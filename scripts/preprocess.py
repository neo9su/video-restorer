"""
preprocess.py — Video pre-processing utilities for the Video Restorer UI.

Functions
---------
auto_crop_black_bars  – detect & crop black bars via ffmpeg cropdetect
extract_audio         – rip audio track to AAC (.aac / .m4a)
scene_split           – split video at scene-change boundaries
get_video_info        – probe metadata with ffprobe
normalize_video       – transcode to H.264 / yuv420p at an optional target FPS

CLI
---
    python preprocess.py info        <input>
    python preprocess.py crop        <input> <output>
    python preprocess.py audio       <input> <output>
    python preprocess.py split       <input> <output_dir> [--threshold 0.4]
    python preprocess.py normalize   <input> <output> [--fps 30]
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run *cmd* via subprocess, log it, and raise on non-zero exit."""
    log.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else subprocess.STDOUT,
        text=True,
        check=check,
    )


def _require_file(path: str | os.PathLike) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. get_video_info
# ---------------------------------------------------------------------------

def get_video_info(input_path: str | os.PathLike) -> dict:
    """Return metadata dict for *input_path* using ffprobe.

    Keys
    ----
    width, height : int
    fps           : float
    duration      : float   (seconds)
    codec         : str     (video codec name, e.g. "h264")
    audio_codec   : str     (audio codec name or "" if absent)
    size_bytes    : int
    """
    src = _require_file(input_path)

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(src),
    ]

    try:
        result = _run(cmd, capture=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        log.error("ffprobe failed for %s: %s", src, exc.stderr or exc)
        raise RuntimeError(f"ffprobe failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        log.error("Could not parse ffprobe output for %s", src)
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    # FPS — stored as a fraction string like "30000/1001"
    fps_raw = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration = float(
        video_stream.get("duration")
        or fmt.get("duration")
        or 0.0
    )

    info = {
        "width":       int(video_stream.get("width", 0)),
        "height":      int(video_stream.get("height", 0)),
        "fps":         round(fps, 4),
        "duration":    round(duration, 4),
        "codec":       video_stream.get("codec_name", ""),
        "audio_codec": audio_stream.get("codec_name", ""),
        "size_bytes":  int(fmt.get("size", 0)),
    }

    log.info("Video info for %s: %s", src.name, info)
    return info


# ---------------------------------------------------------------------------
# 2. auto_crop_black_bars
# ---------------------------------------------------------------------------

def auto_crop_black_bars(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
) -> Path:
    """Detect black bars with ffmpeg *cropdetect* then re-encode with the crop applied.

    Returns the output Path.
    """
    src = _require_file(input_path)
    dst = Path(output_path)
    _ensure_dir(dst.parent)

    log.info("Running cropdetect on %s …", src.name)

    # Pass 1 – run cropdetect on the full video (read only, no output file needed)
    detect_cmd = [
        "ffmpeg",
        "-i", str(src),
        "-vf", "cropdetect=24:16:0",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(
            detect_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        stderr_output = result.stderr
    except subprocess.CalledProcessError as exc:
        log.error("cropdetect failed: %s", exc.stderr)
        raise RuntimeError("cropdetect pass failed") from exc

    # Extract all crop=W:H:X:Y values and take the last (most stable) one
    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", stderr_output)
    if not crops:
        log.warning("No crop values detected – copying source unchanged.")
        # Nothing to crop; just copy
        _run(["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)])
        return dst

    crop_value = crops[-1]
    log.info("Detected crop: %s", crop_value)

    # Pass 2 – apply crop
    crop_cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"crop={crop_value}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        str(dst),
    ]

    try:
        _run(crop_cmd)
        log.info("Cropped video written to %s", dst)
    except subprocess.CalledProcessError as exc:
        log.error("Crop encode failed: %s", exc)
        raise RuntimeError("Crop encode failed") from exc

    return dst


# ---------------------------------------------------------------------------
# 3. extract_audio
# ---------------------------------------------------------------------------

def extract_audio(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
) -> Path:
    """Extract the audio track from *input_path* to *output_path* as AAC.

    The output file extension should be ``.aac`` or ``.m4a``.
    Returns the output Path.
    """
    src = _require_file(input_path)
    dst = Path(output_path)
    _ensure_dir(dst.parent)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",                    # drop video
        "-c:a", "aac",
        "-b:a", "192k",
        str(dst),
    ]

    try:
        _run(cmd)
        log.info("Audio extracted to %s", dst)
    except subprocess.CalledProcessError as exc:
        log.error("Audio extraction failed: %s", exc)
        raise RuntimeError("Audio extraction failed") from exc

    return dst


# ---------------------------------------------------------------------------
# 4. scene_split
# ---------------------------------------------------------------------------

def scene_split(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    threshold: float = 0.4,
) -> List[Path]:
    """Split *input_path* at scene-change boundaries.

    Uses ffmpeg's ``select`` / ``scene`` filter to detect cuts, then writes
    each segment via ``segment`` muxer keyed to those timestamps.

    Parameters
    ----------
    input_path  : source video
    output_dir  : directory where segments are written
    threshold   : scene-change score [0.0 – 1.0]; lower = more sensitive

    Returns
    -------
    List[Path]  : sorted list of created segment paths
    """
    src = _require_file(input_path)
    out_dir = _ensure_dir(output_dir)

    log.info("Detecting scene changes in %s (threshold=%.2f) …", src.name, threshold)

    # Step 1 – collect scene-change timestamps
    detect_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_frames",
        "-select_streams", "v",
        "-read_intervals", "%+#9999",   # up to 9999 frames look-ahead
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time,pict_type",
        str(src),
    ]

    # Use ffmpeg scene filter to get timestamps via stderr
    scene_detect_cmd = [
        "ffmpeg",
        "-i", str(src),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(
            scene_detect_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        showinfo_output = result.stderr
    except subprocess.CalledProcessError as exc:
        log.error("Scene detection failed: %s", exc.stderr)
        raise RuntimeError("Scene detection failed") from exc

    # Parse timestamps from showinfo lines: "pts_time:X.XXXXXX"
    timestamps = [0.0]  # always start from t=0
    for match in re.finditer(r"pts_time:([\d.]+)", showinfo_output):
        t = float(match.group(1))
        if t > 0.0 and t not in timestamps:
            timestamps.append(t)
    timestamps.sort()

    log.info("Found %d scene boundaries: %s", len(timestamps) - 1, timestamps[1:])

    if len(timestamps) == 1:
        log.warning("No scene changes detected – output is a single segment copy.")

    # Step 2 – cut segments
    segments: List[Path] = []
    duration_info = get_video_info(src)
    total_duration = duration_info["duration"]

    for i, start in enumerate(timestamps):
        end = timestamps[i + 1] if i + 1 < len(timestamps) else total_duration
        segment_path = out_dir / f"segment_{i:04d}.mp4"

        cut_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(src),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(segment_path),
        ]

        try:
            _run(cut_cmd)
            segments.append(segment_path)
            log.info("Segment %d written: %s (%.2f – %.2f s)", i, segment_path.name, start, end)
        except subprocess.CalledProcessError as exc:
            log.error("Failed to cut segment %d: %s", i, exc)
            # Continue with remaining segments rather than aborting

    log.info("scene_split complete – %d segment(s) in %s", len(segments), out_dir)
    return sorted(segments)


# ---------------------------------------------------------------------------
# 5. normalize_video
# ---------------------------------------------------------------------------

def normalize_video(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    target_fps: Optional[float] = None,
) -> Path:
    """Normalize *input_path* to H.264 / yuv420p.

    Parameters
    ----------
    input_path  : source video
    output_path : destination file (.mp4 recommended)
    target_fps  : if given, force output frame rate (e.g. 24, 25, 30, 60)

    Returns
    -------
    Path to the normalized output file.
    """
    src = _require_file(input_path)
    dst = Path(output_path)
    _ensure_dir(dst.parent)

    vf_filters = ["format=yuv420p"]
    if target_fps is not None:
        vf_filters.append(f"fps={target_fps}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", ",".join(vf_filters),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ]

    try:
        _run(cmd)
        log.info("Normalized video written to %s", dst)
    except subprocess.CalledProcessError as exc:
        log.error("normalize_video failed: %s", exc)
        raise RuntimeError("normalize_video failed") from exc

    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_info(args: list[str]) -> None:
    if len(args) < 1:
        print("Usage: preprocess.py info <input>")
        sys.exit(1)
    info = get_video_info(args[0])
    print(json.dumps(info, indent=2))


def _cli_crop(args: list[str]) -> None:
    if len(args) < 2:
        print("Usage: preprocess.py crop <input> <output>")
        sys.exit(1)
    out = auto_crop_black_bars(args[0], args[1])
    print(f"Output: {out}")


def _cli_audio(args: list[str]) -> None:
    if len(args) < 2:
        print("Usage: preprocess.py audio <input> <output>")
        sys.exit(1)
    out = extract_audio(args[0], args[1])
    print(f"Output: {out}")


def _cli_split(args: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="preprocess.py split")
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--threshold", type=float, default=0.4)
    parsed = parser.parse_args(args)
    segments = scene_split(parsed.input, parsed.output_dir, parsed.threshold)
    print(f"Created {len(segments)} segment(s):")
    for s in segments:
        print(f"  {s}")


def _cli_normalize(args: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="preprocess.py normalize")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--fps", type=float, default=None)
    parsed = parser.parse_args(args)
    out = normalize_video(parsed.input, parsed.output, target_fps=parsed.fps)
    print(f"Output: {out}")


_COMMANDS = {
    "info":      _cli_info,
    "crop":      _cli_crop,
    "audio":     _cli_audio,
    "split":     _cli_split,
    "normalize": _cli_normalize,
}

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__)
        print("Commands:", ", ".join(_COMMANDS))
        sys.exit(0)

    command = sys.argv[1]
    rest = sys.argv[2:]
    try:
        _COMMANDS[command](rest)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)
