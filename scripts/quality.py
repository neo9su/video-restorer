"""
quality.py — Video quality assessment utilities for video-restorer-ui.

Functions
---------
compute_psnr          : PSNR between two frame images (numpy)
compute_ssim          : SSIM between two frame images (skimage w/ numpy fallback)
compare_videos        : Sample N frames from two videos, return avg PSNR/SSIM
generate_comparison_image : Side-by-side PIL comparison with metrics overlay
generate_quality_report   : Full per-frame report saved to JSON
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Optional skimage import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as _skimage_ssim  # type: ignore
    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional cv2 import for video frame extraction
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_frame_as_array(path: str | Path) -> np.ndarray:
    """Load an image file into a uint8 numpy array (H, W, 3) in RGB order."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _ensure_same_size(arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize arr2 to match arr1 if shapes differ (using PIL)."""
    if arr1.shape == arr2.shape:
        return arr1, arr2
    h, w = arr1.shape[:2]
    img2 = Image.fromarray(arr2).resize((w, h), Image.LANCZOS)
    return arr1, np.asarray(img2, dtype=np.uint8)


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def _try_load_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a PIL font; fall back to the built-in bitmap font if needed."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 1. PSNR
# ---------------------------------------------------------------------------

def compute_psnr(ref_frame_path: str | Path, dist_frame_path: str | Path) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two frame images.

    Parameters
    ----------
    ref_frame_path  : path to the reference (original) frame image.
    dist_frame_path : path to the distorted (processed) frame image.

    Returns
    -------
    PSNR in dB as a float.  Returns math.inf if the images are identical.
    """
    ref = _load_frame_as_array(ref_frame_path).astype(np.float64)
    dist = _load_frame_as_array(dist_frame_path).astype(np.float64)
    ref, dist = _ensure_same_size(
        ref.astype(np.uint8), dist.astype(np.uint8)
    )
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)

    mse = np.mean((ref - dist) ** 2)
    if mse == 0.0:
        return math.inf
    max_pixel = 255.0
    psnr = 10.0 * math.log10((max_pixel ** 2) / mse)
    return float(psnr)


# ---------------------------------------------------------------------------
# 2. SSIM
# ---------------------------------------------------------------------------

def _ssim_numpy(ref: np.ndarray, dist: np.ndarray) -> float:
    """Pure-numpy SSIM implementation (luminance channel, greyscale).

    Follows the original Wang et al. (2004) formula with default constants.
    """
    ref_g = np.mean(ref.astype(np.float64), axis=2)
    dist_g = np.mean(dist.astype(np.float64), axis=2)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = ref_g.mean()
    mu2 = dist_g.mean()
    sigma1_sq = ref_g.var()
    sigma2_sq = dist_g.var()
    sigma12 = np.mean((ref_g - mu1) * (dist_g - mu2))

    numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(numerator / denominator)


def compute_ssim(ref_frame_path: str | Path, dist_frame_path: str | Path) -> float:
    """Compute Structural Similarity Index (SSIM) between two frame images.

    Uses ``skimage.metrics.structural_similarity`` when available; otherwise
    falls back to a pure-numpy implementation.

    Parameters
    ----------
    ref_frame_path  : path to the reference frame image.
    dist_frame_path : path to the distorted frame image.

    Returns
    -------
    SSIM score in [-1, 1] (typically [0, 1] for natural images).
    """
    ref  = _load_frame_as_array(ref_frame_path)
    dist = _load_frame_as_array(dist_frame_path)
    ref, dist = _ensure_same_size(ref, dist)

    if _SKIMAGE_AVAILABLE:
        # skimage ≥ 0.19 requires channel_axis instead of multichannel
        try:
            score = _skimage_ssim(ref, dist, channel_axis=2, data_range=255)
        except TypeError:
            score = _skimage_ssim(ref, dist, multichannel=True, data_range=255)
        return float(score)

    logger.debug("skimage not available — using numpy SSIM fallback")
    return _ssim_numpy(ref, dist)


# ---------------------------------------------------------------------------
# 3. compare_videos
# ---------------------------------------------------------------------------

def _extract_frames_cv2(
    video_path: str | Path,
    frame_indices: list[int],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Use OpenCV to extract specific frame indices from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    saved: list[Path] = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d from %s", idx, video_path)
            continue
        # cv2 → BGR; convert to RGB for consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = out_dir / f"{prefix}_frame_{idx:06d}.png"
        Image.fromarray(frame_rgb).save(out_path)
        saved.append(out_path)

    cap.release()
    return saved


def _get_video_frame_count_cv2(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 1)


def _extract_frames_ffmpeg(
    video_path: str | Path,
    frame_indices: list[int],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Fallback: use ffmpeg CLI to extract frames when cv2 is unavailable."""
    import subprocess  # noqa: PLC0415

    saved: list[Path] = []
    for idx in sorted(frame_indices):
        out_path = out_dir / f"{prefix}_frame_{idx:06d}.png"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{idx})",
            "-vframes", "1",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not out_path.exists():
            logger.warning("ffmpeg failed for frame %d: %s", idx,
                           result.stderr.decode(errors="replace"))
            continue
        saved.append(out_path)
    return saved


def _get_video_frame_count_ffprobe(video_path: str | Path) -> int:
    import subprocess  # noqa: PLC0415
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    try:
        return max(int(result.stdout.strip()), 1)
    except ValueError:
        return 100  # safe fallback


def compare_videos(
    input_video: str | Path,
    output_video: str | Path,
    sample_frames: int = 10,
) -> dict[str, Any]:
    """Compare two videos by sampling frames and computing avg PSNR and SSIM.

    Parameters
    ----------
    input_video   : Path to the original (reference) video.
    output_video  : Path to the processed (distorted) video.
    sample_frames : Number of frames to sample evenly across the video.

    Returns
    -------
    dict with keys:
        'psnr'   : float — average PSNR across sampled frames
        'ssim'   : float — average SSIM across sampled frames
        'frames' : list of dicts, each with 'frame_index', 'psnr', 'ssim'
    """
    input_video  = Path(input_video)
    output_video = Path(output_video)

    # Determine total frame count
    if _CV2_AVAILABLE:
        total_frames = _get_video_frame_count_cv2(input_video)
    else:
        total_frames = _get_video_frame_count_ffprobe(input_video)

    sample_frames = min(sample_frames, total_frames)
    if sample_frames <= 1:
        frame_indices = [0]
    else:
        # Evenly spaced indices across [0, total_frames-1]
        frame_indices = [
            int(round(i * (total_frames - 1) / (sample_frames - 1)))
            for i in range(sample_frames)
        ]
    frame_indices = sorted(set(frame_indices))

    with tempfile.TemporaryDirectory(prefix="quality_cmp_") as tmp:
        tmp_dir = Path(tmp)

        if _CV2_AVAILABLE:
            ref_paths  = _extract_frames_cv2(input_video,  frame_indices, tmp_dir, "ref")
            dist_paths = _extract_frames_cv2(output_video, frame_indices, tmp_dir, "dist")
        else:
            ref_paths  = _extract_frames_ffmpeg(input_video,  frame_indices, tmp_dir, "ref")
            dist_paths = _extract_frames_ffmpeg(output_video, frame_indices, tmp_dir, "dist")

        if len(ref_paths) != len(dist_paths):
            min_len = min(len(ref_paths), len(dist_paths))
            ref_paths  = ref_paths[:min_len]
            dist_paths = dist_paths[:min_len]

        frames_data: list[dict[str, Any]] = []
        psnr_values: list[float] = []
        ssim_values: list[float] = []

        for ref_p, dist_p, fidx in zip(ref_paths, dist_paths, frame_indices):
            try:
                psnr_val = compute_psnr(ref_p, dist_p)
                ssim_val = compute_ssim(ref_p, dist_p)
            except Exception as exc:
                logger.warning("Metric computation failed for frame %d: %s", fidx, exc)
                continue

            finite_psnr = psnr_val if math.isfinite(psnr_val) else 100.0
            psnr_values.append(finite_psnr)
            ssim_values.append(ssim_val)
            frames_data.append({
                "frame_index": fidx,
                "psnr": round(finite_psnr, 4),
                "ssim": round(ssim_val, 6),
            })

    avg_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

    return {
        "psnr":   round(avg_psnr, 4),
        "ssim":   round(avg_ssim, 6),
        "frames": frames_data,
    }


# ---------------------------------------------------------------------------
# 4. generate_comparison_image
# ---------------------------------------------------------------------------

def generate_comparison_image(
    before_frame: str | Path,
    after_frame:  str | Path,
    output_path:  str | Path,
    *,
    psnr: float | None = None,
    ssim: float | None = None,
    label_before: str = "Before",
    label_after:  str = "After",
    bar_height: int = 50,
) -> Path:
    """Create a side-by-side comparison image with an optional metrics overlay.

    Parameters
    ----------
    before_frame : Path to the reference/before frame image.
    after_frame  : Path to the processed/after frame image.
    output_path  : Destination path for the composite image (PNG recommended).
    psnr         : Optional PSNR value to display in the overlay.
    ssim         : Optional SSIM value to display in the overlay.
    label_before : Label printed above the left (before) frame.
    label_after  : Label printed above the right (after) frame.
    bar_height   : Height (px) of the header/footer info bars.

    Returns
    -------
    Path to the saved comparison image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load & normalise sizes ------------------------------------------
    img_before = Image.open(before_frame).convert("RGB")
    img_after  = Image.open(after_frame).convert("RGB")

    # Resize after to match before height if needed
    bw, bh = img_before.size
    aw, ah = img_after.size
    if ah != bh:
        new_aw = int(aw * bh / ah)
        img_after = img_after.resize((new_aw, bh), Image.LANCZOS)
        aw = new_aw

    total_w = bw + aw
    total_h = bh + bar_height * 2   # top bar (labels) + bottom bar (metrics)

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    font_large = _try_load_font(22)
    font_small = _try_load_font(17)
    draw = ImageDraw.Draw(canvas)

    # --- Top label bar ---------------------------------------------------
    # Before label
    draw.rectangle([(0, 0), (bw - 1, bar_height - 1)], fill=(50, 50, 50))
    draw.text((bw // 2, bar_height // 2), label_before,
              font=font_large, fill=(255, 255, 255), anchor="mm")
    # After label
    draw.rectangle([(bw, 0), (total_w - 1, bar_height - 1)], fill=(40, 60, 40))
    draw.text((bw + aw // 2, bar_height // 2), label_after,
              font=font_large, fill=(200, 255, 200), anchor="mm")

    # --- Paste frames ----------------------------------------------------
    canvas.paste(img_before, (0,  bar_height))
    canvas.paste(img_after,  (bw, bar_height))

    # --- Centre divider line ---------------------------------------------
    for y in range(bar_height, bar_height + bh):
        canvas.putpixel((bw - 1, y), (255, 255, 0))
        canvas.putpixel((bw,     y), (255, 255, 0))

    # --- Bottom metrics bar ----------------------------------------------
    metrics_y = bar_height + bh
    draw.rectangle([(0, metrics_y), (total_w - 1, total_h - 1)], fill=(20, 20, 40))

    metric_parts: list[str] = []
    if psnr is not None:
        psnr_display = "∞" if math.isinf(psnr) else f"{psnr:.2f} dB"
        metric_parts.append(f"PSNR: {psnr_display}")
    if ssim is not None:
        metric_parts.append(f"SSIM: {ssim:.4f}")

    if metric_parts:
        metrics_text = "   |   ".join(metric_parts)
        draw.text(
            (total_w // 2, metrics_y + bar_height // 2),
            metrics_text,
            font=font_small,
            fill=(220, 220, 80),
            anchor="mm",
        )
    else:
        draw.text(
            (total_w // 2, metrics_y + bar_height // 2),
            "No metrics available",
            font=font_small,
            fill=(150, 150, 150),
            anchor="mm",
        )

    canvas.save(str(output_path))
    logger.info("Comparison image saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 5. generate_quality_report
# ---------------------------------------------------------------------------

def generate_quality_report(
    task_id:       str,
    work_dir:      str | Path,
    input_video:   str | Path,
    output_video:  str | Path,
    sample_frames: int = 10,
) -> dict[str, Any]:
    """Generate a full quality report comparing input and output videos.

    Per-frame PSNR and SSIM are computed, a summary is produced, and the
    report is saved as ``<work_dir>/quality_report.json``.

    Parameters
    ----------
    task_id      : Unique identifier for the restoration task.
    work_dir     : Directory where the report JSON will be written.
    input_video  : Path to the original (reference) video.
    output_video : Path to the processed (output) video.
    sample_frames: Number of frames to sample for metrics.

    Returns
    -------
    dict containing the full quality report (mirrors the JSON file).
    """
    work_dir     = Path(work_dir)
    input_video  = Path(input_video)
    output_video = Path(output_video)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Starting quality report: %s → %s", task_id, input_video, output_video)

    # --- Core comparison -------------------------------------------------
    comparison = compare_videos(input_video, output_video, sample_frames=sample_frames)
    frames_data: list[dict[str, Any]] = comparison["frames"]

    # --- Per-frame PSNR / SSIM min/max/std statistics -------------------
    psnr_vals = [f["psnr"] for f in frames_data]
    ssim_vals = [f["ssim"] for f in frames_data]

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        arr = np.array(vals, dtype=np.float64)
        return {
            "min":  round(float(arr.min()),  4),
            "max":  round(float(arr.max()),  4),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()),  4),
        }

    # --- Optional comparison images (best / worst / median frame) --------
    comparison_images: list[str] = []
    if frames_data and _CV2_AVAILABLE:
        # Identify notable frames
        notable: dict[str, int] = {}
        if psnr_vals:
            notable["best"]   = frames_data[int(np.argmax(psnr_vals))]["frame_index"]
            notable["worst"]  = frames_data[int(np.argmin(psnr_vals))]["frame_index"]
            mid = len(psnr_vals) // 2
            notable["median"] = frames_data[mid]["frame_index"]

        with tempfile.TemporaryDirectory(prefix="quality_imgs_") as tmp:
            tmp_dir = Path(tmp)
            for label, fidx in notable.items():
                try:
                    ref_paths  = _extract_frames_cv2(input_video,  [fidx], tmp_dir, f"ref_{label}")
                    dist_paths = _extract_frames_cv2(output_video, [fidx], tmp_dir, f"dist_{label}")
                    if ref_paths and dist_paths:
                        cmp_out = work_dir / f"comparison_{label}_frame{fidx}.png"
                        frame_entry = next(
                            (f for f in frames_data if f["frame_index"] == fidx), None
                        )
                        generate_comparison_image(
                            ref_paths[0],
                            dist_paths[0],
                            cmp_out,
                            psnr=frame_entry["psnr"] if frame_entry else None,
                            ssim=frame_entry["ssim"] if frame_entry else None,
                            label_before="Original",
                            label_after="Restored",
                        )
                        comparison_images.append(str(cmp_out))
                except Exception as exc:
                    logger.warning("Could not generate comparison image for frame %d: %s", fidx, exc)

    # --- Assemble report -------------------------------------------------
    report: dict[str, Any] = {
        "task_id":   task_id,
        "input_video":  str(input_video),
        "output_video": str(output_video),
        "sample_frames": len(frames_data),
        "summary": {
            "psnr": _stats(psnr_vals),
            "ssim": _stats(ssim_vals),
            "avg_psnr": comparison["psnr"],
            "avg_ssim": comparison["ssim"],
        },
        "frames": frames_data,
        "comparison_images": comparison_images,
        "quality_grade": _quality_grade(comparison["psnr"], comparison["ssim"]),
    }

    # --- Save JSON -------------------------------------------------------
    report_path = work_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("[%s] Quality report saved to %s", task_id, report_path)
    return report


# ---------------------------------------------------------------------------
# Internal helpers — quality grading
# ---------------------------------------------------------------------------

def _quality_grade(avg_psnr: float, avg_ssim: float) -> str:
    """Return a human-readable quality grade based on PSNR and SSIM values."""
    # Simplified ITU-style grading heuristic
    if avg_psnr >= 40 and avg_ssim >= 0.95:
        return "Excellent"
    if avg_psnr >= 35 and avg_ssim >= 0.90:
        return "Good"
    if avg_psnr >= 30 and avg_ssim >= 0.80:
        return "Fair"
    if avg_psnr >= 25 and avg_ssim >= 0.65:
        return "Poor"
    return "Bad"


# ---------------------------------------------------------------------------
# CLI convenience wrapper (python -m quality or direct invocation)
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Video quality assessment tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # psnr
    p = sub.add_parser("psnr", help="Compute PSNR between two images")
    p.add_argument("ref");  p.add_argument("dist")

    # ssim
    p = sub.add_parser("ssim", help="Compute SSIM between two images")
    p.add_argument("ref");  p.add_argument("dist")

    # compare
    p = sub.add_parser("compare", help="Compare two videos")
    p.add_argument("input_video");  p.add_argument("output_video")
    p.add_argument("--frames", type=int, default=10)

    # report
    p = sub.add_parser("report", help="Generate a full quality report")
    p.add_argument("task_id")
    p.add_argument("work_dir")
    p.add_argument("input_video")
    p.add_argument("output_video")
    p.add_argument("--frames", type=int, default=10)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "psnr":
        print(f"PSNR: {compute_psnr(args.ref, args.dist):.4f} dB")
    elif args.cmd == "ssim":
        print(f"SSIM: {compute_ssim(args.ref, args.dist):.6f}")
    elif args.cmd == "compare":
        result = compare_videos(args.input_video, args.output_video, args.frames)
        print(json.dumps(result, indent=2))
    elif args.cmd == "report":
        result = generate_quality_report(
            args.task_id, args.work_dir,
            args.input_video, args.output_video,
            args.frames,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
