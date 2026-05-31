"""预处理模块 - 格式统一、帧提取"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir, run_ffmpeg, get_video_info, print_info, load_config


def transcode_video(
    input_path: str,
    output_path: str,
    codec: str = "h264",
    crf: int = 18,
    target_fps: float = 0,
    pix_fmt: str = "yuv420p",
) -> bool:
    """统一视频格式"""
    logger.info(f"转码: {input_path} -> {output_path}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", {"h264": "libx264", "h265": "libx265"}.get(codec, codec),
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
    ]
    if target_fps > 0:
        cmd += ["-r", str(target_fps)]
    cmd += [output_path]
    return run_ffmpeg(cmd, desc="视频转码")


def extract_frames(
    video_path: str,
    output_dir: str,
    fmt: str = "png",
    quality: int = 95,
) -> int:
    """将视频拆分为帧序列"""
    ensure_dir(output_dir)
    pattern = os.path.join(output_dir, f"%08d.{fmt}")

    logger.info(f"拆帧: {video_path} -> {output_dir}/ (格式: {fmt})")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if fmt == "jpg" or fmt == "jpeg":
        cmd += ["-qscale:v", str(quality)]
    else:
        cmd += ["-compression_level", "1"]  # PNG 快速压缩
    cmd += [pattern]

    if not run_ffmpeg(cmd, desc="视频拆帧"):
        return 0

    # 统计帧数
    frames = sorted(Path(output_dir).glob(f"*.{fmt}"))
    count = len(frames)
    logger.info(f"拆帧完成: 共 {count} 帧")
    return count


def preprocess_pipeline(
    input_video: str,
    config: Dict[str, Any],
    frames_dir: str = "frames",
    output_video: Optional[str] = None,
) -> Optional[str]:
    """
    完整预处理流程:
    1. 获取视频信息
    2. 统一格式转码
    3. 拆帧
    返回帧目录路径
    """
    cfg = config["preprocess"]
    extract_cfg = config["extract_frames"]

    # 1. 获取视频信息
    info = get_video_info(input_video)
    if not info:
        logger.error("无法读取视频信息")
        return None
    print_info(info)

    # 2. 统一格式
    if not output_video:
        base = Path(input_video).stem
        output_video = os.path.join(frames_dir, f"{base}_normalized.mp4")

    ensure_dir(os.path.dirname(output_video))
    transcode_video(
        input_video, output_video,
        codec=cfg["codec"],
        crf=cfg["crf"],
        target_fps=cfg["target_fps"],
        pix_fmt=cfg["pix_fmt"],
    )

    # 3. 拆帧
    frame_out = ensure_dir(frames_dir)
    total_frames = extract_frames(
        output_video, frame_out,
        fmt=extract_cfg["format"],
        quality=extract_cfg["quality"],
    )
    return frame_out if total_frames > 0 else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频预处理：格式统一 + 拆帧")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("--frames-dir", default="frames", help="帧输出目录")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件")
    args = parser.parse_args()

    config = load_config(args.config)
    result = preprocess_pipeline(args.input, config, frames_dir=args.frames_dir)
    if result:
        print(f"\n[OK] 预处理完成！帧目录: {result}")
    else:
        print(f"\n[FAIL] 预处理失败")
        sys.exit(1)
