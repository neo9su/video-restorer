"""后处理模块 - 帧序列组装为视频 + 色彩优化 + 去抖"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir, run_ffmpeg, load_config


def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: float = 30.0,
    codec: str = "h264",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
    bitrate: str = "",
) -> bool:
    """将帧序列合成为视频"""
    ensure_dir(os.path.dirname(output_path))
    pattern = os.path.join(frame_dir, "%08d.png")

    logger.info(f"组装视频: {frame_dir} -> {output_path}")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", {"h264": "libx264", "h265": "libx265"}.get(codec, codec),
        "-crf", str(crf),
        "-pix_fmt", pix_fmt,
    ]
    if bitrate:
        cmd += ["-b:v", bitrate]
    cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]  # 确保偶数宽高
    cmd += [output_path]

    return run_ffmpeg(cmd, desc="帧序列合成视频")


def apply_color_enhancement(
    frame_dir: str,
    output_dir: str,
    saturation: float = 1.2,
    contrast: float = 1.1,
    brightness: float = 1.0,
) -> int:
    """色彩增强（饱和度/对比度/亮度）"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("需要 opencv-python")
        return 0

    ensure_dir(output_dir)
    frames = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    count = 0
    for fname in frames:
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            count += 1
            continue

        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            continue

        # HSV 饱和度调整
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 对比度 / 亮度
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=(brightness - 1.0) * 50)

        cv2.imwrite(out_path, img)
        count += 1

    logger.info(f"色彩增强: {count} 帧")
    return count


def apply_deshake(frame_dir: str, output_dir: str) -> int:
    """使用 FFmpeg vidstab 去抖"""
    ensure_dir(output_dir)

    # 1. 生成变换数据
    pattern = os.path.join(frame_dir, "%08d.png")
    transforms_file = os.path.join(output_dir, "transforms.trf")
    cmd1 = [
        "ffmpeg", "-y", "-i", pattern,
        "-vf", "vidstabdetect=step=32:shakiness=5:accuracy=15:result=" + transforms_file,
        "-f", "null", "-"
    ]
    if not run_ffmpeg(cmd1, desc="检测抖动"):
        return 0

    # 2. 应用稳定
    cmd2 = [
        "ffmpeg", "-y", "-i", pattern,
        "-vf", f"vidstabtransform=input={transforms_file}:zoom=1:smoothing=30",
        os.path.join(output_dir, "%08d.png")
    ]
    if not run_ffmpeg(cmd2, desc="去抖处理"):
        return 0

    return len(os.listdir(output_dir))


def assemble_video(
    frame_dir: str,
    output_path: str,
    config: Dict[str, Any],
    source_fps: float = 30.0,
) -> bool:
    """
    完整后处理: 可选色彩增强 / 去抖 -> 合成视频
    """
    post_cfg = config["postprocess"]
    current_dir = frame_dir

    # 色彩增强
    if post_cfg["color_enhance"]["enabled"]:
        logger.info("应用色彩增强...")
        color_dir = ensure_dir(os.path.join(frame_dir, "..", "frames_color"))
        apply_color_enhancement(
            current_dir, color_dir,
            saturation=post_cfg["color_enhance"]["saturation"],
            contrast=post_cfg["color_enhance"]["contrast"],
            brightness=post_cfg["color_enhance"]["brightness"],
        )
        current_dir = color_dir

    # 去抖
    if post_cfg.get("deshake", False):
        logger.info("应用视频去抖...")
        deshake_dir = ensure_dir(os.path.join(frame_dir, "..", "frames_stabilized"))
        apply_deshake(current_dir, deshake_dir)
        current_dir = deshake_dir

    # 合成为视频
    fps = post_cfg["output_fps"] if post_cfg["output_fps"] > 0 else source_fps
    return frames_to_video(
        current_dir, output_path,
        fps=fps,
        codec=post_cfg["output_codec"],
        crf=post_cfg["output_crf"],
        pix_fmt=post_cfg["output_pix_fmt"],
        bitrate=post_cfg["bitrate"],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="后处理：帧序列->视频")
    parser.add_argument("--frames-dir", required=True, help="帧目录")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--fps", type=float, default=30.0, help="帧率")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件")
    args = parser.parse_args()

    config = load_config(args.config)
    success = assemble_video(args.frames_dir, args.output, config, args.fps)
    if success:
        print(f"\n[OK] 视频组装完成！输出: {args.output}")
    else:
        print(f"\n[FAIL] 视频组装失败")
        sys.exit(1)
