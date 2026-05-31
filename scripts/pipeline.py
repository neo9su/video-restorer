#!/usr/bin/env python3
"""
============================================================
  Video-Restorer 主流水线
  老片翻新 | 低分辨率提升 | 模糊变高清 | 去水印
============================================================

用法:
    # 完整流水线
    python scripts/pipeline.py input/video.mp4 -o output/result.mp4

    # 跳过某些步骤
    python scripts/pipeline.py input/video.mp4 --no-watermark --no-enhance

    # 2x 超分 + 快速模式
    python scripts/pipeline.py input/video.mp4 --scale 2 --fast
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# 添加到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from utils import logger, ensure_dir, get_video_info, print_info, load_config
from preprocess import preprocess_pipeline
from remove_watermark import run_pipeline as remove_watermark_pipeline
from enhance import run_pipeline as enhance_pipeline
from upscale import run_pipeline as upscale_pipeline
from postprocess import assemble_video


class VideoRestorer:
    """视频修复主流水线"""

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        work_dir: str = ".",
        keep_frames: bool = False,
        fast_mode: bool = False,
    ):
        self.config = load_config(config_path)
        self.work_dir = os.path.abspath(work_dir)
        self.keep_frames = keep_frames
        self.fast_mode = fast_mode

        # 工作目录
        self.frames_dir = os.path.join(self.work_dir, "frames")
        self.frames_normalized = os.path.join(self.work_dir, "frames_normalized")
        self.frames_denoised = os.path.join(self.work_dir, "frames_denoised")
        self.frames_enhanced = os.path.join(self.work_dir, "frames_enhanced")
        self.frames_upscaled = os.path.join(self.work_dir, "frames_upscaled")

        # 如果开了 fast_mode，调整参数
        if fast_mode:
            self.config["super_resolution"]["tile_size"] = 128
            self.config["enhancement"]["tile_size"] = 128
            logger.info("⚡ 快速模式: 使用更小的分块加速推理")

    def cleanup(self):
        """清理中间文件"""
        if self.keep_frames:
            logger.info("保留中间帧 (--keep-frames)")
            return

        dirs_to_clean = [
            self.frames_normalized,
            self.frames_denoised,
            self.frames_enhanced,
            self.frames_upscaled,
        ]
        for d in dirs_to_clean:
            if os.path.exists(d):
                shutil.rmtree(d)
                logger.info(f"清理: {d}")

    def run(
        self,
        input_path: str,
        output_path: str,
        skip_watermark: bool = False,
        skip_enhance: bool = False,
        skip_upscale: bool = False,
    ) -> bool:
        """
        执行完整修复流水线

        流程:
        1. 预处理 (格式统一 + 拆帧)
        2. 去水印 (LaMa / ProPainter)
        3. 画质增强 (去噪/去模糊)
        4. 超分辨率放大
        5. 后处理 (色彩优化 + 合成视频)
        """
        start_time = time.time()
        abs_input = os.path.abspath(input_path)

        if not os.path.exists(abs_input):
            logger.error(f"输入文件不存在: {abs_input}")
            return False

        output_path = os.path.abspath(output_path)
        logger.info(f"{'='*55}")
        logger.info(f"  Video-Restorer 开始处理")
        logger.info(f"  输入: {abs_input}")
        logger.info(f"  输出: {output_path}")
        logger.info(f"{'='*55}")

        # ---- Step 1: 预处理 ----
        logger.info(f"\n{'='*55}")
        logger.info(f"  Step 1/5: 预处理 (格式统一 + 拆帧)")
        logger.info(f"{'='*55}")
        frames = preprocess_pipeline(
            abs_input, self.config,
            frames_dir=self.frames_dir,
        )
        if not frames:
            logger.error("预处理失败")
            return False

        # ---- Step 2: 去水印 ----
        current_frames = frames
        if not skip_watermark:
            logger.info(f"\n{'='*55}")
            logger.info(f"  Step 2/5: 去水印")
            logger.info(f"{'='*55}")
            result = remove_watermark_pipeline(
                current_frames, self.config,
                output_dir=self.frames_denoised,
            )
            if result:
                current_frames = result
            else:
                logger.warning("去水印步骤跳过或失败，继续下一步")
        else:
            logger.info("⏭️  去水印已跳过")

        # ---- Step 3: 画质增强 ----
        if not skip_enhance:
            logger.info(f"\n{'='*55}")
            logger.info(f"  Step 3/5: 画质增强 (去噪/去模糊)")
            logger.info(f"{'='*55}")
            result = enhance_pipeline(
                current_frames, self.config,
                output_dir=self.frames_enhanced,
            )
            if result:
                current_frames = result
        else:
            logger.info("⏭️  画质增强已跳过")

        # ---- Step 4: 超分辨率 ----
        if not skip_upscale:
            logger.info(f"\n{'='*55}")
            logger.info(f"  Step 4/5: 超分辨率放大")
            logger.info(f"{'='*55}")
            result = upscale_pipeline(
                current_frames, self.config,
                output_dir=self.frames_upscaled,
            )
            if result:
                current_frames = result
        else:
            logger.info("⏭️  超分辨率已跳过")

        # ---- Step 5: 后处理 ----
        logger.info(f"\n{'='*55}")
        logger.info(f"  Step 5/5: 后处理 (合成视频)")
        logger.info(f"{'='*55}")

        # 获取原始帧率
        info = get_video_info(abs_input)
        source_fps = info["fps"] if info else 30.0

        success = assemble_video(
            current_frames, output_path, self.config, source_fps=source_fps,
        )

        # ---- 完成 ----
        elapsed = time.time() - start_time
        if success:
            out_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"\n{'='*55}")
            logger.info(f"  [OK] 处理完成!")
            logger.info(f"  输出: {output_path}")
            logger.info(f"  大小: {out_size:.1f} MB")
            logger.info(f"  耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)")
            logger.info(f"{'='*55}")
        else:
            logger.error("[FAIL] 视频合成失败")

        # 清理
        self.cleanup()

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Video-Restorer - 老片翻新/超分/去水印工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程 (去水印 + 增强 + 2x超分)
  python scripts/pipeline.py input/video.mp4 -o output/result.mp4

  # 仅超分 (跳过水印和增强)
  python scripts/pipeline.py input/video.mp4 -o output/hd.mp4 --no-watermark --no-enhance

  # 4x 超分
  python scripts/pipeline.py input/video.mp4 -o output/4k.mp4 --scale 4

  # 快速模式
  python scripts/pipeline.py input/video.mp4 -o output/result.mp4 --fast

  # 保留中间帧 (调试用)
  python scripts/pipeline.py input/video.mp4 -o output/result.mp4 --keep-frames
        """,
    )
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", default=None, help="输出视频路径 (默认: output/目录)")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件路径")
    parser.add_argument("--work-dir", default=".", help="工作目录")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=None, help="超分倍数 (覆盖配置)")
    parser.add_argument("--no-watermark", action="store_true", help="跳过去水印")
    parser.add_argument("--no-enhance", action="store_true", help="跳过画质增强")
    parser.add_argument("--no-upscale", action="store_true", help="跳过超分辨率")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间帧")
    parser.add_argument("--fast", action="store_true", help="快速模式 (减小tile_size)")

    args = parser.parse_args()

    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        input_name = Path(args.input).stem
        ensure_dir("output")
        output_path = os.path.join("output", f"{input_name}_enhanced.mp4")

    # 加载配置并覆写
    config = load_config(args.config)
    if args.scale:
        config["super_resolution"]["scale"] = args.scale

    # 运行流水线
    restorer = VideoRestorer(
        config_path=args.config,
        work_dir=args.work_dir,
        keep_frames=args.keep_frames,
        fast_mode=args.fast,
    )

    success = restorer.run(
        args.input, output_path,
        skip_watermark=args.no_watermark,
        skip_enhance=args.no_enhance,
        skip_upscale=args.no_upscale,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
