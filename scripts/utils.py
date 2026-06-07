"""Video-Restorer 工具模块"""

import os
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video-restorer")


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"配置文件加载: {config_path}")
    return cfg


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def run_ffmpeg(cmd: list, desc: str = "") -> bool:
    """执行 FFmpeg 命令"""
    logger.info(f"FFmpeg: {desc or ' '.join(cmd)}")
    try:
        # 注意: 不使用 capture_output=True，避免 pipe buffer 死锁
        # FFmpeg 大量 stderr 进度输出会填满 64KB pipe，导致进程挂起
        result = subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 失败 (exit={e.returncode})")
        return False


def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """用 ffprobe 获取视频信息"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    try:
        import json
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
        if not video_stream:
            logger.error(f"未找到视频流: {video_path}")
            return None
        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),
            "codec": video_stream.get("codec_name", "unknown"),
            "bitrate": info.get("format", {}).get("bit_rate", "N/A"),
            "duration": float(info.get("format", {}).get("duration", 0)),
            "total_frames": int(video_stream.get("nb_frames", 0)),
        }
    except Exception as e:
        logger.error(f"获取视频信息失败: {e}")
        return None


def print_info(info: Dict[str, Any], title: str = "视频信息"):
    """友好打印视频信息"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  分辨率: {info['width']}x{info['height']}")
    print(f"  帧率:   {info['fps']:.2f} fps")
    print(f"  时长:   {info['duration']:.1f}s ({info['duration']/60:.1f}min)")
    print(f"  总帧数: {info['total_frames']}")
    print(f"  编码:   {info['codec']}")
    print(f"{'='*50}\n")
