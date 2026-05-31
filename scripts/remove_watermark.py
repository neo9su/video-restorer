"""去水印模块 - 支持 LaMa / ProPainter 修复"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir, load_config


def create_mask_from_bbox(
    frame_path: str, mask_path: str,
    bbox: Tuple[int, int, int, int]
) -> bool:
    """
    根据边界框创建掩码图像
    bbox: (x, y, w, h)
    """
    try:
        from PIL import Image, ImageDraw
        img = Image.open(frame_path)
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], fill=255)
        mask.save(mask_path)
        return True
    except Exception as e:
        logger.error(f"创建掩码失败: {e}")
        return False


def create_masks_for_frames(
    frame_dir: str, mask_dir: str,
    bbox: Tuple[int, int, int, int],
    frame_list: Optional[List[str]] = None,
) -> int:
    """为所有帧创建掩码"""
    ensure_dir(mask_dir)
    if frame_list is None:
        frame_list = sorted([
            f for f in os.listdir(frame_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    count = 0
    for frame_name in frame_list:
        mask_path = os.path.join(mask_dir, frame_name)
        if os.path.exists(mask_path):
            continue
        frame_path = os.path.join(frame_dir, frame_name)
        if create_mask_from_bbox(frame_path, mask_path, bbox):
            count += 1

    logger.info(f"生成 {count} 张掩码 -> {mask_dir}")
    return count


def detect_watermark_area(frame_dir: str, sample_count: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    通过边缘检测 + 亮度统计自动检测水印区域（简化版）
    返回 (x, y, w, h) 或 None
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.error("需要 opencv-python")
        return None

    frames = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if not frames:
        return None

    # 采样几帧，取水印区域的交集
    step = max(1, len(frames) // sample_count)
    sampled = frames[::step][:sample_count]

    # 累积梯度图
    acc_grad = None
    for fname in sampled:
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad = cv2.Laplacian(gray, cv2.CV_64F)
        grad = np.abs(grad).astype(np.uint8)
        if acc_grad is None:
            acc_grad = grad.astype(np.float32)
        else:
            acc_grad += grad.astype(np.float32)

    if acc_grad is None:
        return None

    # 梯度均值作为阈值
    acc_grad /= len(sampled)
    _, thresh = cv2.threshold(acc_grad.astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 膨胀连接
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 找最大的轮廓（通常是水印区域）
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    logger.info(f"自动检测水印区域: ({x}, {y}, {w}, {h})")
    return (x, y, w, h)


def inpaint_with_lama(
    frame_dir: str, mask_dir: str, output_dir: str,
    model_path: str = "models/lama/big-lama.pt",
    device: str = "cuda",
) -> int:
    """
    使用 LaMa 模型修复去水印
    LaMa (Large Mask Inpainting) 对大面积水印效果很好
    """
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        return 0

    ensure_dir(output_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载 LaMa 模型
    try:
        from tools.lama.models import LaMa
        model = LaMa(model_path, device=device)
    except ImportError:
        # 尝试使用 OpenCV 自带的 inpaint 作为 fallback
        logger.warning("LaMa 模型未安装，使用 OpenCV Telea 算法（效果较差）")
        return _inpaint_opencv(frame_dir, mask_dir, output_dir)

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
        try:
            img = cv2.imread(os.path.join(frame_dir, fname))
            mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            result = model(img, mask)
            cv2.imwrite(out_path, result)
            count += 1
        except Exception as e:
            logger.warning(f"修复帧 {fname} 失败: {e}")

        if count % 50 == 0:
            logger.info(f"  LaMa 修复进度: {count}/{len(frames)}")

    logger.info(f"LaMa 修复完成: {count}/{len(frames)} 帧")
    return count


def _inpaint_opencv(frame_dir: str, mask_dir: str, output_dir: str) -> int:
    """OpenCV 修复（简单 fallback）"""
    import cv2
    import numpy as np

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
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(out_path, result)
        count += 1
    return count


def run_pipeline(
    frame_dir: str,
    config: Dict[str, Any],
    output_dir: str = "frames_denoised",
) -> Optional[str]:
    """
    去水印流水线
    流程: 自动检测水印区域 -> 生成掩码 -> LaMa/ProPainter 修复
    """
    wm_cfg = config["watermark_removal"]
    if not wm_cfg["enabled"]:
        logger.info("去水印已禁用")
        return frame_dir

    # 1. 确定水印区域
    bbox = tuple(wm_cfg.get("watermark_bbox", []))
    if not bbox or len(bbox) != 4:
        logger.info("自动检测水印区域...")
        bbox = detect_watermark_area(frame_dir)
        if not bbox:
            logger.warning("无法检测水印区域，跳过去水印")
            return frame_dir
    logger.info(f"水印区域: {bbox}")

    # 2. 生成掩码
    mask_dir = ensure_dir(os.path.join(output_dir, "..", "masks"))
    # 用绝对路径
    mask_dir = os.path.abspath(os.path.join(frame_dir, "..", "masks"))
    create_masks_for_frames(frame_dir, mask_dir, bbox)

    # 3. 修复
    out_dir = ensure_dir(output_dir)
    method = wm_cfg.get("method", "lama")

    if method == "propainter":
        logger.info("使用 ProPainter 修复（效果更好，但更慢）...")
        count = _run_propainter(frame_dir, mask_dir, out_dir)
    else:
        count = inpaint_with_lama(frame_dir, mask_dir, out_dir)

    if count == 0:
        logger.error("去水印失败")
        return None

    return out_dir


def _run_propainter(frame_dir: str, mask_dir: str, output_dir: str) -> int:
    """调用 ProPainter 进行视频修复"""
    try:
        result = subprocess.run(
            ["python3", "tools/ProPainter/inference_propainter.py",
             "--video", frame_dir,
             "--mask", mask_dir,
             "--output", output_dir],
            capture_output=True, text=True, timeout=3600
        )
        logger.info(f"ProPainter 输出: {result.stdout[-300:]}")
        return 1
    except Exception as e:
        logger.error(f"ProPainter 调用失败: {e}")
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频去水印")
    parser.add_argument("--frames-dir", required=True, help="帧目录")
    parser.add_argument("--output-dir", default="frames_denoised", help="输出目录")
    parser.add_argument("--config", default="config/settings.yaml", help="配置文件")
    parser.add_argument("--bbox", nargs=4, type=int, default=None,
                        help="水印区域 x y w h (像素)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bbox:
        config["watermark_removal"]["watermark_bbox"] = list(args.bbox)
        config["watermark_removal"]["detect_watermark"] = False

    result = run_pipeline(args.frames_dir, config, args.output_dir)
    if result:
        print(f"\n[OK] 去水印完成！输出: {result}")
    else:
        print(f"\n[FAIL] 去水印失败")
        sys.exit(1)
