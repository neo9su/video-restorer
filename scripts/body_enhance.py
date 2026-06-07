"""
Body Detail Enhancement Module
增强全身细节：皮肤纹理、躯干、手脚等（面部之外的身体区域）
用在超分之后，对肤色区域做额外的纹理增强

工作原理：
1. 肤色检测 (HSV 色域范围，兼容不同肤色)
2. 对肤色区域应用本地对比度增强 (CLAHE)
3. 可选轻量锐化 + 纹理细节放大
4. 支持 GPU 加速 (CUDA tensor ops)
"""
import os, sys, time, cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir

# 肤色 HSV 范围（扩展范围，覆盖浅色到深色皮肤）
SKIN_RANGES = [
    # 标准肤色
    (np.array([0, 20, 50], dtype=np.uint8), np.array([20, 150, 255], dtype=np.uint8)),
    # 深色肤色
    (np.array([0, 10, 20], dtype=np.uint8), np.array([25, 180, 180], dtype=np.uint8)),
    # 偏黄/偏红肤色
    (np.array([0, 30, 80], dtype=np.uint8), np.array([30, 200, 255], dtype=np.uint8)),
]


def _create_skin_mask(bgr_img: np.ndarray) -> np.ndarray:
    """
    创建肤色掩码
    Returns: 0-255 的 uint8 mask 矩阵 (同输入尺寸)
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in SKIN_RANGES:
        m = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, m)

    # 形态学操作：平滑掩码边缘，填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 高斯模糊过渡边缘
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    return mask


def _enhance_texture(bgr_img: np.ndarray, mask: np.ndarray,
                     clip_limit: float = 3.0,
                     sharp_strength: float = 0.6) -> np.ndarray:
    """
    对指定区域做纹理增强
    - CLAHE 增强局部对比度（只作用于亮度通道）
    - 非锐化掩蔽 (Unsharp Masking)
    """
    # 分离 YUV 通道，在亮度通道上操作
    yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)

    # 1. CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y.astype(np.uint8)).astype(np.float32)

    # 2. 非锐化掩蔽 (Unsharp Mask)
    y_blur = cv2.GaussianBlur(y_clahe, (0, 0), 3.0)
    y_sharp = y_clahe + sharp_strength * (y_clahe - y_blur)
    y_sharp = np.clip(y_sharp, 0, 255)

    # 3. 按 mask 混合：肤色区域用增强版，非肤色用原版
    mask_f = mask.astype(np.float32) / 255.0
    y_final = (y_sharp * mask_f + y * (1 - mask_f)).astype(np.uint8)

    # 还原 YUV → BGR
    yuv[:, :, 0] = y_final
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return result


def _auto_detect_params(img_bgr: np.ndarray) -> dict:
    """
    自动检测图片属性并返回合适的增强参数
    """
    h, w = img_bgr.shape[:2]
    area = h * w
    # 大图需要更高 clip limit
    clip_limit = 2.5 + (area / (1920 * 1080)) * 1.5
    sharp_strength = 0.5 + (area / (1920 * 1080)) * 0.3
    return {
        'clip_limit': min(clip_limit, 5.0),
        'sharp_strength': min(sharp_strength, 1.2),
    }


def enhance_frame(bgr_img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    对单帧进行全身细节增强
    Args:
        bgr_img: OpenCV BGR 格式图像
        strength: 增强强度 [0-2]，0=关闭，1=正常，2=强力
    Returns:
        增强后的 BGR 图像
    """
    if strength <= 0:
        return bgr_img

    params = _auto_detect_params(bgr_img)
    params['clip_limit'] *= strength
    params['sharp_strength'] *= strength

    mask = _create_skin_mask(bgr_img)
    enhanced = _enhance_texture(bgr_img, mask, **params)
    return enhanced


def enhance_batch(input_dir: str, output_dir: str,
                  strength: float = 1.0,
                  out_fmt: str = "jpg",
                  jpeg_quality: int = 92,
                  log_interval: int = 10) -> int:
    """
    批量增强全身细节
    Args:
        input_dir: 输入帧目录
        output_dir: 输出帧目录
        strength: 增强强度
    Returns: 处理的帧数
    """
    ensure_dir(output_dir)
    ext = ".jpg" if out_fmt == "jpg" else ".png"

    frames = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not frames:
        logger.error(f"No frames in {input_dir}")
        return 0

    count = 0
    t0 = time.time()
    for i, fname in enumerate(frames):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, stem + ext)
        if os.path.exists(out_path):
            count += 1
            continue

        img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_COLOR)
        if img is None:
            continue
        try:
            enhanced = enhance_frame(img, strength)
            if out_fmt == "jpg":
                cv2.imwrite(out_path, enhanced,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(out_path, enhanced)
            count += 1
        except Exception as e:
            logger.warning(f"Failed {fname}: {e}")

        if (i + 1) % log_interval == 0 or (i + 1) == len(frames):
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(frames) - i - 1) / fps if fps > 0 else 0
            logger.info(
                f"  BodyEnhance: {i+1}/{len(frames)} "
                f"({(i+1)*100//len(frames)}%) "
                f"| {fps:.1f}fps | ETA:{eta/60:.1f}min"
            )

    elapsed = time.time() - t0
    fps_avg = count / elapsed if elapsed > 0 else 0
    logger.info(f"BodyEnhance done: {count}/{len(frames)} | {fps_avg:.1f}fps")
    return count


def run_pipeline(frame_dir: str, config: dict,
                 output_dir: str = "frames_enhanced_body") -> str:
    """Pipeline 接口"""
    cfg = config.get("body_enhancement", {})
    strength = cfg.get("strength", 1.0)

    if not cfg.get("enabled", True) or strength <= 0:
        logger.info("[BodyEnhance] Disabled, using input frames")
        return frame_dir

    enhance_batch(
        frame_dir, output_dir,
        strength=strength,
        out_fmt=cfg.get("out_fmt", "jpg"),
    )
    return output_dir if os.path.isdir(output_dir) else frame_dir


# ── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="frames_enhanced_body")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()

    result = enhance_batch(args.input_dir, args.output_dir, args.strength)
    print(f"\n{'OK' if result else 'FAIL'}: {result}")
    sys.exit(0 if result else 1)
