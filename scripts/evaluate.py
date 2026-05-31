"""质量评估模块 - PSNR / SSIM / LPIPS"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger


def compare_folders(
    folder_a: str,
    folder_b: str,
    use_lpips: bool = False,
) -> List[Tuple[str, float, float, Optional[float]]]:
    """
    比较两个帧目录的质量指标
    返回: [(filename, psnr, ssim, lpips), ...]
    """
    try:
        import cv2
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        return []

    frames = sorted([
        f for f in os.listdir(folder_a)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # LPIPS (感知质量)
    lpips_fn = None
    if use_lpips:
        try:
            import lpips
            import torch
            lpips_fn = lpips.LPIPS(net="alex").cuda()
            logger.info("LPIPS 模型已加载")
        except Exception as e:
            logger.warning(f"LPIPS 加载失败: {e}")

    results = []
    for fname in frames:
        path_a = os.path.join(folder_a, fname)
        path_b = os.path.join(folder_b, fname)
        if not os.path.exists(path_b):
            continue

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            continue

        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:h, :w]
        img_b = img_b[:h, :w]

        # PSNR
        psnr = peak_signal_noise_ratio(img_a, img_b)

        # SSIM (灰度)
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        ssim = structural_similarity(gray_a, gray_b)

        # LPIPS
        lpips_score = None
        if lpips_fn is not None:
            import torch
            from PIL import Image
            from torchvision import transforms
            to_tensor = transforms.ToTensor()
            tensor_a = to_tensor(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
            tensor_b = to_tensor(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
            lpips_score = lpips_fn(tensor_a, tensor_b).item()

        results.append((fname, psnr, ssim, lpips_score))

    return results


def print_results(results: List[Tuple[str, float, float, Optional[float]]]):
    """打印评估结果"""
    if not results:
        print("无评估结果")
        return

    psnr_vals = [r[1] for r in results]
    ssim_vals = [r[2] for r in results]
    lpips_vals = [r[3] for r in results if r[3] is not None]

    print(f"\n{'='*55}")
    print(f"  质量评估报告 ({len(results)} 帧)")
    print(f"{'='*55}")
    print(f"  {'指标':<12} {'平均值':<10} {'最小值':<10} {'最大值':<10}")
    print(f"  {'-'*42}")
    print(f"  {'PSNR (dB)':<12} {sum(psnr_vals)/len(psnr_vals):<10.2f} {min(psnr_vals):<10.2f} {max(psnr_vals):<10.2f}")
    print(f"  {'SSIM':<12} {sum(ssim_vals)/len(ssim_vals):<10.4f} {min(ssim_vals):<10.4f} {max(ssim_vals):<10.4f}")
    if lpips_vals:
        print(f"  {'LPIPS (↓)':<12} {sum(lpips_vals)/len(lpips_vals):<10.4f} {min(lpips_vals):<10.4f} {max(lpips_vals):<10.4f}")
    print(f"{'='*55}")

    # PS: PSNR > 30 通常表示高质量，SSIM > 0.95 很好
    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)
    print(f"\n  说明:")
    print(f"  - PSNR > 30dB = 高质量, > 35dB = 几乎无损")
    print(f"  - SSIM > 0.95 = 极好, 0.90~0.95 = 好")
    if avg_psnr < 25:
        print(f"  ⚠️  PSNR 偏低，可能降质明显")
    print(f"")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频质量评估")
    parser.add_argument("reference", help="参考帧目录 (原始)")
    parser.add_argument("distorted", help="对比帧目录 (处理后)")
    parser.add_argument("--lpips", action="store_true", help="使用 LPIPS 感知指标")
    args = parser.parse_args()

    results = compare_folders(args.reference, args.distorted, args.lpips)
    print_results(results)
