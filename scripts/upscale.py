"""超分辨率放大模块"""

import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir, load_config


def _create_model(model_name):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    if model_name == "RealESRGAN_x4plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    elif model_name == "RealESRGAN_x2plus":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2
    elif model_name == "realesr-animevideov3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"), 4
    elif model_name == "realesr-general-x4v3":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"), 4
    else:
        raise ValueError(f"Unknown model: {model_name}")


def upscale_frames(input_dir, output_dir, scale=2, model_name="RealESRGAN_x4plus", tile_size=256, tile_pad=10, device="cuda"):
    try:
        from realesrgan import RealESRGANer
        from basicsr.utils.download_util import load_file_from_url
        import cv2, numpy as np, torch
    except ImportError as e:
        logger.error(f"Missing dep: {e}")
        return 0

    ensure_dir(output_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    net, netscale = _create_model(model_name)
    logger.info(f"Model: {model_name} (netscale={netscale}, target_scale={scale}x, tile={tile_size})")

    # 找模型文件
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "realesrgan")
    model_path = os.path.join(models_dir, f"{model_name}.pth")
    if not os.path.isfile(model_path):
        logger.info(f"Downloading {model_name}...")
        model_path = load_file_from_url(
            url=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth",
            model_dir=models_dir, progress=True, file_name=f"{model_name}.pth")

    upsampler = RealESRGANer(
        scale=netscale, model_path=model_path, model=net,
        tile=tile_size, tile_pad=tile_pad, pre_pad=0, half=False, gpu_id=0 if device.type == "cuda" else -1,
    )

    frames = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not frames:
        logger.error(f"No frames in {input_dir}")
        return 0

    sample = cv2.imread(os.path.join(input_dir, frames[0]))
    h, w = sample.shape[:2]
    logger.info(f"Resolution: {w}x{h} -> {w*scale}x{h*scale}")

    logger.info(f"Upscaling {len(frames)} frames (3090 24G)...")
    count = 0
    for i, fname in enumerate(frames):
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            count += 1
            continue
        try:
            img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_COLOR)
            if img is None:
                continue
            output, _ = upsampler.enhance(img, outscale=scale)
            cv2.imwrite(out_path, output)
            count += 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"OOM on {fname}, retrying with smaller tile...")
                torch.cuda.empty_cache()
                import gc; gc.collect()
                return upscale_frames(input_dir, output_dir, scale, model_name, max(64, tile_size // 2), tile_pad, device)
            else:
                logger.warning(f"Failed {fname}: {e}")
        except Exception as e:
            logger.warning(f"Failed {fname}: {e}")
        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            logger.info(f"  Progress: {i+1}/{len(frames)}")

    logger.info(f"Done: {count}/{len(frames)} frames")
    return count


def run_pipeline(frame_dir, config, output_dir="frames_upscaled"):
    sr_cfg = config["super_resolution"]
    if not sr_cfg["enabled"]:
        logger.info("Super-resolution disabled")
        return frame_dir
    out_dir = ensure_dir(output_dir)
    count = upscale_frames(frame_dir, out_dir, scale=sr_cfg["scale"], model_name=sr_cfg["model"], tile_size=sr_cfg["tile_size"], tile_pad=sr_cfg["tile_pad"])
    return out_dir if count > 0 else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video super-resolution")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", default="frames_upscaled")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.scale:
        config["super_resolution"]["scale"] = args.scale
    result = run_pipeline(args.frames_dir, config, args.output_dir)
    print(f"\n{[OK] if result else [FAIL]} Done: {result}")
    sys.exit(0 if result else 1)
