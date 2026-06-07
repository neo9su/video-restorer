"""超分辨率放大模块 - 高性能版 (FP16 + 多线程I/O)"""

import os
import sys
import time
import queue
import threading
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir, load_config


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "realesrgan"
)

# 已知模型配置：(arch, num_block, scale)
MODEL_CONFIGS = {
    "realesr-animevideov3": ("srvgg", 16, 4),
    "realesr-general-x4v3": ("srvgg", 32, 4),
    "RealESRGAN_x4plus":    ("rrdb",  23, 4),
    "RealESRGAN_x2plus":    ("rrdb",  23, 2),
    "4x-UltraSharp":        ("rrdb_old", 23, 4),  # old-arch ESRGAN
    "BSRGAN":               ("rrdb_old", 23, 4),  # old-arch ESRGAN
}

MODEL_URLS = {
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "RealESRGAN_x4plus":    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus":    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


def _convert_rrdb_old_to_basicsr(ckpt):
    """将旧版 ESRGAN state_dict (model.0, model.1.sub.N) 转换成 basicsr RRDBNet 格式"""
    new = {}
    for k, v in ckpt.items():
        nk = k
        # conv_first
        nk = re.sub(r'^model\.0\.', 'conv_first.', nk)
        # RRDB body blocks (model.1.sub.0..21 是RRDB块，sub.22 是trunk conv)
        m = re.match(r'^model\.1\.sub\.(\d+)\.(.*)', nk)
        if m:
            idx = int(m.group(1))
            rest = m.group(2)
            max_rrdb = 22  # 0-21 are RRDB, 22 is trunk
            if idx < max_rrdb:
                # RDB1.conv1.0.weight -> rdb1.conv1.weight
                rest2 = re.sub(r'(RDB\d+)\.(conv\d+)\.0\.(weight|bias)',
                    lambda mm: mm.group(1).lower()+'.'+mm.group(2)+'.'+mm.group(3), rest)
                nk = 'body.%d.%s' % (idx, rest2)
            else:
                # trunk conv
                nk = 'conv_body.' + rest.replace('0.weight', 'weight').replace('0.bias', 'bias')
            new[nk] = v
            continue
        # tail layers
        tail_map = [
            (r'^model\.3\.', 'conv_body.'),
            (r'^model\.6\.', 'conv_up1.'),
            (r'^model\.8\.', 'conv_up2.'),
            (r'^model\.10\.', 'conv_last.'),
        ]
        for pat, rep in tail_map:
            if re.match(pat, nk):
                nk = re.sub(pat, rep, nk)
                break
        # RRDB_trunk (BSRGAN format)
        m2 = re.match(r'^RRDB_trunk\.(\d+)\.(.*)', nk)
        if m2:
            idx = int(m2.group(1))
            rest = m2.group(2)
            rest2 = re.sub(r'(RDB\d+)\.(conv\d+)\.(weight|bias)',
                lambda mm: mm.group(1).lower()+'.'+mm.group(2)+'.'+mm.group(3), rest)
            nk = 'body.%d.%s' % (idx, rest2)
        # conv_first / conv_body / conv_up / conv_last (BSRGAN format - already named)
        new[nk] = v

    # 确保 conv_hr 存在（basicsr 要求，旧ESRGAN没有）
    if 'conv_hr.weight' not in new and 'conv_up2.weight' in new:
        new['conv_hr.weight'] = new['conv_up2.weight'].clone()
        new['conv_hr.bias']   = new['conv_up2.bias'].clone()
    return new


def _load_model(model_name, model_path, half, gpu_id=0):
    """加载模型，自动处理新旧格式"""
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Add to MODEL_CONFIGS.")

    arch, num_block, netscale = MODEL_CONFIGS[model_name]

    if arch == "srvgg":
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_conv=num_block, upscale=netscale, act_type='prelu')
        up = RealESRGANer(scale=netscale, model_path=model_path, model=net,
                          tile=0, half=half, gpu_id=gpu_id)
        return up, netscale

    elif arch == "rrdb":
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=num_block, num_grow_ch=32, scale=netscale)
        up = RealESRGANer(scale=netscale, model_path=model_path, model=net,
                          tile=0, half=half, gpu_id=gpu_id)
        return up, netscale

    elif arch == "rrdb_old":
        # 旧格式：手动转换 key 后加载
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                      num_block=num_block, num_grow_ch=32, scale=netscale)
        raw_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        new_ckpt = _convert_rrdb_old_to_basicsr(raw_ckpt)
        missing, unexpected = net.load_state_dict(new_ckpt, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (e.g. {missing[:2]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (e.g. {unexpected[:2]})")
        net = net.cuda().eval()
        if half:
            net = net.half()

        # 包装成统一接口
        class _DirectUpsampler:
            def __init__(self, net, scale, half):
                self._net = net
                self._scale = scale
                self._half = half
            def enhance(self, img_bgr, outscale=2):
                import cv2, numpy as np
                h, w = img_bgr.shape[:2]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda()
                if self._half:
                    t = t.half()
                with torch.no_grad():
                    out = self._net(t)
                out_np = (out.squeeze(0).float().clamp(0, 1).permute(1, 2, 0)
                          .cpu().numpy() * 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
                if outscale != self._scale:
                    th, tw = int(h * outscale), int(w * outscale)
                    out_bgr = cv2.resize(out_bgr, (tw, th), interpolation=cv2.INTER_LANCZOS4)
                return out_bgr, None

        return _DirectUpsampler(net, netscale, half), netscale

    else:
        raise ValueError(f"Unknown arch: {arch}")


def _prefetch_reader(frame_paths, read_q, prefetch=8):
    import cv2
    for p in frame_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        read_q.put((p, img))
        while read_q.qsize() > prefetch:
            time.sleep(0.005)
    read_q.put(None)


def _async_writer(write_q, jpeg_quality=92):
    import cv2
    while True:
        item = write_q.get()
        if item is None:
            break
        out_path, img = item
        if out_path.endswith('.jpg') or out_path.endswith('.jpeg'):
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(out_path, img)
        write_q.task_done()


def upscale_frames(
    input_dir,
    output_dir,
    scale=2,
    model_name="RealESRGAN_x4plus",
    tile_size=0,
    tile_pad=10,
    use_half=True,
    out_fmt="jpg",
    jpeg_quality=92,
    prefetch=8,
    log_interval=10,
):
    import cv2, torch

    ensure_dir(output_dir)
    cuda_ok = torch.cuda.is_available()
    half = use_half and cuda_ok
    logger.info(f"Device: cuda | FP16: {half} | tile: {tile_size or 'full-frame'} | out_fmt: {out_fmt}")

    frames = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not frames:
        logger.error(f"No frames in {input_dir}")
        return 0

    sample = cv2.imread(os.path.join(input_dir, frames[0]))
    h, w = sample.shape[:2]

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.isfile(model_path) or os.path.getsize(model_path) < 1000:
        from basicsr.utils.download_util import load_file_from_url
        url = MODEL_URLS.get(model_name, "")
        if not url:
            logger.error(f"Model not found and no download URL: {model_name}")
            return 0
        model_path = load_file_from_url(url=url, model_dir=MODELS_DIR,
                                         progress=True, file_name=f"{model_name}.pth")

    upsampler, netscale = _load_model(model_name, model_path, half)
    logger.info(f"Model: {model_name} (netscale={netscale}, target_scale={scale}x)")

    ext = ".jpg" if out_fmt == "jpg" else ".png"
    todo = []
    skipped = 0
    for fname in frames:
        stem = os.path.splitext(fname)[0]
        done = False
        for e in [ext, ".png" if ext == ".jpg" else ".jpg"]:
            if os.path.exists(os.path.join(output_dir, stem + e)):
                done = True
                break
        if done:
            skipped += 1
        else:
            todo.append((os.path.join(input_dir, fname),
                         os.path.join(output_dir, stem + ext)))

    logger.info(f"Resolution: {w}x{h} -> {w*scale}x{h*scale} | "
                f"Total: {len(frames)} | Skip: {skipped} | Todo: {len(todo)}")
    if not todo:
        logger.info("All frames already done.")
        return len(frames)

    write_q = queue.Queue(maxsize=16)
    writer_th = threading.Thread(target=_async_writer, args=(write_q, jpeg_quality), daemon=True)
    writer_th.start()

    in_paths = [t[0] for t in todo]
    out_map = {t[0]: t[1] for t in todo}
    read_q = queue.Queue(maxsize=prefetch + 4)
    reader_th = threading.Thread(target=_prefetch_reader,
                                  args=(in_paths, read_q, prefetch), daemon=True)
    reader_th.start()

    count = skipped
    failed = 0
    t0 = time.time()

    for i, _ in enumerate(todo):
        item = read_q.get()
        if item is None:
            break
        in_path, img = item
        if img is None:
            failed += 1
            continue
        try:
            output, _ = upsampler.enhance(img, outscale=scale)
            write_q.put((out_map[in_path], output))
            count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                logger.warning("OOM, retrying with tile=512")
                write_q.put(None)
                writer_th.join()
                return upscale_frames(input_dir, output_dir, scale, model_name,
                                      512, tile_pad, use_half, out_fmt,
                                      jpeg_quality, prefetch, log_interval)
            logger.warning(f"Failed {os.path.basename(in_path)}: {e}")
            failed += 1
        except Exception as e:
            logger.warning(f"Failed {os.path.basename(in_path)}: {e}")
            failed += 1

        if (i + 1) % log_interval == 0 or (i + 1) == len(todo):
            done_total = skipped + i + 1
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i - 1) / fps if fps > 0 else 0
            logger.info(
                f"  Progress: {done_total}/{len(frames)} "
                f"({done_total * 100 // len(frames)}%) "
                f"| {fps:.2f}fps | ETA:{eta/60:.1f}min"
            )

    write_q.put(None)
    writer_th.join()
    elapsed = time.time() - t0
    fps_avg = len(todo) / elapsed if elapsed > 0 else 0
    logger.info(f"Done: {count}/{len(frames)} | avg {fps_avg:.2f}fps | failed: {failed}")
    return count


def run_pipeline(frame_dir, config, output_dir="frames_upscaled"):
    sr_cfg = config["super_resolution"]
    if not sr_cfg.get("enabled", True):
        logger.info("Super-resolution disabled")
        return frame_dir
    out_dir = ensure_dir(output_dir)
    tile = sr_cfg.get("tile_size", 0)
    if tile <= 256:
        tile = 0
    count = upscale_frames(
        frame_dir, out_dir,
        scale=sr_cfg["scale"],
        model_name=sr_cfg["model"],
        tile_size=tile,
        tile_pad=sr_cfg.get("tile_pad", 10),
        use_half=True,
        out_fmt="jpg",
        jpeg_quality=92,
        prefetch=8,
        log_interval=10,
    )
    return out_dir if count > 0 else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", default="frames_upscaled")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.scale:
        config["super_resolution"]["scale"] = args.scale
    result = run_pipeline(args.frames_dir, config, args.output_dir)
    print(f"\n{'OK' if result else 'FAIL'}: {result}")
    sys.exit(0 if result else 1)
