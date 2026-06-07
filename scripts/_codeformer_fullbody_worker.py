"""
CodeFormer full-frame worker - 不检测人脸，直接对全帧做 CodeFormer 增强
用于全身细节提升（皮肤纹理、躯干等）
高 fidelity_weight (0.85-0.95) 保留原结构，仅增强纹理细节
Usage: python3 _codeformer_fullbody_worker.py <params.json>
"""
import sys, os, json, time, cv2, torch
import numpy as np

params = json.load(open(sys.argv[1]))
frame_dir    = params["frame_dir"]
output_dir   = params["output_dir"]
fidelity     = params.get("fidelity_weight", 0.9)
log_interval = params.get("log_interval", 50)
cf_root      = params.get("cf_root", "/home/neo/CodeFormer")

sys.path.insert(0, cf_root)
os.makedirs(output_dir, exist_ok=True)

print(f"CF-FULLBODY worker start: fidelity={fidelity} src={frame_dir}", flush=True)

from basicsr.archs.codeformer_arch import CodeFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
    n_layers=9, connect_list=["32","64","128","256"]).to(device).eval()

ckpt = torch.load(os.path.join(cf_root, "weights/CodeFormer/codeformer.pth"),
    map_location=device, weights_only=False)
net.load_state_dict(ckpt["params_ema"])

frames = sorted([f for f in os.listdir(frame_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
total = len(frames)
print(f"Total frames: {total}", flush=True)

count = skipped = failed = 0
t0 = time.time()

for i, fname in enumerate(frames):
    out_path = os.path.join(output_dir, fname)

    # 断点续处理
    alt = os.path.join(output_dir, os.path.splitext(fname)[0] + ".png")
    if os.path.exists(out_path) or os.path.exists(alt):
        skipped += 1
        count += 1
        continue

    img_bgr = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_COLOR)
    if img_bgr is None:
        failed += 1
        continue

    try:
        h, w = img_bgr.shape[:2]
        # 填充到32倍数 (CodeFormer需要整除32)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            padded = img_bgr
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_in = (torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1)

        # 如果图片太大做分块处理（避免 OOM）
        MAX_PIXELS = 512 * 512  # CodeFormer 训练尺寸
        total_pixels = h * w

        if total_pixels <= MAX_PIXELS * 4:
            # 直接全帧推理
            with torch.no_grad():
                out = net(t_in, w=fidelity, adain=True)[0]
            out_np = ((out.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        else:
            # 大图分块处理
            out_bgr = np.zeros_like(img_bgr)
            TILE = 512
            overlap = 64
            step = TILE - overlap
            weight = np.hanning(TILE).reshape(-1, 1) * np.hanning(TILE).reshape(1, -1)
            weight = weight[None, :, :] if False else weight
            acc = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, step):
                for x in range(0, w, step):
                    y1, y2 = y, min(y + TILE, h)
                    x1, x2 = x, min(x + TILE, w)
                    tile = img_rgb[y1:y2, x1:x2, :]
                    th, tw = tile.shape[:2]
                    tile_t = (torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1)
                    with torch.no_grad():
                        out_tile = net(tile_t, w=fidelity, adain=True)[0]
                    tile_out = ((out_tile.squeeze(0).clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                    tile_bgr = cv2.cvtColor(tile_out, cv2.COLOR_RGB2BGR).astype(np.float32)
                    w_tile = weight[:th, :tw] if weight.shape[:2] != (th, tw) else weight[:th, :tw]
                    w_3 = np.stack([w_tile] * 3, axis=-1)
                    out_bgr[y1:y2, x1:x2] += tile_bgr * w_3
                    acc[y1:y2, x1:x2] += w_tile
                    del tile_t, out_tile
                    torch.cuda.empty_cache()

            acc = np.maximum(acc, 1e-5)
            out_bgr = (out_bgr / np.stack([acc] * 3, axis=-1)).clip(0, 255).astype(np.uint8)

        # 混合：fidelity 控制保留原图的比例
        if pad_h > 0 or pad_w > 0:
            out_bgr = out_bgr[:h, :w]
        result = (img_bgr.astype(np.float32) * fidelity
                  + out_bgr.astype(np.float32) * (1 - fidelity)).clip(0, 255).astype(np.uint8)

        cv2.imwrite(out_path, result)
        count += 1

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"  OOM {fname}, copying original", flush=True)
        else:
            print(f"  WARN {fname}: {e}", flush=True)
        failed += 1
        cv2.imwrite(out_path, img_bgr)
    except Exception as e:
        cv2.imwrite(out_path, img_bgr)
        failed += 1
        print(f"  WARN {fname}: {e}", flush=True)

    if (i + 1) % log_interval == 0 or (i + 1) == total:
        elapsed = time.time() - t0
        fps = (i + 1 - skipped) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / fps if fps > 0 else 0
        print(f"  Progress: {count}/{total} ({count * 100 // total}%) "
              f"| {fps:.2f}fps | ETA:{eta/60:.1f}min", flush=True)

del net
torch.cuda.empty_cache()
elapsed = time.time() - t0
fps_avg = (count - skipped) / elapsed if elapsed > 0 else 0
print(f"CF-FULLBODY done: {count}/{total} frames | {fps_avg:.2f}fps | failed={failed}", flush=True)
