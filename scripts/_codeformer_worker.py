"""
CodeFormer worker - 在 codeformer env 里运行，避免与 video-restorer env 的 basicsr 冲突
Usage: python3 _codeformer_worker.py <params.json>
"""
import sys, os, json, time, cv2, torch, numpy as np

params = json.load(open(sys.argv[1]))
frame_dir    = params["frame_dir"]
output_dir   = params["output_dir"]
fidelity     = params.get("fidelity_weight", 0.7)
log_interval = params.get("log_interval", 50)
cf_root      = params.get("cf_root", "/home/neo/CodeFormer")

sys.path.insert(0, cf_root)
os.makedirs(output_dir, exist_ok=True)

print(f"CF worker start: fidelity={fidelity} src={frame_dir}", flush=True)

from basicsr.archs.codeformer_arch import CodeFormer
from facelib.utils.face_restoration_helper import FaceRestoreHelper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
    n_layers=9, connect_list=["32","64","128","256"]).to(device).eval()
ckpt = torch.load(os.path.join(cf_root,"weights/CodeFormer/codeformer.pth"),
    map_location=device, weights_only=False)
net.load_state_dict(ckpt["params_ema"])

fh = FaceRestoreHelper(1, face_size=512, crop_ratio=(1,1),
    det_model="retinaface_resnet50", save_ext="png",
    use_parse=True, device=device)

frames = sorted([f for f in os.listdir(frame_dir)
    if f.lower().endswith((".png",".jpg",".jpeg"))])
total = len(frames)
print(f"Total frames: {total}", flush=True)

count = skipped = faces_total = failed = 0
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
        fh.clean_all()
        fh.read_image(img_bgr)
        n = fh.get_face_landmarks_5(only_center_face=False,
            resize=640, eye_dist_threshold=5)

        if n == 0:
            # 无人脸直接复制原图
            cv2.imwrite(out_path, img_bgr)
            count += 1
            continue

        faces_total += n
        fh.align_warp_face()

        for face in fh.cropped_faces:
            ft = (torch.from_numpy(face.astype(np.float32) / 255.0)
                  .permute(2,0,1).unsqueeze(0).to(device)) * 2 - 1
            with torch.no_grad():
                out = net(ft, w=fidelity, adain=True)[0]
            out_np = ((out.squeeze(0).clamp(-1,1)+1)/2*255).byte().permute(1,2,0).cpu().numpy()
            fh.add_restored_face(out_np)

        fh.get_inverse_affine(None)
        restored = fh.paste_faces_to_input_image(upsample_img=None)
        cv2.imwrite(out_path, restored)
        count += 1

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            import gc
            torch.cuda.empty_cache(); gc.collect()
            # OOM 时直接复制原图（不中断整体流程）
            print(f"  OOM {fname}, copying original", flush=True)
        else:
            print(f"  WARN {fname}: {e}", flush=True)
        cv2.imwrite(out_path, img_bgr)
        count += 1
        failed += 1
    except Exception as e:
        # 失败时复制原图，不中断
        cv2.imwrite(out_path, img_bgr)
        count += 1
        failed += 1
        print(f"  WARN {fname}: {e}", flush=True)

    if (i+1) % log_interval == 0 or (i+1) == total:
        elapsed = time.time() - t0
        fps = (i+1-skipped) / elapsed if elapsed > 0 else 0
        eta  = (total-i-1) / fps if fps > 0 else 0
        print(f"  Progress: {count}/{total} ({count*100//total}%) "
              f"| faces={faces_total} | {fps:.2f}fps | ETA:{eta/60:.1f}min", flush=True)

del net, fh
torch.cuda.empty_cache()
elapsed = time.time() - t0
fps_avg = (count-skipped)/elapsed if elapsed > 0 else 0
print(f"CF done: {count}/{total} frames | faces={faces_total} | "
      f"{fps_avg:.2f}fps | failed={failed}", flush=True)
