"""
人脸增强模块 - CodeFormer
通过 subprocess 在独立 codeformer env 运行，避免 basicsr 版本冲突
"""
import os, sys, subprocess, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import logger, ensure_dir

CF_PYTHON  = "/home/neo/miniconda3/envs/codeformer/bin/python3"
CF_ROOT    = "/home/neo/CodeFormer"
CF_SCRIPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_codeformer_worker.py")


def run_pipeline(frame_dir, config, output_dir="frames_face_enhanced"):
    face_cfg = config.get("face_enhancement", {})
    if not face_cfg.get("enabled", False):
        logger.info("Face enhancement disabled, skipping")
        return frame_dir

    out_dir = ensure_dir(output_dir)
    fidelity   = face_cfg.get("fidelity_weight", 0.7)
    log_interval = face_cfg.get("log_interval", 50)

    logger.info(f"CodeFormer: fidelity={fidelity} | {frame_dir} -> {out_dir}")

    # 把参数写到临时 json，worker 读取
    params = {
        "frame_dir": frame_dir,
        "output_dir": out_dir,
        "fidelity_weight": fidelity,
        "log_interval": log_interval,
        "cf_root": CF_ROOT,
    }
    params_file = tempfile.mktemp(suffix=".json")
    with open(params_file, "w") as f:
        json.dump(params, f)

    cmd = [CF_PYTHON, CF_SCRIPT, params_file]
    # timeout=None，但用 communicate() 确保 pipe 不死锁
    proc = subprocess.Popen(
        cmd, cwd=CF_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    stdout_data, _ = proc.communicate()  # 正确等待并回收所有输出
    class _Result:
        returncode = proc.returncode
        stdout = stdout_data
    result = _Result()

    os.unlink(params_file)

    # 转发 worker 日志
    for line in result.stdout.splitlines():
        if line.strip():
            logger.info(f"[CF] {line}")

    if result.returncode != 0:
        logger.warning(f"CodeFormer exited {result.returncode}, using original frames")
        return frame_dir

    done = len([f for f in os.listdir(out_dir) if f.lower().endswith((".png",".jpg"))])
    if done == 0:
        logger.warning("CodeFormer produced 0 frames, using original")
        return frame_dir

    logger.info(f"CodeFormer done: {done} frames in {out_dir}")
    return out_dir
