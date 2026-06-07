"""
Video Restorer Pipeline
支持：断点续传（checkpoint）、每步可开关、动态配置覆盖
"""
import os, sys, time, argparse, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, ensure_dir, logger, run_ffmpeg


def _count_frames(d: Path, exts=('jpg', 'jpeg', 'png')) -> int:
    return sum(len(list(d.glob(f'*.{e}'))) for e in exts)


def extract_frames(input_path: str, output_dir: Path, fps=None) -> int:
    """Step 1: 视频抽帧，支持断点（已有帧则跳过）"""
    output_dir = ensure_dir(output_dir)
    existing = _count_frames(output_dir)
    if existing > 0:
        logger.info(f"  [Step 1] Skip: {existing} frames already exist")
        return existing

    cmd = ['ffmpeg', '-y', '-i', input_path]
    if fps:
        cmd += ['-vf', f'fps={fps}']
    cmd += ['-q:v', '2', str(output_dir / '%08d.jpg')]
    run_ffmpeg(cmd)
    count = _count_frames(output_dir)
    logger.info(f"  [Step 1] Extracted {count} frames")
    return count


def denoise_frames(input_dir: Path, output_dir: Path, config: dict) -> Path:
    """Step 2: 降噪"""
    from enhance import run_pipeline
    dn = config.get('denoising', {})
    if not dn.get('enabled', True) or dn.get('level', 1) == 0:
        logger.info("  [Step 2] Denoising disabled, copying frames")
        output_dir = ensure_dir(output_dir)
        existing = _count_frames(output_dir)
        if existing > 0:
            logger.info(f"  [Step 2] Skip: {existing} already exist")
            return output_dir
        for f in sorted(input_dir.glob('*.jpg')) + sorted(input_dir.glob('*.png')):
            shutil.copy2(f, output_dir / f.name)
        return output_dir

    output_dir = ensure_dir(output_dir)
    existing = _count_frames(output_dir)
    total_in = _count_frames(input_dir)
    if existing >= total_in:
        logger.info(f"  [Step 2] Skip: {existing}/{total_in} already done")
        return output_dir

    result = run_pipeline(str(input_dir), config, str(output_dir))
    return Path(result) if result else output_dir


def watermark_removal_frames(input_dir, output_dir, config):
    """Step 1.5: 去水印"""
    from remove_watermark import run_pipeline as wm_pipeline
    wm = config.get("watermark_removal", {})
    if not wm.get("enabled", False):
        logger.info("  [Step 1.5] Watermark removal disabled")
        return input_dir
    output_dir = ensure_dir(output_dir)
    existing = sum(len(list(output_dir.glob(f"*.{e}"))) for e in ("jpg", "jpeg", "png"))
    total_in = sum(len(list(input_dir.glob(f"*.{e}"))) for e in ("jpg", "jpeg", "png"))
    if existing >= total_in:
        logger.info(f"  [Step 1.5] Skip: {existing}/{total_in} already done")
        return output_dir
    result = wm_pipeline(str(input_dir), config, str(output_dir))
    return Path(result) if result else output_dir


def face_enhance_frames(input_dir: Path, output_dir: Path, config: dict) -> Path:
    """Step 3: 面部增强 (CodeFormer)"""
    fe = config.get('face_enhancement', {})
    if not fe.get('enabled', False):
        logger.info("  [Step 3] Face enhancement disabled")
        return input_dir

    from face_enhance import run_pipeline as run_face_enhancement
    output_dir = ensure_dir(output_dir)
    existing = _count_frames(output_dir)
    total_in = _count_frames(input_dir)
    if existing >= total_in:
        logger.info(f"  [Step 3] Skip: {existing}/{total_in} already done")
        return output_dir

    result = run_face_enhancement(str(input_dir), config, str(output_dir))
    return Path(result) if result else output_dir


def body_enhance_frames(input_dir, output_dir, config):
    """Step 3.5 / 4.5: 全身细节增强"""
    from body_enhance import run_pipeline as body_pipeline
    be = config.get("body_enhancement", {})
    if not be.get("enabled", False):
        logger.info("  [BodyEnhance] Disabled")
        return input_dir
    output_dir = ensure_dir(output_dir)
    existing = sum(len(list(output_dir.glob(f"*.{e}"))) for e in ("jpg", "jpeg", "png"))
    total_in = sum(len(list(input_dir.glob(f"*.{e}"))) for e in ("jpg", "jpeg", "png"))
    if existing >= total_in:
        logger.info(f"  [BodyEnhance] Skip: {existing}/{total_in} already done")
        return output_dir
    result = body_pipeline(str(input_dir), config, str(output_dir))
    return Path(result) if result else output_dir



def upscale_frames_step(input_dir: Path, output_dir: Path, config: dict) -> Path:
    """Step 4: 超分辨率"""
    from upscale import run_pipeline as upscale_pipeline
    sr = config.get('super_resolution', {})
    if not sr.get('enabled', True):
        logger.info("  [Step 4] Super-resolution disabled")
        return input_dir

    output_dir = ensure_dir(output_dir)
    existing = _count_frames(output_dir)
    total_in = _count_frames(input_dir)
    if existing >= total_in:
        logger.info(f"  [Step 4] Skip: {existing}/{total_in} already done")
        return output_dir

    result = upscale_pipeline(str(input_dir), config, str(output_dir))
    return Path(result) if result else output_dir


def assemble_video(frames_dir: Path, input_video: str, output_path: str,
                   config: dict, fps: float = None) -> str:
    """Step 5: 合成视频 + 音轨"""
    output_path = str(output_path)
    if Path(output_path).exists() and Path(output_path).stat().st_size > 1024:
        logger.info(f"  [Step 5] Skip: output already exists")
        return output_path

    # Detect frame extension and fps
    jpgs = sorted(frames_dir.glob('*.jpg'))
    pngs = sorted(frames_dir.glob('*.png'))
    frames = jpgs if jpgs else pngs
    if not frames:
        raise RuntimeError(f"No frames in {frames_dir}")
    ext = 'jpg' if jpgs else 'png'

    if fps is None:
        import subprocess, json as _json
        try:
            r = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_streams', input_video],
                capture_output=True, text=True, timeout=10
            )
            info = _json.loads(r.stdout)
            for s in info.get('streams', []):
                if s.get('codec_type') == 'video':
                    fr = s.get('r_frame_rate', '24/1').split('/')
                    fps = float(fr[0]) / float(fr[1]) if len(fr) == 2 else 24.0
                    break
        except:
            fps = 24.0

    tmp_output = output_path.replace('.mp4', '_noaudio.mp4')

    # Encode frames → video
    cmd_video = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', str(frames_dir / f'%08d.{ext}'),
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        tmp_output
    ]
    run_ffmpeg(cmd_video)

    # Mux with audio if keep_audio enabled
    keep_audio = config.get('keep_audio', True)
    if keep_audio:
        cmd_mux = [
            'ffmpeg', '-y',
            '-i', tmp_output,
            '-i', input_video,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0?',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        try:
            run_ffmpeg(cmd_mux)
        except Exception:
            logger.warning("Audio mux failed, using video-only output")
            shutil.move(tmp_output, output_path)
    else:
        shutil.move(tmp_output, output_path)

    # Clean tmp
    if Path(tmp_output).exists():
        Path(tmp_output).unlink()

    size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
    logger.info(f"  [Step 5] Output: {output_path} ({size//1024//1024}MB)")
    return output_path


def run_pipeline(input_path: str, output_dir: str, config: dict,
                 skip_steps: list = None) -> str:
    """
    主流水线，支持断点续传
    skip_steps: list of step numbers to skip, e.g. [3] to skip face enhancement
    """
    skip_steps = skip_steps or []
    input_path = str(input_path)
    work_dir = Path(output_dir)
    ensure_dir(work_dir)

    # Checkpoint file
    checkpoint = work_dir / '.checkpoint'
    completed_steps = set()
    if checkpoint.exists():
        try:
            completed_steps = set(map(int, checkpoint.read_text().split()))
            logger.info(f"Resuming from checkpoint: completed steps {completed_steps}")
        except:
            pass

    def mark_step(n):
        completed_steps.add(n)
        checkpoint.write_text(' '.join(map(str, sorted(completed_steps))))

    t_start = time.time()
    logger.info(f"[Pipeline] Input: {input_path}")
    logger.info(f"[Pipeline] Work dir: {work_dir}")

    # ── Step 1: Extract ──────────────────────────────────────
    frames_raw = work_dir / 'frames_raw'
    if 1 not in completed_steps and 1 not in skip_steps:
        logger.info("[Step 1] Extracting frames...")
        count = extract_frames(input_path, frames_raw)
        if count == 0:
            raise RuntimeError("No frames extracted")
        mark_step(1)
    else:
        logger.info("[Step 1] Skip (checkpoint or config)")

    # ── Step 1.5: Watermark Removal ──────────────────────────
    frames_nwm = work_dir / 'frames_nowatermark'
    if 15 not in completed_steps and 15 not in skip_steps:
        logger.info("[Step 1.5] Watermark removal...")
        frames_nwm = watermark_removal_frames(frames_raw, frames_nwm, config)
        mark_step(15)
    else:
        logger.info("[Step 1.5] Skip")
        if not frames_nwm.exists():
            frames_nwm = frames_raw

    # ── Step 2: Denoise ──────────────────────────────────────
    frames_denoised = work_dir / 'frames_denoised'
    if 2 not in completed_steps and 2 not in skip_steps:
        logger.info("[Step 2] Denoising...")
        frames_denoised = denoise_frames(frames_nwm, frames_denoised, config)
        mark_step(2)
    else:
        logger.info("[Step 2] Skip")
        if not frames_denoised.exists():
            frames_denoised = frames_nwm

    # ── Step 3: Face Enhancement ─────────────────────────────
    frames_face = work_dir / 'frames_face_enhanced'
    if 3 not in completed_steps and 3 not in skip_steps:
        logger.info("[Step 3] Face enhancement...")
        frames_face = face_enhance_frames(frames_denoised, frames_face, config)
        mark_step(3)
    else:
        logger.info("[Step 3] Skip")
        if not frames_face.exists():
            frames_face = frames_denoised

    # ── Step 4: Upscale ──────────────────────────────────────
    frames_up = work_dir / 'frames_upscaled'
    if 4 not in completed_steps and 4 not in skip_steps:
        logger.info("[Step 4] Upscaling...")
        frames_up = upscale_frames_step(frames_face, frames_up, config)
        mark_step(4)
    else:
        logger.info("[Step 4] Skip")
        if not frames_up.exists():
            frames_up = frames_face

    # ── Step 4.5: Body Detail Enhancement ─────────────────
    frames_body = work_dir / 'frames_body_enhanced'
    body_pos = config.get('body_enhancement', {}).get('position', 'after_upscale')
    frame_input = frames_up if body_pos == 'after_upscale' else frames_face
    body_label = 45 if body_pos == 'after_upscale' else 35
    if body_label not in completed_steps and body_label not in skip_steps:
        logger.info(f"[Step {body_label}] Body detail enhancement...")
        frames_body = body_enhance_frames(frame_input, frames_body, config)
        mark_step(body_label)
    else:
        logger.info(f"[Step {body_label}] Skip")
        if not frames_body.exists():
            frames_body = frame_input

    # ── Step 5: Assemble ─────────────────────────────────────
    task_id = work_dir.name
    output_video = Path(work_dir).parent / f'{task_id}_result.mp4'
    if 5 not in completed_steps and 5 not in skip_steps:
        logger.info("[Step 5] Assembling video...")
        assemble_video(frames_body, input_path, str(output_video), config)
        mark_step(5)
    else:
        logger.info("[Step 5] Skip")

    elapsed = time.time() - t_start
    logger.info(f"[Pipeline] Done in {elapsed/60:.1f}min → {output_video}")
    return str(output_video)


# ── CLI ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Restorer Pipeline')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output-dir', required=True, help='Work directory')
    parser.add_argument('--config', default='config/settings.yaml', help='Config file')
    parser.add_argument('--skip-steps', type=int, nargs='*', default=[],
                        help='Steps to skip (1=extract, 2=denoise, 3=face, 4=upscale, 5=assemble)')
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(args.input, args.output_dir, config, args.skip_steps)
    print(f"\nResult: {result}")
    sys.exit(0 if Path(result).exists() else 1)
