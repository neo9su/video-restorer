"""
Video Restorer Web Server
支持：任务管理、断点续传、优先级队列、GPU状态、日志流
"""
import os, sys, json, uuid, time, shutil, threading, subprocess, logging, re
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response, stream_with_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config, ensure_dir, logger as pipe_logger

# ── Config ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = ensure_dir(BASE_DIR / "webapp" / "uploads")
RESULT_DIR = ensure_dir(BASE_DIR / "webapp" / "results")
TASKS_FILE = BASE_DIR / "webapp" / "tasks.json"
CONFIG_FILE = BASE_DIR / "config" / "settings.yaml"
MAX_UPLOAD_MB = 2048

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

# ── Task Store ───────────────────────────────────────────────
_lock = threading.Lock()

def load_tasks() -> dict:
    try:
        if TASKS_FILE.exists():
            return json.loads(TASKS_FILE.read_text())
    except Exception as e:
        log.error(f"load_tasks: {e}")
    return {}

def save_tasks(tasks: dict):
    with _lock:
        TASKS_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))

def get_task(tid: str) -> dict | None:
    return load_tasks().get(tid)

def update_task(tid: str, **kwargs):
    tasks = load_tasks()
    if tid in tasks:
        tasks[tid].update(kwargs)
        save_tasks(tasks)

# ── Pipeline Runner ──────────────────────────────────────────
_running: dict[str, subprocess.Popen] = {}

def _run_pipeline(tid: str, config_override: dict):
    """后台线程：运行 pipeline，实时捕获日志，更新 tasks.json"""
    task = get_task(tid)
    if not task:
        return

    input_path = task['input_path']
    work_dir   = Path(RESULT_DIR) / tid
    ensure_dir(work_dir)

    # 写临时 config
    import yaml
    base_cfg = load_config(str(CONFIG_FILE))
    # 合并 override
    sr = base_cfg.setdefault('super_resolution', {})
    if 'model' in config_override:    sr['model'] = config_override['model']
    if 'scale' in config_override:    sr['scale'] = config_override['scale']
    if 'use_fp16' in config_override: sr['use_fp16'] = config_override['use_fp16']
    fe = base_cfg.setdefault('face_enhancement', {})
    if 'face_enhancement' in config_override:
        fe.update(config_override['face_enhancement'])
    dn = base_cfg.setdefault('denoising', {})
    if 'denoise' in config_override:  dn['level'] = config_override['denoise']

    tmp_cfg = work_dir / 'config.yaml'
    with open(tmp_cfg, 'w') as f:
        yaml.dump(base_cfg, f, allow_unicode=True)

    python = str(Path(sys.executable))
    pipeline_script = str(BASE_DIR / 'scripts' / 'pipeline.py')
    cmd = [python, pipeline_script,
           '--input', input_path,
           '--output-dir', str(work_dir),
           '--config', str(tmp_cfg)]

    log.info(f"[{tid[:8]}] Starting pipeline: {' '.join(cmd)}")
    update_task(tid, status='processing', step=1, percent=0,
                started_at=int(time.time()), recent_logs=[])

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        _running[tid] = proc

        recent_logs = []
        step_re = re.compile(r'\[Step\s*(\d+)', re.I)
        prog_re = re.compile(r'Progress:\s*(\d+)/(\d+)', re.I)
        fps_re  = re.compile(r'([\d.]+)\s*fps', re.I)
        eta_re  = re.compile(r'ETA:([\d.]+)min', re.I)

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            recent_logs.append(line)
            if len(recent_logs) > 50:
                recent_logs = recent_logs[-50:]

            kwargs = {'recent_logs': recent_logs[-10:]}

            # Step detection
            m = step_re.search(line)
            if m:
                kwargs['step'] = int(m.group(1))

            # Progress detection
            m = prog_re.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                task_data = get_task(tid) or {}
                cur_step = task_data.get('step', 1)
                # Each step ~= 20% of total
                step_pct = (cur_step - 1) * 20
                frame_pct = (done / total * 20) if total > 0 else 0
                kwargs['percent'] = min(99, int(step_pct + frame_pct))

            # FPS
            m = fps_re.search(line)
            if m:
                kwargs['fps'] = float(m.group(1))

            # ETA
            m = eta_re.search(line)
            if m:
                kwargs['eta_seconds'] = int(float(m.group(1)) * 60)

            update_task(tid, **kwargs)

        proc.wait()
        _running.pop(tid, None)

        # Find output
        result_file = None
        for pat in ['*_result.mp4', '*.mp4']:
            found = list(Path(RESULT_DIR).glob(f'{tid}{pat}')) + list(work_dir.glob(pat))
            if found:
                result_file = str(found[0])
                break

        # Also check standard path
        std = Path(RESULT_DIR) / f'{tid}_result.mp4'
        if std.exists():
            result_file = str(std)

        if proc.returncode == 0 and result_file:
            size = os.path.getsize(result_file) if result_file else 0
            update_task(tid, status='completed', percent=100, step=5,
                        result_path=result_file, output_size=size,
                        completed_at=int(time.time()), eta_seconds=0)
            log.info(f"[{tid[:8]}] Completed → {result_file} ({size//1024//1024}MB)")
        else:
            update_task(tid, status='error', error=f'Pipeline exit {proc.returncode}')
            log.error(f"[{tid[:8]}] Failed (exit {proc.returncode})")

    except Exception as e:
        _running.pop(tid, None)
        log.exception(f"[{tid[:8]}] Exception: {e}")
        update_task(tid, status='error', error=str(e))

# ── Queue Scheduler ──────────────────────────────────────────
_sched_lock = threading.Lock()
MAX_CONCURRENT = 1  # GPU 同时只跑1个

def _schedule():
    """每5秒检查队列，启动下一个 pending 任务"""
    while True:
        try:
            with _sched_lock:
                tasks = load_tasks()
                running = sum(1 for t in tasks.values() if t.get('status') == 'processing')
                if running < MAX_CONCURRENT:
                    # 找优先级最高的 pending
                    pending = [(tid, t) for tid, t in tasks.items() if t.get('status') == 'pending']
                    if pending:
                        # high priority first, then by created_at
                        pending.sort(key=lambda x: (
                            0 if x[1].get('priority') == 'high' else 1,
                            x[1].get('created_at', 0)
                        ))
                        tid, task = pending[0]
                        cfg = task.get('config_override', {})
                        t = threading.Thread(target=_run_pipeline, args=(tid, cfg), daemon=True)
                        t.start()
        except Exception as e:
            log.error(f"Scheduler error: {e}")
        time.sleep(5)

threading.Thread(target=_schedule, daemon=True).start()

# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('static/index.html')

# Upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(error='No file'), 400
    f = request.files['file']
    if not f.filename:
        return jsonify(error='Empty filename'), 400

    size = request.content_length or 0
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify(error=f'File too large (max {MAX_UPLOAD_MB}MB)'), 413

    tid = uuid.uuid4().hex
    ext = Path(f.filename).suffix.lower() or '.mp4'
    dest = Path(UPLOAD_DIR) / f'{tid}{ext}'
    f.save(str(dest))

    # Parse config override
    cfg = {}
    if 'config' in request.form:
        try: cfg = json.loads(request.form['config'])
        except: pass

    tasks = load_tasks()
    tasks[tid] = {
        'id': tid,
        'filename': f.filename,
        'input_path': str(dest),
        'status': 'pending',
        'step': 0,
        'percent': 0,
        'priority': cfg.get('priority', 'normal'),
        'config_override': cfg,
        'created_at': int(time.time()),
        'recent_logs': [],
        'fps': None,
        'eta_seconds': None,
        'output_size': None,
        'result_path': None,
    }
    save_tasks(tasks)
    log.info(f"[{tid[:8]}] Queued: {f.filename}")
    return jsonify(task_id=tid, status='pending')

# Task List
@app.route('/api/tasks')
def api_tasks():
    return jsonify(load_tasks())

# Task Detail
@app.route('/api/tasks/<tid>')
def api_task(tid):
    t = get_task(tid)
    if not t: return jsonify(error='Not found'), 404
    return jsonify(t)

# Delete Task
@app.route('/api/tasks/<tid>', methods=['DELETE'])
def api_delete_task(tid):
    tasks = load_tasks()
    if tid not in tasks: return jsonify(error='Not found'), 404
    t = tasks[tid]
    # Kill if running
    if tid in _running:
        _running[tid].kill()
        _running.pop(tid, None)
    # Clean files
    try:
        inp = Path(t.get('input_path', ''))
        if inp.exists(): inp.unlink()
        work = Path(RESULT_DIR) / tid
        if work.exists(): shutil.rmtree(work)
        out = t.get('result_path', '')
        if out and Path(out).exists(): Path(out).unlink()
    except Exception as e:
        log.warning(f"Cleanup error: {e}")
    del tasks[tid]
    save_tasks(tasks)
    return jsonify(ok=True)

# Cancel Task
@app.route('/api/tasks/<tid>/cancel', methods=['POST'])
def api_cancel(tid):
    tasks = load_tasks()
    if tid not in tasks: return jsonify(error='Not found'), 404
    if tid in _running:
        _running[tid].kill()
        _running.pop(tid, None)
    tasks[tid]['status'] = 'cancelled'
    tasks[tid]['percent'] = tasks[tid].get('percent', 0)
    save_tasks(tasks)
    return jsonify(ok=True)

# Retry Task
@app.route('/api/tasks/<tid>/retry', methods=['POST'])
def api_retry(tid):
    tasks = load_tasks()
    if tid not in tasks: return jsonify(error='Not found'), 404
    t = tasks[tid]
    if t.get('status') not in ('error', 'cancelled'):
        return jsonify(error='Task not in error/cancelled state'), 400
    tasks[tid].update({
        'status': 'pending',
        'step': 0,
        'percent': 0,
        'recent_logs': [],
        'error': None,
        'fps': None,
        'eta_seconds': None,
    })
    save_tasks(tasks)
    return jsonify(ok=True)

# Download Result
@app.route('/download/<tid>')
def download(tid):
    t = get_task(tid)
    if not t or t.get('status') != 'completed':
        return jsonify(error='Not ready'), 404
    path = t.get('result_path', '')
    if not path or not Path(path).exists():
        return jsonify(error='File not found'), 404
    return send_file(path, as_attachment=True,
                     download_name=f"restored_{Path(t['filename']).stem}.mp4")

# Preview
@app.route('/api/preview/<tid>/<which>')
def preview(tid, which):
    t = get_task(tid)
    if not t: return jsonify(error='Not found'), 404

    if which == 'input':
        inp = t.get('input_path', '')
        frame = Path(RESULT_DIR) / tid / 'frames_denoised' / '00000001.jpg'
        frame_png = Path(RESULT_DIR) / tid / 'frames_denoised' / '00000001.png'
        if frame.exists():
            return send_file(str(frame))
        if frame_png.exists():
            return send_file(str(frame_png))
        # Extract first frame on the fly
        if inp and Path(inp).exists():
            out = Path(RESULT_DIR) / tid / 'preview_input.jpg'
            if not out.exists():
                ensure_dir(out.parent)
                subprocess.run(['ffmpeg', '-y', '-i', inp, '-vframes', '1',
                                '-q:v', '3', str(out)],
                               capture_output=True)
            if out.exists():
                return send_file(str(out))

    elif which == 'output':
        result = t.get('result_path', '')
        out = Path(RESULT_DIR) / tid / 'preview_output.jpg'
        if not out.exists() and result and Path(result).exists():
            ensure_dir(out.parent)
            subprocess.run(['ffmpeg', '-y', '-i', result, '-vframes', '1',
                            '-q:v', '3', str(out)],
                           capture_output=True)
        if out.exists():
            return send_file(str(out))
        # Check upscaled frames
        for d in ['frames_upscaled', 'frames_face_enhanced', 'frames_denoised']:
            frames = sorted((Path(RESULT_DIR) / tid / d).glob('*.jpg'))
            if not frames:
                frames = sorted((Path(RESULT_DIR) / tid / d).glob('*.png'))
            if frames:
                return send_file(str(frames[-1]))

    return jsonify(error='Preview not available'), 404

# GPU Status
@app.route('/api/gpu')
def api_gpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(',')]
            return jsonify(
                available=True,
                memory_used=int(parts[0]) if parts[0].isdigit() else 0,
                memory_total=int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                utilization=int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                temperature=int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
            )
    except Exception as e:
        log.debug(f"GPU query failed: {e}")
    return jsonify(available=False, error='nvidia-smi failed')

# Config
@app.route('/api/config', methods=['GET'])
def api_get_config():
    try:
        cfg = load_config(str(CONFIG_FILE))
        return jsonify(cfg)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/config', methods=['POST'])
def api_set_config():
    try:
        import yaml
        new_cfg = request.json
        if not new_cfg:
            return jsonify(error='No data'), 400
        # Merge with existing
        cfg = load_config(str(CONFIG_FILE))
        def deep_merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
        deep_merge(cfg, new_cfg)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(error=str(e)), 500

# SSE Progress stream
@app.route('/api/tasks/<tid>/stream')
def stream_task(tid):
    def generate():
        last = None
        for _ in range(600):  # max 10min
            t = get_task(tid)
            if not t:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            if t != last:
                yield f"data: {json.dumps(t)}\n\n"
                last = dict(t)
            if t.get('status') in ('completed', 'error', 'cancelled'):
                break
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# Stats
@app.route('/api/tasks/stream/all')
def stream_all_tasks():
    def generate():
        last = None
        for _ in range(1800):
            tasks = load_tasks()
            if tasks != last:
                yield f"data: {json.dumps({'tasks': tasks})}\n\n"
                last = tasks
            time.sleep(1)
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/stats')
def api_stats():
    tasks = load_tasks()
    vals = list(tasks.values())
    return jsonify(
        total=len(vals),
        processing=sum(1 for t in vals if t.get('status') == 'processing'),
        pending=sum(1 for t in vals if t.get('status') == 'pending'),
        completed=sum(1 for t in vals if t.get('status') == 'completed'),
        error=sum(1 for t in vals if t.get('status') == 'error'),
    )

# Health
@app.route('/health')
def health():
    return jsonify(ok=True, time=int(time.time()))

# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    log.info(f"Starting Video Restorer on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
