# Video Restorer

视频修复流水线 — 去水印、降噪、面部增强、超分辨率，支持 GPU 加速。

**GPU**: RTX 3090 (24GB) | **驱动**: NVIDIA 535.309.01 + `NVreg_EnableGpuFirmware=0`  
**Server**: Inspur NF5568M4 (Ubuntu 22.04) | **URL**: `http://10.190.0.222:8000`

---

## 功能特性

### 核心处理管线 (Pipeline)

| Step | 功能 | 工具 | 说明 |
|------|------|------|------|
| 1 | 视频抽帧 | ffmpeg | 从视频提取所有帧 |
| 2 | 视频降噪 | ProPainter | 空域+时域联合去噪 |
| 3 | 面部增强 | CodeFormer | 面部细节修复 (可开关) |
| 4 | 超分辨率 | RealESRGAN / AnimeVideo v3 | 支持 2x/4x，FP16 加速 |
| 5 | 视频合成 | ffmpeg | 帧回视频 + 音轨复用 |

### 超分模型对比

| 模型 | 类型 | 适合场景 | 质量 | 速度 |
|------|------|---------|------|------|
| RealESRGAN_x4plus | RRDB-23层 | **真人视频** | ⭐⭐⭐⭐⭐ | 快 |
| realesr-animevideov3 | SRVGGNet-16层 | 动画/卡通 | ⭐⭐⭐ | 最快 |
| 4x-UltraSharp | 旧格式 ESRGAN | 锐度需求高 | ⭐⭐⭐⭐⭐ | 快 |
| BSRGAN | 旧格式 ESRGAN | 通用 | ⭐⭐⭐⭐ | 快 |

### 新增功能 (v2.0)

- ✅ **批量上传** — 拖放或多文件一次上传
- ✅ **任务队列** — 优先级调度 (普通/高优)
- ✅ **断点续传** — Pipeline 内置 checkpoint，中断后可恢复
- ✅ **实时进度** — SSE 流式推送处理进度和日志
- ✅ **前后对比** — 滑动对比滑块 (Before/After)
- ✅ **质量评估** — PSNR/SSIM 自动计算，生成对比图
- ✅ **预处理** — 黑边裁剪、音频提取、场景切分
- ✅ **移动端适配** — 响应式布局，手机可用
- ✅ **GPU 状态监控** — 显存使用/温度实时显示
- ✅ **一键部署** — systemd 服务 + Docker 支持

---

## 项目结构

```
video-restorer/
├── webapp/
│   ├── server.py              # Flask 后端 (v2: 完整 API + 调度器 + SSE)
│   ├── static/
│   │   └── index.html         # 全新 Web UI
│   ├── uploads/               # 上传目录
│   ├── results/               # 输出目录
│   └── tasks.json             # 任务状态
├── scripts/
│   ├── pipeline.py            # 主流水线 (v2: checkpoint 断点续传)
│   ├── enhance.py             # 降噪 (ProPainter)
│   ├── upscale.py             # 超分辨率 (FP16 + async I/O)
│   ├── face_enhance.py        # CodeFormer 面部增强
│   ├── _codeformer_worker.py  # CF 推理 worker (OOM 降级)
│   ├── postprocess.py         # 视频合成 + 音轨
│   ├── utils.py               # 工具函数
│   ├── preprocess.py          # 预处理 (裁剪/切分/音频) [NEW]
│   └── quality.py             # 质量评估 PSNR/SSIM [NEW]
├── config/
│   └── settings.yaml          # 全局配置
├── models/
│   ├── realesrgan/            # 超分模型权重
│   │   ├── RealESRGAN_x4plus.pth
│   │   ├── realesr-animevideov3.pth
│   │   ├── 4x-UltraSharp.pth
│   │   └── BSRGAN.pth
│   ├── prosaic/               # 降噪模型
│   └── face/                  # 面部增强模型
├── Dockerfile                 # Docker 部署 [NEW]
├── .dockerignore              # Docker 忽略 [NEW]
└── api_docs.md                # API 文档 [NEW]
```

---

## 部署

### 快速启动

```bash
# 一键启动所有服务
bash /mnt/disk3/start-all-services.sh start

# 状态检查 + GPU 信息
bash /mnt/disk3/start-all-services.sh status
```

### systemd 服务

```bash
sudo cp video-restorer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable video-restorer
sudo systemctl start video-restorer
```

### Docker

```bash
docker build -t video-restorer .
docker run -d --gpus all -p 8000:8000 -v /data/results:/app/results video-restorer
```

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | 上传视频 (multipart/form-data) |
| `GET` | `/api/tasks` | 获取所有任务 |
| `GET` | `/api/tasks/<id>` | 获取单个任务 |
| `DELETE` | `/api/tasks/<id>` | 删除任务 |
| `POST` | `/api/tasks/<id>/cancel` | 取消任务 |
| `POST` | `/api/tasks/<id>/retry` | 重试失败任务 |
| `GET` | `/download/<id>` | 下载结果 |
| `GET` | `/api/preview/<id>/<which>` | 预览帧 (input/output) |
| `GET` | `/api/tasks/<id>/stream` | SSE 进度流 |
| `GET` | `/api/gpu` | GPU 状态 |
| `GET`/`POST` | `/api/config` | 配置读取/更新 |
| `GET` | `/api/stats` | 任务统计 |
| `GET` | `/health` | 健康检查 |

---

## GPU 环境配置

### 驱动修复历程

| 问题 | 原因 | 修复 |
|------|------|------|
| `RmInitAdapter 0x24:0x65:1447` | GSP 固件不兼容 | `NVreg_EnableGpuFirmware=0` + 移除 GSP bin |
| `RmInitAdapter 0x31:0x40:2640` | BAR 空间不足 | `nvreg_EnableResizableBar=1` |
| nouveau 抢占 GPU | 开源驱动冲突 | `blacklist nouveau` + `options nouveau modeset=0` |
| PCIe 速度降级 | BIOS Above 4G Decoding 未开启 | 服务器 BIOS → Enabled |

### 关键配置

```bash
# /etc/modprobe.d/nvidia-gsp.conf
options nvidia NVreg_EnableGpuFirmware=0
options nvidia NVreg_EnableResizableBar=1

# /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0

# GRUB 内核参数
GRUB_CMDLINE_LINUX_DEFAULT="... pci=realloc pci=assign-busses nvidia.NVreg_EnableGpuFirmware=0 nvidia-drm.modeset=1"
```

### 性能优化

- **FP16 推理** — RTX 3090 8.6 架构，FP16 比 FP32 快 5x
- **Full-frame** — tile=0 避免分块开销 (tile128 需 48 次 kernel launch)
- **Async I/O** — PNG→JPEG 多线程读写，速度 0.82fps → 2.74fps
- **JPEG 输出** — 超分后输出 JPEG 替代 PNG，IO 快 3x

---

## 技术栈

| 层 | 技术 |
|----|------|
| **UI** | 纯 HTML/CSS/JS，无框架 |
| **后端** | Python 3.11 + Flask |
| **降噪** | ProPainter (基于 Stable Video Diffusion) |
| **面部** | CodeFormer (basicsr + facexlib) |
| **超分** | RealESRGAN / realesrgan-python |
| **合成** | ffmpeg (libx264 + AAC) |
| **质量** | numpy (PSNR/SSIM), PIL (对比图) |

---

## 性能数据

| 操作 | 配置 | 速度 | 备注 |
|------|------|------|------|
| 超分 | RealESRGAN_x4plus 2x FP16 | ~3 fps | full-frame |
| 超分 | RealESRGAN_x4plus 2x FP32 | ~0.6 fps | tile=128 |
| 超分 | realesr-animevideov3 2x FP16 | ~3 fps | 动画视频更快 |
| 面部增强 | CodeFormer fidelity=0.7 | ~50fps | 子进程隔离 |
| 降噪 | ProPainter level=1 | ~0.05 fps | 最慢步骤 |
| **完整管线** | 960×720 → 1920×1440 | 5782MB | 41978帧 |

---

## 已知问题

- **GPU 初始化** — `nvrm` 模块需 `NVreg_EnableGpuFirmware=0`，GSP 固件需移除
- **RealESRGANer 旧格式** — 4x-UltraSharp / BSRGAN 不支持 `params` key，需用直接加载
- **CodeFormer 子进程** — basicsr 版本冲突，用独立 conda env + subprocess 隔离
- **GPU 显存** — 24GB 限制，同时运行 ComfyUI + ProPainter 会 OOM

---

## 未来计划

- [ ] SeedVR2 节点 (ComfyUI) — 更精细的全图超分
- [ ] Webhook 回调 — 任务完成通知
- [ ] 多 GPU 队列 — 支持多卡并行
- [ ] 移动端 App — Flutter 封装
- [ ] 在线预览 — 处理中实时查看进度
