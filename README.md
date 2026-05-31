# 🎬 Video-Restorer

> **老片翻新 · 低分辨率提升 · 模糊变高清 · 去水印**
>
> 专为 **NVIDIA RTX 3090 24GB** GPU 服务器设计的视频修复流水线

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 快速上手

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo> video-restorer
cd video-restorer

# 一键安装 (Ubuntu 22.04 + CUDA)
# 会自动安装: 系统依赖 → Miniconda → PyTorch → Real-ESRGAN → VSR → ProPainter
bash setup.sh

# 激活环境
conda activate video-restorer
```

### 2. 一键运行

```bash
# 完整流水线: 格式统一 → 去水印 → 画质增强 → 2x超分 → 合成视频
python scripts/pipeline.py input/old_video.mp4 -o output/restored.mp4

# 仅超分 (跳过水印和增强)
python scripts/pipeline.py input/video.mp4 -o output/hd.mp4 --no-watermark --no-enhance

# 4x 超分 (适合低分辨率视频)
python scripts/pipeline.py input/360p_video.mp4 -o output/1440p.mp4 --scale 4

# 快速模式 (显存不足时使用)
python scripts/pipeline.py input/video.mp4 -o output/result.mp4 --fast

# 动画/老片专用 (使用 anime 模型)
python scripts/pipeline.py input/cartoon.mp4 -o output/hd_cartoon.mp4 --fast
```

### 3. 分步执行 (调试/定制)

```bash
# Step 1: 格式统一 + 拆帧
python scripts/preprocess.py input/video.mp4 --frames-dir frames

# Step 2: 去水印 (指定水印区域)
python scripts/remove_watermark.py --frames-dir frames --bbox 100 50 200 80

# Step 3: 画质增强
python scripts/enhance.py --frames-dir frames_denoised

# Step 4: 超分辨率
python scripts/upscale.py --frames-dir frames_enhanced --scale 2

# Step 5: 合成视频
python scripts/postprocess.py --frames-dir frames_upscaled --output final.mp4

# 质量评估
python scripts/evaluate.py frames_original frames_upscaled --lpips
```

---

## 🧠 技术架构

```
输入视频 (低分辨率/模糊/带水印)
    │
    ▼
┌─────────────────────┐
│ Step 1: 预处理       │  FFmpeg 格式统一 + 逐帧拆解
│  统一格式 + 拆帧      │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Step 2: 去水印       │  LaMa / ProPainter 修复
│  自动检测 + AI修复    │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Step 3: 画质增强     │  Real-ESRGAN (animevideov3)
│  去噪 + 去模糊       │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Step 4: 超分辨率     │  Real-ESRGAN (x2/x4)
│  2x / 4x 放大       │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Step 5: 后处理       │  FFmpeg 合成 + 色彩优化
│  合成视频 + 色调优化  │
└─────────┬───────────┘
          ▼
输出视频 (高清/无水印)
```

### 核心模型

| 模块 | 模型 | 适用场景 | 显存需求 |
|------|------|---------|---------|
| 去水印 | **LaMa** (big-lama) | 大面积静态水印/字幕 | ~2GB |
| 去水印 | **ProPainter** | 动态水印/复杂场景 | ~6GB |
| 画质增强 | **real-esr-animevideov3** | 动漫/老电影去噪去模糊 | ~3GB |
| 画质增强 | **real-esr-general-x4v3** | 通用视频增强 | ~3GB |
| 超分辨率 | **RealESRGAN_x4plus** | 2x/4x 通用超分 | ~4GB (tile=256) |
| 超分辨率 | **RealESRGAN_x2plus** | 2x 高质量超分 | ~4GB |
| 质量评估 | **LPIPS (AlexNet)** | 感知质量评价 | ~1GB |

> **3090 24GB 足够同时跑多个模型，视频处理时建议 `tile_size=256` 避免 OOM**

---

## ⚙️ 配置说明

编辑 `config/settings.yaml`：

```yaml
# 超分辨率
super_resolution:
  enabled: true
  scale: 2                    # 2 或 4 倍
  model: "RealESRGAN_x4plus"
  tile_size: 256              # 分块大小 (OOM时减小)
  tile_pad: 10

# 去水印
watermark_removal:
  enabled: true
  method: "lama"              # lama 或 propainter
  watermark_bbox: [100, 50, 200, 80]  # [x, y, w, h]
```

完整配置参考 `config/settings.yaml`。

---

## 📊 质量评估

```bash
# 对比原始帧和处理后帧
python scripts/evaluate.py frames_original frames_upscaled --lpips
```

输出示例:
```
=======================================================
  质量评估报告 (300 帧)
=======================================================
  指标         平均值     最小值     最大值
  ------------------------------------------
  PSNR (dB)    32.15     28.44     35.67
  SSIM         0.9641    0.9210    0.9872
  LPIPS (↓)    0.0872    0.0231    0.1543
=======================================================
```

---

## 🔧 服务器部署

```bash
# 1. 上传到服务器
scp -r video-restorer user@10.190.0.222:~/video-restorer

# 2. SSH 登录服务器
ssh user@10.190.0.222

# 3. 一键安装
cd ~/video-restorer
bash setup.sh

# 4. 开始处理
conda activate video-restorer
python scripts/pipeline.py input/old_movie.mp4 -o output/restored.mp4
```

### 手动 CUDA 检查

```bash
nvidia-smi                 # 检查驱动 + GPU 状态
nvcc --version             # 检查 CUDA 版本
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 🗂️ 项目结构

```
video-restorer/
├── config/
│   └── settings.yaml       # 配置文件
├── models/
│   ├── download_models.sh  # 模型下载脚本
│   ├── realesrgan/         # Real-ESRGAN 模型
│   ├── lama/               # LaMa 修复模型
│   └── propainter/         # ProPainter 模型
├── scripts/
│   ├── pipeline.py         # 🎯 主流水线入口
│   ├── preprocess.py       # Step 1: 预处理
│   ├── remove_watermark.py # Step 2: 去水印
│   ├── enhance.py          # Step 3: 画质增强
│   ├── upscale.py          # Step 4: 超分辨率
│   ├── postprocess.py      # Step 5: 后处理
│   ├── evaluate.py         # 质量评估
│   └── utils.py            # 工具函数
├── tools/                  # 第三方工具 (VSR/ProPainter)
├── setup.sh                # 一键安装脚本
├── requirements.txt        # Python 依赖
└── README.md
```

---

## 📝 常见问题

<details>
<summary><b>CUDA out of memory</b></summary>

```bash
# 减小 tile_size
python scripts/pipeline.py input.mp4 -o output.mp4 --fast
# 或编辑 config: tile_size=128, scale=2
```
</details>

<details>
<summary><b>如何提高处理速度？</b></summary>

- 开启 `--fast` 模式减小分块
- 只做 2x 超分而非 4x
- 跳过去水印或增强步骤
- 使用 `half=True` (FP16 推理，显存减半)
</details>

<details>
<summary><b>如何选择超分模型？</b></summary>

- 动漫/老电影 → `realesr-animevideov3`
- 真人视频 → `RealESRGAN_x4plus`
- 2x 超分 → `RealESRGAN_x2plus` (质量更好)
</details>

---

## 📚 参考资源

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - 超分辨率 & 画质增强
- [BasicSR](https://github.com/xinntao/BasicSR) - 视频修复框架
- [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) - 视频去字幕/水印
- [ProPainter](https://github.com/sczhou/ProPainter) - 视频修复 (去水印/划痕)
- [LaMa](https://github.com/saic-mdal/lama) - 图像修复 (去水印)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - 感知质量评估

---

## 📜 License

MIT
