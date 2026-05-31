#!/usr/bin/env bash
# ============================================================
# 下载预训练模型
# ============================================================
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${MODEL_DIR}/realesrgan"
mkdir -p "${MODEL_DIR}/lama"
mkdir -p "${MODEL_DIR}/propainter"

echo "[INFO] 下载 Real-ESRGAN 模型..."
# Real-ESRGAN
REALESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0"
wget -q -nc "${REALESRGAN_URL}/RealESRGAN_x4plus.pth"       -P "${MODEL_DIR}/realesrgan/" 2>/dev/null && echo "  ✅ RealESRGAN_x4plus.pth" || echo "  ⏭️  RealESRGAN_x4plus.pth (已存在或跳过)"
wget -q -nc "${REALESRGAN_URL}/RealESRGAN_x2plus.pth"       -P "${MODEL_DIR}/realesrgan/" 2>/dev/null && echo "  ✅ RealESRGAN_x2plus.pth" || echo "  ⏭️  RealESRGAN_x2plus.pth"

REALESRGAN_ANIME_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4"
wget -q -nc "${REALESRGAN_ANIME_URL}/realesr-animevideov3.pth"  -P "${MODEL_DIR}/realesrgan/" 2>/dev/null && echo "  ✅ realesr-animevideov3.pth" || echo "  ⏭️  realesr-animevideov3.pth"
wget -q -nc "${REALESRGAN_ANIME_URL}/realesr-general-x4v3.pth"  -P "${MODEL_DIR}/realesrgan/" 2>/dev/null && echo "  ✅ realesr-general-x4v3.pth" || echo "  ⏭️  realesr-general-x4v3.pth"

echo "[INFO] 下载 LaMa 模型..."
wget -q -nc "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" -P "${MODEL_DIR}/lama/" 2>/dev/null && echo "  ✅ big-lama.pt" || echo "  ⏭️  big-lama.pt"

echo "[INFO] 下载 ProPainter 模型..."
PROPAINTER_URL="https://github.com/sczhou/ProPainter/releases/download/v0.1.0"
wget -q -nc "${PROPAINTER_URL}/ProPainter.pth"               -P "${MODEL_DIR}/propainter/" 2>/dev/null && echo "  ✅ ProPainter.pth" || echo "  ⏭️  ProPainter.pth"
wget -q -nc "${PROPAINTER_URL}/flow_completion.pth"          -P "${MODEL_DIR}/propainter/" 2>/dev/null && echo "  ✅ flow_completion.pth" || echo "  ⏭️  flow_completion.pth"
wget -q -nc "${PROPAINTER_URL}/raft.pth"                     -P "${MODEL_DIR}/propainter/" 2>/dev/null && echo "  ✅ raft.pth" || echo "  ⏭️  raft.pth"
wget -q -nc "${PROPAINTER_URL}/recurrent_flow_completion.pth" -P "${MODEL_DIR}/propainter/" 2>/dev/null && echo "  ✅ recurrent_flow_completion.pth" || echo "  ⏭️  recurrent_flow_completion.pth"

echo ""
echo "[INFO] 模型下载完成！"
echo "  模型位置: ${MODEL_DIR}"
ls -lh "${MODEL_DIR}/realesrgan/" "${MODEL_DIR}/lama/" "${MODEL_DIR}/propainter/"
