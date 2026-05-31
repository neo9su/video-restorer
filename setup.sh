#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    log_info "检查系统环境..."
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    python3 --version
    ffmpeg -version 2>&1 | head -1
}

install_system_deps() {
    log_info "安装系统依赖..."
    sudo apt-get update -y
    sudo apt-get install -y --no-install-recommends         libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1         wget git curl unzip
}

install_conda() {
    if command -v conda &>/dev/null; then
        log_info "Conda 已安装: $(conda --version)"
        return
    fi
    log_info "安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    log_info "Miniconda 安装完成"
}

create_env() {
    local env_name="${1:-video-restorer}"
    log_info "创建 Conda 环境: ${env_name}"
    conda create -y -n "${env_name}" python=3.10
    source ~/.bashrc
    conda activate "${env_name}"

    # 驱动 535 支持 CUDA 12.x，安装 PyTorch CUDA 12.1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
}

install_python_deps() {
    log_info "安装 Python 依赖..."
    pip install -r ~/video-restorer/requirements.txt
}

install_realesrgan() {
    log_info "安装 Real-ESRGAN..."
    pip install realesrgan
    python3 -c "
from realesrgan import RealESRGANer
print('Real-ESRGAN 安装成功')
"
}

install_vsr() {
    log_info "安装 video-subtitle-remover (VSR)..."
    if [ -d "$HOME/video-restorer/tools/video-subtitle-remover" ]; then
        log_info "VSR 已存在，跳过"
        return
    fi
    mkdir -p $HOME/video-restorer/tools
    cd $HOME/video-restorer/tools
    git clone --depth=1 https://github.com/YaoFANGUK/video-subtitle-remover.git
    cd video-subtitle-remover
    pip install -r requirements.txt 2>/dev/null || true
    cd $HOME
    log_info "VSR 安装完成"
}

install_propainter() {
    log_info "安装 ProPainter..."
    if [ -d "$HOME/video-restorer/tools/ProPainter" ]; then
        log_info "ProPainter 已存在，跳过"
        return
    fi
    mkdir -p $HOME/video-restorer/tools
    cd $HOME/video-restorer/tools
    git clone --depth=1 https://github.com/sczhou/ProPainter.git
    cd ProPainter
    pip install -r requirements.txt 2>/dev/null || true
    cd $HOME
    log_info "ProPainter 安装完成"
}

download_models() {
    log_info "下载预训练模型..."
    cd $HOME/video-restorer
    bash models/download_models.sh || true
}

main() {
    echo "============================================"
    echo "  Video-Restorer 服务器安装脚本"
    echo "============================================"
    echo ""

    check_env
    install_system_deps
    install_conda
    create_env "video-restorer"
    install_python_deps
    install_realesrgan
    install_vsr
    install_propainter
    download_models

    echo ""
    log_info "✅ 安装完成！"
    echo ""
    echo "  使用:"
    echo "    conda activate video-restorer"
    echo "    cd ~/video-restorer"
    echo "    python scripts/pipeline.py --help"
    echo ""
}

main "$@"
