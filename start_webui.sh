#!/bin/bash

cd "$(dirname "$0")"

# 查找 conda 安装路径（兼容 macOS / Linux / WSL2）
CONDA_BASE=""
for path in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
            "/opt/conda" "/opt/miniconda3" "/opt/anaconda3" \
            "/usr/local/miniconda3" "/usr/local/anaconda3"; do
    if [ -f "$path/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$path"
        break
    fi
done

if [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate cosyvoice
    python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
elif command -v conda &> /dev/null; then
    conda run -n cosyvoice python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
else
    echo "错误: 未找到 conda，请先安装 conda 或手动激活 cosyvoice 环境后运行："
    echo "  python webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B"
    exit 1
fi
