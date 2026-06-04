#!/bin/bash
# Cloud Studio 环境配置脚本 — GeoVLM Teacher Labeling
# 用法: bash scripts/cloud_setup.sh

set -e

echo "=== 安装 Python 依赖 ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes modelscope Pillow peft qwen-vl-utils

echo ""
echo "=== 下载 Teacher 模型 (Qwen3-VL-8B-Instruct) ==="
mkdir -p models
python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='./models')
print('Model downloaded OK')
"

echo ""
echo "=== 解压图片数据 ==="
if [ -f data/mapillary_500.tar.gz ]; then
    tar -xzf data/mapillary_500.tar.gz -C data/
    echo "Images extracted: $(ls data/mapillary_500/*.jpg | wc -l)"
else
    echo "WARNING: data/mapillary_500.tar.gz not found. Upload it first!"
fi

echo ""
echo "=== 目录结构 ==="
mkdir -p output

echo ""
echo "=== 配置完成 ==="
echo "下一步："
echo "  # 先跑5张测试"
echo "  python3 scripts/generate_teacher_labels.py --max-samples 5"
echo "  # 确认无误后跑500张"
echo "  python3 scripts/generate_teacher_labels.py"
