#!/bin/bash
# Cloud Studio 环境配置脚本 — GeoVLM Teacher Labeling
# 用法: bash cloud_setup.sh

set -e

echo "=== 安装 Python 依赖 ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes modelscope Pillow peft

echo ""
echo "=== 下载 Teacher 模型 (Qwen3-VL-8B-Instruct) ==="
python3 -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='./models')
print('Model downloaded OK')
"

echo ""
echo "=== 目录结构 ==="
mkdir -p data/mapillary output

echo ""
echo "=== 配置完成 ==="
echo "下一步："
echo "1. 上传图片到 data/mapillary/"
echo "2. 确认 output/test_50_selected.json 中路径正确"
echo "3. 运行测试: python3 scripts/generate_teacher_labels.py --max-samples 5"
echo "4. 批量运行: python3 scripts/generate_teacher_labels.py --max-samples 500"
