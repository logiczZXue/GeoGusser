# DK-2500 部署指南（Ubuntu + Python 3.10）

## 1. 联网后先装系统依赖

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git
```

## 2. 创建虚拟环境

```bash
cd /home/你的用户名
python3 -m venv tuxun-env
source tuxun-env/bin/activate
```

## 3. 安装PyTorch + CUDA

```bash
# 先确认DK-2500的GPU驱动和CUDA版本
nvidia-smi

# 根据CUDA版本安装PyTorch（假设CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 4. 安装项目依赖

```bash
pip install transformers accelerate peft pillow tqdm huggingface-hub bitsandbytes qwen-vl-utils
```

## 5. 下载模型（需要联网）

```bash
# 设置HuggingFace镜像（国内网络）
export HF_ENDPOINT=https://hf-mirror.com

# 下载Qwen2-VL-2B（约4GB）
python -c "from transformers import Qwen2VLForConditionalGeneration; Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"
```

## 6. 拷贝项目代码

从Windows笔记本把 D:\Geocomp 整个文件夹拷到DK-2500上（U盘或网络传输）：

```bash
# 在DK-2500上
cp -r /media/usb/Geocomp ~/Geocomp
cd ~/Geocomp
```

## 7. 运行测试

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# 用你自己的街景图测试
python scripts/test_real_image.py --image 你的图片路径.jpg
```

## 8. 验证OpenVINO（如果DK-2500预装了）

```bash
python -c "import openvino; print(openvino.__version__)"
```

---

## 故障排查

### 显存不够（OOM）
- 确保图片不超过630x630像素（脚本会自动缩放）
- 或使用更小的模型：Qwen2-VL-2B

### HuggingFace下载超时
- 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 或使用离线模式：`export HF_HUB_OFFLINE=1`（需先下载好模型）

### bitsandbytes安装失败（Ubuntu）
- 需要：`sudo apt install -y build-essential gcc`
- 然后：`pip install bitsandbytes`
