"""Human-in-the-loop dataset builder for GeoCoT.

Workflow:
  1. Scan images → run GeoCoT inference → show model prediction
  2. Ask: "Do you know the correct location of this image?"
  3. If yes → collect correct answer + reasoning → save to dataset
  4. Repeat until you have enough data, then train LoRA

Usage:
    # Add images to dataset (interactive)
    python scripts/build_dataset.py --image-dir "D:/Desktop/测试图片/"

    # Check dataset status
    python scripts/build_dataset.py --status

    # Train LoRA when dataset is ready (e.g. 100+ examples)
    python scripts/build_dataset.py --train
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "data/vlm_finetune"
DATASET_FILE = DATASET_DIR / "train.jsonl"
CORRECTIONS_DIR = PROJECT_ROOT / "data/corrections"
STATUS_FILE = DATASET_DIR / "status.json"


# ── Status management ──────────────────────────────────────────────────

def load_status():
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    return {"total": 0, "last_updated": None, "images": []}


def save_status(status):
    status["last_updated"] = datetime.now().isoformat()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def count_dataset():
    if not DATASET_FILE.exists():
        return 0
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ── Dataset operations ─────────────────────────────────────────────────

def add_to_dataset(image_path: str, city: str, country: str, continent: str,
                   coordinates: str, reasoning: str, notes: str = ""):
    """Add one correction to the training dataset."""
    entry = {
        "image": os.path.abspath(image_path),
        "city": city,
        "country": country,
        "continent": continent,
        "coordinates": coordinates,
        "reasoning": reasoning,
        "notes": notes,
        "added_at": datetime.now().isoformat(),
    }

    # Save individual correction
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    corr_id = f"{city}_{country}".replace(" ", "_").lower()
    corr_file = CORRECTIONS_DIR / f"{corr_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    corr_file.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append to training JSONL (format for LoRA training)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    instruction = (
        "Analyze this street view image using the GeoCoT 3-stage geolocation framework:\n"
        "Stage 1 [MACRO]: Identify continent/region from climate, topography, vegetation, architecture.\n"
        "Stage 2 [REGIONAL]: Identify country from language, signage, traffic, cultural markers.\n"
        "Stage 3 [LOCAL]: Identify city from landmarks, street details, local features.\n"
        "Provide coordinates and final location."
    )
    training_record = {
        "image": os.path.abspath(image_path),
        "instruction": instruction,
        "response": reasoning,
    }
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(training_record, ensure_ascii=False) + "\n")

    # Update status
    status = load_status()
    status["total"] = count_dataset()
    status["images"].append({
        "file": os.path.basename(image_path),
        "city": city,
        "country": country,
        "added_at": datetime.now().isoformat(),
    })
    save_status(status)


# ── Inference ──────────────────────────────────────────────────────────

MAX_PIXELS = 500000

def resize_img(img, max_px=MAX_PIXELS):
    w, h = img.size
    if w * h > max_px:
        scale = (max_px / (w * h)) ** 0.5
        return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img


def run_geocot_inference(image_path, model, processor):
    """Run GeoCoT 3-stage inference and return outputs."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    import torch

    raw = Image.open(image_path).convert("RGB")
    image = resize_img(raw)

    macro_prompt = (
        "Analyze this street view image for broad geographic context. Consider:\n"
        "- Climate zone (tropical, temperate, arid, polar)\n"
        "- Topography (flat, mountainous, coastal, inland)\n"
        "- Vegetation type and distinctive flora\n"
        "- Architecture style and urbanization level\n"
        "- Rock/soil color if visible\n\n"
        "Which continent and broad region is this image most likely from?"
    )

    regional_prompt = (
        "Based on the macro analysis:\n\"{macro}\"\n\n"
        "Now examine country-specific indicators:\n"
        "- Language on signs, buildings, or advertisements\n"
        "- Architectural style (roof shapes, materials, building era)\n"
        "- Traffic direction, road markings, license plates\n"
        "- Utility poles, fire hydrants, street furniture\n"
        "- Cultural markers (flags, religious symbols, clothing)\n\n"
        "Which specific country does this image come from? Be precise."
    )

    local_prompt = (
        "Based on:\n- Macro: {macro}\n- Regional: {regional}\n\n"
        "Now identify city-level details:\n"
        "- Sidewalk patterns, street furniture\n"
        "- Specific landmarks or recognizable buildings\n"
        "- Vehicle makes/models\n"
        "- Local business names or advertisements\n\n"
        "Provide final geolocation. Format:\n"
        "COORDINATES: lat, lng\n"
        "LOCATION: city, country, continent"
    )

    def _generate(prompt, max_tokens=250):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[txt], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens)
        result = processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        torch.cuda.empty_cache()
        return result.strip()

    macro = _generate(macro_prompt)
    regional = _generate(regional_prompt.replace("{macro}", macro))
    local = _generate(local_prompt.replace("{macro}", macro).replace("{regional}", regional), max_tokens=350)

    return {"macro": macro, "regional": regional, "local": local}


# ── Interactive mode ───────────────────────────────────────────────────

def interactive_session(image_dir: str):
    """Process images one by one, ask user for corrections."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Collect images
    images = []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(image_dir, f))

    if not images:
        print("未找到图片文件。")
        return

    # Load model
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"加载模型: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"共 {len(images)} 张图片待处理\n")

    count = count_dataset()
    added = 0

    for i, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        print(f"[{i}/{len(images)}] {fname}")
        print("-" * 60)

        # Run inference
        t0 = time.time()
        try:
            results = run_geocot_inference(img_path, model, processor)
        except Exception as e:
            print(f"  推理失败: {e}")
            continue
        elapsed = time.time() - t0

        print(f"  模型预测 ({elapsed:.0f}s):")
        # Extract location from local stage
        local_text = results["local"]
        for line in local_text.split("\n"):
            if "LOCATION:" in line or "ocation:" in line or "city" in line.lower():
                print(f"    {line.strip()}")

        print(f"\n  完整推理 (local stage):")
        print(f"    {local_text[:300]}...")
        print()

        # Ask user
        print("  >>> 你知道这张图的真实位置吗?")
        know = input("  [y=知道, 录入正确答案 / n=不知道, 跳过 / q=退出]: ").strip().lower()

        if know == "q":
            print("退出。")
            break
        elif know == "y":
            print("\n  --- 录入正确信息 ---")
            city = input("  城市: ").strip()
            country = input("  国家: ").strip()
            continent = input("  大洲: ").strip()
            coords = input("  经纬度 (lat, lng): ").strip()

            print("\n  --- 推理链 (自由文本, 可参考 Stage1→2→3 格式) ---")
            print("  输入你的完整推理过程。输入空行结束:")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            reasoning = "\n".join(lines)

            if not reasoning:
                # Use a simple template if user doesn't provide reasoning
                reasoning = (
                    f"[MACRO]\n该图片位于{continent}，{country}地区。\n\n"
                    f"[REGIONAL]\n建筑风格和道路特征指向{country}。\n\n"
                    f"[LOCAL]\n综合所有线索，定位在{city}, {country}, {continent}。\n\n"
                    f"COORDINATES: {coords}\n"
                    f"LOCATION: {city}, {country}, {continent}"
                )

            notes = input("  备注 (可选): ").strip()

            add_to_dataset(img_path, city, country, continent, coords, reasoning, notes)
            count += 1
            added += 1
            print(f"  [OK] 已保存! 当前数据集: {count} 条\n")
        else:
            print("  已跳过。\n")

    # Summary
    print("=" * 60)
    print(f"本次新增: {added} 条")
    print(f"数据集总量: {count} 条")
    if count >= 100:
        print(f"\n  >>> 已达到 {count} 条，可以开始训练了!")
        print(f"  运行: python scripts/build_dataset.py --train")
    elif count > 0:
        print(f"  还需 {100 - count} 条到 100。")


# ── Training ───────────────────────────────────────────────────────────

def train_lora():
    """Train LoRA adapter on the accumulated dataset."""
    count = count_dataset()
    if count == 0:
        print("数据集为空，请先添加数据。")
        return

    print(f"数据集: {count} 条")
    print(f"开始 LoRA 训练...")

    import subprocess
    cmd = [
        sys.executable, "-m", "src.Geocot.train_vlm_lora",
        "--data", str(DATASET_FILE),
        "--output", str(PROJECT_ROOT / "output/vlm_lora"),
        "--epochs", "3",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--rank", "8",
        "--alpha", "16",
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


# ── Status ─────────────────────────────────────────────────────────────

def show_status():
    count = count_dataset()
    status = load_status()

    print(f"数据集状态")
    print(f"{'='*50}")
    print(f"  总样本数: {count}")
    print(f"  上次更新: {status.get('last_updated', 'N/A')}")
    print(f"  存储位置: {DATASET_FILE}")
    print(f"  单条存档: {CORRECTIONS_DIR}")

    if status.get("images"):
        print(f"\n  已录入图片:")
        for img in status["images"]:
            print(f"    - {img['file']} → {img['city']}, {img['country']}")

    print(f"\n  距离 100 条目标: {'已达到!' if count >= 100 else f'还差 {100 - count} 条'}")

    if count >= 100:
        print(f"  >>> 可以训练了: python scripts/build_dataset.py --train")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GeoCoT dataset builder with human feedback")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    parser.add_argument("--train", action="store_true", help="Train LoRA on accumulated dataset")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.train:
        train_lora()
    elif args.image_dir:
        interactive_session(args.image_dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python scripts/build_dataset.py --image-dir D:/Desktop/测试图片/")
        print("  python scripts/build_dataset.py --status")
        print("  python scripts/build_dataset.py --train")


if __name__ == "__main__":
    main()
