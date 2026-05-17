"""Run GeoCoT inference with few-shot examples enabled.

Usage:
    python scripts/test_with_fewshot.py --image "D:/Desktop/测试图片/xxx.jpg"
    python scripts/test_with_fewshot.py --image-dir "D:/Desktop/测试图片/"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MAX_PIXELS = 500000


def resize_img(img, max_px=MAX_PIXELS):
    w, h = img.size
    if w * h > max_px:
        scale = (max_px / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def load_fewshot_examples():
    """Load few-shot examples from all available sources."""
    examples = []
    sources = [
        PROJECT_ROOT / "src/Geocot/prompts/fewshot_china.json",
        PROJECT_ROOT / "src/Geocot/prompts/few_shot_examples.json",
    ]
    for src in sources:
        if src.exists():
            data = json.loads(src.read_text(encoding="utf-8"))
            for ex in data:
                if ex.get("output"):
                    examples.append(ex["output"])
    return examples


def build_prompt_with_fewshot(stage, prev_outputs=None):
    """Build a stage prompt with few-shot examples prepended."""
    fewshots = load_fewshot_examples()

    # Select relevant few-shots (simple: take up to 3 most recent)
    fewshot_text = ""
    if fewshots:
        # Use the last 3 examples (user's corrections are appended last)
        relevant = fewshots[-3:]
        fewshot_text = "以下是地理定位推理的参考范例:\n\n"
        for i, ex in enumerate(relevant, 1):
            fewshot_text += f"=== 范例 {i} ===\n{ex}\n\n"
        fewshot_text += "---\n现在，请用同样的推理方式分析这张新图片:\n\n"

    prompts = {
        "macro": fewshot_text + (
            "Analyze this street view image for broad geographic context. Consider:\n"
            "- Climate zone (tropical, temperate, arid, polar) based on vegetation, sky, and soil\n"
            "- General topography (flat, mountainous, coastal, inland)\n"
            "- Vegetation type and any distinctive flora\n"
            "- Rock/soil color and type (limestone? granite? sandstone?)\n"
            "- Overall urbanization level (dense city, suburban, rural)\n\n"
            "Provide a [MACRO] analysis identifying the continent and broad region."
        ),
        "regional": (
            "Based on the macro analysis above, examine country-specific indicators:\n"
            "- Language on signs, buildings, or vehicles\n"
            "- Architectural style (roof shapes, materials, building age)\n"
            "- Traffic direction and road infrastructure\n"
            "- License plate format and color\n"
            "- Utility pole design\n"
            "- Cultural markers (religious buildings, flags, symbols)\n\n"
            "Provide a [REGIONAL] analysis identifying the specific country or region.\n"
            "IMPORTANT: distinguish from similar-looking regions."
        ),
        "local": (
            "Based on macro and regional analyses, identify city-level details:\n"
            "- Sidewalk patterns and street furniture\n"
            "- Specific landmarks or recognizable buildings\n"
            "- Vehicle makes/models common to the region\n"
            "- Local business names or advertisements\n"
            "- Any visible coordinates or address information\n\n"
            "Provide a [LOCAL] synthesis with final conclusion in format:\n"
            "COORDINATES: lat, lng\n"
            "LOCATION: city, country, continent"
        ),
    }
    return prompts.get(stage, "")


def run_inference(image_path, model, processor):
    """Run 3-stage GeoCoT inference with few-shot examples."""
    raw = Image.open(image_path).convert("RGB")
    orig_size = raw.size
    image = resize_img(raw)

    print(f"图片: {os.path.basename(image_path)} ({orig_size[0]}x{orig_size[1]} -> {image.size[0]}x{image.size[1]})")
    print("=" * 70)

    outputs = {}

    for stage in ["macro", "regional", "local"]:
        prompt = build_prompt_with_fewshot(stage, outputs)
        if stage == "regional":
            prompt = f'Based on the macro analysis:\n"{outputs.get("macro", "")}"\n\n{prompt}'
        elif stage == "local":
            prompt = f'Macro analysis: {outputs.get("macro", "")}\nRegional analysis: {outputs.get("regional", "")}\n\n{prompt}'

        t0 = time.time()
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[txt], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=300 if stage == "local" else 200)
        res = processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        torch.cuda.empty_cache()

        outputs[stage] = res.strip()
        elapsed = time.time() - t0
        print(f"\n--- [{stage.upper()}] ({elapsed:.1f}s) ---")
        print(res.strip())

    print("\n" + "=" * 70)
    print("最终推理结果:")
    print(outputs["local"])

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Test GeoCoT with few-shot examples")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("需要 --image 或 --image-dir")

    # Load model
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"加载模型: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Show how many few-shot examples are loaded
    fewshots = load_fewshot_examples()
    print(f"已加载 {len(fewshots)} 个 few-shot 范例")

    # Collect images
    images = []
    if args.image:
        images.append(args.image)
    else:
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(args.image_dir, f))

    for img_path in images:
        run_inference(img_path, model, processor)
        print()


if __name__ == "__main__":
    main()
