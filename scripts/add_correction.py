"""Add correction examples to few-shot prompts and training dataset.

Usage:
    # Interactive mode — walk through adding a correction
    python scripts/add_correction.py --image "D:/Desktop/测试图片/xxx.jpg" --interactive

    # Batch mode — load corrections from a JSON file
    python scripts/add_correction.py --batch data/corrections/my_corrections.json

    # Generate training JSONL from existing few-shot examples
    python scripts/add_correction.py --generate-training

Each correction contains:
    - image: path to the street view image
    - correct_location: city, country, continent
    - reasoning: full 3-stage GeoCoT reasoning chain
    - scene_type: (optional) urban/rural/mountain/coastal/etc.
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEWSHOT_CHINA = PROJECT_ROOT / "src/Geocot/prompts/fewshot_china.json"
TRAINING_JSONL = PROJECT_ROOT / "data/vlm_finetune/train.jsonl"

STAGE_PROMPTS = {
    "macro": (
        "[MACRO]\n"
        "气候带: {climate}\n"
        "地形: {topography}\n"
        "植被: {vegetation}\n"
        "土壤: {soil}\n"
        "城市化: {urbanization}\n"
        "判断: {macro_conclusion}\n"
    ),
    "regional": (
        "[REGIONAL]\n"
        "语言/文字: {language}\n"
        "建筑风格: {architecture}\n"
        "植被细节: {vegetation_detail}\n"
        "道路/交通: {transport}\n"
        "文化元素: {culture}\n"
        "区分: {differentiation}\n"
        "判断: {regional_conclusion}\n"
    ),
    "local": (
        "[LOCAL]\n"
        "{synthesis}\n\n"
        "COORDINATES: {coordinates}\n"
        "LOCATION: {location}"
    ),
}


def interactive_add():
    """Walk user through adding a correction."""
    image_path = input("图片路径: ").strip()
    if not os.path.exists(image_path):
        print(f"[ERROR] 文件不存在: {image_path}")
        return

    print("\n--- 正确的地理位置 ---")
    city = input("城市: ").strip()
    country = input("国家: ").strip()
    continent = input("大洲 (Asia/Europe/North America/etc.): ").strip()
    coordinates = input("经纬度 (lat, lng): ").strip()

    print("\n--- Stage 1: 宏观分析 (大陆/气候/地形) ---")
    climate = input("气候带: ").strip()
    topography = input("地形: ").strip()
    vegetation = input("植被: ").strip()
    soil = input("土壤/岩石: ").strip()
    urbanization = input("城市化程度: ").strip()
    macro_conclusion = input("宏观结论: ").strip()

    print("\n--- Stage 2: 区域分析 (国家/地区) ---")
    language = input("语言/文字: ").strip()
    architecture = input("建筑风格: ").strip()
    vegetation_detail = input("植被细节(区别于其他地区的特征): ").strip()
    transport = input("道路/交通特征: ").strip()
    culture = input("文化元素: ").strip()
    differentiation = input("如何区分于相似地区: ").strip()
    regional_conclusion = input("区域结论: ").strip()

    print("\n--- Stage 3: 本地定位 (城市/地标) ---")
    synthesis = input("综合推理(自由文本): ").strip()

    # Build full reasoning chain
    macro = STAGE_PROMPTS["macro"].format(
        climate=climate, topography=topography, vegetation=vegetation,
        soil=soil, urbanization=urbanization, macro_conclusion=macro_conclusion,
    )
    regional = STAGE_PROMPTS["regional"].format(
        language=language, architecture=architecture,
        vegetation_detail=vegetation_detail, transport=transport,
        culture=culture, differentiation=differentiation,
        regional_conclusion=regional_conclusion,
    )
    local = STAGE_PROMPTS["local"].format(
        synthesis=synthesis, coordinates=coordinates,
        location=f"{city}, {country}, {continent}",
    )
    full_reasoning = "\n\n".join([macro, regional, local])

    # Create entry
    entry = {
        "id": f"{city}_{country}".replace(" ", "_").lower(),
        "image": image_path,
        "city": city,
        "country": country,
        "continent": continent,
        "coordinates": coordinates,
        "output": full_reasoning,
    }

    # Save to corrections directory
    corr_dir = PROJECT_ROOT / "data/corrections"
    corr_dir.mkdir(parents=True, exist_ok=True)
    corr_file = corr_dir / f"{entry['id']}.json"
    with open(corr_file, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 修正文件已保存: {corr_file}")

    # Add to few-shot
    add_to_fewshot(entry)

    # Add to training JSONL
    add_to_training_jsonl(entry)

    print(f"\n完成! 已添加到:")
    print(f"  - Few-shot: {FEWSHOT_CHINA}")
    print(f"  - 训练集:   {TRAINING_JSONL}")


def add_to_fewshot(entry: dict):
    """Add a correction as a few-shot example."""
    fewshot = []
    if FEWSHOT_CHINA.exists():
        fewshot = json.loads(FEWSHOT_CHINA.read_text(encoding="utf-8"))

    fewshot.append({
        "id": entry["id"],
        "scene_type": entry.get("scene_type", "urban"),
        "region": f"{entry['city']}, {entry['country']}",
        "output": entry["output"],
    })

    FEWSHOT_CHINA.write_text(json.dumps(fewshot, ensure_ascii=False, indent=2), encoding="utf-8")


def add_to_training_jsonl(entry: dict):
    """Add a correction to the training JSONL (for LoRA fine-tuning)."""
    # instruction = 3-stage GeoCoT prompt for the LOCAL stage
    parts = entry["output"].split("\n\n")
    macro_text = parts[0] if len(parts) > 0 else ""
    regional_text = parts[1] if len(parts) > 1 else ""

    instruction = (
        "Analyze this street view image using the GeoCoT 3-stage framework:\n"
        "1. [MACRO] Identify continent-level context (climate, topography, vegetation)\n"
        "2. [REGIONAL] Identify country-specific indicators (language, architecture, traffic)\n"
        "3. [LOCAL] Identify city-level details and provide final coordinates\n\n"
        f"Based on the macro analysis ({macro_text[:200]}...) and "
        f"regional analysis ({regional_text[:200]}...), "
        "provide the final [LOCAL] stage reasoning, coordinates, and location."
    )

    record = {
        "image": entry["image"],
        "instruction": instruction,
        "response": entry["output"],
    }

    TRAINING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def batch_add(batch_file: str):
    """Load corrections from a batch JSON file."""
    with open(batch_file, "r", encoding="utf-8") as f:
        corrections = json.load(f)

    for i, entry in enumerate(corrections):
        # Validate required fields
        required = ["image", "city", "country", "continent", "output"]
        missing = [k for k in required if k not in entry]
        if missing:
            print(f"[SKIP] Entry {i}: missing {missing}")
            continue

        if not os.path.exists(entry["image"]):
            print(f"[SKIP] Entry {i}: image not found - {entry['image']}")
            continue

        entry.setdefault("id", f"{entry['city']}_{entry['country']}_{i}".replace(" ", "_").lower())
        entry.setdefault("coordinates", "")

        add_to_fewshot(entry)
        add_to_training_jsonl(entry)
        print(f"[OK] Entry {i}: {entry['id']}")

    print(f"\nBatch complete. {len(corrections)} corrections processed.")
    print(f"  Few-shot: {FEWSHOT_CHINA}")
    print(f"  Training: {TRAINING_JSONL}")


def generate_training_jsonl():
    """Generate training JSONL from existing few-shot examples (for bootstrapping)."""
    sources = [
        FEWSHOT_CHINA,
        PROJECT_ROOT / "src/Geocot/prompts/few_shot_examples.json",
    ]

    count = 0
    for src in sources:
        if not src.exists():
            continue
        examples = json.loads(src.read_text(encoding="utf-8"))
        for ex in examples:
            # Few-shot examples don't have images — skip for training
            if not ex.get("output"):
                continue
            # For few_shot_examples.json entries without image, skip
            # Only fewshot_china.json entries have meaningful full outputs
            if "output" in ex and len(ex["output"]) > 500:
                record = {
                    "image": ex.get("image", ""),
                    "instruction": "Analyze this street view image using GeoCoT 3-stage reasoning. Provide [MACRO], [REGIONAL], and [LOCAL] analysis with final COORDINATES and LOCATION.",
                    "response": ex["output"],
                }
                with open(TRAINING_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Generated {count} training records from few-shot examples → {TRAINING_JSONL}")


def main():
    parser = argparse.ArgumentParser(description="Add correction examples for GeoCoT")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", type=str, help="Batch JSON file with corrections")
    parser.add_argument("--generate-training", action="store_true",
                        help="Generate training JSONL from few-shot examples")
    args = parser.parse_args()

    if args.interactive:
        interactive_add()
    elif args.batch:
        batch_add(args.batch)
    elif args.generate_training:
        generate_training_jsonl()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python scripts/add_correction.py --interactive")


if __name__ == "__main__":
    main()
