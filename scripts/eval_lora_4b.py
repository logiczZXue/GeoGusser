"""Evaluate LoRA 4B-Instruct vs base on GeoCoT element extraction quality.

Compares element-level output quality, not just coordinates.
Uses 5 test images from the original 5-image test set.
"""
import os, sys, time, math, json
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
_src_dir = _scripts_dir.parent / "src"
sys.path.insert(0, str(_src_dir))

import torch
from PIL import Image
from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl, create_qwen2vl_model_fn

MODEL_4B = "D:/Geocomp/models/Qwen3-VL-4B-Instruct/qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "D:/Geocomp/output/vlm_lora_test/lora_adapter"

# 5 test images: Xishuangbanna temple + 4 diverse scenes picked from merged_geotagged
# We'll use known images from the v2.3 test set
TEST_DIR = Path("D:/Geocomp/data/mapillary")
TEST_IMAGES = []

# Pick images that exist and have valid GPS
with open("D:/Geocomp/data/merged_geotagged.json") as f:
    geotagged = json.load(f)

# Collect some images from the test set
for path, (lat, lng) in geotagged.items():
    if len(TEST_IMAGES) >= 5:
        break
    p = Path(path)
    if p.exists() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        TEST_IMAGES.append((str(p), lat, lng))

print(f"Test images: {len(TEST_IMAGES)}")


def haversine(lat1, lng1, lat2, lng2):
    if None in (lat1, lng1, lat2, lng2):
        return float('inf')
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlng/2)**2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def test_model(model_fn, name, images):
    """Run GeoCoT on images and return element outputs + timing."""
    pipeline = GeoCoTPipeline(model_fn, {'prompts_dir': str(_src_dir / 'Geocot' / 'prompts')})
    results = []

    for img_path, true_lat, true_lng in images:
        image = Image.open(img_path).convert("RGB")
        print(f"  [{name}] {Path(img_path).name}...", end=" ", flush=True)

        t0 = time.time()
        result = pipeline.run(image)
        dt = time.time() - t0

        macro = result.stage_outputs.get("macro", "")
        regional = result.stage_outputs.get("regional", "")
        local = result.stage_outputs.get("local", "")

        results.append({
            "image": Path(img_path).name,
            "true_lat": true_lat, "true_lng": true_lng,
            "macro": macro,
            "regional": regional,
            "local": local,
            "time_s": round(dt, 1),
        })
        print(f"{dt:.0f}s")

    return results


def main():
    print("=" * 70)
    print("Loading BASE model (no LoRA)...")
    model, processor, _ = load_qwen2vl(
        model_name=MODEL_4B,
        load_in_4bit=True,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.8,
        enable_thinking=False,
    )
    base_fn = create_qwen2vl_model_fn(model, processor, max_new_tokens=4096, temperature=0.6, top_p=0.8, enable_thinking=False)

    print("\nRunning BASE model on 5 images...")
    base_results = test_model(base_fn, "BASE", TEST_IMAGES)

    # Free memory
    del model, processor, base_fn
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Loading LoRA model...")
    lora_model, lora_proc, _ = load_qwen2vl(
        model_name=MODEL_4B,
        load_in_4bit=True,
        lora_path=LORA_PATH,
        merge_lora=False,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.8,
        enable_thinking=False,
    )
    lora_fn = create_qwen2vl_model_fn(lora_model, lora_proc, max_new_tokens=4096, temperature=0.6, top_p=0.8, enable_thinking=False)

    print("\nRunning LoRA model on 5 images...")
    lora_results = test_model(lora_fn, "LORA", TEST_IMAGES)

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for i, (base_r, lora_r) in enumerate(zip(base_results, lora_results)):
        print(f"\n── Image {i+1}: {base_r['image']} ──")
        print(f"  Truth: ({base_r['true_lat']:.4f}, {base_r['true_lng']:.4f})")
        print(f"  Base:  {len(base_r['macro']):4d} chars macro, {base_r['time_s']:5.0f}s")
        print(f"  LoRA:  {len(lora_r['macro']):4d} chars macro, {lora_r['time_s']:5.0f}s")
        print(f"  Base macro (first 200):")
        print(f"    {base_r['macro'][:200]}")
        print(f"  LoRA macro (first 200):")
        print(f"    {lora_r['macro'][:200]}")

    # Save full results
    out_path = "D:/Geocomp/output/vlm_lora_test/eval_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "lora": lora_results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
