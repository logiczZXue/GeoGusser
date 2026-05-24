"""Batch test: run GeoCoT on all 8 test images and print summary.

Usage:
    python scripts/batch_test_8_images.py [--image-dir "primary test/"]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.Geocot.Geocot import GeoCoTPipeline, load_qwen2vl
from src.Geocot.prediction_extractor import format_prediction


def find_unique_images(image_dir: str) -> list[str]:
    """Find unique images, deduplicating by base name (same image in JPG+PNG)."""
    seen = {}
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(f)[0]
        if base not in seen:
            seen[base] = os.path.join(image_dir, f)
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Batch test GeoCoT on 8 images")
    parser.add_argument("--image-dir", type=str, default="primary test")
    parser.add_argument("--output-dir", type=str, default="output/geocot")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization (for 7B on 8GB VRAM)")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offloading (slower fallback)")
    parser.add_argument("--gpu-memory", type=str, default=None, help="Max GPU memory, e.g. 6GB")
    args = parser.parse_args()

    images = find_unique_images(args.image_dir)
    print(f"Found {len(images)} unique images in '{args.image_dir}'")

    # Load model
    print(f"\nLoading {args.model}...")
    t0 = time.time()
    offload_folder = "output/offload" if args.offload else None
    model, processor, model_fn = load_qwen2vl(
        args.model,
        load_in_4bit=args.__dict__.get("4bit", False),
        offload_folder=offload_folder,
        gpu_memory=args.gpu_memory or ("6GB" if args.offload else None),
    )
    print(f"Loaded in {time.time() - t0:.0f}s")

    # Build pipeline (uses the improved prompts from src/Geocot/prompts/)
    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "src", "Geocot", "prompts")
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": prompts_dir})

    # Output directories
    reasoning_dir = Path(args.output_dir) / "reasoning"
    predictions_dir = Path(args.output_dir) / "predictions"
    reasoning_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\nProcessing {len(images)} images...")
    print("=" * 70)

    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[IMG] {name}")
        print("-" * 50)

        t1 = time.time()
        image = Image.open(img_path).convert("RGB")

        try:
            result = pipeline.run(image)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append((name, "ERROR", "ERROR", "ERROR", 0))
            continue

        elapsed = time.time() - t1
        pred = format_prediction(result.final_prediction)

        # Print brief results (full output saved to files)
        print(f"  Stage 1 (Macro):    {result.stage_outputs.get('macro', '')[:120]}...")
        print(f"  Stage 2 (Regional): {result.stage_outputs.get('regional', '')[:120]}...")
        print(f"  Stage 3 (Local):    {result.stage_outputs.get('local', '')[:120]}...")
        print(f"  >> PREDICTION: {pred}  ({elapsed:.1f}s)")

        # Save
        (reasoning_dir / f"{name}.txt").write_text(result.reasoning_chain, encoding="utf-8")
        (predictions_dir / f"{name}.txt").write_text(pred, encoding="utf-8")

        lat_str = f"{result.final_prediction.latitude:.2f}" if result.final_prediction.latitude else "?"
        lng_str = f"{result.final_prediction.longitude:.2f}" if result.final_prediction.longitude else "?"
        city = result.final_prediction.city or "?"
        country = result.final_prediction.country or "?"
        continent = result.final_prediction.continent or "?"
        results.append((name, city, country, continent, lat_str, lng_str, elapsed))

    # Summary table
    print("\n")
    print("=" * 95)
    print("SUMMARY - GeoCoT Batch Test Results")
    print("=" * 95)
    print(f"{'Image':<18s} {'City':<15s} {'Country':<15s} {'Continent':<12s} {'Lat':>8s} {'Lng':>8s} {'Time':>7s}")
    print("-" * 95)
    for name, city, country, continent, lat, lng, elapsed in results:
        print(f"{name:<18s} {city:<15s} {country:<15s} {continent:<12s} {lat:>8s} {lng:>8s} {elapsed:>6.1f}s")
    print("-" * 95)

    total_time = sum(r[6] for r in results)
    print(f"Total: {len(results)} images, {total_time:.0f}s total ({total_time/len(results):.0f}s avg)")

    print(f"\nFull reasoning saved to: {reasoning_dir}/")
    print(f"Predictions saved to:    {predictions_dir}/")


if __name__ == "__main__":
    main()
