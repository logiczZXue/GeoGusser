"""Compare three inference modes on test images.

Modes:
  1. Pure MoE:      visual features → MoE → coordinates
  2. Text Cascade:  GeoCoT → Parser → SpatialKB → Constraint → MoE
  3. Feature Fusion: visual + VLM hidden states → FusedMoE → coordinates

Usage:
    python scripts/eval_fused_moe.py
    python scripts/eval_fused_moe.py --moe output/moe/moe_final.pt --fused output/fused_moe/fused_moe_final.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_model():
    """Load VLM once for all modes."""
    from Geocot.Geocot import load_qwen2vl
    return load_qwen2vl("Qwen/Qwen2-VL-2B-Instruct")


def run_moe(image, model, processor, moe_path: str):
    """Mode 1: Pure MoE regression."""
    from regression.feature_extractor import VisualFeatureExtractor
    from regression.moe import MoERegressor

    extractor = VisualFeatureExtractor(model, processor)
    visual = extractor.extract(image).float().unsqueeze(0)

    state = torch.load(moe_path, map_location="cpu", weights_only=True)
    moe = MoERegressor()
    moe.load_state_dict(state)
    moe.to(model.device)
    moe.eval()

    with torch.no_grad():
        lat, lng = moe.predict(visual)
    return lat.item(), lng.item()


def run_cascade(image, model, processor, model_fn, moe_path: str):
    """Mode 2: Text cascade (GeoCoT → Parser → SpatialKB → Constraint → MoE)."""
    from regression.pipeline import load_geocot_with_regression

    _, _, pipeline = load_geocot_with_regression(
        "Qwen/Qwen2-VL-2B-Instruct",
        moe_path=moe_path,
    )
    # Override with our model to avoid double-loading
    pipeline._geocot._model_fn = model_fn
    pipeline._extractor = None  # force use of existing model setup

    # Actually, simpler: just use the pipeline's own run with cascade
    result = pipeline.run(image)
    lat = result.final_prediction.latitude
    lng = result.final_prediction.longitude
    return lat, lng if (lat and lng) else (None, None)


def run_fused(image, model, processor, fused_moe_path: str):
    """Mode 3: Feature fusion (visual + VLM hidden states → FusedMoE)."""
    from regression.pipeline import load_geocot_with_regression

    _, _, pipeline = load_geocot_with_regression(
        "Qwen/Qwen2-VL-2B-Instruct",
        fused_moe_path=fused_moe_path,
    )
    result = pipeline.run(image)
    lat = result.final_prediction.latitude
    lng = result.final_prediction.longitude
    return lat, lng if (lat and lng) else (None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Pure MoE vs Text Cascade vs Feature Fusion"
    )
    parser.add_argument("--moe", type=str, default="output/moe/moe_final.pt")
    parser.add_argument("--fused", type=str, default="output/fused_moe/fused_moe_final.pt")
    parser.add_argument("--images", type=str, nargs="*", default=None,
                        help="Specific test images; defaults to 8-image test set")
    parser.add_argument("--index", type=str, default="data/merged_geotagged.json")
    args = parser.parse_args()

    # ── Determine test images ────────────────────────────────────────
    with open(args.index, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    if args.images:
        test_paths = args.images
    else:
        # Use the standard 8-image test set paths from batch_test_8_images
        test_paths = [
            "data/mapillary/1163967815472102.jpg",  # Karst (Huangshan)
        ]
        # Fill from geo_data
        existing = [p for p in geo_data if os.path.exists(p)]
        test_paths = existing[:8]

    test_paths = [p for p in test_paths if os.path.exists(p)]
    if not test_paths:
        print("No valid test images found!")
        return

    print(f"Test images: {len(test_paths)}")
    print(f"Truth file: {args.index}")

    # ── Load VLM ─────────────────────────────────────────────────────
    print("\nLoading VLM (once, shared)...")
    model, processor, model_fn = load_model()

    # ── Check available modes ─────────────────────────────────────────
    has_moe = os.path.exists(args.moe)
    has_fused = os.path.exists(args.fused)
    has_cascade = has_moe  # cascade needs MoE

    print(f"  Pure MoE:      {'AVAILABLE' if has_moe else 'missing'}")
    print(f"  Text Cascade:  {'AVAILABLE' if has_cascade else 'missing'}")
    print(f"  Feature Fusion:{'AVAILABLE' if has_fused else 'missing'}")

    # ── Run comparison ────────────────────────────────────────────────
    results = {"moe": [], "cascade": [], "fused": [], "truth": []}

    for i, img_path in enumerate(test_paths):
        # Resolve path to match geo_data keys (may be relative or absolute)
        coords = None
        if img_path in geo_data:
            coords = geo_data[img_path]
        else:
            abs_path = str(Path(img_path).resolve())
            if abs_path in geo_data:
                coords = geo_data[abs_path]
        if coords is None:
            print(f"  SKIP: {img_path} not in truth index")
            continue
        true_lat, true_lng = coords
        print(f"\n--- Image {i+1}/{len(test_paths)}: {Path(img_path).name}")
        print(f"    Truth: ({true_lat:.4f}, {true_lng:.4f})")

        results["truth"].append((true_lat, true_lng))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    ERROR loading image: {e}")
            continue

        # Mode 1: Pure MoE
        if has_moe:
            t0 = time.time()
            try:
                lat, lng = run_moe(image, model, processor, args.moe)
                err = haversine_km(lat, lng, true_lat, true_lng)
                print(f"    Pure MoE:     ({lat:.4f}, {lng:.4f}) error={err:.1f}km [{time.time()-t0:.1f}s]")
                results["moe"].append((lat, lng, err))
            except Exception as e:
                print(f"    Pure MoE:     ERROR: {e}")
                results["moe"].append(None)

        # Mode 2: Text Cascade
        if has_cascade:
            t0 = time.time()
            try:
                lat, lng = run_cascade(image, model, processor, model_fn, args.moe)
                if lat is not None:
                    err = haversine_km(lat, lng, true_lat, true_lng)
                    print(f"    Text Cascade: ({lat:.4f}, {lng:.4f}) error={err:.1f}km [{time.time()-t0:.1f}s]")
                    results["cascade"].append((lat, lng, err))
                else:
                    print(f"    Text Cascade: no prediction")
                    results["cascade"].append(None)
            except Exception as e:
                print(f"    Text Cascade: ERROR: {e}")
                results["cascade"].append(None)

        # Mode 3: Feature Fusion
        if has_fused:
            t0 = time.time()
            try:
                lat, lng = run_fused(image, model, processor, args.fused)
                if lat is not None:
                    err = haversine_km(lat, lng, true_lat, true_lng)
                    print(f"    Feature Fusion:({lat:.4f}, {lng:.4f}) error={err:.1f}km [{time.time()-t0:.1f}s]")
                    results["fused"].append((lat, lng, err))
                else:
                    print(f"    Feature Fusion: no prediction")
                    results["fused"].append(None)
            except Exception as e:
                print(f"    Feature Fusion: ERROR: {e}")
                results["fused"].append(None)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for mode_name, mode_key in [("Pure MoE", "moe"), ("Text Cascade", "cascade"),
                                  ("Feature Fusion", "fused")]:
        valid = [r for r in results[mode_key] if r is not None]
        if not valid:
            print(f"\n{mode_name}: no results")
            continue
        errors = [r[2] for r in valid]
        mean_err = sum(errors) / len(errors)
        med_err = sorted(errors)[len(errors) // 2]
        print(f"\n{mode_name} ({len(valid)} images):")
        print(f"  Mean:   {mean_err:.1f} km")
        print(f"  Median: {med_err:.1f} km")
        print(f"  Min:    {min(errors):.1f} km")
        print(f"  Max:    {max(errors):.1f} km")

    # ── Direct comparison on images with all three ────────────────────
    print("\n" + "=" * 80)
    print("PER-IMAGE COMPARISON")
    print("=" * 80)
    header = f"{'Image':<35} {'Truth':>20} {'MoE':>10} {'Cascade':>10} {'Fused':>10}"
    print(header)
    print("-" * 85)

    for i, (path, truth) in enumerate(zip(test_paths, results["truth"])):
        name = Path(path).name[:32]
        truth_str = f"({truth[0]:.2f},{truth[1]:.2f})"

        moe_str = f"{results['moe'][i][2]:.0f}km" if i < len(results["moe"]) and results["moe"][i] else "---"
        cas_str = f"{results['cascade'][i][2]:.0f}km" if i < len(results["cascade"]) and results["cascade"][i] else "---"
        fus_str = f"{results['fused'][i][2]:.0f}km" if i < len(results["fused"]) and results["fused"][i] else "---"

        print(f"{name:<35} {truth_str:>20} {moe_str:>10} {cas_str:>10} {fus_str:>10}")


if __name__ == "__main__":
    main()
