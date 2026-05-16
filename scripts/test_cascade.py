"""End-to-end cascade pipeline test: compare Cascade vs MoE-only on labeled images.

Usage:
    python scripts/test_cascade.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.pipeline import load_geocot_with_regression
from regression.spatial_constraint import ConstraintBox


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def load_test_images():
    """Load test images with known ground truth from scene_labels."""
    labels_path = "output/regression/scene_labels.json"
    data_path = "data/merged_geotagged.json"

    if not os.path.exists(labels_path) or not os.path.exists(data_path):
        print("Missing labels or data file")
        return []

    with open(labels_path) as f:
        labels = json.load(f)
    with open(data_path) as f:
        data = json.load(f)

    # Pick 2 images per scene type with known coordinates
    by_type = {0: [], 1: [], 2: [], 3: []}
    for path, st in labels.items():
        st_int = int(st)
        if path in data and os.path.exists(path):
            lat, lng = data[path][0], data[path][1]
            by_type[st_int].append((path, lat, lng))

    # Take first 2 from each type
    test_cases = []
    type_names = {0: "urban", 1: "karst_granite", 2: "alpine_snow", 3: "other"}
    for st_int, paths in by_type.items():
        for path, lat, lng in paths[:2]:
            test_cases.append({
                "path": path,
                "true_lat": lat,
                "true_lng": lng,
                "scene": type_names[st_int],
                "scene_type": st_int,
            })

    return test_cases


def main():
    print("=" * 70)
    print("  Cascade Pipeline End-to-End Test")
    print("=" * 70)

    # ── Load models ──────────────────────────────────────────────────
    print("\n[1/4] Loading Qwen2-VL-2B + MoE...")
    model, processor, pipeline = load_geocot_with_regression(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        moe_path="output/moe/moe_final.pt",
        load_in_4bit=False,
        enable_knowledge=True,
    )
    print(f"  MoE loaded: {pipeline.has_moe}")
    print(f"  Cascade loaded: {pipeline.has_cascade}")

    # ── Load test images ─────────────────────────────────────────────
    print("\n[2/4] Loading test images...")
    test_cases = load_test_images()
    if not test_cases:
        print("  No test images found! Using hardcoded paths.")
        # Fallback test cases
        test_cases = [
            {
                "path": "data/mapillary/1163967815472102.jpg",
                "true_lat": 30.0961, "true_lng": 118.1802,
                "scene": "karst_granite", "scene_type": 1,
            },
            {
                "path": "data/mapillary/1094119808150839.jpg",
                "true_lat": 27.0807, "true_lng": 100.2650,
                "scene": "alpine_snow", "scene_type": 2,
            },
        ]

    valid_cases = []
    for tc in test_cases:
        if os.path.exists(tc["path"]):
            valid_cases.append(tc)
            print(f"  [{tc['scene']}] {tc['path']} -> ({tc['true_lat']:.4f}, {tc['true_lng']:.4f})")
        else:
            print(f"  [SKIP] {tc['path']} not found")

    if not valid_cases:
        print("No valid test images!")
        return

    # ── Run inference ────────────────────────────────────────────────
    print("\n[3/4] Running inference (Cascade vs MoE-only)...")
    results = []

    for i, tc in enumerate(valid_cases):
        print(f"\n  --- Test {i+1}/{len(valid_cases)}: {tc['scene']} ---")
        image = Image.open(tc["path"]).convert("RGB")
        print(f"  Image size: {image.size}")
        true_lat, true_lng = tc["true_lat"], tc["true_lng"]

        # ---- CASCADE PATH ----
        print("  [Cascade] Running...")
        torch.cuda.empty_cache()
        t0 = time.time()
        cascade_result = pipeline.run(image, use_cascade=True, use_moe=False)
        cascade_time = time.time() - t0

        cascade_lat = cascade_result.final_prediction.latitude
        cascade_lng = cascade_result.final_prediction.longitude
        cascade_err = (
            haversine_km(cascade_lat, cascade_lng, true_lat, true_lng)
            if cascade_lat is not None else float("inf")
        )

        # Extract cascade-specific info
        cascade_info = {}
        if cascade_result.explanation:
            d = cascade_result.explanation.to_dict()
            cascade_info["confidence"] = d["confidence"]["stars"]
            cascade_info["primary_scene"] = d["scene_analysis"]["primary_scene"]
        cascade_raw = cascade_result.stage_outputs.get("cascade", "")
        cascade_reg = cascade_result.stage_outputs.get("regression", "")

        print(f"  [Cascade] ({cascade_lat:.4f}, {cascade_lng:.4f}) error={cascade_err:.1f}km time={cascade_time:.1f}s")
        if cascade_raw:
            import re
            for line in cascade_raw.split(","):
                if "narrow" in line or "radius" in line or "confidence" in line:
                    print(f"           {line.strip()}")

        # ---- MOE-ONLY PATH ----
        print("  [MoE-only] Running...")
        torch.cuda.empty_cache()
        t0 = time.time()
        moe_result = pipeline.run(image, use_cascade=False, use_moe=True)
        moe_time = time.time() - t0

        moe_lat = moe_result.final_prediction.latitude
        moe_lng = moe_result.final_prediction.longitude
        moe_err = (
            haversine_km(moe_lat, moe_lng, true_lat, true_lng)
            if moe_lat is not None else float("inf")
        )

        moe_info = {}
        if moe_result.explanation:
            d = moe_result.explanation.to_dict()
            moe_info["confidence"] = d["confidence"]["stars"]
            moe_info["primary_scene"] = d["scene_analysis"]["primary_scene"]

        print(f"  [MoE-only] ({moe_lat:.4f}, {moe_lng:.4f}) error={moe_err:.1f}km time={moe_time:.1f}s")

        results.append({
            "scene": tc["scene"],
            "true": (true_lat, true_lng),
            "cascade": (cascade_lat, cascade_lng, cascade_err, cascade_time, cascade_info),
            "moe": (moe_lat, moe_lng, moe_err, moe_time, moe_info),
        })

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Scene':<16s} {'True':>20s} {'Cascade':>20s} {'C-Err':>8s} {'MoE':>20s} {'M-Err':>8s} {'Delta':>8s}")
    print("-" * 90)

    for r in results:
        scene = r["scene"]
        true_str = f"({r['true'][0]:.3f}, {r['true'][1]:.3f})"
        c_str = f"({r['cascade'][0]:.3f}, {r['cascade'][1]:.3f})" if r['cascade'][0] else "N/A"
        c_err = f"{r['cascade'][2]:.1f}km" if r['cascade'][2] < 1e9 else "N/A"
        m_str = f"({r['moe'][0]:.3f}, {r['moe'][1]:.3f})" if r['moe'][0] else "N/A"
        m_err = f"{r['moe'][2]:.1f}km" if r['moe'][2] < 1e9 else "N/A"
        delta = (r['moe'][2] - r['cascade'][2]) if (r['cascade'][2] < 1e9 and r['moe'][2] < 1e9) else 0
        delta_str = f"{delta:+.1f}km"
        print(f"{scene:<16s} {true_str:>20s} {c_str:>20s} {c_err:>8s} {m_str:>20s} {m_err:>8s} {delta_str:>8s}")

    # Average error (excluding inf)
    cascade_errs = [r['cascade'][2] for r in results if r['cascade'][2] < 1e9]
    moe_errs = [r['moe'][2] for r in results if r['moe'][2] < 1e9]
    if cascade_errs:
        print(f"\n  Cascade avg error: {sum(cascade_errs)/len(cascade_errs):.1f} km ({len(cascade_errs)} samples)")
    if moe_errs:
        print(f"  MoE avg error:     {sum(moe_errs)/len(moe_errs):.1f} km ({len(moe_errs)} samples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
