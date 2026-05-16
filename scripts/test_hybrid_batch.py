"""Batch test GeoCoT + regression hybrid pipeline."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from src.regression.pipeline import load_geocot_with_regression
from src.regression.coord_regressor import haversine_distance_km


def test():
    geotagged = json.load(open("data/merged_geotagged.json"))

    # Pick diverse test samples
    samples = {}
    for path, coords in geotagged.items():
        lat, lng = coords[0], coords[1]
        fname = os.path.basename(path)
        region = "other"
        # Categorize by rough region
        if 29.5 <= lat <= 31.5 and 117 <= lng <= 119:
            region = "huangshan"
        elif 26.8 <= lat <= 27.5 and 100 <= lng <= 100.4:
            region = "tiger_leaping"
        elif 30.5 <= lat <= 31.5 and 103 <= lng <= 104:
            region = "siguniang"
        elif 27.5 <= lat <= 28 and 117.5 <= lng <= 118:
            region = "wuyishan"
        elif 29 <= lat <= 29.5 and 110 <= lng <= 111:
            region = "zhangjiajie"

        if region not in samples and region != "other":
            samples[region] = (path, coords)

    if len(samples) < 3:
        # Fallback: pick random diverse lats and lngs
        items = list(geotagged.items())
        import random
        random.shuffle(items)
        for path, coords in items[:5]:
            samples[f"sample_{len(samples)}"] = (path, coords)

    print(f"Testing {len(samples)} images from different regions...")
    for r, (p, c) in samples.items():
        print(f"  {r}: {os.path.basename(p)} -> ({c[0]:.2f}, {c[1]:.2f})")

    print("\nLoading hybrid pipeline...")
    t0 = time.time()
    model, processor, pipeline = load_geocot_with_regression(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        regressor_path="output/regression/best_model.pt",
    )
    print(f"Loaded in {time.time()-t0:.0f}s\n")

    results = []
    for region, (img_path, true_coords) in samples.items():
        print(f"[{region}]", end=" ", flush=True)
        image = Image.open(img_path).convert("RGB")
        t0 = time.time()
        result = pipeline.run(image, use_regression=True)
        elapsed = time.time() - t0

        pred = result.final_prediction
        err_km = haversine_distance_km(
            torch.tensor([pred.latitude]), torch.tensor([pred.longitude]),
            torch.tensor([true_coords[0]]), torch.tensor([true_coords[1]])
        ).item()

        print(f"true=({true_coords[0]:.3f},{true_coords[1]:.3f}) "
              f"pred=({pred.latitude:.3f},{pred.longitude:.3f}) "
              f"err={err_km:.1f}km {elapsed:.0f}s")

        results.append({
            "region": region,
            "true_lat": true_coords[0], "true_lng": true_coords[1],
            "pred_lat": pred.latitude, "pred_lng": pred.longitude,
            "error_km": err_km,
            "city": pred.city, "country": pred.country,
            "time_s": elapsed,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    errors = [r["error_km"] for r in results]
    times = [r["time_s"] for r in results]
    print(f"  Samples: {len(results)}")
    print(f"  Errors (km): min={min(errors):.1f}, median={sorted(errors)[len(errors)//2]:.1f}, max={max(errors):.1f}, mean={sum(errors)/len(errors):.1f}")
    print(f"  Time (s):    min={min(times):.0f}, max={max(times):.0f}, mean={sum(times)/len(times):.0f}")
    print(f"  Training val_median: 84.7 km")
    print(f"  Test improvement:    {84.7 - sorted(errors)[len(errors)//2]:.0f} km better than training median" if sorted(errors)[len(errors)//2] < 84.7 else f"  Test: worse than training")

    print(f"\n{'='*60}")
    print(f"ARCHITECTURE REVIEW")
    print(f"{'='*60}")
    print(f"  Stage 1: GeoCoT 3-stage reasoning (Macro->Regional->Local)")
    print(f"  Stage 2: Qwen2-VL encoder -> 1536-dim features -> MLP -> (lat,lng)")
    print(f"  Stage 3: Fuse regression coords + GeoCoT location names")
    print(f"  Model: Qwen2-VL-2B-Instruct (frozen) + CoordRegressor (935K params)")
    print(f"  Training data: 7,489 image+GPS pairs")
    print(f"  Loss: Haversine (great-circle distance in km)")

    print(f"\n{'='*60}")
    print(f"BOTTLENECKS & NEXT STEPS")
    print(f"{'='*60}")
    print(f"  1. Speed: {min(times):.0f}-{max(times):.0f}s/image (target <5s for DK-2500)")
    print(f"     -> Model quantization (4-bit NF4) + OpenVINO export")
    print(f"  2. Accuracy: median ~{sorted(errors)[len(errors)//2]:.0f}km")
    print(f"     -> More training data, especially wilderness photos")
    print(f"  3. Wilderness focus: current data ~93% urban")
    print(f"     -> Need mountain trail photos for hiking glasses use case")


if __name__ == "__main__":
    test()
