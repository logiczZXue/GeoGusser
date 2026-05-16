"""Test GeoCoT + regression hybrid pipeline on sample images."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.regression.pipeline import load_geocot_with_regression


def main():
    # Pick a few test images from our data
    geotagged = json.load(open("data/merged_geotagged.json"))

    # Pick one urban and one trail image
    urban_img = None
    trail_img = None
    for path in geotagged:
        if "mapillary_trails" in path and trail_img is None:
            trail_img = path
        if "mapillary\\" in path and urban_img is None:
            urban_img = path
        if urban_img and trail_img:
            break

    test_images = []
    if trail_img:
        test_images.append(("野外轨迹", trail_img, geotagged[trail_img]))
    if urban_img:
        test_images.append(("城市", urban_img, geotagged[urban_img]))

    print("Loading hybrid pipeline (Qwen2-VL-2B + regression head)...")
    t0 = time.time()
    model, processor, pipeline = load_geocot_with_regression(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        regressor_path="output/regression/best_model.pt",
    )
    print(f"Loaded in {time.time()-t0:.0f}s")

    for label, img_path, true_coords in test_images:
        print(f"\n{'='*60}")
        print(f"[{label}] {os.path.basename(img_path)}")
        print(f"  真实坐标: {true_coords[0]:.4f}, {true_coords[1]:.4f}")

        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        result = pipeline.run(image, use_regression=True)
        elapsed = time.time() - t0

        pred = result.final_prediction
        print(f"  预测坐标: {pred.latitude:.4f}, {pred.longitude:.4f}")
        print(f"  预测位置: {pred.city}, {pred.country}, {pred.continent}")
        print(f"  推理耗时: {elapsed:.1f}s")

        # Compute error
        from src.regression.coord_regressor import haversine_distance_km
        import torch
        err = haversine_distance_km(
            torch.tensor([pred.latitude]), torch.tensor([pred.longitude]),
            torch.tensor([true_coords[0]]), torch.tensor([true_coords[1]])
        )
        print(f"  定位误差: {err.item():.1f} km")

        if "regression" in result.stage_outputs:
            print(f"\n  [回归头输出]")
            print(f"  {result.stage_outputs['regression'][:200]}")


if __name__ == "__main__":
    main()
