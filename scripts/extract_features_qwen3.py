"""Extract Qwen3-VL-2B visual features (2048-dim) from training images.

Replaces the Qwen2-VL-2B 1536-dim features. Only runs vision encoder forward
pass (no text generation), so it's fast (~0.2s per image).

Output format matches train_moe.py expectations:
  {"features": (N, 2048), "coords": (N, 2)}

Usage:
    python scripts/extract_features_qwen3.py
    python scripts/extract_features_qwen3.py --max-images 100  # quick test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL-2B visual features for MoE training"
    )
    parser.add_argument("--data", type=str, default="data/merged_geotagged.json")
    parser.add_argument("--output", type=str, default="output/regression/features_qwen3.pt")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # ── Load image paths and coordinates ──────────────────────────────
    with open(args.data, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    all_paths = list(geo_data.keys())
    if args.max_images > 0:
        all_paths = all_paths[:args.max_images]

    valid_paths = []
    valid_coords = []
    for p in all_paths:
        abs_path = p
        if not os.path.isabs(p):
            abs_path = str(Path(p).resolve())
        if os.path.exists(abs_path):
            valid_paths.append(abs_path)
            valid_coords.append(geo_data[p])
        else:
            # Try as relative
            alt = str(Path(p))
            if os.path.exists(alt):
                valid_paths.append(alt)
                valid_coords.append(geo_data[p])

    print(f"Images: {len(valid_paths)} valid out of {len(all_paths)}")

    # ── Load Qwen3-VL-2B ──────────────────────────────────────────────
    from Geocot.Geocot import load_qwen2vl
    from regression.feature_extractor import VisualFeatureExtractor

    print("Loading Qwen3-VL-2B...")
    model, processor, _ = load_qwen2vl()
    extractor = VisualFeatureExtractor(model, processor)
    print(f"  Feature dim: {extractor.feature_dim}")

    # ── Extract features ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_features = []
    all_coords = []

    for i in tqdm(range(0, len(valid_paths), args.batch_size), desc="Extracting"):
        batch_paths = valid_paths[i:i + args.batch_size]
        batch_coords_raw = valid_coords[i:i + args.batch_size]

        batch_features = []
        batch_coords_norm = []

        for img_path, (true_lat, true_lng) in zip(batch_paths, batch_coords_raw):
            try:
                image = Image.open(img_path).convert("RGB")
                feat = extractor.extract(image).cpu()
                batch_features.append(feat)

                lat_norm = true_lat / 90.0
                lng_norm = true_lng / 180.0
                batch_coords_norm.append(torch.tensor([lat_norm, lng_norm]))

            except Exception as e:
                tqdm.write(f"  Error [{img_path}]: {e}")
                continue

        if batch_features:
            all_features.extend(batch_features)
            all_coords.extend(batch_coords_norm)

        # Incremental save
        if (i + args.batch_size) % args.save_interval == 0:
            features_t = torch.stack(all_features) if all_features else torch.empty(0, extractor.feature_dim)
            coords_t = torch.stack(all_coords) if all_coords else torch.empty(0, 2)
            ckpt_path = args.output.replace(".pt", f"_ckpt_{len(all_features)}.pt")
            torch.save({"features": features_t, "coords": coords_t}, ckpt_path)
            print(f"  Saved checkpoint: {len(all_features)} records")

    # ── Final save ────────────────────────────────────────────────────
    features_t = torch.stack(all_features) if all_features else torch.empty(0, extractor.feature_dim)
    coords_t = torch.stack(all_coords) if all_coords else torch.empty(0, 2)
    torch.save({"features": features_t, "coords": coords_t}, args.output)
    print(f"\nDone: {len(all_features)} records saved to {args.output}")
    print(f"  features: {features_t.shape}")
    print(f"  coords:   {coords_t.shape}")


if __name__ == "__main__":
    main()
