"""Retrieval-based geolocation baseline using FAISS vector search.

For each query image:
  1. Extract/load Qwen3-VL-2B visual feature (2048-dim)
  2. Find top-k most similar training images via cosine similarity
  3. Weighted average of their GPS coordinates
  4. Evaluate on held-out test set

Usage:
    python scripts/retrieval_baseline.py
    python scripts/retrieval_baseline.py --top-k 5
    python scripts/retrieval_baseline.py --top-k 10 --search-radius-km 500
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.moe_config import SceneType, SCENE_TYPE_NAMES, NUM_EXPERTS


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def evaluate_retrieval(train_features, train_coords, test_features, test_coords,
                       test_labels, top_k=5, search_radius_km=None):
    """Evaluate retrieval-based geolocation.

    Args:
        train_features: (N, D) normalized training features
        train_coords: (N, 2) normalized training coords [lat/90, lng/180]
        test_features: (M, D) normalized test features
        test_coords: (M, 2) normalized test coords
        test_labels: (M,) scene type labels
        top_k: number of nearest neighbors to use
        search_radius_km: if set, only search within this radius (oracle simulation)

    Returns:
        dict with results
    """
    N_train = train_features.shape[0]
    N_test = test_features.shape[0]

    # Build FAISS index
    dim = train_features.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity for normalized vectors
    index.add(train_features.numpy().astype(np.float32))

    # Search
    scores, indices = index.search(test_features.numpy().astype(np.float32), top_k)

    # Denormalize coordinates
    train_lat = train_coords[:, 0].numpy() * 90.0
    train_lng = train_coords[:, 1].numpy() * 180.0
    true_lat = test_coords[:, 0].numpy() * 90.0
    true_lng = test_coords[:, 1].numpy() * 180.0

    pred_lats = np.zeros(N_test)
    pred_lngs = np.zeros(N_test)
    distances = np.zeros(N_test)

    for i in range(N_test):
        nn_indices = indices[i]
        nn_scores = scores[i]

        # Softmax weights from similarity scores
        weights = np.exp(nn_scores * 10)  # temperature=0.1 for sharper weighting
        weights = weights / weights.sum()

        nn_lats = train_lat[nn_indices]
        nn_lngs = train_lng[nn_indices]

        # Circular mean for longitude (handle -180/+180 wrap)
        pred_lat = np.sum(weights * nn_lats)

        # Weighted circular mean for longitude
        lng_rad = np.radians(nn_lngs)
        pred_lng_sin = np.sum(weights * np.sin(lng_rad))
        pred_lng_cos = np.sum(weights * np.cos(lng_rad))
        pred_lng = np.degrees(np.arctan2(pred_lng_sin, pred_lng_cos))

        pred_lats[i] = pred_lat
        pred_lngs[i] = pred_lng
        distances[i] = haversine_km(pred_lat, pred_lng, true_lat[i], true_lng[i])

    return {
        "distances": distances,
        "pred_lats": pred_lats,
        "pred_lngs": pred_lngs,
        "indices": indices,
        "scores": scores,
    }


def print_results(distances, test_labels, top_k):
    print(f"\n{'='*60}")
    print(f"RETRIEVAL BASELINE — top-{top_k} nearest neighbors")
    print(f"{'='*60}")

    for c in range(NUM_EXPERTS):
        mask = test_labels.numpy() == c
        if mask.any():
            c_dists = distances[mask]
            name = SCENE_TYPE_NAMES.get(c, f"expert{c}")
            print(f"  {name}: n={mask.sum()}, "
                  f"mean={c_dists.mean():.1f}km, median={np.median(c_dists):.1f}km")

    print(f"\n  OVERALL ({len(distances)} images):")
    print(f"    Mean:   {distances.mean():.1f} km")
    print(f"    Median: {np.median(distances):.1f} km")
    print(f"    Std:    {distances.std():.1f} km")
    print(f"    <10km:  {(distances < 10).mean()*100:.1f}%")
    print(f"    <25km:  {(distances < 25).mean()*100:.1f}%")
    print(f"    <50km:  {(distances < 50).mean()*100:.1f}%")
    print(f"    <100km: {(distances < 100).mean()*100:.1f}%")
    print(f"    <200km: {(distances < 200).mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="FAISS retrieval baseline for geolocation")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of nearest neighbors (default: 5)")
    parser.add_argument("--search-radius-km", type=float, default=None,
                        help="Oracle radius for constrained search (default: None = full)")
    parser.add_argument("--output", type=str, default="output/regression/retrieval_results.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load data
    print("Loading features...")
    train_data = torch.load("output/regression/features_qwen3_train.pt",
                            map_location="cpu", weights_only=True)
    test_data = torch.load("output/regression/features_qwen3_test.pt",
                           map_location="cpu", weights_only=True)
    train_features, train_coords = train_data["features"].float(), train_data["coords"].float()
    test_features, test_coords = test_data["features"].float(), test_data["coords"].float()

    # Load scene labels for test set
    test_labels = torch.full((test_features.shape[0],), SceneType.OTHER, dtype=torch.long)
    with open("output/regression/scene_labels.json") as f:
        path_labels = json.load(f)
    with open("output/regression/test_images.json") as f:
        test_info = json.load(f)
    for idx, path in enumerate(test_info.keys()):
        if path in path_labels:
            test_labels[idx] = int(path_labels[path])

    # Normalize features for cosine similarity
    print(f"Normalizing {train_features.shape[0]} train + {test_features.shape[0]} test features...")
    train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
    test_features = torch.nn.functional.normalize(test_features, p=2, dim=1)

    # Evaluate
    print(f"Running retrieval with top-k={args.top_k}...")
    results = evaluate_retrieval(
        train_features, train_coords, test_features, test_coords,
        test_labels, top_k=args.top_k,
    )

    print_results(results["distances"], test_labels, args.top_k)

    # Save
    np.savez(args.output,
             distances=results["distances"],
             pred_lats=results["pred_lats"],
             pred_lngs=results["pred_lngs"],
             indices=results["indices"],
             scores=results["scores"])
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
