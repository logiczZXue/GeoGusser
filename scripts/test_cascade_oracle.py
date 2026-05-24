"""Oracle test: measure cascade upper bound by constraining MoE to correct province.

Key question: If GeoCoT correctly identifies the province, what accuracy can MoE achieve?

Usage:
    python scripts/test_cascade_oracle.py
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.moe import MoERegressor, get_expert_configs
from regression.moe_config import (
    SceneType, SCENE_TYPE_NAMES, NUM_EXPERTS,
)
from regression.coord_regressor import haversine_distance_km
from regression.spatial_constraint import ConstraintBox

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Province bounding boxes (lat_min, lat_max, lng_min, lng_max)
PROVINCE_BBOXES = {
    "北京": (39.5, 41.0, 115.5, 117.5),
    "天津": (38.5, 40.5, 116.5, 118.5),
    "河北": (36.0, 42.5, 113.5, 120.0),
    "山西": (34.5, 41.0, 110.0, 114.5),
    "内蒙古": (37.5, 53.5, 97.0, 126.5),
    "辽宁": (38.5, 43.5, 119.0, 126.0),
    "吉林": (40.5, 46.5, 121.5, 131.5),
    "黑龙江": (43.0, 54.0, 121.0, 135.5),
    "上海": (30.5, 32.0, 120.5, 122.5),
    "江苏": (30.5, 35.5, 116.0, 122.0),
    "浙江": (27.0, 31.5, 118.0, 123.0),
    "安徽": (29.0, 35.0, 114.5, 120.0),
    "福建": (23.5, 28.5, 115.5, 121.0),
    "江西": (24.0, 30.5, 113.5, 118.5),
    "山东": (34.0, 38.5, 114.5, 123.0),
    "河南": (31.0, 36.5, 110.0, 117.0),
    "湖北": (29.0, 33.5, 108.0, 116.5),
    "湖南": (24.5, 30.5, 108.5, 114.5),
    "广东": (20.0, 25.5, 109.5, 117.5),
    "广西": (20.5, 26.5, 104.0, 112.5),
    "海南": (18.0, 20.5, 108.0, 111.5),
    "重庆": (28.0, 32.5, 105.0, 110.5),
    "四川": (26.0, 34.5, 97.0, 108.5),
    "贵州": (24.5, 29.5, 103.5, 110.0),
    "云南": (21.0, 29.5, 97.5, 106.5),
    "西藏": (26.5, 36.5, 78.0, 99.5),
    "陕西": (31.5, 39.5, 105.5, 111.5),
    "甘肃": (32.5, 43.0, 92.0, 109.0),
    "青海": (31.0, 39.5, 89.0, 103.5),
    "宁夏": (35.0, 39.5, 104.0, 107.5),
    "新疆": (34.0, 49.5, 73.0, 96.5),
    "台湾": (21.5, 25.5, 120.0, 122.5),
    "香港": (22.0, 23.0, 113.5, 114.5),
    "澳门": (22.0, 22.5, 113.0, 113.5),
}


def latlng_to_province(lat, lng):
    """Simple point-in-bbox province lookup."""
    for prov, (lat_min, lat_max, lng_min, lng_max) in PROVINCE_BBOXES.items():
        if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
            return prov
    return None


def province_to_constraint(prov_name, padding_km=50):
    """Convert province name to ConstraintBox."""
    if prov_name not in PROVINCE_BBOXES:
        return None
    lat_min, lat_max, lng_min, lng_max = PROVINCE_BBOXES[prov_name]
    # Expand by padding (approximate: 1 degree ≈ 111 km lat, ~102 km lng at 30N)
    pad_lat = padding_km / 111.0
    pad_lng = padding_km / 102.0
    lat_min -= pad_lat
    lat_max += pad_lat
    lng_min -= pad_lng
    lng_max += pad_lng

    center_lat = (lat_min + lat_max) / 2
    center_lng = (lng_min + lng_max) / 2
    # Radius as half-diagonal
    radius = haversine_distance_km(
        torch.tensor(center_lat), torch.tensor(center_lng),
        torch.tensor(lat_max), torch.tensor(lng_max)
    ).item()

    return ConstraintBox(
        lat_min=lat_min, lat_max=lat_max,
        lng_min=lng_min, lng_max=lng_max,
        center_lat=center_lat, center_lng=center_lng,
        radius_km=radius, confidence=1.0, source="oracle_province"
    )


def test_constrained_moe(moe, test_features, test_coords, test_paths, test_labels):
    """Test MoE with oracle province constraints."""
    N = test_features.shape[0]

    # No constraint (baseline)
    print("Testing UNCONSTRAINED MoE...")
    moe.eval()
    unconstrained_dists = []
    with torch.no_grad():
        for i in tqdm(range(0, N, 64), desc="Unconstrained"):
            batch = test_features[i:i+64].to(DEVICE).float()
            fused = moe.forward(batch)
            pred_lat = fused[:, 0].cpu() * 90.0
            pred_lng = fused[:, 1].cpu() * 180.0
            true_lat = test_coords[i:i+64, 0] * 90.0
            true_lng = test_coords[i:i+64, 1] * 180.0
            d = haversine_distance_km(pred_lat, pred_lng, true_lat, true_lng)
            unconstrained_dists.append(d)
    unconstrained_dists = torch.cat(unconstrained_dists)

    # Constrained by oracle province
    print("Testing PROVINCE-CONSTRAINED MoE...")
    constrained_dists = []
    failed_constraint = 0
    with torch.no_grad():
        for i in tqdm(range(N), desc="Constrained"):
            feat = test_features[i:i+1].to(DEVICE).float()
            true_lat = test_coords[i, 0].item() * 90.0
            true_lng = test_coords[i, 1].item() * 180.0

            prov = latlng_to_province(true_lat, true_lng)
            constraint = province_to_constraint(prov) if prov else None

            if constraint is None:
                # Fall back to unconstrained
                fused = moe.forward(feat)
                failed_constraint += 1
            else:
                fused, weights, expert_preds = moe.forward(feat, return_weights=True)
                # Blend: 70% constrained center, 30% expert prediction
                # Simulating predict_constrained behavior
                constraint_lat = constraint.center_lat / 90.0
                constraint_lng = constraint.center_lng / 180.0
                constraint_center = torch.tensor([[constraint_lat, constraint_lng]],
                                                  device=DEVICE, dtype=torch.float32)

                # Soft blend towards constraint center
                blend = 0.3  # weight for expert prediction
                fused = blend * fused + (1 - blend) * constraint_center

            pred_lat = fused[0, 0].cpu().item() * 90.0
            pred_lng = fused[0, 1].cpu().item() * 180.0
            d = haversine_distance_km(
                torch.tensor(pred_lat), torch.tensor(pred_lng),
                torch.tensor(true_lat), torch.tensor(true_lng)
            ).item()
            constrained_dists.append(d)

    constrained_dists = torch.tensor(constrained_dists)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"ORACLE PROVINCE CONSTRAINT — MoE Accuracy Upper Bound")
    print(f"{'='*60}")
    print(f"  Images outside China bbox: {failed_constraint}/{N}")

    print(f"\n  UNCONSTRAINED:")
    print(f"    Mean:   {unconstrained_dists.mean():.1f} km")
    print(f"    Median: {unconstrained_dists.median():.1f} km")
    print(f"    <10km:  {(unconstrained_dists < 10).float().mean()*100:.1f}%")
    print(f"    <50km:  {(unconstrained_dists < 50).float().mean()*100:.1f}%")
    print(f"    <100km: {(unconstrained_dists < 100).float().mean()*100:.1f}%")

    print(f"\n  CONSTRAINED (oracle province):")
    print(f"    Mean:   {constrained_dists.mean():.1f} km")
    print(f"    Median: {constrained_dists.median():.1f} km")
    print(f"    <10km:  {(constrained_dists < 10).float().mean()*100:.1f}%")
    print(f"    <50km:  {(constrained_dists < 50).float().mean()*100:.1f}%")
    print(f"    <100km: {(constrained_dists < 100).float().mean()*100:.1f}%")

    # By scene type
    print(f"\n  BY SCENE TYPE:")
    for c in range(NUM_EXPERTS):
        mask = test_labels == c
        if mask.any():
            un_d = unconstrained_dists[mask]
            co_d = constrained_dists[mask]
            name = SCENE_TYPE_NAMES.get(c, f"expert{c}")
            print(f"    {name}: unconstrained={un_d.mean():.0f}km → constrained={co_d.mean():.0f}km")

    return unconstrained_dists, constrained_dists


def main():
    output_dir = "output/moe_qwen3_rigorous"
    os.makedirs(output_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = torch.load("output/regression/features_qwen3_test.pt",
                           map_location="cpu", weights_only=True)
    test_features, test_coords = test_data["features"], test_data["coords"]

    # Load test paths
    with open("output/regression/test_images.json") as f:
        test_info = json.load(f)
    test_paths = list(test_info.keys())

    # Load scene labels
    with open("output/regression/scene_labels.json") as f:
        path_labels = json.load(f)
    test_labels = torch.full((test_features.shape[0],), SceneType.OTHER, dtype=torch.long)
    for idx, path in enumerate(test_paths):
        if path in path_labels:
            test_labels[idx] = int(path_labels[path])

    # Load MoE
    print("Loading MoE model...")
    train_data = torch.load("output/regression/features_qwen3_train.pt",
                            map_location="cpu", weights_only=True)
    input_dim = train_data["features"].shape[1]

    train_labels = torch.load("output/regression/scene_labels_train.pt",
                              map_location="cpu", weights_only=True)
    train_labels = torch.where(train_labels < 0, torch.tensor(SceneType.OTHER), train_labels)

    sample_counts = {}
    for c in range(NUM_EXPERTS):
        sample_counts[c] = (train_labels == c).sum().item()
    expert_configs = get_expert_configs(sample_counts)

    moe = MoERegressor(input_dim=input_dim, expert_configs=expert_configs)
    moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_final.pt"),
                                   map_location=DEVICE, weights_only=True))
    moe.to(DEVICE)
    moe.eval()

    # Run oracle test
    un_d, co_d = test_constrained_moe(
        moe, test_features, test_coords, test_paths, test_labels)

    # Save results
    torch.save({"unconstrained": un_d, "constrained": co_d},
               os.path.join(output_dir, "oracle_results.pt"))
    print(f"\nSaved to {output_dir}/oracle_results.pt")


if __name__ == "__main__":
    main()
