"""Run full evaluation comparing GeoCoT vs baseline.

Usage:
    python scripts/run_evaluation.py --geocot-dir output/geocot --baseline-dir output/baseline --gt-dir src/Geoeval/Ground_Truth
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.Geoeval.OfflineInferenceScorer import OfflineInferenceScorer, score_directory
from src.Geoeval.OfflineLatLngPredictor import OfflineLatLngPredictor


def compare_predictions(geocot_pred_dir: str, baseline_pred_dir: str, gt_pred_dir: str):
    """Compare GeoCoT and baseline predictions against ground truth."""
    gt_files = set(os.listdir(gt_pred_dir)) if os.path.exists(gt_pred_dir) else set()

    geocot_correct = {"city": 0, "country": 0, "continent": 0}
    baseline_correct = {"city": 0, "country": 0, "continent": 0}
    total = 0

    for fname in sorted(gt_files):
        if not fname.endswith(".txt"):
            continue

        gt_path = os.path.join(gt_pred_dir, fname)
        geocot_path = os.path.join(geocot_pred_dir, fname)
        baseline_path = os.path.join(baseline_pred_dir, fname)

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_parts = [p.strip() for p in f.read().strip().split(",")]

        if len(gt_parts) >= 3:
            total += 1
            # Check GeoCoT
            if os.path.exists(geocot_path):
                with open(geocot_path, "r", encoding="utf-8") as f:
                    pred_parts = [p.strip() for p in f.read().strip().split(",")]
                if len(pred_parts) >= 3:
                    if pred_parts[0].lower() == gt_parts[0].lower():
                        geocot_correct["city"] += 1
                    if pred_parts[1].lower() == gt_parts[1].lower():
                        geocot_correct["country"] += 1
                    if pred_parts[2].lower() == gt_parts[2].lower():
                        geocot_correct["continent"] += 1

            # Check baseline
            if os.path.exists(baseline_path):
                with open(baseline_path, "r", encoding="utf-8") as f:
                    pred_parts = [p.strip() for p in f.read().strip().split(",")]
                if len(pred_parts) >= 3:
                    if pred_parts[0].lower() == gt_parts[0].lower():
                        baseline_correct["city"] += 1
                    if pred_parts[1].lower() == gt_parts[1].lower():
                        baseline_correct["country"] += 1
                    if pred_parts[2].lower() == gt_parts[2].lower():
                        baseline_correct["continent"] += 1

    if total == 0:
        print("No matching ground truth files found.")
        return

    print(f"\n{'='*50}")
    print(f"{'Metric':<20} {'GeoCoT':>10} {'Baseline':>10}")
    print(f"{'='*50}")
    for level in ["city", "country", "continent"]:
        g_acc = geocot_correct[level] / total
        b_acc = baseline_correct[level] / total
        print(f"{level+' accuracy':<20} {g_acc:>10.3f} {b_acc:>10.3f}")
    print(f"{'='*50}")
    print(f"{'Total samples':<20} {total:>10}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoCoT vs baseline")
    parser.add_argument("--geocot-dir", type=str, required=True, help="GeoCoT output directory")
    parser.add_argument("--baseline-dir", type=str, default=None, help="Baseline output directory")
    parser.add_argument("--gt-dir", type=str, default=None, help="Ground truth predictions directory")
    args = parser.parse_args()

    gt_dir = args.gt_dir or os.path.join(args.geocot_dir, "..", "ground_truth", "predictions")

    # Classification comparison
    if args.baseline_dir:
        geocot_pred = os.path.join(args.geocot_dir, "predictions")
        baseline_pred = os.path.join(args.baseline_dir, "predictions")
        compare_predictions(geocot_pred, baseline_pred, gt_dir)

    # Reasoning quality
    geocot_reasoning = os.path.join(args.geocot_dir, "reasoning")
    gt_reasoning = os.path.join(args.geocot_dir, "..", "ground_truth", "reasoning")

    if os.path.exists(geocot_reasoning):
        print("\n=== Reasoning Quality (GeoCoT) ===")
        score_directory(geocot_reasoning, gt_reasoning)


if __name__ == "__main__":
    main()
