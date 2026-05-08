"""Offline inference quality scorer — replaces GPT-4o-based InferenceQualityScorer.

Uses semantic similarity + rule-based heuristics to score reasoning quality
without requiring internet access.
"""

import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class OfflineInferenceScorer:
    """Score reasoning quality using local models and heuristics."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def semantic_score(self, prediction_text: str, ground_truth_text: str) -> float:
        """Compute cosine similarity between prediction and ground truth."""
        emb_pred = self._model.encode([prediction_text])
        emb_gt = self._model.encode([ground_truth_text])
        return float(cosine_similarity(emb_pred, emb_gt)[0][0])

    def clue_coverage_score(self, prediction_text: str) -> float:
        """Check if the prediction covers key geographic clue categories."""
        clue_categories = {
            "climate": r"(?:tropical|temperate|arid|polar|climate|weather|rain|sun|snow)",
            "vegetation": r"(?:tree|plant|palm|grass|forest|vegetation|flower|bush)",
            "architecture": r"(?:building|roof|wall|architect|structure|house|church|temple)",
            "language": r"(?:sign|language|text|writing|letter|character|script|word)",
            "traffic": r"(?:traffic|road|drive|car|vehicle|lane|direction|left|right)",
            "infrastructure": r"(?:pole|hydrant|bollard|sidewalk|pavement|curb|fence|lamp)",
        }
        covered = sum(
            1 for pattern in clue_categories.values()
            if re.search(pattern, prediction_text, re.IGNORECASE)
        )
        return covered / len(clue_categories)

    def reasoning_chain_score(self, prediction_text: str) -> float:
        """Check if the prediction follows a structured reasoning chain."""
        has_inference = bool(re.search(r"(?:indicates?|suggests?|points? to|implies?|shows?)", prediction_text, re.IGNORECASE))
        has_final = bool(re.search(r"(?:most likely|probably|conclude|therefore|thus)", prediction_text, re.IGNORECASE))
        has_location = bool(re.search(r"(?:taken in|located in|found in|from)", prediction_text, re.IGNORECASE))
        return sum([has_inference, has_final, has_location]) / 3.0

    def score(self, prediction_text: str, ground_truth_text: str) -> dict:
        """Compute all scores for a single prediction."""
        return {
            "semantic_similarity": round(self.semantic_score(prediction_text, ground_truth_text), 4),
            "clue_coverage": round(self.clue_coverage_score(prediction_text), 4),
            "reasoning_chain": round(self.reasoning_chain_score(prediction_text), 4),
        }


def score_directory(pred_dir: str, gt_dir: str, output_path: str = None):
    """Score all predictions against ground truth."""
    scorer = OfflineInferenceScorer()
    results = []

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".txt"):
            continue
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue

        with open(os.path.join(pred_dir, fname), "r", encoding="utf-8") as f:
            pred_text = f.read()
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()

        scores = scorer.score(pred_text, gt_text)
        scores["file"] = fname
        results.append(scores)

    if output_path and results:
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Print averages
    if results:
        avg_sem = sum(r["semantic_similarity"] for r in results) / len(results)
        avg_clue = sum(r["clue_coverage"] for r in results) / len(results)
        avg_chain = sum(r["reasoning_chain"] for r in results) / len(results)
        print(f"Averages over {len(results)} samples:")
        print(f"  Semantic similarity: {avg_sem:.4f}")
        print(f"  Clue coverage:      {avg_clue:.4f}")
        print(f"  Reasoning chain:    {avg_chain:.4f}")

    return results
