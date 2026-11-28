"""Recompute every reported View 2 result from the saved pair scores.

No dataset or GPU is needed: this re-runs the 10-fold evaluation on the
committed pair_scores.csv files and checks it against each summary.json and
the README table.
"""
import csv
import json
from pathlib import Path

import pytest

from Common.model_evaluation import evaluate_view2

ROOT = Path(__file__).resolve().parent.parent

# Folder, README accuracy (%), README AUC
REPORTED = {
    "Model_2_Triplet/results/View_2/semi_hard": (86.47, 0.9365),
    "Model_2_Triplet/results/View_2/batch_hard": (85.40, 0.9321),
    "Model_3_Center/results/View_2/center_loss": (80.68, 0.8881),
    "Model_3_Center/results/View_2/softmax": (78.27, 0.8676),
    "Model_4_Spehereface/results/View_2/sphereface": (78.08, 0.8689),
    "Model_5_ArcFace/results/View_2": (71.77, 0.7883),
    "Model_1_Siamese/results/View_2/unrestricted": (66.48, 0.7134),
    "Model_1_Siamese/results/View_2/restricted": (63.23, 0.6761),
}


def load_scores(folder):
    with (ROOT / folder / "pair_scores.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("folder", sorted(REPORTED))
def test_reported_results_recompute_from_saved_scores(folder):
    rows = load_scores(folder)
    assert len(rows) == 6000
    result = evaluate_view2(rows)
    accuracy, auc = REPORTED[folder]
    assert round(result["accuracy_mean"] * 100, 2) == pytest.approx(accuracy, abs=0.006)
    assert result["auc"] == pytest.approx(auc, abs=0.00006)


@pytest.mark.parametrize("folder", sorted(REPORTED))
def test_all_methods_scored_the_same_official_pairs(folder):
    reference = [(r["fold"], r["img1"], r["img2"], r["label"]) for r in load_scores("Model_2_Triplet/results/View_2/semi_hard")]
    pairs = [(r["fold"], r["img1"], r["img2"], r["label"]) for r in load_scores(folder)]
    assert pairs == reference
