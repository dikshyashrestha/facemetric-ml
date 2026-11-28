import numpy as np
import pytest

from Common.model_evaluation import evaluate_view2, select_accuracy_threshold


def synthetic_view2(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(1, 11):
        for label in (1, 0):
            for _ in range(300):
                base = 0.7 if label else 0.1
                rows.append({"fold": fold, "label": label, "score": base + noise * rng.standard_normal()})
    return rows


def test_threshold_separates_perfectly_separable_scores():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1]
    t = select_accuracy_threshold(scores, labels)
    assert 0.3 < t <= 0.8


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        select_accuracy_threshold([], [])
    with pytest.raises(ValueError):
        select_accuracy_threshold([0.1, 0.2], [1, 1])


def test_view2_protocol_on_separable_scores():
    result = evaluate_view2(synthetic_view2())
    assert result["accuracy_mean"] == pytest.approx(1.0)
    assert result["auc"] == pytest.approx(1.0)
    assert result["eer"] == pytest.approx(0.0, abs=1e-9)
    assert len(result["folds"]) == 10


def test_view2_protocol_on_noisy_scores_is_between_chance_and_perfect():
    result = evaluate_view2(synthetic_view2(noise=0.3))
    assert 0.5 < result["accuracy_mean"] < 1.0
    assert 0.5 < result["auc"] < 1.0


def test_view2_requires_6000_pairs():
    with pytest.raises(ValueError):
        evaluate_view2(synthetic_view2()[:-1])
