"""
Final comparison of all LFW face-verification methods (View 2).

Reads only the saved View 2 results of every model folder; nothing is trained or
re-tuned. Run from the repository root:

    python Comparison_Final/compare_methods.py

Outputs (all in Comparison_Final/):
    comparison_table.csv / .md     accuracy mean ± sample SD, AUC, EER, TAR@FAR=1%
    fold_accuracies.csv            the 10 per-fold accuracies of every method (cosine)
    statistical_tests.csv / .md    paired tests over the 10 folds (with caveats)
    training_cost.csv / .md        training time, epochs, memory, hardware
    view1_ablations.csv / .md      View 1 hyperparameter sweeps as recorded
    roc_combined.png / .pdf        ROC curves, log FAR axis, restricted Siamese separate
    verification_report.md         every consistency check and its result

Every check that fails is reported, never silently corrected.
"""

import csv
import json
import sys
from pathlib import Path

# torch before scikit-learn / scipy: avoids an OpenMP clash in Anaconda on macOS.
import torch  # noqa: F401

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Common.model_evaluation import evaluate_view2  # noqa: E402

OUT_DIR = ROOT / "Comparison_Final"
MANIFEST = ROOT / "Dataset" / "Manifests" / "view2_pairs.csv"
VIEW2_IMAGES = ROOT / "Dataset" / "Manifests" / "view2_images.csv"
HELD_OUT_SCHEME = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
TOLERANCE = 1e-9


# ============================================================
# Methods (paths and how each team member recorded its View 1 selection)
# ============================================================

METHODS = [
    {"key": "softmax", "label": "Softmax baseline", "protocol": "Unrestricted",
     "dir": "Model_3_Center/results/View_2/softmax", "family": "classification",
     "tuning": "Model_3_Center/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "softmax"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["final_hyperparameters"],
     "selection_fields": {"lambda": "lambda", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: "λ=0 (cross-entropy only)"},
    {"key": "center_loss", "label": "Softmax + Center Loss", "protocol": "Unrestricted",
     "dir": "Model_3_Center/results/View_2/center_loss", "family": "classification",
     "tuning": "Model_3_Center/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "center_loss"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["final_hyperparameters"],
     "selection_fields": {"lambda": "lambda", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"λ={s['final_hyperparameters']['lambda']}, α={s['final_hyperparameters']['center_alpha']}"},
    {"key": "siamese_restricted", "label": "Siamese (restricted)", "protocol": "Image-restricted",
     "dir": "Model_1_Siamese/results/View_2/restricted", "family": "restricted_pairs",
     "tuning": "Model_1_Siamese/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "restricted"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["final_hyperparameters"],
     "selection_fields": {"margin": "margin", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"margin={s['final_hyperparameters']['margin']}", "has_l2": True},
    {"key": "siamese_unrestricted", "label": "Siamese (unrestricted)", "protocol": "Unrestricted",
     "dir": "Model_1_Siamese/results/View_2/unrestricted", "family": "identity_pairs",
     "tuning": "Model_1_Siamese/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "unrestricted"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["final_hyperparameters"],
     "selection_fields": {"margin": "margin", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"margin={s['final_hyperparameters']['margin']}", "has_l2": True},
    {"key": "triplet_semi_hard", "label": "Triplet (semi-hard)", "protocol": "Unrestricted",
     "dir": "Model_2_Triplet/results/View_2/semi_hard", "family": "identity_all",
     "tuning": "Model_2_Triplet/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "semi_hard"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["frozen_hyperparameters"],
     "selection_fields": {"margin": "margin", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"α={s['frozen_hyperparameters']['margin']}, P×K=8×4", "has_l2": True},
    {"key": "triplet_batch_hard", "label": "Triplet (batch-hard)", "protocol": "Unrestricted",
     "dir": "Model_2_Triplet/results/View_2/batch_hard", "family": "identity_all",
     "tuning": "Model_2_Triplet/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "batch_hard"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["frozen_hyperparameters"],
     "selection_fields": {"margin": "margin", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"α={s['frozen_hyperparameters']['margin']}, P×K=8×4", "has_l2": True},
    {"key": "sphereface", "label": "SphereFace", "protocol": "Unrestricted",
     "dir": "Model_4_Spehereface/results/View_2/sphereface", "family": "classification",
     "tuning": "Model_4_Spehereface/results/View_1/tuning_summary.csv",
     "tuning_filter": {"variant": "sphereface"}, "rank_column": "rank_within_variant",
     "selected": lambda s: s["final_hyperparameters"],
     "selection_fields": {"margin": "margin", "learning_rate": "learning_rate", "epochs": "epochs"},
     "setting": lambda s: f"m={s['final_hyperparameters']['margin']}, λ annealed to 5"},
    {"key": "arcface", "label": "ArcFace", "protocol": "Unrestricted",
     "dir": "Model_5_ArcFace/results/View_2", "family": "classification",
     "tuning": "Model_5_ArcFace/results/View_1/tuning_summary.csv",
     "tuning_filter": {"phase": "grid"}, "rank_column": "rank_within_grid",
     "selected": lambda s: s["selected_from_view1"],
     "selection_fields": {"scale": "scale", "margin": "margin", "learning_rate": "learning_rate",
                          "epochs": "epochs"},
     "setting": lambda s: f"s={s['selected_from_view1']['scale']:g}, m={s['selected_from_view1']['margin']}"},
]

# Categorical palette (validated: adjacent CVD ΔE ≥ 9, normal-vision ΔE ≥ 19).
# Colour follows the method; line style is a second, non-colour cue.
STYLE = {
    "softmax": ("#2a78d6", (0, ())),
    "center_loss": ("#eb6834", (0, (6, 2))),
    "siamese_unrestricted": ("#1baf7a", (0, (2, 2))),
    "triplet_semi_hard": ("#eda100", (0, ())),
    "triplet_batch_hard": ("#e87ba4", (0, (6, 2))),
    "sphereface": ("#008300", (0, (1, 1.5))),
    "arcface": ("#6250d6", (0, (6, 2, 2, 2))),
    "siamese_restricted": ("#e34948", (0, ())),
}


# ============================================================
# Small helpers
# ============================================================

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def read_scores(path, extra_column=None):
    rows = read_csv(path)
    out = []
    for r in rows:
        row = {"fold": int(r["fold"]), "img1": r["img1"], "img2": r["img2"],
               "label": int(r["label"]), "score": float(r["score"])}
        if extra_column and extra_column in r:
            row[extra_column] = float(r[extra_column])
        out.append(row)
    return out, list(rows[0].keys())


def independent_thresholds(scores, labels, folds):
    """Re-derive each fold's accuracy-maximizing threshold from the other nine folds,
    without the shared helper. Ties go to the highest threshold, as in Common."""
    result = []
    for fold in range(1, 11):
        test = folds == fold
        s, y = scores[~test], labels[~test]
        candidates = np.unique(s)[::-1]
        # accuracy of "same if score >= t" for every candidate t, via cumulative counts
        order = np.argsort(-s, kind="stable")
        s_sorted, y_sorted = s[order], y[order]
        positives_above = np.cumsum(y_sorted)
        negatives_above = np.cumsum(1 - y_sorted)
        last_index = np.searchsorted(-s_sorted, -candidates, side="right") - 1
        tp = positives_above[last_index]
        fp = negatives_above[last_index]
        accuracy = (tp + ((1 - y).sum() - fp)) / len(y)
        all_different = (1 - y).sum() / len(y)          # threshold above every score
        best = int(np.argmax(accuracy))
        threshold = (float(np.nextafter(s.max(), np.inf)) if all_different > accuracy[best]
                     else float(candidates[best]))
        result.append({"fold": fold, "threshold": threshold,
                       "accuracy": float(np.mean((scores[test] >= threshold) == labels[test]))})
    return result


# ============================================================
# Verification of one method
# ============================================================

def verify_method(method, manifest_rows, view2_images):
    folder = ROOT / method["dir"]
    summary = json.loads((folder / "summary.json").read_text())
    checks = []

    def check(name, ok, detail=""):
        checks.append({"method": method["label"], "check": name,
                       "result": "PASS" if ok else "FAIL", "detail": detail})
        return ok

    # --- pair-level submission file ---
    rows, columns = read_scores(folder / "pair_scores.csv")
    check("pair_scores.csv has exactly fold,img1,img2,label,score",
          columns == ["fold", "img1", "img2", "label", "score"], str(columns))
    check("6,000 scored pairs", len(rows) == 6000, str(len(rows)))
    same_order = [(r["fold"], r["img1"], r["img2"], r["label"]) for r in rows] == manifest_rows
    check("pairs, labels and folds identical to official View 2 list (same order)", same_order)
    folds = np.array([r["fold"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    balanced = all((folds == f).sum() == 600 and labels[folds == f].sum() == 300
                   for f in range(1, 11))
    check("each fold: 600 pairs, 300 genuine + 300 impostor", balanced)

    # --- fold assignment: five runs, each scoring exactly its two held-out folds ---
    fold_to_run = {int(k): v for k, v in summary["fold_to_run"].items()}
    expected = {f: j for j, pair in enumerate(HELD_OUT_SCHEME, 1) for f in pair}
    check("summary fold_to_run = (1,2),(3,4),(5,6),(7,8),(9,10) scheme",
          fold_to_run == expected, str(fold_to_run))
    run_rows = []
    for run_number, held_out in enumerate(HELD_OUT_SCHEME, 1):
        run_dir = folder / f"run_{run_number}"
        config = json.loads((run_dir / "config.json").read_text())
        complete = json.loads((run_dir / "run_complete.json").read_text())
        scored, _ = read_scores(run_dir / "heldout_scores.csv")
        ok = (list(config["held_out_folds"]) == list(held_out)
              and list(complete["held_out_folds"]) == list(held_out)
              and {r["fold"] for r in scored} == set(held_out) and len(scored) == 1200)
        check(f"run_{run_number} trained without and scored only folds {held_out}", ok)
        run_rows += scored
    check("five runs' held-out scores reproduce pair_scores.csv exactly",
          [(r["fold"], r["img1"], r["img2"], r["label"], r["score"]) for r in run_rows] ==
          [(r["fold"], r["img1"], r["img2"], r["label"], r["score"]) for r in rows])

    # --- training data isolation, re-derived from the image manifest ---
    for run_number, held_out in enumerate(HELD_OUT_SCHEME, 1):
        data = json.loads((folder / f"run_{run_number}" / "run_complete.json").read_text())["training_data"]
        overlap = data.get("evaluation_identity_overlap", data.get("held_out_identity_overlap"))
        train_images = [r for r in view2_images if int(r["fold"]) not in held_out]
        counts = {}
        for r in train_images:
            counts[r["identity"]] = counts.get(r["identity"], 0) + 1
        if method["family"] == "classification":
            expected_n = (sum(1 for c in counts.values() if c >= 2),
                          sum(c for c in counts.values() if c >= 2))
            recorded = (data.get("classes", data.get("eligible_identity_classes")),
                        data.get("training_images", data.get("eligible_training_images")))
            what = "classes (identities ≥2 images) and images"
        elif method["family"] == "restricted_pairs":
            expected_n, recorded = (4800, 2400), (data["training_pairs"], data["genuine_pairs"])
            what = "official training pairs and genuine pairs"
        else:
            expected_n, recorded = (len(counts), len(train_images)), (
                data["training_identities"], data["training_images"])
            what = "identities and images"
        check(f"run_{run_number} training {what} match the 8 training folds; recorded overlap 0",
              overlap == 0 and tuple(recorded) == tuple(expected_n),
              f"recorded {recorded}, expected {expected_n}, overlap {overlap}")

    # --- metrics recomputed with the shared evaluation code ---
    recomputed = evaluate_view2(rows, score_key="score", higher_is_same=True)
    saved = summary["cosine"]
    metric_keys = ["accuracy_mean", "accuracy_std", "auc", "eer", "tar_at_far_0.01"]
    diffs = {k: abs(recomputed[k] - saved[k]) for k in metric_keys}
    check("summary.json cosine metrics reproduced from pair_scores.csv",
          max(diffs.values()) < TOLERANCE, json.dumps({k: f"{v:.1e}" for k, v in diffs.items()}))

    fold_file = read_csv(folder / "fold_accuracy.csv")
    ok = len(fold_file) == 10 and all(
        int(f["fold"]) == r["fold"]
        and abs(float(f["threshold"]) - r["threshold"]) < TOLERANCE
        and abs(float(f["accuracy"]) - r["accuracy"]) < TOLERANCE
        and int(f["correct"]) == round(r["accuracy"] * 600) and int(f["n_pairs"]) == 600
        for f, r in zip(fold_file, recomputed["folds"]))
    check("fold_accuracy.csv thresholds/accuracies match the recomputation", ok)

    independent = independent_thresholds(scores, labels, folds)
    ok = all(abs(a["threshold"] - b["threshold"]) < TOLERANCE and abs(a["accuracy"] - b["accuracy"]) < TOLERANCE
             for a, b in zip(independent, recomputed["folds"]))
    check("thresholds independently re-derived from the other nine folds (5,400 pairs)", ok)

    flipped = labels.copy()
    flipped[folds == 1] = 1 - flipped[folds == 1]
    own_labels_unused = independent_thresholds(scores, flipped, folds)[0]["threshold"] == independent[0]["threshold"]
    check("a fold's threshold does not depend on its own labels (fold 1 labels flipped)",
          own_labels_unused)

    genuine_higher = scores[labels == 1].mean() > scores[labels == 0].mean()
    check("score direction: cosine, higher = same person (genuine mean > impostor mean)",
          genuine_higher and recomputed["auc"] > 0.5 and scores.min() >= -1 - 1e-6 and scores.max() <= 1 + 1e-6,
          f"genuine {scores[labels == 1].mean():.3f}, impostor {scores[labels == 0].mean():.3f}")

    # --- squared Euclidean distance (Siamese and Triplet) ---
    l2_result = None
    if method.get("has_l2"):
        l2_rows, l2_columns = read_scores(folder / "pair_scores_squared_l2.csv")
        check("pair_scores_squared_l2.csv: same pairs, same five columns",
              l2_columns == columns and [(r["fold"], r["img1"], r["img2"], r["label"]) for r in l2_rows]
              == [(r["fold"], r["img1"], r["img2"], r["label"]) for r in rows])
        distances = np.array([r["score"] for r in l2_rows])
        max_diff = float(np.max(np.abs(distances - (2 - 2 * scores))))
        check("squared L2 = 2 − 2·cosine for unit vectors", max_diff < 1e-5, f"max |diff| {max_diff:.1e}")
        check("score direction: squared L2, lower = same person",
              distances[labels == 1].mean() < distances[labels == 0].mean())
        l2_input = [dict(r, squared_l2=d) for r, d in zip(rows, distances)]
        l2_result = evaluate_view2(l2_input, score_key="squared_l2", higher_is_same=False)
        diffs = {k: abs(l2_result[k] - summary["squared_l2"][k]) for k in metric_keys}
        check("summary.json squared-L2 metrics reproduced", max(diffs.values()) < TOLERANCE,
              json.dumps({k: f"{v:.1e}" for k, v in diffs.items()}))
        l2_fold_file = read_csv(folder / "fold_accuracy_squared_l2.csv")
        check("fold_accuracy_squared_l2.csv matches the recomputation",
              all(abs(float(f["accuracy"]) - r["accuracy"]) < TOLERANCE
                  and abs(float(f["threshold"]) - r["threshold"]) < 1e-6
                  for f, r in zip(l2_fold_file, l2_result["folds"])))

    # --- View 1 selection as recorded (no re-selection here) ---
    tuning = [r for r in read_csv(ROOT / method["tuning"])
              if all(r[k] == v for k, v in method["tuning_filter"].items())]
    rank_1 = [r for r in tuning if r[method["rank_column"]] == "1"]
    selected = method["selected"](summary)
    ok = len(rank_1) == 1 and all(
        abs(float(rank_1[0][tuning_key]) - float(selected[selected_key])) < 1e-12
        for tuning_key, selected_key in method["selection_fields"].items())
    check("View 2 hyperparameters = rank-1 row of the View 1 tuning summary", ok,
          f"View 1 rank 1: {({k: rank_1[0][k] for k in method['selection_fields']} if rank_1 else None)}")

    run_epochs = [json.loads((folder / f"run_{j}" / "run_complete.json").read_text())["epochs_completed"]
                  for j in range(1, 6)]
    check("every View 2 run trained for the View 1-selected number of epochs",
          all(e == int(float(selected["epochs"])) for e in run_epochs), str(run_epochs))

    return {"summary": summary, "rows": rows, "scores": scores, "labels": labels,
            "folds": folds, "cosine": recomputed, "squared_l2": l2_result, "checks": checks}


# ============================================================
# Outputs
# ============================================================

def comparison_table(methods, results):
    rows = []
    for m in methods:
        r = results[m["key"]]
        for score_name, metrics in [("cos", r["cosine"]), ("L2²", r["squared_l2"])]:
            if metrics is None:
                continue
            rows.append({
                "method": m["label"], "protocol": m["protocol"], "score": score_name,
                "selected_setting": m["setting"](r["summary"]),
                "accuracy_mean": round(metrics["accuracy_mean"], 4),
                "accuracy_sd": round(metrics["accuracy_std"], 4),
                "accuracy": f"{metrics['accuracy_mean'] * 100:.2f} ± {metrics['accuracy_std'] * 100:.2f}",
                "auc": round(metrics["auc"], 4), "eer": round(metrics["eer"], 4),
                "tar_at_far_1pct": round(metrics["tar_at_far_0.01"], 4),
            })
    return rows


def fold_accuracy_table(methods, results):
    return [{"fold": f, **{m["key"]: round(results[m["key"]]["cosine"]["folds"][f - 1]["accuracy"], 6)
                           for m in methods}} for f in range(1, 11)]


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def statistical_tests(methods, results):
    accuracy = {m["key"]: np.array([f["accuracy"] for f in results[m["key"]]["cosine"]["folds"]])
                for m in methods}
    label = {m["key"]: m["label"] for m in methods}
    best = max(accuracy, key=lambda k: accuracy[k].mean())

    def test(a, b, family):
        difference = accuracy[a] - accuracy[b]
        return {"family": family, "method_a": label[a], "method_b": label[b],
                "mean_difference_pct_points": round(difference.mean() * 100, 2),
                "sd_difference_pct_points": round(difference.std(ddof=1) * 100, 2),
                "folds_a_better": int((difference > 0).sum()),
                "paired_t_p": float(stats.ttest_rel(accuracy[a], accuracy[b]).pvalue),
                "wilcoxon_p": float(stats.wilcoxon(accuracy[a], accuracy[b]).pvalue)
                if np.any(difference != 0) else 1.0}

    families = [
        ("best vs every other method", [(best, k) for k in accuracy if k != best]),
        ("planned ablation pairs", [("siamese_unrestricted", "siamese_restricted"),
                                    ("triplet_semi_hard", "triplet_batch_hard"),
                                    ("center_loss", "softmax")]),
    ]
    rows = []
    for family, pairs in families:
        block = [test(a, b, family) for a, b in pairs]
        for key in ("paired_t_p", "wilcoxon_p"):
            for row, adjusted in zip(block, holm(np.array([r[key] for r in block]))):
                row[key + "_holm"] = float(adjusted)
        rows += block
    for row in rows:
        for key in list(row):
            if key.endswith("_p") or key.endswith("_holm"):
                row[key] = float(f"{row[key]:.3g}")
    return rows, label[best]


def training_cost(methods, results):
    rows = []
    for m in methods:
        folder = ROOT / m["dir"]
        summary = results[m["key"]]["summary"]
        runs = [json.loads((folder / f"run_{j}" / "run_complete.json").read_text()) for j in range(1, 6)]
        seconds = [r["training_seconds"] for r in runs]
        epochs = [r["epochs_completed"] for r in runs]
        hardware = summary["hardware"]
        mps = [r.get("peak_sampled_mps_driver_mib") for r in runs if r.get("peak_sampled_mps_driver_mib")]
        cuda = [r.get("peak_gpu_allocated_mib") for r in runs if r.get("peak_gpu_allocated_mib")]
        if cuda:
            memory, memory_kind = max(cuda), "CUDA max_memory_allocated (true peak of tensors)"
        elif mps:
            memory, memory_kind = max(mps), "MPS driver memory, max of per-batch samples (not a true peak)"
        else:
            memory, memory_kind = "not recorded", ""
        rows.append({
            "method": m["label"],
            "accelerator": ("Apple Silicon GPU (MPS)" if hardware["device"] == "mps"
                            else f"{hardware['accelerator_name']} ({hardware['device'].upper()})"),
            "platform": hardware["python_platform"],
            "pytorch": hardware["pytorch_version"],
            "runs": len(runs),
            "epochs_per_run": "/".join(str(e) for e in sorted(set(epochs))),
            "total_training_hours": round(sum(seconds) / 3600, 2),
            "mean_hours_per_run": round(np.mean(seconds) / 3600, 2),
            "mean_minutes_per_epoch": round(np.mean([s / e for s, e in zip(seconds, epochs)]) / 60, 2),
            "peak_memory_mib": memory if isinstance(memory, str) else round(memory),
            "memory_measurement": memory_kind,
        })
    return rows


def view1_ablations(methods):
    rows = []
    seen = set()
    for m in methods:
        if m["tuning"] in seen:
            continue
        seen.add(m["tuning"])
        for r in read_csv(ROOT / m["tuning"]):
            setting = {k: r[k] for k in ("variant", "phase", "lambda", "margin", "scale", "annealing", "learning_rate")
                       if k in r and r[k] != ""}
            group = setting.pop("variant", setting.pop("phase", ""))
            rank = r.get("rank_within_variant") or r.get("rank_within_grid") or ""
            rows.append({
                "model_folder": m["tuning"].split("/")[0], "group": group,
                "setting": ", ".join(f"{k}={v}" for k, v in setting.items()),
                "epochs": r["epochs"],
                "view1_final_accuracy": round(float(r["final_validation_accuracy"]), 4),
                "view1_final_auc": round(float(r["final_validation_auc"]), 4),
                "rank": rank,
            })
    return rows


def plot_roc(methods, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panels = [
        (axes[0], [m for m in methods if m["protocol"] == "Unrestricted"],
         "(a) Unrestricted protocol (identity labels of the training folds)"),
        (axes[1], [m for m in methods if m["key"] in ("siamese_restricted", "siamese_unrestricted")],
         "(b) Image-restricted Siamese (official pairs only),\nunrestricted Siamese for reference"),
    ]
    for ax, members, title in panels:
        members = sorted(members, key=lambda m: -results[m["key"]]["cosine"]["auc"])
        for m in members:
            r = results[m["key"]]
            far, tar, _ = roc_curve(r["labels"], r["scores"])
            color, dashes = STYLE[m["key"]]
            ax.plot(far, tar, color=color, linestyle=dashes, linewidth=2,
                    label=f"{m['label']}  AUC {r['cosine']['auc']:.3f}")
        floor = np.logspace(np.log10(1 / 3000), 0, 200)
        ax.plot(floor, floor, color=muted, linewidth=1, linestyle=(0, (1, 2)), label="chance")
        ax.axvline(0.01, color=muted, linewidth=1)
        ax.set(xscale="log", xlim=(1 / 3000, 1), ylim=(0, 1),
               xlabel="False accept rate (log scale; 3,000 impostor pairs; vertical line = 1%)",
               title=title)
        ax.grid(color=grid, linewidth=0.6)
        ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.9, handlelength=3.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("True accept rate")
    fig.suptitle("LFW View 2 ROC, cosine similarity on out-of-fold scores (5 runs × 2 held-out folds)",
                 fontsize=10, color=ink)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"roc_combined.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    manifest_rows = [(int(r["fold"]), r["img1"], r["img2"], int(r["label"])) for r in read_csv(MANIFEST)]
    view2_images = read_csv(VIEW2_IMAGES)

    results = {m["key"]: verify_method(m, manifest_rows, view2_images) for m in METHODS}
    all_checks = [c for m in METHODS for c in results[m["key"]]["checks"]]
    failures = [c for c in all_checks if c["result"] == "FAIL"]

    table = comparison_table(METHODS, results)
    write_csv(OUT_DIR / "comparison_table.csv", table)
    (OUT_DIR / "comparison_table.md").write_text(
        "# LFW View 2 verification results\n\n"
        "Accuracy = mean ± sample SD (ddof=1) over the 10 View 2 folds, in %. "
        "The SD measures fold-to-fold variation of one seed (42); it is not seed variance. "
        "AUC, EER and TAR@FAR=1% are computed from all 6,000 out-of-fold pair scores. "
        "Every method uses the reduced five-run scheme (PDF §4.2).\n\n"
        + markdown_table(table, ["method", "protocol", "score", "selected_setting", "accuracy",
                                 "auc", "eer", "tar_at_far_1pct"]) + "\n", encoding="utf-8")

    write_csv(OUT_DIR / "fold_accuracies.csv", fold_accuracy_table(METHODS, results))

    tests, best = statistical_tests(METHODS, results)
    write_csv(OUT_DIR / "statistical_tests.csv", tests)
    (OUT_DIR / "statistical_tests.md").write_text(
        f"# Paired tests over the 10 View 2 folds (cosine accuracy)\n\n"
        f"Best mean accuracy: **{best}**. Holm correction is applied within each family. "
        "Caveats: (1) the two folds scored by the same run share a model, so the 10 differences are "
        "not fully independent and p-values are likely optimistic; (2) with n = 10 the smallest "
        "two-sided Wilcoxon p is ≈ 0.002; (3) restricted vs unrestricted Siamese also differ in "
        "protocol, by design.\n\n"
        + markdown_table(tests, ["family", "method_a", "method_b", "mean_difference_pct_points",
                                 "folds_a_better", "paired_t_p", "paired_t_p_holm", "wilcoxon_p",
                                 "wilcoxon_p_holm"]) + "\n", encoding="utf-8")

    cost = training_cost(METHODS, results)
    write_csv(OUT_DIR / "training_cost.csv", cost)
    (OUT_DIR / "training_cost.md").write_text(
        "# Training cost (View 2, five runs per method)\n\n"
        "Times are wall-clock training time summed over epochs (held-out scoring excluded), as "
        "recorded by each run. **They are not directly comparable across methods:** runs used "
        "different machines (Siamese: Apple M1, 8 GB, macOS 15.5; Triplet, Center and SphereFace: "
        "Apple Silicon Macs on macOS 26.x, whose chip model was not recorded; ArcFace: a CUDA "
        "Tesla T4), different PyTorch versions, and "
        "different amounts of work per epoch (4,800 pairs = 9,600 images for Siamese; 300 × 36 images "
        "for Triplet; one pass over ~7,000–7,800 images in batches of 64 for Softmax / Center / "
        "SphereFace; 300 × 32 images for ArcFace). MPS memory is the largest of per-batch samples of "
        "driver-allocated memory (includes allocator cache, not a true peak); CUDA memory is PyTorch's "
        "peak allocated tensor memory. The two are different quantities.\n\n"
        + markdown_table(cost, ["method", "accelerator", "platform", "pytorch", "epochs_per_run",
                                "total_training_hours", "mean_hours_per_run", "mean_minutes_per_epoch",
                                "peak_memory_mib", "memory_measurement"]) + "\n", encoding="utf-8")

    ablations = view1_ablations(METHODS)
    write_csv(OUT_DIR / "view1_ablations.csv", ablations)
    (OUT_DIR / "view1_ablations.md").write_text(
        "# View 1 hyperparameter sweeps (as recorded by each model owner)\n\n"
        "Validation accuracy uses the best threshold on the 1,000 pairsDevTest pairs: a tuning metric, "
        "optimistic and not comparable to View 2 accuracy. Rank 1 is the configuration used on View 2.\n\n"
        + markdown_table(ablations, ["model_folder", "group", "setting", "epochs",
                                     "view1_final_accuracy", "view1_final_auc", "rank"]) + "\n",
        encoding="utf-8")

    plot_roc(METHODS, results)

    lines = ["# Verification report", "",
             f"{len(all_checks)} checks, {len(failures)} failed. Generated by compare_methods.py from "
             "saved results only.", ""]
    for m in METHODS:
        lines += [f"## {m['label']}  (`{m['dir']}`)", "",
                  markdown_table(results[m["key"]]["checks"], ["check", "result", "detail"]), ""]
    (OUT_DIR / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(markdown_table(table, ["method", "protocol", "score", "accuracy", "auc", "eer", "tar_at_far_1pct"]))
    print(f"\nChecks: {len(all_checks)}, failed: {len(failures)}")
    for c in failures:
        print("  FAIL", c["method"], "|", c["check"], "|", c["detail"])


if __name__ == "__main__":
    main()
