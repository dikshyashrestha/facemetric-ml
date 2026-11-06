"""
Train and evaluate the softmax and center-loss models on LFW.

The softmax variant uses cross-entropy alone. The center-loss variant adds
lambda * (1/2) * ||f(x) - c_y||^2, following Wen et al. (2016). Set STAGE
below to run a smoke test, tune settings on View 1, or evaluate on View 2.

Run from the repository root with:
    python -m Model_3_Center.center_loss

Results are saved each epoch. Interrupted runs resume from their last checkpoint,
and completed runs are skipped when restarted.
"""

import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from Common.base_model import FaceEmbeddingNet
from Common.dataset_loader import LFWIdentityDataset, LFWPairDataset, read_manifest
from Common.model_evaluation import (
    evaluate_view2,
    save_pair_scores,
    save_roc_plot,
    score_pair_dataset,
    select_accuracy_threshold,
)
from Common.model_utils import (
    TrainingMonitor,
    count_trainable_parameters,
    get_device,
    hardware_info,
    save_json,
    seed_everything,
)


# Training configuration

STAGE = "view2_final"          # Choose "smoke_test", "view1_tuning", or "view2_final".
VARIANTS = ["softmax", "center_loss"]

SEED = 42                     # Shared by each run for reproducibility.
EPOCHS = 30                   # Epochs per run.
BATCH_SIZE = 64               # Number of images in each training batch.
NUM_WORKERS = 0
MIN_IMAGES_PER_CLASS = 2      # Ignore identities with fewer than two training images.

# View 1 search space
LAMBDAS = {
    "softmax": [0.0],
    "center_loss": [0.001, 0.003, 0.01, 0.03],
}
CENTER_ALPHA = 0.5            # Center update rate from the paper.
LEARNING_RATES = [1e-4]       # Shared learning rate for all variants.
WEIGHT_DECAY = 5e-4           # AdamW weight decay.
ADAMW_BETAS = (0.9, 0.999)

# Best View 1 settings for each variant.
FINAL_CONFIG = {
    "softmax": {"lambda": 0.0, "learning_rate": 1e-4, "epochs": 30},
    "center_loss": {"lambda": 0.01, "learning_rate": 1e-4, "epochs": 30},
}

VIEW2_HELD_OUT = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# Smoke-test settings
SMOKE_BATCHES = 3
SMOKE_EPOCHS = 2
SMOKE_VAL_PAIRS = 64

MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODEL_DIR / "results"
SMOKE_DIR = MODEL_DIR / "outputs" / "smoke_test"      # Gitignored.
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"            # Gitignored.


# Center loss

class CenterLoss(nn.Module):
    """Compute the mean center loss, (1/2) * ||f(x) - c_y||^2.

    Centers are stored as a buffer and updated separately after each batch.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.register_buffer("centers", torch.zeros(num_classes, 512))

    def forward(self, features, labels):
        diff = features - self.centers[labels]
        return (diff ** 2).sum(dim=1).mean() / 2

    @torch.no_grad()
    def update_centers(self, features, labels):
        # Move each center toward the features assigned to its class.
        for label in labels.unique():
            mask = labels == label
            delta = (self.centers[label] - features[mask]).sum(dim=0) / (1 + mask.sum())
            self.centers[label] -= CENTER_ALPHA * delta


def mean_inter_center_distance(centers):
    centers = centers.cpu()
    n = len(centers)
    return (torch.cdist(centers, centers).sum() / (n * (n - 1))).item()


# Training data

def image_identity(image_path):
    return image_path.split("/")[0]


def load_training_data(stage, held_out):
    """Load eligible identities and remap their labels to contiguous class indices."""
    if stage == "view1":
        dataset = LFWIdentityDataset(split="view1_train", train=True)
    else:
        dataset = LFWIdentityDataset(split="view2", held_out_folds=list(held_out), train=True)

    kept = [label for label, indices in dataset.indices_by_identity.items()
            if len(indices) >= MIN_IMAGES_PER_CLASS]
    new_label = {old: new for new, old in enumerate(sorted(kept))}

    samples, labels = [], []
    for row, label in zip(dataset.samples, dataset.labels):
        if label in new_label:
            samples.append(row)
            labels.append(new_label[label])

    dataset.samples = samples
    dataset.labels = labels
    return dataset, len(kept)


def check_training_data(dataset, stage, held_out):
    """Check that no evaluation identity appears in the training split."""
    train_ids = {row["identity"] for row in dataset.samples}

    if stage == "view1":
        evaluation_ids = {
            image_identity(p)
            for row in read_manifest("view1_validation_pairs.csv")
            for p in (row["img1"], row["img2"])
        }
    else:
        evaluation_ids = {
            row["identity"] for row in read_manifest("view2_images.csv")
            if int(row["fold"]) in held_out
        }
        if any(int(row["fold"]) in held_out for row in dataset.samples):
            raise RuntimeError("LEAKAGE: training image from a held-out fold")

    overlap = train_ids & evaluation_ids
    if overlap:
        raise RuntimeError(
            f"LEAKAGE: {len(overlap)} evaluation identities in training data, "
            f"e.g. {sorted(overlap)[:5]}"
        )

    return {
        "classes": len(train_ids),
        "training_images": len(dataset.samples),
        "min_images_per_class": MIN_IMAGES_PER_CLASS,
        "evaluation_identity_overlap": 0,
    }


# Output helpers

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_scores_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return [
            {"fold": int(r["fold"]), "img1": r["img1"], "img2": r["img2"],
             "label": int(r["label"]), "score": float(r["score"]),
             "squared_l2": float(r["squared_l2"])}
            for r in csv.DictReader(file)
        ]


def software_versions():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=MODEL_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(git("status", "--porcelain")),
    }


def checkpoint_path(run_dir):
    return CHECKPOINT_DIR / Path(run_dir).relative_to(MODEL_DIR) / "last_epoch.pt"


def plot_training_curves(run_dir, history, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="total")
    axes[0].plot(epochs, [h["classification_loss"] for h in history], label="softmax (cross-entropy)")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Training loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["mean_intra_class_distance"] for h in history], label="feature to own center")
    axes[1].plot(epochs, [h["mean_inter_center_distance"] for h in history], label="between centers")
    axes[1].set(xlabel="epoch", ylabel="mean Euclidean distance", title="Compactness")
    axes[1].legend()

    if history[0]["validation_accuracy"] != "":
        axes[2].plot(epochs, [h["validation_accuracy"] for h in history], label="accuracy")
        axes[2].plot(epochs, [h["validation_auc"] for h in history], label="AUC")
        axes[2].set(xlabel="epoch", title="View 1 validation (pairsDevTest)")
        axes[2].legend()
    else:
        axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "training_curves.png", dpi=120)
    plt.close(fig)


# One training run

def train_one_run(run_dir, stage, settings, held_out=(), smoke=False):
    """Train one configuration and save its results."""
    run_dir = Path(run_dir)
    complete_file = run_dir / "run_complete.json"
    if complete_file.is_file():
        print(f"Already complete, skipping: {run_dir}")
        return json.loads(complete_file.read_text())

    device = get_device()
    seed_everything(SEED)
    held_out = tuple(held_out)
    variant = settings["variant"]
    lam = settings["lambda"]
    epochs = settings["epochs"]
    run_number = VIEW2_HELD_OUT.index(held_out) + 1 if held_out else None

    train_dataset, num_classes = load_training_data(stage, held_out)
    data_info = check_training_data(train_dataset, stage, held_out)

    validation_dataset = None
    if stage == "view1":
        validation_dataset = LFWPairDataset(split="view1_validation", train=False)
        if smoke:
            half = SMOKE_VAL_PAIRS // 2
            validation_dataset.samples = (validation_dataset.samples[:half] +
                                          validation_dataset.samples[-half:])

    model = FaceEmbeddingNet().to(device)
    classifier = nn.Linear(512, num_classes).to(device)
    center_loss = CenterLoss(num_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                                  lr=settings["learning_rate"], betas=ADAMW_BETAS,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    key = {**settings, "stage": stage, "held_out_folds": list(held_out), "seed": SEED,
           "batch_size": BATCH_SIZE, "center_alpha": CENTER_ALPHA, "smoke_test": smoke}

    config = {
        "model": "Softmax + Center Loss" if lam > 0 else "Softmax baseline (lambda = 0)",
        "variant": variant,
        "lfw_protocol": "unrestricted, labeled outside data",
        "stage": "View 1" if stage == "view1" else "View 2",
        "run_number": run_number,
        "held_out_folds": list(held_out),
        "fold_scheme": (None if stage == "view1" else
                        "reduced five-run scheme (PDF 4.2): run j holds out folds 2j-1 and 2j"),
        "seed": SEED,
        "backbone": "Common.base_model.FaceEmbeddingNet (modified ResNet-18, random init)",
        "embedding_dim": 512,
        "trainable_parameters": (count_trainable_parameters(model) +
                                 count_trainable_parameters(classifier)),
        "loss": {
            "name": "softmax cross-entropy + center loss (Wen et al., 2016)",
            "formula": "CE + lambda * (1/2) * ||f(x) - c_y||^2, mean over the batch",
            "lambda": lam,
            "center_update": "c_j -= alpha * sum(c_j - x_i) / (1 + n_j), after every batch, "
                             "not by the optimizer",
            "center_alpha": CENTER_ALPHA,
            "centers_init": "zeros",
            "classifier": "Linear(512, classes) with bias, discarded at test time",
        },
        "optimizer": {"name": "AdamW", "learning_rate": settings["learning_rate"],
                      "weight_decay": WEIGHT_DECAY, "betas": list(ADAMW_BETAS)},
        "lr_schedule": {"name": "CosineAnnealingLR", "T_max_epochs": epochs,
                        "eta_min": 0.0, "step": "once per epoch"},
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": len(train_dataset) // BATCH_SIZE,
        "max_epochs": epochs,
        "sampling": "uniform shuffle of all training images each epoch, last partial batch dropped",
        "training_data": data_info,
        "augmentation": "random horizontal flip p=0.5",
        "normalization": "RGB 112x112, (pixel - 127.5) / 128",
        "test_time_features": "original + flipped embeddings summed, then L2-normalized",
        "validation": (None if stage == "view2" else
                       "View 1 pairsDevTest (1,000 pairs): accuracy at the best threshold "
                       "on those pairs (tuning metric, optimistic) and ROC-AUC"),
        "hardware": hardware_info(device),
        "software": software_versions(),
        "settings_key": key,
    }
    if smoke:
        config["smoke_test"] = {"batches_per_epoch": SMOKE_BATCHES,
                                "validation_pairs": SMOKE_VAL_PAIRS}

    history = []
    ckpt_file = checkpoint_path(run_dir)

    if ckpt_file.is_file():
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if checkpoint["settings"] != key:
            raise RuntimeError(
                f"Settings changed since this run started.\nSaved: {checkpoint['settings']}\n"
                f"Now:   {key}\nRestore the settings or delete {ckpt_file}"
            )
        model.load_state_dict(checkpoint["model"])
        classifier.load_state_dict(checkpoint["classifier"])
        center_loss.load_state_dict(checkpoint["center_loss"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "mps" and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        history = checkpoint["history"]
        print(f"Resuming {run_dir.name} after epoch {len(history)}")
    else:
        save_json(run_dir / "config.json", config)

    print(f"\n=== {run_dir.relative_to(MODEL_DIR)} | lambda={lam} lr={settings['learning_rate']} "
          f"| held out {list(held_out)} | {data_info}")

    for epoch in range(len(history) + 1, epochs + 1):
        generator = torch.Generator().manual_seed(SEED * 1000 + epoch)
        loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                            generator=generator, num_workers=NUM_WORKERS)
        epoch_lr = optimizer.param_groups[0]["lr"]

        model.train()
        classifier.train()
        monitor = TrainingMonitor(device)
        monitor.start()
        sums = {"loss": 0.0, "ce": 0.0, "center": 0.0, "intra": 0.0, "correct": 0, "images": 0}
        batches = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            n = labels.size(0)

            optimizer.zero_grad(set_to_none=True)
            features = model(images)
            logits = classifier(features)
            ce = F.cross_entropy(logits, labels)
            c_loss = center_loss(features, labels)
            loss = ce + lam * c_loss
            loss.backward()
            optimizer.step()

            # Measure distances before updating the centers. The softmax baseline
            # also tracks centers so its training metrics can be compared.
            features = features.detach()
            intra = (features - center_loss.centers[labels]).norm(dim=1).mean()
            center_loss.update_centers(features, labels)

            sums["loss"] += loss.item() * n
            sums["ce"] += ce.item() * n
            sums["center"] += c_loss.item() * n
            sums["intra"] += intra.item() * n
            sums["correct"] += (logits.argmax(dim=1) == labels).sum().item()
            sums["images"] += n
            batches += 1
            monitor.sample_memory()
            if smoke and batches >= SMOKE_BATCHES:
                break

        scheduler.step()
        monitor.stop()
        report = monitor.report()
        n_seen = sums["images"]

        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / n_seen,
            "validation_accuracy": "",
            "validation_auc": "",
            "validation_threshold": "",
            "learning_rate": epoch_lr,
            "epoch_time_seconds": report["elapsed_seconds"],
            "validation_time_seconds": "",
            "classification_loss": sums["ce"] / n_seen,
            "center_loss": sums["center"] / n_seen,
            "weighted_center_loss": lam * sums["center"] / n_seen,
            "train_classification_accuracy": sums["correct"] / n_seen,
            "mean_intra_class_distance": sums["intra"] / n_seen,
            "mean_inter_center_distance": mean_inter_center_distance(center_loss.centers),
            "train_images": n_seen,
            "batches": batches,
            "peak_sampled_mps_driver_mib": report.get("peak_sampled_mps_driver_mib", ""),
            "peak_gpu_allocated_mib": report.get("peak_gpu_allocated_mib", ""),
        }

        # Use View 1 validation scores to compare tuning settings.
        if validation_dataset is not None:
            start = time.perf_counter()
            scored = score_pair_dataset(model, validation_dataset, device, batch_size=32)
            labels_v = [r["label"] for r in scored]
            scores_v = [r["score"] for r in scored]
            threshold = select_accuracy_threshold(scores_v, labels_v)
            accuracy = float(np.mean((np.array(scores_v) >= threshold) == np.array(labels_v)))
            row.update({
                "validation_accuracy": accuracy,
                "validation_auc": float(roc_auc_score(labels_v, scores_v)),
                "validation_threshold": threshold,
                "validation_time_seconds": round(time.perf_counter() - start, 3),
            })
            write_csv(run_dir / "validation_scores.csv",
                      [{"epoch": epoch, **r} for r in scored])

        history.append(row)
        write_csv(run_dir / "training_history.csv", history)

        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = ckpt_file.with_suffix(".tmp")
        torch.save({
            "settings": key,
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "center_loss": center_loss.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            "history": history,
        }, temporary)
        temporary.replace(ckpt_file)

        validation_text = (f" | val acc {row['validation_accuracy']:.4f} "
                           f"AUC {row['validation_auc']:.4f}"
                           if validation_dataset is not None else "")
        print(f"epoch {epoch:2d}/{epochs} | loss {row['train_loss']:.4f} "
              f"(ce {row['classification_loss']:.4f}, center {row['center_loss']:.2f}) "
              f"| intra {row['mean_intra_class_distance']:.2f} "
              f"inter {row['mean_inter_center_distance']:.2f}"
              f"{validation_text} | {row['epoch_time_seconds']:.0f}s", flush=True)

    # Score the held-out pairs for this run.
    if stage == "view2":
        held_out_scores = []
        for fold in held_out:
            fold_dataset = LFWPairDataset(split="view2", fold=fold, train=False)
            held_out_scores += score_pair_dataset(model, fold_dataset, device, batch_size=32)
        write_csv(run_dir / "heldout_scores.csv", held_out_scores)

    validated = [h for h in history if h["validation_accuracy"] != ""]
    best = max(validated, key=lambda h: h["validation_accuracy"]) if validated else None
    memory = [h["peak_sampled_mps_driver_mib"] for h in history
              if h["peak_sampled_mps_driver_mib"] not in ("", None)]

    summary = {
        "variant": variant,
        "stage": config["stage"],
        "run_number": run_number,
        "held_out_folds": list(held_out),
        "lambda": lam,
        "learning_rate": settings["learning_rate"],
        "seed": SEED,
        "epochs_completed": len(history),
        "training_seconds": round(sum(h["epoch_time_seconds"] for h in history), 1),
        "validation_seconds": round(sum(h["validation_time_seconds"] or 0 for h in history), 1),
        "final_train_loss": history[-1]["train_loss"],
        "final_classification_loss": history[-1]["classification_loss"],
        "final_center_loss": history[-1]["center_loss"],
        "final_intra_class_distance": history[-1]["mean_intra_class_distance"],
        "final_inter_center_distance": history[-1]["mean_inter_center_distance"],
        "final_validation_accuracy": history[-1]["validation_accuracy"] if validated else None,
        "final_validation_auc": history[-1]["validation_auc"] if validated else None,
        "best_validation_accuracy": best["validation_accuracy"] if best else None,
        "best_validation_epoch": best["epoch"] if best else None,
        "peak_sampled_mps_driver_mib": max(memory) if memory else None,
        "memory_note": "MPS: maximum of values sampled after each batch, not a true peak",
        "training_data": data_info,
        "device": str(device),
    }

    plot_training_curves(run_dir, history,
                         f"Center loss {variant} | {config['stage']} | lambda {lam}")
    save_json(complete_file, summary)

    # Remove the checkpoint before the next run starts from fresh weights.
    ckpt_file.unlink(missing_ok=True)
    return summary


# View 1 tuning

def write_tuning_summary(root):
    rows = []
    for file in sorted((root / "View_1").glob("*/*/run_complete.json")):
        s = json.loads(file.read_text())
        rows.append({
            "variant": s["variant"],
            "lambda": s["lambda"],
            "learning_rate": s["learning_rate"],
            "epochs": s["epochs_completed"],
            "final_validation_accuracy": s["final_validation_accuracy"],
            "final_validation_auc": s["final_validation_auc"],
            "best_validation_accuracy": s["best_validation_accuracy"],
            "best_validation_epoch": s["best_validation_epoch"],
            "final_train_loss": s["final_train_loss"],
            "final_intra_class_distance": s["final_intra_class_distance"],
            "final_inter_center_distance": s["final_inter_center_distance"],
            "training_minutes": round(s["training_seconds"] / 60, 1),
            "run_folder": file.parent.name,
        })

    # Rank by final-epoch accuracy, with AUC as the tie-breaker.
    rows.sort(key=lambda r: (r["variant"], -r["final_validation_accuracy"],
                             -r["final_validation_auc"]))
    rank = Counter()
    for r in rows:
        rank[r["variant"]] += 1
        r["rank_within_variant"] = rank[r["variant"]]

    if rows:
        write_csv(root / "View_1" / "tuning_summary.csv", rows)
        print("\nView 1 tuning summary (rank 1 = highest final accuracy, then AUC):")
        for r in rows:
            print(f"  {r['variant']:12s} lambda {r['lambda']:<6} lr {r['learning_rate']:<8g} "
                  f"acc {r['final_validation_accuracy']:.4f} AUC {r['final_validation_auc']:.4f}"
                  f"  rank {r['rank_within_variant']}")


def run_view1_tuning(smoke=False):
    root = SMOKE_DIR if smoke else RESULTS_DIR
    epochs = SMOKE_EPOCHS if smoke else EPOCHS

    for variant in VARIANTS:
        lambdas = LAMBDAS[variant][:1] if smoke else LAMBDAS[variant]
        for learning_rate in LEARNING_RATES:
            for lam in lambdas:
                run_dir = root / "View_1" / variant / f"lambda_{lam:g}_lr_{learning_rate:g}"
                settings = {"variant": variant, "lambda": lam,
                            "learning_rate": learning_rate, "epochs": epochs}
                train_one_run(run_dir, "view1", settings, smoke=smoke)
                write_tuning_summary(root)


# View 2 evaluation

def fold_table(rows, result):
    """Build per-fold accuracy rows using evaluate_view2's thresholds."""
    table = []
    for fold_result in result["folds"]:
        fold_rows = [r for r in rows if r["fold"] == fold_result["fold"]]
        threshold = fold_result["threshold"]
        correct = sum((r["score"] >= threshold) == bool(r["label"]) for r in fold_rows)
        if not math.isclose(correct / len(fold_rows), fold_result["accuracy"]):
            raise RuntimeError(f"Fold {fold_result['fold']}: accuracy mismatch")
        table.append({"fold": fold_result["fold"], "threshold": threshold,
                      "correct": correct, "n_pairs": len(fold_rows),
                      "accuracy": fold_result["accuracy"]})
    return table


def save_score_histogram(rows, folds, filename, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    genuine = [r["score"] for r in rows if r["label"] == 1]
    impostor = [r["score"] for r in rows if r["label"] == 0]
    ax.hist(genuine, bins=60, alpha=0.6, label="genuine (same person)")
    ax.hist(impostor, bins=60, alpha=0.6, label="impostor (different people)")
    thresholds = [f["threshold"] for f in folds]
    ax.axvline(np.mean(thresholds), color="black", linestyle="--",
               label=f"mean fold threshold {np.mean(thresholds):.3f}")
    ax.axvspan(min(thresholds), max(thresholds), color="grey", alpha=0.2,
               label="range of the 10 fold thresholds")
    ax.set(xlabel="cosine similarity", ylabel="pairs", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def aggregate_view2(variant_dir, variant, final, run_summaries, view1_row=None):
    rows = []
    fold_owner = {}
    for run_number, held_out in enumerate(VIEW2_HELD_OUT, start=1):
        run_rows = read_scores_csv(variant_dir / f"run_{run_number}" / "heldout_scores.csv")
        if {r["fold"] for r in run_rows} != set(held_out):
            raise RuntimeError(f"run_{run_number} scored folds other than {held_out}")
        for fold in held_out:
            if fold in fold_owner:
                raise RuntimeError(f"Fold {fold} scored twice")
            fold_owner[fold] = run_number
        rows += run_rows

    if sorted(fold_owner) != list(range(1, 11)):
        raise RuntimeError("Not every fold was evaluated exactly once")

    manifest = Counter((int(r["fold"]), r["img1"], r["img2"], int(r["label"]))
                       for r in read_manifest("view2_pairs.csv"))
    scored = Counter((r["fold"], r["img1"], r["img2"], r["label"]) for r in rows)
    if len(rows) != 6000 or scored != manifest:
        raise RuntimeError("Scored pairs do not match the 6,000 official View 2 pairs")

    # Select each fold's threshold from scores in the other nine folds.
    cosine = evaluate_view2(rows, score_key="score", higher_is_same=True)

    save_pair_scores(rows, variant_dir / "pair_scores.csv", score_key="score")
    cosine_table = fold_table(rows, cosine)
    write_csv(variant_dir / "fold_accuracy.csv", cosine_table)

    labels = np.array([r["label"] for r in rows])
    cos_scores = np.array([r["score"] for r in rows])
    far, tar, thresholds = roc_curve(labels, cos_scores, drop_intermediate=False)
    write_csv(variant_dir / "roc_data.csv",
              [{"far": f, "tar": t, "threshold": th} for f, t, th in zip(far, tar, thresholds)])

    save_roc_plot({f"Center loss {variant} (cosine)": (cos_scores, labels)},
                  variant_dir / "roc.png")
    save_score_histogram(rows, cosine_table, variant_dir / "score_histogram.png",
                         f"Center loss {variant}: View 2 out-of-fold scores")

    # Record paths and scores for error analysis, without copying image data.
    fold_thresholds = {f["fold"]: f["threshold"] for f in cosine_table}
    errors = []
    for r in rows:
        threshold = fold_thresholds[r["fold"]]
        predicted_same = r["score"] >= threshold
        if predicted_same != bool(r["label"]):
            errors.append({
                "error_type": "false_accept" if predicted_same else "false_reject",
                "confidence": abs(r["score"] - threshold),
                "fold": r["fold"], "img1": r["img1"], "img2": r["img2"],
                "label": r["label"], "cosine": r["score"],
                "fold_threshold": threshold, "scored_by_run": fold_owner[r["fold"]],
            })
    errors.sort(key=lambda e: (e["error_type"], -e["confidence"]))
    write_csv(variant_dir / "confident_errors.csv", errors)

    training_seconds = [s["training_seconds"] for s in run_summaries]
    memory = [s["peak_sampled_mps_driver_mib"] for s in run_summaries
              if s["peak_sampled_mps_driver_mib"] is not None]

    summary = {
        "model": "Softmax + Center Loss" if final["lambda"] > 0 else "Softmax baseline",
        "variant": variant,
        "lfw_protocol": "unrestricted, labeled outside data",
        "fold_scheme": "reduced five-run scheme (PDF 4.2): run j held out folds 2j-1, 2j",
        "fold_to_run": {str(k): v for k, v in sorted(fold_owner.items())},
        "n_training_runs": len(run_summaries),
        "seed": SEED,
        "seeds_note": "one seed per run (42); additional seeds not run",
        "final_hyperparameters": final | {
            "center_alpha": CENTER_ALPHA, "batch_size": BATCH_SIZE, "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY, "lr_schedule": "cosine",
            "min_images_per_class": MIN_IMAGES_PER_CLASS},
        "view1_selection": view1_row,
        "cosine": {k: cosine[k] for k in
                   ["accuracy_mean", "accuracy_std", "auc", "eer", "tar_at_far_0.01"]}
                  | {"fold_accuracy_file": "fold_accuracy.csv"},
        "accuracy_std_note": "sample standard deviation (ddof=1) over the 10 folds",
        "pair_scores_files": ["pair_scores.csv"],
        "classes_per_run": [s["training_data"]["classes"] for s in run_summaries],
        "training_images_per_run": [s["training_data"]["training_images"] for s in run_summaries],
        "training_time_total_seconds": round(sum(training_seconds), 1),
        "training_time_mean_per_run_seconds": round(float(np.mean(training_seconds)), 1),
        "epochs_per_run": [s["epochs_completed"] for s in run_summaries],
        "per_run": run_summaries,
        "hardware": hardware_info(get_device()),
        "peak_sampled_mps_driver_mib": max(memory) if memory else None,
        "memory_note": "MPS: maximum of values sampled after each batch, not a true peak",
        "software": software_versions(),
    }
    save_json(variant_dir / "summary.json", summary)

    print(f"\n{variant}: cosine accuracy {cosine['accuracy_mean']:.4f} "
          f"± {cosine['accuracy_std']:.4f}, AUC {cosine['auc']:.4f}, EER {cosine['eer']:.4f}, "
          f"TAR@FAR=1% {cosine['tar_at_far_0.01']:.4f}")


def view1_selection(variant, final):
    """Check FINAL_CONFIG against the top-ranked View 1 row and return that row."""
    with (RESULTS_DIR / "View_1" / "tuning_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["variant"] == variant]

    if len(rows) != len(LAMBDAS[variant]) * len(LEARNING_RATES):
        raise RuntimeError(f"View 1 grid for {variant} is incomplete: {len(rows)} runs")

    best = min(rows, key=lambda row: int(row["rank_within_variant"]))
    if (float(best["lambda"]) != final["lambda"]
            or float(best["learning_rate"]) != final["learning_rate"]
            or int(best["epochs"]) != final["epochs"]):
        raise RuntimeError(f"FINAL_CONFIG[{variant!r}] = {final} does not match the "
                           f"View 1 selection {best}")

    return best | {"selection_criterion": "highest final-epoch View 1 accuracy "
                                          "(best threshold on pairsDevTest), ties by AUC"}


def run_view2_final(smoke=False):
    root = SMOKE_DIR if smoke else RESULTS_DIR

    for variant in VARIANTS:
        if smoke:
            final = {"lambda": LAMBDAS[variant][0], "learning_rate": LEARNING_RATES[0],
                     "epochs": SMOKE_EPOCHS}
            selection = None
        else:
            final = FINAL_CONFIG[variant]
            if final is None:
                print(f"Skipping View 2 {variant}: freeze FINAL_CONFIG from View 1 first.")
                continue
            selection = view1_selection(variant, final)

        variant_dir = root / "View_2" / variant
        settings = {"variant": variant, **final}
        summaries = []
        for held_out in VIEW2_HELD_OUT:
            run_number = VIEW2_HELD_OUT.index(held_out) + 1
            summaries.append(train_one_run(variant_dir / f"run_{run_number}", "view2",
                                           settings, held_out=held_out, smoke=smoke))
        aggregate_view2(variant_dir, variant, final, summaries, selection)


if __name__ == "__main__":
    if STAGE == "smoke_test":
        run_view1_tuning(smoke=True)
        run_view2_final(smoke=True)
    elif STAGE == "view1_tuning":
        run_view1_tuning()
    elif STAGE == "view2_final":
        run_view2_final()
    else:
        raise ValueError(f"Unknown STAGE: {STAGE}")
