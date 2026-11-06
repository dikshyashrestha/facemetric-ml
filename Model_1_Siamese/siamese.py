"""
Train and evaluate the Siamese contrastive-loss model on LFW.

The restricted variant uses the official training pairs. The unrestricted
variant samples balanced pairs from training identities. Set STAGE below to
run a smoke test, tune settings on View 1, or evaluate on View 2.

Run from the repository root with:
    python -m Model_1_Siamese.siamese

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
from collections import Counter, defaultdict
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
from Common.dataset_loader import LFWPairDataset, read_manifest
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

STAGE = "view2_final"         # Choose "smoke_test", "view1_tuning", or "view2_final".
VARIANTS = ["restricted", "unrestricted"]

SEED = 42                     # Shared by each run and recorded in config.json.
EPOCHS = 30                   # Epochs per run.
BATCH_PAIRS = 16              # Each batch has 8 genuine and 8 impostor pairs.
NUM_WORKERS = 0               # Image loading takes about 2% of epoch time on M1.

# Start with margin 1.0 as a pilot run.
MARGINS = [1.0, 0.5, 0.75, 1.25]
LEARNING_RATES = [1e-4]
WEIGHT_DECAY = 5e-4           # AdamW weight decay.
ADAMW_BETAS = (0.9, 0.999)

# Best View 1 settings for each variant, ranked by accuracy then AUC.
FINAL_CONFIG = {
    "restricted": {"margin": 1.25, "learning_rate": 1e-4, "epochs": 30},    # View 1 accuracy 0.644, AUC 0.683.
    "unrestricted": {"margin": 1.25, "learning_rate": 1e-4, "epochs": 30},  # View 1 accuracy 0.625, AUC 0.650.
}

VIEW1_TRAIN_PAIRS = 2200
VIEW2_TRAIN_PAIRS = 4800      # Eight folds with 600 pairs each.
VIEW2_HELD_OUT = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

# Smoke-test settings
SMOKE_BATCHES = 3
SMOKE_EPOCHS = 2
SMOKE_VAL_PAIRS = 64

MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODEL_DIR / "results"
SMOKE_DIR = MODEL_DIR / "outputs" / "smoke_test"      # Gitignored.
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"            # Gitignored.


# Contrastive loss

class ContrastiveLoss(nn.Module):
    """Contrastive loss from Hadsell, Chopra, and LeCun (2006).

    L = y * d^2 + (1 - y) * max(0, m - d)^2,   d = ||f(x1) - f(x2)||

    Labels are 1 for genuine pairs and 0 for impostor pairs. Embeddings are
    L2-normalized before the loss, so distances range from 0 to 2.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, labels):
        distances = F.pairwise_distance(embedding1, embedding2)

        labels = labels.float()

        positive_loss = labels * distances.pow(2)

        negative_loss = (1 - labels) * (
            torch.clamp(self.margin - distances, min=0).pow(2)
        )

        return (positive_loss + negative_loss).mean()


# Training data

def image_identity(image_path):
    return image_path.split("/")[0]


def restricted_training_pairs(stage, held_out):
    """Load official labeled pairs without generating any additional pairs."""
    if stage == "view1":
        return read_manifest("view1_train_pairs.csv")
    return [
        row for row in read_manifest("view2_pairs.csv")
        if int(row["fold"]) not in held_out
    ]


def identity_pools(stage, held_out):
    """Group training image paths by identity for pair sampling."""
    if stage == "view1":
        rows = read_manifest("view1_train_images.csv")
    else:
        rows = [
            row for row in read_manifest("view2_images.csv")
            if int(row["fold"]) not in held_out
        ]

    pools = defaultdict(list)
    for row in rows:
        pools[row["identity"]].append(row["image_path"])

    return {identity: sorted(paths) for identity, paths in sorted(pools.items())}


def generate_unrestricted_pairs(pools, n_pairs, rng):
    """Sample balanced genuine and impostor pairs from training identities.

    Genuine pairs use two distinct images of one identity. Impostor pairs use
    images from two distinct identities; identities with one image are allowed.
    """
    identities = list(pools)
    multi_image = [identity for identity in identities if len(pools[identity]) >= 2]
    rows = []

    for _ in range(n_pairs // 2):
        person = multi_image[rng.integers(len(multi_image))]
        a, b = rng.choice(len(pools[person]), size=2, replace=False)
        rows.append({"fold": "0", "img1": pools[person][a],
                     "img2": pools[person][b], "label": "1"})

    for _ in range(n_pairs // 2):
        i, j = rng.choice(len(identities), size=2, replace=False)
        person1, person2 = identities[i], identities[j]
        rows.append({
            "fold": "0",
            "img1": pools[person1][rng.integers(len(pools[person1]))],
            "img2": pools[person2][rng.integers(len(pools[person2]))],
            "label": "0",
        })

    return rows


def balanced_batches(rows, batch_pairs, rng):
    """Shuffle pairs into batches with equal genuine and impostor counts."""
    positives = [i for i, row in enumerate(rows) if int(row["label"]) == 1]
    negatives = [i for i, row in enumerate(rows) if int(row["label"]) == 0]

    if len(positives) != len(negatives):
        raise ValueError("Training pairs must be balanced 1:1")

    positives = rng.permutation(positives)
    negatives = rng.permutation(negatives)
    half = batch_pairs // 2

    return [
        [int(i) for i in positives[start:start + half]] +
        [int(i) for i in negatives[start:start + half]]
        for start in range(0, len(positives), half)
    ]


def check_training_data(variant, stage, held_out, pairs=None, pools=None):
    """Check that the training data does not overlap the evaluation split."""

    if pairs is not None:
        train_ids = {image_identity(p) for row in pairs for p in (row["img1"], row["img2"])}
    else:
        train_ids = set(pools)

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

    overlap = train_ids & evaluation_ids
    if overlap:
        raise RuntimeError(
            f"LEAKAGE: {len(overlap)} evaluation identities in training data, "
            f"e.g. {sorted(overlap)[:5]}"
        )

    info = {"training_identities": len(train_ids), "evaluation_identity_overlap": 0}

    if pairs is not None:
        expected = VIEW1_TRAIN_PAIRS if stage == "view1" else VIEW2_TRAIN_PAIRS
        n_positive = sum(int(row["label"]) for row in pairs)
        if len(pairs) != expected or 2 * n_positive != expected:
            raise RuntimeError(f"Expected {expected} balanced pairs, got {len(pairs)}")

        if stage == "view2":
            held_out_pairs = {
                (row["img1"], row["img2"]) for row in read_manifest("view2_pairs.csv")
                if int(row["fold"]) in held_out
            }
            if any((row["img1"], row["img2"]) in held_out_pairs for row in pairs):
                raise RuntimeError("LEAKAGE: a held-out pair is in the training pairs")
            if any(int(row["fold"]) in held_out for row in pairs):
                raise RuntimeError("LEAKAGE: training pair from a held-out fold")

        info.update({
            "training_pairs": len(pairs),
            "genuine_pairs": n_positive,
            "impostor_pairs": len(pairs) - n_positive,
            "training_images": len({p for row in pairs for p in (row["img1"], row["img2"])}),
        })
    else:
        info.update({
            "training_images": sum(len(paths) for paths in pools.values()),
            "identities_with_2plus_images": sum(len(paths) >= 2 for paths in pools.values()),
        })

    return info


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
    axes[0].plot(epochs, [h["positive_pair_loss"] for h in history], label="genuine pairs")
    axes[0].plot(epochs, [h["negative_pair_loss"] for h in history], label="impostor pairs")
    axes[0].set(xlabel="epoch", ylabel="contrastive loss", title="Training loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["genuine_dist_mean"] for h in history], label="genuine")
    axes[1].plot(epochs, [h["impostor_dist_mean"] for h in history], label="impostor")
    axes[1].set(xlabel="epoch", ylabel="mean training distance d",
                title="Training pair distances")
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

def train_one_run(run_dir, variant, stage, margin, learning_rate, epochs,
                  held_out=(), run_number=None, smoke=False):
    run_dir = Path(run_dir)
    complete_file = run_dir / "run_complete.json"
    if complete_file.is_file():
        print(f"Already complete, skipping: {run_dir}")
        return json.loads(complete_file.read_text())

    device = get_device()
    seed_everything(SEED)
    held_out = tuple(held_out)

    if variant == "restricted":
        pairs = restricted_training_pairs(stage, held_out)
        pools = None
        data_info = check_training_data(variant, stage, held_out, pairs=pairs)
        sampling = "official labelled pairs only, reshuffled each epoch into balanced batches"
    else:
        pairs = None
        pools = identity_pools(stage, held_out)
        data_info = check_training_data(variant, stage, held_out, pools=pools)
        sampling = ("fresh pairs every epoch from training identities: genuine = uniform "
                    "identity with >=2 images, two distinct images; impostor = two distinct "
                    "uniform identities (singletons allowed), one random image each")

    n_pairs = VIEW1_TRAIN_PAIRS if stage == "view1" else VIEW2_TRAIN_PAIRS
    batches_per_epoch = math.ceil((n_pairs // 2) / (BATCH_PAIRS // 2))

    # Use the shared pair loader and training-time image transforms.
    train_dataset = LFWPairDataset(split="view1_train", train=True)
    if pairs is not None:
        train_dataset.samples = pairs

    validation_dataset = None
    if stage == "view1":
        validation_dataset = LFWPairDataset(split="view1_validation", train=False)
        if smoke:
            half = SMOKE_VAL_PAIRS // 2
            validation_dataset.samples = (validation_dataset.samples[:half] +
                                          validation_dataset.samples[-half:])

    model = FaceEmbeddingNet().to(device)
    loss_fn = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=ADAMW_BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    settings = {
        "variant": variant, "stage": stage, "held_out_folds": list(held_out),
        "margin": margin, "learning_rate": learning_rate, "epochs": epochs,
        "batch_pairs": BATCH_PAIRS, "seed": SEED, "weight_decay": WEIGHT_DECAY,
        "pairs_per_epoch": n_pairs, "smoke_test": smoke,
    }

    config = {
        "model": "Siamese network, contrastive loss",
        "variant": variant,
        "lfw_protocol": ("image-restricted" if variant == "restricted"
                         else "unrestricted, labeled outside data"),
        "stage": "View 1" if stage == "view1" else "View 2",
        "run_number": run_number,
        "held_out_folds": list(held_out),
        "fold_scheme": (None if stage == "view1" else
                        "reduced five-run scheme (PDF 4.2): run j holds out folds 2j-1 and 2j"),
        "seed": SEED,
        "backbone": "Common.base_model.FaceEmbeddingNet (modified ResNet-18, random init)",
        "embedding_dim": 512,
        "trainable_parameters": count_trainable_parameters(model),
        "loss": {
            "name": "contrastive (Hadsell et al., 2006)",
            "formula": "y*d^2 + (1-y)*max(0, m-d)^2, mean over pairs",
            "distance": "Euclidean distance between L2-normalized embeddings",
            "margin": margin,
        },
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                      "weight_decay": WEIGHT_DECAY, "betas": list(ADAMW_BETAS)},
        "lr_schedule": {"name": "CosineAnnealingLR", "T_max_epochs": epochs,
                        "eta_min": 0.0, "step": "once per epoch"},
        "batch": {"pairs_per_batch": BATCH_PAIRS, "genuine_per_batch": BATCH_PAIRS // 2,
                  "impostor_per_batch": BATCH_PAIRS // 2,
                  "forward_pass": "both images of all pairs in one batch of 32 images"},
        "pairs_per_epoch": n_pairs,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_updates_planned": batches_per_epoch * epochs,
        "max_epochs": epochs,
        "pair_sampling": sampling,
        "pair_sampling_seed": f"numpy default_rng([{SEED}, epoch])",
        "training_data": data_info,
        "augmentation": "random horizontal flip p=0.5, independently per image",
        "normalization": "RGB 112x112, (pixel - 127.5) / 128",
        "test_time_features": "original + flipped embeddings summed, then L2-normalized",
        "validation": (None if stage == "view2" else
                       "View 1 pairsDevTest (1,000 pairs): accuracy at the best threshold "
                       "on those pairs (tuning metric, optimistic) and ROC-AUC"),
        "hardware": hardware_info(device),
        "software": software_versions(),
        "settings_key": settings,
    }
    if smoke:
        config["smoke_test"] = {"batches_per_epoch": SMOKE_BATCHES,
                                "validation_pairs": SMOKE_VAL_PAIRS}

    history = []
    updates = 0
    ckpt_file = checkpoint_path(run_dir)

    if ckpt_file.is_file():
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if checkpoint["settings"] != settings:
            raise RuntimeError(
                f"Settings changed since this run started.\nSaved: {checkpoint['settings']}\n"
                f"Now:   {settings}\nRestore the settings or delete {ckpt_file}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "mps" and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        history = checkpoint["history"]
        updates = checkpoint["updates"]
        print(f"Resuming {run_dir.name} after epoch {len(history)}")
    else:
        save_json(run_dir / "config.json", config)

    print(f"\n=== {run_dir.relative_to(MODEL_DIR)} | {variant} | margin={margin} "
          f"lr={learning_rate} | held out {list(held_out)} | {data_info}")

    for epoch in range(len(history) + 1, epochs + 1):
        rng = np.random.default_rng([SEED, epoch])
        if variant == "unrestricted":
            train_dataset.samples = generate_unrestricted_pairs(pools, n_pairs, rng)

        batches = balanced_batches(train_dataset.samples, BATCH_PAIRS, rng)
        if smoke:
            batches = batches[:SMOKE_BATCHES]

        loader = DataLoader(train_dataset, batch_sampler=batches, num_workers=NUM_WORKERS)
        epoch_lr = optimizer.param_groups[0]["lr"]

        model.train()
        monitor = TrainingMonitor(device)
        monitor.start()
        total_loss = 0.0
        pairs_seen = 0
        genuine_distances, impostor_distances = [], []

        for image1, image2, labels, _ in loader:
            n = labels.size(0)
            images = torch.cat([image1, image2]).to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Run both images through the shared network in one forward pass.
            embeddings = model(images, normalize=True)
            embedding1, embedding2 = embeddings[:n], embeddings[n:]

            loss = loss_fn(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()
            updates += 1

            distances = F.pairwise_distance(embedding1.detach(), embedding2.detach()).cpu()
            is_genuine = labels.cpu().bool()
            genuine_distances.append(distances[is_genuine])
            impostor_distances.append(distances[~is_genuine])
            total_loss += loss.item() * n
            pairs_seen += n
            monitor.sample_memory()

        scheduler.step()
        monitor.stop()
        report = monitor.report()

        genuine = torch.cat(genuine_distances).numpy()
        impostor = torch.cat(impostor_distances).numpy()

        row = {
            "epoch": epoch,
            "train_loss": total_loss / pairs_seen,
            "validation_accuracy": "",
            "validation_auc": "",
            "validation_threshold": "",
            "learning_rate": epoch_lr,
            "epoch_time_seconds": report["elapsed_seconds"],
            "validation_time_seconds": "",
            "positive_pair_loss": float(np.mean(genuine ** 2)),
            "negative_pair_loss": float(np.mean(np.clip(margin - impostor, 0, None) ** 2)),
            "active_negative_fraction": float(np.mean(impostor < margin)),
            "genuine_dist_mean": float(genuine.mean()),
            "genuine_dist_std": float(genuine.std()),
            "genuine_dist_p05": float(np.percentile(genuine, 5)),
            "genuine_dist_p50": float(np.percentile(genuine, 50)),
            "genuine_dist_p95": float(np.percentile(genuine, 95)),
            "impostor_dist_mean": float(impostor.mean()),
            "impostor_dist_std": float(impostor.std()),
            "impostor_dist_p05": float(np.percentile(impostor, 5)),
            "impostor_dist_p50": float(np.percentile(impostor, 50)),
            "impostor_dist_p95": float(np.percentile(impostor, 95)),
            "train_pairs": pairs_seen,
            "batches": len(batches),
            "optimizer_updates_total": updates,
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
            "settings": settings,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
            "history": history,
            "updates": updates,
        }, temporary)
        temporary.replace(ckpt_file)

        validation_text = (f" | val acc {row['validation_accuracy']:.4f} "
                           f"AUC {row['validation_auc']:.4f}"
                           if validation_dataset is not None else "")
        print(f"epoch {epoch:2d}/{epochs} | loss {row['train_loss']:.4f} "
              f"(pos {row['positive_pair_loss']:.4f}, neg {row['negative_pair_loss']:.4f}) "
              f"| d gen {row['genuine_dist_mean']:.3f} imp {row['impostor_dist_mean']:.3f}"
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
        "margin": margin,
        "learning_rate": learning_rate,
        "seed": SEED,
        "epochs_completed": len(history),
        "optimizer_updates": updates,
        "training_seconds": round(sum(h["epoch_time_seconds"] for h in history), 1),
        "validation_seconds": round(sum(h["validation_time_seconds"] or 0 for h in history), 1),
        "final_train_loss": history[-1]["train_loss"],
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
                         f"Siamese {variant} | {config['stage']} | margin {margin}")
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
            "margin": s["margin"],
            "learning_rate": s["learning_rate"],
            "epochs": s["epochs_completed"],
            "final_validation_accuracy": s["final_validation_accuracy"],
            "final_validation_auc": s["final_validation_auc"],
            "best_validation_accuracy": s["best_validation_accuracy"],
            "best_validation_epoch": s["best_validation_epoch"],
            "final_train_loss": s["final_train_loss"],
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
            print(f"  {r['variant']:12s} margin {r['margin']:<5} lr {r['learning_rate']:<8g} "
                  f"acc {r['final_validation_accuracy']:.4f} AUC {r['final_validation_auc']:.4f}"
                  f"  rank {r['rank_within_variant']}")


def run_view1_tuning(smoke=False):
    root = SMOKE_DIR if smoke else RESULTS_DIR
    margins = MARGINS[:1] if smoke else MARGINS
    epochs = SMOKE_EPOCHS if smoke else EPOCHS

    for variant in VARIANTS:
        for learning_rate in LEARNING_RATES:
            for margin in margins:
                run_dir = root / "View_1" / variant / f"margin_{margin}_lr_{learning_rate:g}"
                train_one_run(run_dir, variant, "view1", margin, learning_rate, epochs,
                              smoke=smoke)
                write_tuning_summary(root)


# View 2 evaluation

def fold_table(rows, result, key, higher_is_same):
    """Build per-fold accuracy rows using evaluate_view2's thresholds."""
    table = []
    for fold_result in result["folds"]:
        fold_rows = [r for r in rows if r["fold"] == fold_result["fold"]]
        threshold = fold_result["threshold"]
        correct = sum(
            ((r[key] >= threshold) if higher_is_same else (r[key] <= threshold)) == bool(r["label"])
            for r in fold_rows
        )
        if not math.isclose(correct / len(fold_rows), fold_result["accuracy"]):
            raise RuntimeError(f"Fold {fold_result['fold']}: accuracy mismatch")
        table.append({"fold": fold_result["fold"], "threshold": threshold,
                      "correct": correct, "n_pairs": len(fold_rows),
                      "accuracy": fold_result["accuracy"]})
    return table


def save_score_histogram(rows, cosine_folds, l2_folds, filename, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, folds, label in [
        (axes[0], "score", cosine_folds, "cosine similarity"),
        (axes[1], "squared_l2", l2_folds, "squared Euclidean distance"),
    ]:
        genuine = [r[key] for r in rows if r["label"] == 1]
        impostor = [r[key] for r in rows if r["label"] == 0]
        ax.hist(genuine, bins=60, alpha=0.6, label="genuine (same person)")
        ax.hist(impostor, bins=60, alpha=0.6, label="impostor (different people)")
        thresholds = [f["threshold"] for f in folds]
        ax.axvline(np.mean(thresholds), color="black", linestyle="--",
                   label=f"mean fold threshold {np.mean(thresholds):.3f}")
        ax.axvspan(min(thresholds), max(thresholds), color="grey", alpha=0.2,
                   label="range of the 10 fold thresholds")
        ax.set(xlabel=label, ylabel="pairs")
        ax.legend(fontsize=8)
    fig.suptitle(title)
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
    squared_l2 = evaluate_view2(rows, score_key="squared_l2", higher_is_same=False)

    save_pair_scores(rows, variant_dir / "pair_scores.csv", score_key="score")
    save_pair_scores(rows, variant_dir / "pair_scores_squared_l2.csv", score_key="squared_l2")
    cosine_table = fold_table(rows, cosine, "score", True)
    l2_table = fold_table(rows, squared_l2, "squared_l2", False)
    write_csv(variant_dir / "fold_accuracy.csv", cosine_table)
    write_csv(variant_dir / "fold_accuracy_squared_l2.csv", l2_table)

    labels = np.array([r["label"] for r in rows])
    cos_scores = np.array([r["score"] for r in rows])
    distances = np.array([r["squared_l2"] for r in rows])

    # For unit vectors, squared distance equals 2 - 2 times cosine similarity.
    l2_vs_cos = float(np.max(np.abs(distances - (2 - 2 * cos_scores))))

    for key, scores, name in [("score", cos_scores, "roc_data.csv"),
                              ("squared_l2", -distances, "roc_data_squared_l2.csv")]:
        far, tar, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        write_csv(variant_dir / name, [
            {"far": f, "tar": t, "threshold": (th if key == "score" else -th)}
            for f, t, th in zip(far, tar, thresholds)
        ])

    save_roc_plot({f"Siamese {variant} (cosine)": (cos_scores, labels),
                   f"Siamese {variant} (squared L2, negated)": (-distances, labels)},
                  variant_dir / "roc.png")
    save_score_histogram(rows, cosine_table, l2_table, variant_dir / "score_histogram.png",
                         f"Siamese {variant}: View 2 out-of-fold scores")

    # Record paths and scores for error analysis, without copying image data.
    thresholds = {f["fold"]: f["threshold"] for f in cosine_table}
    errors = []
    for r in rows:
        threshold = thresholds[r["fold"]]
        predicted_same = r["score"] >= threshold
        if predicted_same != bool(r["label"]):
            errors.append({
                "error_type": "false_accept" if predicted_same else "false_reject",
                "confidence": abs(r["score"] - threshold),
                "fold": r["fold"], "img1": r["img1"], "img2": r["img2"],
                "label": r["label"], "cosine": r["score"], "squared_l2": r["squared_l2"],
                "fold_threshold": threshold, "scored_by_run": fold_owner[r["fold"]],
            })
    errors.sort(key=lambda e: (e["error_type"], -e["confidence"]))
    write_csv(variant_dir / "confident_errors.csv", errors)

    training_seconds = [s["training_seconds"] for s in run_summaries]
    epochs = [s["epochs_completed"] for s in run_summaries]
    memory = [s["peak_sampled_mps_driver_mib"] for s in run_summaries
              if s["peak_sampled_mps_driver_mib"] is not None]

    def metrics(result, fold_file):
        return {k: result[k] for k in
                ["accuracy_mean", "accuracy_std", "auc", "eer", "tar_at_far_0.01"]} | {
                "fold_accuracy_file": fold_file}

    summary = {
        "model": "Siamese network, contrastive loss",
        "variant": variant,
        "lfw_protocol": ("image-restricted" if variant == "restricted"
                         else "unrestricted, labeled outside data"),
        "fold_scheme": "reduced five-run scheme (PDF 4.2): run j held out folds 2j-1, 2j",
        "fold_to_run": {str(k): v for k, v in sorted(fold_owner.items())},
        "n_training_runs": len(run_summaries),
        "seed": SEED,
        "seeds_note": "one seed per run (42); additional seeds not run",
        "final_hyperparameters": final | {
            "batch_pairs": BATCH_PAIRS, "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "lr_schedule": "cosine", "pairs_per_epoch": VIEW2_TRAIN_PAIRS},
        "view1_selection": view1_row,
        "cosine": metrics(cosine, "fold_accuracy.csv"),
        "squared_l2": metrics(squared_l2, "fold_accuracy_squared_l2.csv"),
        "accuracy_std_note": "sample standard deviation (ddof=1) over the 10 folds",
        "squared_l2_vs_2_minus_2cos_max_abs_diff": l2_vs_cos,
        "pair_scores_files": ["pair_scores.csv", "pair_scores_squared_l2.csv"],
        "training_time_total_seconds": round(sum(training_seconds), 1),
        "training_time_mean_per_run_seconds": round(float(np.mean(training_seconds)), 1),
        "epochs_per_run": epochs,
        "per_run": run_summaries,
        "hardware": hardware_info(get_device()),
        "peak_sampled_mps_driver_mib": max(memory) if memory else None,
        "memory_note": "MPS: maximum of values sampled after each batch, not a true peak",
        "software": software_versions(),
    }
    save_json(variant_dir / "summary.json", summary)

    print(f"\n{variant}: cosine accuracy {cosine['accuracy_mean']:.4f} "
          f"± {cosine['accuracy_std']:.4f}, AUC {cosine['auc']:.4f}, EER {cosine['eer']:.4f}, "
          f"TAR@FAR=1% {cosine['tar_at_far_0.01']:.4f} | squared L2 accuracy "
          f"{squared_l2['accuracy_mean']:.4f} | max |d^2-(2-2cos)| = {l2_vs_cos:.2e}")


def view1_selection(variant, final):
    """Check FINAL_CONFIG against the top-ranked View 1 row and return that row."""
    with (RESULTS_DIR / "View_1" / "tuning_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["variant"] == variant]

    if len(rows) != len(MARGINS) * len(LEARNING_RATES):
        raise RuntimeError(f"View 1 grid for {variant} is incomplete: {len(rows)} runs")

    best = min(rows, key=lambda row: int(row["rank_within_variant"]))
    if (float(best["margin"]) != final["margin"]
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
            final = {"margin": MARGINS[0], "learning_rate": LEARNING_RATES[0],
                     "epochs": SMOKE_EPOCHS}
            selection = None
        else:
            final = FINAL_CONFIG[variant]
            if final is None:
                print(f"Skipping View 2 {variant}: freeze FINAL_CONFIG from View 1 first.")
                continue
            selection = view1_selection(variant, final)

        variant_dir = root / "View_2" / variant
        summaries = []
        for run_number, held_out in enumerate(VIEW2_HELD_OUT, start=1):
            summaries.append(train_one_run(
                variant_dir / f"run_{run_number}", variant, "view2",
                final["margin"], final["learning_rate"], final["epochs"],
                held_out=held_out, run_number=run_number, smoke=smoke,
            ))
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
