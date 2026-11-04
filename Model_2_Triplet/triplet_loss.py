"""Triplet loss training pipeline for the LFW project."""

import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add the project root to the Python path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep torch import order stable on macOS.
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import sklearn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from Common.base_model import FaceEmbeddingNet
from Common.dataset_loader import LFWIdentityDataset, LFWPairDataset, read_manifest
from Common.dataset_transforms import get_eval_transform
from Common.model_evaluation import (
    evaluate_view2, save_pair_scores, save_roc_plot,
    score_pair_dataset, select_accuracy_threshold, test_time_embeddings,
)
from Common.model_utils import (
    TrainingMonitor, count_trainable_parameters, get_device,
    hardware_info, save_json, seed_everything,
)


# Training config.
STAGE = "view2_final"  # "smoke_test", "view1_tuning", "view2_final"
MINING_METHODS = ["semi_hard", "batch_hard"]
MARGINS = [0.1, 0.2, 0.3]  # Margin sweep
LEARNING_RATES = [1e-4, 3e-4]  # LR sweep
EPOCHS = 30                # Max View 1 epochs
VIEW1_MIN_EPOCHS = 20      # Min epochs for plateau check
VIEW1_PLATEAU_PATIENCE = 6 # Plateau patience
VIEW1_MIN_IMPROVEMENT = 0.002  # Min validation gain
SEED = 42                 # Fixed seed
WEIGHT_DECAY = 5e-4
ADAMW_BETAS = (0.9, 0.999)

# Batch setup for the local workstation.
# Use only identities with at least two distinct images.
P_IDENTITIES = 8
K_IMAGES = 4
SINGLETON_NEGATIVES = 4
NUM_WORKERS = 0

# Keep optimizer steps aligned with the Siamese setup.
VIEW1_STEPS_PER_EPOCH = math.ceil(2200 / 16)
VIEW2_STEPS_PER_EPOCH = math.ceil(4800 / 16)
VIEW2_HELD_OUT = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

SMOKE_EPOCHS = 1
SMOKE_STEPS = 3
SMOKE_VAL_PAIRS = 64
SMOKE_FOLD_PAIRS = 32
EMBEDDINGS_PER_FOLD = 128  # Local plot samples

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent
RESULTS_DIR = MODEL_DIR / "results"
SMOKE_DIR = MODEL_DIR / "outputs" / "smoke_test"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
IMAGE_ROOT = ROOT / "Dataset" / "Processed" / "deepfunneled_112"


# Output helpers.
def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def software_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = None
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(),
            "git_commit": commit}


def ckpt_path(run_dir):
    return CHECKPOINT_DIR / run_dir.relative_to(MODEL_DIR) / "last_epoch.pt"


def view1_plateau(history):
    """Check whether validation accuracy has flatlined."""
    best_accuracy = -float("inf")
    last_improvement_epoch = 0
    for row in history:
        if row["validation_accuracy"] == "":
            continue
        accuracy = float(row["validation_accuracy"])
        if accuracy > best_accuracy + VIEW1_MIN_IMPROVEMENT:
            best_accuracy = accuracy
            last_improvement_epoch = int(row["epoch"])
    epochs_without_improvement = (int(history[-1]["epoch"]) - last_improvement_epoch
                                  if history else 0)
    reached = (bool(history) and len(history) >= VIEW1_MIN_EPOCHS
               and epochs_without_improvement >= VIEW1_PLATEAU_PATIENCE)
    return {"reached": bool(reached),
            "last_meaningful_improvement_epoch": last_improvement_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "minimum_epochs": VIEW1_MIN_EPOCHS,
            "patience": VIEW1_PLATEAU_PATIENCE,
            "min_accuracy_improvement": VIEW1_MIN_IMPROVEMENT}


# Batch sampling.
def identity_indices(dataset):
    pools = defaultdict(list)
    for idx, row in enumerate(dataset.samples):
        pools[row["identity"]].append(idx)
    multi = sorted(i for i, paths in pools.items() if len(paths) >= 2)
    singleton = sorted(i for i, paths in pools.items() if len(paths) == 1)
    if len(multi) < P_IDENTITIES:
        raise RuntimeError(f"Need >= {P_IDENTITIES} multi-image identities; got {len(multi)}")
    return pools, multi, singleton


def sample_batch_indices(pools, multi, singleton, rng):
    selected_ids = rng.choice(multi, size=P_IDENTITIES, replace=False)
    indices = []
    for identity in selected_ids:
        items = pools[identity]
        # Start with unique images, then repeat if needed.
        chosen = rng.choice(items, size=min(K_IMAGES, len(items)), replace=False).tolist()
        if len(chosen) < K_IMAGES:
            chosen += rng.choice(items, size=K_IMAGES-len(chosen), replace=True).tolist()
        indices.extend(int(i) for i in chosen)

    # Singletons are only used as negatives.
    if SINGLETON_NEGATIVES and singleton:
        chosen_ids = rng.choice(singleton,
                                size=min(SINGLETON_NEGATIVES, len(singleton)),
                                replace=False)
        indices.extend(int(pools[identity][0]) for identity in chosen_ids)
    return indices


def make_batches(dataset, steps, epoch):
    pools, multi, singleton = identity_indices(dataset)
    rng = np.random.default_rng([SEED, epoch])
    return [sample_batch_indices(pools, multi, singleton, rng) for _ in range(steps)]


def check_identity_isolation(dataset, stage, held_out):
    rows = dataset.samples
    train_ids = {r["identity"] for r in rows}
    if stage == "view1":
        validation = read_manifest("view1_validation_pairs.csv")
        held_ids = {path.split("/")[0] for r in validation
                    for path in (r["img1"], r["img2"])}
    else:
        if any(int(r["fold"]) in held_out for r in rows):
            raise RuntimeError("LEAKAGE: training dataset includes held-out fold")
        # Keep each held-out fold's full identity set together.
        held_ids = {r["identity"] for r in read_manifest("view2_images.csv")
                    if int(r["fold"]) in held_out}
    overlap = train_ids & held_ids
    if overlap:
        raise RuntimeError(f"LEAKAGE: {len(overlap)} evaluation identities in training; "
                           f"examples {sorted(overlap)[:3]}")
    pools, multi, singleton = identity_indices(dataset)
    return {"training_images": len(rows), "training_identities": len(pools),
            "multi_image_identities": len(multi), "singleton_identities": len(singleton),
            "held_out_identity_overlap": 0}


# Triplet mining and loss.
def mine_triplets(embeddings, labels, image_paths, margin, method, anchor_count):
    """Return triplet loss and mining stats."""
    z = F.normalize(embeddings, p=2, dim=1)
    labels = labels.to(z.device)
    with torch.no_grad():
        dist = (2 - 2 * z.detach() @ z.detach().T).clamp_min(0)
        n = len(labels)
        same = labels[:, None] == labels[None, :]
        different_image = torch.tensor(
            [[image_paths[i] != image_paths[j] for j in range(n)] for i in range(n)],
            dtype=torch.bool, device=z.device)
        pos_mask = same & different_image
        neg_mask = ~same

        aa, pp, nn = [], [], []
        possible = 0
        for anchor in range(min(anchor_count, n)):
            positives = torch.where(pos_mask[anchor])[0]
            negatives = torch.where(neg_mask[anchor])[0]
            if len(positives) == 0 or len(negatives) == 0:
                continue

            if method == "batch_hard":
                possible += 1
                positive = positives[torch.argmax(dist[anchor, positives])]
                negative = negatives[torch.argmin(dist[anchor, negatives])]
                aa.append(anchor)
                pp.append(int(positive))
                nn.append(int(negative))
            elif method == "semi_hard":
                for positive in positives:
                    possible += 1
                    dp = dist[anchor, positive]
                    dn = dist[anchor, negatives]
                    valid = (dn > dp) & (dn < dp + margin)
                    if valid.any():
                        candidates = negatives[valid]
                        negative = candidates[torch.argmin(dist[anchor, candidates])]
                        aa.append(anchor)
                        pp.append(int(positive))
                        nn.append(int(negative))
            else:
                raise ValueError(method)

    if not aa:
        # Return a zero loss tensor so the caller can skip the update cleanly.
        return z.sum() * 0, {"eligible": possible, "mined": 0,
                              "active": 0, "mean_ap2": None, "mean_an2": None}

    ap2 = (z[aa] - z[pp]).square().sum(dim=1)
    an2 = (z[aa] - z[nn]).square().sum(dim=1)
    per_triplet = F.relu(ap2 - an2 + margin)
    return per_triplet.mean(), {
        "eligible": possible, "mined": len(aa),
        "active": int((per_triplet.detach() > 0).sum().item()),
        "mean_ap2": float(ap2.detach().mean()),
        "mean_an2": float(an2.detach().mean()),
    }


# Training and validation flow.
def plot_history(run_dir, history, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ep = [r["epoch"] for r in history]
    ax[0].plot(ep, [r["train_loss"] for r in history]); ax[0].set(xlabel="epoch", title="Triplet loss")
    ax[1].plot(ep, [r["active_candidate_fraction"] for r in history], label="mined / eligible")
    ax[1].plot(ep, [r["active_mined_fraction"] for r in history], label="active / mined")
    ax[1].set(xlabel="epoch", ylim=(0, 1.05), title="Triplet mining activity"); ax[1].legend()
    if history[0]["validation_accuracy"] != "":
        ax[2].plot(ep, [r["validation_accuracy"] for r in history], label="accuracy")
        ax[2].plot(ep, [r["validation_auc"] for r in history], label="AUC")
        ax[2].legend(); ax[2].set(xlabel="epoch", title="View 1 validation")
    else:
        ax[2].axis("off")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=140)
    plt.close(fig)


def train_one_run(run_dir, stage, method, margin, learning_rate, epochs,
                  held_out=(), run_number=None, smoke=False):
    run_dir = Path(run_dir)
    complete = run_dir / "run_complete.json"

    seed_everything(SEED)
    device = get_device()
    held_out = tuple(held_out)
    train_dataset = (LFWIdentityDataset(split="view1_train", train=True)
                     if stage == "view1" else
                     LFWIdentityDataset(split="view2", held_out_folds=list(held_out), train=True))
    info = check_identity_isolation(train_dataset, stage, held_out)
    steps = SMOKE_STEPS if smoke else (VIEW1_STEPS_PER_EPOCH if stage == "view1"
                                      else VIEW2_STEPS_PER_EPOCH)
    validation = None
    if stage == "view1":
        validation = LFWPairDataset(split="view1_validation", train=False)
        if smoke:
            n = SMOKE_VAL_PAIRS // 2
            validation.samples = validation.samples[:n] + validation.samples[-n:]

    model = FaceEmbeddingNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  betas=ADAMW_BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    params = {"method": method, "margin": margin, "lr": learning_rate,
              "epochs": epochs, "seed": SEED, "stage": stage, "held_out": list(held_out),
              "P": P_IDENTITIES, "K": K_IMAGES, "extra_singletons": SINGLETON_NEGATIVES,
              "steps_per_epoch": steps, "smoke": smoke,
              "weight_decay": WEIGHT_DECAY, "adamw_betas": list(ADAMW_BETAS),
              "plateau_rule": ({"min_epochs": VIEW1_MIN_EPOCHS,
                                "patience": VIEW1_PLATEAU_PATIENCE,
                                "min_improvement": VIEW1_MIN_IMPROVEMENT}
                               if stage == "view1" and not smoke else None)}
    config = {
        "model": "Triplet Loss (FaceNet) with online mining",
        "variant": method, "stage": stage, "run_number": run_number,
        "held_out_folds": list(held_out),
        "training_scheme": "View 1: peopleDevTrain; View 2: five runs, each holding out 2 folds",
        "architecture": "Common.base_model.FaceEmbeddingNet, random ResNet-18, 512-D",
        "trainable_parameters": count_trainable_parameters(model),
        "loss": "relu(||a-p||^2 - ||a-n||^2 + margin) on L2-normalized features",
        "mining": {"method": method, "margin": margin,
                   "valid_positive": "different image of same person",
                   "singleton_images": "negative only"},
        "sampling": {"P": P_IDENTITIES, "K": K_IMAGES,
                     "extra_singleton_negatives": SINGLETON_NEGATIVES,
                     "images_per_batch": P_IDENTITIES * K_IMAGES + SINGLETON_NEGATIVES,
                     "sampling_seed": "np.default_rng([SEED, epoch])"},
        "optimizer": {"type": "AdamW", "learning_rate": learning_rate,
                      "weight_decay": WEIGHT_DECAY, "betas": ADAMW_BETAS},
        "scheduler": {"type": "CosineAnnealingLR", "epochs": epochs},
        "epochs": epochs, "steps_per_epoch": steps,
        "normalization": "112x112 RGB, (pixel-127.5)/128, random horizontal flip",
        "test_time": "sum original+flipped features, L2 normalize",
        "validation": ("View 1: best threshold on 1000 dev pairs, AUC; optimistic tuning metric"
                       if stage == "view1" else "no View 2 validation; held-out folds scored only"),
        "training_data": info, "hardware": hardware_info(device),
        "software": software_info(), "settings_key": params,
    }
    # Don't reuse a completed run if the settings changed.
    if complete.exists():
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise RuntimeError(f"Completed run has no config.json: {run_dir}")
        saved_config = json.loads(config_path.read_text())
        saved_summary = json.loads(complete.read_text())
        if (saved_config.get("settings_key") != params
                or saved_summary.get("settings_key") != params):
            raise RuntimeError(
                f"Completed run settings differ: {run_dir}\n"
                "Move/archive this run's results before launching a different "
                "configuration. Refusing to reuse them."
            )
        required = [run_dir / "training_history.csv"]
        if stage == "view2":
            required.append(run_dir / "heldout_scores.csv")
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Completed run is missing output files: {run_dir}")
        print("Skipping matching completed run:", run_dir)
        return saved_summary

    history = []
    total_updates = 0
    checkpoint = ckpt_path(run_dir)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["settings"] != params:
            raise RuntimeError("Resume settings changed; restore settings or move the checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        history = state["history"]
        total_updates = state["total_updates"]
        print(f"Resuming {method} at epoch {len(history)+1}")
    else:
        save_json(run_dir / "config.json", config)

    print(f"\n{run_dir.relative_to(MODEL_DIR)} | data={info} | device={device}", flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        batches = make_batches(train_dataset, steps, epoch)
        loader = DataLoader(train_dataset, batch_sampler=batches, num_workers=NUM_WORKERS)
        model.train()
        monitor = TrainingMonitor(device)
        monitor.start()
        lr_at_epoch = optimizer.param_groups[0]["lr"]
        loss_sum, seen_steps, all_eligible, all_mined, all_active = 0., 0, 0, 0, 0
        weighted_ap, weighted_an = 0., 0.
        batch_log = []

        for ids, (images, labels) in zip(batches, loader):
            paths = [train_dataset.samples[i]["image_path"] for i in ids]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(images, normalize=True)
            loss, diag = mine_triplets(z, labels, paths, margin, method,
                                       P_IDENTITIES * K_IMAGES)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite triplet loss in epoch {epoch}")
            if diag["active"]:  # skip updates when no triplet violates the margin
                loss.backward()
                optimizer.step()
                total_updates += 1
            # Count mined triplets even when the loss is zero.
            if diag["mined"]:
                weighted_ap += diag["mean_ap2"] * diag["mined"]
                weighted_an += diag["mean_an2"] * diag["mined"]
            all_eligible += diag["eligible"]
            all_mined += diag["mined"]
            all_active += diag["active"]
            loss_value = float(loss.detach())
            loss_sum += loss_value
            seen_steps += 1
            batch_log.append({
                "epoch": epoch, "batch": seen_steps, "loss": loss_value,
                "eligible_anchor_positive_pairs": diag["eligible"],
                "mined_triplets": diag["mined"], "active_triplets": diag["active"],
                "fraction_eligible_mined": diag["mined"] / max(1, diag["eligible"]),
                "fraction_mined_active": diag["active"] / max(1, diag["mined"]),
                "mean_positive_squared_distance": diag["mean_ap2"],
                "mean_negative_squared_distance": diag["mean_an2"],
            })
            monitor.sample_memory()

        scheduler.step()
        monitor.stop()
        report = monitor.report()
        row = {"epoch": epoch, "train_loss": loss_sum / seen_steps,
               "validation_accuracy": "", "validation_auc": "", "validation_threshold": "",
               "learning_rate": lr_at_epoch, "epoch_time_seconds": report["elapsed_seconds"],
               "validation_time_seconds": "", "eligible_anchor_positive_pairs": all_eligible,
               "mined_triplets": all_mined, "active_triplets": all_active,
               "active_candidate_fraction": all_mined / max(1, all_eligible),
               "active_mined_fraction": all_active / max(1, all_mined),
               "mean_positive_squared_distance": weighted_ap / all_mined if all_mined else "",
               "mean_negative_squared_distance": weighted_an / all_mined if all_mined else "",
               "steps": seen_steps, "optimizer_updates_total": total_updates,
               "peak_sampled_mps_driver_mib": report.get("peak_sampled_mps_driver_mib", ""),
               "peak_gpu_allocated_mib": report.get("peak_gpu_allocated_mib", "")}

        if validation is not None:
            t0 = time.perf_counter()
            scored = score_pair_dataset(model, validation, device, batch_size=32)
            y = np.asarray([r["label"] for r in scored]); s = np.asarray([r["score"] for r in scored])
            threshold = select_accuracy_threshold(s, y)
            row.update(validation_accuracy=float(np.mean((s >= threshold) == y)),
                       validation_auc=float(roc_auc_score(y, s)),
                       validation_threshold=threshold,
                       validation_time_seconds=round(time.perf_counter() - t0, 2))
            # Keep each epoch's validation metrics.
            write_csv(run_dir / "validation_scores" / f"epoch_{epoch:03d}.csv",
                      [{"epoch": epoch, **r} for r in scored])

        if stage == "view1" and not smoke:
            status = view1_plateau(history + [row])
            row["view1_plateau_reached"] = status["reached"]
            row["epochs_without_val_improvement"] = status["epochs_without_improvement"]

        # Log active triplets per batch.
        write_csv(run_dir / "mining_batches" / f"epoch_{epoch:03d}.csv", batch_log)
        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({"settings": params, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                    "history": history, "total_updates": total_updates}, tmp)
        tmp.replace(checkpoint)
        print(f"epoch {epoch:2d}/{epochs}: loss={row['train_loss']:.4f}, "
              f"mined/eligible={row['active_candidate_fraction']:.3f}, "
              f"active/mined={row['active_mined_fraction']:.3f}, "
              f"val_acc={row['validation_accuracy']}, time={row['epoch_time_seconds']:.1f}s",
              flush=True)
        if stage == "view1" and not smoke and status["reached"]:
            print(f"View 1 plateau reached after epoch {epoch} "
                  f"({VIEW1_PLATEAU_PATIENCE} epochs without an improvement "
                  f"> {VIEW1_MIN_IMPROVEMENT:g}).")
            break

    if stage == "view2":
        heldout_scores = []
        for fold in held_out:
            dataset = LFWPairDataset(split="view2", fold=fold, train=False)
            if smoke:
                # Keep the class split roughly balanced when possible.
                n = SMOKE_FOLD_PAIRS // 2
                dataset.samples = dataset.samples[:n] + dataset.samples[-n:]
            heldout_scores.extend(score_pair_dataset(model, dataset, device, batch_size=32))
        write_csv(run_dir / "heldout_scores.csv", heldout_scores)
        if not smoke and EMBEDDINGS_PER_FOLD > 0:
            save_embedding_samples(model, heldout_scores, held_out, device, run_dir)

    validation_rows = [h for h in history if h["validation_accuracy"] != ""]
    best = max(validation_rows, key=lambda h: h["validation_accuracy"]) if validation_rows else None
    memory = [r["peak_sampled_mps_driver_mib"] for r in history
              if r["peak_sampled_mps_driver_mib"] not in ("", None)]
    gpu_memory = [
                float(r["peak_gpu_allocated_mib"])
                for r in history
                if r.get("peak_gpu_allocated_mib") not in ("", None)
            ]
    plateau = view1_plateau(history) if stage == "view1" and not smoke else None
    summary = {"variant": method, "margin": margin, "learning_rate": learning_rate,
               "stage": stage, "held_out_folds": list(held_out), "seed": SEED,
               "settings_key": params, "max_epoch_budget": epochs,
               "view1_plateau": plateau,
               "epochs_completed": len(history), "optimizer_updates": total_updates,
               "training_seconds": sum(r["epoch_time_seconds"] for r in history),
               "validation_seconds": sum(r["validation_time_seconds"] or 0 for r in history),
               "final_train_loss": history[-1]["train_loss"],
               "final_validation_accuracy": validation_rows[-1]["validation_accuracy"] if best else None,
               "final_validation_auc": validation_rows[-1]["validation_auc"] if best else None,
               "best_validation_accuracy": best["validation_accuracy"] if best else None,
               "best_validation_epoch": best["epoch"] if best else None,
               "peak_sampled_mps_driver_mib": max(memory) if memory else None,
               "peak_gpu_allocated_mib": max(gpu_memory) if gpu_memory else None,
               "training_data": info, "device": str(device)}
    plot_history(run_dir, history, f"Triplet {method}, margin={margin}, {stage}")
    if stage == "view2" and not smoke:
        weights_dir = MODEL_DIR / "saved_models" / method
        weights_dir.mkdir(parents=True, exist_ok=True)

        weights_path = weights_dir / f"run_{run_number}.pt"
        temp_path = weights_path.with_suffix(".pt.tmp")

        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "architecture": "Common.base_model.FaceEmbeddingNet",
                "mining_method": method,
                "margin": margin,
                "learning_rate": learning_rate,
                "epochs": len(history),
                "seed": SEED,
                "run_number": run_number,
                "held_out_folds": list(held_out),
            },
            temp_path,
        )

        temp_path.replace(weights_path)

        summary["saved_weights"] = str(weights_path.relative_to(MODEL_DIR))
        print(f"Saved final model: {weights_path}", flush=True)
    save_json(complete, summary)
    if plateau is not None and not plateau["reached"]:
        print(f"WARNING: View 1 accuracy has not plateaued by the maximum of "
              f"{epochs} epochs ({run_dir}). View 2 will refuse to use "
              "this configuration until convergence is established.")
    # Keep completed View 2 checkpoints.
    if stage == "view2" and not smoke:
        print(f"Training checkpoint preserved: {checkpoint}", flush=True)
    else:
        checkpoint.unlink(missing_ok=True)

    return summary


# Local embedding samples for plots.
@torch.inference_mode()
def save_embedding_samples(model, scored, folds, device, run_dir):
    """Save held-out embeddings for plotting."""
    transform = get_eval_transform()
    for fold in folds:
        paths = sorted({p for r in scored if int(r["fold"]) == fold
                        for p in (r["img1"], r["img2"])})
        by_identity = defaultdict(list)
        for path in paths:
            by_identity[path.split("/")[0]].append(path)
        # Prefer identities with multiple examples for better clustering.
        rng = np.random.default_rng([SEED, fold])
        multi = [i for i in sorted(by_identity) if len(by_identity[i]) >= 2]
        singles = [i for i in sorted(by_identity) if len(by_identity[i]) == 1]
        selected = []
        for identity in rng.permutation(multi):
            if len(selected) + 2 > EMBEDDINGS_PER_FOLD:
                break
            selected += by_identity[identity][:min(4, len(by_identity[identity]),
                                               EMBEDDINGS_PER_FOLD-len(selected))]
        for identity in rng.permutation(singles):
            if len(selected) >= EMBEDDINGS_PER_FOLD:
                break
            selected += by_identity[identity]
        if not selected:
            continue
        features = []
        model.eval()
        for i in range(0, len(selected), 32):
            tensors = []
            for path in selected[i:i+32]:
                with Image.open(IMAGE_ROOT / path) as image:
                    tensors.append(transform(image.convert("RGB")))
            features.append(test_time_embeddings(model, torch.stack(tensors).to(device)).cpu().numpy())
        np.savez_compressed(run_dir / f"fold_{fold}_embeddings_LOCAL.npz",
                            paths=np.asarray(selected),
                            identities=np.asarray([p.split('/')[0] for p in selected]),
                            embeddings=np.concatenate(features))


# View 1 search and selection.
def write_tuning_summary():
    rows = []
    for file in sorted((RESULTS_DIR / "View_1").glob("*/*/run_complete.json")):
        s = json.loads(file.read_text())
        rows.append({"variant": s["variant"], "margin": s["margin"],
                     "learning_rate": s["learning_rate"], "epochs": s["epochs_completed"],
                     "max_epoch_budget": s["max_epoch_budget"],
                     "view1_plateau_reached": s["view1_plateau"]["reached"],
                     "final_validation_accuracy": s["final_validation_accuracy"],
                     "final_validation_auc": s["final_validation_auc"],
                     "best_validation_accuracy": s["best_validation_accuracy"],
                     "best_validation_epoch": s["best_validation_epoch"],
                     "training_seconds": s["training_seconds"], "run_folder": file.parent.name})
    rows.sort(key=lambda r: (r["variant"], -r["final_validation_accuracy"],
                             -r["final_validation_auc"]))
    ranks = Counter()
    for row in rows:
        ranks[row["variant"]] += 1
        row["rank_within_variant"] = ranks[row["variant"]]
    if rows:
        write_csv(RESULTS_DIR / "View_1" / "tuning_summary.csv", rows)


def run_view1(smoke=False):
    if smoke:
        for method in MINING_METHODS:
            train_one_run(SMOKE_DIR / "View_1" / method, "view1", method, MARGINS[0],
                          LEARNING_RATES[0], SMOKE_EPOCHS, smoke=True)
        return
    for method in MINING_METHODS:
        for lr in LEARNING_RATES:
            for margin in MARGINS:
                out = RESULTS_DIR / "View_1" / method / f"margin_{margin}_lr_{lr:g}"
                train_one_run(out, "view1", method, margin, lr, EPOCHS)
                write_tuning_summary()


def selected_hyperparameters(method):
    path = RESULTS_DIR / "View_1" / "tuning_summary.csv"
    if not path.exists():
        raise RuntimeError("Run STAGE='view1_tuning' before View 2")
    rows = [r for r in read_csv(path) if r["variant"] == method]
    if len(rows) != len(MARGINS) * len(LEARNING_RATES):
        raise RuntimeError(f"Incomplete View 1 tuning for {method}: {len(rows)} configurations")
    best = min(rows, key=lambda r: int(r["rank_within_variant"]))
    if int(best["max_epoch_budget"]) != EPOCHS:
        raise RuntimeError("View 1 selection and current EPOCHS budget differ; "
                           "do not silently mix experimental settings")
    if str(best["view1_plateau_reached"]).lower() != "true":
        print(
            f"Best View 1 {method} configuration did not "
            f"reach the defined plateau criterion within "
            f"{best['epochs']} epochs. "
            "Proceeding with the best configuration under the "
            "fixed training budget. "
            "This will be documented in the report.",
            flush=True,
        )   
    return {"margin": float(best["margin"]), "learning_rate": float(best["learning_rate"]),
            "epochs": int(best["epochs"]), "view1_selection": best}


# View 2 aggregation and outputs.
def fold_table(rows, metrics, key, higher_is_same):
    output = []
    for fold in metrics["folds"]:
        sub = [r for r in rows if int(r["fold"]) == int(fold["fold"])]
        t = fold["threshold"]
        correct = sum(((r[key] >= t) if higher_is_same else (r[key] <= t)) == bool(r["label"])
                      for r in sub)
        if not math.isclose(correct / len(sub), fold["accuracy"]):
            raise RuntimeError("Fold threshold/accuracy mismatch")
        output.append({"fold": fold["fold"], "threshold": t, "correct": correct,
                       "n_pairs": len(sub), "accuracy": fold["accuracy"]})
    return output


def save_histograms(rows, cosine_table, squared_table, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for axis, key, folds, title in [(ax[0], "score", cosine_table, "Cosine similarity"),
                                     (ax[1], "squared_l2", squared_table, "Squared Euclidean")]:
        axis.hist([r[key] for r in rows if r["label"] == 1], bins=50, alpha=.6, label="genuine")
        axis.hist([r[key] for r in rows if r["label"] == 0], bins=50, alpha=.6, label="impostor")
        ts = [r["threshold"] for r in folds]
        axis.axvline(np.mean(ts), color="black", ls="--", label="mean fold threshold")
        axis.axvspan(min(ts), max(ts), color="gray", alpha=.15, label="fold threshold range")
        axis.set(xlabel=title, ylabel="number of pairs"); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)


def aggregate_view2(variant_dir, method, selection, summaries):
    rows, owners = [], {}
    for run_number, folds in enumerate(VIEW2_HELD_OUT, 1):
        file = variant_dir / f"run_{run_number}" / "heldout_scores.csv"
        parsed = read_csv(file)
        if Counter(int(r["fold"]) for r in parsed) != Counter({folds[0]:600, folds[1]:600}):
            raise RuntimeError(f"Run {run_number} did not score precisely {folds}")
        for fold in folds:
            owners[fold] = run_number
        rows += [{"fold": int(r["fold"]), "img1": r["img1"], "img2": r["img2"],
                  "label": int(r["label"]), "score": float(r["score"]),
                  "squared_l2": float(r["squared_l2"])} for r in parsed]

    expected = Counter((int(r["fold"]), r["img1"], r["img2"], int(r["label"]))
                       for r in read_manifest("view2_pairs.csv"))
    actual = Counter((r["fold"], r["img1"], r["img2"], r["label"]) for r in rows)
    if len(rows) != 6000 or actual != expected:
        raise RuntimeError("Pair-score manifest mismatch: View 2 must have exactly the 6000 official pairs")
    cosine = evaluate_view2(rows, score_key="score", higher_is_same=True)
    squared = evaluate_view2(rows, score_key="squared_l2", higher_is_same=False)
    ct = fold_table(rows, cosine, "score", True)
    st = fold_table(rows, squared, "squared_l2", False)
    save_pair_scores(rows, variant_dir / "pair_scores.csv")
    save_pair_scores(rows, variant_dir / "pair_scores_squared_l2.csv", score_key="squared_l2")
    write_csv(variant_dir / "fold_accuracy.csv", ct)
    write_csv(variant_dir / "fold_accuracy_squared_l2.csv", st)

    labels = np.asarray([r["label"] for r in rows])
    sim = np.asarray([r["score"] for r in rows])
    dist = np.asarray([r["squared_l2"] for r in rows])
    error_cos_l2 = float(np.max(np.abs(dist - (2 - 2 * sim))))
    for scores, path, sign in [(sim, "roc_data.csv", 1), (-dist, "roc_data_squared_l2.csv", -1)]:
        far, tar, thresholds = roc_curve(labels, scores, drop_intermediate=False)
        write_csv(variant_dir / path,
                  [{"far": f, "tar": t, "threshold": th * sign}
                   for f, t, th in zip(far, tar, thresholds)])
    save_roc_plot({f"Triplet {method} cosine": (sim, labels),
                   f"Triplet {method} squared L2 (negated)": (-dist, labels)},
                  variant_dir / "roc.png")
    save_histograms(rows, ct, st, variant_dir / "score_histogram.png")

    fold_thresholds = {int(r["fold"]): float(r["threshold"]) for r in ct}
    errors = []
    for r in rows:
        threshold = fold_thresholds[r["fold"]]
        same = r["score"] >= threshold
        if same != bool(r["label"]):
            errors.append({"error_type": "false_accept" if same else "false_reject",
                           "confidence": abs(r["score"]-threshold), "fold": r["fold"],
                           "img1": r["img1"], "img2": r["img2"], "label": r["label"],
                           "cosine_score": r["score"], "squared_l2": r["squared_l2"],
                           "threshold": threshold, "scored_by_run": owners[r["fold"]]})
    errors.sort(key=lambda r: (r["error_type"], -r["confidence"]))
    write_csv(variant_dir / "confident_errors.csv", errors,
              fields=["error_type", "confidence", "fold", "img1", "img2", "label",
                      "cosine_score", "squared_l2", "threshold", "scored_by_run"])

    def extract(m):
        return {k:m[k] for k in ("accuracy_mean", "accuracy_std", "auc", "eer", "tar_at_far_0.01")}
    times = [s["training_seconds"] for s in summaries]
    memory = [s["peak_sampled_mps_driver_mib"] for s in summaries
              if s["peak_sampled_mps_driver_mib"] is not None]
    save_json(variant_dir / "summary.json", {
        "model": "Triplet Loss", "mining": method, "seed": SEED,
        "training_runs": 5, "evaluated_folds": 10,
        "held_out_pairs": VIEW2_HELD_OUT, "fold_to_run": owners,
        "selection_from_view1": selection["view1_selection"],
        "frozen_hyperparameters": {"margin": selection["margin"],
                                   "learning_rate": selection["learning_rate"],
                                   "epochs": selection["epochs"]},
        "cosine": extract(cosine), "squared_l2": extract(squared),
        "max_abs_squared_l2_minus_2_minus_2cos": error_cos_l2,
        "training_total_seconds": sum(times), "training_mean_seconds": float(np.mean(times)),
        "epochs_per_run": [s["epochs_completed"] for s in summaries],
        "peak_sampled_mps_driver_mib": max(memory) if memory else None,
        "memory_note": "MPS value sampled once per batch, not guaranteed hardware peak",
        "hardware": hardware_info(get_device()), "software": software_info(),
        "runs": summaries,
        "files": ["pair_scores.csv", "pair_scores_squared_l2.csv", "fold_accuracy.csv",
                  "fold_accuracy_squared_l2.csv", "roc_data.csv", "roc.png",
                  "score_histogram.png", "confident_errors.csv"],
    })
    print(f"\nTriplet {method}: acc={cosine['accuracy_mean']:.4f} ± "
          f"{cosine['accuracy_std']:.4f}, AUC={cosine['auc']:.4f}, "
          f"EER={cosine['eer']:.4f}, TAR@FAR=1%={cosine['tar_at_far_0.01']:.4f}",
          flush=True)


def run_view2(smoke=False):
    root = SMOKE_DIR if smoke else RESULTS_DIR
    comparison = []
    for method in MINING_METHODS:
        selection = ({"margin": MARGINS[0], "learning_rate": LEARNING_RATES[0],
                      "epochs": SMOKE_EPOCHS, "view1_selection": None}
                     if smoke else selected_hyperparameters(method))
        summaries = []
        for number, held_out in enumerate(VIEW2_HELD_OUT, 1):
            # Smoke mode only needs one run.
            if smoke and number > 1:
                break
            path = root / "View_2" / method / f"run_{number}"
            summaries.append(train_one_run(path, "view2", method, selection["margin"],
                                           selection["learning_rate"], selection["epochs"],
                                           held_out=held_out, run_number=number, smoke=smoke))
        if not smoke:
            variant_dir = root / "View_2" / method
            aggregate_view2(variant_dir, method, selection, summaries)
            report = json.loads((variant_dir / "summary.json").read_text())
            comparison.append({"variant": method, "margin": selection["margin"],
                               "learning_rate": selection["learning_rate"],
                               "accuracy_mean": report["cosine"]["accuracy_mean"],
                               "accuracy_std": report["cosine"]["accuracy_std"],
                               "auc": report["cosine"]["auc"],
                               "eer": report["cosine"]["eer"],
                               "tar_at_far_0.01": report["cosine"]["tar_at_far_0.01"],
                               "training_total_seconds": report["training_total_seconds"]})
    if comparison:
        write_csv(root / "View_2" / "mining_comparison.csv", comparison)


if __name__ == "__main__":
    if STAGE == "smoke_test":
        run_view1(smoke=True)
        run_view2(smoke=True)
    elif STAGE == "view1_tuning":
        run_view1()
    elif STAGE == "view2_final":
        run_view2()
    else:
        raise ValueError(f"Invalid STAGE: {STAGE}")
