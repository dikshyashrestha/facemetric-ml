"""ArcFace training pipeline for the LFW project."""

import csv
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep import order consistent on macOS.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import sklearn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader

from Common.base_model import FaceEmbeddingNet
from Common.dataset_loader import LFWIdentityDataset, LFWPairDataset, read_manifest
from Common.dataset_transforms import get_eval_transform
from Common.model_evaluation import (
    evaluate_view2, save_pair_scores, save_roc_plot,
    score_pair_dataset, select_accuracy_threshold, test_time_embeddings,
)
from Common.model_utils import (
    TrainingMonitor, count_trainable_parameters, get_device, hardware_info,
    save_json, seed_everything,
)


# Training config.
STAGE = "view2_final"  # "smoke_test", "view1_tuning", "view2_final"
SCALES = [16, 30, 64]                 # Scale sweep
MARGINS = [0.2, 0.3, 0.5]             # Margin sweep
LEARNING_RATES = [1e-4, 3e-4]        # LR sweep
ABLATED_MARGIN = 0.0                 # Zero-margin baseline
SEED = 42
EPOCHS = 30                        # Max View 1 epochs
VIEW1_MIN_EPOCHS = 20
VIEW1_PLATEAU_PATIENCE = 6
VIEW1_MIN_IMPROVEMENT = 0.002         # Min validation gain
WEIGHT_DECAY = 5e-4
ADAMW_BETAS = (0.9, 0.999)

# Batch setup for the ArcFace classifier.
P_IDENTITIES = 8
K_IMAGES = 4                        # Images per ID
NUM_WORKERS = 0
VIEW1_STEPS_PER_EPOCH = math.ceil(2200 / 16)  # Steps per epoch
VIEW2_STEPS_PER_EPOCH = math.ceil(4800 / 16)  # Steps per epoch
VIEW2_HELD_OUT = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

SMOKE_EPOCHS = 1
SMOKE_STEPS = 3
SMOKE_VAL_PAIRS = 64
SMOKE_FOLD_PAIRS = 32
EMBEDDINGS_PER_FOLD = 128  # Local plot samples

MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODEL_DIR / "results"
SMOKE_DIR = MODEL_DIR / "outputs" / "smoke_test"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
IMAGE_ROOT = PROJECT_ROOT / "Dataset" / "Processed" / "deepfunneled_112"


# Output helpers.
def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def software_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        commit = None
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(),
            "git_commit": commit}


def checkpoint_path(run_dir):
    return CHECKPOINT_DIR / Path(run_dir).relative_to(MODEL_DIR) / "last_epoch.pt"


def view1_plateau(history):
    """Check whether the View 1 run has plateaued."""
    best = -float("inf")
    last_improvement = 0
    for row in history:
        if row["validation_accuracy"] == "":
            continue
        score = float(row["validation_accuracy"])
        if score > best + VIEW1_MIN_IMPROVEMENT:
            best, last_improvement = score, int(row["epoch"])
    n = int(history[-1]["epoch"]) if history else 0
    stale = n - last_improvement
    return {"reached": bool(n >= VIEW1_MIN_EPOCHS and stale >= VIEW1_PLATEAU_PATIENCE),
            "last_meaningful_improvement_epoch": last_improvement,
            "epochs_without_improvement": stale,
            "minimum_epochs": VIEW1_MIN_EPOCHS,
            "patience": VIEW1_PLATEAU_PATIENCE,
            "minimum_accuracy_improvement": VIEW1_MIN_IMPROVEMENT}


# ArcFace head.
class ArcFaceHead(nn.Module):
    """ArcFace margin head for classification training."""

    def __init__(self, embedding_dim, num_classes, scale, margin):
        super().__init__()
        if num_classes < 2 or scale <= 0 or not (0 <= margin < math.pi / 2):
            raise ValueError("Invalid ArcFace classes, scale, or margin")
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        features = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(features, weights).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.margin == 0:
            logits = cosine * self.scale
        else:
            target_cos = cosine.gather(1, labels.view(-1, 1))
            sin_theta = torch.sqrt((1 - target_cos.square()).clamp_min(1e-7))
            phi = target_cos * math.cos(self.margin) - sin_theta * math.sin(self.margin)
            threshold = math.cos(math.pi - self.margin)
            phi = torch.where(
                target_cos > threshold, phi,
                target_cos - self.margin * math.sin(self.margin),
            )
            logits = (cosine.scatter(1, labels.view(-1, 1), phi)) * self.scale
        return logits


# Training dataset setup.
class ArcFaceTrainingDataset(Dataset):
    """Dataset with only eligible identities."""

    def __init__(self, base):
        self.base = base
        pools = defaultdict(list)
        for index, row in enumerate(base.samples):
            pools[row["identity"]].append(index)
        self.singletons = sum(len(indices) == 1 for indices in pools.values())
        self.classes = sorted(name for name, indices in pools.items() if len(indices) >= 2)
        if len(self.classes) < P_IDENTITIES:
            raise RuntimeError("Insufficient multi-image identities for P×K sampling")
        self.class_to_label = {name: i for i, name in enumerate(self.classes)}
        self.indices = [idx for name in self.classes for idx in pools[name]]
        self.samples = [base.samples[i] for i in self.indices]
        self.by_class = defaultdict(list)
        for new_index, row in enumerate(self.samples):
            self.by_class[self.class_to_label[row["identity"]]].append(new_index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, _ = self.base[original_index]
        identity = self.samples[index]["identity"]
        return image, self.class_to_label[identity]


def check_training_data(base_dataset, dataset, stage, held_out):
    train_ids = set(dataset.class_to_label)
    if stage == "view1":
        val = read_manifest("view1_validation_pairs.csv")
        held_ids = {p.split("/")[0] for row in val for p in (row["img1"], row["img2"])}
    else:
        if any(int(row["fold"]) in held_out for row in base_dataset.samples):
            raise RuntimeError("LEAKAGE: held-out fold included in training")
        held_ids = {row["identity"] for row in read_manifest("view2_images.csv")
                    if int(row["fold"]) in held_out}
    overlapping = train_ids & held_ids
    if overlapping:
        raise RuntimeError(f"LEAKAGE: {len(overlapping)} held-out identities in training")
    return {"eligible_identity_classes": len(dataset.classes),
            "eligible_training_images": len(dataset),
            "excluded_singleton_identities": dataset.singletons,
            "held_out_identity_overlap": 0}


def sample_batches(dataset, steps, epoch):
    """Build balanced ArcFace batches."""
    rng = np.random.default_rng([SEED, epoch])
    ids = np.array(sorted(dataset.by_class))
    batches = []
    for _ in range(steps):
        selected = rng.choice(ids, size=P_IDENTITIES, replace=False)
        indices = []
        for label in selected:
            pool = dataset.by_class[int(label)]
            first = rng.choice(pool, size=min(K_IMAGES, len(pool)), replace=False).tolist()
            if len(first) < K_IMAGES:
                first += rng.choice(pool, size=K_IMAGES-len(first), replace=True).tolist()
            indices.extend(map(int, first))
        batches.append(indices)
    return batches


# Training and reporting.
def plot_history(run_dir, history, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ep = [row["epoch"] for row in history]
    axes[0].plot(ep, [row["train_loss"] for row in history]); axes[0].set(title="ArcFace cross-entropy", xlabel="epoch")
    axes[1].plot(ep, [row["train_classification_accuracy"] for row in history]); axes[1].set(title="Training class accuracy", xlabel="epoch")
    if history[0]["validation_accuracy"] != "":
        axes[2].plot(ep, [row["validation_accuracy"] for row in history], label="verification accuracy")
        axes[2].plot(ep, [row["validation_auc"] for row in history], label="ROC-AUC")
        axes[2].legend(); axes[2].set(title="View 1 verification", xlabel="epoch")
    else:
        axes[2].axis("off")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=140)
    plt.close(fig)


def train_one_run(run_dir, stage, scale, margin, learning_rate, epochs,
                  held_out=(), run_number=None, smoke=False):
    run_dir = Path(run_dir)
    complete = run_dir / "run_complete.json"
    seed_everything(SEED)
    device = get_device()
    held_out = tuple(held_out)

    base = (LFWIdentityDataset(split="view1_train", train=True)
            if stage == "view1" else
            LFWIdentityDataset(split="view2", held_out_folds=list(held_out), train=True))
    dataset = ArcFaceTrainingDataset(base)
    info = check_training_data(base, dataset, stage, held_out)
    steps = (SMOKE_STEPS if smoke else
             VIEW1_STEPS_PER_EPOCH if stage == "view1" else VIEW2_STEPS_PER_EPOCH)
    validation = None
    if stage == "view1":
        validation = LFWPairDataset(split="view1_validation", train=False)
        if smoke:
            n = SMOKE_VAL_PAIRS // 2
            validation.samples = validation.samples[:n] + validation.samples[-n:]

    model = FaceEmbeddingNet().to(device)
    head = ArcFaceHead(512, len(dataset.classes), scale, margin).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=learning_rate, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS,
    )
    # Keep the LR schedule consistent across View 1 and View 2.
    scheduler_tmax = epochs if smoke else EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_tmax)

    settings = {"stage": stage, "scale": scale, "margin": margin, "lr": learning_rate,
                "epochs": epochs, "scheduler_tmax": scheduler_tmax,
                "seed": SEED, "held_out": list(held_out),
                "P": P_IDENTITIES, "K": K_IMAGES, "steps": steps,
                "weight_decay": WEIGHT_DECAY, "betas": list(ADAMW_BETAS),
                "run_number": run_number, "smoke": smoke,
                "class_count": len(dataset.classes),
                "plateau": {"minimum_epochs": VIEW1_MIN_EPOCHS,
                            "patience": VIEW1_PLATEAU_PATIENCE,
                            "minimum_accuracy_improvement": VIEW1_MIN_IMPROVEMENT}
                if stage == "view1" and not smoke else None}
    config = {
        "model": "ArcFace: additive angular-margin softmax", "stage": stage,
        "lfw_protocol": "unrestricted, labeled outside data",
        "run_number": run_number, "held_out_folds": list(held_out),
        "view2_scheme": "five models; each holds out consecutive two folds",
        "backbone": "Common.base_model.FaceEmbeddingNet, random modified ResNet-18, 512-D",
        "trainable_parameters": count_trainable_parameters(model) + count_trainable_parameters(head),
        "head": {"embedding_dim": 512, "classes": len(dataset.classes),
                 "scale": scale, "angular_margin_radians": margin,
                 "formula": "s*cos(theta_y+m) for target, s*cos(theta_j) otherwise",
                 "stable_fallback": "if cos(theta_y)<cos(pi-m): cos(theta_y)-m*sin(m)"},
        "optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                      "weight_decay": WEIGHT_DECAY, "betas": list(ADAMW_BETAS)},
        "lr_scheduler": {"name": "CosineAnnealingLR", "T_max_epochs": scheduler_tmax},
        "batch": {"P": P_IDENTITIES, "K": K_IMAGES, "images_per_step": P_IDENTITIES*K_IMAGES,
                  "steps_per_epoch": steps,
                  "sampling": "uniform eligible identities, without replacement; images unique then replacement"},
        "augmentation": "training horizontal flip; 112x112 RGB; (pixel-127.5)/128",
        "test_time_embedding": "original plus horizontal flip, then L2 normalization",
        "validation": ("View 1: 1000 development pairs, best-on-same-pairs threshold for tuning only"
                       if stage == "view1" else
                       "View 2: test held-out folds only, thresholds from other 9 folds"),
        "training_data": info, "hardware": hardware_info(device),
        "software": software_info(), "settings_key": settings,
    }

    if complete.exists():
        config_file = run_dir / "config.json"
        if not config_file.exists():
            raise RuntimeError(f"Completed run missing config: {run_dir}")
        saved_config = json.loads(config_file.read_text())
        saved_summary = json.loads(complete.read_text())
        if saved_config.get("settings_key") != settings or saved_summary.get("settings_key") != settings:
            raise RuntimeError(f"Completed run settings changed: {run_dir}. Archive old results first.")
        expected = [run_dir / "training_history.csv"]
        if stage == "view2":
            expected.append(run_dir / "heldout_scores.csv")
        if not all(path.exists() for path in expected):
            raise RuntimeError(f"Completed run missing required output: {run_dir}")
        print("Skipping matching completed run:", run_dir)
        return saved_summary

    history = []
    total_updates = 0
    ckpt = checkpoint_path(run_dir)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise RuntimeError(f"Checkpoint settings changed: {ckpt}; archive old run first")
        model.load_state_dict(state["model"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device=device)
        if device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        history = state["history"]
        total_updates = state["updates"]
        print(f"Resuming {run_dir} after epoch {len(history)}")
    else:
        save_json(run_dir / "config.json", config)

    print(f"\n{run_dir.relative_to(MODEL_DIR)} | ArcFace(s={scale}, m={margin}, lr={learning_rate}) "
          f"| data={info} | device={device}", flush=True)
    for epoch in range(len(history)+1, epochs+1):
        batches = sample_batches(dataset, steps, epoch)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=NUM_WORKERS)
        monitor = TrainingMonitor(device)
        monitor.start()
        model.train(); head.train()
        running_loss, correct, seen = 0.0, 0, 0
        epoch_lr = optimizer.param_groups[0]["lr"]

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images, normalize=False)
            logits = head(embeddings, labels)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            n = labels.size(0)
            running_loss += float(loss.detach()) * n
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += n
            monitor.sample_memory()

        scheduler.step(); monitor.stop()
        report = monitor.report()
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_classification_accuracy": correct / seen,
            "validation_accuracy": "", "validation_auc": "", "validation_threshold": "",
            "validation_time_seconds": "", "learning_rate": epoch_lr,
            "epoch_time_seconds": report["elapsed_seconds"],
            "train_images_seen": seen, "batches": len(batches),
            "optimizer_updates_total": total_updates + len(batches),
            "peak_sampled_mps_driver_mib": report.get("peak_sampled_mps_driver_mib", ""),
            "peak_gpu_allocated_mib": report.get("peak_gpu_allocated_mib", ""),
        }
        total_updates += len(batches)
        if validation is not None:
            start = time.perf_counter()
            scored = score_pair_dataset(model, validation, device, batch_size=32)
            y = np.asarray([r["label"] for r in scored])
            scores = np.asarray([r["score"] for r in scored])
            threshold = select_accuracy_threshold(scores, y)
            row.update(validation_accuracy=float(np.mean((scores >= threshold) == y)),
                       validation_auc=float(roc_auc_score(y, scores)),
                       validation_threshold=float(threshold),
                       validation_time_seconds=round(time.perf_counter()-start, 2))
            write_csv(run_dir / "validation_scores" / f"epoch_{epoch:03d}.csv",
                      [{"epoch": epoch, **r} for r in scored])
        if stage == "view1" and not smoke:
            status = view1_plateau(history + [row])
            row.update(view1_plateau_reached=status["reached"],
                       epochs_without_val_improvement=status["epochs_without_improvement"])

        history.append(row)
        write_csv(run_dir / "training_history.csv", history)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        temp = ckpt.with_suffix(".tmp")
        torch.save({"settings": settings,
                    "model": model.state_dict(), "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                    "history": history, "updates": total_updates}, temp)
        temp.replace(ckpt)
        print(f"epoch {epoch:2d}/{epochs} | loss={row['train_loss']:.4f} "
              f"class_acc={row['train_classification_accuracy']:.3f} "
              f"val_acc={row['validation_accuracy']} "
              f"time={row['epoch_time_seconds']:.1f}s", flush=True)
        if stage == "view1" and not smoke and status["reached"]:
            print(f"View 1 validation plateau reached at epoch {epoch}")
            break

    if stage == "view2":
        heldout = []
        for fold in held_out:
            pair_dataset = LFWPairDataset(split="view2", fold=fold, train=False)
            if smoke:
                n = SMOKE_FOLD_PAIRS // 2
                pair_dataset.samples = pair_dataset.samples[:n] + pair_dataset.samples[-n:]
            heldout.extend(score_pair_dataset(model, pair_dataset, device, batch_size=32))
        write_csv(run_dir / "heldout_scores.csv", heldout)
        if not smoke and EMBEDDINGS_PER_FOLD:
            save_embedding_samples(model, heldout, held_out, device, run_dir)

    valid = [row for row in history if row["validation_accuracy"] != ""]
    best = max(valid, key=lambda row: row["validation_accuracy"]) if valid else None
    memory = [row["peak_sampled_mps_driver_mib"] for row in history
              if row["peak_sampled_mps_driver_mib"] not in (None, "")]
    gpu_memory = [
        float(row["peak_gpu_allocated_mib"])
        for row in history
        if row.get("peak_gpu_allocated_mib") not in ("", None)
    ]
    plateau = view1_plateau(history) if stage == "view1" and not smoke else None
    summary = {
        "model": "ArcFace", "scale": scale, "margin": margin, "learning_rate": learning_rate,
        "stage": stage, "run_number": run_number, "held_out_folds": list(held_out),
        "seed": SEED, "settings_key": settings, "view1_plateau": plateau,
        "max_epoch_budget": epochs, "epochs_completed": len(history),
        "optimizer_updates": total_updates,
        "training_seconds": sum(row["epoch_time_seconds"] for row in history),
        "validation_seconds": sum(row["validation_time_seconds"] or 0 for row in history),
        "final_train_loss": history[-1]["train_loss"],
        "final_validation_accuracy": valid[-1]["validation_accuracy"] if valid else None,
        "final_validation_auc": valid[-1]["validation_auc"] if valid else None,
        "best_validation_accuracy": best["validation_accuracy"] if best else None,
        "best_validation_epoch": best["epoch"] if best else None,
        "peak_sampled_mps_driver_mib": max(memory) if memory else None,
        "peak_gpu_allocated_mib": max(gpu_memory) if gpu_memory else None,
        "training_data": info, "device": str(device),
    }
    plot_history(run_dir, history, f"ArcFace s={scale}, m={margin}, {stage}")
    if stage == "view2" and not smoke:
        weights_dir = MODEL_DIR / "saved_models"
        weights_dir.mkdir(parents=True, exist_ok=True)

        weights_path = weights_dir / f"run_{run_number}.pt"
        temp_path = weights_path.with_suffix(".pt.tmp")

        torch.save({
            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "arcface_head_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()
            },
            "scale": scale,
            "margin": margin,
            "learning_rate": learning_rate,
            "epochs": len(history),
            "seed": SEED,
            "held_out_folds": list(held_out),
            "num_classes": len(dataset.classes),
            "class_names": dataset.classes,
        }, temp_path)

        temp_path.replace(weights_path)

        summary["saved_weights"] = str(
            weights_path.relative_to(MODEL_DIR)
        )
        print(f"Saved final model: {weights_path}", flush=True)
    save_json(complete, summary)
    if plateau is not None and not plateau["reached"]:
        print(f"WARNING: View 1 not plateaued by epoch {len(history)}. "
              "View 2 will not use an unconverged selected configuration.")
    if stage == "view2" and not smoke:
        print(f"Preserved checkpoint: {ckpt}", flush=True)
    else:
        ckpt.unlink(missing_ok=True)

    return summary


# Held-out embedding samples for plots.
@torch.inference_mode()
def save_embedding_samples(model, scored_rows, folds, device, run_dir):
    transform = get_eval_transform()
    model.eval()
    for fold in folds:
        paths = sorted({p for row in scored_rows if int(row["fold"]) == fold
                        for p in (row["img1"], row["img2"])})
        by_id = defaultdict(list)
        for path in paths:
            by_id[path.split("/")[0]].append(path)
        rng = np.random.default_rng([SEED, fold])
        multi = [name for name in sorted(by_id) if len(by_id[name]) >= 2]
        singles = [name for name in sorted(by_id) if len(by_id[name]) == 1]
        chosen = []
        for identity in rng.permutation(multi):
            if len(chosen) >= EMBEDDINGS_PER_FOLD:
                break
            chosen += by_id[identity][:min(4, EMBEDDINGS_PER_FOLD - len(chosen))]
        for identity in rng.permutation(singles):
            if len(chosen) >= EMBEDDINGS_PER_FOLD:
                break
            chosen += by_id[identity][:EMBEDDINGS_PER_FOLD - len(chosen)]
        if not chosen:
            continue
        features = []
        for i in range(0, len(chosen), 32):
            batch = []
            for path in chosen[i:i+32]:
                with Image.open(IMAGE_ROOT / path) as image:
                    batch.append(transform(image.convert("RGB")))
            features.append(test_time_embeddings(model, torch.stack(batch).to(device)).cpu().numpy())
        np.savez_compressed(Path(run_dir) / f"fold_{fold}_embeddings_LOCAL.npz",
                            paths=np.asarray(chosen),
                            identities=np.asarray([p.split("/")[0] for p in chosen]),
                            embeddings=np.concatenate(features))


# View 1 tuning.
def write_tuning_summary():
    entries = []
    for phase in ("grid", "ablation"):
        for file in sorted((RESULTS_DIR / "View_1" / phase).glob("*/run_complete.json")):
            s = json.loads(file.read_text())
            entries.append({
                "phase": phase, "scale": s["scale"], "margin": s["margin"],
                "learning_rate": s["learning_rate"], "epochs": s["epochs_completed"],
                "max_epoch_budget": s["max_epoch_budget"],
                "view1_plateau_reached": s["view1_plateau"]["reached"],
                "final_validation_accuracy": s["final_validation_accuracy"],
                "final_validation_auc": s["final_validation_auc"],
                "best_validation_accuracy": s["best_validation_accuracy"],
                "best_validation_epoch": s["best_validation_epoch"],
                "final_train_loss": s["final_train_loss"],
                "training_seconds": s["training_seconds"],
                "run_folder": str(file.parent.relative_to(RESULTS_DIR)),
            })
    entries.sort(key=lambda r: (r["phase"], -r["final_validation_accuracy"],
                                -r["final_validation_auc"]))
    for index, row in enumerate([r for r in entries if r["phase"] == "grid"], 1):
        row["rank_within_grid"] = index
    for row in entries:
        row.setdefault("rank_within_grid", "")
    if entries:
        write_csv(RESULTS_DIR / "View_1" / "tuning_summary.csv", entries)
    return entries


def best_main_grid():
    rows = [row for row in write_tuning_summary() if row["phase"] == "grid"]
    expected = len(SCALES) * len(MARGINS) * len(LEARNING_RATES)
    if len(rows) != expected:
        raise RuntimeError(f"View 1 grid incomplete: {len(rows)}/{expected} candidates")
    if any(int(row["max_epoch_budget"]) != EPOCHS for row in rows):
        raise RuntimeError("View 1 epoch budget changed; archive incompatible runs")
    return rows[0]


def selected_hyperparameters():
    chosen = best_main_grid()
    if str(chosen["view1_plateau_reached"]).lower() != "true":
        raise RuntimeError("Selected ArcFace View 1 configuration did not plateau. "
                           "Inspect learning curves and extend the epoch budget before View 2.")
    ablation = (RESULTS_DIR / "View_1" / "ablation" /
                f"s_{float(chosen['scale']):g}_lr_{float(chosen['learning_rate']):g}_m_0" /
                "run_complete.json")
    if not ablation.exists():
        raise RuntimeError("Required m=0 ablation at chosen scale/LR is missing. Finish View 1.")
    abl = json.loads(ablation.read_text())
    if (float(abl["scale"]) != float(chosen["scale"])
            or float(abl["learning_rate"]) != float(chosen["learning_rate"])):
        raise RuntimeError("Ablation has mismatched hyperparameters")
    return {"scale": float(chosen["scale"]), "margin": float(chosen["margin"]),
            "learning_rate": float(chosen["learning_rate"]),
            "epochs": int(chosen["epochs"]), "view1_selection": chosen,
            "view1_m0_ablation": {"validation_accuracy": abl["final_validation_accuracy"],
                                   "validation_auc": abl["final_validation_auc"],
                                   "epochs": abl["epochs_completed"]}}


def run_view1(smoke=False):
    if smoke:
        for margin in (MARGINS[0], ABLATED_MARGIN):
            run_dir = SMOKE_DIR / "View_1" / f"m_{margin:g}"
            train_one_run(run_dir, "view1", SCALES[0], margin,
                          LEARNING_RATES[0], SMOKE_EPOCHS, smoke=True)
        return
    for scale in SCALES:
        for margin in MARGINS:
            for lr in LEARNING_RATES:
                run_dir = (RESULTS_DIR / "View_1" / "grid" /
                           f"s_{scale:g}_m_{margin:g}_lr_{lr:g}")
                train_one_run(run_dir, "view1", scale, margin, lr, EPOCHS)
                write_tuning_summary()
    chosen = best_main_grid()
    scale, lr = float(chosen["scale"]), float(chosen["learning_rate"])
    run_dir = RESULTS_DIR / "View_1" / "ablation" / f"s_{scale:g}_lr_{lr:g}_m_0"
    train_one_run(run_dir, "view1", scale, 0.0, lr, EPOCHS)
    write_tuning_summary()
    save_json(RESULTS_DIR / "View_1" / "selected_from_view1.json",
              {"best_main": chosen, "ablation": str(run_dir.relative_to(RESULTS_DIR))})
    print("\nView 1 completed. Inspect tuning_summary.csv and training curves.")


# View 2 aggregation.
def fold_table(rows, evaluated):
    result = []
    for fold in evaluated["folds"]:
        subset = [r for r in rows if r["fold"] == int(fold["fold"])]
        threshold = float(fold["threshold"])
        correct = sum((r["score"] >= threshold) == bool(r["label"]) for r in subset)
        if not math.isclose(correct / len(subset), float(fold["accuracy"])):
            raise RuntimeError("Per-fold accuracy mismatch")
        result.append({"fold": int(fold["fold"]), "threshold": threshold,
                       "correct": correct, "n_pairs": len(subset),
                       "accuracy": float(fold["accuracy"])})
    return result


def save_histogram(rows, fold_results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    genuine = [r["score"] for r in rows if r["label"] == 1]
    impostor = [r["score"] for r in rows if r["label"] == 0]
    thresholds = [r["threshold"] for r in fold_results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(genuine, bins=55, alpha=.6, label="genuine")
    ax.hist(impostor, bins=55, alpha=.6, label="impostor")
    ax.axvline(np.mean(thresholds), color="black", ls="--", label="mean fold threshold")
    ax.axvspan(min(thresholds), max(thresholds), color="gray", alpha=.15,
               label="range of 10 fold thresholds")
    ax.set(xlabel="Cosine similarity", ylabel="number of pairs")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def aggregate_view2(final_dir, selected, run_summaries):
    rows, ownership = [], {}
    for run_number, held_out in enumerate(VIEW2_HELD_OUT, 1):
        path = final_dir / f"run_{run_number}" / "heldout_scores.csv"
        raw = read_csv(path)
        if Counter(int(row["fold"]) for row in raw) != Counter({held_out[0]: 600, held_out[1]: 600}):
            raise RuntimeError(f"Run {run_number} did not score precisely the held-out folds {held_out}")
        for fold in held_out:
            if fold in ownership:
                raise RuntimeError("Fold scored more than once")
            ownership[fold] = run_number
        for row in raw:
            rows.append({"fold": int(row["fold"]), "img1": row["img1"],
                         "img2": row["img2"], "label": int(row["label"]),
                         "score": float(row["score"])})

    expected = Counter((int(r["fold"]), r["img1"], r["img2"], int(r["label"]))
                       for r in read_manifest("view2_pairs.csv"))
    actual = Counter((r["fold"], r["img1"], r["img2"], r["label"]) for r in rows)
    if len(rows) != 6000 or actual != expected:
        raise RuntimeError("View 2 official manifest mismatch (expected exactly 6000 pairs)")

    metrics = evaluate_view2(rows, score_key="score", higher_is_same=True)
    folds = fold_table(rows, metrics)
    save_pair_scores(rows, final_dir / "pair_scores.csv")
    write_csv(final_dir / "fold_accuracy.csv", folds)

    labels = np.asarray([r["label"] for r in rows])
    scores = np.asarray([r["score"] for r in rows])
    far, tar, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    write_csv(final_dir / "roc_data.csv", [
        {"far": f, "tar": t, "threshold": th}
        for f, t, th in zip(far, tar, thresholds)])
    save_roc_plot({"ArcFace (cosine)": (scores, labels)}, final_dir / "roc.png")
    save_histogram(rows, folds, final_dir / "score_histogram.png")

    thresholds_by_fold = {r["fold"]: r["threshold"] for r in folds}
    errors = []
    for row in rows:
        cutoff = thresholds_by_fold[row["fold"]]
        predicted_same = row["score"] >= cutoff
        if predicted_same != bool(row["label"]):
            errors.append({
                "error_type": "false_accept" if predicted_same else "false_reject",
                "confidence": abs(row["score"] - cutoff),
                "fold": row["fold"], "img1": row["img1"], "img2": row["img2"],
                "label": row["label"], "score": row["score"], "threshold": cutoff,
                "scored_by_run": ownership[row["fold"]],
            })
    errors.sort(key=lambda r: (r["error_type"], -r["confidence"]))
    write_csv(final_dir / "confident_errors.csv", errors,
              fields=["error_type", "confidence", "fold", "img1", "img2", "label",
                      "score", "threshold", "scored_by_run"])

    times = [r["training_seconds"] for r in run_summaries]
    memories = [r["peak_sampled_mps_driver_mib"] for r in run_summaries
                if r["peak_sampled_mps_driver_mib"] is not None]
    gpu_memories = [
        r["peak_gpu_allocated_mib"]
        for r in run_summaries
        if r["peak_gpu_allocated_mib"] is not None
    ]
    summary = {
        "model": "ArcFace", "protocol": "unrestricted, labeled outside data",
        "seed": SEED, "seed_note": "one final seed (42); additional seeds if compute permits",
        "training_runs": 5, "evaluated_folds": 10, "heldout_scheme": VIEW2_HELD_OUT,
        "fold_to_run": ownership, "selected_from_view1": selected,
        "cosine": {k: metrics[k] for k in
                   ("accuracy_mean", "accuracy_std", "auc", "eer", "tar_at_far_0.01")},
        "training_total_seconds": float(sum(times)),
        "training_mean_seconds": float(np.mean(times)),
        "epochs_per_run": [r["epochs_completed"] for r in run_summaries],
        "peak_sampled_mps_driver_mib": max(memories) if memories else None,
        "peak_gpu_allocated_mib": max(gpu_memories) if gpu_memories else None,
        "memory_note": "MPS value sampled after each batch; not guaranteed true peak",
        "hardware": hardware_info(get_device()), "software": software_info(),
        "runs": run_summaries,
        "files": ["pair_scores.csv", "fold_accuracy.csv", "roc_data.csv",
                  "roc.png", "score_histogram.png", "confident_errors.csv"],
    }
    save_json(final_dir / "summary.json", summary)
    print(f"\nArcFace: accuracy={metrics['accuracy_mean']:.4f} ± "
          f"{metrics['accuracy_std']:.4f}, AUC={metrics['auc']:.4f}, "
          f"EER={metrics['eer']:.4f}, TAR@FAR=1%={metrics['tar_at_far_0.01']:.4f}")


def run_view2(smoke=False):
    selected = ({"scale": SCALES[0], "margin": MARGINS[0],
                 "learning_rate": LEARNING_RATES[0], "epochs": SMOKE_EPOCHS,
                 "view1_selection": None}
                if smoke else selected_hyperparameters())
    destination = (SMOKE_DIR if smoke else RESULTS_DIR) / "View_2"
    summaries = []
    for number, held_out in enumerate(VIEW2_HELD_OUT, 1):
        if smoke and number > 1:
            break
        run_dir = destination / f"run_{number}"
        summaries.append(train_one_run(run_dir, "view2", selected["scale"],
                                       selected["margin"], selected["learning_rate"],
                                       selected["epochs"], held_out=held_out,
                                       run_number=number, smoke=smoke))
    if not smoke:
        aggregate_view2(destination, selected, summaries)


if __name__ == "__main__":
    if STAGE == "smoke_test":
        run_view1(smoke=True)
        run_view2(smoke=True)
    elif STAGE == "view1_tuning":
        run_view1()
    elif STAGE == "view2_final":
        run_view2()
    else:
        raise ValueError(f"Unknown STAGE: {STAGE}")
