## Utility helpers for devices, seeding, and timing.

from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


def get_device() -> torch.device:
    ## Pick the fastest available device for the current machine.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    ## Seed Python, NumPy, and PyTorch for repeatable runs.
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def synchronize_device(device: torch.device | str) -> None:
    ## Sync the device before timing or measuring memory.
    device = torch.device(device)
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def hardware_info(device: torch.device | str | None = None) -> dict:
    ## Return a short description of the current hardware setup.
    device = torch.device(device) if device is not None else get_device()
    info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "python_platform": platform.platform(),
        "processor_architecture": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        info["accelerator_name"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        info["accelerator_name"] = "Apple Silicon (MPS)"
    else:
        info["accelerator_name"] = "CPU"
    return info


class TrainingMonitor:
    ## Track elapsed time and a few basic memory stats.

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.started_at = None
        self.elapsed_seconds = None
        self.max_sampled_allocated_bytes = None
        self.max_sampled_driver_bytes = None

    def start(self) -> None:
        synchronize_device(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started_at = time.perf_counter()
        self.elapsed_seconds = None
        self.sample_memory()

    def sample_memory(self) -> None:
        if self.device.type != "mps" or not torch.backends.mps.is_available():
            return
        if hasattr(torch.mps, "current_allocated_memory"):
            current = torch.mps.current_allocated_memory()
            self.max_sampled_allocated_bytes = max(
                self.max_sampled_allocated_bytes or 0, current
            )
        if hasattr(torch.mps, "driver_allocated_memory"):
            current_driver = torch.mps.driver_allocated_memory()
            self.max_sampled_driver_bytes = max(
                self.max_sampled_driver_bytes or 0, current_driver
            )

    def stop(self) -> float:
        if self.started_at is None:
            raise RuntimeError("Call start() before stop()")
        synchronize_device(self.device)
        self.sample_memory()
        self.elapsed_seconds = time.perf_counter() - self.started_at
        return self.elapsed_seconds

    def report(self) -> dict:
        if self.elapsed_seconds is None:
            raise RuntimeError("Call stop() before report()")
        result = {"elapsed_seconds": round(self.elapsed_seconds, 3)}
        if self.device.type == "cuda":
            result["peak_gpu_allocated_mib"] = round(
                torch.cuda.max_memory_allocated(self.device) / (1024**2), 2
            )
        elif self.device.type == "mps":
            result["peak_sampled_mps_allocated_mib"] = (
                round(self.max_sampled_allocated_bytes / (1024**2), 2)
                if self.max_sampled_allocated_bytes is not None else None
            )
            result["peak_sampled_mps_driver_mib"] = (
                round(self.max_sampled_driver_bytes / (1024**2), 2)
                if self.max_sampled_driver_bytes is not None else None
            )
        return result


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path: str | Path, data: dict) -> None:
    ## Save a dictionary as formatted JSON.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
