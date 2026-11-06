"""Environment and configuration helpers for the initial Colab setup."""
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    """Read and validate the initial experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping.")
    for key in ("scale_factor", "hr_crop_size", "batch_size"):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if config["scale_factor"] != 4:
        raise ValueError("This project currently targets 4x super-resolution.")
    if config["hr_crop_size"] % config["scale_factor"]:
        raise ValueError("HR crop size must be divisible by the scale factor.")
    return config


def seed_everything(seed):
    """Seed random generators; exact reproducibility across hardware is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report():
    """Report versions and the available accelerator without requiring a GPU."""
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def prepare_directories(project_root, output_root):
    """Keep working data local and persistent outputs under the selected output root."""
    paths = {
        "data": Path(project_root) / "data",
        "checkpoints": Path(output_root) / "checkpoints",
        "results": Path(output_root) / "results",
        "logs": Path(output_root) / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
