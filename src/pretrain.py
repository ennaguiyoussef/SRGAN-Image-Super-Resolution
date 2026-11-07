"""Generator-only MSE pretraining with resumable epoch-boundary checkpoints."""
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def atomic_torch_save(value, path):
    """Write a temporary sibling before replacing the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_weights(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.inference_mode()
def validate(model, loader, device):
    """RGB center-crop metrics; no border shave. PSNR is mean per-image dB.

    MSE is in [-1, 1]. PSNR uses the equivalent [0, 1] error (MSE / 4).
    A 1e-12 error floor caps PSNR at 120 dB for exact reconstructions.
    """
    previous_mode = model.training
    model.eval()
    mse_sum, psnr_sum, count = 0.0, 0.0, 0
    try:
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            if sr.shape != hr.shape:
                raise ValueError("Generated image shape differs from the HR reference.")
            mse = (sr - hr).square().flatten(1).mean(1)
            if not torch.isfinite(mse).all():
                raise FloatingPointError("Non-finite validation error.")
            psnr = -10 * torch.log10((mse / 4).clamp_min(1e-12))
            mse_sum += mse.sum().item()
            psnr_sum += psnr.sum().item()
            count += lr.shape[0]
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Validation loader is empty.")
    return {"val_mse": mse_sum / count, "val_psnr_rgb": psnr_sum / count}


def initialize_from_checkpoint(model, checkpoint, experiment=None):
    """Load trained best generator weights, without transferring Adam or epochs."""
    checkpoint = Path(checkpoint)
    payload = checkpoint.read_bytes()
    source = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if source.get("format_version") != 1 or "best_model" not in source:
        raise ValueError("Source must be a phase-4A last.pt checkpoint.")
    if source.get("best_epoch", 0) < 1:
        raise ValueError("Source best weights are still epoch 0; inspect the source run first.")
    source_experiment = source.get("settings", {}).get("experiment", {})
    target_experiment = experiment or {}
    for key in ("scale_factor", "hr_crop_size", "manifest_sha256", "validation_names",
                "degradation", "lr_range", "hr_range"):
        if key in target_experiment and source_experiment.get(key) != target_experiment[key]:
            raise ValueError(f"Warm-start source has a different {key}.")
    if "train_names" in target_experiment:
        if not set(source_experiment.get("train_names", [])) <= set(target_experiment["train_names"]):
            raise ValueError("Source training images are not contained in the target training set.")
    model.load_state_dict(source["best_model"], strict=True)
    return {
        "kind": "best_generator_weights_only", "source_checkpoint": str(checkpoint),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_best_epoch": source["best_epoch"],
        "source_completed_epoch": source["epoch"],
        "source_best_val_mse": source["best_val_mse"],
        "source_commit": source.get("last_source_commit", source.get("source_commit", "")),
        "optimizer_reset": True,
    }


def fit_pretrain(model, train_dataset, validation_dataset, run_dir, epochs,
                 batch_size=16, learning_rate=1e-4, seed=42,
                 gradient_clip_norm=1.0, device="cpu", resume=True,
                 experiment=None, source_commit="", show_progress=True,
                 initial_checkpoint=None):
    """Train to a total epoch target, preserving model, Adam, history, and best weights.

    Each epoch reseeds Python/NumPy/Torch and its shuffle generator from seed+epoch.
    With num_workers=0 this replays crop/shuffle randomness after an epoch-boundary
    resume. An interrupted partial epoch is repeated from the last saved boundary.
    Exact equivalence across devices or software versions is not guaranteed.
    initial_checkpoint warm-starts a new run from best generator weights only.
    An existing destination last.pt takes priority and does not need the source file.
    """
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive.")
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Training and validation datasets must be nonempty.")
    device, run_dir = torch.device(device), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # JSON round trip makes saved metadata independent of caller-owned containers.
    experiment = json.loads(json.dumps(experiment or {}))
    settings = {
        "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
        "gradient_clip_norm": gradient_clip_norm, "num_workers": 0,
        "experiment": experiment,
        "model": type(model).__qualname__,
        "state_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        "train_images": len(train_dataset), "validation_images": len(validation_dataset),
    }
    if initial_checkpoint is not None:
        settings["initial_checkpoint"] = str(Path(initial_checkpoint))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("Run already exists. Use resume=True or a new run_name.")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state.get("settings") != settings:
            raise ValueError("Checkpoint settings differ. Restore the original settings or use a new run_name.")
        if state["epoch"] > epochs:
            raise ValueError("Target epochs is smaller than the saved epoch.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("torch_version") != str(torch.__version__):
            print("PyTorch version changed since checkpoint; exact numerical replay is not guaranteed.")
        print(f"Resuming after epoch {state['epoch']}; target total epochs: {epochs}")
    else:
        initialization = {"kind": "fresh_model"}
        if initial_checkpoint is not None:
            if Path(initial_checkpoint).resolve() == checkpoint_path.resolve():
                raise ValueError("Source and destination checkpoints must be different.")
            initialization = initialize_from_checkpoint(model, initial_checkpoint, experiment)
            print(f"Loaded source best weights from epoch {initialization['source_best_epoch']}; Adam starts fresh.")
        baseline = validate(model, validation_loader, device)
        state = {
            "format_version": 1, "epoch": 0, "settings": settings,
            "initialization": initialization,
            "source_commit": source_commit, "torch_version": str(torch.__version__),
            "history": [{"epoch": 0, "train_mse": None, **baseline}],
            "best_val_mse": baseline["val_mse"], "best_epoch": 0,
            "best_model": cpu_weights(model),
        }
        state["model"], state["optimizer"] = cpu_weights(model), optimizer.state_dict()
        atomic_torch_save(state, checkpoint_path)
        print(f"Initial baseline: MSE={baseline['val_mse']:.6f}, RGB PSNR={baseline['val_psnr_rgb']:.2f} dB")
    for epoch in range(state["epoch"] + 1, epochs + 1):
        started = time.perf_counter()
        epoch_seed = seed + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2 ** 32))
        torch.manual_seed(epoch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(epoch_seed)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(epoch_seed),
            pin_memory=device.type == "cuda",
        )
        model.train()
        total_loss, count = 0.0, 0
        if show_progress:
            from tqdm.auto import tqdm
            batches = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        else:
            batches = loader
        for lr, hr in batches:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            sr = model(lr)
            if sr.shape != hr.shape:
                raise ValueError("Generated image shape differs from HR.")
            loss = criterion(sr, hr)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; last completed epoch remains saved.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * lr.shape[0]
            count += lr.shape[0]
        metrics = validate(model, validation_loader, device)
        state["history"].append({
            "epoch": epoch, "train_mse": total_loss / count, **metrics,
            "seconds": time.perf_counter() - started,
        })
        if metrics["val_mse"] < state["best_val_mse"]:
            state["best_val_mse"], state["best_epoch"] = metrics["val_mse"], epoch
            state["best_model"] = cpu_weights(model)
        state["epoch"] = epoch
        state["model"], state["optimizer"] = cpu_weights(model), optimizer.state_dict()
        state["last_source_commit"] = source_commit
        # last.pt is authoritative and includes both latest and best model states.
        atomic_torch_save(state, checkpoint_path)
        atomic_json_save(state["history"], run_dir / "history.json")
        print(
            f"Epoch {epoch:03d}/{epochs}: train MSE={total_loss/count:.6f}, "
            f"val MSE={metrics['val_mse']:.6f}, "
            f"RGB PSNR={metrics['val_psnr_rgb']:.2f} dB, "
            f"time={state['history'][-1]['seconds']:.1f}s; saved"
        )
    # Export is regenerated from the authoritative checkpoint, including on a no-op rerun.
    atomic_torch_save(state["best_model"], run_dir / "best_generator.pt")
    atomic_json_save(state["history"], run_dir / "history.json")
    return state
