"""SRGAN adversarial training with VGG features and resumable checkpoints."""
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.pretrain import (
    atomic_json_save,
    atomic_torch_save,
    cpu_weights,
    initialize_from_checkpoint,
    validate,
)
from src.setup import seed_everything


class VGGFeatureLoss(nn.Module):
    """Compare frozen VGG19 relu2_2 activations of RGB crops in [-1, 1]."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        weights = VGG19_Weights.IMAGENET1K_V1
        # Retain only the early feature layers and release the classifier.
        self.features = vgg19(weights=weights).features[:9]
        self.features.requires_grad_(False)
        self.features.eval()
        preprocessing = weights.transforms()
        self.register_buffer("mean", torch.tensor(preprocessing.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(preprocessing.std).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr_normalized = ((sr + 1) / 2 - self.mean) / self.std
        hr_normalized = ((hr + 1) / 2 - self.mean) / self.std
        with torch.no_grad():
            target_features = self.features(hr_normalized)
        return F.mse_loss(self.features(sr_normalized), target_features)


def adversarial_step(generator, discriminator, feature_loss, optim_g, optim_d,
                     lr, hr, loss_weights, gradient_clip_norm):
    """Update D with detached images, then G through the frozen D."""
    generator.train()
    discriminator.train()
    optim_g.zero_grad(set_to_none=True)
    optim_d.zero_grad(set_to_none=True)

    sr = generator(lr)
    if sr.shape != hr.shape:
        raise ValueError("Generator output shape differs from the HR reference.")

    real_logits, fake_logits = discriminator(torch.cat([hr, sr.detach()])).chunk(2)
    loss_d = (
        F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    ) / 2
    if not torch.isfinite(loss_d):
        raise FloatingPointError("Non-finite discriminator loss.")
    loss_d.backward()
    nn.utils.clip_grad_norm_(discriminator.parameters(), gradient_clip_norm, error_if_nonfinite=True)
    optim_d.step()
    optim_d.zero_grad(set_to_none=True)

    # Evaluation mode also prevents BatchNorm statistics changing during G's update.
    discriminator.eval().requires_grad_(False)
    try:
        generated_logits = discriminator(sr)
        pixel = F.mse_loss(sr, hr)
        feature = feature_loss(sr, hr)
        adversarial = F.binary_cross_entropy_with_logits(
            generated_logits, torch.ones_like(generated_logits)
        )
        loss_g = (
            loss_weights["pixel"] * pixel
            + loss_weights["feature"] * feature
            + loss_weights["adversarial"] * adversarial
        )
        if not torch.isfinite(loss_g):
            raise FloatingPointError("Non-finite generator loss.")
        loss_g.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optim_g.step()
    finally:
        discriminator.requires_grad_(True).train()

    return {
        "g_total": loss_g.item(), "d_bce": loss_d.item(),
        "g_pixel_mse": pixel.item(), "g_feature_mse": feature.item(),
        "g_adversarial_bce": adversarial.item(),
    }


def fit_adversarial(generator, discriminator, feature_loss, train_dataset, validation_dataset,
                    run_dir, initial_checkpoint, *, epochs, batch_size, generator_lr,
                    discriminator_lr, loss_weights, gradient_clip_norm, seed=42,
                    device="cpu", resume=True, experiment=None, source_commit="",
                    show_progress=True):
    """Run the GAN to a total epoch target with independent G and D optimizer states."""
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    rates = (generator_lr, discriminator_lr, gradient_clip_norm)
    if any(not math.isfinite(value) or value <= 0 for value in rates):
        raise ValueError("Learning rates and gradient_clip_norm must be finite and positive.")
    if set(loss_weights) != {"pixel", "feature", "adversarial"}:
        raise ValueError("Set pixel, feature, and adversarial loss weights.")
    if any(not math.isfinite(value) or value <= 0 for value in loss_weights.values()):
        raise ValueError("All three loss weights must be finite and positive.")
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Training and validation datasets must be nonempty.")

    device, run_dir = torch.device(device), Path(run_dir)
    last_path = run_dir / "last.pt"
    initial_checkpoint = Path(initial_checkpoint)
    if last_path.resolve() == initial_checkpoint.resolve():
        raise ValueError("Source and GAN checkpoints must be in separate directories.")
    generator.to(device)
    discriminator.to(device)
    feature_loss.to(device).eval().requires_grad_(False)
    optim_g = torch.optim.Adam(generator.parameters(), lr=generator_lr)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr)
    experiment = json.loads(json.dumps(experiment or {}))
    settings = {
        "batch_size": batch_size, "seed": seed, "num_workers": 0,
        "generator_lr": generator_lr, "discriminator_lr": discriminator_lr,
        "gradient_clip_norm": gradient_clip_norm, "loss_weights": dict(loss_weights),
        "train_images": len(train_dataset), "validation_images": len(validation_dataset),
        "feature_loss": type(feature_loss).__qualname__,
        "initial_checkpoint": str(initial_checkpoint), "experiment": experiment,
    }
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0)
    if last_path.is_file():
        if not resume:
            raise FileExistsError("GAN run already exists. Enable resume or choose a new run name.")
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state.get("kind") != "srgan_adversarial_v1" or state.get("settings") != settings:
            raise ValueError("GAN checkpoint settings differ. Use the original settings or a new run name.")
        if state["epoch"] > epochs:
            raise ValueError("Target epochs is smaller than the saved epoch.")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optim_g.load_state_dict(state["optimizer_g"])
        optim_d.load_state_dict(state["optimizer_d"])
        print(f"Resuming GAN after epoch {state['epoch']}; target total epochs: {epochs}")
    else:
        initialization = initialize_from_checkpoint(generator, initial_checkpoint, experiment)
        baseline = validate(generator, validation_loader, device)
        print(f"Pretrained baseline: MSE={baseline['val_mse']:.6f}, RGB PSNR={baseline['val_psnr_rgb']:.2f} dB")
        state = {
            "kind": "srgan_adversarial_v1", "epoch": 0, "settings": settings,
            "initialization": initialization, "initial_generator": cpu_weights(generator),
            "history": [{"epoch": 0, **baseline}], "best_epoch": None,
            "best_model": None, "best_val_mse": None,
        }

    for epoch in range(state["epoch"] + 1, epochs + 1):
        started = time.perf_counter()
        seed_everything(seed + epoch)
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed + epoch),
            pin_memory=device.type == "cuda",
        )
        if show_progress:
            from tqdm.auto import tqdm
            batches = tqdm(loader, desc=f"GAN epoch {epoch}/{epochs}", leave=False)
        else:
            batches = loader
        totals, count = {}, 0
        for lr, hr in batches:
            lr, hr = lr.to(device), hr.to(device)
            losses = adversarial_step(
                generator, discriminator, feature_loss, optim_g, optim_d,
                lr, hr, loss_weights, gradient_clip_norm,
            )
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value * lr.shape[0]
            count += lr.shape[0]

        means = {name: total / count for name, total in totals.items()}
        metrics = validate(generator, validation_loader, device)
        state["history"].append({
            "epoch": epoch, **means, **metrics, "seconds": time.perf_counter() - started,
        })
        if state["best_val_mse"] is None or metrics["val_mse"] < state["best_val_mse"]:
            state["best_val_mse"] = metrics["val_mse"]
            state["best_epoch"] = epoch
            state["best_model"] = cpu_weights(generator)
        state.update({
            "epoch": epoch, "generator": cpu_weights(generator),
            "discriminator": cpu_weights(discriminator),
            "optimizer_g": optim_g.state_dict(), "optimizer_d": optim_d.state_dict(),
            "last_source_commit": source_commit, "torch_version": str(torch.__version__),
        })
        atomic_torch_save(state, last_path)
        atomic_json_save(state["history"], run_dir / "history.json")
        print(
            f"GAN epoch {epoch:02d}/{epochs}: G={means['g_total']:.5f}, D={means['d_bce']:.5f}, "
            f"val MSE={metrics['val_mse']:.6f}, RGB PSNR={metrics['val_psnr_rgb']:.2f} dB, "
            f"time={state['history'][-1]['seconds']:.1f}s. Checkpoint saved."
        )

    atomic_torch_save(state["best_model"], run_dir / "best_gan_generator.pt")
    atomic_json_save(state["history"], run_dir / "history.json")
    return state
