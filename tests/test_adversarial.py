"""Exercise GAN gradient routing and resumption without downloading VGG weights."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.adversarial import adversarial_step, fit_adversarial
from src.pretrain import cpu_weights


LOSS_WEIGHTS = {"pixel": 1.0, "feature": 0.01, "adversarial": 0.001}


def small_models():
    generator = nn.Sequential(nn.Upsample(scale_factor=4, mode="nearest"), nn.Conv2d(3, 3, 1), nn.Tanh())
    discriminator = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.LeakyReLU(0.2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 1),
    )
    return generator, discriminator


class AdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradients_only_reach_the_model_being_updated(self):
        torch.manual_seed(42)
        generator, discriminator = small_models()
        original_g, original_d = cpu_weights(generator), cpu_weights(discriminator)
        optim_g = torch.optim.Adam(generator.parameters())
        optim_d = torch.optim.Adam(discriminator.parameters())

        def inspect_d_update(optimizer, args, kwargs):
            self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))

        def inspect_g_update(optimizer, args, kwargs):
            self.assertFalse(discriminator.training)
            self.assertTrue(all(parameter.grad is None for parameter in discriminator.parameters()))

        optim_d.register_step_pre_hook(inspect_d_update)
        optim_g.register_step_pre_hook(inspect_g_update)
        losses = adversarial_step(
            generator, discriminator, nn.MSELoss(), optim_g, optim_d,
            torch.rand(2, 3, 2, 2), torch.rand(2, 3, 8, 8) * 2 - 1,
            LOSS_WEIGHTS, 1.0,
        )
        self.assertGreater(losses["g_adversarial_bce"], 0)
        self.assertFalse(torch.equal(original_g["1.weight"], generator[1].weight))
        self.assertFalse(torch.equal(original_d["0.weight"], discriminator[0].weight))
        self.assertEqual(discriminator[1].num_batches_tracked.item(), 1)
        self.assertTrue(discriminator.training)
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))

    def test_resume_restores_both_models_and_optimizers_after_source_moves(self):
        torch.manual_seed(42)
        generator, discriminator = small_models()
        dataset = TensorDataset(torch.rand(4, 3, 2, 2), torch.rand(4, 3, 8, 8) * 2 - 1)
        options = dict(
            batch_size=2, generator_lr=1e-3, discriminator_lr=1e-3,
            loss_weights=LOSS_WEIGHTS, gradient_clip_norm=1.0, show_progress=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            torch.save({
                "format_version": 1, "best_model": cpu_weights(generator),
                "best_epoch": 18, "epoch": 20, "best_val_mse": 0.024,
            }, source)
            full = fit_adversarial(
                copy.deepcopy(generator), copy.deepcopy(discriminator), nn.MSELoss(),
                dataset, dataset, root / "full", source, epochs=2, **options,
            )
            fit_adversarial(
                copy.deepcopy(generator), copy.deepcopy(discriminator), nn.MSELoss(),
                dataset, dataset, root / "resume", source, epochs=1, **options,
            )
            source.rename(root / "moved_source.pt")
            resumed = fit_adversarial(
                *small_models(), nn.MSELoss(), dataset, dataset,
                root / "resume", source, epochs=2, **options,
            )
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["initialization"]["source_best_epoch"], 18)
            self.assertEqual([row["epoch"] for row in resumed["history"]], [0, 1, 2])
            for key in ("generator", "discriminator"):
                for name, value in full[key].items():
                    torch.testing.assert_close(resumed[key][name], value, rtol=0, atol=0)
            for key in ("optimizer_g", "optimizer_d"):
                for value in resumed[key]["state"].values():
                    self.assertEqual(value["step"].item(), 4)
            self.assertTrue((root / "resume" / "best_gan_generator.pt").is_file())


if __name__ == "__main__":
    unittest.main()
