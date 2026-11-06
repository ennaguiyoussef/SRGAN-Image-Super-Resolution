"""Offline training/resume and metric checks on synthetic data."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import Generator
from src.pretrain import fit_pretrain, validate


class RandomCropToyDataset(torch.utils.data.Dataset):
    """Exercise Python and NumPy randomness used by crop/data transforms."""
    def __len__(self):
        return 4

    def __getitem__(self, index):
        import random
        import numpy as np
        value = (index + random.random() + np.random.rand()) / 6
        return torch.full((3, 2, 2), value), torch.full((3, 8, 8), -0.5)


def toy_model():
    return nn.Sequential(nn.Upsample(scale_factor=4, mode="nearest"), nn.Conv2d(3, 3, 1), nn.Tanh())


class PretrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_use_correct_dynamic_range_and_restore_mode(self):
        x, target = torch.zeros(2, 3, 8, 8), torch.ones(2, 3, 8, 8)
        model = nn.Identity().train()
        metrics = validate(model, DataLoader(TensorDataset(x, target), batch_size=1), "cpu")
        self.assertAlmostEqual(metrics["val_mse"], 1.0)
        self.assertAlmostEqual(metrics["val_psnr_rgb"], 6.0206, places=4)
        self.assertTrue(model.training)

    def test_learning_and_epoch_boundary_resume_match_uninterrupted(self):
        torch.manual_seed(4)
        initial = toy_model()
        train = RandomCropToyDataset()
        val = TensorDataset(torch.zeros(4, 3, 2, 2), torch.full((4, 3, 8, 8), -0.5))
        with tempfile.TemporaryDirectory() as tmp:
            options = dict(batch_size=2, learning_rate=0.02, show_progress=False, seed=42)
            uninterrupted = copy.deepcopy(initial)
            full = fit_pretrain(uninterrupted, train, val, Path(tmp)/"full", epochs=4, **options)
            partial = copy.deepcopy(initial)
            fit_pretrain(partial, train, val, Path(tmp)/"resume", epochs=2, **options)
            resumed_model = toy_model()
            resumed = fit_pretrain(resumed_model, train, val, Path(tmp)/"resume", epochs=4, **options)
            self.assertLess(full["best_val_mse"], full["history"][0]["val_mse"])
            self.assertEqual(resumed["epoch"], 4)
            self.assertEqual(len(resumed["history"]), 5)
            for key, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(full["best_epoch"], resumed["best_epoch"])
            stored = torch.load(Path(tmp)/"resume/last.pt", weights_only=True)
            self.assertTrue(stored["optimizer"]["state"])
            best = torch.load(Path(tmp)/"resume/best_generator.pt", weights_only=True)
            for key in best:
                torch.testing.assert_close(best[key], stored["best_model"][key])
            same = fit_pretrain(toy_model(), train, val, Path(tmp)/"resume", epochs=4, **options)
            self.assertEqual(len(same["history"]), 5)
            with self.assertRaises(ValueError):
                fit_pretrain(toy_model(), train, val, Path(tmp)/"resume", epochs=5,
                             **{**options, "learning_rate":0.01})
            with self.assertRaises(FileExistsError):
                fit_pretrain(toy_model(), train, val, Path(tmp)/"resume", epochs=5,
                             resume=False, **options)

    def test_actual_generator_updates_and_saves_batchnorm_state(self):
        torch.manual_seed(5)
        model = Generator()
        before = model.head[0].weight.detach().clone()
        lr = torch.rand(2, 3, 24, 24)
        hr = torch.rand(2, 3, 96, 96) * 2 - 1
        dataset = TensorDataset(lr, hr)
        with tempfile.TemporaryDirectory() as tmp:
            state = fit_pretrain(model, dataset, dataset, tmp, epochs=1, batch_size=2, show_progress=False)
            self.assertFalse(torch.equal(before, model.head[0].weight))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            stored = torch.load(Path(tmp)/"last.pt", weights_only=True)
            self.assertEqual(stored["epoch"], 1)
            self.assertEqual(
                stored["model"]["residual_blocks.0.body.1.num_batches_tracked"].item(), 1
            )
            self.assertTrue(Path(tmp, "best_generator.pt").is_file())


if __name__ == "__main__":
    unittest.main()
