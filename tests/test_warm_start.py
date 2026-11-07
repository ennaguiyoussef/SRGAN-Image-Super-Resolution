"""Warm-start transfer and destination-resume integration checks."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.pretrain import fit_pretrain, initialize_from_checkpoint


def small_model():
    return nn.Sequential(nn.Upsample(scale_factor=4, mode="nearest"), nn.Conv2d(3, 3, 1), nn.Tanh())


class WarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_transfer_best_weights_reset_adam_preserve_source_and_resume(self):
        torch.manual_seed(7)
        small_data = TensorDataset(torch.zeros(2, 3, 2, 2), torch.full((2, 3, 8, 8), -0.6))
        full_data = TensorDataset(torch.zeros(4, 3, 2, 2), torch.full((4, 3, 8, 8), -0.6))
        shared = {"scale_factor": 4, "manifest_sha256": "fixed", "validation_names": ["v1"]}
        source_experiment = {**shared, "train_names": ["t1", "t2"]}
        target_experiment = {**shared, "train_names": ["t1", "t2", "t3", "t4"]}
        options = dict(batch_size=2, learning_rate=0.01, show_progress=False, seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_state = fit_pretrain(
                small_model(), small_data, small_data, root/"source", epochs=3,
                experiment=source_experiment, **options
            )
            source_path = root/"source/last.pt"
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.assertGreater(source_state["best_epoch"], 0)
            candidate = small_model()
            provenance = initialize_from_checkpoint(candidate, source_path, target_experiment)
            self.assertTrue(provenance["optimizer_reset"])
            for key, tensor in candidate.state_dict().items():
                torch.testing.assert_close(tensor, source_state["best_model"][key], rtol=0, atol=0)
            target_state = fit_pretrain(
                candidate, full_data, small_data, root/"full", epochs=1,
                initial_checkpoint=source_path, experiment=target_experiment, **options
            )
            self.assertEqual(target_state["epoch"], 1)
            self.assertEqual(len(target_state["history"]), 2)
            self.assertEqual(target_state["initialization"]["source_sha256"], source_hash)
            for state in target_state["optimizer"]["state"].values():
                self.assertEqual(state["step"].item(), 2)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), source_hash)
            source_path.rename(root/"source/archived_source.pt")
            resumed = fit_pretrain(
                small_model(), full_data, small_data, root/"full", epochs=2,
                initial_checkpoint=source_path, experiment=target_experiment, **options
            )
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["initialization"], target_state["initialization"])

    def test_reject_incompatible_manifest_without_creating_target_checkpoint(self):
        data = TensorDataset(torch.zeros(2, 3, 2, 2), torch.full((2, 3, 8, 8), -0.6))
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_pretrain(small_model(), data, data, root/"source", epochs=3, batch_size=2,
                         learning_rate=0.01, show_progress=False, experiment={"manifest_sha256":"first"})
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                fit_pretrain(small_model(), data, data, root/"target", epochs=1, batch_size=2,
                             show_progress=False, initial_checkpoint=root/"source/last.pt",
                             experiment={"manifest_sha256":"different"})
            self.assertFalse((root/"target/last.pt").exists())


if __name__ == "__main__":
    unittest.main()
