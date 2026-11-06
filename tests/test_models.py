"""CPU model integration checks; no dataset download or optimizer step."""
import io
import unittest

import numpy as np
import torch
from PIL import Image

from src.models import Discriminator, Generator


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_generator_default_batch_and_non_square_input(self):
        model = Generator().eval()
        with torch.no_grad():
            sr = model(torch.rand(16, 3, 24, 24))
            rectangular = model(torch.rand(1, 3, 9, 13))
        self.assertEqual(tuple(sr.shape), (16, 3, 96, 96))
        self.assertEqual(tuple(rectangular.shape), (1, 3, 36, 52))
        self.assertTrue(torch.isfinite(sr).all())
        self.assertGreaterEqual(sr.min().item(), -1)
        self.assertLessEqual(sr.max().item(), 1)

    def test_discriminator_real_and_generated_inputs_and_gradient_routing(self):
        generator, discriminator = Generator(), Discriminator()
        lr = torch.rand(2, 3, 24, 24)
        hr = torch.rand(2, 3, 96, 96) * 2 - 1
        sr = generator(lr)
        real_logits = discriminator(hr)
        fake_logits = discriminator(sr.detach())
        self.assertEqual(tuple(real_logits.shape), (2, 1))
        criterion = torch.nn.BCEWithLogitsLoss()
        d_loss = criterion(real_logits, torch.ones_like(real_logits)) + criterion(
            fake_logits, torch.zeros_like(fake_logits)
        )
        d_loss.backward()
        self.assertTrue(all(p.grad is None for p in generator.parameters()))
        self.assertTrue(all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in discriminator.parameters()
        ))
        discriminator.zero_grad(set_to_none=True)
        discriminator.requires_grad_(False)
        discriminator.eval()
        logits = discriminator(sr)
        g_loss = criterion(logits, torch.ones_like(logits)) + torch.nn.functional.mse_loss(sr, hr)
        g_loss.backward()
        self.assertTrue(all(p.grad is None for p in discriminator.parameters()))
        self.assertTrue(all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in generator.parameters()
        ))

    def test_generator_weight_round_trip(self):
        model = Generator().eval()
        x = torch.rand(1, 3, 24, 24)
        with torch.no_grad():
            expected = model(x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = Generator().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x))

    def test_invalid_shapes_fail_with_clear_errors(self):
        with self.assertRaises(ValueError):
            Generator()(torch.rand(1, 1, 24, 24))
        with self.assertRaises(ValueError):
            Discriminator()(torch.rand(1, 3, 48, 48))

    def test_dataset_to_model_contract(self):
        import tempfile
        from pathlib import Path
        from src.dataset import SuperResolutionDataset

        with tempfile.TemporaryDirectory() as tmp:
            image = np.random.default_rng(42).integers(0, 256, (120, 120, 3), dtype=np.uint8)
            Image.fromarray(image).save(Path(tmp) / "sample.png")
            lr, hr = SuperResolutionDataset(tmp, ["sample.png"])[0]
            self.assertEqual(lr.dtype, torch.float32)
            self.assertEqual(hr.dtype, torch.float32)
            self.assertGreaterEqual(lr.min().item(), 0)
            self.assertLessEqual(lr.max().item(), 1)
            generator, discriminator = Generator().eval(), Discriminator().eval()
            with torch.no_grad():
                sr = generator(lr.unsqueeze(0))
                self.assertEqual(sr.shape, hr.unsqueeze(0).shape)
                self.assertEqual(tuple(discriminator(sr).shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
