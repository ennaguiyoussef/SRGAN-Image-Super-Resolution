"""Verify full-image pairing, tile seams, and metric conventions."""
import unittest

import numpy as np
import torch
from PIL import Image

from src.evaluation import image_metrics, make_test_pair, resolve_image, rgb_array
from src.models import Generator


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tiling_matches_single_pass_at_internal_and_image_edges(self):
        torch.manual_seed(42)
        pixels = np.random.default_rng(42).integers(0, 256, (79, 91, 3), dtype=np.uint8)
        lr = Image.fromarray(pixels)
        generator = Generator(channels=4).eval()
        tiled = resolve_image(generator, lr, tile_size=31)
        single = resolve_image(generator, lr, tile_size=100)
        self.assertEqual(tiled.shape, (316, 364, 3))
        np.testing.assert_allclose(tiled, single, rtol=1e-5, atol=2e-6)

    def test_pair_alignment_and_metric_range(self):
        lr, hr = make_test_pair(Image.new("L", (99, 101), 127))
        self.assertEqual((lr.mode, hr.mode), ("RGB", "RGB"))
        self.assertEqual((lr.size, hr.size), ((24, 25), (96, 100)))
        reference = rgb_array(hr)
        identical = image_metrics(reference, reference)
        self.assertAlmostEqual(identical["ssim"], 1.0)
        self.assertAlmostEqual(identical["psnr_db"], 120.0)
        wrong = reference.copy()
        wrong[:4] = 0
        self.assertAlmostEqual(image_metrics(reference, wrong)["psnr_db"], 120.0)
        self.assertAlmostEqual(image_metrics(reference, reference + 0.25)["psnr_db"], 12.0412, places=4)
        self.assertLess(image_metrics(reference, wrong, border=0)["ssim"], 1.0)
        with self.assertRaises(ValueError):
            image_metrics(reference, wrong, border=49)


if __name__ == "__main__":
    unittest.main()
