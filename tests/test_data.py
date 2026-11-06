"""Offline checks for splitting, crop alignment, archive safety, and saved splits."""
import io
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import extract_hr_archive, paired_crop, save_manifest, split_manifest


class DataTests(unittest.TestCase):
    def test_disjoint_reproducible_image_splits(self):
        split = split_manifest()
        self.assertEqual(split, split_manifest())
        groups = [set(split[key]) for key in ("train", "validation", "test")]
        self.assertEqual([len(g) for g in groups], [800, 50, 50])
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))
        self.assertTrue(set(split["train_smoke"]) <= groups[0])
        self.assertEqual(len(split["train_smoke"]), 100)

    def test_center_crop_alignment_and_rgb(self):
        values = np.arange(120 * 132 * 3, dtype=np.uint8).reshape(120, 132, 3)
        image = Image.fromarray(values)
        lr, hr = paired_crop(image)
        self.assertEqual(lr.size, (24, 24))
        self.assertEqual(hr.size, (96, 96))
        np.testing.assert_array_equal(np.array(hr), values[12:108, 18:114])
        np.testing.assert_array_equal(
            np.array(lr), np.array(hr.resize((24, 24), Image.Resampling.BICUBIC))
        )
        self.assertEqual(paired_crop(image.convert("L"))[0].mode, "RGB")
        with self.assertRaises(ValueError):
            paired_crop(Image.new("RGB", (40, 40)))
        with self.assertRaises(ValueError):
            paired_crop(image, hr_size=95)

    def test_training_crop_is_repeatable_with_seed(self):
        image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (150, 150, 3), dtype=np.uint8))
        first = paired_crop(image, training=True, rng=random.Random(42))
        second = paired_crop(image, training=True, rng=random.Random(42))
        np.testing.assert_array_equal(np.array(first[1]), np.array(second[1]))

    def test_extract_expected_files_only_and_reject_missing_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = io.BytesIO()
            Image.new("RGB", (96, 96)).save(data, format="PNG")
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("images/0001.png", data.getvalue())
                handle.writestr("../outside.txt", "unexpected")
            extract_hr_archive(archive, root / "output", "images", [1])
            self.assertTrue((root / "output/images/0001.png").exists())
            self.assertFalse((root / "outside.txt").exists())
            with self.assertRaises(ValueError):
                extract_hr_archive(archive, root / "output", "images", [1, 2])

    def test_manifest_does_not_silently_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_manifest(path, split_manifest())
            save_manifest(path, split_manifest())
            with self.assertRaises(ValueError):
                save_manifest(path, split_manifest(seed=7))


if __name__ == "__main__":
    unittest.main()
