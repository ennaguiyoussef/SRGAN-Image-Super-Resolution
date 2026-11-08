"""Full-image 4x inference and metrics for the held-out DIV2K split."""
import math

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


SCALE = 4


def make_test_pair(image):
    """Trim the right/bottom edges to multiples of four, then generate LR."""
    width, height = image.size
    width, height = width - width % SCALE, height - height % SCALE
    if width == 0 or height == 0:
        raise ValueError("HR image must be at least 4 pixels wide and tall.")
    hr = image.convert("RGB").crop((0, 0, width, height))
    lr = hr.resize((width // SCALE, height // SCALE), Image.Resampling.BICUBIC)
    return lr, hr


def rgb_array(image):
    """Read RGB pixels as a float32 HWC array in [0, 1]."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@torch.inference_mode()
def resolve_image(generator, image, *, tile_size=192, halo=48):
    """Upscale an RGB image and return float32 RGB HWC pixels in [0, 1].

    LR halo pixels provide context for each tile and are discarded in the output.
    The default halo covers the receptive field of the 16-block generator.
    """
    if generator.training:
        raise ValueError("Call generator.eval() before inference.")
    if type(tile_size) is not int or tile_size < 1 or type(halo) is not int or halo < 0:
        raise ValueError("tile_size must be a positive integer and halo a nonnegative integer.")
    device = next(generator.parameters()).device
    lr = torch.from_numpy(rgb_array(image)).permute(2, 0, 1).unsqueeze(0)
    height, width = lr.shape[-2:]
    output = torch.empty(3, height * SCALE, width * SCALE, dtype=torch.float32)
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        start_y, end_y = max(0, top - halo), min(height, bottom + halo)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            start_x, end_x = max(0, left - halo), min(width, right + halo)
            tile = lr[:, :, start_y:end_y, start_x:end_x].contiguous().to(device)
            sr = generator(tile)
            expected = (1, 3, (end_y - start_y) * SCALE, (end_x - start_x) * SCALE)
            if tuple(sr.shape) != expected:
                raise ValueError(f"Expected 4x RGB output {expected}, got {tuple(sr.shape)}.")
            core_top, core_left = (top - start_y) * SCALE, (left - start_x) * SCALE
            core_h, core_w = (bottom - top) * SCALE, (right - left) * SCALE
            output[:, top * SCALE:bottom * SCALE, left * SCALE:right * SCALE] = sr[
                0, :, core_top:core_top + core_h, core_left:core_left + core_w
            ].cpu()
    if not torch.isfinite(output).all():
        raise FloatingPointError("Generator produced non-finite pixels.")
    return output.mul_(0.5).add_(0.5).clamp_(0, 1).permute(1, 2, 0).numpy()


def image_metrics(reference, prediction, *, border=4):
    """Compute RGB PSNR and SSIM for one [0, 1] image pair."""
    if reference.shape != prediction.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference and prediction must have the same HWC RGB shape.")
    if type(border) is not int or border < 0 or min(reference.shape[:2]) - 2 * border < 7:
        raise ValueError("A nonnegative border must leave at least 7x7 pixels for SSIM.")
    if border:
        reference = reference[border:-border, border:-border]
        prediction = prediction[border:-border, border:-border]
    mse = float(np.mean(np.square(reference - prediction), dtype=np.float64))
    if not math.isfinite(mse):
        raise FloatingPointError("Image pair has a non-finite reconstruction error.")
    return {
        "psnr_db": -10 * math.log10(max(mse, 1e-12)),
        "ssim": float(structural_similarity(
            reference, prediction, data_range=1.0, channel_axis=-1, win_size=7,
            gaussian_weights=False, use_sample_covariance=True,
        )),
    }
