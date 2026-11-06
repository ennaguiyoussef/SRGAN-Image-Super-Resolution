# SRGAN — Image Super-Resolution

A step-by-step PyTorch project for **4× single-image super-resolution**, designed for Google Colab GPU runtimes.

[![Open setup in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/01_setup_colab.ipynb)

## Current status

**Phase 1: initial environment setup.** This repository currently provides the Colab setup notebook, configuration, and environment utilities. Dataset processing, models, training, inference, and evaluation will be implemented in later phases. No trained weights or benchmark results are provided.

## Start in Colab

1. Open the setup notebook using the badge above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order and authorize Google Drive mounting.
4. Confirm the final message: **Phase 1 complete: GPU, configuration, and persistent storage checks passed.**

The notebook clones the public repository into `/content/SRGAN-Image-Super-Resolution`. An existing checkout is preserved on reruns; it is not automatically pulled or reset. For updated code, use a fresh Colab runtime or deliberately update your checkout after saving local edits.

Colab supplies PyTorch and torchvision. `requirements.txt` installs supporting libraries without replacing that CUDA-compatible pair. Outside Colab, install a matching PyTorch/torchvision pair from the [official PyTorch installer](https://pytorch.org/get-started/locally/) first.

## Project layout

```text
SRGAN-Image-Super-Resolution/
├── configs/
│   └── srgan.yaml
├── notebooks/
│   └── 01_setup_colab.ipynb
├── src/
│   ├── __init__.py
│   ├── setup.py
│   └── models/
│       └── __init__.py
├── data/
├── checkpoints/
├── results/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Files for later phases are added when implemented, so the repository does not contain empty training scripts.

## Storage

| Location | Contents | Persistence |
| --- | --- | --- |
| GitHub | Source, configuration, notebooks, documentation | Versioned |
| Colab `/content` | Working checkout and datasets | Lost when runtime storage resets |
| Drive `MyDrive/SRGAN-Image-Super-Resolution/` | `checkpoints/`, `results/`, `logs/` | Persists between runtimes |

The setup notebook verifies Drive writes and records environment versions and the Git commit. Checkpoint saving and training resume will be added with training. Data, weights, results, and local secrets are excluded from Git.

## Roadmap

1. **Environment:** verify GPU operations, configuration, and persistent storage.
2. **Data:** choose a dataset, separate train/validation/test splits, generate bicubic LR images, and inspect paired crops.
3. **Models:** implement and inspect the generator and discriminator.
4. **Training:** pretrain the generator with pixel loss, then introduce perceptual and adversarial losses; implement checkpoints and resume.
5. **Evaluation:** compare bicubic, pretrained generator, and SRGAN using visual examples, PSNR, and SSIM with a documented evaluation protocol.

## Initial configuration

- Scale: **4×**
- HR training crop: **96 × 96**
- Derived LR training crop: **24 × 24**
- Initial batch size: **16**, to be validated against GPU memory during training
- Seed: **42** (exact reproducibility across devices and software versions is not guaranteed)

These patch sizes describe training crops, not a restriction to those image dimensions at inference. The setup does not download a dataset or start training.

## Reference

Ledig et al., [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802). Implementation details and deviations will be documented as models and losses are added.

## License

See [LICENSE](LICENSE).
