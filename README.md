# SRGAN — Image Super-Resolution

A step-by-step PyTorch project for **4× single-image super-resolution**, designed for Google Colab GPU runtimes.

[![Open setup in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/01_setup_colab.ipynb)

## Current status

**Phases 1–2: environment and data pipeline.** The repository provides Colab setup, DIV2K download/caching, fixed image-level splits, paired crop datasets, visualization, and batch checks. Models, training, inference, and evaluation will follow. No trained weights or benchmark results are provided.

## Phase 2: prepare DIV2K

[Open data preparation in Colab](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/02_data_preparation.ipynb).

Run its cells in order. It updates an existing clean checkout with a fast-forward pull, mounts Drive, downloads the official training and validation HR archives, and saves a fixed split manifest. Local tracked edits cause the update to stop without discarding them.

- Default smoke mode: 100 training images; set `SMOKE_MODE = False` for all 800.
- Custom validation/test: 50/50 images from the official validation HR set, split before cropping. Our held-out test is not the official challenge test.
- The smoke subset does **not** reduce archive download size. Budget several GB of downloads, approximately 12 GB free local space and 5 GB for the Drive archive cache. Download time depends on the server and connection.
- HR crops are 96×96; LR crops are 24×24. Pillow bicubic creates LR from the matching HR crop. This differs from the official MATLAB degradation, so results are not directly comparable to the official leaderboard.
- LR tensors use `[0, 1]`; HR tensors use `[-1, 1]`. Training crops are random; validation center crops are deterministic. Final full-image evaluation is deferred to phase 5.

The notebook saves `splits/div2k_seed42.json`, `results/phase2_lr_bicubic_hr.png`, and `logs/phase2_data_settings.json` under the Drive project folder. Test images are not visualized or used for tuning. Data and preview outputs are excluded from Git.

The [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is provided for academic research; image copyrights belong to their owners. The repository's code license does not license these images. Cite Agustsson and Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, and the relevant challenge report as requested by the dataset authors.

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
│   ├── 01_setup_colab.ipynb
│   └── 02_data_preparation.ipynb
├── src/
│   ├── __init__.py
│   ├── setup.py
│   ├── data.py
│   ├── dataset.py
│   └── models/
│       └── __init__.py
├── data/
├── checkpoints/
├── results/
├── tests/
│   └── test_data.py
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

## Offline data checks

With NumPy and Pillow installed, run `python -m unittest discover -s tests -v`. These checks cover reproducible disjoint splits, aligned crops, safe expected-member extraction, and persisted manifest protection. They use synthetic images and do not download DIV2K. The notebook performs the real PyTorch tensor and batch checks in Colab.

