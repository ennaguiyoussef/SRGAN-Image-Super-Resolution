# SRGAN — Image Super-Resolution

A step-by-step PyTorch project for **4× single-image super-resolution**, designed for Google Colab GPU runtimes.

[![Open setup in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/01_setup_colab.ipynb)

## Current status

**Phases 1–4B: setup, data, models, pretraining, and adversarial fine-tuning.** Both training stages have resumable checkpoints and validation previews. Phase 5 covers inference and final evaluation.

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

## Phase 3: inspect the models

[Open model architecture in Colab](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/03_model_architecture.ipynb).

Run the notebook after phase 2 with a GPU runtime. It traces a 16-image batch through the generator, checks discriminator outputs, restores validation images from the Drive cache when needed, and saves an untrained output comparison. No optimizer steps are performed.

| Model | Input | Output |
| --- | --- | --- |
| Generator | RGB LR `(N, 3, H, W)` in [0, 1] | RGB SR `(N, 3, 4H, 4W)` in [-1, 1] |
| Discriminator | RGB HR or SR `(N, 3, 96, 96)` in [-1, 1] | Raw logits `(N, 1)` |

The generator uses a 9x9 input convolution, 16 residual blocks (3x3 convolutions, batch normalization, and channel-wise PReLU), a long skip, two learned PixelShuffle x2 blocks, and a 9x9 RGB output convolution with Tanh. Both local residual additions and the long skip preserve feature information. The default PyTorch parameter initialization is used.

The discriminator uses eight 3x3 convolution blocks (64 through 512 channels), LeakyReLU with slope 0.2, batch normalization after all but the first convolution, and a dense 1024-unit classifier. It is fixed to 96x96 crops; only the generator is used for future SR inference. It returns logits without sigmoid for future `BCEWithLogitsLoss`.

This is an SRGAN-style implementation rather than a claim of exact paper reproduction. Explicit choices include channel-wise PReLU, Tanh RGB output, LR [0, 1] / HR [-1, 1], Pillow degradation, PyTorch defaults, and a raw-logit discriminator. An untrained generator can produce correct shapes without reconstructing useful details. Its initial discriminator scores have no useful quality interpretation.

The notebook saves `results/phase3_untrained_generator.png` and `logs/phase3_model_inspection.json` on Drive. No weights are saved in phase 3.

## Phase 4A: pretrain the generator

[Open generator pretraining in Colab](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/04a_generator_pretraining.ipynb).

The first pilot uses 100 training images, all 50 validation images, 20 total epochs, batch size 16, float32, Adam (lr 1e-4, betas 0.9/0.999), and MSE in [-1, 1]. Gradients are clipped to norm 1.0. The discriminator is not instantiated. One random crop per image per epoch gives only 140 optimizer steps; this is a learning/pipeline pilot, not a convergence claim or the paper's full schedule.

Settings live in `configs/pretrain.yaml`. The notebook restores selected images from the Drive caches and uses a separate `checkpoints/<run_name>_smoke100/` or `checkpoints/<run_name>_full800/` directory. Changes to the data subset, learning rate, seed, or batch size require a separate run name. Increasing `epochs` extends the same run to a new **total** epoch target.

At each completed epoch, `last.pt` stores the latest generator, Adam state, history, experiment identity, and the best validation generator weights. A temporary sibling is written before replacement. On resume, model/optimizer settings are checked; partial epochs are repeated. Epoch-based random seeds and zero loader workers support crop/shuffle replay on the same environment. Exact equivalence across hardware or PyTorch versions is not guaranteed. Allow Drive writes to finish before disconnecting.

`last.pt` is authoritative. `best_generator.pt` is a derived weights-only export regenerated when the training function finishes, including a no-op resumed run. If an interrupted run has no export yet, the best weights remain available inside `last.pt` under `best_model`. If no trained epoch improves validation MSE, epoch 0 remains the best and is explicitly reported.

Results under `results/<run_id>/` include learning curves, LR/bicubic/untrained/best/HR comparisons, and a JSON report. Validation PSNR is mean per-image **RGB** PSNR on fixed 96x96 center crops with no border shaving; MSE is mapped from [-1, 1] to [0, 1] for this metric, with a 1e-12 error floor. These are pilot metrics, not official benchmark scores. Held-out test images are never loaded.

Inspect the curves and visual results before moving to a longer run or adversarial phase 4B.

## Phase 4A: expand to all 800 training images

[Open full-dataset pretraining in Colab](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/04a_full_dataset_pretraining.ipynb).

Use this after completing the 100-epoch smoke run. It loads the source `checkpoints/pretrain_smoke_v1_smoke100/last.pt` from Drive and initializes from its **best generator weights**. The source file is never rewritten. BatchNorm running statistics transfer with the model; Adam moments and epoch count do not.

The new run uses all 800 training images and the same 50 validation images, batch size 16, MSE, float32, and 20 full-data epochs by default. This adds 1,000 optimizer steps. Budget roughly 35–50 minutes on the previously observed T4, plus restoration of missing local images; use the printed first-epoch duration for a better estimate. These settings do not imply convergence or guarantee an improvement over bicubic.

Settings are in `configs/pretrain_full.yaml`. Destination files are under `checkpoints/pretrain_full_v1_full800/`. Full-data epoch 0 measures the transferred baseline; epoch 1 begins the new experiment. A SHA-256 source fingerprint, source best/completed epochs, source validation MSE, and optimizer reset flag are stored in the new checkpoint. Manifest, validation identities, ranges, crop size, degradation, architecture, and source-train subset membership are checked before transfer.

On rerun, the **destination checkpoint takes priority** and restores its own model, Adam, history, and best weights. It can resume without the source file. If the source is missing or no longer matches its original fingerprint, its image-preview column is omitted; the full run can still resume. Increasing `epochs` extends this run's total target. Other incompatible settings require a new run name.

The full notebook preserves the existing smoke notebook and configuration. Results compare the source baseline, the best full-data generator, and bicubic on identical validation crops. Held-out test images remain unused. Phase 4B uses the full-data best weights to start a separate GAN experiment.

## Phase 4B: adversarial fine-tuning

[Open adversarial training in Colab](https://colab.research.google.com/github/ennaguiyoussef/SRGAN-Image-Super-Resolution/blob/main/notebooks/04b_adversarial_training.ipynb).

The default pilot starts from the best generator in the full-data phase 4A checkpoint, with a fresh discriminator and two Adam optimizers. It runs three epochs over 800 training images, batch size 8, in float32. Settings are in `configs/adversarial.yaml`.

Each batch updates D on real HR and detached SR, then updates G through D with D parameters and BatchNorm statistics frozen. G's objective uses pixel MSE (1.0), VGG19 feature MSE (0.01), and adversarial BCE (0.001). The feature extractor uses pretrained ImageNet VGG19 through relu2_2 with RGB normalization at native crop size. The first run caches approximately 548 MB of weights under the Drive project folder. The additional pixel term and the loss weights define this pilot's objective.

The GAN run has its own `last.pt` with G, D, both optimizers, source provenance, and metrics. Resume starts from the last completed epoch. The weights export `best_gan_generator.pt` selects the trained GAN epoch with the lowest MSE on the 50 validation crops. The report compares that epoch, the phase 4A baseline, and bicubic. The held-out test split is reserved for phase 5.

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
│   ├── srgan.yaml
│   ├── pretrain.yaml
│   ├── pretrain_full.yaml
│   └── adversarial.yaml
├── notebooks/
│   ├── 01_setup_colab.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04a_generator_pretraining.ipynb
│   ├── 04a_full_dataset_pretraining.ipynb
│   └── 04b_adversarial_training.ipynb
├── src/
│   ├── __init__.py
│   ├── setup.py
│   ├── data.py
│   ├── dataset.py
│   ├── pretrain.py
│   ├── adversarial.py
│   └── models/
│       ├── __init__.py
│       ├── generator.py
│       └── discriminator.py
├── data/
├── checkpoints/
├── results/
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_pretrain.py
│   ├── test_warm_start.py
│   └── test_adversarial.py
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

The setup notebook verifies Drive writes and records environment versions and the Git commit. Generator pretraining now saves epoch-boundary checkpoints with Adam state and resumes from them. Data, weights, results, and local secrets are excluded from Git.

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


## Model integration checks

With a matching PyTorch installation plus NumPy and Pillow, run `python -m unittest discover -s tests -v`. Model checks exercise the default 16-image generator batch, non-square input, data-to-model ranges/shapes, discriminator/generator gradient isolation, and generator weight serialization. These are CPU checks; Colab verifies the GPU path.

## Pretraining checks

The unittest suite also checks the metric range conversion, parameter/BatchNorm updates on the real generator, and uninterrupted versus epoch-boundary-resumed training on synthetic random data. It checks optimizer/history restoration, best-weight export, no-op resume, and rejection of incompatible settings. No DIV2K download is required for these CPU checks.

Warm-start tests additionally verify exact best-weight transfer, source checkpoint immutability, reset Adam step counters, provenance recording, manifest compatibility, and destination resume when the source has been moved.
