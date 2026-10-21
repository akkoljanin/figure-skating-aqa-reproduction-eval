# Figure Skating Action Quality Assessment: Reproduction & Evaluation

Reproduction and critical evaluation of **"Learning Long-Range Action Representation by Two-Stream Mamba Pyramid Network for Figure Skating Assessment"** (Wang et al., 2025), done as a project for the *Deep Learning Methods and Techniques* course at the Faculty of Technical Sciences (FTN), University of Novi Sad.

The model was trained and evaluated under limited compute (free Google Colab, Tesla T4), and the work includes a detailed error analysis of the failure cases.

> Original implementation: [ycwfs/Figure-Skating-Action-Quality-Assessment](https://github.com/ycwfs/Figure-Skating-Action-Quality-Assessment)

---

## Background

Figure skating is a demanding domain for Action Quality Assessment (AQA), because the official scoring has two separate components:

- **TES (Technical Element Score)**: the sum of scores of individual technical elements (jumps, spins, steps), judged visually only.
- **PCS (Program Component Score)**: artistic components, including interpretation of music, so audio is part of the judgement.

The paper proposes a **Two-Stream Mamba Pyramid Network** that respects this split:

- a **visual-only stream** (I3D features) detects, classifies and scores each technical element for TES;
- an **audio-visual stream** (I3D + VGGish features, fused with cross-attention) predicts PCS;
- a **multi-scale Mamba pyramid** (6 levels) models long-range temporal dependencies with linear complexity, which matters for performances up to ~4 minutes long.

## What this repository contains

| Item | Description |
|---|---|
| `FigureSkating_EVAL_epoch020.ipynb` | Colab notebook: environment setup, Mamba/NMS installation and dependency fixes, evaluation of the epoch-20 checkpoint, error analysis and plots |
| `report/` | Project report (PDF, in Serbian) with the full write-up |

The model code itself comes from the original repository, which the notebook clones.

## Setup

- **Dataset:** FineFS (1,604 videos with temporal boundaries and per-element scores), using precomputed I3D and VGGish features. The dataset is **not included** in this repo.
- **Environment:** Google Colab, Tesla T4 GPU, PyTorch 2.10 (CUDA 12.8).
- **Training:** 20 of the 50 epochs used in the paper (the authors used an A800 80 GB GPU); the checkpoint is loaded from the EMA model.
- **Evaluation split:** 187 videos (the last 20% of the sorted annotation files).
- **Config:** 6 pyramid levels, 24 element classes, embedding dim 512, max sequence length 288.

## Results (epoch 20 / 50)

**Temporal action localization (mean over tIoU 0.50 to 0.95)**

| Setting | Average accuracy |
|---|---|
| With classification | 53.37 % |
| Class-agnostic (`cls_ignore`) | 64.18 % |

**Spearman correlation**

| Target | ρ |
|---|---|
| Per-element score | 0.285 |
| Total TES per performance | 0.715 |
| PCS per performance | **0.971** |

**Score error (MAE):** TES 10.64, PCS 0.038.

Highlights:
- PCS prediction is already very strong after 20 epochs (ρ = 0.971, vs. 0.96 reported in the paper), which supports the paper's decision to use audio only for PCS.
- The ~17 point gap between classified and class-agnostic localization suggests the classifier converges more slowly than the boundary detector.

## Error analysis

The 10 worst TES predictions share a clear pattern:

- all are **free skating** performances (12 elements) with true TES of roughly **88 to 102**;
- the model heavily **underestimates the hardest elements** (true scores ~13 to 19, predicted ~3 to 4), a regression-toward-the-mean effect, likely caused by quad jumps being rare in the dataset, visually similar to triples, and by the shortened training;
- PCS error on the same videos stays tiny (e.g. video 817: TES error 42.22, PCS error 0.00), so the issue is not data loading or video quality but TES head convergence.

## Practical issues documented

- Dataset hosted on Baidu NetDisk, which is hard to reach without a Chinese phone number.
- `mamba-ssm` compiles from source (~40 min per Colab session).
- `mamba-ssm` / `transformers` version conflict (`ImportError`), solved by pinning a compatible `transformers` version, patching the import, and restarting the runtime.
- Limited GPU time on free Colab, handled with per-epoch checkpointing to Google Drive.

## Possible next steps

1. Continue training from the saved checkpoint up to epoch 50.
2. Report error per element category (Axel, Lutz, Loop, ...) instead of one aggregate error.
3. Oversample high-TES performances (quad jumps) during training.

## How to run

1. Open the notebook in Google Colab with a GPU runtime.
2. Put the FineFS features archive and your checkpoint on Google Drive and adjust the paths in the first cells.
3. Run the cells in order. Installing Mamba takes a while, and the dependency fix cell is needed because of the `transformers` conflict.

## Reference

Wang et al. (2025). *Learning Long-Range Action Representation by Two-Stream Mamba Pyramid Network for Figure Skating Assessment.*
