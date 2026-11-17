# Resource-Efficient, Multi-Method Explainable AI for Brain Tumor Segmentation

Reference implementation and reproduction materials for the manuscript
*"Resource Efficient, Multi-Method Explainable AI for Brain Tumor Segmentation with
Cross-Population Validation"* (submitted to **IEEE Access**, manuscript ID
Access-2026-53071).

Everything in this repository runs on a **consumer-grade CPU**. No GPU is required or
used, for training, inference or explanation. All reported measurements were taken on an
Intel Core i5-1235U with 7.4 GB RAM and eight PyTorch threads.

---

## 1. The two datasets

Neither dataset is redistributed here. Both are obtained from their original providers.

### 1.1 Medical Segmentation Decathlon — Task01_BrainTumour (484 cases)

Used for all cross-validation, explainability, pruning and architectural-benchmark
experiments.

| | |
|---|---|
| Registry | <https://registry.opendata.aws/msd/> |
| Project site | <http://medicaldecathlon.com/> |
| Direct download | `s3://msd-for-monai/Task01_BrainTumour.tar` (us-west-2) |
| EU mirror | `s3://msd-for-monai-eu/Task01_BrainTumour.tar` (eu-west-2) |
| Size | ~7.3 GB (tar) |
| Licence | CC BY-SA 4.0 |
| Access | Open. No account or agreement required. |

```bash
# No AWS account needed
aws s3 cp --no-sign-request \
    s3://msd-for-monai/Task01_BrainTumour.tar \
    data/Task01_BrainTumour.tar
```

Cite as: Antonelli M., Reinke A., Bakas S., *et al.*, "The Medical Segmentation
Decathlon," *Nature Communications*, vol. 13, no. 1, p. 4128, 2022.

### 1.2 BraTS-Africa (146 cases)

Used only for the external cross-population evaluation in Section 4.4. The model is never
trained or fine-tuned on it.

| | |
|---|---|
| Synapse project | <https://www.synapse.org/Synapse:syn51156910> (BraTS 2023 Challenge) |
| Collection | BraTS-Africa (Sub-Saharan Africa glioma cohort) |
| Dataset paper | Adewole M., *et al.*, "The BraTS-Africa dataset," *Radiology: Artificial Intelligence*, vol. 7, no. 4, p. e240528, 2025 |
| Challenge paper | <https://arxiv.org/abs/2305.19369> |
| Access | **Restricted.** Requires a free Synapse account and acceptance of the challenge data-use agreement. |

> **This dataset cannot be redistributed.** Obtain it directly from Synapse under the
> data-use agreement. The cross-population scripts here expect a local cache built from
> your own copy; see `scripts/cache_bratsafrica.py`.

**Consequence for reproduction:** everything in this repository except Section 4.4 /
Table 3 can be reproduced from the open Decathlon data alone. Table 3 additionally
requires BraTS-Africa access.

---

## 2. Environment

Python **3.12.3**. PyTorch is CPU-only and must be installed from the CPU wheel index
first, because the default PyPI wheel pulls in CUDA.

```bash
git clone https://github.com/muhammadnass/brain-tumor-xai-cpu.git
cd brain-tumor-xai-cpu
python3 -m venv .venv && source .venv/bin/activate

pip install torch==2.5.1+cpu torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Exact versions used for every published number are pinned in `requirements.txt`.

---

## 3. Reproducing the published values

The `results/metrics/*.json` files in this repository **are the files the manuscript was
written from**. Every published number can therefore be checked without re-running
anything. Each file also carries a `provenance` block recording the checkpoint path, its
SHA-256, the fold definitions file, thread count and all method parameters.

### 3.1 Check a value without recomputing

```bash
# Table 7 / Figure 10 — pooled architectural comparison
python -c "import json;d=json.load(open('results/metrics/baseline_comparison_3fold.json'))['pooled'];\
print(d['mean_difference'], d['mean_difference_95ci'], d['wilcoxon']['p_two_sided'], d['cohens_dz'])"
# -> 0.010127 [-0.000917, 0.021272] 0.39687203458585807 0.081419
```

which is the `+0.0101 (95% CI −0.0009 to +0.0213; p = 0.397; Cohen's dz = 0.081)` printed
in Section 4.7.

### 3.2 Value → script → output file

Run every command from the repository root with the virtualenv active.

| Manuscript item | Command | Output file |
|---|---|---|
| Fold definitions (seed 42) | `python scripts/make_folds.py` | `data/brats_dataset/folds_seed42_3fold.json` |
| Pre-processed cache | `python scripts/precompute_cache.py` | `data/brats_dataset/cache_128/` |
| Table 2 — cross-validation | `python scripts/train_cv.py --fold {0,1,2} --epochs 12` | `results/metrics/cv_fold{0,1,2}.json` |
| Table 3 — cross-population | `python scripts/evaluate_cross_population.py --checkpoints models/segmentation/unet3d_cv_fold0.pth models/segmentation/unet3d_cv_fold1.pth models/segmentation/unet3d_cv_fold2.pth --decathlon-cache data/brats_dataset/cache_128 --folds data/brats_dataset/folds_seed42_3fold.json --africa-cache data/brats_dataset/cache_africa` | `results/metrics/cross_population.json` |
| Table 5 — agreement, n = 50 | `python scripts/compare_xai_methods.py --cases 50 --out results/metrics/xai_comparison_n50.json` | `results/metrics/xai_comparison_n50.json` |
| Table 5 — stability, n = 15 | `python scripts/explanation_stability.py --cases 15 --out results/metrics/xai_stability_n15.json` | `results/metrics/xai_stability_n15.json` |
| Table 6 — resources | `python scripts/measure_resources.py --checkpoint models/segmentation/unet3d_cv_fold0.pth --cache data/brats_dataset/cache_128` | `results/metrics/resources.json` |
| Table 6 — XAI latency, controlled | `python scripts/measure_xai_latency.py` | `results/metrics/xai_latency_controlled.json` |
| Table 7, Figure 10 — three-fold benchmark | `python scripts/train_baseline.py --fold {0,1,2} --epochs 12` then `python scripts/compare_paired_3fold.py` | `results/metrics/baseline_comparison_3fold.json` |
| Table 8 — exact pruning | `python scripts/prune_bottleneck.py --checkpoint models/segmentation/unet3d_cv_fold{N}.pth --save-model models/segmentation/unet3d_pruned_fold{N}.pth --out results/metrics/pruned_fold{N}.json` | `results/metrics/pruned_fold{0,1,2}.json` |
| Table 8 — pruned latency | `python scripts/measure_pruned_latency.py --checkpoint models/segmentation/unet3d_cv_fold{N}.pth --out results/metrics/pruned_latency_fold{N}.json` | `results/metrics/pruned_latency_fold{0,1,2}.json` |
| Section 3.16 — deterministic rebuild | `python scripts/restore_decathlon_cache.py --tar data/Task01_BrainTumour.tar` then `python scripts/verify_restored_cache.py --checkpoint models/segmentation/unet3d_cv_fold0.pth --fold 0` | `results/metrics/cache_verification.json` |
| Section 4.5 — bottleneck ablation | `python scripts/ablate_bottleneck.py --checkpoint models/segmentation/unet3d_cv_fold0.pth --cache data/brats_dataset/cache_128 --folds data/brats_dataset/folds_seed42_3fold.json` | `results/metrics/bottleneck_ablation.json` |
| Section 4.5 — attribution-target ablation | `python scripts/ablate_explainer_target.py` | `results/metrics/explainer_target_ablation.json` |
| Section 4.5 — Grad-CAM layer scan | `python scripts/test_gradcam_layers.py` | `results/metrics/gradcam_layer_scan.json` |

A value-by-value table, giving the exact JSON key for each number printed in the
manuscript, is in [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

### 3.3 Two self-checking scripts

Two scripts fail loudly rather than silently producing a different answer, and are the
quickest way for a reviewer to confirm the claims:

```bash
# Rebuilds all 484 volumes from the public tar, re-scores fold 0 against the
# per-case Dice recorded at submission, and exits non-zero on any deviation
# above --tolerance (default 1e-4). Published result: 162/162 at exactly 0.0.
python scripts/restore_decathlon_cache.py --tar data/Task01_BrainTumour.tar
python scripts/verify_restored_cache.py \
    --checkpoint models/segmentation/unet3d_cv_fold0.pth --fold 0

# Re-derives both attribution targets on the original 15 cases. Exits non-zero
# unless it reproduces the published 0.3200 (whole-volume, as submitted) and
# 0.6221 (predicted-mask, corrected) within --tolerance (default 0.005).
python scripts/ablate_explainer_target.py
```

### 3.4 Cost

| Step | Time on the reference CPU |
|---|---|
| Download + rebuild the cache | ~30 min (plus download) |
| Train one U-Net fold, 12 epochs | ~6 h |
| Train one SegResNet fold, 12 epochs | ~6 h |
| Full cross-validation, both architectures | ~35 h |
| Attribution agreement, 50 cases | ~1.5 h |
| Stability, 15 cases × 4 noise levels | ~2.5 h |
| Pruning + verification, one fold | ~4 min |

Analysis scripts that only read existing JSON (`compare_paired_3fold.py`) run in seconds.

---

## 4. Model checkpoints

Checkpoints are attached to the **[latest release](https://github.com/muhammadnass/brain-tumor-xai-cpu/releases/latest)**
rather than tracked in git, because the training checkpoints exceed GitHub's file-size
limit.

| File | Size | What it is |
|---|---|---|
| `unet3d_pruned_fold{0,1,2}.pth` | 44 MB each | **Deployment configuration.** 10,959,139 parameters. Reproduces the trained model bit-for-bit. |
| `unet3d_cv_fold{0,1,2}.pth` | 272 MB each | Trained models as evaluated, including optimiser state. 22,627,875 parameters. |
| `baseline_segresnet_fold{0,1,2}.pth` | 57 MB each | SegResNet baseline for Table 7. 4,702,227 parameters. |

SHA-256 of the fold 0 U-Net checkpoint, as recorded in every results file that used it:

```
d26a1533e784f6f138c9b1236b6e0c07dcc46bf50028bf7d89d8b10814562519
```

Place downloaded checkpoints in `models/segmentation/`.

---

## 5. Repository layout

```
src/
  ai_models/segmentation.py        3D U-Net, 22,627,875 parameters
  xai_explainers/
    gradient_methods.py            Vanilla Gradient, Integrated Gradients
    attention_visualizer.py        Grad-CAM (withdrawn; see Section 4.5)
    mesh_generator.py              3D surface extraction
  data_processing/preprocessor.py  Pre-processing pipeline (Section 3.3)
  surgical_planning/               Planning-oriented outputs (illustrative only)
  evaluation/                      Clinician-evaluation module (Phase 2)
  app.py                           Flask API
frontend/                          Streamlit interface
scripts/                           Every experiment and analysis script
tests/                             Unit tests, incl. pruning-equivalence test
results/metrics/*.json             The result files the manuscript was written from
data/brats_dataset/
  folds_seed42_3fold.json          Fold definitions (the only data file included)
docs/REPRODUCING.md                Value-by-value provenance
```

Run the unit tests with:

```bash
python -m pytest tests/ -v
```

---

## 6. Corrections recorded in this repository

The revised manuscript discloses three errors found during revision. The code and results
here are the corrected versions, and the scripts that established each correction are
included so the corrections themselves can be checked:

1. **Attribution target.** The explainer summed the tumour-class output over the whole
   volume instead of over predicted-tumour voxels, which let background contributions
   cancel the gradient. Corrected in `src/xai_explainers/gradient_methods.py`.
   Mean agreement 0.3200 → 0.6014. Demonstrated by `scripts/ablate_explainer_target.py`.
2. **Grad-CAM diagnosis.** The failure was caused by hooking a functionally inert
   bottleneck, not by layer-name resolution. Demonstrated by
   `scripts/test_gradcam_layers.py`. Grad-CAM remains withdrawn: layer choice flips the
   sign of its agreement with Vanilla Gradient.
3. **Architectural claim.** The single-fold SegResNet comparison did not survive
   extension to three folds (p = 0.397). All claims of architectural superiority are
   withdrawn. Demonstrated by `scripts/compare_paired_3fold.py`.

---

## 7. Intended use

Research and educational use only. The planning-oriented outputs are **illustrative and
have not been clinically validated**, and must not be used for clinical
decision-making. The system is not a medical device and has no regulatory clearance.

No patient data is contained in this repository.

---

## 8. Licence and citation

Code is released under the MIT Licence (see [`LICENSE`](LICENSE)). The datasets carry
their own separate licences and access terms, described in Section 1.

Citation metadata is in [`CITATION.cff`](CITATION.cff). Please cite the manuscript:

> R. A. Rasheed, M. S. Nasir, A. Kamsin, and N. J. Ismail, "Resource Efficient,
> Multi-Method Explainable AI for Brain Tumor Segmentation with Cross-Population
> Validation," submitted to *IEEE Access*, 2026.
