# What is in `results/metrics/`

These are the result files the manuscript was written from, kept exactly as the scripts
produced them. Each carries a `provenance` block recording the checkpoint used, its
SHA-256, the fold definitions file, the thread count and all method parameters.

The directory also retains the earlier runs, including those superseded by corrections
made during revision. They are kept deliberately, as the record of what was reported at
submission and what changed. **Section 2 below lists them; do not read those files as
current results.**

`docs/REPRODUCING.md` maps every published number to its file and JSON key.

---

## 1. Current results — cited in the revised manuscript

| File | Supports |
|---|---|
| `cv_fold{0,1,2}.json` | Table 2, three-fold cross-validation |
| `cross_population.json` | Table 3, cross-population comparison (Section 4.4) |
| `xai_comparison_n50.json` | Table 5, attribution agreement, n = 50 |
| `xai_stability_n15.json` | Table 5, perturbation stability, n = 15, four noise levels |
| `resources.json` | Table 6, computational resource measurements |
| `xai_latency_controlled.json` | Table 6, explanation latency and peak memory, controlled protocol |
| `baseline_segresnet_fold{0,1,2}.json` | Table 7, SegResNet per-fold results |
| `baseline_comparison_3fold.json` | Table 7 and Figure 10, pooled paired comparison |
| `pruned_fold{0,1,2}.json` | Table 8, exact-pruning equivalence and parameter counts |
| `pruned_latency_fold{0,1,2}.json` | Table 8, pruned inference latency |
| `cache_verification.json` | Section 3.16, deterministic rebuild of the pre-processing cache |
| `ablation_fold{0,1,2}.json` | Section 4.5, bottleneck zero-ablation over all 484 cases |
| `explainer_target_ablation.json` | Section 4.5, the attribution-target defect and its correction |
| `gradcam_layer_scan.json` | Section 4.5, Grad-CAM layer sensitivity |

---

## 2. Superseded — retained as the record, not as results

### 2.1 Produced with the defective attribution target

Both files below were generated before the correction described in Section 4.5, in which
the backward target summed the tumour-class output over the whole volume instead of over
predicted-tumour voxels. **Their numbers are withdrawn.**

| File | Contains | Superseded by |
|---|---|---|
| `xai_comparison_n15.json` | Mean agreement 0.3200 | `xai_comparison_n50.json` (0.6014) |
| `stability_n8.json` | Vanilla Gradient 0.4395 at σ = 0.10 | `xai_stability_n15.json` (0.9849) |

`explainer_target_ablation.json` re-derives both targets on the same 15 cases and
reproduces 0.3200 and 0.6221 exactly, which is what establishes that the target
definition alone accounts for the difference.

### 2.2 Superseded by a wider or more controlled run

| File | Why superseded |
|---|---|
| `baseline_comparison.json` | Single-fold SegResNet comparison. Extended to three folds in `baseline_comparison_3fold.json`, where the difference ceases to be significant (p = 0.397). |
| `gradcam_diag_065.json`, `gradcam_diag_072.json`, `gradcam_diag_bottleneck.json`, `gradcam_diag_bottleneck_065.json` | Early Grad-CAM diagnostics. Superseded by the systematic layer scan in `gradcam_layer_scan.json`. |
| `explanation_stability_20260718_172055.json` | Early stability run, pre-correction and smaller. |
| `xai_comparison_20260717_204443.json`, `xai_comparison_20260911_094025.json` | Early agreement runs, pre-correction. |
| `benchmark_phase2.json` | Earlier resource benchmark, superseded by `resources.json` and the controlled latency protocol. |
| `attribution_depth.json` | Exploratory analysis of attribution by network depth. Not cited in the manuscript. |

### 2.3 Development-stage experiments

These come from the 30- and 80-case experiments described in Section 4.2 as ablations of
sampling and training-data volume. They do not contribute to any final performance
estimate, which uses all 484 cases.

| File | Stage |
|---|---|
| `evaluation_summary_20260713_083950.json` | early 30-case run |
| `evaluation_summary_20260713_174653.json` | early 30-case run |
| `evaluation_summary_20260714_081305.json` | early 80-case run |
| `cross_population_pilot.json` | cross-population pilot, 15 cases |
| `bratsafrica_summary_20260716_231533.json` | early BraTS-Africa evaluation |
| `bratsafrica_summary_20260716_235038.json` | early BraTS-Africa evaluation |

---

## 3. Not included

`*.progress.jsonl` files are resumption state written during long runs, not results, and
are excluded by `.gitignore`. Training and run logs are likewise excluded.

No per-patient imaging data appears in any file here. The per-case entries hold case
identifiers and derived scores only.
