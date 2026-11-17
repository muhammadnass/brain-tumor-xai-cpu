# Value-by-value provenance

Every number printed in the manuscript, the file it comes from, and the JSON key that
holds it. All paths are relative to the repository root.

Each results file carries a `provenance` block recording the checkpoint used, its
SHA-256, the fold definitions file, the thread count and all method parameters, so a
value can be traced to the exact configuration that produced it.

A helper for reading any key:

```bash
python - <<'EOF'
import json
d = json.load(open("results/metrics/baseline_comparison_3fold.json"))
print(d["pooled"]["mean_difference"])
EOF
```

---

## Table 3 — Cross-population comparison (Section 4.4)

File: `results/metrics/cross_population.json`
Key root: `comparisons.metrics[]`, one entry per metric; per-case values under
`per_case.decathlon` and `per_case.brats_africa_by_patient`.

| Manuscript value | Key |
|---|---|
| Mean difference and CI | `mean_difference`, `mean_difference_95ci` |
| Median difference and CI | `median_difference`, `median_difference_95ci` |
| Hedges' g | `hedges_g` |
| Cliff's δ | `cliffs_delta` |
| p | `mann_whitney.p_two_sided` |

Published: Dice 0.7460 → 0.6625; precision Cliff's δ = 0.046 with p = 0.402, against a
mean difference of 0.0710 — the mean/rank divergence discussed in Section 4.4.

Decathlon sensitivity and precision are reported to five decimals (0.77659 and 0.77664)
because they coincide at four. Recompute from `per_case.decathlon`:

```bash
python - <<'EOF'
import json, numpy as np
pc = json.load(open("results/metrics/cross_population.json"))["per_case"]["decathlon"]
it = pc if isinstance(pc, list) else pc.values()
for k in ("sensitivity", "precision"):
    v = np.array([r[k] for r in it if r.get(k) is not None], float)
    print(k, round(v.mean(), 5))
EOF
# -> sensitivity 0.77659 / precision 0.77664
```

---

## Table 5 — Explainability (Section 4.5)

**Agreement, n = 50.** File: `results/metrics/xai_comparison_n50.json`, key
`agreement_pearson`.

| Published | Key | Value |
|---|---|---|
| Mean Pearson r | `agreement_pearson.mean` | 0.6014 |
| Median | `agreement_pearson.median` | 0.6650 |
| 95% CI | `agreement_pearson.mean_95ci` | [0.5523, 0.6497] |
| Range | `agreement_pearson.min` / `.max` | 0.1234 / 0.8116 |
| Degenerate maps | `degenerate` | 0 of 50, both methods |

**Stability, n = 15.** File: `results/metrics/xai_stability_n15.json`, key
`by_sigma.<σ>.<method>.mean` and `.sd`.

| σ | Vanilla Gradient | Integrated Gradients |
|---|---|---|
| 0.01 | 0.9968 (SD 0.0053) | 0.9990 (SD 0.0007) |
| 0.05 | 0.9913 (SD 0.0081) | 0.9933 (SD 0.0047) |
| 0.10 | 0.9849 (SD 0.0128) | 0.9853 (SD 0.0100) |
| 0.20 | 0.9740 (SD 0.0205) | 0.9681 (SD 0.0219) |

Note the σ keys are `"0.01"`, `"0.05"`, `"0.1"`, `"0.2"` — unpadded.

---

## Table 6 — Resources (Section 4.6)

Latency and peak memory: `results/metrics/xai_latency_controlled.json`. Everything else:
`results/metrics/resources.json`.

| Published | Key | Value |
|---|---|---|
| Vanilla Gradient latency | `vanilla_gradient.median_s` | 5.535 s |
| its 95% CI | `vanilla_gradient.median_95ci` | [5.2375, 5.7742] |
| Integrated Gradients latency | `integrated_gradients.median_s` | 112.038 s |
| its 95% CI | `integrated_gradients.median_95ci` | [110.243, 113.6625] |
| Cost ratio | `cost_ratio_ig_over_vg.median` | 20.52× |
| Peak RSS | `peak_rss_gb` | 2.8 GB |

The protocol that makes these comparable is recorded in `protocol`: 10 cases, 1 warmup
case discarded, both methods run consecutively on the same case with the within-case
order alternated between cases (`interleaved: true`,
`within_case_order_alternated: true`), 8 threads, 20 integration steps.

These supersede the 8.95 s and 177.77 s of the original submission, which were measured
without the controlled protocol. No improvement is claimed; see Section 4.6.

---

## Table 7 and Figure 10 — Architectural benchmark (Section 4.7)

File: `results/metrics/baseline_comparison_3fold.json`. `a` is the proposed U-Net, `b` is
SegResNet, so a positive difference favours SegResNet.

**Pooled** (`pooled`):

| Published | Key | Value |
|---|---|---|
| Mean Dice, proposed | `mean_dice_a` | 0.745962 |
| Mean Dice, SegResNet | `mean_dice_b` | 0.756089 |
| Median Dice, proposed | `median_dice_a` | 0.832698 |
| Median Dice, SegResNet | `median_dice_b` | 0.823150 |
| Paired mean difference | `mean_difference` | +0.010127 |
| its 95% CI | `mean_difference_95ci` | [−0.000917, +0.021272] |
| Paired median difference | `median_difference` | −0.004283 |
| its 95% CI | `median_difference_95ci` | [−0.009646, +0.001422] |
| Wilcoxon p | `wilcoxon.p_two_sided` | 0.39687 |
| Cohen's dz | `cohens_dz` | 0.081419 |
| Cases won, proposed / SegResNet / ties | `wins_a` / `wins_b` / `ties` | 251 / 229 / 4 |

**Per fold** (`per_fold.<0\|1\|2>`):

| Fold | n | Proposed | SegResNet | Mean diff | p |
|---|---|---|---|---|---|
| 0 | 162 | 0.729458 | 0.754088 | +0.024631 | 0.694945 |
| 1 | 161 | 0.766698 | 0.762813 | −0.003885 | 0.239259 |
| 2 | 161 | 0.741833 | 0.751379 | +0.009546 | 0.436013 |

The sign reverses in fold 1, which is why the single-fold result of the original
submission did not survive extension.

`pairing_checks` records the integrity constraints the script enforces: identical case
sets on both sides, each case in exactly one fold, and counts matching the folds file.
The script exits non-zero if any fails.

Figure 10 is drawn from `per_case_difference` (panel A) and `per_fold` plus `pooled`
(panel B).

---

## Table 8 — Exact pruning (Section 4.8)

Two files, with different key names.

**Equivalence and parameter counts** — `results/metrics/pruned_fold{0,1,2}.json`:

| Published | Key | Value |
|---|---|---|
| Parameters, trained | `parameters.intact_total` | 22,627,875 |
| Parameters, pruned | `parameters.pruned_total` | 10,959,139 |
| Parameters removed | `parameters.removed_net` | 11,668,736 |
| Reduction | `parameters.reduction_fraction` | 0.51568 (51.6%) |
| Bottleneck share | `parameters.bottleneck_fraction` | 0.469328 (46.9%) |
| Storage, trained / pruned | `storage.intact_parameter_mib` / `.pruned_parameter_mib` | 86.32 / 41.81 MiB |
| Max absolute logit deviation | `equivalence.worst_max_abs_logit_diff_vs_intact` | 0.0 |
| Voxel disagreements | `equivalence.total_mask_disagreements_vs_intact` | 0 |
| Bitwise identical | `equivalence.all_outputs_bitwise_identical_to_ablated` | true |
| BatchNorm diagnostics (Section 4.5) | `bottleneck_batchnorm.bn1` | γ 0.46598, β 3.129 × 10⁻³⁷, running_var 5.605 × 10⁻⁴⁵ |

> **Note on the equivalence inputs.** In the recorded runs, `equivalence.n_real` is 0 and
> `n_synthetic` is 4: the Decathlon cache had been deleted to free disk space when the
> pruning was performed, so the bitwise check ran on synthetic volumes of the full
> 128 × 128 × 128 input shape. The claim that no *real* prediction changes rests on the
> separate 484-case ablation recorded in `prior_ablation_on_record` (and in
> `results/metrics/ablation_fold{0,1,2}.json`), where `cases_changed` is 0 and
> `max_abs_change` is 0.0 for every fold. With the cache restored, re-running
> `prune_bottleneck.py` picks up `--real-cases 8` and repeats the bitwise check on real
> volumes.

**Latency** — `results/metrics/pruned_latency_fold{0,1,2}.json`:

| Published | Key | Value (fold 0) |
|---|---|---|
| Parameters | `parameters.intact` / `.pruned` / `.removed` | 22,627,875 / 10,959,139 / 11,668,736 |
| Median latency, trained | `intact_s.median` | 2.4765 s |
| Median latency, pruned | `pruned_s.median` | 2.2658 s |
| Ratio and CI | `paired.median_ratio_pruned_over_intact`, `paired.median_ratio_95ci` | 0.9229 |
| Reduction | `paired.median_reduction_percent` | — |
| Pairs pruned was faster | `paired.pairs_pruned_faster` / `paired.n_pairs` | 73/75 pooled over three folds |
| Wilcoxon p | `paired.wilcoxon.p_two_sided` | 8.9 × 10⁻¹⁴ pooled |
| Order-effect check | `order_effect` | +0.034, +0.004, −0.034 s across folds |

The pooled figures quoted in the manuscript (median 2.479 → 2.263 s, −8.4%, 95% CI
7.7–9.6%) come from pooling `raw.intact_s` and `raw.pruned_s` across the three fold
files:

```bash
python - <<'EOF'
import json, numpy as np
from scipy.stats import wilcoxon
I, P = [], []
for f in range(3):
    d = json.load(open(f"results/metrics/pruned_latency_fold{f}.json"))
    I += d["raw"]["intact_s"]; P += d["raw"]["pruned_s"]
I, P = np.array(I), np.array(P)
r = P / I
rng = np.random.default_rng(42)
boot = [np.median(r[rng.integers(0, len(r), len(r))]) for _ in range(10000)]
print("median %.4f -> %.4f" % (np.median(I), np.median(P)))
print("reduction %.2f%%  CI %.1f-%.1f%%" % (100*(1-np.median(r)),
      100*(1-np.percentile(boot, 97.5)), 100*(1-np.percentile(boot, 2.5))))
print("faster in %d of %d  p=%.3g" % ((P < I).sum(), len(I), wilcoxon(I, P).pvalue))
EOF
```

Bit-for-bit equivalence is asserted inside `scripts/prune_bottleneck.py`, which exits
non-zero if the pruned network's logits differ from the zero-ablated network's at all.
`tests/test_prune_logic.py` tests the same algebra on a small network that can run
anywhere, without the trained checkpoints.

---

## Section 3.16 — Deterministic rebuild

File: `results/metrics/cache_verification.json`.

| Published | Key | Value |
|---|---|---|
| Cases compared | `n_compared` | 162 |
| Maximum absolute deviation | `max_abs_deviation` | 0.0 |
| Mean / median deviation | `mean_abs_deviation` / `median_abs_deviation` | 0.0 / 0.0 |
| Tolerance used | `tolerance` | 1 × 10⁻⁴ |
| Passed | `passed` | true |

The reference is the per-case Dice recorded in `results/metrics/cross_population.json` at
submission time. `verify_restored_cache.py` exits non-zero above the tolerance, so a
successful run is itself the evidence.

---

## Section 4.5 — The attribution-target defect

File: `results/metrics/explainer_target_ablation.json`.

| Published | Key | Value |
|---|---|---|
| Whole-volume target (as submitted) | `summary.whole_volume.mean` | 0.3200 |
| Predicted-mask target (corrected) | `summary.predicted.mean` | 0.6221 |
| Both reproduce the recorded values | `reproduction_checks` | `true`, `true` |
| Correlation with tumour size | `tumour_size_effect.pearson_predvoxels_vs_delta` | +0.5497 |
| Softmax change is inert | `change_1_softmax_status` | `"INERT — forward() already returns logits for this model class"` |

The last row matters: the change from `forward()` to `forward_logits()` made no
difference for this model class, so the whole effect is attributable to the target
definition alone. The script holds the backward mechanism, normalisation and agreement
metric identical across the two configurations.

---

## Section 4.5 — Grad-CAM layer scan

File: `results/metrics/gradcam_layer_scan.json`, key root `results.<layer>.summary`.

Layers scanned: `bottleneck` (control), `encoder.4`, `decoder.0`, `decoder.1`,
`decoder.2`, `decoder.3`.

| Published | Key |
|---|---|
| Degenerate in all 5 cases at the bottleneck | `results.bottleneck.summary.degenerate_cases` = 5 |
| Non-degenerate at every live layer | `results.<layer>.summary.viable` |
| Mean maximum activation per layer | `results.<layer>.summary.activation_max_abs_mean` |
| Agreement with Vanilla Gradient | `results.<layer>.summary.mean_pearson_vs_vg` |

| Layer | Max activation | Agreement with VG |
|---|---|---|
| `bottleneck` (control) | 0.0 | null — map degenerate |
| `encoder.4` | 6.583887 | +0.1189 |
| `decoder.0` | 12.704059 | −0.0215 |
| `decoder.1` | 16.205222 | +0.2162 |
| `decoder.2` | 23.728281 | +0.1556 |
| `decoder.3` | 32.508068 | −0.1915 |

Grad-CAM is viable in this network, but its agreement with Vanilla Gradient ranges from
−0.192 to +0.216 and is negative at two of the five live layers, purely as a function of
which layer is hooked. That is why it remains withdrawn: no layer choice is defensible
without localisation validation against ground truth, which was not performed.

---

## Not reproducible from this repository

| Item | Why |
|---|---|
| Table 3 and Section 4.4 | Require BraTS-Africa, which cannot be redistributed. Obtain it from Synapse under the data-use agreement, then build the cache with `scripts/cache_bratsafrica.py --data-root <your copy>`. |
| Figures 3 and 8 | Interface screenshot and qualitative overlays generated before the attribution-target correction. Regenerate with `scripts/plot_vg_ig_comparison.py --cases <ids>`; the current code produces corrected maps. |
| Clinician-evaluation results | Evaluation data is held locally and is not published. The module code is in `src/evaluation/`. |
