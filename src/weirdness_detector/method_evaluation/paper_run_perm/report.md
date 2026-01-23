# Cutoff Calibration & Feature Influence Report

## Overview
Goal: learn an automated coherence filter for synthetically generated sentences, using sentence-level scoring features and noisy human labels.

This report documents a full pipeline:
1. k-fold cross-validated training of a logistic regression (batch GD, L2).
2. Precision-constrained threshold selection (raw precision or Wilson LCB).
3. Feature influence analysis via standardized weights.
4. Feature influence analysis via permutation importance.
5. Automatic ablation / pruning.
6. Final model fitting and export for pipeline use.

## Data Filtering
- Input CSV: `dev.csv`
- Raw rows: 4794
- Kept evaluable examples: 3729
- Ignored (non-numeric human_label): 0
- Ignored (borderline labels): 1065
- Ignored (missing/non-numeric feature): 0

## Experimental Settings
- `pos_threshold`: 2.0
- `neg_threshold`: -1.0
- `kfold`: 5
- `seed`: 0
- `steps`: 201
- `lr`: 0.05
- `epochs`: 4000
- `l2`: 0.001
- `use_lcb`: True
- `min_precision`: 0.0
- `min_precision_lcb`: 0.9
- `perm_repeats`: 5
- `perm_seed`: 123
- `objective`: maximize coherent_kept_rate subject to precision constraint
- `dataset`: {'usable_examples': 3729, 'coherent': 2273, 'incoherent': 1456, 'prevalence': 0.6095467953875033}
- `selected_step`: 6
- `prune_tol`: 0.01

## Step 1 — Baseline Model (All Features)
- Features used: 18
- Chosen threshold (median across folds): **t = 0.8650**

Performance at chosen thresholds (across folds):
- coherent_kept_rate: **0.0644 ± 0.0217**
- precision: **0.9189 ± 0.0125**
- precision_LCB: **0.8092 ± 0.0257**
- coverage: **0.0700 ± 0.0232**
- recall: **0.1059 ± 0.0367**

## Step 2 — Feature Influence (Standardized Weights)
Weights are in standardized feature space (train-fold z-scoring), so magnitudes are comparable. Sign consistency indicates stability across folds.

| rank | feature | mean_abs(weight) | mean(weight) ± std | sign_consistency |
|---:|---|---:|---:|---:|
| 1 | `n_tokens` | 0.9001 | -0.9001 ± 0.0139 | 1.00 |
| 2 | `avg_nll` | 0.2945 | -0.2945 ± 0.0522 | 1.00 |
| 3 | `max_surp` | 0.1856 | -0.1856 ± 0.0296 | 1.00 |
| 4 | `spike_frac_35` | 0.1732 | -0.1732 ± 0.0551 | 1.00 |
| 5 | `p95` | 0.1700 | -0.1700 ± 0.0178 | 1.00 |
| 6 | `top1_prob_mean` | 0.1663 | 0.1663 ± 0.0446 | 1.00 |
| 7 | `rep_1gram` | 0.1650 | 0.1650 ± 0.0803 | 1.00 |
| 8 | `p99` | 0.1483 | -0.1483 ± 0.0201 | 1.00 |
| 9 | `top1_minus_true_mean` | 0.1469 | -0.1469 ± 0.0314 | 1.00 |
| 10 | `digit_ratio` | 0.1343 | 0.1343 ± 0.0210 | 1.00 |
| 11 | `upper_ratio` | 0.1273 | -0.1273 ± 0.0357 | 1.00 |
| 12 | `rep_2gram` | 0.0527 | 0.0527 ± 0.0463 | 1.00 |
| 13 | `spike_frac_50` | 0.0514 | 0.0514 ± 0.0173 | 1.00 |
| 14 | `punct_ratio` | 0.0324 | -0.0260 ± 0.0392 | 0.80 |
| 15 | `run_norm_35` | 0.0174 | -0.0059 ± 0.0222 | 0.60 |
| 16 | `uniq_token_ratio` | 0.0159 | 0.0083 ± 0.0191 | 0.60 |
| 17 | `paren_imbalance_abs` | 0.0000 | 0.0000 ± 0.0000 | 0.00 |
| 18 | `bracket_imbalance_abs` | 0.0000 | 0.0000 ± 0.0000 | 0.00 |

## Step 3 — Feature Influence (Permutation Importance)
Permutation importance measures the drop in performance when a single feature is randomized within each validation fold (repeated shuffles). We report mean±std drops across folds.

Primary importance metric shown: **Δ coherent_kept_rate** (baseline minus permuted).

| rank | feature | Δ coherent_kept_rate (mean ± std) | Δ precision_LCB (mean ± std) |
|---:|---|---:|---:|
| 1 | `p95` | 0.0196 ± 0.0057 | 0.0546 ± 0.0194 |
| 2 | `p99` | 0.0181 ± 0.0060 | 0.0529 ± 0.0189 |
| 3 | `max_surp` | 0.0130 ± 0.0069 | 0.0428 ± 0.0168 |
| 4 | `avg_nll` | 0.0009 ± 0.0101 | 0.0603 ± 0.0354 |
| 5 | `rep_1gram` | 0.0006 ± 0.0012 | 0.0116 ± 0.0152 |
| 6 | `run_norm_35` | 0.0002 ± 0.0012 | 0.0083 ± 0.0082 |
| 7 | `rep_2gram` | 0.0002 ± 0.0012 | -0.0006 ± 0.0063 |
| 8 | `uniq_token_ratio` | 0.0002 ± 0.0006 | 0.0058 ± 0.0119 |
| 9 | `paren_imbalance_abs` | 0.0000 ± 0.0000 | -0.0000 ± 0.0000 |
| 10 | `bracket_imbalance_abs` | 0.0000 ± 0.0000 | -0.0000 ± 0.0000 |
| 11 | `spike_frac_50` | -0.0004 ± 0.0025 | 0.0117 ± 0.0096 |
| 12 | `upper_ratio` | -0.0023 ± 0.0043 | 0.0241 ± 0.0218 |
| 13 | `punct_ratio` | -0.0028 ± 0.0048 | 0.0096 ± 0.0166 |
| 14 | `top1_minus_true_mean` | -0.0058 ± 0.0017 | 0.0504 ± 0.0334 |
| 15 | `digit_ratio` | -0.0064 ± 0.0024 | 0.0111 ± 0.0085 |
| 16 | `top1_prob_mean` | -0.0068 ± 0.0051 | 0.0376 ± 0.0215 |
| 17 | `spike_frac_35` | -0.0076 ± 0.0079 | 0.0417 ± 0.0258 |
| 18 | `n_tokens` | -0.0192 ± 0.0126 | 0.1915 ± 0.0338 |

## Steps 4–5 — Automatic Ablation / Pruning
We iteratively remove the weakest remaining feature (by mean_abs standardized weight), re-train, and evaluate. This approximates a greedy backward feature elimination.

| step | n_features | removed | coherent_kept_rate | precision | precision_LCB | coverage | recall | chosen_t |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 18 | `(none)` | 0.0644 | 0.9189 | 0.8092 | 0.0700 | 0.1059 | 0.8650 |
| 1 | 17 | `bracket_imbalance_abs` | 0.0644 | 0.9189 | 0.8092 | 0.0700 | 0.1059 | 0.8650 |
| 2 | 16 | `paren_imbalance_abs` | 0.0644 | 0.9189 | 0.8092 | 0.0700 | 0.1059 | 0.8650 |
| 3 | 15 | `uniq_token_ratio` | 0.0963 | 0.9075 | 0.8129 | 0.1086 | 0.1568 | 0.8650 |
| 4 | 14 | `run_norm_35` | 0.0681 | 0.9157 | 0.8081 | 0.0743 | 0.1120 | 0.8700 |
| 5 | 13 | `punct_ratio` | 0.1022 | 0.9011 | 0.8107 | 0.1156 | 0.1663 | 0.8500 |
| 6 | 12 | `rep_2gram` | 0.1030 | 0.9042 | 0.8143 | 0.1164 | 0.1675 | 0.8500 |
| 7 | 11 | `spike_frac_50` | 0.0904 | 0.9235 | 0.8149 | 0.1035 | 0.1463 | 0.8700 |
| 8 | 10 | `digit_ratio` | 0.0834 | 0.9134 | 0.8171 | 0.0928 | 0.1359 | 0.8600 |
| 9 | 9 | `upper_ratio` | 0.0679 | 0.9361 | 0.8201 | 0.0746 | 0.1106 | 0.8550 |
| 10 | 8 | `top1_minus_true_mean` | 0.0579 | 0.9280 | 0.8141 | 0.0630 | 0.0951 | 0.8650 |
| 11 | 7 | `top1_prob_mean` | 0.0705 | 0.9236 | 0.8220 | 0.0772 | 0.1158 | 0.8550 |
| 12 | 6 | `p99` | 0.0705 | 0.9236 | 0.8220 | 0.0772 | 0.1158 | 0.8550 |
| 13 | 5 | `rep_1gram` | 0.0968 | 0.9133 | 0.8215 | 0.1094 | 0.1597 | 0.8450 |
| 14 | 4 | `max_surp` | 0.0493 | 0.9562 | 0.8257 | 0.0528 | 0.0812 | 0.8900 |
| 15 | 3 | `spike_frac_35` | 0.0735 | 0.9306 | 0.8300 | 0.0799 | 0.1202 | 0.8450 |
| 16 | 2 | `n_tokens` | 0.0869 | 0.8958 | 0.7694 | 0.1019 | 0.1426 | 0.7750 |

## Step 6 — Final Selected Feature Set
- Selected features: 12
```text
n_tokens, avg_nll, p95, p99, max_surp, spike_frac_35, spike_frac_50, rep_1gram, top1_prob_mean, top1_minus_true_mean, digit_ratio, upper_ratio
```
- Chosen threshold (median across folds): **t = 0.8500**

Final performance at chosen thresholds (across folds):
- coherent_kept_rate: **0.1030 ± 0.0627**
- precision: **0.9042 ± 0.0374**
- precision_LCB: **0.8143 ± 0.0227**
- coverage: **0.1164 ± 0.0776**
- recall: **0.1675 ± 0.0982**

### Final weights (standardized) — top 15 by |weight|

| rank | feature | mean_abs(weight) | mean(weight) ± std |
|---:|---|---:|---:|
| 1 | `n_tokens` | 0.8883 | -0.8883 ± 0.0258 |
| 2 | `avg_nll` | 0.2980 | -0.2980 ± 0.0518 |
| 3 | `max_surp` | 0.1839 | -0.1839 ± 0.0277 |
| 4 | `spike_frac_35` | 0.1733 | -0.1733 ± 0.0488 |
| 5 | `p95` | 0.1709 | -0.1709 ± 0.0151 |
| 6 | `top1_prob_mean` | 0.1658 | 0.1658 ± 0.0453 |
| 7 | `rep_1gram` | 0.1620 | 0.1620 ± 0.0807 |
| 8 | `top1_minus_true_mean` | 0.1485 | -0.1485 ± 0.0315 |
| 9 | `p99` | 0.1462 | -0.1462 ± 0.0181 |
| 10 | `upper_ratio` | 0.1420 | -0.1420 ± 0.0261 |
| 11 | `digit_ratio` | 0.1285 | 0.1285 ± 0.0235 |
| 12 | `spike_frac_50` | 0.0505 | 0.0505 ± 0.0176 |

### Final permutation importance — top 15 by Δ coherent_kept_rate

| rank | feature | Δ coherent_kept_rate (mean ± std) | Δ precision_LCB (mean ± std) |
|---:|---|---:|---:|
| 1 | `p95` | 0.0239 ± 0.0087 | 0.0305 ± 0.0079 |
| 2 | `p99` | 0.0199 ± 0.0049 | 0.0171 ± 0.0142 |
| 3 | `max_surp` | 0.0168 ± 0.0093 | 0.0383 ± 0.0226 |
| 4 | `avg_nll` | 0.0033 ± 0.0066 | 0.0493 ± 0.0210 |
| 5 | `rep_1gram` | 0.0010 ± 0.0014 | 0.0149 ± 0.0154 |
| 6 | `upper_ratio` | -0.0014 ± 0.0040 | 0.0268 ± 0.0334 |
| 7 | `spike_frac_50` | -0.0020 ± 0.0017 | 0.0105 ± 0.0075 |
| 8 | `top1_prob_mean` | -0.0027 ± 0.0063 | 0.0353 ± 0.0206 |
| 9 | `n_tokens` | -0.0046 ± 0.0136 | 0.1947 ± 0.0537 |
| 10 | `digit_ratio` | -0.0049 ± 0.0014 | 0.0178 ± 0.0072 |
| 11 | `spike_frac_35` | -0.0049 ± 0.0062 | 0.0385 ± 0.0295 |
| 12 | `top1_minus_true_mean` | -0.0073 ± 0.0038 | 0.0352 ± 0.0254 |

## Reproducibility Notes
- Model: logistic regression trained with batch gradient descent; L2 regularization.
- Features standardized using training-fold mean/std; applied to validation fold.
- Threshold selected per fold to maximize coherent kept subject to precision constraint.
- Wilson LCB constraint recommended for noisy labels.
- Permutation importance computed on validation folds at fold-chosen threshold (kept fixed).
