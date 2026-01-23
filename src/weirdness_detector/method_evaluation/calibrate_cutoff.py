#!/usr/bin/env python3
"""
calibrate_cutoff.py

Automatic cutoff calibration + feature influence analysis for filtering coherent synthetic sentences.

Pipeline (automated):
1) Train + evaluate with ALL features (k-fold CV).
2) Feature influence via standardized weights + fold stability.
3) Automatic ablation: iteratively remove weakest feature, re-train/evaluate.
4) Track performance vs feature count during pruning.
5) Select minimal feature set that preserves performance within tolerance.
6) Fit final model on ALL data with selected features and export model JSON for pipeline.

Adds permutation importance (paper-friendly) and writes:
- report.md
- weights_all.csv, weights_final.csv
- ablation.csv
- curve_all.csv, curve_final.csv
- perm_importance_all.csv, perm_importance_final.csv
- model.json

Progress printing:
- Stage markers in main()
- Per-fold progress during CV
- Throttled progress inside permutation importance (about ~50 updates)

Input CSV requirements:
  - must include columns: sentence, human_label
  - any number of additional numeric feature columns
  - rows with non-numeric human_label ignored
  - borderline labels excluded:
      coherent   if human_label >= --pos-threshold
      incoherent if human_label <= --neg-threshold
      otherwise ignored

Definitions:
  - coverage: fraction of evaluable examples kept by cutoff.
  - precision: fraction of kept that are coherent.
  - recall: fraction of coherent examples kept.

Example:
  python calibrate_cutoff.py scored_features.csv \
    --kfold 5 --seed 0 \
    --pos-threshold 2 --neg-threshold -1 \
    --use-lcb --min-precision-lcb 0.90 \
    --steps 201 --perm-repeats 5 \
    --prune-tol 0.01 \
    --outdir calib_out
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def parse_float(value: str) -> Optional[float]:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def dot(w: List[float], x: List[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return m, math.sqrt(var)


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Example:
    x: List[float]
    y: int  # 1 coherent, 0 incoherent


@dataclass
class MetricsAtT:
    t: float
    coverage: float
    precision: float
    recall: float
    incoherent_rate: float
    kept: int
    coh_kept: int
    inc_kept: int
    total: int
    total_coh: int

    @property
    def coherent_kept_rate(self) -> float:
        return self.coh_kept / self.total if self.total else 0.0

    @property
    def precision_lcb(self) -> float:
        return wilson_lower_bound(self.coh_kept, self.kept) if self.kept else 0.0


@dataclass
class FoldResult:
    weights: List[float]     # standardized weights, includes bias
    means: List[float]
    stds: List[float]
    chosen: MetricsAtT       # chosen threshold metrics on validation fold

    # Stored for permutation importance
    X_val_raw: List[List[float]]
    y_val: List[int]
    p_val: List[float]       # baseline probabilities on val (unpermuted)


@dataclass
class EvalResult:
    feature_names: List[str]
    fold_results: List[FoldResult]
    curve_by_t: Dict[float, List[MetricsAtT]]  # threshold -> list across folds

    chosen_t_median: float
    coherent_kept_rate_mean: float
    coherent_kept_rate_std: float
    precision_mean: float
    precision_std: float
    precision_lcb_mean: float
    precision_lcb_std: float
    coverage_mean: float
    coverage_std: float
    recall_mean: float
    recall_std: float


@dataclass
class WeightSummary:
    feature: str
    mean: float
    std: float
    mean_abs: float
    sign_consistency: float


@dataclass
class PermImportanceRow:
    feature: str
    delta_coherent_kept_rate_mean: float
    delta_coherent_kept_rate_std: float
    delta_precision_mean: float
    delta_precision_std: float
    delta_precision_lcb_mean: float
    delta_precision_lcb_std: float
    delta_coverage_mean: float
    delta_coverage_std: float
    delta_recall_mean: float
    delta_recall_std: float


# -----------------------------
# Standardization
# -----------------------------

def standardize(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not X:
        return X, [], []
    n = len(X)
    d = len(X[0])

    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds: List[float] = []
    for j in range(d):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(1, n - 1)
        sd = math.sqrt(var)
        stds.append(sd if sd > 1e-8 else 1.0)

    Xs = [[(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]
    return Xs, means, stds


def add_bias(X: List[List[float]]) -> List[List[float]]:
    return [[1.0] + row for row in X]


# -----------------------------
# Logistic regression training (batch GD, L2)
# -----------------------------

def train_logreg(
    Xb: List[List[float]],
    y: List[int],
    lr: float,
    epochs: int,
    l2: float,
    seed: int,
) -> List[float]:
    n = len(Xb)
    d = len(Xb[0])
    w = [0.0] * d
    idxs = list(range(n))
    rng = random.Random(seed)

    for _ in range(epochs):
        rng.shuffle(idxs)
        grad = [0.0] * d

        for i in idxs:
            xi = Xb[i]
            yi = y[i]
            p = sigmoid(dot(w, xi))
            err = p - yi
            for j in range(d):
                grad[j] += err * xi[j]

        for j in range(d):
            grad[j] /= n

        if l2 > 0.0:
            for j in range(1, d):
                grad[j] += l2 * w[j]

        for j in range(d):
            w[j] -= lr * grad[j]

    return w


def predict_proba_raw(x_raw: List[float], w: List[float], means: List[float], stds: List[float]) -> float:
    z = [(x_raw[j] - means[j]) / stds[j] for j in range(len(x_raw))]
    xb = [1.0] + z
    return sigmoid(dot(w, xb))


# -----------------------------
# Threshold evaluation and selection
# -----------------------------

def metrics_at_threshold(y_true: List[int], p: List[float], t: float) -> MetricsAtT:
    total = len(y_true)
    total_coh = sum(y_true)
    kept_idx = [i for i, pi in enumerate(p) if pi >= t]
    kept = len(kept_idx)
    coh_kept = sum(y_true[i] for i in kept_idx)
    inc_kept = kept - coh_kept

    coverage = kept / total if total else 0.0
    precision = coh_kept / kept if kept else 0.0
    recall = coh_kept / total_coh if total_coh else 0.0
    incoh = 1.0 - precision if kept else 0.0

    return MetricsAtT(
        t=t,
        coverage=coverage,
        precision=precision,
        recall=recall,
        incoherent_rate=incoh,
        kept=kept,
        coh_kept=coh_kept,
        inc_kept=inc_kept,
        total=total,
        total_coh=total_coh,
    )


def eval_thresholds(y_true: List[int], p: List[float], steps: int) -> List[MetricsAtT]:
    thresholds = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.5]
    return [metrics_at_threshold(y_true, p, t) for t in thresholds]


def choose_threshold(
    curve: List[MetricsAtT],
    use_lcb: bool,
    min_precision: float,
    min_precision_lcb: float,
) -> Optional[MetricsAtT]:
    best: Optional[MetricsAtT] = None
    for m in curve:
        if m.kept == 0:
            continue

        if use_lcb:
            if m.precision_lcb < min_precision_lcb:
                continue
        else:
            if m.precision < min_precision:
                continue

        # primary objective: keep as many coherent examples as possible
        if best is None or m.coh_kept > best.coh_kept:
            best = m
    return best


# -----------------------------
# k-fold splitting
# -----------------------------

def make_kfold_indices(n: int, k: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)
    return folds


# -----------------------------
# CSV reading (auto feature detection)
# -----------------------------

def read_labeled_csv(
    path: Path,
    pos_threshold: float,
    neg_threshold: float,
    min_numeric_frac: float,
) -> Tuple[List[Example], List[str], Dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit("CSV has no header row.")

        lower_map = {name.strip().lower(): name for name in reader.fieldnames}
        if "sentence" not in lower_map or "human_label" not in lower_map:
            raise SystemExit("CSV must include columns: sentence, human_label")

        sentence_col = lower_map["sentence"]
        label_col = lower_map["human_label"]

        candidate_cols = [c for c in reader.fieldnames if c not in {sentence_col, label_col}]
        raw_rows = [row for row in reader]

    kept_features: List[str] = []
    for col in candidate_cols:
        non_empty = 0
        numeric = 0
        for row in raw_rows:
            v = (row.get(col, "") or "").strip()
            if not v:
                continue
            non_empty += 1
            if parse_float(v) is not None:
                numeric += 1
        if non_empty == 0:
            continue
        if numeric / non_empty >= min_numeric_frac:
            kept_features.append(col)

    if not kept_features:
        raise SystemExit("No numeric feature columns found (besides human_label).")

    ignored_non_numeric_label = 0
    ignored_borderline = 0
    ignored_missing_feature = 0
    kept = 0

    examples: List[Example] = []
    for row in raw_rows:
        lab = parse_float(row.get(label_col, "") or "")
        if lab is None:
            ignored_non_numeric_label += 1
            continue

        if lab >= pos_threshold:
            y = 1
        elif lab <= neg_threshold:
            y = 0
        else:
            ignored_borderline += 1
            continue

        x: List[float] = []
        ok = True
        for col in kept_features:
            v = parse_float(row.get(col, "") or "")
            if v is None:
                ok = False
                break
            x.append(float(v))
        if not ok:
            ignored_missing_feature += 1
            continue

        examples.append(Example(x=x, y=y))
        kept += 1

    stats = {
        "raw_rows": len(raw_rows),
        "kept_examples": kept,
        "ignored_non_numeric_label": ignored_non_numeric_label,
        "ignored_borderline": ignored_borderline,
        "ignored_missing_feature": ignored_missing_feature,
    }
    return examples, kept_features, stats


# -----------------------------
# Feature subset helpers
# -----------------------------

def subset_examples(examples: List[Example], keep_idx: List[int]) -> List[Example]:
    return [Example(x=[e.x[i] for i in keep_idx], y=e.y) for e in examples]


# -----------------------------
# Evaluation for a given feature set
# -----------------------------

def evaluate_feature_set(
    examples: List[Example],
    feature_names: List[str],
    *,
    kfold: int,
    seed: int,
    steps: int,
    lr: float,
    epochs: int,
    l2: float,
    use_lcb: bool,
    min_precision: float,
    min_precision_lcb: float,
    progress_prefix: str = "",
) -> EvalResult:
    n = len(examples)
    k = max(2, kfold)
    if k > n:
        raise SystemExit(f"kfold={k} is larger than number of usable rows={n}.")

    folds = make_kfold_indices(n, k, seed)

    fold_results: List[FoldResult] = []
    curve_by_t: Dict[float, List[MetricsAtT]] = {}

    t_start = time.time()
    for fold_idx in range(k):
        fold_t0 = time.time()
        val_idx = set(folds[fold_idx])
        train = [examples[i] for i in range(n) if i not in val_idx]
        val = [examples[i] for i in range(n) if i in val_idx]

        X_train = [e.x for e in train]
        y_train = [e.y for e in train]

        Xs_train, means, stds = standardize(X_train)
        Xb_train = add_bias(Xs_train)

        w = train_logreg(
            Xb_train,
            y_train,
            lr=lr,
            epochs=epochs,
            l2=l2,
            seed=seed + fold_idx,
        )

        X_val_raw = [e.x for e in val]
        y_val = [e.y for e in val]
        p_val = [predict_proba_raw(x, w, means, stds) for x in X_val_raw]

        curve = eval_thresholds(y_val, p_val, steps=steps)
        for m in curve:
            curve_by_t.setdefault(m.t, []).append(m)

        chosen = choose_threshold(
            curve,
            use_lcb=use_lcb,
            min_precision=min_precision,
            min_precision_lcb=min_precision_lcb,
        )

        if chosen is None:
            # fallback: best available
            if use_lcb:
                chosen = max((m for m in curve if m.kept > 0), key=lambda m: m.precision_lcb)
            else:
                chosen = max((m for m in curve if m.kept > 0), key=lambda m: m.precision)

        fold_results.append(
            FoldResult(
                weights=w,
                means=means,
                stds=stds,
                chosen=chosen,
                X_val_raw=X_val_raw,
                y_val=y_val,
                p_val=p_val,
            )
        )

        fold_dt = time.time() - fold_t0
        elapsed = time.time() - t_start
        print(
            f"{progress_prefix}  fold {fold_idx+1}/{k}: "
            f"t*={chosen.t:.3f} kept={chosen.kept}/{chosen.total} "
            f"prec={chosen.precision:.3f} lcb={chosen.precision_lcb:.3f} "
            f"cov={chosen.coverage:.3f} rec={chosen.recall:.3f} "
            f"({fold_dt:.1f}s, elapsed {elapsed:.1f}s)"
        )

    chosen_ts = [fr.chosen.t for fr in fold_results]
    chosen_t_median = float(median(chosen_ts))

    coherent_kept_rates = [fr.chosen.coherent_kept_rate for fr in fold_results]
    precisions = [fr.chosen.precision for fr in fold_results]
    lcbs = [fr.chosen.precision_lcb for fr in fold_results]
    coverages = [fr.chosen.coverage for fr in fold_results]
    recalls = [fr.chosen.recall for fr in fold_results]

    ck_m, ck_s = mean_std(coherent_kept_rates)
    p_m, p_s = mean_std(precisions)
    l_m, l_s = mean_std(lcbs)
    c_m, c_s = mean_std(coverages)
    r_m, r_s = mean_std(recalls)

    return EvalResult(
        feature_names=feature_names,
        fold_results=fold_results,
        curve_by_t=curve_by_t,
        chosen_t_median=chosen_t_median,
        coherent_kept_rate_mean=ck_m,
        coherent_kept_rate_std=ck_s,
        precision_mean=p_m,
        precision_std=p_s,
        precision_lcb_mean=l_m,
        precision_lcb_std=l_s,
        coverage_mean=c_m,
        coverage_std=c_s,
        recall_mean=r_m,
        recall_std=r_s,
    )


# -----------------------------
# Feature influence: weights summary
# -----------------------------

def summarize_weights(eval_res: EvalResult) -> List[WeightSummary]:
    k = len(eval_res.fold_results)
    d = len(eval_res.feature_names)

    w_lists: List[List[float]] = [[] for _ in range(d)]
    for fr in eval_res.fold_results:
        w = fr.weights
        for j in range(d):
            w_lists[j].append(w[j + 1])

    summaries: List[WeightSummary] = []
    for j, name in enumerate(eval_res.feature_names):
        ws = w_lists[j]
        m, s = mean_std(ws)
        ma = sum(abs(x) for x in ws) / len(ws) if ws else 0.0
        mean_sign = 1.0 if m > 0 else (-1.0 if m < 0 else 0.0)
        if mean_sign == 0.0:
            sign_consistency = 0.0
        else:
            sign_consistency = sum(
                1 for x in ws
                if (x > 0 and mean_sign > 0) or (x < 0 and mean_sign < 0)
            ) / len(ws)
        summaries.append(
            WeightSummary(
                feature=name,
                mean=m,
                std=s,
                mean_abs=ma,
                sign_consistency=sign_consistency,
            )
        )

    summaries.sort(key=lambda s: s.mean_abs, reverse=True)
    return summaries


# -----------------------------
# Permutation importance
# -----------------------------

def permutation_importance(
    eval_res: EvalResult,
    *,
    perm_repeats: int,
    seed: int,
    progress_label: str = "perm",
) -> List[PermImportanceRow]:
    """
    Permutation importance computed on validation folds.

    For each feature j:
      - For each fold:
          - Repeat R times:
              - Shuffle feature j across validation examples
              - Recompute p(x) with the trained fold model
              - Evaluate metrics at the fold-chosen threshold t_fold (kept fixed)
          - Average metrics across repeats
      - Compare to baseline fold metrics and record mean deltas.

    Deltas:
      delta = baseline_metric - permuted_metric
    So higher delta means "more important".
    """
    rng = random.Random(seed)
    feature_names = eval_res.feature_names
    d = len(feature_names)
    k = len(eval_res.fold_results)

    total_shuffles = k * d * perm_repeats
    done_shuffles = 0
    t0 = time.time()

    # throttle prints to about ~50 updates
    report_every = max(1, total_shuffles // 50)
    print(
        f"[{progress_label}] permutation importance: folds={k} features={d} repeats={perm_repeats} "
        f"total_shuffles={total_shuffles}"
    )

    per_feat_deltas: Dict[int, Dict[str, List[float]]] = {
        j: {"ck": [], "prec": [], "lcb": [], "cov": [], "rec": []} for j in range(d)
    }

    for fold_idx, fr in enumerate(eval_res.fold_results):
        X_val = fr.X_val_raw
        y_val = fr.y_val
        t_fold = fr.chosen.t

        baseline = metrics_at_threshold(y_val, fr.p_val, t_fold)

        n_val = len(X_val)
        if n_val <= 1:
            continue

        cols = [[X_val[i][j] for i in range(n_val)] for j in range(d)]

        for j in range(d):
            ck_vals: List[float] = []
            prec_vals: List[float] = []
            lcb_vals: List[float] = []
            cov_vals: List[float] = []
            rec_vals: List[float] = []

            for _ in range(perm_repeats):
                perm = cols[j][:]
                rng.shuffle(perm)

                p_perm: List[float] = []
                for i in range(n_val):
                    x = X_val[i][:]
                    x[j] = perm[i]
                    p_perm.append(predict_proba_raw(x, fr.weights, fr.means, fr.stds))

                m_perm = metrics_at_threshold(y_val, p_perm, t_fold)

                ck_vals.append(m_perm.coherent_kept_rate)
                prec_vals.append(m_perm.precision)
                lcb_vals.append(m_perm.precision_lcb)
                cov_vals.append(m_perm.coverage)
                rec_vals.append(m_perm.recall)

                done_shuffles += 1
                if done_shuffles % report_every == 0 or done_shuffles == total_shuffles:
                    pct = 100.0 * done_shuffles / total_shuffles
                    dt = time.time() - t0
                    rate = done_shuffles / max(1e-9, dt)
                    eta = (total_shuffles - done_shuffles) / max(1e-9, rate)
                    print(
                        f"[{progress_label}] {done_shuffles}/{total_shuffles} ({pct:5.1f}%) "
                        f"elapsed={dt:6.1f}s eta={eta:6.1f}s "
                        f"(fold {fold_idx+1}/{k}, feat {j+1}/{d})"
                    )

            ck_avg = sum(ck_vals) / len(ck_vals)
            prec_avg = sum(prec_vals) / len(prec_vals)
            lcb_avg = sum(lcb_vals) / len(lcb_vals)
            cov_avg = sum(cov_vals) / len(cov_vals)
            rec_avg = sum(rec_vals) / len(rec_vals)

            per_feat_deltas[j]["ck"].append(baseline.coherent_kept_rate - ck_avg)
            per_feat_deltas[j]["prec"].append(baseline.precision - prec_avg)
            per_feat_deltas[j]["lcb"].append(baseline.precision_lcb - lcb_avg)
            per_feat_deltas[j]["cov"].append(baseline.coverage - cov_avg)
            per_feat_deltas[j]["rec"].append(baseline.recall - rec_avg)

    rows: List[PermImportanceRow] = []
    for j, name in enumerate(feature_names):
        ck_m, ck_s = mean_std(per_feat_deltas[j]["ck"])
        p_m, p_s = mean_std(per_feat_deltas[j]["prec"])
        l_m, l_s = mean_std(per_feat_deltas[j]["lcb"])
        c_m, c_s = mean_std(per_feat_deltas[j]["cov"])
        r_m, r_s = mean_std(per_feat_deltas[j]["rec"])

        rows.append(
            PermImportanceRow(
                feature=name,
                delta_coherent_kept_rate_mean=ck_m,
                delta_coherent_kept_rate_std=ck_s,
                delta_precision_mean=p_m,
                delta_precision_std=p_s,
                delta_precision_lcb_mean=l_m,
                delta_precision_lcb_std=l_s,
                delta_coverage_mean=c_m,
                delta_coverage_std=c_s,
                delta_recall_mean=r_m,
                delta_recall_std=r_s,
            )
        )

    rows.sort(
        key=lambda r: (r.delta_coherent_kept_rate_mean, r.delta_precision_lcb_mean),
        reverse=True,
    )
    return rows


def write_perm_importance_csv(path: Path, rows: List[PermImportanceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "feature",
            "delta_coherent_kept_rate_mean", "delta_coherent_kept_rate_std",
            "delta_precision_mean", "delta_precision_std",
            "delta_precision_lcb_mean", "delta_precision_lcb_std",
            "delta_coverage_mean", "delta_coverage_std",
            "delta_recall_mean", "delta_recall_std",
        ])
        for r in rows:
            w.writerow([
                r.feature,
                f"{r.delta_coherent_kept_rate_mean:.6f}", f"{r.delta_coherent_kept_rate_std:.6f}",
                f"{r.delta_precision_mean:.6f}", f"{r.delta_precision_std:.6f}",
                f"{r.delta_precision_lcb_mean:.6f}", f"{r.delta_precision_lcb_std:.6f}",
                f"{r.delta_coverage_mean:.6f}", f"{r.delta_coverage_std:.6f}",
                f"{r.delta_recall_mean:.6f}", f"{r.delta_recall_std:.6f}",
            ])


# -----------------------------
# Other outputs
# -----------------------------

def write_curve_csv(path: Path, curve_by_t: Dict[float, List[MetricsAtT]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "threshold",
            "coverage_mean", "coverage_std",
            "precision_mean", "precision_std",
            "precision_lcb_mean", "precision_lcb_std",
            "recall_mean", "recall_std",
            "coherent_kept_rate_mean", "coherent_kept_rate_std",
        ])
        for t in sorted(curve_by_t.keys()):
            ms = curve_by_t[t]
            cov_m, cov_s = mean_std([m.coverage for m in ms])
            prec_m, prec_s = mean_std([m.precision for m in ms])
            lcb_m, lcb_s = mean_std([m.precision_lcb for m in ms])
            rec_m, rec_s = mean_std([m.recall for m in ms])
            ck_m, ck_s = mean_std([m.coherent_kept_rate for m in ms])
            w.writerow([
                f"{t:.6f}",
                f"{cov_m:.6f}", f"{cov_s:.6f}",
                f"{prec_m:.6f}", f"{prec_s:.6f}",
                f"{lcb_m:.6f}", f"{lcb_s:.6f}",
                f"{rec_m:.6f}", f"{rec_s:.6f}",
                f"{ck_m:.6f}", f"{ck_s:.6f}",
            ])


def write_weights_csv(path: Path, weight_summaries: List[WeightSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "weight_mean", "weight_std", "weight_mean_abs", "sign_consistency"])
        for s in weight_summaries:
            w.writerow([s.feature, f"{s.mean:.6f}", f"{s.std:.6f}", f"{s.mean_abs:.6f}", f"{s.sign_consistency:.3f}"])


def linear_rule_for_threshold_multi(
    w: List[float], means: List[float], stds: List[float], prob_t: float
) -> Tuple[List[float], float]:
    L = logit(prob_t)
    a_list = [w[j + 1] / stds[j] for j in range(len(means))]
    c = w[0] - sum((w[j + 1] * means[j] / stds[j]) for j in range(len(means))) - L
    return a_list, float(c)


def write_model_json(
    path: Path,
    *,
    threshold: float,
    feature_names: List[str],
    weights: List[float],
    means: List[float],
    stds: List[float],
    a_list: List[float],
    c_bias: float,
    settings: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    w_std = {"bias": float(weights[0])}
    for j, name in enumerate(feature_names):
        w_std[name] = float(weights[j + 1])

    mean_map = {name: float(means[j]) for j, name in enumerate(feature_names)}
    std_map = {name: float(stds[j]) for j, name in enumerate(feature_names)}
    a_map = {name: float(a_list[j]) for j, name in enumerate(feature_names)}

    payload = {
        "schema_version": 4,
        "model_type": "logistic_regression_gd",
        "features": feature_names,
        "threshold_p": float(threshold),
        "weights_standardized": w_std,
        "standardization": {"means": mean_map, "stds": std_map},
        "linear_rule_raw": {
            "a": a_map,
            "c_bias": float(c_bias),
            "keep_if": "sum(a[f]*x[f]) + c_bias >= 0",
        },
        "training_settings": settings,
        "notes": [
            "Compute z[f]=(x[f]-mean[f])/std[f]. Then p=sigmoid(bias + sum(w[f]*z[f])). Keep if p>=threshold_p.",
            "Equivalent raw rule: sum(a[f]*x[f]) + c_bias >= 0 (same as p>=threshold_p).",
        ],
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_report_md(
    path: Path,
    *,
    input_csv: str,
    stats: Dict[str, int],
    all_eval: EvalResult,
    all_weights: List[WeightSummary],
    perm_all: List[PermImportanceRow],
    ablation_rows: List[Dict[str, object]],
    final_eval: EvalResult,
    final_features: List[str],
    final_weights: List[WeightSummary],
    perm_final: List[PermImportanceRow],
    settings: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_pm(m: float, s: float) -> str:
        return f"{m:.4f} ± {s:.4f}"

    lines: List[str] = []
    lines.append("# Cutoff Calibration & Feature Influence Report\n\n")

    lines.append("## Overview\n")
    lines.append(
        "Goal: learn an automated coherence filter for synthetically generated sentences, "
        "using sentence-level scoring features and noisy human labels.\n\n"
    )
    lines.append("This report documents a full pipeline:\n")
    lines.append("1. k-fold cross-validated training of a logistic regression (batch GD, L2).\n")
    lines.append("2. Precision-constrained threshold selection (raw precision or Wilson LCB).\n")
    lines.append("3. Feature influence analysis via standardized weights.\n")
    lines.append("4. Feature influence analysis via permutation importance.\n")
    lines.append("5. Automatic ablation / pruning.\n")
    lines.append("6. Final model fitting and export for pipeline use.\n\n")

    lines.append("## Data Filtering\n")
    lines.append(f"- Input CSV: `{input_csv}`\n")
    lines.append(f"- Raw rows: {stats['raw_rows']}\n")
    lines.append(f"- Kept evaluable examples: {stats['kept_examples']}\n")
    lines.append(f"- Ignored (non-numeric human_label): {stats['ignored_non_numeric_label']}\n")
    lines.append(f"- Ignored (borderline labels): {stats['ignored_borderline']}\n")
    lines.append(f"- Ignored (missing/non-numeric feature): {stats['ignored_missing_feature']}\n\n")

    lines.append("## Experimental Settings\n")
    for k, v in settings.items():
        lines.append(f"- `{k}`: {v}\n")
    lines.append("\n")

    lines.append("## Step 1 — Baseline Model (All Features)\n")
    lines.append(f"- Features used: {len(all_eval.feature_names)}\n")
    lines.append(f"- Chosen threshold (median across folds): **t = {all_eval.chosen_t_median:.4f}**\n\n")
    lines.append("Performance at chosen thresholds (across folds):\n")
    lines.append(f"- coherent_kept_rate: **{fmt_pm(all_eval.coherent_kept_rate_mean, all_eval.coherent_kept_rate_std)}**\n")
    lines.append(f"- precision: **{fmt_pm(all_eval.precision_mean, all_eval.precision_std)}**\n")
    lines.append(f"- precision_LCB: **{fmt_pm(all_eval.precision_lcb_mean, all_eval.precision_lcb_std)}**\n")
    lines.append(f"- coverage: **{fmt_pm(all_eval.coverage_mean, all_eval.coverage_std)}**\n")
    lines.append(f"- recall: **{fmt_pm(all_eval.recall_mean, all_eval.recall_std)}**\n\n")

    lines.append("## Step 2 — Feature Influence (Standardized Weights)\n")
    lines.append(
        "Weights are in standardized feature space (train-fold z-scoring), so magnitudes are comparable. "
        "Sign consistency indicates stability across folds.\n\n"
    )
    lines.append("| rank | feature | mean_abs(weight) | mean(weight) ± std | sign_consistency |\n")
    lines.append("|---:|---|---:|---:|---:|\n")
    for i, ws in enumerate(all_weights[:25], start=1):
        lines.append(
            f"| {i} | `{ws.feature}` | {ws.mean_abs:.4f} | {ws.mean:.4f} ± {ws.std:.4f} | {ws.sign_consistency:.2f} |\n"
        )
    if len(all_weights) > 25:
        lines.append("\n(Top 25 shown.)\n\n")
    else:
        lines.append("\n")

    lines.append("## Step 3 — Feature Influence (Permutation Importance)\n")
    lines.append(
        "Permutation importance measures the drop in performance when a single feature is randomized "
        "within each validation fold (repeated shuffles). We report mean±std drops across folds.\n\n"
        "Primary importance metric shown: **Δ coherent_kept_rate** (baseline minus permuted).\n\n"
    )
    lines.append("| rank | feature | Δ coherent_kept_rate (mean ± std) | Δ precision_LCB (mean ± std) |\n")
    lines.append("|---:|---|---:|---:|\n")
    for i, r in enumerate(perm_all[:25], start=1):
        lines.append(
            f"| {i} | `{r.feature}` | {r.delta_coherent_kept_rate_mean:.4f} ± {r.delta_coherent_kept_rate_std:.4f} | "
            f"{r.delta_precision_lcb_mean:.4f} ± {r.delta_precision_lcb_std:.4f} |\n"
        )
    if len(perm_all) > 25:
        lines.append("\n(Top 25 shown.)\n\n")
    else:
        lines.append("\n")

    lines.append("## Steps 4–5 — Automatic Ablation / Pruning\n")
    lines.append(
        "We iteratively remove the weakest remaining feature (by mean_abs standardized weight), re-train, "
        "and evaluate. This approximates a greedy backward feature elimination.\n\n"
    )
    lines.append("| step | n_features | removed | coherent_kept_rate | precision | precision_LCB | coverage | recall | chosen_t |\n")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for row in ablation_rows:
        lines.append(
            f"| {row['step']} | {row['n_features']} | `{row['removed']}` | "
            f"{row['coherent_kept_rate']:.4f} | {row['precision']:.4f} | {row['precision_lcb']:.4f} | "
            f"{row['coverage']:.4f} | {row['recall']:.4f} | {row['chosen_t']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Step 6 — Final Selected Feature Set\n")
    lines.append(f"- Selected features: {len(final_features)}\n")
    lines.append("```text\n" + ", ".join(final_features) + "\n```\n")
    lines.append(f"- Chosen threshold (median across folds): **t = {final_eval.chosen_t_median:.4f}**\n\n")
    lines.append("Final performance at chosen thresholds (across folds):\n")
    lines.append(f"- coherent_kept_rate: **{fmt_pm(final_eval.coherent_kept_rate_mean, final_eval.coherent_kept_rate_std)}**\n")
    lines.append(f"- precision: **{fmt_pm(final_eval.precision_mean, final_eval.precision_std)}**\n")
    lines.append(f"- precision_LCB: **{fmt_pm(final_eval.precision_lcb_mean, final_eval.precision_lcb_std)}**\n")
    lines.append(f"- coverage: **{fmt_pm(final_eval.coverage_mean, final_eval.coverage_std)}**\n")
    lines.append(f"- recall: **{fmt_pm(final_eval.recall_mean, final_eval.recall_std)}**\n\n")

    lines.append("### Final weights (standardized) — top 15 by |weight|\n\n")
    lines.append("| rank | feature | mean_abs(weight) | mean(weight) ± std |\n")
    lines.append("|---:|---|---:|---:|\n")
    for i, ws in enumerate(final_weights[:15], start=1):
        lines.append(f"| {i} | `{ws.feature}` | {ws.mean_abs:.4f} | {ws.mean:.4f} ± {ws.std:.4f} |\n")
    lines.append("\n")

    lines.append("### Final permutation importance — top 15 by Δ coherent_kept_rate\n\n")
    lines.append("| rank | feature | Δ coherent_kept_rate (mean ± std) | Δ precision_LCB (mean ± std) |\n")
    lines.append("|---:|---|---:|---:|\n")
    for i, r in enumerate(perm_final[:15], start=1):
        lines.append(
            f"| {i} | `{r.feature}` | {r.delta_coherent_kept_rate_mean:.4f} ± {r.delta_coherent_kept_rate_std:.4f} | "
            f"{r.delta_precision_lcb_mean:.4f} ± {r.delta_precision_lcb_std:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Reproducibility Notes\n")
    lines.append("- Model: logistic regression trained with batch gradient descent; L2 regularization.\n")
    lines.append("- Features standardized using training-fold mean/std; applied to validation fold.\n")
    lines.append("- Threshold selected per fold to maximize coherent kept subject to precision constraint.\n")
    lines.append("- Wilson LCB constraint recommended for noisy labels.\n")
    lines.append("- Permutation importance computed on validation folds at fold-chosen threshold (kept fixed).\n")

    path.write_text("".join(lines), encoding="utf-8")


# -----------------------------
# Pruning helper
# -----------------------------

def pick_weakest_feature(weight_summaries: List[WeightSummary], protected: set[str]) -> Optional[str]:
    # weight_summaries are sorted descending by mean_abs; weakest is at the end.
    for ws in reversed(weight_summaries):
        if ws.feature not in protected:
            return ws.feature
    return None


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatic calibration + weights + permutation importance + ablation pruning.")
    p.add_argument("csv_path", help="Labeled CSV containing sentence,human_label + numeric feature columns")

    p.add_argument("--pos-threshold", type=float, default=2.0)
    p.add_argument("--neg-threshold", type=float, default=-1.0)
    p.add_argument("--min-numeric-frac", type=float, default=0.60)

    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=201)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--l2", type=float, default=1e-3)

    p.add_argument("--use-lcb", action="store_true")
    p.add_argument("--min-precision", type=float, default=0.0)
    p.add_argument("--min-precision-lcb", type=float, default=0.90)

    p.add_argument("--perm-repeats", type=int, default=5,
                   help="Number of shuffles per fold for permutation importance.")
    p.add_argument("--perm-seed", type=int, default=123,
                   help="Seed controlling permutation shuffles (independent of fold seed).")

    p.add_argument("--prune-tol", type=float, default=0.01)
    p.add_argument("--max-prune-steps", type=int, default=50)
    p.add_argument("--min-features", type=int, default=2)
    p.add_argument("--protect", default="", help="Comma-separated list of features to never remove.")

    p.add_argument("--outdir", default="calib_out")
    p.add_argument("--model-json", default="model.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_all0 = time.time()
    print("=== calibrate_cutoff.py ===")
    print(f"Input: {args.csv_path}")
    print(f"Outdir: {outdir.resolve()}")
    print(f"Settings: kfold={args.kfold} steps={args.steps} epochs={args.epochs} lr={args.lr} l2={args.l2}")
    if args.use_lcb:
        print(f"Precision constraint: Wilson LCB >= {args.min_precision_lcb}")
    else:
        print(f"Precision constraint: raw precision >= {args.min_precision}")

    print("\n[0] Reading and filtering CSV...")
    examples, feature_names, stats = read_labeled_csv(
        Path(args.csv_path),
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        min_numeric_frac=args.min_numeric_frac,
    )
    total = len(examples)
    pos = sum(e.y for e in examples)
    neg = total - pos
    prev = pos / total if total else 0.0

    print(f"[0] Raw rows: {stats['raw_rows']}")
    print(f"[0] Kept evaluable examples: {total}  coherent(+): {pos}  incoherent(-): {neg}  prevalence(+): {prev:.4f}")
    print(f"[0] Ignored non-numeric labels: {stats['ignored_non_numeric_label']}")
    print(f"[0] Ignored borderline labels: {stats['ignored_borderline']}")
    print(f"[0] Ignored rows with missing/non-numeric feature: {stats['ignored_missing_feature']}")
    print(f"[0] Features detected: {len(feature_names)}")
    print(f"[0] First 10 features: {feature_names[:10]}")

    settings: Dict[str, object] = {
        "pos_threshold": args.pos_threshold,
        "neg_threshold": args.neg_threshold,
        "kfold": args.kfold,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "epochs": args.epochs,
        "l2": args.l2,
        "use_lcb": bool(args.use_lcb),
        "min_precision": args.min_precision,
        "min_precision_lcb": args.min_precision_lcb,
        "perm_repeats": args.perm_repeats,
        "perm_seed": args.perm_seed,
        "objective": "maximize coherent_kept_rate subject to precision constraint",
        "dataset": {
            "usable_examples": total,
            "coherent": pos,
            "incoherent": neg,
            "prevalence": prev,
        },
    }

    # Step 1: all-features evaluation
    print("\n[1/6] Evaluating ALL features with k-fold CV...")
    t1 = time.time()
    all_eval = evaluate_feature_set(
        examples,
        feature_names,
        kfold=args.kfold,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        use_lcb=bool(args.use_lcb),
        min_precision=args.min_precision,
        min_precision_lcb=args.min_precision_lcb,
        progress_prefix="[all]",
    )
    print(f"[1/6] Done in {time.time()-t1:.1f}s. chosen_t_median={all_eval.chosen_t_median:.4f} "
          f"prec={all_eval.precision_mean:.4f} lcb={all_eval.precision_lcb_mean:.4f} "
          f"cov={all_eval.coverage_mean:.4f} rec={all_eval.recall_mean:.4f} "
          f"ck_rate={all_eval.coherent_kept_rate_mean:.4f}")

    print("[1/6] Writing curve_all.csv...")
    write_curve_csv(outdir / "curve_all.csv", all_eval.curve_by_t)

    # Step 2: weights summary
    print("\n[2/6] Summarizing standardized weights...")
    all_weights = summarize_weights(all_eval)
    write_weights_csv(outdir / "weights_all.csv", all_weights)
    print("[2/6] Wrote weights_all.csv")

    # Step 3: permutation importance for all-features
    print("\n[3/6] Permutation importance (ALL features)...")
    tpi = time.time()
    perm_all = permutation_importance(
        all_eval,
        perm_repeats=args.perm_repeats,
        seed=args.perm_seed,
        progress_label="perm-all",
    )
    write_perm_importance_csv(outdir / "perm_importance_all.csv", perm_all)
    print(f"[3/6] Done in {time.time()-tpi:.1f}s. Wrote perm_importance_all.csv")

    # Steps 4-5: ablation pruning
    print("\n[4-5/6] Pruning / ablation...")
    protected = set([s.strip() for s in args.protect.split(",") if s.strip()])
    if protected:
        print(f"[4-5/6] Protected features (never removed): {sorted(protected)}")

    remaining = feature_names[:]
    best_ck = all_eval.coherent_kept_rate_mean
    best_eval = all_eval
    best_features = remaining[:]

    ablation_rows: List[Dict[str, object]] = []
    ablation_rows.append({
        "step": 0,
        "n_features": len(remaining),
        "removed": "(none)",
        "coherent_kept_rate": all_eval.coherent_kept_rate_mean,
        "precision": all_eval.precision_mean,
        "precision_lcb": all_eval.precision_lcb_mean,
        "coverage": all_eval.coverage_mean,
        "recall": all_eval.recall_mean,
        "chosen_t": all_eval.chosen_t_median,
    })

    prune_t0 = time.time()
    for step in range(1, args.max_prune_steps + 1):
        if len(remaining) <= args.min_features:
            print(f"[4-5/6] Stop: reached min_features={args.min_features}")
            break

        # Evaluate current if needed
        if remaining == best_features:
            current_eval = best_eval
        else:
            keep_idx_cur = [feature_names.index(f) for f in remaining]
            current_eval = evaluate_feature_set(
                subset_examples(examples, keep_idx_cur),
                remaining,
                kfold=args.kfold,
                seed=args.seed,
                steps=args.steps,
                lr=args.lr,
                epochs=args.epochs,
                l2=args.l2,
                use_lcb=bool(args.use_lcb),
                min_precision=args.min_precision,
                min_precision_lcb=args.min_precision_lcb,
                progress_prefix=f"[prune{step:02d}]",
            )

        current_weights = summarize_weights(current_eval)
        weakest = pick_weakest_feature(current_weights, protected)
        if weakest is None:
            print("[4-5/6] Stop: no removable feature found (all protected?)")
            break

        new_remaining = [f for f in remaining if f != weakest]
        if len(new_remaining) == len(remaining):
            print("[4-5/6] Stop: removal made no change (unexpected).")
            break

        print(f"[4-5/6] Step {step}: removing '{weakest}' -> {len(new_remaining)} features remain")

        keep_idx_new = [feature_names.index(f) for f in new_remaining]
        new_eval = evaluate_feature_set(
            subset_examples(examples, keep_idx_new),
            new_remaining,
            kfold=args.kfold,
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
            epochs=args.epochs,
            l2=args.l2,
            use_lcb=bool(args.use_lcb),
            min_precision=args.min_precision,
            min_precision_lcb=args.min_precision_lcb,
            progress_prefix=f"[prune{step:02d}]",
        )

        ablation_rows.append({
            "step": step,
            "n_features": len(new_remaining),
            "removed": weakest,
            "coherent_kept_rate": new_eval.coherent_kept_rate_mean,
            "precision": new_eval.precision_mean,
            "precision_lcb": new_eval.precision_lcb_mean,
            "coverage": new_eval.coverage_mean,
            "recall": new_eval.recall_mean,
            "chosen_t": new_eval.chosen_t_median,
        })

        if new_eval.coherent_kept_rate_mean > best_ck:
            best_ck = new_eval.coherent_kept_rate_mean
            best_eval = new_eval
            best_features = new_remaining[:]
            print(f"[4-5/6] New best coherent_kept_rate_mean={best_ck:.4f}")

        remaining = new_remaining

    print(f"[4-5/6] Pruning done in {time.time()-prune_t0:.1f}s. Best coherent_kept_rate_mean={best_ck:.4f}")

    # Select minimal feature set within tolerance
    target_ck = best_ck * (1.0 - args.prune_tol)
    print(f"[5/6] Selecting minimal feature set within prune_tol={args.prune_tol} "
          f"(target coherent_kept_rate_mean >= {target_ck:.4f})")

    step_features: Dict[int, List[str]] = {0: feature_names[:]}
    cur = feature_names[:]
    for row in ablation_rows[1:]:
        removed = str(row["removed"])
        cur = [f for f in cur if f != removed]
        step_features[int(row["step"])] = cur[:]

    chosen_step = 0
    for row in reversed(ablation_rows):
        if float(row["coherent_kept_rate"]) >= target_ck:
            chosen_step = int(row["step"])
            break

    final_features = step_features[chosen_step]
    print(f"[5/6] Chosen step={chosen_step} -> final_features={len(final_features)}")

    keep_idx_final = [feature_names.index(f) for f in final_features]
    final_examples = subset_examples(examples, keep_idx_final)

    # Step 6: final evaluation + export
    print("\n[6/6] Final evaluation on selected feature set...")
    t6 = time.time()
    final_eval = evaluate_feature_set(
        final_examples,
        final_features,
        kfold=args.kfold,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        use_lcb=bool(args.use_lcb),
        min_precision=args.min_precision,
        min_precision_lcb=args.min_precision_lcb,
        progress_prefix="[final]",
    )
    write_curve_csv(outdir / "curve_final.csv", final_eval.curve_by_t)
    final_weights = summarize_weights(final_eval)
    write_weights_csv(outdir / "weights_final.csv", final_weights)
    print(f"[6/6] Done in {time.time()-t6:.1f}s. chosen_t_median={final_eval.chosen_t_median:.4f} "
          f"prec={final_eval.precision_mean:.4f} lcb={final_eval.precision_lcb_mean:.4f} "
          f"cov={final_eval.coverage_mean:.4f} rec={final_eval.recall_mean:.4f} "
          f"ck_rate={final_eval.coherent_kept_rate_mean:.4f}")

    print("\n[final] Permutation importance (FINAL feature set)...")
    tpi2 = time.time()
    perm_final = permutation_importance(
        final_eval,
        perm_repeats=args.perm_repeats,
        seed=args.perm_seed + 1,
        progress_label="perm-final",
    )
    write_perm_importance_csv(outdir / "perm_importance_final.csv", perm_final)
    print(f"[final] Done in {time.time()-tpi2:.1f}s. Wrote perm_importance_final.csv")

    # Write ablation.csv
    ablation_csv = outdir / "ablation.csv"
    with ablation_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "n_features", "removed", "coherent_kept_rate", "precision", "precision_lcb", "coverage", "recall", "chosen_t"])
        for row in ablation_rows:
            w.writerow([
                row["step"],
                row["n_features"],
                row["removed"],
                f"{float(row['coherent_kept_rate']):.6f}",
                f"{float(row['precision']):.6f}",
                f"{float(row['precision_lcb']):.6f}",
                f"{float(row['coverage']):.6f}",
                f"{float(row['recall']):.6f}",
                f"{float(row['chosen_t']):.6f}",
            ])
    print("[write] Wrote ablation.csv")

    # Fit final model on ALL data and export JSON
    print("\n[export] Training final model on ALL data with selected features...")
    X_all = [e.x for e in final_examples]
    y_all = [e.y for e in final_examples]
    Xs_all, means_all, stds_all = standardize(X_all)
    Xb_all = add_bias(Xs_all)

    w_all = train_logreg(
        Xb_all,
        y_all,
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed + 999,
    )

    t_prod = final_eval.chosen_t_median
    a_list, c_bias = linear_rule_for_threshold_multi(w_all, means_all, stds_all, prob_t=t_prod)

    model_json_path = outdir / args.model_json
    write_model_json(
        model_json_path,
        threshold=t_prod,
        feature_names=final_features,
        weights=w_all,
        means=means_all,
        stds=stds_all,
        a_list=a_list,
        c_bias=c_bias,
        settings={**settings, "selected_step": chosen_step, "prune_tol": args.prune_tol},
    )
    print(f"[export] Wrote model JSON: {model_json_path}")

    # Report
    print("[write] Writing report.md...")
    report_path = outdir / "report.md"
    write_report_md(
        report_path,
        input_csv=str(args.csv_path),
        stats=stats,
        all_eval=all_eval,
        all_weights=all_weights,
        perm_all=perm_all,
        ablation_rows=ablation_rows,
        final_eval=final_eval,
        final_features=final_features,
        final_weights=final_weights,
        perm_final=perm_final,
        settings={**settings, "selected_step": chosen_step, "prune_tol": args.prune_tol},
    )
    print(f"[write] Wrote report: {report_path}")

    # Final console summary
    dt_total = time.time() - t_all0
    print("\n=== DONE ===")
    print(f"Total time: {dt_total:.1f}s")
    print(f"Output directory: {outdir.resolve()}")
    print(f"- report:                 {report_path}")
    print(f"- model:                  {model_json_path}")
    print(f"- ablation:               {ablation_csv}")
    print(f"- weights(all):           {outdir / 'weights_all.csv'}")
    print(f"- weights(final):         {outdir / 'weights_final.csv'}")
    print(f"- perm_importance(all):   {outdir / 'perm_importance_all.csv'}")
    print(f"- perm_importance(final): {outdir / 'perm_importance_final.csv'}")
    print(f"- curve(all):             {outdir / 'curve_all.csv'}")
    print(f"- curve(final):           {outdir / 'curve_final.csv'}")

    print("\nFinal selected features:")
    print("  " + ", ".join(final_features))
    print(f"\nFinal production threshold (median across folds): t = {t_prod:.4f}")
    print("Raw-space linear rule saved in JSON for fast pipeline use.")


if __name__ == "__main__":
    main()
