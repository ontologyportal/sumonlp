#!/usr/bin/env python3
"""
svm_discriminator.py

Train an SVM-based coherence discriminator on dev.csv and apply to test.csv.

Assumptions:
- Input CSV has numeric feature columns + a column named 'human_label'
- human_label > 0 : coherent (positive)
- human_label < 0 : incoherent (negative)
- human_label == 0 : ignored (excluded from training/eval)

Outputs:
- Writes test_scored.csv with an added 'label' column:
    label = +1 if accepted (p >= threshold)
    label = -1 if rejected

Thresholding:
- Select threshold on a validation split of dev.csv:
    - either enforce min precision (>= --min-precision)
    - or enforce Wilson lower confidence bound on precision (>= --min-precision-lcb) with --use-lcb
- also enforce a minimum number of accepted examples on validation:
    accepted_val >= --min-accepted
- Among thresholds satisfying the constraint(s), choose the one maximizing coherent-kept rate on val:
    coherent_kept_rate = TP / N_total_val

Fallback behavior (IMPORTANT):
- If no threshold satisfies the precision constraint (with min_accepted), we DO NOT return t=1.0 blindly.
- Instead:
    - If --use-lcb: pick the threshold (with accepted>=min_accepted) that maximizes Wilson LCB.
    - Else: pick the threshold (with accepted>=min_accepted) that maximizes raw precision.
- If even accepted>=min_accepted is impossible, we fall back to accept-none (t=1.0).

Model:
- SVM with RBF kernel + probability calibration:
    Pipeline(StandardScaler -> SVC(probability=True))
"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def wilson_lcb(k: int, n: int, z: float = 1.96) -> float:
    """
    Wilson score interval lower bound for a binomial proportion.
    k successes out of n. z=1.96 ~ 95% confidence.
    """
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    radius = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - radius)


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    total: int = 0
    accepted: int = 0


def compute_counts(y_true: np.ndarray, accept: np.ndarray) -> EvalCounts:
    """
    y_true: +1 or -1
    accept: boolean True=accepted
    """
    c = EvalCounts()
    c.total = int(y_true.shape[0])
    c.accepted = int(np.sum(accept))
    c.tp = int(np.sum((accept) & (y_true > 0)))
    c.fp = int(np.sum((accept) & (y_true < 0)))
    c.tn = int(np.sum((~accept) & (y_true < 0)))
    c.fn = int(np.sum((~accept) & (y_true > 0)))
    return c


def metrics_from_counts(c: EvalCounts) -> Dict[str, float]:
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    coverage = c.accepted / c.total if c.total else 0.0
    coherent_kept_rate = c.tp / c.total if c.total else 0.0
    incoherent_leakage = c.fp / c.accepted if c.accepted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
        "coherent_kept_rate": coherent_kept_rate,
        "incoherent_leakage": incoherent_leakage,
    }


def load_xy(path: Path, label_col: str = "human_label") -> Tuple[List[str], np.ndarray, np.ndarray, List[str], int]:
    """
    Returns:
      sentences, X, y (+1/-1), feature_names, ignored_count
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        if label_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain '{label_col}' column.")
        if "sentence" not in reader.fieldnames:
            raise ValueError("CSV must contain 'sentence' column.")

        feat_cols = [c for c in reader.fieldnames if c not in ("sentence", label_col)]
        if not feat_cols:
            raise ValueError("No feature columns found (expected numeric feature columns).")

        sentences: List[str] = []
        rows_X: List[List[float]] = []
        rows_y: List[int] = []
        ignored = 0

        for row in reader:
            hl = parse_float(row.get(label_col, ""))
            if hl is None or hl == 0:
                ignored += 1
                continue

            y = 1 if hl > 0 else -1

            x: List[float] = []
            ok = True
            for c in feat_cols:
                v = parse_float(row.get(c, ""))
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    ok = False
                    break
                x.append(float(v))
            if not ok:
                ignored += 1
                continue

            sentences.append(row["sentence"])
            rows_X.append(x)
            rows_y.append(y)

    X = np.asarray(rows_X, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.int32)
    return sentences, X, y, feat_cols, ignored


def choose_threshold(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    steps: int,
    use_lcb: bool,
    min_precision: float,
    min_precision_lcb: float,
    min_accepted: int,
    z: float = 1.96,
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds in [0,1]. A threshold is eligible if:
      - accepted >= min_accepted
      - AND precision constraint holds (raw precision or Wilson LCB)

    Primary selection:
      - Among eligible thresholds, pick the one maximizing coherent_kept_rate.

    Fallback (if no eligible threshold exists):
      - If use_lcb: choose threshold (with accepted>=min_accepted) that maximizes Wilson LCB.
      - Else: choose threshold (with accepted>=min_accepted) that maximizes raw precision.
      - If even accepted>=min_accepted is impossible, fall back to accept none.
    """
    best_t = 1.0
    best_ckr = -1.0
    best_stats: Dict[str, float] = {}

    # Best among thresholds satisfying ONLY min_accepted (debug)
    best_ckr_minacc_only = -1.0
    best_ckr_minacc_only_t = 1.0
    best_ckr_minacc_only_stats: Dict[str, float] = {}

    # Fallback candidates (must satisfy min_accepted)
    best_lcb = -1.0
    best_lcb_t = 1.0
    best_lcb_stats: Dict[str, float] = {}

    best_prec = -1.0
    best_prec_t = 1.0
    best_prec_stats: Dict[str, float] = {}

    for i in range(steps):
        t = i / (steps - 1)
        accept = prob_pos >= t
        c = compute_counts(y_true, accept)
        m = metrics_from_counts(c)

        if c.accepted < min_accepted:
            continue

        # best coherent-kept-rate under only min_accepted (debugging)
        if m["coherent_kept_rate"] > best_ckr_minacc_only:
            best_ckr_minacc_only = m["coherent_kept_rate"]
            best_ckr_minacc_only_t = t
            best_ckr_minacc_only_stats = dict(m)

        # Track fallback "best achievable"
        if use_lcb:
            lcb = wilson_lcb(c.tp, c.tp + c.fp, z=z)
            m_with_lcb = dict(m)
            m_with_lcb["precision_lcb"] = lcb
            if lcb > best_lcb:
                best_lcb = lcb
                best_lcb_t = t
                best_lcb_stats = dict(m_with_lcb)
            ok = lcb >= min_precision_lcb
        else:
            if m["precision"] > best_prec:
                best_prec = m["precision"]
                best_prec_t = t
                best_prec_stats = dict(m)
            ok = m["precision"] >= min_precision

        # Primary selection among thresholds satisfying constraints
        if ok and m["coherent_kept_rate"] > best_ckr:
            best_ckr = m["coherent_kept_rate"]
            best_t = t
            best_stats = dict(m) if not use_lcb else dict(m_with_lcb)

    # If constraint cannot be satisfied, fallback
    if best_ckr < 0.0:
        print("[threshold] No threshold satisfied constraints.")
        print(f"[threshold] Constraint: accepted_val >= {min_accepted}")
        if use_lcb:
            print(f"[threshold] Constraint: Wilson LCB >= {min_precision_lcb:.4f} (z={z})")
            if best_lcb_stats:
                print(f"[threshold] Best achievable LCB (with accepted>=min_accepted): {best_lcb:.4f} at t={best_lcb_t:.4f}")
                print(f"[threshold] Stats at best achievable LCB: {json.dumps({k: round(v,4) for k,v in best_lcb_stats.items()}, indent=None)}")
                print("[threshold] WARNING: Using best-achievable LCB fallback (constraint not met).")
                return best_lcb_t, best_lcb_stats
            else:
                print("[threshold] No thresholds had accepted>=min_accepted; falling back to accept-none.")
        else:
            print(f"[threshold] Constraint: precision >= {min_precision:.4f}")
            if best_prec_stats:
                print(f"[threshold] Best achievable precision (with accepted>=min_accepted): {best_prec:.4f} at t={best_prec_t:.4f}")
                print(f"[threshold] Stats at best achievable precision: {json.dumps({k: round(v,4) for k,v in best_prec_stats.items()}, indent=None)}")
                print("[threshold] WARNING: Using best-achievable precision fallback (constraint not met).")
                return best_prec_t, best_prec_stats
            else:
                print("[threshold] No thresholds had accepted>=min_accepted; falling back to accept-none.")

        if best_ckr_minacc_only_stats:
            print(
                "[threshold] Best coherent-kept-rate with accepted>=min_accepted (ignoring precision constraint): "
                f"t={best_ckr_minacc_only_t:.4f} stats={json.dumps({k: round(v,4) for k,v in best_ckr_minacc_only_stats.items()}, indent=None)}"
            )

        return 1.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "coverage": 0.0, "coherent_kept_rate": 0.0, "incoherent_leakage": 0.0}

    return best_t, best_stats


def add_label_column_and_write(
    in_path: Path,
    out_path: Path,
    model: Pipeline,
    feature_cols: List[str],
    threshold: float,
) -> Tuple[int, int]:
    missing = 0
    written = 0

    with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        fieldnames = list(reader.fieldnames)
        if "label" not in fieldnames:
            fieldnames.append("label")

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            x = []
            ok = True
            for c in feature_cols:
                v = parse_float(row.get(c, ""))
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    ok = False
                    break
                x.append(float(v))

            if not ok:
                missing += 1
                row["label"] = "-1"
            else:
                p = float(model.predict_proba(np.asarray([x], dtype=np.float64))[0, 1])
                row["label"] = "1" if p >= threshold else "-1"

            writer.writerow(row)
            written += 1

    return written, missing


def eval_on_test(path: Path) -> None:
    tp = fp = tn = fn = 0
    total = 0
    accepted = 0
    ignored = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "human_label" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'human_label' and 'label' columns.")

        for row in reader:
            hl = parse_float(row.get("human_label", ""))
            dl = parse_float(row.get("label", ""))

            if hl is None or hl == 0 or dl is None:
                ignored += 1
                continue

            y_true = 1 if hl > 0 else -1
            y_pred = 1 if dl > 0 else -1

            total += 1
            if y_pred > 0:
                accepted += 1
                if y_true > 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if y_true < 0:
                    tn += 1
                else:
                    fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    coverage = accepted / total if total else 0.0
    coherent_kept_rate = tp / total if total else 0.0
    incoherent_leakage = fp / accepted if accepted else 0.0

    print("\n=== SVM Discriminator Evaluation ===")
    print(f"File: {path}")
    print(f"Evaluated examples: {total}")
    print(f"Ignored examples:   {ignored}\n")
    print("Confusion counts:")
    print(f"  TP (accepted coherent):   {tp}")
    print(f"  FP (accepted incoherent): {fp}")
    print(f"  TN (rejected incoherent): {tn}")
    print(f"  FN (rejected coherent):   {fn}\n")
    print("Metrics:")
    print(f"  Precision (accepted coherent): {precision:.4f}")
    print(f"  Recall (coherent accepted):    {recall:.4f}")
    print(f"  F1 score:                      {f1:.4f}")
    print(f"  Coverage (accepted total):     {coverage:.4f}")
    print(f"  Coherent-kept rate:            {coherent_kept_rate:.4f}")
    print(f"  Incoherent leakage:            {incoherent_leakage:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dev_csv", help="Development CSV with features + human_label")
    p.add_argument("test_csv", help="Test CSV with features + human_label")
    p.add_argument("--out", default="test_scored_svm.csv", help="Output scored test CSV")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of dev used for threshold selection")
    p.add_argument("--steps", type=int, default=201, help="Threshold sweep steps in [0,1]")

    p.add_argument("--use-lcb", action="store_true", help="Use Wilson lower bound constraint on precision")
    p.add_argument("--min-precision", type=float, default=0.90, help="Min precision (if not using LCB)")
    p.add_argument("--min-precision-lcb", type=float, default=0.90, help="Min precision LCB (if using LCB)")
    p.add_argument("--z", type=float, default=1.96, help="z value for Wilson LCB (1.96 ~ 95%)")

    p.add_argument(
        "--min-accepted",
        type=int,
        default=1,
        help="Minimum number of accepted examples required on validation when selecting a threshold.",
    )

    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--gamma", default="scale", help="SVC gamma: 'scale', 'auto', or a float")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev_path = Path(args.dev_csv)
    test_path = Path(args.test_csv)
    out_path = Path(args.out)

    print("=== svm_discriminator.py ===")
    print(f"Dev:  {dev_path}")
    print(f"Test: {test_path}")
    print(f"Out:  {out_path}")
    print(f"Settings: val_frac={args.val_frac} steps={args.steps} use_lcb={args.use_lcb}")
    print(f"Min accepted on val: {args.min_accepted}")
    if args.use_lcb:
        print(f"Precision constraint: Wilson LCB >= {args.min_precision_lcb} (z={args.z})")
    else:
        print(f"Precision constraint: precision >= {args.min_precision}")

    t0 = time.time()
    _, X, y, feat_cols, ignored_dev = load_xy(dev_path, label_col="human_label")
    print(f"[load] dev usable={len(y)} ignored={ignored_dev} features={len(feat_cols)}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.val_frac, random_state=args.seed, stratify=y
    )
    print(f"[split] train={len(y_tr)} val={len(y_va)}")

    gamma = args.gamma
    if isinstance(gamma, str):
        try:
            gamma = float(gamma)
        except Exception:
            pass

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", SVC(C=args.C, kernel="rbf", gamma=gamma, probability=True, random_state=args.seed)),
        ]
    )

    print("[fit] training SVM...")
    model.fit(X_tr, (y_tr > 0).astype(int))
    print(f"[fit] done in {time.time()-t0:.1f}s")

    print("[threshold] scoring validation...")
    prob_va = model.predict_proba(X_va)[:, 1]
    t_star, stats = choose_threshold(
        y_true=y_va,
        prob_pos=prob_va,
        steps=args.steps,
        use_lcb=args.use_lcb,
        min_precision=args.min_precision,
        min_precision_lcb=args.min_precision_lcb,
        min_accepted=args.min_accepted,
        z=args.z,
    )
    print(f"[threshold] chosen t={t_star:.4f}  stats={json.dumps({k: round(v,4) for k,v in stats.items()}, indent=None)}")

    print("[refit] training on full dev...")
    model.fit(X, (y > 0).astype(int))
    print("[refit] done")

    print("[apply] writing labeled test CSV...")
    written, missing = add_label_column_and_write(
        in_path=test_path,
        out_path=out_path,
        model=model,
        feature_cols=feat_cols,
        threshold=t_star,
    )
    print(f"[apply] wrote rows={written}  rows_missing_features={missing}")

    eval_on_test(out_path)


if __name__ == "__main__":
    main()
