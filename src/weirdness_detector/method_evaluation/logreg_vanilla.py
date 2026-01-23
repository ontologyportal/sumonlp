#!/usr/bin/env python3
"""
logreg_discriminator.py

Vanilla scikit-learn Logistic Regression baseline.

- Trains on dev.csv
- Applies to test.csv
- Uses default sklearn decision rule (p >= 0.5)
- No precision constraints, no threshold tuning
- Calls calc_stats_on_test_scored.py for evaluation

Assumptions:
- CSV contains numeric feature columns + 'sentence' + 'human_label'
- human_label > 0 : coherent
- human_label < 0 : incoherent
- human_label == 0 : ignored
"""

import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_xy(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Returns:
      X, y (+1/-1), feature_names, ignored_count
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        if "human_label" not in reader.fieldnames:
            raise ValueError("CSV must contain 'human_label'.")
        if "sentence" not in reader.fieldnames:
            raise ValueError("CSV must contain 'sentence'.")

        feature_cols = [c for c in reader.fieldnames if c not in ("sentence", "human_label")]
        if not feature_cols:
            raise ValueError("No feature columns found.")

        X_rows = []
        y_rows = []
        ignored = 0

        for row in reader:
            hl = parse_float(row.get("human_label", ""))
            if hl is None or hl == 0:
                ignored += 1
                continue

            y = 1 if hl > 0 else -1

            x = []
            ok = True
            for c in feature_cols:
                v = parse_float(row.get(c, ""))
                if v is None or math.isnan(v) or math.isinf(v):
                    ok = False
                    break
                x.append(v)

            if not ok:
                ignored += 1
                continue

            X_rows.append(x)
            y_rows.append(y)

    return (
        np.asarray(X_rows, dtype=np.float64),
        np.asarray(y_rows, dtype=np.int32),
        feature_cols,
        ignored,
    )


def apply_and_write(
    in_csv: Path,
    out_csv: Path,
    model: Pipeline,
    feature_cols: List[str],
) -> None:
    """
    Writes out_csv with added 'label' column.
    label = +1 if predicted coherent, else -1
    """
    with in_csv.open("r", encoding="utf-8", newline="") as fin, out_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
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
                if v is None or math.isnan(v) or math.isinf(v):
                    ok = False
                    break
                x.append(v)

            if ok:
                pred = model.predict(np.asarray([x]))[0]
                row["label"] = "1" if pred > 0 else "-1"
            else:
                row["label"] = "-1"

            writer.writerow(row)


def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_discriminator.py dev.csv test.csv")
        sys.exit(1)

    dev_csv = Path(sys.argv[1])
    test_csv = Path(sys.argv[2])
    out_csv = Path("test_scored_logreg.csv")

    print("=== Vanilla Logistic Regression Discriminator ===")
    print(f"Dev:  {dev_csv}")
    print(f"Test: {test_csv}")
    print(f"Out:  {out_csv}")

    # Load dev
    X_dev, y_dev, feature_cols, ignored = load_xy(dev_csv)
    print(f"[load] dev usable={len(y_dev)} ignored={ignored} features={len(feature_cols)}")

    # Vanilla sklearn logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )

    print("[fit] training logistic regression...")
    model.fit(X_dev, (y_dev > 0).astype(int))
    print("[fit] done")

    print("[apply] scoring test set...")
    apply_and_write(test_csv, out_csv, model, feature_cols)
    print("[apply] done")

    print("\n=== Evaluation ===")
    subprocess.run(
        ["python", "calc_stats_on_test_scored.py", str(out_csv)],
        check=False,
    )


if __name__ == "__main__":
    main()
