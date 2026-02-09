#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_xy(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        feat_cols = [c for c in reader.fieldnames if c not in ("sentence", "human_label")]

        X, y = [], []
        rows = []

        for row in reader:
            hl = float(row["human_label"])
            if hl == 0:
                continue

            x = [float(row[c]) for c in feat_cols]
            X.append(x)
            y.append(1 if hl > 0 else 0)
            rows.append(row)

    return np.array(X), np.array(y), feat_cols, rows


def main():
    if len(sys.argv) != 4:
        print("Usage: python svm_vanilla.py train.csv test.csv out.csv")
        sys.exit(1)

    train_csv = Path(sys.argv[1])
    test_csv = Path(sys.argv[2])
    out_csv = Path(sys.argv[3])

    X_train, y_train, feat_cols, _ = load_xy(train_csv)
    X_test, _, _, test_rows = load_xy(test_csv)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC())   # <-- defaults
    ])

    print("[fit] Training vanilla SVM...")
    model.fit(X_train, y_train)
    print("[fit] Done")

    print("[predict] Predicting test set...")
    preds = model.predict(X_test)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()) + ["label"])
        writer.writeheader()

        for row, p in zip(test_rows, preds):
            row["label"] = 1 if p == 1 else -1
            writer.writerow(row)

    print(f"[done] Wrote {out_csv}")


if __name__ == "__main__":
    main()
