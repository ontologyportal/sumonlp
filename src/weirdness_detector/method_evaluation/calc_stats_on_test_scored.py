#!/usr/bin/env python3
"""
Evaluate a discriminator on test_scored.csv.

Assumptions:
- human_label > 0 : coherent (positive)
- human_label < 0 : incoherent (negative)
- human_label == 0 : ignored
- label > 0 : accepted by discriminator
- label < 0 : rejected

Outputs:
- Confusion counts
- Precision, Recall, F1
- Coverage, Coherent-kept rate, Incoherent leakage
"""

import csv
import sys
from pathlib import Path


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_test_scored.py test_scored.csv")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    tp = fp = tn = fn = 0
    total = 0
    accepted = 0
    ignored = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "human_label" not in reader.fieldnames or "label" not in reader.fieldnames:
            print("CSV must contain 'human_label' and 'label' columns.")
            sys.exit(1)

        for row in reader:
            hl = parse_float(row["human_label"])
            dl = parse_float(row["label"])

            if hl is None or hl == 0:
                ignored += 1
                continue
            if dl is None:
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    coverage = accepted / total if total else 0.0
    coherent_kept_rate = tp / total if total else 0.0
    incoherent_leakage = fp / accepted if accepted else 0.0

    print("\n=== Discriminator Evaluation ===")
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


if __name__ == "__main__":
    main()
