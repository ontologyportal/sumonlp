#!/usr/bin/env python3
"""
Split a labeled CSV into dev/test sets with stratification.

- Randomly shuffles rows
- Stratifies by human_label
- Default 80/20 split
- Writes dev.csv and test.csv
- Prints summary statistics suitable for reporting

Usage:
  python split_dev_test.py scored_features.csv --seed 0
"""

import argparse
import csv
import random
from collections import defaultdict
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified 80/20 dev-test split")
    p.add_argument("input_csv", help="Input CSV with human_label as last column")
    p.add_argument("--dev-frac", type=float, default=0.8, help="Fraction for dev set")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--dev-out", default="dev.csv", help="Dev output CSV")
    p.add_argument("--test-out", default="test.csv", help="Test output CSV")
    return p.parse_args()


def is_number(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=== split_dev_test.py ===")
    print(f"Input file: {args.input_csv}")
    print(f"Dev fraction: {args.dev_frac}")
    print(f"Seed: {args.seed}")

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    print(f"Total rows read: {len(rows)}")

    # Group rows by label (last column)
    by_label: Dict[str, List[List[str]]] = defaultdict(list)
    dropped = 0

    for r in rows:
        label = r[-1].strip()
        if not is_number(label):
            dropped += 1
            continue
        by_label[label].append(r)

    if dropped > 0:
        print(f"Dropped rows with non-numeric labels: {dropped}")

    dev_rows = []
    test_rows = []

    print("\nLabel-wise split summary:")
    for label, group in sorted(by_label.items(), key=lambda x: float(x[0])):
        random.shuffle(group)
        n = len(group)
        n_dev = int(round(args.dev_frac * n))
        dev_rows.extend(group[:n_dev])
        test_rows.extend(group[n_dev:])
        print(f"  label={label:>6}  total={n:>4}  dev={n_dev:>4}  test={n-n_dev:>4}")

    # Final shuffle so dev/test are not grouped by label
    random.shuffle(dev_rows)
    random.shuffle(test_rows)

    with open(args.dev_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dev_rows)

    with open(args.test_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    print("\nFinal counts:")
    print(f"  Dev set:  {len(dev_rows)} rows  -> {args.dev_out}")
    print(f"  Test set: {len(test_rows)} rows -> {args.test_out}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
