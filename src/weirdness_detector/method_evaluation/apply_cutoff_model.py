#!/usr/bin/env python3
"""
Apply a calibrated cutoff model (from calibrate_cutoff.py schema_version=4) to a CSV.

Adds:
  label = +1 (keep / coherent)
        = -1 (reject / incoherent)

Uses the exported RAW linear rule:
  keep if sum(a[f] * x[f]) + c_bias >= 0

Usage:
  python apply_cutoff_model.py test.csv model.json test_scored.csv
"""

import argparse
import csv
import json
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply cutoff model to CSV (schema v4)")
    p.add_argument("input_csv", help="Input CSV (e.g., test.csv)")
    p.add_argument("model_json", help="Model JSON produced by calibrate_cutoff.py")
    p.add_argument("output_csv", help="Output CSV with predicted label column")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=== apply_cutoff_model.py ===")
    print(f"Input CSV:  {args.input_csv}")
    print(f"Model JSON: {args.model_json}")
    print(f"Output CSV: {args.output_csv}")

    # Load model.json
    with open(args.model_json, "r", encoding="utf-8") as f:
        model: Dict[str, Any] = json.load(f)

    # Validate expected schema (your file has schema_version=4)
    schema_version = model.get("schema_version")
    if schema_version != 4:
        print(f"Warning: expected schema_version=4, got {schema_version!r}. Trying anyway.")

    features: List[str] = model["features"]
    raw = model["linear_rule_raw"]
    a: Dict[str, float] = raw["a"]
    c_bias: float = float(raw["c_bias"])

    # Read input CSV
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Ensure required feature columns exist
    missing = [feat for feat in features if feat not in fieldnames]
    if missing:
        raise SystemExit(f"Missing required feature columns in input CSV: {missing}")

    # Ensure model has coefficients for all features
    missing_coef = [feat for feat in features if feat not in a]
    if missing_coef:
        raise SystemExit(f"Model missing raw coefficients for features: {missing_coef}")

    out_fields = fieldnames + (["label"] if "label" not in fieldnames else [])

    kept = rejected = 0

    # Write output
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for r in rows:
            score = c_bias
            for feat in features:
                score += float(a[feat]) * float(r[feat])

            label = 1 if score >= 0.0 else -1
            if label > 0:
                kept += 1
            else:
                rejected += 1

            r_out = dict(r)
            r_out["label"] = label
            writer.writerow(r_out)

    total = kept + rejected
    print("\nModel rule applied:")
    print(raw.get("keep_if", "sum(a[f]*x[f]) + c_bias >= 0"))
    print("\nSummary:")
    print(f"  Total rows:  {total}")
    print(f"  Kept (+1):   {kept} ({kept/total:.3%})")
    print(f"  Reject (-1): {rejected} ({rejected/total:.3%})")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
