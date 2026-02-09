#!/usr/bin/env python3
"""
Create test_llama_3_2_strict.csv by copying test.csv and
adding a 'label' column from llama3_2_strict.csv.

Matching is done by exact sentence string.
"""

import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_llama_labels.py test.csv llama3_2_strict.csv")
        sys.exit(1)

    test_path = Path(sys.argv[1])
    llama_path = Path(sys.argv[2])
    out_path = Path("test_llama_3_2_strict.csv")

    if not test_path.exists() or not llama_path.exists():
        print("Error: one or more input files do not exist.")
        sys.exit(1)

    print("Reading llama3_2_strict.csv...")
    llama_labels = {}

    with llama_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "sentence" not in reader.fieldnames or "label" not in reader.fieldnames:
            print("llama3_2_strict.csv must contain 'sentence' and 'label' columns.")
            sys.exit(1)

        for row in reader:
            llama_labels[row["sentence"]] = row["label"]

    print(f"Loaded {len(llama_labels)} labeled sentences from llama3_2_strict.csv")

    print("Processing test.csv...")
    missing = 0
    total = 0

    with test_path.open("r", encoding="utf-8", newline="") as infile, \
         out_path.open("w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["label"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in reader:
            total += 1
            sentence = row.get("sentence", "")

            if sentence in llama_labels:
                row["label"] = llama_labels[sentence]
            else:
                row["label"] = ""
                missing += 1

            writer.writerow(row)

    print("\n=== Done ===")
    print(f"Total rows processed: {total}")
    print(f"Missing labels:       {missing}")
    print(f"Wrote output file:    {out_path.resolve()}")


if __name__ == "__main__":
    main()
