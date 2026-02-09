#!/usr/bin/env python3
"""
Combine method evaluation result text files into per-model, per-method CSVs.

Outputs:
  - gpt_oss_strict.csv
  - gpt_oss_relaxed.csv
  - llama3_2_strict.csv
  - llama3_2_relaxed.csv
"""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple


MODEL_DIRS: Dict[str, str] = {
    "gpt-oss120b": "gpt_oss",
    "Llama3.2": "llama3_2",
}

METHOD_DIRS: Dict[str, str] = {
    "Strict": "strict",
    "Relaxed": "relaxed",
}


def _iter_result_files(root: str, model_dir: str, method_dir: str) -> List[str]:
    base = os.path.join(root, model_dir, method_dir)
    if not os.path.isdir(base):
        return []
    paths: List[str] = []
    for entry in os.listdir(base):
        if entry.endswith(".txt"):
            paths.append(os.path.join(base, entry))
    return sorted(paths)


def _parse_section_header(line: str) -> Optional[int]:
    upper = line.strip().upper()
    if not upper.startswith("----"):
        return None
    if "INVALID SENTENCES" in upper:
        return -1
    if "VALID SENTENCES" in upper:
        return 1
    if "UNCLASSIFIED" in upper:
        return 0
    return None


def _parse_results_file(path: str) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    current_label: Optional[int] = None

    with open(path, "r", encoding="utf-8") as infile:
        for raw in infile:
            line = raw.strip()
            if not line:
                continue
            section_label = _parse_section_header(line)
            if section_label is not None:
                current_label = section_label
                continue
            if current_label in (1, -1):
                results.append((line, current_label))

    return results


def _write_csv(rows: Iterable[Tuple[str, int]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["sentence", "label"])
        for sentence, label in rows:
            writer.writerow([sentence, label])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine strict/relaxed evaluation outputs into per-model CSVs."
    )
    p.add_argument(
        "--results-root",
        default=os.path.join(
            os.path.dirname(__file__),
            "results",
        ),
        help="Root results directory containing model/method subfolders.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV outputs (defaults to results root).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    output_dir = os.path.abspath(args.output_dir or results_root)

    for model_dir, model_slug in MODEL_DIRS.items():
        for method_dir, method_slug in METHOD_DIRS.items():
            files = _iter_result_files(results_root, model_dir, method_dir)
            combined: List[Tuple[str, int]] = []
            for path in files:
                combined.extend(_parse_results_file(path))
            out_name = f"{model_slug}_{method_slug}.csv"
            out_path = os.path.join(output_dir, out_name)
            _write_csv(combined, out_path)
            print(
                f"Wrote {out_path} from {len(files)} files with {len(combined)} sentences."
            )


if __name__ == "__main__":
    main()
