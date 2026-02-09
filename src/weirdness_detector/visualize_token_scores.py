#!/usr/bin/env python3
"""
Create a 3D scatter plot from the ollama-model-token CSV output.

Axes:
  - x: avg_nll
  - y: p95
  - z: run

Each point corresponds to one sentence. Points are colored by label.

Also writes 2D scatter plots for each pairwise combination of axes.
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def _load_label_map(path: str, source_file: str = "") -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required = {"sentence", "label"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns: {sorted(required)}")
        if source_file and "source_file" not in set(reader.fieldnames):
            raise ValueError("CSV missing required column: source_file")

        for row in reader:
            if source_file and row.get("source_file", "").strip() != source_file:
                continue
            sentence = row["sentence"].strip()
            raw_label = row["label"].strip()
            if not raw_label:
                continue
            try:
                labels[sentence] = int(raw_label)
            except ValueError:
                continue

    return labels


def _resolve_label_path(path: str) -> str:
    if not os.path.exists(path):
        raise SystemExit(f"Missing label file: {path}")
    return path


def _load_points(
    path: str, label_path: str, source_file: str = ""
) -> Tuple[List[float], List[float], List[float], List[int]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    labels: List[int] = []

    label_map = _load_label_map(_resolve_label_path(label_path), source_file=source_file)

    with open(path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required = {"sentence", "avg_nll", "p95", "run"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing required columns: {sorted(required)}")

        for row in reader:
            sentence = row["sentence"].strip()
            if sentence not in label_map:
                continue
            xs.append(float(row["avg_nll"]))
            ys.append(float(row["p95"]))
            zs.append(float(row["run"]))
            labels.append(label_map[sentence])

    return xs, ys, zs, labels


def _label_to_color(label: int) -> str:
    if label == 2 or label == 3:
        return "#2ca02c"  # green
    if label == 0:
        return "#ff7f0e"  # orange
    return "#d62728"  # red


def plot_points(
    xs: List[float], ys: List[float], zs: List[float], labels: List[int], out_path: str
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = [_label_to_color(l) for l in labels]
    ax.scatter(xs, ys, zs, c=colors, s=12, alpha=0.75)

    ax.set_xlabel("avg_nll")
    ax.set_ylabel("p95")
    ax.set_zlabel("run")
    ax.set_title("Sentence Coherence Metrics")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _plot_points_2d(
    xs: List[float],
    ys: List[float],
    labels: List[int],
    x_label: str,
    y_label: str,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [_label_to_color(l) for l in labels]
    ax.scatter(xs, ys, c=colors, s=12, alpha=0.75)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _ensure_extension(path: str, default_ext: str) -> str:
    root, ext = os.path.splitext(path)
    if ext:
        return path
    return root + default_ext


def write_pairwise_plots(
    xs: List[float],
    ys: List[float],
    zs: List[float],
    labels: List[int],
    out_path: str,
) -> List[str]:
    base_path = _ensure_extension(out_path, ".png")
    root, ext = os.path.splitext(base_path)
    outputs: List[str] = []

    pairs = [
        ("avg_nll", xs, "p95", ys),
        ("avg_nll", xs, "run", zs),
        ("p95", ys, "run", zs),
    ]
    for x_label, x_vals, y_label, y_vals in pairs:
        pair_out = f"{root}_{x_label}_{y_label}{ext}"
        _plot_points_2d(
            x_vals,
            y_vals,
            labels,
            x_label=x_label,
            y_label=y_label,
            title=f"Sentence Coherence Metrics ({x_label} vs {y_label})",
            out_path=pair_out,
        )
        outputs.append(pair_out)

    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ollama-model-token CSV output.")
    p.add_argument("--input", required=True, help="Input CSV from ollama-model-token.py")
    p.add_argument(
        "--output",
        required=True,
        help="Output image file (png recommended); pairwise plots are written with suffixes.",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="CSV file containing sentence/label columns for coloring.",
    )
    p.add_argument(
        "--source-file",
        default="",
        help="Filter labels to a specific source_file value.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xs, ys, zs, labels = _load_points(
        args.input, args.labels, source_file=args.source_file
    )
    plot_points(xs, ys, zs, labels, args.output)
    pair_outputs = write_pairwise_plots(xs, ys, zs, labels, args.output)
    print(f"Wrote plot to {args.output}")
    for path in pair_outputs:
        print(f"Wrote plot to {path}")


if __name__ == "__main__":
    main()
