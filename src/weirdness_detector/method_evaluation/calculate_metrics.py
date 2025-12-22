"""Calculate precision, recall, prevalence, and accuracy for a classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_sentence_valid import SentenceValidityExtractor
from read_results_file import ResultsFileReader

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for a binary classifier using sentence-level data.",
    )
    parser.add_argument(
        "results_file",
        help="Path to the classifier results file to read",
    )
    parser.add_argument(
        "actual_csv",
        help="Path to the CSV file with actual sentence validity labels",
    )
    return parser.parse_args()


def normalize_actual(value: str, sentence: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        print(f"Actual validity missing for sentence: {sentence}")
        return None
    try:
        numeric = float(stripped)
    except ValueError:
        print(f"Actual validity not numeric for sentence: {sentence}")
        return None
    if numeric == 0:
        print(f"Actual validity is 0; ignoring sentence: {sentence}")
        return None
    return 1 if numeric > 0 else -1


def main() -> None:
    args = parse_args()

    classifier_results = ResultsFileReader.read_results(Path(args.results_file))
    actual_results = SentenceValidityExtractor.extract_columns(Path(args.actual_csv))

    tp = fp = tn = fn = 0

    for sentence, actual_value in actual_results.items():
        if not sentence:
            continue
        actual_label = normalize_actual(actual_value, sentence)
        if actual_label is None:
            continue

        predicted = classifier_results.get(sentence)
        if predicted is None:
            print(f"Sentence not classified by classifier: {sentence}")
            continue
        if predicted == 0:
            print(f"Sentence unclassified by classifier: {sentence}")
            continue

        if predicted > 0 and actual_label > 0:
            tp += 1
        elif predicted > 0 and actual_label < 0:
            fp += 1
        elif predicted < 0 and actual_label < 0:
            tn += 1
        elif predicted < 0 and actual_label > 0:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    prevalence = (tp + fn) / total if total else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Prevalence: {prevalence:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Counts -> TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, Total: {total}")


if __name__ == "__main__":
    main()
