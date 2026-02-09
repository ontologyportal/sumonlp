"""Extract Sentence/Valid columns from a weirdness detector CSV.

The CSV files have several preamble rows before the first header row that
contains "Sentence" and "Valid/Invalid" columns. This script locates that
header and stores the rows in a dictionary keyed by sentence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


class SentenceValidityExtractor:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Extract the Sentence and Valid/Invalid columns from a CSV file",
        )
        parser.add_argument(
            "csv_path",
            help=(
                "Path to a CSV file (e.g., test.csv) or 'all' to process every CSV in csv_data/"
            ),
        )
        return parser.parse_args()

    @staticmethod
    def find_header_row(reader: csv.reader) -> list[str]:
        for row in reader:
            cleaned = [col.strip() for col in row]
            if len(cleaned) >= 2 and cleaned[0] == "Sentence" and cleaned[1].startswith(
                "Valid/Invalid"
            ):
                return cleaned
        raise SystemExit(
            "Could not locate the header row with Sentence and Valid/Invalid columns."
        )

    @classmethod
    def extract_columns(cls, csv_path: Path) -> dict[str, str]:
        if not csv_path.is_file():
            raise SystemExit(f"Input file not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = cls.find_header_row(reader)

            try:
                sentence_idx = header.index("Sentence")
                validity_idx = header.index("Valid/Invalid")
            except ValueError as exc:
                raise SystemExit("Required columns are missing in the header row.") from exc

            extracted: dict[str, str] = {}

            for row in reader:
                if not row:
                    continue
                sentence = row[sentence_idx] if sentence_idx < len(row) else ""
                validity = row[validity_idx] if validity_idx < len(row) else ""
                extracted[sentence] = validity

        return extracted

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        csv_arg = args.csv_path.strip()
        if csv_arg.lower() == "all":
            csv_dir = Path(__file__).resolve().parent / "csv_data"
            if not csv_dir.is_dir():
                raise SystemExit(f"CSV directory not found: {csv_dir}")
            csv_files = sorted(csv_dir.glob("*.csv"))
            if not csv_files:
                raise SystemExit(f"No CSV files found in {csv_dir}")

            for idx, csv_file in enumerate(csv_files):
                print(f"--- {csv_file.name} ---")
                sentence_status = cls.extract_columns(csv_file)
                for sentence, validity in sentence_status.items():
                    print(f"{sentence}: {validity}")
                if idx < len(csv_files) - 1:
                    print()
        else:
            sentence_status = cls.extract_columns(Path(csv_arg))
            for sentence, validity in sentence_status.items():
                print(f"{sentence}: {validity}")


if __name__ == "__main__":
    SentenceValidityExtractor.main()
