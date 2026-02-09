"""Read a weirdness detector results file and map sentences to validity labels."""

from __future__ import annotations

import argparse
from pathlib import Path


class ResultsFileReader:
    SECTION_VALUES = {
        "valid": 1,
        "invalid": -1,
        "unclassified": 0,
    }

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Parse a results file and map sentences to validity values",
        )
        parser.add_argument(
            "results_path",
            help="Path to a results file inside the results/ folder",
        )
        return parser.parse_args()

    @classmethod
    def _section_from_line(cls, line: str) -> str | None:
        upper = line.strip().upper()
        if "INVALID" in upper and "SENTENCE" in upper:
            return "invalid"
        if "UNCLASSIFIED" in upper and "SENTENCE" in upper:
            return "unclassified"
        if "VALID" in upper and "SENTENCE" in upper:
            return "valid"
        return None

    @classmethod
    def read_results(cls, results_path: Path) -> dict[str, int]:
        if not results_path.is_file():
            raise SystemExit(f"Results file not found: {results_path}")

        sentences: dict[str, int] = {}
        current_section: str | None = None

        with results_path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue

                section = cls._section_from_line(stripped)
                if section:
                    current_section = section
                    continue

                if current_section is None:
                    continue

                sentences[stripped] = cls.SECTION_VALUES[current_section]

        if not sentences:
            raise SystemExit(
                "No sentences were found. Check that the file contains VALID/INVALID/UNCLASSIFIED sections."
            )

        return sentences

    @classmethod
    def main(cls) -> None:
        args = cls.parse_args()
        sentence_map = cls.read_results(Path(args.results_path))
        for sentence, value in sentence_map.items():
            print(f"{sentence}: {value}")


if __name__ == "__main__":
    ResultsFileReader.main()
