from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_harness.schemas import validate_example


def main() -> None:
    args = parse_args()
    path = Path(args.dataset)
    counts: Counter[str] = Counter()
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSONL row: {exc}")
                continue
            row_errors = validate_example(example)
            if row_errors:
                errors.extend(f"line {line_number}: {error}" for error in row_errors)
            metadata = example.get("metadata", {})
            counts[str(metadata.get("family", "unknown"))] += 1

    if errors:
        for error in errors[:25]:
            print(error)
        if len(errors) > 25:
            print(f"... {len(errors) - 25} more errors")
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "event": "dataset_valid",
                "dataset": str(path),
                "examples": sum(counts.values()),
                "families": dict(sorted(counts.items())),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TycoonLE Tinker SFT JSONL data.")
    parser.add_argument("--dataset", default="results/tinker/datasets/sft-smoke.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
