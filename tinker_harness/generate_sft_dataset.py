from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
for path in (ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax

from tycoonle_jax import TycoonLE
from tycoonle_jax.constants import FAMILY_NAMES

from tinker_harness.heuristic_labels import build_assistant_content
from tinker_harness.render_world import RenderOptions, render_user_prompt, world_from_state
from tinker_harness.schemas import SYSTEM_PROMPT, validate_example


def main() -> None:
    args = parse_args()
    families = parse_families(args.families)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for idx in range(args.examples):
        family = families[idx % len(families)]
        seed = args.seed_start + idx
        env = TycoonLE(split=args.split, family=family)
        state, _ = env.reset(jax.random.PRNGKey(seed))
        world = world_from_state(
            state,
            seed=seed,
            split=args.split,
            family=family,
            options=RenderOptions(candidate_limit=args.candidate_limit),
        )
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_prompt(world)},
                {"role": "assistant", "content": build_assistant_content(world)},
            ],
            "metadata": {
                "family": family,
                "seed": seed,
                "split": args.split,
                "labeler": "candidate_heuristic_v1",
            },
        }
        errors = validate_example(example)
        if errors:
            joined = "; ".join(errors)
            raise RuntimeError(f"generated invalid example for {family}/{seed}: {joined}")
        examples.append(example)

    with out.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "event": "dataset_written",
                "out": str(out),
                "examples": len(examples),
                "families": families,
                "split": args.split,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TycoonLE chat SFT data for Tinker.")
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--families", default=",".join(FAMILY_NAMES))
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--out", default="results/tinker/datasets/sft-smoke.jsonl")
    return parser.parse_args()


def parse_families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(families) - set(FAMILY_NAMES))
    if unknown:
        raise ValueError(f"unknown families: {', '.join(unknown)}")
    if not families:
        raise ValueError("at least one family is required")
    return families


if __name__ == "__main__":
    main()
