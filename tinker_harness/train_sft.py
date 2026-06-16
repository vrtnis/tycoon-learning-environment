from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_harness.schemas import validate_example


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    summary = validate_dataset(dataset)
    log_path = Path(args.log_path)

    config_preview = {
        "dataset": str(dataset),
        "datasetSummary": summary,
        "model": args.model,
        "renderer": args.renderer,
        "logPath": str(log_path),
        "batchSize": args.batch_size,
        "maxLength": args.max_length,
        "learningRate": args.learning_rate,
        "numEpochs": args.num_epochs,
        "maxSteps": args.max_steps,
        "loraRank": args.lora_rank,
        "saveEvery": args.save_every,
        "evalEvery": args.eval_every,
    }

    if args.dry_run:
        print(json.dumps({"event": "tinker_train_dry_run", **config_preview}, indent=2))
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Set it in your shell before launching real training.")

    try:
        from tinker_cookbook.supervised import train
        from tinker_cookbook.renderers import TrainOnWhat
        try:
            from tinker_cookbook.supervised import ChatDatasetBuilderCommonConfig, FromConversationFileBuilder
        except ImportError:
            from tinker_cookbook.supervised.data import FromConversationFileBuilder
            from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as exc:
        raise SystemExit(
            "Tinker packages are not installed. Run: .\\.venv\\Scripts\\python.exe -m pip install -r tinker_harness\\requirements.txt"
        ) from exc

    dataset_builder = FromConversationFileBuilder(
        file_path=str(dataset),
        test_size=args.test_size,
        shuffle_seed=args.shuffle_seed,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model,
            renderer_name=args.renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ),
    )
    config = train.Config(
        log_path=str(log_path),
        model_name=args.model,
        recipe_name=args.recipe_name,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_steps=args.max_steps,
        ttl_seconds=args.ttl_seconds,
    )
    print(json.dumps({"event": "tinker_train_start", **config_preview}, indent=2))
    asyncio.run(train.main(config))


def validate_dataset(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"dataset does not exist: {path}")
    examples = 0
    families: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"line {line_number}: invalid JSONL row: {exc}") from exc
            errors = validate_example(example)
            if errors:
                raise SystemExit(f"line {line_number}: {'; '.join(errors)}")
            family = str(example.get("metadata", {}).get("family", "unknown"))
            families[family] = families.get(family, 0) + 1
            examples += 1
    if examples == 0:
        raise SystemExit(f"dataset has no examples: {path}")
    return {"examples": examples, "families": families}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Tinker SFT run for TycoonLE planning JSON.")
    parser.add_argument("--dataset", default="results/tinker/datasets/sft-smoke.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--renderer", default="qwen3_disable_thinking")
    parser.add_argument("--log-path", default="results/tinker/runs/qwen3-8b-sft")
    parser.add_argument("--recipe-name", default="tycoonle_json_planner_sft")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", default="linear", choices=("linear", "cosine", "constant"))
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--ttl-seconds", type=int, default=604800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
