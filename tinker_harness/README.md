# Tinker Harness

This folder is an isolated proof-of-concept harness for fine-tuning a Tinker LoRA model on TycoonLE planning examples.

The harness has two phases:

1. Generate local supervised fine-tuning data from the TycoonLE environment.
2. Launch a Tinker SFT run against that JSONL dataset once `TINKER_API_KEY` is set.

The generated dataset uses chat records with:

- a system prompt that requires JSON-only logistics plans
- a user prompt rendered from a procedural TycoonLE world and candidate action table
- an assistant message produced by a deterministic heuristic labeler

## Setup

Install the repo and Tinker dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install -r tinker_harness\requirements.txt
```

Set the API key only in your local shell:

```powershell
$env:TINKER_API_KEY="..."
```

Do not commit the key.

## Generate Data

Create a small smoke dataset without calling Tinker:

```powershell
npm run tinker:data -- --examples 50 --out results\tinker\datasets\sft-smoke.jsonl
npm run tinker:validate -- --dataset results\tinker\datasets\sft-smoke.jsonl
```

## Train

Dry-run the training config:

```powershell
npm run tinker:train -- --dataset results\tinker\datasets\sft-smoke.jsonl --dry-run
```

Launch a real Qwen3-8B LoRA SFT run:

```powershell
npm run tinker:train -- --dataset results\tinker\datasets\sft-smoke.jsonl --max-steps 10
```

The default model is `Qwen/Qwen3-8B` with the `qwen3_disable_thinking` renderer, so training targets JSON answers directly instead of reasoning traces.
