# Tycoon Learning Environment

![TycoonLE replay interface](https://raw.githubusercontent.com/vrtnis/tycoon-learning-environment/main/assets/tycoonLE.png)

Prime/Verifiers environment for evaluating LLM agents on TycoonLE logistics planning.

The model receives a TycoonLE world state and a visible table of executable candidate actions. On each turn it must return JSON:

```json
{"action_index": 3, "reason": "short reason"}
```

The environment parses the action, executes it in TycoonLE, returns the updated state, and scores the rollout using final TycoonLE score, JSON validity, and valid-action rate.

## Local Smoke Eval

Install Prime CLI and authenticate separately:

```powershell
pip install prime
prime config set-api-key
```

Run a small eval after the environment is installed:

```powershell
prime eval run tycoon-learning-environment -p .\environments -n 3 -r 1 -m openai/gpt-4.1-mini --max-tokens 128
```

For local development without spending credits, import `load_environment()` and call the environment parser/scorer directly.

## Environment Args

- `split`: TycoonLE split, default `dev`
- `families`: comma-separated family names, default all families
- `seed_start`: first procedural seed, default `20000`
- `num_examples`: dataset size, default `5`
- `candidate_limit`: maximum visible candidates per turn, default `12`
- `action_budget`: maximum executed TycoonLE actions per rollout, default `6`
- `max_turns`: Prime/Verifiers turn limit; when provided, sets `action_budget` to `max_turns - 1`
