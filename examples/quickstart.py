from __future__ import annotations

import json
from pathlib import Path

import jax

from tycoonle_jax import TycoonLE, export_replay, rollout_first_valid


def main() -> None:
    env = TycoonLE(split="dev", family="chain")
    rollout = rollout_first_valid(env, jax.random.PRNGKey(0), num_steps=12)
    replay = export_replay(rollout)
    out = Path("runs/quickstart/replay.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    print({"replay": str(out), "summary": replay["summary"]})


if __name__ == "__main__":
    main()
