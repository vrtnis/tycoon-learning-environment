from __future__ import annotations

import json
from pathlib import Path

import jax

from tycoonle_jax import TycoonLE, export_replay, rollout_first_valid

SEED = 0
NUM_STEPS = 12
REPLAY_PATH = Path("runs/quickstart/replay.json")


def main() -> None:
    env = TycoonLE(split="dev", family="chain")
    rollout = rollout_first_valid(env, jax.random.PRNGKey(SEED), num_steps=NUM_STEPS)
    replay = export_replay(rollout)
    REPLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPLAY_PATH.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    print({"replay": str(REPLAY_PATH), "summary": replay["summary"]})


if __name__ == "__main__":
    main()
