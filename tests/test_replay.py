from __future__ import annotations

import json

import jax

from tycoonle_jax import TycoonLE, export_replay, rollout_first_valid


def test_scanned_rollout_exports_browser_replay_schema() -> None:
    env = TycoonLE(split="dev", family="chain")
    rollout = rollout_first_valid(env, jax.random.PRNGKey(10), num_steps=4)
    replay = export_replay(rollout)
    assert replay["schema"] == "tycoonle-replay-v1"
    assert replay["events"]
    event = replay["events"][0]
    assert event["before"]["schema"] == "tycoonle-observation-v1"
    assert event["action"]["type"] in {"build_route", "add_vehicle", "wait", "take_loan", "repay_loan", "invalid"}
    assert "focus" in event
    assert "rewardDetails" in event["info"]
    assert event["info"]["candidateActions"]
    assert json.dumps(replay)
