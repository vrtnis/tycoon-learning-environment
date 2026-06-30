from __future__ import annotations

from typing import Any

import jax
import numpy as np

from tycoonle_hud.paths import ensure_repo_python_path

ensure_repo_python_path()

from tycoonle_jax.constants import MAX_CANDIDATES
from tycoonle_jax.replay import decode_candidate, decode_observation
from tycoonle_jax.types import State


def compact_observation(state: State) -> dict[str, Any]:
    obs = decode_observation(state)
    world = obs["world"]
    return {
        "world": {
            "id": world["id"],
            "split": world["split"],
            "family": world["family"],
            "seed": world["seed"],
            "size": {"width": world["width"], "height": world["height"]},
            "terrain": terrain_counts(world["terrain"]),
            "budget": world["budget"],
            "objective": world["objective"],
        },
        "time": obs["time"],
        "company": obs["company"],
        "metrics": obs["metrics"],
        "nodes": obs["nodes"],
        "routes": obs["routes"],
        "lastEvent": obs["lastEvent"],
    }


def full_observation(state: State) -> dict[str, Any]:
    return decode_observation(state)


def visible_actions(state: State, *, limit: int, show_rank_score: bool) -> list[dict[str, Any]]:
    state = jax.device_get(state)
    mask = np.asarray(state.action_mask, dtype=bool)
    actions: list[dict[str, Any]] = []
    for idx in range(MAX_CANDIDATES):
        if not bool(mask[idx]):
            continue
        candidate = decode_candidate(state, idx)
        item = {
            "actionIndex": idx,
            "kind": candidate["kind"],
            "description": candidate["description"],
            "action": candidate["action"],
            "requiresLoan": candidate["requiresLoan"],
            "estimates": candidate["estimates"],
            "diagnostics": candidate["diagnostics"],
        }
        if show_rank_score:
            item["rankScore"] = candidate["rankScore"]
        actions.append(item)
        if len(actions) >= limit:
            break
    return actions


def metrics_from_state(state: State) -> dict[str, Any]:
    return decode_observation(state)["metrics"]


def objective_from_state(state: State) -> dict[str, Any]:
    return decode_observation(state)["world"]["objective"]


def terrain_counts(terrain: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in terrain:
        for tile in row:
            counts[tile] = counts.get(tile, 0) + 1
    return dict(sorted(counts.items()))
