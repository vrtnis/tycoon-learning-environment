from __future__ import annotations

from functools import cached_property
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from tycoonle_jax.candidates import make_observation, refresh_derived
from tycoonle_jax.constants import (
    CANDIDATE_FEATURES,
    MAX_CANDIDATES,
    MAX_HEIGHT,
    MAX_NODES,
    MAX_PATH,
    MAX_ROUTES,
    MAX_WIDTH,
    METRIC_COUNT,
    NODE_FEATURES,
    ROUTE_FEATURES,
    SCORE_BREAKDOWN_COUNT,
    TERRAIN_COUNT,
    family_id,
    split_id,
)
from tycoonle_jax.dynamics import step_state
from tycoonle_jax.generator import initial_state
from tycoonle_jax.types import Observation, State


class TycoonLE(Environment[State, specs.DiscreteArray, Observation]):
    """TycoonLE environment implemented with JAX/Jumanji APIs.

    The environment exposes a scalar candidate-index action. The candidate table is
    stored in state and policy-facing candidate features plus the executable mask
    are emitted in the observation.
    """

    def __init__(self, *, split: str | int = "dev", family: str | int = "chain", max_candidates: int = MAX_CANDIDATES) -> None:
        if int(max_candidates) != MAX_CANDIDATES:
            raise ValueError(f"this JAX build uses a static max_candidates={MAX_CANDIDATES}")
        self.split_id = split_id(split)
        self.family_id = family_id(family)
        super().__init__()

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        state = refresh_derived(initial_state(key, split=self.split_id, family=self.family_id))
        observation = make_observation(state)
        timestep = restart(observation=observation, extras=_reset_extras(state), shape=(), dtype=jnp.float32)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep[Observation]]:
        next_state, extras = step_state(state, action)
        observation = make_observation(next_state)
        reward = extras["reward"]

        def done_timestep(_: Any) -> TimeStep[Observation]:
            return jax.lax.cond(
                extras["terminated"],
                lambda __: termination(reward=reward, observation=observation, extras=extras, shape=(), dtype=jnp.float32),
                lambda __: truncation(reward=reward, observation=observation, discount=jnp.array(1.0, dtype=jnp.float32), extras=extras, shape=(), dtype=jnp.float32),
                operand=None,
            )

        def mid_timestep(_: Any) -> TimeStep[Observation]:
            return transition(reward=reward, observation=observation, discount=jnp.array(1.0, dtype=jnp.float32), extras=extras, shape=(), dtype=jnp.float32)

        timestep = jax.lax.cond(next_state.done, done_timestep, mid_timestep, operand=None)
        return next_state, timestep

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        return specs.Spec(
            Observation,
            "Observation",
            company=specs.BoundedArray((4,), jnp.float32, minimum=-10.0, maximum=10.0, name="company"),
            time=specs.BoundedArray((2,), jnp.float32, minimum=0.0, maximum=2.0, name="time"),
            objective=specs.BoundedArray((4,), jnp.float32, minimum=-1.0, maximum=5.0, name="objective"),
            metrics=specs.BoundedArray((METRIC_COUNT,), jnp.float32, minimum=-1_000_000.0, maximum=1_000_000.0, name="metrics"),
            score_breakdown=specs.BoundedArray((SCORE_BREAKDOWN_COUNT,), jnp.float32, minimum=-100.0, maximum=100.0, name="score_breakdown"),
            terrain=specs.BoundedArray((MAX_HEIGHT, MAX_WIDTH), jnp.int32, minimum=0, maximum=TERRAIN_COUNT - 1, name="terrain"),
            terrain_summary=specs.BoundedArray((TERRAIN_COUNT,), jnp.float32, minimum=0.0, maximum=1.0, name="terrain_summary"),
            node_features=specs.BoundedArray((MAX_NODES, NODE_FEATURES), jnp.float32, minimum=-10.0, maximum=10.0, name="node_features"),
            route_features=specs.BoundedArray((MAX_ROUTES, ROUTE_FEATURES), jnp.float32, minimum=-10.0, maximum=10.0, name="route_features"),
            candidate_features=specs.BoundedArray((MAX_CANDIDATES, CANDIDATE_FEATURES), jnp.float32, minimum=-10.0, maximum=10.0, name="candidate_features"),
            action_mask=specs.BoundedArray((MAX_CANDIDATES,), jnp.bool_, minimum=False, maximum=True, name="action_mask"),
            node_mask=specs.BoundedArray((MAX_NODES,), jnp.bool_, minimum=False, maximum=True, name="node_mask"),
            route_mask=specs.BoundedArray((MAX_ROUTES,), jnp.bool_, minimum=False, maximum=True, name="route_mask"),
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=MAX_CANDIDATES, dtype=jnp.int32, name="action")

    def render(self, state: State) -> np.ndarray:
        terrain = np.asarray(jax.device_get(state.terrain))
        mask = np.asarray(jax.device_get(state.terrain_mask))
        colors = np.array(
            [
                [76, 139, 62],
                [119, 112, 84],
                [48, 105, 146],
                [140, 127, 99],
            ],
            dtype=np.uint8,
        )
        frame = colors[np.clip(terrain, 0, 3)]
        frame = np.where(mask[..., None], frame, np.array([28, 44, 38], dtype=np.uint8))
        for x, y in zip(np.asarray(jax.device_get(state.node_x[state.node_mask])), np.asarray(jax.device_get(state.node_y[state.node_mask])), strict=False):
            x0, x1 = max(0, x - 1), min(frame.shape[1], x + 2)
            y0, y1 = max(0, y - 1), min(frame.shape[0], y + 2)
            frame[y0:y1, x0:x1] = np.array([245, 235, 180], dtype=np.uint8)
        return np.repeat(np.repeat(frame, 8, axis=0), 8, axis=1)


def _reset_extras(state: State) -> dict[str, jnp.ndarray]:
    return {
        "action_index": jnp.array(0, dtype=jnp.int32),
        "selected_kind": jnp.array(0, dtype=jnp.int32),
        "valid_action": jnp.array(True),
        "terminated": jnp.array(False),
        "truncated": jnp.array(False),
        "reward": jnp.array(0.0, dtype=jnp.float32),
        "score_delta": jnp.array(0.0, dtype=jnp.float32),
        "cargo_delta": jnp.array(0.0, dtype=jnp.float32),
        "profit_delta": jnp.array(0.0, dtype=jnp.float32),
        "cash_delta": jnp.array(0.0, dtype=jnp.float32),
        "loan_delta": jnp.array(0.0, dtype=jnp.float32),
        "invalid_action": jnp.array(0.0, dtype=jnp.float32),
        "selected_source": jnp.array(0, dtype=jnp.int32),
        "selected_destination": jnp.array(0, dtype=jnp.int32),
        "selected_cargo": jnp.array(0, dtype=jnp.int32),
        "selected_mode": jnp.array(0, dtype=jnp.int32),
        "selected_route": jnp.array(0, dtype=jnp.int32),
        "selected_months": jnp.array(0, dtype=jnp.int32),
        "selected_amount": jnp.array(0.0, dtype=jnp.float32),
        "selected_path": jnp.zeros((MAX_PATH, 2), dtype=jnp.int32),
        "selected_path_length": jnp.array(0, dtype=jnp.int32),
        "selected_diagnostics": jnp.zeros((8,), dtype=jnp.bool_),
        "action_mask": state.action_mask,
        "metrics": state.metrics,
        "score_breakdown": state.score_breakdown,
    }
