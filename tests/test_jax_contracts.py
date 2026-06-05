from __future__ import annotations

import jax
import jax.numpy as jnp
import jumanji

from tycoonle_jax import FAMILY_NAMES, TycoonLE
from tycoonle_jax.constants import MAX_CANDIDATES, MAX_HEIGHT, MAX_NODES, MAX_ROUTES, MAX_WIDTH


def test_reset_and_step_shapes() -> None:
    env = TycoonLE(split="dev", family="chain")
    state, timestep = env.reset(jax.random.PRNGKey(0))
    assert timestep.observation.terrain.shape == (MAX_HEIGHT, MAX_WIDTH)
    assert timestep.observation.node_features.shape[0] == MAX_NODES
    assert timestep.observation.route_features.shape[0] == MAX_ROUTES
    assert timestep.observation.candidate_features.shape[0] == MAX_CANDIDATES
    assert timestep.observation.action_mask.shape == (MAX_CANDIDATES,)
    assert bool(jnp.any(timestep.observation.action_mask))

    action = jnp.argmax(timestep.observation.action_mask.astype(jnp.int32))
    next_state, next_timestep = env.step(state, action)
    assert int(next_state.step) == 1
    assert next_timestep.reward.shape == ()
    assert "selected_kind" in next_timestep.extras


def test_jumanji_registration() -> None:
    import tycoonle_jax  # noqa: F401

    env = jumanji.make("TycoonLE-v0", family="chain")
    assert isinstance(env, TycoonLE)


def test_jax_transforms_reset_step_vmap_and_scan() -> None:
    env = TycoonLE(split="dev", family="chain")
    state, timestep = jax.jit(env.reset)(jax.random.PRNGKey(1))
    action = jnp.argmax(timestep.observation.action_mask.astype(jnp.int32))
    state, timestep = jax.jit(env.step)(state, action)
    assert int(state.step) == 1

    keys = jax.random.split(jax.random.PRNGKey(2), 3)
    states, timesteps = jax.vmap(env.reset)(keys)
    actions = jnp.argmax(timesteps.observation.action_mask.astype(jnp.int32), axis=1)
    states, timesteps = jax.vmap(env.step)(states, actions)
    assert states.step.shape == (3,)

    def body(carry, _):
        action = jnp.argmax(carry.action_mask.astype(jnp.int32))
        next_state, ts = env.step(carry, action)
        return next_state, ts.reward

    final_state, rewards = jax.lax.scan(body, state, None, length=4)
    assert rewards.shape == (4,)
    assert int(final_state.step) == 5


def test_all_families_have_executable_reset_action() -> None:
    for family in FAMILY_NAMES:
        env = TycoonLE(split="dev", family=family)
        state, timestep = env.reset(jax.random.PRNGKey(7))
        assert bool(jnp.any(timestep.observation.action_mask)), family
        action = jnp.argmax(timestep.observation.action_mask.astype(jnp.int32))
        next_state, next_timestep = env.step(state, action)
        assert int(next_state.metrics[6]) == 0
        assert next_timestep.extras["valid_action"].shape == ()


def test_invalid_action_is_logged_without_crashing() -> None:
    env = TycoonLE(split="dev", family="chain")
    state, _ = env.reset(jax.random.PRNGKey(4))
    state, timestep = env.step(state, jnp.array(999))
    assert int(state.invalid_actions) == 1
    assert float(timestep.extras["invalid_action"]) == 1.0
