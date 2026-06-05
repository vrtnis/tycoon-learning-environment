from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from tycoonle_jax import TycoonLE
from tycoonle_jax.constants import MAX_CANDIDATES
from tycoonle_jax.training import (
    OBSERVATION_SIZE,
    PPOConfig,
    create_train_state,
    deterministic_actions,
    encode_observation,
    make_eval_fn,
    make_pmap_train_step,
    make_train_step,
    sample_actions,
)


def _small_config() -> PPOConfig:
    return replace(PPOConfig(), num_envs=4, rollout_length=4, update_epochs=1, hidden_sizes=(32,))


def test_observation_encoder_size() -> None:
    env = TycoonLE(split="dev", family="chain")
    _, timestep = env.reset(jax.random.PRNGKey(0))
    features = encode_observation(timestep.observation)
    assert features.shape == (OBSERVATION_SIZE,)
    assert bool(jnp.all(jnp.isfinite(features)))


def test_masked_actions_respect_action_mask() -> None:
    env = TycoonLE(split="dev", family="chain")
    _, timestep = env.reset(jax.random.PRNGKey(1))
    logits = jnp.linspace(-1.0, 1.0, MAX_CANDIDATES)
    action, log_prob, entropy = sample_actions(jax.random.PRNGKey(2), logits, timestep.observation.action_mask)
    greedy = deterministic_actions(logits, timestep.observation.action_mask)
    assert bool(timestep.observation.action_mask[action])
    assert bool(timestep.observation.action_mask[greedy])
    assert bool(jnp.isfinite(log_prob))
    assert bool(jnp.isfinite(entropy))


def test_compiled_ppo_train_step_updates_params() -> None:
    env = TycoonLE(split="dev", family="chain")
    config = _small_config()
    state = create_train_state(jax.random.PRNGKey(3), config)
    train_step = make_train_step(env, config)
    next_state, metrics = train_step(state, jax.random.PRNGKey(4))
    assert int(next_state.step) == 1
    assert bool(jnp.isfinite(metrics.loss))
    assert bool(jnp.isfinite(metrics.mean_reward))


def test_compiled_policy_eval_shapes() -> None:
    env = TycoonLE(split="dev", family="single_route")
    config = _small_config()
    state = create_train_state(jax.random.PRNGKey(5), config)
    evaluate = make_eval_fn(env, num_envs=3, max_steps=4)
    metrics = evaluate(state.params, jax.random.PRNGKey(6))
    assert metrics.mean_return.shape == ()
    assert metrics.mean_score.shape == ()
    assert bool(jnp.isfinite(metrics.mean_return))
    assert bool(jnp.isfinite(metrics.mean_score))


def test_pmap_train_step_smoke() -> None:
    env = TycoonLE(split="dev", family="single_route")
    config = replace(_small_config(), num_envs=1, rollout_length=2, hidden_sizes=(16,))
    state = create_train_state(jax.random.PRNGKey(7), config)
    replicated_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (jax.local_device_count(),) + x.shape), state)
    keys = jax.random.split(jax.random.PRNGKey(8), jax.local_device_count())
    train_step = make_pmap_train_step(env, config)
    next_state, metrics = train_step(replicated_state, keys)
    assert next_state.step.shape == (jax.local_device_count(),)
    assert bool(jnp.all(jnp.isfinite(metrics.loss)))
