from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax

from tycoonle_jax.constants import (
    CANDIDATE_FEATURES,
    MAX_CANDIDATES,
    MAX_HEIGHT,
    MAX_NODES,
    MAX_ROUTES,
    MAX_WIDTH,
    METRIC_COUNT,
    NODE_FEATURES,
    ROUTE_FEATURES,
    SCORE_BREAKDOWN_COUNT,
    TERRAIN_COUNT,
)
from tycoonle_jax.env import TycoonLE
from tycoonle_jax.types import Observation

OBSERVATION_SIZE = (
    4
    + 2
    + 4
    + METRIC_COUNT
    + SCORE_BREAKDOWN_COUNT
    + TERRAIN_COUNT
    + MAX_NODES * NODE_FEATURES
    + MAX_ROUTES * ROUTE_FEATURES
    + MAX_CANDIDATES * CANDIDATE_FEATURES
    + MAX_CANDIDATES
    + MAX_NODES
    + MAX_ROUTES
    + MAX_HEIGHT * MAX_WIDTH
)

_METRIC_SCALE = jnp.array(
    [
        100.0,
        2000.0,
        150_000.0,
        100.0,
        MAX_ROUTES,
        24.0,
        10.0,
        96.0,
        1.0,
        1.0,
        1000.0,
        1.0,
        8.0,
        3000.0,
        2.0,
        24.0,
    ],
    dtype=jnp.float32,
)


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 128
    rollout_length: int = 32
    update_epochs: int = 4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)


class LayerParams(NamedTuple):
    weight: chex.Array
    bias: chex.Array


class ActorCriticParams(NamedTuple):
    policy: tuple[LayerParams, ...]
    value: tuple[LayerParams, ...]


class TrainState(NamedTuple):
    params: ActorCriticParams
    opt_state: Any
    step: chex.Array


class Transition(NamedTuple):
    features: chex.Array
    action_mask: chex.Array
    actions: chex.Array
    old_log_probs: chex.Array
    values: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    done: chex.Array


class RolloutBatch(NamedTuple):
    features: chex.Array
    action_mask: chex.Array
    actions: chex.Array
    old_log_probs: chex.Array
    values: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    advantages: chex.Array
    returns: chex.Array


class TrainMetrics(NamedTuple):
    loss: chex.Array
    policy_loss: chex.Array
    value_loss: chex.Array
    entropy: chex.Array
    approx_kl: chex.Array
    clip_fraction: chex.Array
    mean_reward: chex.Array
    done_rate: chex.Array
    mean_score: chex.Array


class EvalMetrics(NamedTuple):
    mean_return: chex.Array
    mean_score: chex.Array
    success_rate: chex.Array
    done_rate: chex.Array


class RolloutMetrics(NamedTuple):
    mean_reward: chex.Array
    done_rate: chex.Array
    mean_score: chex.Array


def encode_observation(observation: Observation) -> chex.Array:
    """Flatten the fixed TycoonLE observation into a policy input vector."""

    terrain = observation.terrain.astype(jnp.float32) / jnp.maximum(1.0, TERRAIN_COUNT - 1)
    metrics = jnp.clip(observation.metrics / _METRIC_SCALE, -10.0, 10.0)
    return jnp.concatenate(
        [
            observation.company.astype(jnp.float32).reshape(-1),
            observation.time.astype(jnp.float32).reshape(-1),
            observation.objective.astype(jnp.float32).reshape(-1),
            metrics.reshape(-1),
            (observation.score_breakdown.astype(jnp.float32) / 100.0).reshape(-1),
            observation.terrain_summary.astype(jnp.float32).reshape(-1),
            observation.node_features.astype(jnp.float32).reshape(-1),
            observation.route_features.astype(jnp.float32).reshape(-1),
            observation.candidate_features.astype(jnp.float32).reshape(-1),
            observation.action_mask.astype(jnp.float32).reshape(-1),
            observation.node_mask.astype(jnp.float32).reshape(-1),
            observation.route_mask.astype(jnp.float32).reshape(-1),
            terrain.reshape(-1),
        ],
        axis=0,
    )


def encode_observation_batch(observation: Observation) -> chex.Array:
    return jax.vmap(encode_observation)(observation)


def create_train_state(key: chex.PRNGKey, config: PPOConfig = PPOConfig()) -> TrainState:
    params = _init_actor_critic(key, OBSERVATION_SIZE, config.hidden_sizes, MAX_CANDIDATES)
    optimizer = _optimizer(config)
    return TrainState(params=params, opt_state=optimizer.init(params), step=jnp.array(0, dtype=jnp.int32))


def apply_actor_critic(params: ActorCriticParams, features: chex.Array) -> tuple[chex.Array, chex.Array]:
    logits = _mlp_apply(params.policy, features)
    value = jnp.squeeze(_mlp_apply(params.value, features), axis=-1)
    return logits, value


def sample_actions(key: chex.PRNGKey, logits: chex.Array, action_mask: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    masked_logits = _mask_logits(logits, action_mask)
    actions = jax.random.categorical(key, masked_logits).astype(jnp.int32)
    log_probs, entropy = masked_log_prob_entropy(logits, action_mask, actions)
    return actions, log_probs, entropy


def deterministic_actions(logits: chex.Array, action_mask: chex.Array) -> chex.Array:
    return jnp.argmax(_mask_logits(logits, action_mask), axis=-1).astype(jnp.int32)


def masked_log_prob_entropy(logits: chex.Array, action_mask: chex.Array, actions: chex.Array) -> tuple[chex.Array, chex.Array]:
    masked_logits = _mask_logits(logits, action_mask)
    log_probs_all = jax.nn.log_softmax(masked_logits, axis=-1)
    probs = jnp.exp(log_probs_all)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)
    entropy = -jnp.sum(jnp.where(action_mask, probs * log_probs_all, 0.0), axis=-1)
    return log_probs, entropy


def make_train_step(env: TycoonLE, config: PPOConfig = PPOConfig()):
    optimizer = _optimizer(config)

    @jax.jit
    def train_step(train_state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, TrainMetrics]:
        batch, rollout_metrics = _collect_rollout(env, train_state.params, key, config)
        next_state, update_metrics = _ppo_update(train_state, batch, optimizer, config)
        metrics = TrainMetrics(
            loss=update_metrics.loss,
            policy_loss=update_metrics.policy_loss,
            value_loss=update_metrics.value_loss,
            entropy=update_metrics.entropy,
            approx_kl=update_metrics.approx_kl,
            clip_fraction=update_metrics.clip_fraction,
            mean_reward=rollout_metrics.mean_reward,
            done_rate=rollout_metrics.done_rate,
            mean_score=rollout_metrics.mean_score,
        )
        return next_state, metrics

    return train_step


def make_pmap_train_step(env: TycoonLE, config: PPOConfig = PPOConfig(), *, axis_name: str = "devices"):
    optimizer = _optimizer(config)

    def train_step(train_state: TrainState, key: chex.PRNGKey) -> tuple[TrainState, TrainMetrics]:
        batch, rollout_metrics = _collect_rollout(env, train_state.params, key, config)
        next_state, update_metrics = _ppo_update(train_state, batch, optimizer, config, axis_name=axis_name)
        metrics = TrainMetrics(
            loss=update_metrics.loss,
            policy_loss=update_metrics.policy_loss,
            value_loss=update_metrics.value_loss,
            entropy=update_metrics.entropy,
            approx_kl=update_metrics.approx_kl,
            clip_fraction=update_metrics.clip_fraction,
            mean_reward=rollout_metrics.mean_reward,
            done_rate=rollout_metrics.done_rate,
            mean_score=rollout_metrics.mean_score,
        )
        metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name), metrics)
        return next_state, metrics

    return jax.pmap(train_step, axis_name=axis_name)


def make_eval_fn(env: TycoonLE, *, num_envs: int = 64, max_steps: int = 128):
    @jax.jit
    def evaluate(params: ActorCriticParams, key: chex.PRNGKey) -> EvalMetrics:
        keys = jax.random.split(key, num_envs)
        states, timesteps = jax.vmap(env.reset)(keys)
        done = jnp.zeros((num_envs,), dtype=jnp.bool_)
        total_return = jnp.zeros((num_envs,), dtype=jnp.float32)

        def body(carry, _):
            states, observations, done, total_return = carry
            features = encode_observation_batch(observations)
            logits, _ = apply_actor_critic(params, features)
            actions = deterministic_actions(logits, observations.action_mask)

            def step_one(state, observation, action, already_done):
                def active(_):
                    next_state, timestep = env.step(state, action)
                    return next_state, timestep.observation, timestep.reward, next_state.done

                def inactive(_):
                    return state, observation, jnp.array(0.0, dtype=jnp.float32), jnp.array(True)

                return jax.lax.cond(already_done, inactive, active, operand=None)

            next_states, next_observations, rewards, step_done = jax.vmap(step_one)(states, observations, actions, done)
            next_done = done | step_done
            total_return = total_return + jnp.where(done, 0.0, rewards)
            return (next_states, next_observations, next_done, total_return), None

        (final_states, _, done, total_return), _ = jax.lax.scan(
            body,
            (states, timesteps.observation, done, total_return),
            None,
            length=max_steps,
        )
        scores = final_states.metrics[:, 0]
        return EvalMetrics(
            mean_return=jnp.mean(total_return),
            mean_score=jnp.mean(scores),
            success_rate=jnp.mean((scores >= 92.0).astype(jnp.float32)),
            done_rate=jnp.mean(done.astype(jnp.float32)),
        )

    return evaluate


def _collect_rollout(env: TycoonLE, params: ActorCriticParams, key: chex.PRNGKey, config: PPOConfig) -> tuple[RolloutBatch, RolloutMetrics]:
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, config.num_envs)
    states, timesteps = jax.vmap(env.reset)(reset_keys)

    def body(carry, _):
        states, observations, key = carry
        key, action_key, reset_key = jax.random.split(key, 3)
        action_keys = jax.random.split(action_key, config.num_envs)
        reset_keys = jax.random.split(reset_key, config.num_envs)

        features = encode_observation_batch(observations)
        logits, values = apply_actor_critic(params, features)
        actions, log_probs, _ = jax.vmap(sample_actions)(action_keys, logits, observations.action_mask)
        next_states, next_timesteps = jax.vmap(env.step)(states, actions)
        done = next_states.done
        discounts = config.discount * (~done).astype(jnp.float32)

        reset_states, reset_timesteps = jax.vmap(env.reset)(reset_keys)
        carry_states = _where_batch(done, reset_states, next_states)
        carry_observations = _where_batch(done, reset_timesteps.observation, next_timesteps.observation)

        transition = Transition(
            features=features,
            action_mask=observations.action_mask,
            actions=actions,
            old_log_probs=log_probs,
            values=values,
            rewards=next_timesteps.reward,
            discounts=discounts,
            done=done,
        )
        return (carry_states, carry_observations, key), transition

    (final_states, final_observations, _), transitions = jax.lax.scan(
        body,
        (states, timesteps.observation, key),
        None,
        length=config.rollout_length,
    )
    last_features = encode_observation_batch(final_observations)
    _, last_values = apply_actor_critic(params, last_features)
    batch = _with_advantages(transitions, last_values, config)
    metrics = RolloutMetrics(
        mean_reward=jnp.mean(transitions.rewards),
        done_rate=jnp.mean(transitions.done.astype(jnp.float32)),
        mean_score=jnp.mean(final_states.metrics[:, 0]),
    )
    return batch, metrics


def _with_advantages(transitions: Transition, last_values: chex.Array, config: PPOConfig) -> RolloutBatch:
    def body(carry, data):
        next_advantage, next_value = carry
        reward, discount, value = data
        delta = reward + discount * next_value - value
        advantage = delta + discount * config.gae_lambda * next_advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        body,
        (jnp.zeros_like(last_values), last_values),
        (transitions.rewards, transitions.discounts, transitions.values),
        reverse=True,
    )
    returns = advantages + transitions.values
    return RolloutBatch(
        features=transitions.features,
        action_mask=transitions.action_mask,
        actions=transitions.actions,
        old_log_probs=transitions.old_log_probs,
        values=transitions.values,
        rewards=transitions.rewards,
        discounts=transitions.discounts,
        advantages=advantages,
        returns=returns,
    )


def _ppo_update(
    train_state: TrainState,
    batch: RolloutBatch,
    optimizer: optax.GradientTransformation,
    config: PPOConfig,
    *,
    axis_name: str | None = None,
) -> tuple[TrainState, TrainMetrics]:
    flat_batch = _flatten_batch(batch)
    advantages = (flat_batch.advantages - jnp.mean(flat_batch.advantages)) / (jnp.std(flat_batch.advantages) + 1e-8)
    flat_batch = flat_batch._replace(advantages=advantages)

    def update_epoch(state: TrainState, _):
        (loss, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(state.params, flat_batch, config)
        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name)
            loss = jax.lax.pmean(loss, axis_name)
            metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name), metrics)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        next_state = TrainState(params=params, opt_state=opt_state, step=state.step + 1)
        metrics = metrics._replace(loss=loss)
        return next_state, metrics

    next_state, metrics = jax.lax.scan(update_epoch, train_state, None, length=config.update_epochs)
    return next_state, jax.tree.map(lambda x: x[-1], metrics)


def _ppo_loss(params: ActorCriticParams, batch: RolloutBatch, config: PPOConfig) -> tuple[chex.Array, TrainMetrics]:
    logits, values = apply_actor_critic(params, batch.features)
    log_probs, entropy = masked_log_prob_entropy(logits, batch.action_mask, batch.actions)
    ratio = jnp.exp(log_probs - batch.old_log_probs)
    unclipped = ratio * batch.advantages
    clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean(jnp.square(batch.returns - values))
    entropy_mean = jnp.mean(entropy)
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
    approx_kl = jnp.mean(batch.old_log_probs - log_probs)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32))
    metrics = TrainMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        mean_reward=jnp.array(0.0, dtype=jnp.float32),
        done_rate=jnp.array(0.0, dtype=jnp.float32),
        mean_score=jnp.array(0.0, dtype=jnp.float32),
    )
    return loss, metrics


def _flatten_batch(batch: RolloutBatch) -> RolloutBatch:
    return RolloutBatch(
        features=batch.features.reshape((-1, batch.features.shape[-1])),
        action_mask=batch.action_mask.reshape((-1, batch.action_mask.shape[-1])),
        actions=batch.actions.reshape(-1),
        old_log_probs=batch.old_log_probs.reshape(-1),
        values=batch.values.reshape(-1),
        rewards=batch.rewards.reshape(-1),
        discounts=batch.discounts.reshape(-1),
        advantages=batch.advantages.reshape(-1),
        returns=batch.returns.reshape(-1),
    )


def _optimizer(config: PPOConfig) -> optax.GradientTransformation:
    return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))


def _init_actor_critic(key: chex.PRNGKey, input_size: int, hidden_sizes: tuple[int, ...], action_size: int) -> ActorCriticParams:
    policy_key, value_key = jax.random.split(key)
    policy_sizes = (input_size, *hidden_sizes, action_size)
    value_sizes = (input_size, *hidden_sizes, 1)
    return ActorCriticParams(policy=_init_mlp(policy_key, policy_sizes), value=_init_mlp(value_key, value_sizes))


def _init_mlp(key: chex.PRNGKey, layer_sizes: tuple[int, ...]) -> tuple[LayerParams, ...]:
    keys = jax.random.split(key, len(layer_sizes) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:], strict=False):
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        weight = jax.random.uniform(layer_key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=jnp.float32)
        bias = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append(LayerParams(weight=weight, bias=bias))
    return tuple(layers)


def _mlp_apply(layers: tuple[LayerParams, ...], x: chex.Array) -> chex.Array:
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer.weight + layer.bias)
    final = layers[-1]
    return x @ final.weight + final.bias


def _mask_logits(logits: chex.Array, action_mask: chex.Array) -> chex.Array:
    return jnp.where(action_mask, logits, jnp.array(-1.0e9, dtype=logits.dtype))


def _where_batch(mask: chex.Array, on_true: Any, on_false: Any) -> Any:
    def select(true_leaf, false_leaf):
        leaf_mask = mask.reshape(mask.shape + (1,) * (true_leaf.ndim - mask.ndim))
        return jnp.where(leaf_mask, true_leaf, false_leaf)

    return jax.tree.map(select, on_true, on_false)
