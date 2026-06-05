from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from tycoonle_jax import TycoonLE, export_replay
from tycoonle_jax.training import PPOConfig, create_train_state, deterministic_actions, encode_observation, make_eval_fn, make_pmap_train_step, make_train_step
from tycoonle_jax.training.ppo import apply_actor_critic


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO, keep the best eval checkpoint, and export a browser replay.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--family", default="chain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=720)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--pmap", action="store_true", help="split PPO rollouts and gradient updates across all local JAX devices")
    parser.add_argument("--replay-candidates", type=int, default=64, help="number of deterministic eval seeds to search for the replay")
    parser.add_argument("--replay-seed", type=int, default=10_000)
    parser.add_argument("--out", default="public/replays/best-eval-replay.json")
    parser.add_argument("--checkpoint-out", default=None, help="optional pickle path for the best params and metadata")
    args = parser.parse_args()

    hidden_sizes = tuple(int(item.strip()) for item in args.hidden_sizes.split(",") if item.strip())
    device_count = jax.local_device_count() if args.pmap else 1
    if args.pmap and args.num_envs % device_count != 0:
        raise ValueError(f"--num-envs must be divisible by local_device_count={device_count} when using --pmap")

    config = replace(
        PPOConfig(),
        num_envs=args.num_envs // device_count,
        rollout_length=args.rollout_length,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        hidden_sizes=hidden_sizes,
    )
    env = TycoonLE(split=args.split, family=args.family)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    train_state = create_train_state(init_key, config)
    train_step = make_pmap_train_step(env, config) if args.pmap else make_train_step(env, config)
    if args.pmap:
        train_state = _replicate(train_state, device_count)
    evaluate = make_eval_fn(env, num_envs=args.eval_envs, max_steps=args.eval_steps)

    start_time = time.perf_counter()
    best_score = -float("inf")
    best_update = 0
    best_eval: dict[str, float] | None = None
    best_params: Any | None = None

    print(
        json.dumps(
            {
                "event": "start",
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "config": {
                    "split": args.split,
                    "family": args.family,
                    "updates": args.updates,
                    "num_envs": args.num_envs,
                    "envs_per_device": config.num_envs,
                    "pmap": args.pmap,
                    "rollout_length": config.rollout_length,
                    "update_epochs": config.update_epochs,
                    "hidden_sizes": config.hidden_sizes,
                },
            }
        ),
        flush=True,
    )

    for update in range(1, args.updates + 1):
        key, step_key = jax.random.split(key)
        if args.pmap:
            step_key = jax.random.split(step_key, device_count)
        train_state, metrics = train_step(train_state, step_key)

        payload: dict[str, Any] = {
            "event": "train",
            "update": update,
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "transitions": update * args.num_envs * config.rollout_length,
            **_metrics_to_dict(metrics),
        }
        should_eval = args.eval_every > 0 and (update == 1 or update % args.eval_every == 0 or update == args.updates)
        if should_eval:
            key, eval_key = jax.random.split(key)
            params = _unreplicate(train_state.params) if args.pmap else train_state.params
            eval_metrics = _metrics_to_dict(evaluate(params, eval_key))
            payload["eval"] = eval_metrics
            if eval_metrics["mean_score"] > best_score:
                best_score = eval_metrics["mean_score"]
                best_update = update
                best_eval = eval_metrics
                best_params = jax.device_get(params)
                payload["best"] = True
        print(json.dumps(payload), flush=True)

    if best_params is None:
        best_params = jax.device_get(_unreplicate(train_state.params) if args.pmap else train_state.params)
        best_update = int(args.updates)
        best_eval = {}

    rollout_key, rollout_score, rollout_return, rollout_done = _select_replay_key(
        env,
        best_params,
        seed=args.replay_seed,
        candidates=args.replay_candidates,
        max_steps=args.eval_steps,
    )
    rollout = _rollout_policy(env, best_params, rollout_key, max_steps=args.eval_steps)
    replay = export_replay(rollout)
    replay["training"] = {
        "algorithm": "ppo",
        "checkpoint": "best_eval",
        "bestUpdate": best_update,
        "bestEval": best_eval,
        "selectedReplay": {
            "score": round(float(rollout_score), 6),
            "return": round(float(rollout_return), 6),
            "done": bool(rollout_done),
        },
        "config": {
            "split": args.split,
            "family": args.family,
            "seed": args.seed,
            "updates": args.updates,
            "numEnvTransitions": args.updates * args.num_envs * config.rollout_length,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, indent=2), encoding="utf-8")

    if args.checkpoint_out:
        checkpoint = {"params": best_params, "best_update": best_update, "best_eval": best_eval, "config": config, "args": vars(args)}
        checkpoint_out = Path(args.checkpoint_out)
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_out.open("wb") as file:
            pickle.dump(checkpoint, file)

    print(
        json.dumps(
            {
                "event": "replay_exported",
                "out": str(out),
                "checkpoint": args.checkpoint_out,
                "best_update": best_update,
                "best_eval": best_eval,
                "selected_replay_score": round(float(rollout_score), 6),
                "selected_replay_return": round(float(rollout_return), 6),
                "selected_replay_done": bool(rollout_done),
                "summary": replay["summary"],
            }
        ),
        flush=True,
    )


def _rollout_policy(env: TycoonLE, params: Any, key: jax.Array, *, max_steps: int) -> dict[str, Any]:
    state, first_timestep = env.reset(key)

    def body(carry: tuple[Any, Any], _: jnp.ndarray):
        state, observation = carry
        features = encode_observation(observation)
        logits, _ = apply_actor_critic(params, features)
        action = deterministic_actions(logits[jnp.newaxis, :], observation.action_mask[jnp.newaxis, :])[0]
        next_state, timestep = env.step(state, action)
        return (next_state, timestep.observation), (state, next_state, action, timestep)

    (final_state, _), scanned = jax.lax.scan(body, (state, first_timestep.observation), jnp.arange(max_steps))
    before_states, after_states, actions, timesteps = scanned
    return {
        "initial_state": state,
        "initial_timestep": first_timestep,
        "before_states": before_states,
        "after_states": after_states,
        "actions": actions,
        "timesteps": timesteps,
        "final_state": final_state,
    }


def _select_replay_key(
    env: TycoonLE,
    params: Any,
    *,
    seed: int,
    candidates: int,
    max_steps: int,
) -> tuple[jax.Array, float, float, bool]:
    score_one = _make_score_rollout(env, max_steps=max_steps)
    keys = jax.random.split(jax.random.PRNGKey(seed), candidates)
    scores, returns, done = jax.vmap(lambda rollout_key: score_one(params, rollout_key))(keys)
    best_idx = int(jnp.argmax(scores))
    return keys[best_idx], float(scores[best_idx]), float(returns[best_idx]), bool(done[best_idx])


def _make_score_rollout(env: TycoonLE, *, max_steps: int):
    @jax.jit
    def score_rollout(params: Any, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state, timestep = env.reset(key)
        done = jnp.array(False)
        total_return = jnp.array(0.0, dtype=jnp.float32)

        def body(carry: tuple[Any, Any, jax.Array, jax.Array], _: jnp.ndarray):
            state, observation, done, total_return = carry
            features = encode_observation(observation)
            logits, _ = apply_actor_critic(params, features)
            action = deterministic_actions(logits[jnp.newaxis, :], observation.action_mask[jnp.newaxis, :])[0]

            def active(_: None):
                next_state, next_timestep = env.step(state, action)
                return next_state, next_timestep.observation, next_state.done, total_return + next_timestep.reward

            def inactive(_: None):
                return state, observation, done, total_return

            return jax.lax.cond(done, inactive, active, operand=None), None

        (final_state, _, done, total_return), _ = jax.lax.scan(
            body,
            (state, timestep.observation, done, total_return),
            jnp.arange(max_steps),
        )
        return final_state.metrics[0], total_return, done

    return score_rollout


def _metrics_to_dict(metrics: Any) -> dict[str, float]:
    return {field: round(float(jnp.mean(jax.device_get(getattr(metrics, field)))), 6) for field in metrics._fields}


def _unreplicate(tree: Any) -> Any:
    return jax.tree.map(lambda x: x[0], tree)


def _replicate(tree: Any, count: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (count,) + x.shape), tree)


if __name__ == "__main__":
    main()
