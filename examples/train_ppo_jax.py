from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from tycoonle_jax import TycoonLE
from tycoonle_jax.training import PPOConfig, create_train_state, make_eval_fn, make_pmap_train_step, make_train_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent on TycoonLE.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--family", default="chain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--pmap", action="store_true", help="split PPO rollouts and gradient updates across all local JAX devices")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after this many wall-clock seconds, checked between PPO updates")
    parser.add_argument("--logdir", default=None, help="optional TensorBoard root directory")
    parser.add_argument("--run-name", default=None, help="optional run directory name under --logdir")
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
    if args.pmap:
        train_state = _replicate(train_state, device_count)
        train_step = make_pmap_train_step(env, config)
    else:
        train_step = make_train_step(env, config)
    evaluate = make_eval_fn(env, num_envs=args.eval_envs, max_steps=args.eval_steps)
    logger = _make_tensorboard_logger(args.logdir, args.run_name, args.family, args.seed)
    start_time = time.perf_counter()
    run_path = str(logger.path) if logger is not None else None

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
                "tensorboard": run_path,
            }
        ),
        flush=True,
    )

    try:
        for update in range(1, args.updates + 1):
            key, step_key = jax.random.split(key)
            if args.pmap:
                step_key = jax.random.split(step_key, device_count)
            train_state, metrics = train_step(train_state, step_key)
            elapsed = time.perf_counter() - start_time
            transitions = update * args.num_envs * config.rollout_length
            payload: dict[str, Any] = {
                "event": "train",
                "update": update,
                "elapsed_seconds": round(elapsed, 3),
                "transitions": transitions,
                **_metrics_to_dict(metrics),
            }
            if args.eval_every > 0 and (update == 1 or update % args.eval_every == 0 or update == args.updates):
                key, eval_key = jax.random.split(key)
                params = _unreplicate(train_state.params) if args.pmap else train_state.params
                payload["eval"] = _metrics_to_dict(evaluate(params, eval_key))
            if logger is not None:
                _write_tensorboard(logger, payload)
            print(json.dumps(payload), flush=True)
            if args.max_seconds is not None and elapsed >= args.max_seconds:
                print(json.dumps({"event": "stopped", "reason": "max_seconds", "elapsed_seconds": round(elapsed, 3), "update": update, "tensorboard": run_path}), flush=True)
                break
    finally:
        if logger is not None:
            logger.close()


def _metrics_to_dict(metrics: Any) -> dict[str, float]:
    return {field: round(float(jnp.mean(jax.device_get(getattr(metrics, field)))), 6) for field in metrics._fields}


def _unreplicate(tree: Any) -> Any:
    return jax.tree.map(lambda x: x[0], tree)


def _replicate(tree: Any, count: int) -> Any:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (count,) + x.shape), tree)


class TensorBoardLogger:
    def __init__(self, path: Path) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(str(path))

    def scalar(self, tag: str, step: int, value: float) -> None:
        summary = self._summary_cls(value=[self._summary_cls.Value(tag=tag, simple_value=float(value))])
        self._writer.add_event(self._event_cls(wall_time=time.time(), step=int(step), summary=summary))

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _make_tensorboard_logger(logdir: str | None, run_name: str | None, family: str, seed: int) -> TensorBoardLogger | None:
    if not logdir:
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    name = run_name or f"{family}_seed{seed}_{timestamp}"
    try:
        return TensorBoardLogger(Path(logdir) / name)
    except ImportError as exc:
        raise RuntimeError("TensorBoard logging requires: python -m pip install -e '.[logging]'") from exc


def _write_tensorboard(logger: TensorBoardLogger, payload: dict[str, Any]) -> None:
    step = int(payload["update"])
    train_keys = ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
    rollout_keys = ("mean_reward", "done_rate", "mean_score")
    for key in train_keys:
        logger.scalar(f"train/{key}", step, float(payload[key]))
    for key in rollout_keys:
        logger.scalar(f"rollout/{key}", step, float(payload[key]))
    logger.scalar("charts/elapsed_seconds", step, float(payload["elapsed_seconds"]))
    logger.scalar("charts/transitions", step, float(payload["transitions"]))
    if "eval" in payload:
        for key, value in payload["eval"].items():
            logger.scalar(f"eval/{key}", step, float(value))
    logger.flush()


if __name__ == "__main__":
    main()
