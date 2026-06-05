from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from tycoonle_jax import TycoonLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TycoonLE throughput.")
    parser.add_argument("--family", default="chain")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--pmap", action="store_true", help="split the batch across all local JAX devices")
    args = parser.parse_args()

    env = TycoonLE(split="dev", family=args.family)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def single_rollout(key):
        state, _ = env.reset(key)

        def body(carry, _):
            action = jnp.argmax(carry.action_mask.astype(jnp.int32))
            next_state, timestep = env.step(carry, action)
            return next_state, timestep.reward

        return jax.lax.scan(body, state, None, length=args.steps)

    if args.pmap:
        device_count = jax.local_device_count()
        if args.batch % device_count != 0:
            raise ValueError(f"--batch must be divisible by local_device_count={device_count} when using --pmap")
        keys = jax.random.split(key, args.batch).reshape(device_count, args.batch // device_count, 2)
        batched = jax.pmap(jax.vmap(single_rollout))
    else:
        device_count = 1
        keys = jax.random.split(key, args.batch)
        batched = jax.jit(jax.vmap(single_rollout))

    jax.block_until_ready(batched(keys))
    start = time.perf_counter()
    _, rewards = batched(keys)
    jax.block_until_ready(rewards)
    elapsed = time.perf_counter() - start
    transitions = args.batch * args.steps
    print(
        {
            "batch": args.batch,
            "steps": args.steps,
            "devices": device_count,
            "seconds": round(elapsed, 4),
            "transitions_per_second": round(transitions / elapsed, 1),
        }
    )


if __name__ == "__main__":
    main()
