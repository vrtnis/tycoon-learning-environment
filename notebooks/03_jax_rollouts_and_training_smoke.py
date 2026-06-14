import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import time
    from dataclasses import replace
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import pandas as pd
    import marimo as mo

    REPO_ROOT = Path(__file__).resolve().parents[1]
    PYTHON_SRC = REPO_ROOT / "python"
    if PYTHON_SRC.exists() and str(PYTHON_SRC) not in sys.path:
        sys.path.insert(0, str(PYTHON_SRC))

    from tycoonle_jax import FAMILY_NAMES, TycoonLE
    from tycoonle_jax.training import PPOConfig, create_train_state, make_train_step

    def run_batched_first_valid(env, key, batch_size, rollout_steps):
        keys = jax.random.split(key, batch_size)

        def single_rollout(reset_key):
            state, _ = env.reset(reset_key)

            def body(carry, _):
                action = jnp.argmax(carry.action_mask.astype(jnp.int32))
                next_state, timestep = env.step(carry, action)
                return next_state, timestep.reward

            final_state, rewards = jax.lax.scan(body, state, None, length=rollout_steps)
            return jnp.sum(rewards), final_state.metrics[0], final_state.metrics[1], final_state.metrics[2]

        batched = jax.jit(jax.vmap(single_rollout))
        jax.block_until_ready(batched(keys))
        start = time.perf_counter()
        returns, scores, cargo, profit = batched(keys)
        jax.block_until_ready(scores)
        elapsed = time.perf_counter() - start
        return {
            "elapsed_seconds": elapsed,
            "transitions_per_second": batch_size * rollout_steps / max(elapsed, 1e-9),
            "returns": returns,
            "scores": scores,
            "cargo": cargo,
            "profit": profit,
        }

    def scalar_metrics(metrics):
        metrics = jax.device_get(metrics)
        return {field: round(float(getattr(metrics, field)), 6) for field in metrics._fields}

    def plot_columns(frame, columns, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        for column in columns:
            ax.plot(frame["update"], frame[column], marker="o", label=column)
        ax.set_title(title)
        ax.set_xlabel("update")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return fig

    return (
        FAMILY_NAMES,
        PPOConfig,
        TycoonLE,
        create_train_state,
        jax,
        make_train_step,
        mo,
        pd,
        plot_columns,
        replace,
        run_batched_first_valid,
        scalar_metrics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JAX rollouts and PPO smoke training

    TycoonLE is built around a fixed-shape Jumanji-style API: observations, candidate features, and action masks keep the same shapes across generated worlds. That makes the environment compatible with JAX transformations such as `jit`, `vmap`, and `scan`.

    This notebook demonstrates those contracts with batched rollouts, then runs a very small PPO loop as an end-to-end training smoke test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Configure a small experiment

    The controls below keep the default workload intentionally small so it runs on a laptop. Increase the batch, rollout length, or PPO updates when you want stronger timing or learning signals.
    """)
    return


@app.cell
def _(FAMILY_NAMES, mo):
    family_pick = mo.ui.dropdown(options=list(FAMILY_NAMES), value="chain", label="scenario type")
    seed_pick = mo.ui.slider(start=0, stop=999, step=1, value=0, include_input=True, label="seed")
    batch_pick = mo.ui.slider(start=1, stop=64, step=1, value=8, include_input=True, label="batch")
    rollout_steps_pick = mo.ui.slider(start=1, stop=64, step=1, value=8, include_input=True, label="rollout steps")
    train_updates_pick = mo.ui.slider(start=0, stop=5, step=1, value=1, include_input=True, label="PPO updates")
    mo.vstack(
        [
            mo.md("### Experiment controls"),
            mo.hstack(
                [family_pick, seed_pick, batch_pick, rollout_steps_pick, train_updates_pick],
                justify="start",
                wrap=True,
            ),
        ]
    )
    return (
        batch_pick,
        family_pick,
        rollout_steps_pick,
        seed_pick,
        train_updates_pick,
    )


@app.cell
def _(TycoonLE, family_pick):
    env = TycoonLE(split="dev", family=family_pick.value)
    return (env,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Run a batched first-valid rollout

    This cell uses `jax.vmap` to reset and step many worlds in parallel, `jax.lax.scan` to loop over time, and `jax.jit` to compile the batched rollout. The policy is still simple; the point is to show the environment's transform-friendly shape contract.
    """)
    return


@app.cell
def _(
    batch_pick,
    env,
    jax,
    pd,
    rollout_steps_pick,
    run_batched_first_valid,
    seed_pick,
):
    rollout_result = run_batched_first_valid(
        env,
        jax.random.PRNGKey(int(seed_pick.value)),
        int(batch_pick.value),
        int(rollout_steps_pick.value),
    )
    rollout_summary = pd.DataFrame(
        [
            {
                "batch": int(batch_pick.value),
                "steps": int(rollout_steps_pick.value),
                "elapsed_seconds": round(rollout_result["elapsed_seconds"], 5),
                "transitions_per_second": round(rollout_result["transitions_per_second"], 1),
                "mean_return": round(float(rollout_result["returns"].mean()), 4),
                "mean_score": round(float(rollout_result["scores"].mean()), 4),
                "mean_cargo": round(float(rollout_result["cargo"].mean()), 4),
                "mean_profit": round(float(rollout_result["profit"].mean()), 4),
            }
        ]
    )
    rollout_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run a tiny PPO smoke train

    This is not meant to produce a strong agent. It verifies the policy network, masked action sampling, rollout collection, advantage calculation, and PPO update path all work together on the selected scenario type.
    """)
    return


@app.cell
def _(
    PPOConfig,
    create_train_state,
    env,
    jax,
    make_train_step,
    pd,
    replace,
    scalar_metrics,
    seed_pick,
    train_updates_pick,
):
    train_updates = int(train_updates_pick.value)
    train_rows = []
    if train_updates > 0:
        train_config = replace(PPOConfig(), num_envs=4, rollout_length=4, update_epochs=1, hidden_sizes=(32,))
        train_state = create_train_state(jax.random.PRNGKey(int(seed_pick.value) + 10_000), train_config)
        train_step = make_train_step(env, train_config)
        train_key = jax.random.PRNGKey(int(seed_pick.value) + 20_000)
        for update in range(1, train_updates + 1):
            train_key, step_key = jax.random.split(train_key)
            train_state, metrics = train_step(train_state, step_key)
            train_rows.append({"update": update, **scalar_metrics(metrics)})
    training_table = pd.DataFrame(train_rows)
    training_table
    return (training_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Read the training metrics

    `loss`, `mean_reward`, and `mean_score` should be treated as smoke-test signals at these settings. Use more environments, longer rollouts, and more updates before interpreting them as a meaningful learning curve.
    """)
    return


@app.cell
def _(mo, plot_columns, training_table):
    if training_table.empty:
        training_output = mo.md("Set `PPO updates` above zero to run the PPO smoke train.")
    else:
        training_output = plot_columns(training_table, ["loss", "mean_reward", "mean_score"], "PPO smoke metrics")
    training_output
    return


if __name__ == "__main__":
    app.run()
