import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import pandas as pd
    import marimo as mo

    REPO_ROOT = Path(__file__).resolve().parents[1]
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PYTHON_SRC = REPO_ROOT / "python"
    if PYTHON_SRC.exists() and str(PYTHON_SRC) not in sys.path:
        sys.path.insert(0, str(PYTHON_SRC))
    if str(NOTEBOOK_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOK_DIR))

    from opengfx_render import render_opengfx_map
    from tycoonle_jax import FAMILY_NAMES, TycoonLE
    from tycoonle_jax.constants import MAX_CANDIDATES
    from tycoonle_jax.replay import (
        build_execution_trace,
        decode_candidate,
        decode_candidate_action,
        decode_diagnostics,
        decode_metrics,
    )

    def candidate_frame_all(state):
        mask = list(jax.device_get(state.action_mask))
        rows = []
        for index in range(MAX_CANDIDATES):
            candidate = decode_candidate(state, index)
            estimates = candidate["estimates"]
            rows.append(
                {
                    "index": index,
                    "mask": bool(mask[index]),
                    "kind": candidate["kind"],
                    "feasible": candidate["feasible"],
                    "executable": candidate["directlyExecutable"],
                    "loan_needed": candidate["requiresLoan"],
                    "rank": candidate["rankScore"],
                    "cost": estimates["totalCost"],
                    "monthly_profit": estimates["monthlyProfit"],
                    "path_length": estimates["pathLength"],
                    "diagnostics": ", ".join(candidate["diagnostics"]),
                    "description": candidate["description"],
                }
            )
        return pd.DataFrame(rows)

    def action_summary_frame(candidate, action):
        rows = [{"field": "candidate_id", "value": candidate["id"]}]
        rows.extend({"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in action.items())
        return pd.DataFrame(rows)

    def metric_delta_frame(before_state, after_state):
        before = decode_metrics(before_state)
        after = decode_metrics(after_state)
        rows = []
        for key, before_value in before.items():
            if key == "breakdown" or not isinstance(before_value, (int, float)):
                continue
            after_value = after[key]
            delta = after_value - before_value if isinstance(after_value, (int, float)) else None
            rows.append({"metric": key, "before": before_value, "after": after_value, "delta": delta})
        return pd.DataFrame(rows)

    def frame_pair_figure(before_state, after_state):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(render_opengfx_map(before_state, REPO_ROOT))
        axes[0].set_title("Before")
        axes[1].imshow(render_opengfx_map(after_state, REPO_ROOT))
        axes[1].set_title("After")
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout()
        return fig

    return (
        FAMILY_NAMES,
        MAX_CANDIDATES,
        TycoonLE,
        action_summary_frame,
        build_execution_trace,
        candidate_frame_all,
        decode_candidate,
        decode_candidate_action,
        decode_diagnostics,
        frame_pair_figure,
        jax,
        jnp,
        metric_delta_frame,
        mo,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Candidate actions and reward effects

    TycoonLE agents do not issue arbitrary commands. At every step, the environment builds a fixed candidate frontier of route, vehicle, finance, repay, and wait actions. The agent chooses a candidate index, and the action mask decides what is executable.

    This notebook focuses on one decision at a time: inspect the frontier, choose an index, execute it, and compare the immediate reward and state changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Reset a generated world

    Pick a scenario type and seed. The reset state includes terrain, industries, towns, company finances, objective targets, and the initial candidate action table.
    """)
    return


@app.cell
def _(FAMILY_NAMES, mo):
    family_pick = mo.ui.dropdown(options=list(FAMILY_NAMES), value="chain", label="scenario type")
    seed_pick = mo.ui.slider(start=0, stop=999, step=1, value=0, include_input=True, label="seed")
    mo.vstack([mo.md("### Reset state"), mo.hstack([family_pick, seed_pick], justify="start", wrap=True)])
    return family_pick, seed_pick


@app.cell
def _(TycoonLE, family_pick, jax, seed_pick):
    env = TycoonLE(split="dev", family=family_pick.value)
    state, _ = env.reset(jax.random.PRNGKey(int(seed_pick.value)))
    return env, state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Read the candidate table

    The candidate table is the policy-facing action interface. Rows with `mask = True` are valid choices. Diagnostic tags explain constraints such as financing, route conflicts, bridge needs, or upstream supply requirements.
    """)
    return


@app.cell
def _(candidate_frame_all, state):
    candidate_table = candidate_frame_all(state)
    candidate_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Choose and execute one action

    Select a candidate index from the table above. You can deliberately pick an invalid or padded index to see how the environment records invalid actions without crashing.
    """)
    return


@app.cell
def _(MAX_CANDIDATES, jax, mo, state):
    valid_indices = [int(index) for index, active in enumerate(jax.device_get(state.action_mask)) if bool(active)]
    default_action = valid_indices[0] if valid_indices else 0
    action_index = mo.ui.slider(
        start=0,
        stop=MAX_CANDIDATES - 1,
        step=1,
        value=default_action,
        include_input=True,
        label="action index",
    )
    mo.vstack([mo.md("### Execute one action"), action_index])
    return (action_index,)


@app.cell
def _(
    action_index,
    build_execution_trace,
    decode_candidate,
    decode_candidate_action,
    decode_diagnostics,
    env,
    jax,
    jnp,
    state,
):
    chosen_index = int(action_index.value)
    selected_candidate = decode_candidate(state, chosen_index)
    chosen_action = decode_candidate_action(state, chosen_index)
    after_state, after_timestep = env.step(state, jnp.array(chosen_index, dtype=jnp.int32))
    extras = jax.device_get(after_timestep.extras)
    execution_trace = build_execution_trace(state, after_state, chosen_action)
    selected_diagnostics = decode_diagnostics(state.candidate.diagnostics[chosen_index])
    return (
        after_state,
        chosen_action,
        chosen_index,
        execution_trace,
        extras,
        selected_candidate,
        selected_diagnostics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect what happened

    The selected action summary shows the decoded command, validity, reward, and diagnostics. The next tables and maps compare the before and after state so the effect of the action is concrete.
    """)
    return


@app.cell
def _(
    action_summary_frame,
    chosen_action,
    chosen_index,
    extras,
    mo,
    selected_candidate,
    selected_diagnostics,
):
    selected_action_table = action_summary_frame(selected_candidate, chosen_action)
    mo.vstack(
        [
            mo.md(
                f"Selected index `{chosen_index}` has type `{chosen_action['type']}`. "
                f"Valid action: `{bool(extras['valid_action'])}`. "
                f"Reward: `{float(extras['reward']):.4f}`. "
                f"Diagnostics: `{', '.join(selected_diagnostics) or 'none'}`."
            ),
            selected_action_table,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Immediate metric changes

    Rewards are local step feedback, but TycoonLE is a delayed-return environment. A build action often spends cash now, while cargo and profit arrive only after operating the route for later months.
    """)
    return


@app.cell
def _(after_state, metric_delta_frame, state):
    reward_metrics_table = metric_delta_frame(state, after_state)
    reward_metrics_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build trace

    For route-building actions, the execution trace breaks the action into preparation, stations, track or road segments, vehicles, and final route readiness. Non-build actions leave this table empty.
    """)
    return


@app.cell
def _(execution_trace, pd):
    execution_steps_table = pd.DataFrame(execution_trace["steps"]) if execution_trace else pd.DataFrame()
    execution_steps_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Before and after map

    These OpenGFX sprite maps are useful for checking whether a chosen action changed the visible network. For example, a build action should add a route, while a wait action usually changes operations and metrics rather than terrain.
    """)
    return


@app.cell
def _(after_state, frame_pair_figure, state):
    comparison_figure = frame_pair_figure(state, after_state)
    comparison_figure
    return


if __name__ == "__main__":
    app.run()
