import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import jax
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
    from tycoonle_jax import FAMILY_NAMES, TycoonLE, export_replay, rollout_first_valid

    def state_at(rollout, key, index):
        return jax.tree.map(lambda leaf: leaf[index], rollout[key])

    def build_candidate_frame(event):
        rows = []
        mask = event["info"]["actionMask"]
        for index, candidate in enumerate(event["info"]["candidateActions"]):
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
                    "monthly_delivered": estimates["monthlyDelivered"],
                    "diagnostics": ", ".join(candidate["diagnostics"]),
                    "description": candidate["description"],
                }
            )
        return pd.DataFrame(rows)

    def metrics_frame(event):
        before = event["before"]["metrics"]
        after = event["after"]["metrics"]
        rows = []
        for key, before_value in before.items():
            if key == "breakdown" or not isinstance(before_value, (int, float)):
                continue
            after_value = after[key]
            delta = after_value - before_value if isinstance(after_value, (int, float)) else None
            rows.append({"metric": key, "before": before_value, "after": after_value, "delta": delta})
        return pd.DataFrame(rows)

    def frame_figure(state, title):
        frame = render_opengfx_map(state, REPO_ROOT)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(frame)
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    return (
        FAMILY_NAMES,
        REPO_ROOT,
        TycoonLE,
        build_candidate_frame,
        export_replay,
        frame_figure,
        jax,
        json,
        metrics_frame,
        mo,
        pd,
        rollout_first_valid,
        state_at,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TycoonLE Quickstart Rollout

    Tycoon Learning Environment (TycoonLE) is a JAX reinforcement learning environment for logistics planning. Agents build routes, move cargo, manage cash and debt, and try to improve delayed economic outcomes over a generated transport world.

    This notebook is the shortest end-to-end tour: choose a scenario, run a deterministic baseline policy, inspect one decision step, and export a replay for the browser UI.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pick a scenario

    A **scenario type** controls the kind of planning problem, such as a single route, low-cash financing, a supply chain, or terrain-constrained routing. The **seed** chooses a generated world within that scenario type. The rollout length controls how many candidate decisions the baseline policy takes.
    """)
    return


@app.cell
def _(FAMILY_NAMES, mo):
    family_pick = mo.ui.dropdown(options=list(FAMILY_NAMES), value="chain", label="scenario type")
    seed_pick = mo.ui.slider(start=0, stop=999, step=1, value=0, include_input=True, label="seed")
    steps_pick = mo.ui.slider(start=1, stop=48, step=1, value=12, include_input=True, label="rollout steps")
    mo.vstack(
        [
            mo.md("### Scenario"),
            mo.hstack([family_pick, seed_pick, steps_pick], justify="start", wrap=True),
        ]
    )
    return family_pick, seed_pick, steps_pick


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Run a baseline rollout

    The environment exposes a fixed-size action table plus an `action_mask`. For this quickstart, the policy simply takes the first valid candidate at every step. This is intentionally simple: it gives us a reproducible trace to inspect before introducing learning.
    """)
    return


@app.cell
def _(
    TycoonLE,
    export_replay,
    family_pick,
    jax,
    rollout_first_valid,
    seed_pick,
    steps_pick,
):
    env = TycoonLE(split="dev", family=family_pick.value)
    rollout = rollout_first_valid(env, jax.random.PRNGKey(int(seed_pick.value)), num_steps=int(steps_pick.value))
    replay = export_replay(rollout)
    events = replay["events"]
    summary = replay["summary"]
    return env, events, replay, rollout, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Read the rollout summary

    The summary is the high-level outcome of the rollout: how many steps ran, final score, delivered cargo, and operating profit. It is useful for comparing seeds, scenario types, or policies before digging into individual decisions.
    """)
    return


@app.cell
def _(pd, replay, summary):
    scenario = replay["scenario"]
    summary_table = pd.DataFrame(
        [
            {
                "split": scenario["split"],
                "scenario_type": scenario["family"],
                "seed": scenario["seed"],
                **summary,
            }
        ]
    )
    summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect one decision

    Move the event slider to look at the state before a selected action. The map uses the same OpenGFX sprite assets as the replay UI, while the tables below show the candidate frontier and metric changes for that step.
    """)
    return


@app.cell
def _(events, mo):
    max_event_index = max(0, len(events) - 1)
    step_pick = mo.ui.slider(
        start=0,
        stop=max_event_index,
        step=1,
        value=0,
        include_input=True,
        label="event index",
    )
    mo.vstack([mo.md("### Step inspector"), step_pick])
    return (step_pick,)


@app.cell
def _(events, rollout, state_at, step_pick):
    event_index = min(int(step_pick.value), max(0, len(events) - 1))
    selected_event = events[event_index]
    selected_state = state_at(rollout, "before_states", event_index)
    return event_index, selected_event, selected_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### World snapshot

    This sprite map shows the generated industries, towns, terrain, and later any routes built by the policy. It is meant to connect the numeric action table to the world the agent is acting in.
    """)
    return


@app.cell
def _(event_index, frame_figure, selected_event, selected_state):
    map_figure = frame_figure(
        selected_state,
        f"Before event {event_index + 1}: {selected_event['action']['type']}",
    )
    map_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Candidate frontier

    Each row is a candidate action the agent may choose. `mask` tells whether the action is executable now, `loan_needed` explains financing pressure, and `rank` is the environment's heuristic ordering signal rather than a learned policy score.
    """)
    return


@app.cell
def _(build_candidate_frame, selected_event):
    candidate_table = build_candidate_frame(selected_event)
    candidate_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metric delta

    This table compares metrics before and after the selected action. For route-building steps, cash may drop immediately while score, cargo, or profit improve only after later wait steps deliver goods.
    """)
    return


@app.cell
def _(metrics_frame, selected_event):
    metric_table = metrics_frame(selected_event)
    metric_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Export a replay

    The replay JSON captures the rollout as an inspectable audit trace. Load this file in the browser UI to step through route choices, cargo flow, financing behavior, rewards, and score over time.
    """)
    return


@app.cell
def _(REPO_ROOT, json, mo, replay):
    replay_relative_path = "runs/marimo/quickstart/replay.json"
    replay_path = REPO_ROOT / replay_relative_path
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    mo.md(f"Replay exported to `{replay_relative_path}`.")
    return


if __name__ == "__main__":
    app.run()
