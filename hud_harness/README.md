# TycoonLE HUD Harness

HUD harness for running TycoonLE as a tool-using agent environment.

The harness exposes TycoonLE through a HUD `mcp` capability. Agents receive a seeded logistics task, inspect the world with tools, choose executable candidate action indices, and leave behind a final simulator state that the HUD task grades.

## Layout

```text
hud_harness/
  env.py              # HUD environment and task template
  tasks.py            # concrete TycoonLE taskset
  Dockerfile.hud      # HUD deploy image
  pyproject.toml      # HUD harness dependencies
  tycoonle_hud/       # prompts, MCP tools, session state, scoring
```

## Local Setup

Install the main repo and the HUD harness in the same Python environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install -e .\hud_harness
```

Install the HUD CLI separately if it is not already available:

```powershell
uv tool install hud-python --python 3.12
```

## Run Locally

List the tasks:

```powershell
hud task list --source .\hud_harness
```

Run the first task against an agent:

```powershell
hud eval .\hud_harness\tasks.py openai --model gpt-5 --max-steps 30
```

Run all seeded tasks and repeat each task for reward-spread checks:

```powershell
hud eval .\hud_harness\tasks.py openai --model gpt-5 --all --group 3 --max-steps 30
```

## Deploy

From `hud_harness/`:

```powershell
hud set HUD_API_KEY=...
hud deploy
hud sync tasks tycoonle-hud-smoke
```

`Dockerfile.hud` installs the published TycoonLE package from GitHub for hosted runs. Local runs prefer the checkout's `python/` package when this harness is inside the repo.

## Tool Contract

The agent should use the `tycoonle` MCP capability:

- `observe_world(detail="summary")`
- `list_actions(limit=None)`
- `step(action_index, reason="")`
- `finish(summary="")`

The grader scores the final TycoonLE metrics, valid-action rate, and objective progress. The task prompt intentionally requires executable candidate indices rather than free-form route descriptions.
