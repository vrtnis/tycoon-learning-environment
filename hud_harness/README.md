# TycoonLE HUD Harness

<a href="https://hud.ai"><img alt="HUD" src="https://mintcdn.com/hud-f5fd7c15/DDRwdBWTW6XznZVO/logo/hud_logo.svg?fit=max&amp;auto=format&amp;n=DDRwdBWTW6XznZVO&amp;q=85&amp;s=eb2d6ce2615c6e20f2124dc6451ac9ff" width="44" /></a>

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
.\.venv\Scripts\hud.exe task list --source .\hud_harness
```

Run the first task against an agent:

```powershell
.\.venv\Scripts\hud.exe eval .\hud_harness\tasks.py openai --model gpt-5 --max-steps 30 --gateway --yes
```

Run all seeded tasks and repeat each task for reward-spread checks:

```powershell
.\.venv\Scripts\hud.exe eval .\hud_harness\tasks.py openai --model gpt-5 --all --group 3 --max-steps 30 --gateway --yes
```

## Sample HUD Runs

These sample links are smoke runs on the same `single_route` task (`seed=4101`, `max_steps=10`). They verify that agents can use the HUD tool loop and produce comparable TycoonLE outcomes. Treat them as harness checks, not benchmark claims.

| Model | Reward | Final score | Cargo delivered | Operating profit | Valid actions | Sample trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt-5` | `0.5654` | `55.892` | `386.490` | `$970.582` | `6/6` | [job](https://hud.ai/jobs/12a5db0056f44039b881076f45bee6b5) / [trace](https://hud.ai/trace/35529ba4-8411-40d3-90be-e5ce1c388641) |
| `claude-sonnet-4-6` | `0.4994` | `49.309` | `194.031` | `-$11,606.644` | `6/6` | [job](https://hud.ai/jobs/ccf2b997aaa34f5ba3ca4b7ed33bf8aa) / [trace](https://hud.ai/trace/733714d0-f001-4fb1-bceb-c81e33615d5c) |
| `gemini-3.5-flash` | `0.5654` | `55.892` | `386.490` | `$970.582` | `6/6` | [job](https://hud.ai/jobs/4e5b2c878c5d41959753b5733b8d4fab) / [trace](https://hud.ai/trace/11a792f8-97d7-47e7-8590-c1b4baac768e) |

In these runs, `gpt-5` and `gemini-3.5-flash` took a loan to unlock and build the higher-throughput rail coal route. `claude-sonnet-4-6` chose the immediately affordable road coal route, which was valid but lower-scoring on this seed.

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
