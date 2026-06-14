# Tycoon Learning Environment

[![JAX Accelerated](assets/jax-accelerated.svg)](https://github.com/google/jax) [![Discord](https://img.shields.io/badge/Discord-Join%20server-5865F2?logo=discord&logoColor=white)](https://discord.gg/GPwEgANZKX)

![TycoonLE replay interface](assets/tycoonLE.png)

Tycoon Learning Environment (TycoonLE) is a reinforcement learning environment for economically grounded, long-horizon planning. Agents operate in a simulated logistics economy where they allocate capital, build transport routes, move cargo, manage debt, and optimize delayed returns.

It is designed to study action legality, candidate-frontier decision interfaces, financing timing, delayed rewards, procedural variation, and replayable audit traces.

TycoonLE uses a fixed-shape interface. Agents choose among valid route, finance, and wait candidates, making rollouts compatible with JAX transformations such as `jit`, `vmap`, and `scan`.

The replay UI makes policies inspectable through route choices, cargo flow, financing behavior, reward, score, and profit over time.

TycoonBench provides a companion benchmark report for comparing agent and model performance on TycoonLE planning tasks: [vrtnis.github.io/tycoonbench](https://vrtnis.github.io/tycoonbench/).

## Install

Use Python 3.11 or 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
npm install
```

## Quickstart

```python
import jax
from tycoonle_jax import TycoonLE

env = TycoonLE(split="dev", family="chain")
state, timestep = env.reset(jax.random.PRNGKey(0))
action = timestep.observation.action_mask.argmax()
state, timestep = env.step(state, action)
```

Export a replay:

```powershell
.\.venv\Scripts\python.exe examples\quickstart.py
npm run dev
```

Open the browser UI and load `runs/quickstart/replay.json`.

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
npm run build
```

## Interactive notebooks

Install the notebook dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,notebooks]"
```

Open a marimo notebook:

```powershell
.\.venv\Scripts\python.exe -m marimo edit notebooks\01_quickstart_rollout.py
```

For the read-only app view:

```powershell
.\.venv\Scripts\python.exe -m marimo run notebooks\01_quickstart_rollout.py
```

The molab badges open server-mode notebooks so TycoonLE can use JAX and the repo package. Molab may ask you to sign in before launching the server. The plain GitHub preview is read-only and may show source or dependency errors; the exact local app view is still `marimo run`.

The notebook set covers:

| Notebook | What it covers | Run on molab |
| --- | --- | --- |
| [`01_quickstart_rollout.py`](notebooks/01_quickstart_rollout.py) | Interactive reset, rollout, OpenGFX sprite map render, candidate table, metrics, and replay export. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/vrtnis/tycoon-learning-environment/blob/main/notebooks/01_quickstart_rollout.py/server) |
| [`02_action_candidates_and_rewards.py`](notebooks/02_action_candidates_and_rewards.py) | Inspect candidate actions, execute a selected action, and compare reward/metric deltas with before/after sprite maps. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/vrtnis/tycoon-learning-environment/blob/main/notebooks/02_action_candidates_and_rewards.py/server) |
| [`03_jax_rollouts_and_training_smoke.py`](notebooks/03_jax_rollouts_and_training_smoke.py) | Run JAX `jit`/`vmap`/`scan` rollouts and a tiny PPO smoke train. | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/vrtnis/tycoon-learning-environment/blob/main/notebooks/03_jax_rollouts_and_training_smoke.py/server) |

## TycoonBench OpenRouter runs

The benchmark report is generated from JSON artifacts in `src/benchmark/generated/`.

Create the fixed benchmark task files:

```powershell
npm run benchmark:generate
```

Preview OpenRouter requests without making API calls:

```powershell
npm run benchmark:run:preview -- --models gpt-55 --tasks singleRoute
```

Run configured OpenRouter models:

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
npm run benchmark:run:openrouter -- --models gpt-55,gemini-35-flash --tasks singleRoute,chain
npm run benchmark:extract
npm run build
```

Model slugs live in `benchmark/config/openrouter-models.mjs`. Keep `provider.allow_fallbacks` disabled for benchmark runs unless the benchmark target is the router itself.

## Training

Run a small PPO smoke train:

```powershell
.\.venv\Scripts\python.exe examples\train_ppo_jax.py --updates 1 --num-envs 4 --rollout-length 4 --update-epochs 1 --hidden-sizes 32
```

## Citation

If you find this work useful, consider citing:

```bibtex
@software{tycoonle,
  title = {TycoonLE},
  author = {TycoonLE contributors},
  year = {2026},
  url = {https://github.com/vrtnis/tycoon-learning-environment}
}
```

## Artwork Credits

TycoonLE uses sprite artwork from [OpenGFX](https://github.com/OpenTTD/OpenGFX), an open-source graphics base set for [OpenTTD](https://www.openttd.org/).
