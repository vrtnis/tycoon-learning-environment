# Tycoon Learning Environment

![TycoonLE replay interface](assets/tycoonLE.png)

TycoonLE is a reinforcement learning environment for transport-tycoon style logistics planning. It is implemented with JAX and Jumanji-style `reset` and `step` APIs, using fixed-shape observations for vectorized rollouts and a browser replay viewer for inspection.

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

## Training

Run a small PPO smoke train:

```powershell
.\.venv\Scripts\python.exe examples\train_ppo_jax.py --updates 1 --num-envs 4 --rollout-length 4 --update-epochs 1 --hidden-sizes 32
```

## Citation

If you use TycoonLE, cite this repository:

```bibtex
@software{tycoonle,
  title = {TycoonLE},
  author = {TycoonLE contributors},
  year = {2026},
  url = {https://github.com/vrtnis/tycoonLE}
}
```
