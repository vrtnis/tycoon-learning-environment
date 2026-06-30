from __future__ import annotations

from hud import Taskset

from env import tycoonle_plan


_TASKS = [
    tycoonle_plan(family="single_route", seed=4101, action_budget=6, candidate_limit=12),
    tycoonle_plan(family="low_cash", seed=4127, action_budget=7, candidate_limit=12),
    tycoonle_plan(family="chain", seed=4159, action_budget=8, candidate_limit=12),
    tycoonle_plan(family="mixed_network", seed=4201, action_budget=8, candidate_limit=12),
    tycoonle_plan(family="terrain_gap", seed=4231, action_budget=7, candidate_limit=12),
    tycoonle_plan(family="chain", seed=20_000, action_budget=8, candidate_limit=12, show_rank_score=False),
]

tasks = Taskset("tycoonle-hud-smoke", _TASKS)
