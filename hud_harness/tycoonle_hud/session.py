from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

from tycoonle_hud.paths import ensure_repo_python_path
from tycoonle_hud.world import compact_observation, full_observation, objective_from_state, visible_actions

ensure_repo_python_path()

from tycoonle_jax import TycoonLE
from tycoonle_jax.constants import FAMILY_NAMES, MAX_CANDIDATES, SPLIT_NAMES


@dataclass
class TycoonLESession:
    split: str = "dev"
    family: str = "chain"
    seed: int = 20_000
    candidate_limit: int = 12
    action_budget: int = 6
    show_rank_score: bool = True
    env: TycoonLE | None = None
    state: Any | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    invalid_tool_calls: int = 0
    finished: bool = False
    final_answer: str = ""
    final_summary: str = ""

    def reset(
        self,
        *,
        split: str,
        family: str,
        seed: int,
        candidate_limit: int,
        action_budget: int,
        show_rank_score: bool,
    ) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown TycoonLE split: {split}")
        if family not in FAMILY_NAMES:
            raise ValueError(f"unknown TycoonLE family: {family}")
        self.split = split
        self.family = family
        self.seed = int(seed)
        self.candidate_limit = max(1, min(int(candidate_limit), MAX_CANDIDATES))
        self.action_budget = max(1, int(action_budget))
        self.show_rank_score = bool(show_rank_score)
        self.env = TycoonLE(split=split, family=family)
        self.state, _ = self.env.reset(jax.random.PRNGKey(self.seed))
        self.steps = []
        self.invalid_tool_calls = 0
        self.finished = False
        self.final_answer = ""
        self.final_summary = ""

    def initial_prompt_state(self) -> dict[str, Any]:
        self._require_started()
        return {
            "split": self.split,
            "family": self.family,
            "seed": self.seed,
            "candidateLimit": self.candidate_limit,
            "actionBudget": self.action_budget,
            "showRankScore": self.show_rank_score,
            "observation": compact_observation(self.state),
            "visibleActions": self.list_actions(),
        }

    def observe(self, detail: str = "summary") -> dict[str, Any]:
        self._require_started()
        if detail == "full":
            return full_observation(self.state)
        return compact_observation(self.state)

    def list_actions(self, limit: int | None = None) -> list[dict[str, Any]]:
        self._require_started()
        resolved_limit = self.candidate_limit if limit is None else max(1, min(int(limit), self.candidate_limit))
        return visible_actions(self.state, limit=resolved_limit, show_rank_score=self.show_rank_score)

    def step(self, action_index: int, reason: str = "") -> dict[str, Any]:
        self._require_started()
        if self.finished:
            self.invalid_tool_calls += 1
            return {"ok": False, "error": "rollout is already finished", "final": self.finish()}
        if len(self.steps) >= self.action_budget:
            self.finished = True
            self.invalid_tool_calls += 1
            return {"ok": False, "error": "action budget exhausted", "final": self.finish()}

        action_index = int(action_index)
        before_actions = {item["actionIndex"]: item for item in self.list_actions(limit=self.candidate_limit)}
        valid_before = action_index in before_actions
        selected = before_actions.get(action_index)

        next_state, timestep = self.env.step(self.state, np.int32(action_index))
        reward = float(np.asarray(jax.device_get(timestep.reward)))
        self.state = next_state
        metrics = compact_observation(self.state)["metrics"]
        done = bool(np.asarray(jax.device_get(self.state.done)))
        step_record = {
            "step": len(self.steps) + 1,
            "actionIndex": action_index,
            "validAction": valid_before,
            "selected": selected,
            "reason": str(reason or "")[:500],
            "reward": round(reward, 4),
            "metrics": metrics,
            "done": done,
        }
        self.steps.append(step_record)
        if done or len(self.steps) >= self.action_budget:
            self.finished = True
        return {
            "ok": True,
            "step": step_record,
            "finished": self.finished,
            "observation": compact_observation(self.state),
            "visibleActions": [] if self.finished else self.list_actions(),
        }

    def finish(self, summary: str = "") -> dict[str, Any]:
        self._require_started()
        self.finished = True
        if summary:
            self.final_summary = str(summary)[:1000]
        return {
            "finished": True,
            "stepsUsed": len(self.steps),
            "actionBudget": self.action_budget,
            "finalMetrics": compact_observation(self.state)["metrics"],
            "summary": self.final_summary,
        }

    def record_final_answer(self, answer: Any) -> None:
        self.final_answer = "" if answer is None else str(answer)[:4000]

    def snapshot(self) -> dict[str, Any]:
        self._require_started()
        observation = compact_observation(self.state)
        return {
            "split": self.split,
            "family": self.family,
            "seed": self.seed,
            "candidateLimit": self.candidate_limit,
            "actionBudget": self.action_budget,
            "showRankScore": self.show_rank_score,
            "finished": self.finished,
            "steps": list(self.steps),
            "invalidToolCalls": self.invalid_tool_calls,
            "finalAnswer": self.final_answer,
            "finalSummary": self.final_summary,
            "observation": observation,
            "objective": objective_from_state(self.state),
            "metrics": observation["metrics"],
        }

    def _require_started(self) -> None:
        if self.env is None or self.state is None:
            raise RuntimeError("TycoonLE session has not been initialized for a HUD task")
