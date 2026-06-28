from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import verifiers as vf
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
for path in (ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tycoonle_jax import TycoonLE
from tycoonle_jax.constants import (
    ACTION_ADD_VEHICLE,
    ACTION_BUILD_ROUTE,
    ACTION_INVALID,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    ACTION_WAIT,
    CARGO_LABELS,
    CARGO_NAMES,
    FAMILY_NAMES,
    MAX_CANDIDATES,
    MODE_NAMES,
    NODE_KIND_NAMES,
    TERRAIN_NAMES,
)


SYSTEM_PROMPT = (
    "You are controlling a transport company in TycoonLE, a reinforcement-learning logistics simulator. "
    "Choose exactly one visible executable candidate action per turn. Return JSON only."
)

ACTION_CONTRACT = (
    'Return only valid JSON with this shape: {"action_index": 0, "reason": "short operational reason"}. '
    "The action_index must be one of the visible candidate indices."
)

ACTION_NAMES = {
    ACTION_INVALID: "invalid",
    ACTION_BUILD_ROUTE: "build_route",
    ACTION_ADD_VEHICLE: "add_vehicle",
    ACTION_WAIT: "wait",
    ACTION_TAKE_LOAN: "take_loan",
    ACTION_REPAY_LOAN: "repay_loan",
}

DIAGNOSTIC_NAMES = (
    "requires_loan",
    "cannot_finance",
    "invalid_path",
    "water_crossing",
    "town_crossing",
    "crosses_existing_route",
    "shares_existing_route",
    "missing_upstream",
)


@dataclass(frozen=True)
class ActionParse:
    action_index: int | None
    raw: str
    error: str | None = None


@dataclass(frozen=True)
class StepResult:
    action_index: int
    reward: float
    done: bool
    metrics: dict[str, float]


class TycoonLEPrimeEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        split: str = "dev",
        families: list[str] | None = None,
        seed_start: int = 20_000,
        num_examples: int = 5,
        candidate_limit: int = 12,
        action_budget: int = 6,
    ) -> None:
        self.split = split
        self.families = families or list(FAMILY_NAMES)
        self.seed_start = int(seed_start)
        self.num_examples = int(num_examples)
        self.candidate_limit = int(candidate_limit)
        self.action_budget = int(action_budget)
        dataset = _build_dataset(
            split=self.split,
            families=self.families,
            seed_start=self.seed_start,
            num_examples=self.num_examples,
            candidate_limit=self.candidate_limit,
            action_budget=self.action_budget,
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            rubric=_build_rubric(),
            max_turns=self.action_budget + 1,
        )

    async def setup_state(self, state: vf.State) -> None:
        info = state["info"]
        tycoon_env = TycoonLE(split=info["split"], family=info["family"])
        tycoon_state, _ = tycoon_env.reset(jax.random.PRNGKey(int(info["seed"])))
        state["tycoon_env"] = tycoon_env
        state["tycoon_state"] = tycoon_state
        state["tycoon_steps"] = []
        state["tycoon_parse_errors"] = 0
        state["tycoon_invalid_actions"] = 0
        state["tycoon_total_reward"] = 0.0
        state["tycoon_final_metrics"] = _metrics_from_state(tycoon_state)
        state["tycoon_done"] = False
        state["visible_action_indices"] = _visible_action_indices(tycoon_state, self.candidate_limit)
        await super().setup_state(state)

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> vf.Messages:
        parsed = _parse_action(_last_assistant_content(messages))
        if parsed.error is not None or parsed.action_index is None:
            state["tycoon_parse_errors"] += 1
            return self._finish(state, f"Invalid response: {parsed.error or 'missing action_index'}")

        visible = set(state.get("visible_action_indices", []))
        if parsed.action_index not in visible:
            state["tycoon_invalid_actions"] += 1
            return self._finish(
                state,
                f"Invalid action_index {parsed.action_index}. Choose one of the visible candidates: {sorted(visible)}.",
            )

        result = _step_tycoon(state, parsed.action_index)
        if result.done or len(state["tycoon_steps"]) >= self.action_budget:
            return self._finish(state, _final_feedback(state))

        state["visible_action_indices"] = _visible_action_indices(state["tycoon_state"], self.candidate_limit)
        content = _render_prompt(
            state["tycoon_state"],
            split=state["info"]["split"],
            family=state["info"]["family"],
            seed=state["info"]["seed"],
            candidate_limit=self.candidate_limit,
            step_feedback=_step_feedback(result),
        )
        return [{"role": "user", "content": content}]

    def _finish(self, state: vf.State, content: str) -> vf.Messages:
        state["tycoon_done"] = True
        final = [{"role": "user", "content": f"Evaluation finished.\n{content}"}]
        state["final_env_response"] = final
        return final


def load_environment(
    split: str = "dev",
    families: str | list[str] | None = None,
    seed_start: int = 20_000,
    num_examples: int = 5,
    candidate_limit: int = 12,
    action_budget: int = 6,
    max_turns: int | None = None,
    **_: Any,
) -> vf.Environment:
    if max_turns is not None:
        action_budget = max(1, int(max_turns) - 1)
    return TycoonLEPrimeEnv(
        split=split,
        families=_parse_families(families),
        seed_start=seed_start,
        num_examples=num_examples,
        candidate_limit=candidate_limit,
        action_budget=action_budget,
    )


def _build_dataset(
    *,
    split: str,
    families: list[str],
    seed_start: int,
    num_examples: int,
    candidate_limit: int,
    action_budget: int,
) -> Dataset:
    rows = []
    for idx in range(num_examples):
        family = families[idx % len(families)]
        seed = seed_start + idx
        env = TycoonLE(split=split, family=family)
        tycoon_state, _ = env.reset(jax.random.PRNGKey(seed))
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _render_prompt(
                            tycoon_state,
                            split=split,
                            family=family,
                            seed=seed,
                            candidate_limit=candidate_limit,
                        ),
                    },
                ],
                "info": json.dumps(
                    {
                        "split": split,
                        "family": family,
                        "seed": seed,
                        "candidate_limit": candidate_limit,
                        "action_budget": action_budget,
                    }
                ),
            }
        )
    return Dataset.from_list(rows)


def _build_rubric() -> vf.Rubric:
    rubric = vf.Rubric()
    rubric.add_reward_func(json_format_reward, weight=0.10)
    rubric.add_reward_func(valid_action_rate_reward, weight=0.20)
    rubric.add_reward_func(final_score_reward, weight=0.70)
    rubric.add_metric(final_score_metric)
    rubric.add_metric(cargo_delivered_metric)
    rubric.add_metric(operating_profit_metric)
    rubric.add_metric(executed_steps_metric)
    rubric.add_metric(parse_error_metric)
    rubric.add_metric(invalid_action_metric)
    return rubric


async def json_format_reward(state: vf.State) -> float:
    return 0.0 if state.get("tycoon_parse_errors", 0) else 1.0


async def valid_action_rate_reward(state: vf.State) -> float:
    steps = state.get("tycoon_steps", [])
    invalid = float(state.get("tycoon_invalid_actions", 0))
    total = len(steps) + invalid
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, len(steps) / total))


async def final_score_reward(state: vf.State) -> float:
    score = float(state.get("tycoon_final_metrics", {}).get("score", 0.0))
    return max(-1.0, min(1.0, score / 100.0))


async def final_score_metric(state: vf.State) -> float:
    return float(state.get("tycoon_final_metrics", {}).get("score", 0.0))


async def cargo_delivered_metric(state: vf.State) -> float:
    return float(state.get("tycoon_final_metrics", {}).get("cargo_delivered", 0.0))


async def operating_profit_metric(state: vf.State) -> float:
    return float(state.get("tycoon_final_metrics", {}).get("operating_profit", 0.0))


async def executed_steps_metric(state: vf.State) -> float:
    return float(len(state.get("tycoon_steps", [])))


async def parse_error_metric(state: vf.State) -> float:
    return float(state.get("tycoon_parse_errors", 0))


async def invalid_action_metric(state: vf.State) -> float:
    return float(state.get("tycoon_invalid_actions", 0))


def _step_tycoon(state: vf.State, action_index: int) -> StepResult:
    tycoon_env: TycoonLE = state["tycoon_env"]
    tycoon_state, timestep = tycoon_env.step(state["tycoon_state"], np.int32(action_index))
    reward = float(np.asarray(jax.device_get(timestep.reward)))
    metrics = _metrics_from_state(tycoon_state)
    done = bool(np.asarray(jax.device_get(tycoon_state.done)))
    state["tycoon_state"] = tycoon_state
    state["tycoon_total_reward"] = float(state.get("tycoon_total_reward", 0.0)) + reward
    state["tycoon_final_metrics"] = metrics
    result = StepResult(action_index=action_index, reward=reward, done=done, metrics=metrics)
    state["tycoon_steps"].append(
        {
            "step": len(state["tycoon_steps"]) + 1,
            "action_index": action_index,
            "reward": round(reward, 4),
            "score": round(metrics["score"], 4),
            "cargo_delivered": round(metrics["cargo_delivered"], 4),
            "operating_profit": round(metrics["operating_profit"], 4),
        }
    )
    return result


def _render_prompt(
    state: Any,
    *,
    split: str,
    family: str,
    seed: int,
    candidate_limit: int,
    step_feedback: str | None = None,
) -> str:
    world = _world_from_state(state, seed=seed, split=split, family=family, candidate_limit=candidate_limit)
    lines = [
        "Choose the next TycoonLE action.",
        "",
        "Current world:",
        f"- split: {world['split']}, family: {world['family']}, seed: {world['seed']}",
        f"- step/month: {world['step']}/{world['month']} of {world['maxSteps']} steps / {world['maxMonths']} months",
        f"- map: {world['width']}x{world['height']}, terrain: {_format_mapping(world['terrain'])}",
        f"- cash: ${world['cash']:,.0f}, loan: ${world['loan']:,.0f}, max loan: ${world['maxLoan']:,.0f}, interest: {world['interestRate']:.3f}",
        f"- objective cargo: {world['objectiveCargo']}, delivered target: {world['deliveredTarget']:,.0f}, profit target: ${world['profitTarget']:,.0f}, route target: {world['routeTarget']}",
        f"- current score: {world['score']:.3f}, delivered: {world['cargoDelivered']:.1f}, profit: ${world['operatingProfit']:,.0f}",
    ]
    if step_feedback:
        lines.extend(["", "Previous result:", step_feedback])

    lines.extend(["", "Nodes:"])
    for node in world["nodes"]:
        lines.append(
            f"- {node['label']} at ({node['x']},{node['y']}), population {node['population']:.0f}, "
            f"produces [{_format_mapping(node['produces'])}], accepts [{_format_mapping(node['accepts'])}]"
        )

    lines.extend(["", "Visible executable candidate actions:"])
    if world["candidates"]:
        for candidate in world["candidates"]:
            lines.append(_format_candidate(candidate))
    else:
        lines.append("- none")

    lines.extend(["", ACTION_CONTRACT])
    return "\n".join(lines)


def _world_from_state(state: Any, *, seed: int, split: str, family: str, candidate_limit: int) -> dict[str, Any]:
    state = jax.device_get(state)
    terrain = np.asarray(state.terrain)
    terrain_mask = np.asarray(state.terrain_mask, dtype=bool)
    terrain_counts = {
        TERRAIN_NAMES[idx]: int(np.sum((terrain == idx) & terrain_mask))
        for idx in range(len(TERRAIN_NAMES))
    }

    nodes = []
    for idx, enabled in enumerate(np.asarray(state.node_mask, dtype=bool)):
        if not enabled:
            continue
        nodes.append(
            {
                "id": idx,
                "label": f"{NODE_KIND_NAMES[int(state.node_kind[idx])]}_{idx}",
                "kind": NODE_KIND_NAMES[int(state.node_kind[idx])],
                "x": int(state.node_x[idx]),
                "y": int(state.node_y[idx]),
                "population": round(float(state.node_population[idx]), 1),
                "produces": _cargo_amounts(np.asarray(state.node_produces[idx])),
                "accepts": _cargo_amounts(np.asarray(state.node_accepts[idx])),
            }
        )

    action_mask = np.asarray(state.action_mask, dtype=bool)
    candidates = []
    for idx in range(min(MAX_CANDIDATES, len(state.candidate.kind))):
        if not action_mask[idx]:
            continue
        candidate = _candidate_dict(state, idx, nodes)
        candidates.append(candidate)
        if len(candidates) >= candidate_limit:
            break

    objective_cargo = int(state.objective_cargo)
    metrics = np.asarray(state.metrics)
    return {
        "split": split,
        "family": family,
        "seed": seed,
        "scenarioSeed": int(state.seed),
        "width": int(state.width),
        "height": int(state.height),
        "step": int(state.step),
        "month": int(state.month),
        "maxSteps": int(state.max_steps),
        "maxMonths": int(state.max_months),
        "cash": round(float(state.cash), 2),
        "loan": round(float(state.loan), 2),
        "maxLoan": round(float(state.max_loan), 2),
        "interestRate": round(float(state.interest_rate), 4),
        "objectiveCargo": CARGO_NAMES[objective_cargo] if objective_cargo >= 0 else "all_cargo",
        "deliveredTarget": round(float(state.delivered_target), 2),
        "profitTarget": round(float(state.profit_target), 2),
        "routeTarget": round(float(state.route_target), 2),
        "terrain": terrain_counts,
        "nodes": nodes,
        "candidates": candidates,
        "score": round(float(metrics[0]), 3),
        "cargoDelivered": round(float(metrics[1]), 3),
        "operatingProfit": round(float(metrics[2]), 3),
    }


def _candidate_dict(state: Any, idx: int, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    cargo = int(state.candidate.cargo[idx])
    kind = int(state.candidate.kind[idx])
    source = int(state.candidate.source[idx])
    destination = int(state.candidate.destination[idx])
    return {
        "index": idx,
        "kind": ACTION_NAMES.get(kind, "unknown"),
        "source": source,
        "destination": destination,
        "sourceLabel": _node_label(nodes, source),
        "destinationLabel": _node_label(nodes, destination),
        "cargo": CARGO_NAMES[cargo],
        "cargoLabel": CARGO_LABELS[cargo],
        "mode": MODE_NAMES[int(state.candidate.mode[idx])],
        "vehicles": int(state.candidate.vehicles[idx]),
        "months": int(state.candidate.months[idx]),
        "amount": round(float(state.candidate.amount[idx]), 2),
        "totalCost": round(float(state.candidate.total_cost[idx]), 2),
        "monthlyProfit": round(float(state.candidate.monthly_profit[idx]), 2),
        "monthlyDelivered": round(float(state.candidate.monthly_delivered[idx]), 2),
        "rankScore": round(float(state.candidate.rank_score[idx]), 5),
        "requiresLoan": round(float(state.candidate.requires_loan[idx]), 2),
        "terrainCost": round(float(state.candidate.terrain_cost[idx]), 3),
        "congestion": round(float(state.candidate.congestion[idx]), 3),
        "diagnostics": _diagnostics(np.asarray(state.candidate.diagnostics[idx], dtype=bool)),
    }


def _format_candidate(candidate: dict[str, Any]) -> str:
    if candidate["kind"] == "build_route":
        return (
            f"- [{candidate['index']}] build_route {candidate['mode']} {candidate['cargo']} "
            f"{candidate['sourceLabel']} -> {candidate['destinationLabel']}; "
            f"vehicles {candidate['vehicles']}; cost ${candidate['totalCost']:,.0f}; "
            f"loan needed ${candidate['requiresLoan']:,.0f}; monthly profit ${candidate['monthlyProfit']:,.0f}; "
            f"monthly delivered {candidate['monthlyDelivered']:,.1f}; terrain {candidate['terrainCost']:.2f}; "
            f"congestion {candidate['congestion']:.2f}; rank {candidate['rankScore']:.3f}; "
            f"diagnostics [{', '.join(candidate['diagnostics']) or 'none'}]"
        )
    if candidate["kind"] == "take_loan":
        return f"- [{candidate['index']}] take_loan amount ${candidate['amount']:,.0f}; rank {candidate['rankScore']:.3f}"
    if candidate["kind"] == "repay_loan":
        return f"- [{candidate['index']}] repay_loan amount ${candidate['amount']:,.0f}; rank {candidate['rankScore']:.3f}"
    if candidate["kind"] == "add_vehicle":
        return (
            f"- [{candidate['index']}] add_vehicle route {candidate['sourceLabel']} -> "
            f"{candidate['destinationLabel']}; cost ${candidate['totalCost']:,.0f}; rank {candidate['rankScore']:.3f}"
        )
    if candidate["kind"] == "wait":
        months = candidate["months"] or 1
        return f"- [{candidate['index']}] wait {months} month(s); rank {candidate['rankScore']:.3f}"
    return f"- [{candidate['index']}] {candidate['kind']}; rank {candidate['rankScore']:.3f}"


def _parse_action(content: str) -> ActionParse:
    text = content.strip()
    if not text:
        return ActionParse(action_index=None, raw=content, error="empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return ActionParse(action_index=None, raw=content, error="response is not JSON")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return ActionParse(action_index=None, raw=content, error=f"invalid JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return ActionParse(action_index=None, raw=content, error="JSON must be an object")
    raw_index = payload.get("action_index", payload.get("actionIndex"))
    if isinstance(raw_index, bool) or raw_index is None:
        return ActionParse(action_index=None, raw=content, error="missing numeric action_index")
    try:
        action_index = int(raw_index)
    except (TypeError, ValueError):
        return ActionParse(action_index=None, raw=content, error="action_index must be an integer")
    return ActionParse(action_index=action_index, raw=content)


def _last_assistant_content(messages: vf.Messages) -> str:
    for message in reversed(messages):
        role = _message_value(message, "role")
        if role == "assistant":
            return _content_text(_message_value(message, "content"))
    return ""


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _visible_action_indices(state: Any, candidate_limit: int) -> list[int]:
    action_mask = np.asarray(jax.device_get(state.action_mask), dtype=bool)
    indices = [idx for idx, enabled in enumerate(action_mask[:MAX_CANDIDATES]) if enabled]
    return indices[:candidate_limit]


def _metrics_from_state(state: Any) -> dict[str, float]:
    state = jax.device_get(state)
    metrics = np.asarray(state.metrics)
    return {
        "score": float(metrics[0]),
        "cargo_delivered": float(metrics[1]),
        "operating_profit": float(metrics[2]),
        "route_count": float(metrics[4]),
        "invalid_actions": float(metrics[6]),
    }


def _step_feedback(result: StepResult) -> str:
    return (
        f"Executed action_index {result.action_index}. Reward {result.reward:.3f}. "
        f"Score {result.metrics['score']:.3f}; cargo delivered {result.metrics['cargo_delivered']:.1f}; "
        f"operating profit ${result.metrics['operating_profit']:,.0f}."
    )


def _final_feedback(state: vf.State) -> str:
    metrics = state.get("tycoon_final_metrics", {})
    return (
        f"Final score {float(metrics.get('score', 0.0)):.3f}; "
        f"cargo delivered {float(metrics.get('cargo_delivered', 0.0)):.1f}; "
        f"operating profit ${float(metrics.get('operating_profit', 0.0)):,.0f}; "
        f"executed steps {len(state.get('tycoon_steps', []))}."
    )


def _cargo_amounts(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for idx, value in enumerate(values):
        amount = float(value)
        if amount > 0.0:
            result[CARGO_NAMES[idx]] = round(amount, 1)
    return result


def _node_label(nodes: list[dict[str, Any]], node_id: int) -> str:
    for node in nodes:
        if node["id"] == node_id:
            return node["label"]
    return f"node_{node_id}"


def _diagnostics(flags: np.ndarray) -> list[str]:
    return [name for name, enabled in zip(DIAGNOSTIC_NAMES, flags, strict=False) if bool(enabled)]


def _format_mapping(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in mapping.items())


def _parse_families(families: str | list[str] | None) -> list[str] | None:
    if families is None:
        return None
    if isinstance(families, str):
        parsed = [item.strip() for item in families.split(",") if item.strip()]
    else:
        parsed = families
    unknown = sorted(set(parsed) - set(FAMILY_NAMES))
    if unknown:
        raise ValueError(f"unknown TycoonLE families: {', '.join(unknown)}")
    return parsed or None
