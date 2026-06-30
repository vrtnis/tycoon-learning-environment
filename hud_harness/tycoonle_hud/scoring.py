from __future__ import annotations

from typing import Any


def grade_session(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot["metrics"]
    objective = snapshot["objective"]
    steps = snapshot["steps"]
    valid_steps = sum(1 for step in steps if step.get("validAction"))
    attempted = len(steps) + int(snapshot.get("invalidToolCalls", 0))

    score_gain = _clamp((float(metrics["score"]) - 20.0) / 80.0)
    valid_action_rate = valid_steps / attempted if attempted else 0.0
    objective_progress = _objective_progress(metrics, objective)
    reward = _round4(0.70 * score_gain + 0.20 * valid_action_rate + 0.10 * objective_progress)

    subscores = {
        "score_gain": _round4(score_gain),
        "valid_action_rate": _round4(valid_action_rate),
        "objective_progress": _round4(objective_progress),
    }
    return {
        "score": reward,
        "content": (
            f"TycoonLE score {float(metrics['score']):.3f}; "
            f"valid actions {valid_steps}/{attempted}; "
            f"objective progress {objective_progress:.3f}; reward {reward:.3f}."
        ),
        "info": {
            "subscores": subscores,
            "metrics": metrics,
            "objective": objective,
            "steps": steps,
            "invalidToolCalls": snapshot.get("invalidToolCalls", 0),
            "finalSummary": snapshot.get("finalSummary", ""),
            "finalAnswer": snapshot.get("finalAnswer", ""),
        },
    }


def _objective_progress(metrics: dict[str, Any], objective: dict[str, Any]) -> float:
    delivered_target = max(1.0, float(objective.get("deliveredTarget") or 1.0))
    profit_target = max(1.0, float(objective.get("profitTarget") or 1.0))
    route_target = max(1.0, float(objective.get("routeTarget") or 1.0))
    cargo = _clamp(float(metrics["cargoDelivered"]) / delivered_target)
    profit = _clamp(max(0.0, float(metrics["operatingProfit"])) / profit_target)
    routes = _clamp(float(metrics["routeCount"]) / route_target)
    debt_ratio = float(metrics.get("debtRatio", 0.0))
    max_debt_ratio = max(0.01, float(objective.get("maxDebtRatio") or 1.0))
    debt = 1.0 - _clamp(max(0.0, debt_ratio - max_debt_ratio) / max_debt_ratio)
    return _clamp(0.35 * cargo + 0.30 * profit + 0.25 * routes + 0.10 * debt)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _round4(value: float) -> float:
    return round(float(value), 4)
