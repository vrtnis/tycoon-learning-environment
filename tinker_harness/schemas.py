from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a benchmark planning agent for a reinforcement-learning logistics environment. "
    "Given a simulated world state and valid action candidates, choose a concrete operating plan. "
    "Return JSON only. Do not include Markdown fences or reasoning text."
)

PLAN_SHAPE = {
    "summary": "one sentence",
    "firstAction": {
        "type": "build_route | take_loan | add_vehicle | wait",
        "mode": "road | rail",
        "cargo": "cargo name",
        "reason": "why this action comes first",
    },
    "financing": {
        "takeLoan": False,
        "reserveCash": 0,
        "repaymentTrigger": "when repayment starts",
    },
    "plan": [
        {"step": 1, "action": "specific action", "reason": "short operational reason"},
        {"step": 2, "action": "specific action", "reason": "short operational reason"},
    ],
    "riskControls": ["risk and mitigation"],
    "expectedOutcome": {
        "profitDirection": "up | flat | down",
        "mainBottleneck": "bottleneck to monitor",
    },
}

ACTION_TYPES = {"build_route", "take_loan", "add_vehicle", "wait"}
MODES = {"road", "rail"}
PROFIT_DIRECTIONS = {"up", "flat", "down"}


def response_contract_text() -> str:
    return "Return only valid JSON with this shape:\n" + json.dumps(PLAN_SHAPE, indent=2)


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return ["example.messages must contain system, user, and assistant messages"]

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles[:3] != ["system", "user", "assistant"]:
        errors.append("first three message roles must be system, user, assistant")

    assistant = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if not isinstance(assistant, str):
        errors.append("assistant content must be a string")
        return errors

    try:
        plan = json.loads(assistant)
    except json.JSONDecodeError as exc:
        errors.append(f"assistant content is not valid JSON: {exc}")
        return errors

    errors.extend(validate_plan(plan))
    return errors


def validate_plan(plan: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be an object"]

    if not _nonempty_string(plan.get("summary")):
        errors.append("summary must be a non-empty string")

    first_action = plan.get("firstAction")
    if not isinstance(first_action, dict):
        errors.append("firstAction must be an object")
    else:
        if first_action.get("type") not in ACTION_TYPES:
            errors.append("firstAction.type is unsupported")
        if first_action.get("mode") not in MODES:
            errors.append("firstAction.mode is unsupported")
        if not _nonempty_string(first_action.get("cargo")):
            errors.append("firstAction.cargo must be a non-empty string")
        if not _nonempty_string(first_action.get("reason")):
            errors.append("firstAction.reason must be a non-empty string")

    financing = plan.get("financing")
    if not isinstance(financing, dict):
        errors.append("financing must be an object")
    else:
        if not isinstance(financing.get("takeLoan"), bool):
            errors.append("financing.takeLoan must be boolean")
        if not isinstance(financing.get("reserveCash"), int | float):
            errors.append("financing.reserveCash must be numeric")
        if not _nonempty_string(financing.get("repaymentTrigger")):
            errors.append("financing.repaymentTrigger must be a non-empty string")

    steps = plan.get("plan")
    if not isinstance(steps, list) or len(steps) < 2:
        errors.append("plan must contain at least two steps")
    else:
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                errors.append(f"plan[{idx}] must be an object")
                continue
            if not isinstance(step.get("step"), int):
                errors.append(f"plan[{idx}].step must be an integer")
            if not _nonempty_string(step.get("action")):
                errors.append(f"plan[{idx}].action must be a non-empty string")
            if not _nonempty_string(step.get("reason")):
                errors.append(f"plan[{idx}].reason must be a non-empty string")

    risk_controls = plan.get("riskControls")
    if not isinstance(risk_controls, list) or not risk_controls or not all(isinstance(item, str) and item for item in risk_controls):
        errors.append("riskControls must be a non-empty string list")

    expected = plan.get("expectedOutcome")
    if not isinstance(expected, dict):
        errors.append("expectedOutcome must be an object")
    else:
        if expected.get("profitDirection") not in PROFIT_DIRECTIONS:
            errors.append("expectedOutcome.profitDirection is unsupported")
        if not _nonempty_string(expected.get("mainBottleneck")):
            errors.append("expectedOutcome.mainBottleneck must be a non-empty string")

    return errors


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
