from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

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
    MODE_NAMES,
    NODE_KIND_NAMES,
    TERRAIN_NAMES,
)

from tinker_harness.schemas import response_contract_text

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
class RenderOptions:
    candidate_limit: int = 12


def world_from_state(state: Any, *, seed: int, split: str, family: str, options: RenderOptions) -> dict[str, Any]:
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
        produces = _cargo_amounts(np.asarray(state.node_produces[idx]))
        accepts = _cargo_amounts(np.asarray(state.node_accepts[idx]))
        nodes.append(
            {
                "id": idx,
                "label": f"{NODE_KIND_NAMES[int(state.node_kind[idx])]}_{idx}",
                "kind": NODE_KIND_NAMES[int(state.node_kind[idx])],
                "x": int(state.node_x[idx]),
                "y": int(state.node_y[idx]),
                "population": round(float(state.node_population[idx]), 1),
                "produces": produces,
                "accepts": accepts,
            }
        )

    candidates = []
    for idx in range(len(state.candidate.kind)):
        kind = int(state.candidate.kind[idx])
        if kind == ACTION_INVALID or not bool(state.candidate.feasible[idx]):
            continue
        candidate = {
            "index": idx,
            "kind": ACTION_NAMES[kind],
            "source": int(state.candidate.source[idx]),
            "destination": int(state.candidate.destination[idx]),
            "sourceLabel": _node_label(nodes, int(state.candidate.source[idx])),
            "destinationLabel": _node_label(nodes, int(state.candidate.destination[idx])),
            "cargo": CARGO_NAMES[int(state.candidate.cargo[idx])],
            "cargoLabel": CARGO_LABELS[int(state.candidate.cargo[idx])],
            "mode": MODE_NAMES[int(state.candidate.mode[idx])],
            "vehicles": int(state.candidate.vehicles[idx]),
            "months": int(state.candidate.months[idx]),
            "amount": round(float(state.candidate.amount[idx]), 2),
            "totalCost": round(float(state.candidate.total_cost[idx]), 2),
            "monthlyProfit": round(float(state.candidate.monthly_profit[idx]), 2),
            "monthlyDelivered": round(float(state.candidate.monthly_delivered[idx]), 2),
            "rankScore": round(float(state.candidate.rank_score[idx]), 5),
            "requiresLoan": round(float(state.candidate.requires_loan[idx]), 2),
            "directlyExecutable": bool(state.candidate.directly_executable[idx]),
            "terrainCost": round(float(state.candidate.terrain_cost[idx]), 3),
            "congestion": round(float(state.candidate.congestion[idx]), 3),
            "diagnostics": _diagnostics(np.asarray(state.candidate.diagnostics[idx], dtype=bool)),
        }
        candidates.append(candidate)
        if len(candidates) >= options.candidate_limit:
            break

    objective_cargo = int(state.objective_cargo)
    return {
        "split": split,
        "family": family,
        "familyId": int(state.family),
        "seed": seed,
        "scenarioSeed": int(state.seed),
        "width": int(state.width),
        "height": int(state.height),
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
        "maxDebtRatio": round(float(state.max_debt_ratio), 3),
        "terrain": terrain_counts,
        "nodes": nodes,
        "candidates": candidates,
    }


def render_user_prompt(world: dict[str, Any]) -> str:
    lines = [
        "You are operating a transport company inside a reinforcement-learning logistics simulator.",
        "Choose the strongest near-term operating plan from the valid candidate actions.",
        "",
        "World:",
        f"- family: {world['family']}",
        f"- seed: {world['seed']} / scenario seed: {world['scenarioSeed']}",
        f"- map: {world['width']}x{world['height']}, terrain: {_format_mapping(world['terrain'])}",
        f"- cash: ${world['cash']:,.0f}, loan: ${world['loan']:,.0f}, max loan: ${world['maxLoan']:,.0f}, interest rate: {world['interestRate']:.3f}",
        f"- horizon: {world['maxSteps']} steps / {world['maxMonths']} months",
        f"- objective cargo: {world['objectiveCargo']}, delivered target: {world['deliveredTarget']:,.0f}, profit target: ${world['profitTarget']:,.0f}, route target: {world['routeTarget']}",
        "",
        "Nodes:",
    ]
    for node in world["nodes"]:
        lines.append(
            f"- {node['label']} at ({node['x']},{node['y']}), population {node['population']:.0f}, "
            f"produces [{_format_mapping(node['produces'])}], accepts [{_format_mapping(node['accepts'])}]"
        )

    lines.extend(["", "Valid action candidates:"])
    for candidate in world["candidates"]:
        lines.append(_format_candidate(candidate))

    lines.extend(["", response_contract_text()])
    return "\n".join(lines)


def _format_candidate(candidate: dict[str, Any]) -> str:
    if candidate["kind"] == "build_route":
        return (
            f"- [{candidate['index']}] build_route {candidate['mode']} {candidate['cargo']} "
            f"{candidate['sourceLabel']} -> {candidate['destinationLabel']}; "
            f"vehicles {candidate['vehicles']}; cost ${candidate['totalCost']:,.0f}; "
            f"requires loan ${candidate['requiresLoan']:,.0f}; monthly profit ${candidate['monthlyProfit']:,.0f}; "
            f"monthly delivered {candidate['monthlyDelivered']:,.1f}; terrain cost {candidate['terrainCost']:.2f}; "
            f"congestion {candidate['congestion']:.2f}; rank {candidate['rankScore']:.3f}; "
            f"diagnostics [{', '.join(candidate['diagnostics']) or 'none'}]"
        )
    if candidate["kind"] == "take_loan":
        return f"- [{candidate['index']}] take_loan amount ${candidate['amount']:,.0f}; rank {candidate['rankScore']:.3f}"
    if candidate["kind"] == "repay_loan":
        return f"- [{candidate['index']}] repay_loan amount ${candidate['amount']:,.0f}; rank {candidate['rankScore']:.3f}"
    if candidate["kind"] == "wait":
        months = candidate["months"] or 1
        return f"- [{candidate['index']}] wait {months} month(s); rank {candidate['rankScore']:.3f}"
    return f"- [{candidate['index']}] {candidate['kind']}; rank {candidate['rankScore']:.3f}"


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
