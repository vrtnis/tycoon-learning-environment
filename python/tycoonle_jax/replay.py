from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tycoonle_jax.candidates import make_observation
from tycoonle_jax.constants import (
    ACTION_ADD_VEHICLE,
    ACTION_BUILD_ROUTE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    ACTION_WAIT,
    CARGO_LABELS,
    CARGO_NAMES,
    FAMILY_NAMES,
    MAX_CANDIDATES,
    MAX_PATH,
    MODE_NAMES,
    NODE_KIND_NAMES,
    SPLIT_NAMES,
    TERRAIN_NAMES,
)
from tycoonle_jax.env import TycoonLE
from tycoonle_jax.types import State


def rollout_first_valid(env: TycoonLE, key: jax.Array, num_steps: int = 16) -> dict[str, Any]:
    state, first_timestep = env.reset(key)

    def body(carry: State, _: jnp.ndarray):
        action = jnp.argmax(carry.action_mask.astype(jnp.int32)).astype(jnp.int32)
        next_state, timestep = env.step(carry, action)
        return next_state, (carry, next_state, action, timestep)

    final_state, scanned = jax.lax.scan(body, state, jnp.arange(num_steps))
    before_states, after_states, actions, timesteps = scanned
    return {
        "initial_state": state,
        "initial_timestep": first_timestep,
        "before_states": before_states,
        "after_states": after_states,
        "actions": actions,
        "timesteps": timesteps,
        "final_state": final_state,
    }


def export_replay(rollout: dict[str, Any]) -> dict[str, Any]:
    before_states = _unstack_states(rollout["before_states"])
    after_states = _unstack_states(rollout["after_states"])
    actions = np.asarray(jax.device_get(rollout["actions"]))
    timesteps = jax.device_get(rollout["timesteps"])
    events: list[dict[str, Any]] = []
    for step, (before, after, action) in enumerate(zip(before_states, after_states, actions, strict=False), start=1):
        before_obs = decode_observation(before)
        after_obs = decode_observation(after)
        selected = decode_candidate_action(before, int(action))
        reward = float(np.asarray(timesteps.reward)[step - 1])
        info = {
            "selectedAction": selected,
            "executionTrace": build_execution_trace(before, after, selected),
            "rewardDetails": {
                "reward": round(reward, 4),
                "components": {
                    "scoreDelta": round(reward, 4),
                    "cargoDelta": round(float(after.metrics[1] - before.metrics[1]), 3),
                    "profitDelta": round(float(after.metrics[2] - before.metrics[2]), 3),
                    "cashDelta": round(float(after.cash - before.cash), 3),
                    "loanDelta": round(float(after.loan - before.loan), 3),
                    "invalidAction": 0 if bool(before.action_mask[int(action)]) else 1,
                },
                "milestones": _milestones(before, after),
                "diagnostics": decode_diagnostics(before.candidate.diagnostics[int(action)]),
            },
            "actionMask": [1 if value else 0 for value in np.asarray(before.action_mask).tolist()],
            "candidateActions": decode_candidates(before),
        }
        events.append(
            {
                "step": step,
                "before": before_obs,
                "action": selected,
                "after": after_obs,
                "reward": round(reward, 4),
                "info": info,
                "focus": build_action_focus(before_obs, selected, after_obs),
            }
        )
        if bool(after.done):
            break
    final_obs = events[-1]["after"] if events else decode_observation(rollout["initial_state"])
    return {
        "schema": "tycoonle-replay-v1",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worldId": final_obs["world"]["id"],
        "scenario": {"split": final_obs["world"]["split"], "family": final_obs["world"]["family"], "seed": final_obs["world"]["seed"]},
        "events": events,
        "summary": {
            "steps": final_obs["time"]["step"],
            "finalScore": final_obs["metrics"]["score"],
            "cargoDelivered": final_obs["metrics"]["cargoDelivered"],
            "operatingProfit": final_obs["metrics"]["operatingProfit"],
        },
    }


def decode_observation(state: State) -> dict[str, Any]:
    state = jax.device_get(state)
    terrain = [[TERRAIN_NAMES[int(tile)] for tile in row[: int(state.width)]] for row in np.asarray(state.terrain)[: int(state.height)]]
    nodes = decode_nodes(state)
    world = {
        "id": _world_id(state),
        "split": SPLIT_NAMES[int(state.split)],
        "family": FAMILY_NAMES[int(state.family)],
        "seed": int(state.seed),
        "width": int(state.width),
        "height": int(state.height),
        "terrain": terrain,
        "nodes": nodes,
        "budget": {
            "maxSteps": int(state.max_steps),
            "maxMonths": int(state.max_months),
            "startingCash": float(state.cash) if int(state.step) == 0 else None,
            "maxLoan": float(state.max_loan),
            "interestRate": float(state.interest_rate),
        },
        "objective": {
            "id": f"{FAMILY_NAMES[int(state.family)]}_objective",
            "label": _objective_label(state),
            "cargo": None if int(state.objective_cargo) < 0 else CARGO_NAMES[int(state.objective_cargo)],
            "deliveredTarget": float(state.delivered_target),
            "profitTarget": float(state.profit_target),
            "routeTarget": int(state.route_target),
            "maxDebtRatio": float(state.max_debt_ratio),
        },
    }
    return {
        "schema": "tycoonle-observation-v1",
        "world": world,
        "time": {"month": int(state.month), "step": int(state.step), "maxMonths": int(state.max_months), "maxSteps": int(state.max_steps)},
        "company": {"cash": round(float(state.cash), 2), "loan": round(float(state.loan), 2), "maxLoan": float(state.max_loan)},
        "nodes": nodes,
        "routes": decode_routes(state),
        "metrics": decode_metrics(state),
        "candidateActions": decode_candidates(state),
        "lastEvent": _last_event(state),
    }


def decode_nodes(state: State) -> list[dict[str, Any]]:
    nodes = []
    for idx, valid in enumerate(np.asarray(state.node_mask).tolist()):
        if not valid:
            continue
        produces = _cargo_map(state.node_produces[idx])
        accepts = _cargo_map(state.node_accepts[idx])
        storage = _cargo_map(state.node_storage[idx])
        converts = {}
        if int(state.node_convert_from[idx]) >= 0 and int(state.node_convert_to[idx]) >= 0:
            converts[CARGO_NAMES[int(state.node_convert_from[idx])]] = CARGO_NAMES[int(state.node_convert_to[idx])]
        nodes.append(
            {
                "id": _node_id(state, idx),
                "name": _node_name(state, idx),
                "kind": NODE_KIND_NAMES[int(state.node_kind[idx])],
                "x": int(state.node_x[idx]),
                "y": int(state.node_y[idx]),
                "produces": produces,
                "accepts": accepts,
                "converts": converts or None,
                "storage": storage,
                "population": int(state.node_population[idx]) if int(state.node_kind[idx]) == 3 else None,
                "baseProduction": _cargo_map(state.node_base_production[idx]),
                "productionIndex": round(float(state.node_production_index[idx]), 3),
                "serviceMonths": int(state.node_service_months[idx]),
                "lastServedMonth": None if int(state.node_last_served_month[idx]) < 0 else int(state.node_last_served_month[idx]),
                "rating": round(float(state.node_rating[idx]), 3),
            }
        )
    return nodes


def decode_routes(state: State) -> list[dict[str, Any]]:
    routes = []
    for idx, valid in enumerate(np.asarray(state.route_mask).tolist()):
        if not valid:
            continue
        path = [{"x": int(x), "y": int(y)} for x, y in np.asarray(state.route_path[idx])[: int(state.route_path_length[idx])]]
        routes.append(
            {
                "id": f"R{idx + 1:03d}",
                "sourceId": _node_id(state, int(state.route_source[idx])),
                "destinationId": _node_id(state, int(state.route_destination[idx])),
                "cargo": CARGO_NAMES[int(state.route_cargo[idx])],
                "mode": MODE_NAMES[int(state.route_mode[idx])],
                "distance": float(state.route_distance[idx]),
                "terrainCost": round(float(state.route_terrain_cost[idx]), 3),
                "buildCost": round(float(state.route_build_cost[idx]), 2),
                "vehicleCost": round(float(state.route_vehicle_cost[idx]), 2),
                "vehicles": int(state.route_vehicles[idx]),
                "delivered": round(float(state.route_delivered[idx]), 3),
                "revenue": round(float(state.route_revenue[idx]), 3),
                "operatingCost": round(float(state.route_operating_cost[idx]), 3),
                "profit": round(float(state.route_profit[idx]), 3),
                "utilization": round(float(state.route_utilization[idx]), 3),
                "ageMonths": int(state.route_age_months[idx]),
                "path": path,
                "travelTimeMonths": int(state.route_travel_time_months[idx]),
                "inTransit": [
                    {"cargo": CARGO_NAMES[int(state.route_cargo[idx])], "amount": round(float(amount), 3), "monthsRemaining": int(rem), "loadedMonth": 0}
                    for amount, rem in zip(np.asarray(state.route_transit_amount[idx]), np.asarray(state.route_transit_remaining[idx]), strict=False)
                    if amount > 0
                ],
                "lastDelivered": round(float(state.route_last_delivered[idx]), 3),
                "lastDelay": int(state.route_last_delay[idx]),
                "reliability": round(float(state.route_reliability[idx]), 3),
                "congestion": round(float(state.route_congestion[idx]), 3),
                "stationRating": round(float(state.route_station_rating[idx]), 3),
                "constraintFlags": [],
            }
        )
    return routes


def decode_metrics(state: State) -> dict[str, Any]:
    m = np.asarray(state.metrics)
    b = np.asarray(state.score_breakdown)
    return {
        "score": round(float(m[0]), 3),
        "cargoDelivered": round(float(m[1]), 3),
        "operatingProfit": round(float(m[2]), 3),
        "networkValue": round(float(m[3]), 3),
        "routeCount": int(m[4]),
        "vehicles": int(m[5]),
        "invalidActions": int(m[6]),
        "firstDeliveryMonth": None if int(m[7]) < 0 else int(m[7]),
        "debtRatio": round(float(m[8]), 3),
        "utilization": round(float(m[9]), 3),
        "inTransitCargo": round(float(m[10]), 3),
        "averageReliability": round(float(m[11]), 3),
        "congestion": round(float(m[12]), 3),
        "townGrowth": round(float(m[13]), 3),
        "productionIndex": round(float(m[14]), 3),
        "lateShipments": int(m[15]),
        "breakdown": {
            "cargo": round(float(b[0]), 3),
            "profit": round(float(b[1]), 3),
            "routes": round(float(b[2]), 3),
            "debt": round(float(b[3]), 3),
            "firstDelivery": round(float(b[4]), 3),
            "utilization": round(float(b[5]), 3),
            "reliability": round(float(b[6]), 3),
            "congestionPenalty": round(float(b[7]), 3),
            "invalidPenalty": round(float(b[8]), 3),
        },
    }


def decode_candidates(state: State) -> list[dict[str, Any]]:
    return [decode_candidate(state, idx) for idx in range(MAX_CANDIDATES) if int(state.candidate.kind[idx]) != 0]


def decode_candidate(state: State, idx: int) -> dict[str, Any]:
    kind = int(state.candidate.kind[idx])
    action = decode_candidate_action(state, idx)
    return {
        "id": _candidate_id(state, idx),
        "kind": action["type"],
        "action": action,
        "feasible": bool(state.candidate.feasible[idx]),
        "directlyExecutable": bool(state.candidate.directly_executable[idx]),
        "requiresLoan": round(float(state.candidate.requires_loan[idx]), 2),
        "rankScore": round(float(state.candidate.rank_score[idx]), 5),
        "estimates": {
            "totalCost": round(float(state.candidate.total_cost[idx]), 2),
            "monthlyProfit": round(float(state.candidate.monthly_profit[idx]), 2),
            "monthlyDelivered": round(float(state.candidate.monthly_delivered[idx]), 2),
            "terrainCost": round(float(state.candidate.terrain_cost[idx]), 3),
            "congestion": round(float(state.candidate.congestion[idx]), 3),
            "pathLength": int(state.candidate.path_length[idx]),
        },
        "diagnostics": decode_diagnostics(state.candidate.diagnostics[idx]),
        "description": _candidate_description(state, idx, kind),
    }


def decode_candidate_action(state: State, idx: int) -> dict[str, Any]:
    c = state.candidate
    kind = int(c.kind[idx])
    if kind == ACTION_BUILD_ROUTE:
        return {
            "type": "build_route",
            "sourceId": _node_id(state, int(c.source[idx])),
            "destinationId": _node_id(state, int(c.destination[idx])),
            "cargo": CARGO_NAMES[int(c.cargo[idx])],
            "mode": MODE_NAMES[int(c.mode[idx])],
            "vehicles": int(c.vehicles[idx]),
        }
    if kind == ACTION_ADD_VEHICLE:
        return {"type": "add_vehicle", "routeId": f"R{int(c.route[idx]) + 1:03d}", "count": int(c.vehicles[idx])}
    if kind == ACTION_WAIT:
        return {"type": "wait", "months": int(c.months[idx])}
    if kind == ACTION_TAKE_LOAN:
        return {"type": "take_loan", "amount": round(float(c.amount[idx]), 2)}
    if kind == ACTION_REPAY_LOAN:
        return {"type": "repay_loan", "amount": round(float(c.amount[idx]), 2)}
    return {"type": "invalid"}


def decode_diagnostics(bits: Any) -> list[str]:
    names = [
        "requires_financing",
        "insufficient_cash_even_with_loan",
        "route_build_constraints",
        "bridge_required",
        "town_permit_cost",
        "crossing_conflict",
        "shared_corridor",
        "requires_upstream_supply",
    ]
    return [name for name, active in zip(names, np.asarray(bits).tolist(), strict=False) if active]


def build_execution_trace(before: State, after: State, action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("type") != "build_route":
        return None
    route = next((item for item in decode_routes(after) if item["sourceId"] == action["sourceId"] and item["destinationId"] == action["destinationId"] and item["cargo"] == action["cargo"]), None)
    if route is None:
        return None
    cash = float(before.cash)
    path = route["path"]
    total_cost = max(0.0, float(before.cash - after.cash))
    segment_count = max(1, len(path) - 1)
    steps = []
    for index, tile in enumerate(path[1:], start=1):
        cost = total_cost * 0.12 / segment_count
        cash -= cost
        steps.append({"kind": "prepare_land", "label": f"Prepare land {index}/{segment_count}", "cost": cost, "cashAfter": round(cash, 2), "tile": tile, "index": index, "total": segment_count})
    for label, tile in (("Build source station", path[0]), ("Build destination station", path[-1])):
        cost = route["buildCost"] * 0.11
        cash -= cost
        steps.append({"kind": "build_station", "label": label, "cost": cost, "cashAfter": round(cash, 2), "tile": tile})
    for index, tile in enumerate(path[1:], start=1):
        cost = route["buildCost"] * 0.58 / segment_count
        cash -= cost
        steps.append({"kind": "lay_segment", "label": f"Lay {route['mode']} segment {index}/{segment_count}", "cost": cost, "cashAfter": round(cash, 2), "from": path[index - 1], "to": tile, "tile": tile, "index": index, "total": segment_count})
    steps.append({"kind": "buy_vehicle", "label": f"Buy {route['vehicles']} {route['mode']} vehicle", "cost": route["vehicleCost"] * route["vehicles"], "cashAfter": round(float(after.cash), 2), "count": route["vehicles"]})
    steps.append({"kind": "route_ready", "label": f"{route['id']} ready", "cost": 0, "cashAfter": round(float(after.cash), 2), "tile": path[-1]})
    return {
        "kind": "build_route",
        "routeId": route["id"],
        "mode": route["mode"],
        "cargo": route["cargo"],
        "sourceId": route["sourceId"],
        "destinationId": route["destinationId"],
        "path": path,
        "cashBefore": round(float(before.cash), 2),
        "cashAfter": round(float(after.cash), 2),
        "totalCost": round(total_cost, 2),
        "costBreakdown": {"landPrep": total_cost * 0.12, "stations": route["buildCost"] * 0.22, "track": route["buildCost"] * 0.58, "bridges": 0, "crossings": 0, "sharedCredit": 0, "vehicles": route["vehicleCost"] * route["vehicles"]},
        "steps": steps,
    }


def build_action_focus(before: dict[str, Any], action: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if action.get("type") == "build_route":
        source = _node_by_id(after["nodes"], action["sourceId"])
        destination = _node_by_id(after["nodes"], action["destinationId"])
        route = next((item for item in after["routes"] if item["sourceId"] == action["sourceId"] and item["destinationId"] == action["destinationId"] and not any(old["id"] == item["id"] for old in before["routes"])), None)
        return _route_focus("route", route["id"] if route else "Built route", route, source, destination, 1.2)
    if action.get("type") == "wait" and after["routes"]:
        route = sorted(after["routes"], key=lambda item: (item["delivered"], item["profit"]), reverse=True)[0]
        return _route_focus("network", f"Operating {route['id']}", route, _node_by_id(after["nodes"], route["sourceId"]), _node_by_id(after["nodes"], route["destinationId"]), 1.1)
    nodes = after["nodes"][:3] or [{"x": 0, "y": 0, "id": "origin", "kind": "origin"}]
    return _focus_from_tiles("world", after["world"]["id"], None, [{"x": node["x"], "y": node["y"], "role": node["kind"], "nodeId": node["id"]} for node in nodes], 1.0)


def _route_focus(kind: str, label: str, route: dict[str, Any] | None, source: dict[str, Any] | None, destination: dict[str, Any] | None, zoom: float) -> dict[str, Any]:
    if not source or not destination:
        return _focus_from_tiles(kind, label, route["id"] if route else None, [], zoom)
    return _focus_from_tiles(
        kind,
        label,
        route["id"] if route else None,
        [{"x": source["x"], "y": source["y"], "role": "source", "nodeId": source["id"]}, {"x": destination["x"], "y": destination["y"], "role": "destination", "nodeId": destination["id"]}],
        zoom,
    )


def _focus_from_tiles(kind: str, label: str, route_id: str | None, nodes: list[dict[str, Any]], zoom: float) -> dict[str, Any]:
    tiles = [{"x": item["x"], "y": item["y"], "role": item["role"]} for item in nodes] or [{"x": 0, "y": 0, "role": "origin"}]
    node_ids = [item["nodeId"] for item in nodes if item.get("nodeId")]
    xs = [tile["x"] for tile in tiles]
    ys = [tile["y"] for tile in tiles]
    bounds = {"minX": min(xs), "minY": min(ys), "maxX": max(xs), "maxY": max(ys)}
    return {
        "kind": kind,
        "label": label,
        "routeId": route_id,
        "nodeIds": node_ids,
        "tiles": tiles,
        "bounds": bounds,
        "camera": {"centerX": (bounds["minX"] + bounds["maxX"]) / 2, "centerY": (bounds["minY"] + bounds["maxY"]) / 2, "padding": 6, "zoom": zoom},
    }


def _unstack_states(batched: State) -> list[State]:
    leaves = jax.tree.leaves(batched)
    count = int(np.asarray(leaves[0]).shape[0])
    return [jax.tree.map(lambda x, i=i: x[i], batched) for i in range(count)]


def _cargo_map(values: Any) -> dict[str, float]:
    return {CARGO_NAMES[idx]: round(float(value), 3) for idx, value in enumerate(np.asarray(values).tolist()) if abs(float(value)) > 1e-6}


def _world_id(state: State) -> str:
    return f"tycoonle_{SPLIT_NAMES[int(state.split)]}_{FAMILY_NAMES[int(state.family)]}_{int(state.seed):04d}"


def _node_id(state: State, idx: int) -> str:
    family = FAMILY_NAMES[int(state.family)]
    if family == "chain":
        return ("raw_source", "processor", "sink", "town_1", "town_2", "node_5")[idx]
    if family == "mixed_network":
        return ("source_a", "dest_a", "source_b", "dest_b", "town_1", "town_2")[idx]
    return ("source_a", "dest_a", "town_1", "town_2", "node_4", "node_5")[idx]


def _node_name(state: State, idx: int) -> str:
    kind = NODE_KIND_NAMES[int(state.node_kind[idx])].replace("_", " ").title()
    produced = np.flatnonzero(np.asarray(state.node_produces[idx]) > 0)
    accepted = np.flatnonzero(np.asarray(state.node_accepts[idx]) > 0)
    cargo = produced[0] if len(produced) else accepted[0] if len(accepted) else None
    return f"{kind} {CARGO_LABELS[int(cargo)]}" if cargo is not None else f"{kind} {idx + 1}"


def _objective_label(state: State) -> str:
    family = FAMILY_NAMES[int(state.family)]
    if family == "mixed_network":
        return "Build a profitable two-route network across generated demand."
    cargo = CARGO_LABELS[int(state.objective_cargo)] if int(state.objective_cargo) >= 0 else "cargo"
    if family == "chain":
        return f"Build a two-stage chain and deliver {cargo}."
    if family == "low_cash":
        return f"Use financing carefully to bootstrap {cargo} delivery."
    if family == "terrain_gap":
        return f"Find a route across costly terrain and deliver {cargo}."
    return f"Build and operate the strongest {cargo} route."


def _last_event(state: State) -> str:
    if int(state.step) == 0:
        return f"{_world_id(state)} ready."
    return f"Step {int(state.step)} complete: score {float(state.metrics[0]):.1f}, cargo {float(state.metrics[1]):.1f}."


def _candidate_id(state: State, idx: int) -> str:
    action = decode_candidate_action(state, idx)
    if action["type"] == "build_route":
        return f"build:{action['sourceId']}:{action['destinationId']}:{action['cargo']}:{action['mode']}"
    if action["type"] == "add_vehicle":
        return f"add_vehicle:{action['routeId']}"
    if action["type"] == "wait":
        return f"wait:{action['months']}"
    return f"finance:{action['type']}"


def _candidate_description(state: State, idx: int, kind: int) -> str:
    action = decode_candidate_action(state, idx)
    if kind == ACTION_BUILD_ROUTE:
        return f"Build {action['mode']} {action['cargo']} route from {action['sourceId']} to {action['destinationId']}."
    if kind == ACTION_ADD_VEHICLE:
        return f"Add one vehicle to {action['routeId']}."
    if kind == ACTION_WAIT:
        return f"Wait {action['months']} month(s) for operations."
    if kind == ACTION_TAKE_LOAN:
        return "Take debt to unlock high-ranked actions."
    if kind == ACTION_REPAY_LOAN:
        return "Repay debt while retaining operating cash."
    return "Invalid padded action."


def _milestones(before: State, after: State) -> list[str]:
    result = []
    if int(after.metrics[4]) > int(before.metrics[4]):
        result.append("route_built")
    if int(after.metrics[5]) > int(before.metrics[5]):
        result.append("vehicles_added")
    if int(after.first_delivery_month) >= 0 and int(before.first_delivery_month) < 0:
        result.append("first_delivery")
    if float(after.metrics[1]) > float(before.metrics[1]):
        result.append("cargo_delivered")
    if float(after.metrics[2]) > float(before.metrics[2]):
        result.append("profit_improved")
    return result


def _node_by_id(nodes: list[dict[str, Any]], node_id: str) -> dict[str, Any] | None:
    return next((node for node in nodes if node["id"] == node_id), None)
