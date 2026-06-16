from __future__ import annotations

import json
from typing import Any


def build_assistant_content(world: dict[str, Any]) -> str:
    plan = build_plan(world)
    return json.dumps(plan, indent=2)


def build_plan(world: dict[str, Any]) -> dict[str, Any]:
    route = _select_route(world)
    loan = _select_kind(world, "take_loan")
    wait = _select_kind(world, "wait")
    needs_loan = bool(route and route["requiresLoan"] > 0)
    first_kind = "take_loan" if needs_loan and loan else "build_route"
    first_route = route or _fallback_route(world)

    if first_kind == "take_loan":
        first_reason = (
            f"Borrow enough to unlock the {first_route['mode']} {first_route['cargo']} route while preserving runway."
        )
    else:
        first_reason = (
            f"Start with the highest-ranked positive-return {first_route['mode']} route that directly matches demand."
        )

    reserve_cash = _reserve_cash(world, first_route, needs_loan)
    family = world["family"]
    plan_steps = _family_steps(family, first_kind, first_route, loan, wait)
    risk_controls = _risk_controls(family, first_route, reserve_cash)

    return {
        "summary": _summary(family, first_route, needs_loan),
        "firstAction": {
            "type": first_kind,
            "mode": first_route["mode"],
            "cargo": first_route["cargo"],
            "reason": first_reason,
        },
        "financing": {
            "takeLoan": needs_loan,
            "reserveCash": reserve_cash,
            "repaymentTrigger": "after first deliveries create stable positive cash flow and reserve cash remains above target",
        },
        "plan": plan_steps,
        "riskControls": risk_controls,
        "expectedOutcome": {
            "profitDirection": "up" if first_route["monthlyProfit"] > 0 else "flat",
            "mainBottleneck": _bottleneck(family, first_route),
        },
    }


def _select_route(world: dict[str, Any]) -> dict[str, Any] | None:
    routes = [candidate for candidate in world["candidates"] if candidate["kind"] == "build_route"]
    if not routes:
        return None

    family = world["family"]
    if family == "chain":
        upstream = [
            candidate
            for candidate in routes
            if "processor" in candidate["destinationLabel"] and "missing_upstream" not in candidate["diagnostics"]
        ]
        if upstream:
            return max(upstream, key=_route_score)
    if family == "terrain_gap":
        rail = [candidate for candidate in routes if candidate["mode"] == "rail"]
        road = [candidate for candidate in routes if candidate["mode"] == "road"]
        if rail and road:
            best_rail = max(rail, key=_route_score)
            best_road = max(road, key=_route_score)
            if best_rail["rankScore"] >= best_road["rankScore"] - 0.02:
                return best_rail

    return max(routes, key=_route_score)


def _fallback_route(world: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "build_route",
        "mode": "road",
        "cargo": world.get("objectiveCargo", "cargo"),
        "sourceLabel": "source",
        "destinationLabel": "destination",
        "vehicles": 1,
        "requiresLoan": 0,
        "monthlyProfit": 0,
        "monthlyDelivered": 0,
        "terrainCost": 1,
        "congestion": 0,
        "rankScore": 0,
        "diagnostics": [],
    }


def _select_kind(world: dict[str, Any], kind: str) -> dict[str, Any] | None:
    matches = [candidate for candidate in world["candidates"] if candidate["kind"] == kind]
    return matches[0] if matches else None


def _route_score(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (float(candidate["rankScore"]), float(candidate["monthlyProfit"]), -float(candidate["requiresLoan"]))


def _reserve_cash(world: dict[str, Any], route: dict[str, Any], needs_loan: bool) -> int:
    base = 18_000 if world["family"] == "low_cash" else 28_000
    terrain_buffer = 8_000 if route["terrainCost"] > 1.35 or route["congestion"] > 0.5 else 0
    loan_buffer = 10_000 if needs_loan else 0
    return int(base + terrain_buffer + loan_buffer)


def _summary(family: str, route: dict[str, Any], needs_loan: bool) -> str:
    financing = "with a loan-backed runway" if needs_loan else "while preserving cash reserves"
    if family == "chain":
        return f"Build the upstream {route['cargo']} link first, then sequence the downstream processor route {financing}."
    if family == "terrain_gap":
        return f"Use {route['mode']} for the terrain-heavy opening route and protect cash until the payoff arrives."
    if family == "mixed_network":
        return f"Open the strongest {route['cargo']} route first, then diversify capacity across the network."
    if family == "low_cash":
        return f"Bootstrap the {route['cargo']} route {financing} before adding capacity."
    return f"Open the highest-return {route['cargo']} route {financing}."


def _family_steps(
    family: str,
    first_kind: str,
    route: dict[str, Any],
    loan: dict[str, Any] | None,
    wait: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if first_kind == "take_loan" and loan:
        steps.append(
            {
                "step": 1,
                "action": f"take loan of about ${loan['amount']:,.0f}",
                "reason": "loan timing unlocks the preferred route without starving operating cash",
            }
        )
        next_step = 2
    else:
        next_step = 1

    steps.append(
        {
            "step": next_step,
            "action": (
                f"build {route['mode']} route moving {route['cargo']} from {route['sourceLabel']} "
                f"to {route['destinationLabel']} with {route['vehicles']} vehicle(s)"
            ),
            "reason": "route rank, demand match, capacity, and expected ROI are strongest among candidates",
        }
    )
    next_step += 1

    if family == "chain":
        steps.append(
            {
                "step": next_step,
                "action": "wait for upstream deliveries, then build the downstream processor-output route",
                "reason": "sequencing upstream input before downstream output avoids a missing-upstream bottleneck",
            }
        )
    elif family == "mixed_network":
        steps.append(
            {
                "step": next_step,
                "action": "reserve capital for the second complementary route instead of overbuilding one corridor",
                "reason": "portfolio allocation captures network effects and town-growth demand without overbuild risk",
            }
        )
    elif family == "terrain_gap":
        steps.append(
            {
                "step": next_step,
                "action": "monitor terrain drag and add vehicles only after reliability and utilization are stable",
                "reason": "terrain cost and congestion can erase long-term payoff if capacity is added too early",
            }
        )
    elif wait:
        steps.append(
            {
                "step": next_step,
                "action": f"wait {max(1, wait['months'])} month(s) after opening before expanding",
                "reason": "first revenue confirms cash flow before additional capital is committed",
            }
        )
    else:
        steps.append(
            {
                "step": next_step,
                "action": "add capacity only when utilization is high and reserve cash remains above target",
                "reason": "vehicle sizing should follow observed demand and runway constraints",
            }
        )
    return steps


def _risk_controls(family: str, route: dict[str, Any], reserve_cash: int) -> list[str]:
    controls = [
        f"keep at least ${reserve_cash:,.0f} reserve cash for runway and maintenance",
        "delay extra vehicles until demand, utilization, and cash flow confirm the route ROI",
    ]
    if route["congestion"] > 0.4:
        controls.append("watch congestion and avoid crossing or sharing too many existing route tiles")
    if route["terrainCost"] > 1.35 or family == "terrain_gap":
        controls.append("compare rail and road terrain cost before expanding across hills or water")
    if family == "chain":
        controls.append("build upstream input before downstream output to prevent processor starvation")
    if family == "low_cash":
        controls.append("repay debt only after first delivery revenue and cash runway are secure")
    if family == "mixed_network":
        controls.append("avoid overbuilding one route before the portfolio split proves demand growth")
    return controls


def _bottleneck(family: str, route: dict[str, Any]) -> str:
    if family == "chain":
        return "upstream input supply and downstream processor capacity balance"
    if family == "terrain_gap":
        return "terrain cost, maintenance drag, and early negative cash period"
    if family == "mixed_network":
        return "capital allocation between high-margin routes and town-growth routes"
    if family == "low_cash":
        return "cash runway before first delivery and debt repayment timing"
    if route["monthlyDelivered"] <= 0:
        return "first delivery timing"
    return "vehicle capacity and sustained accepted demand"
