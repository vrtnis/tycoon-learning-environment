from __future__ import annotations

from typing import Any


def build_task_prompt(state: dict[str, Any]) -> str:
    obs = state["observation"]
    objective = obs["world"]["objective"]
    metrics = obs["metrics"]
    company = obs["company"]
    lines = [
        "You are controlling a transport company in TycoonLE.",
        "Use the tycoonle MCP tools to inspect the world and execute candidate actions.",
        "",
        "Tool contract:",
        "- observe_world(detail='summary') returns the current world, metrics, nodes, routes, and company state.",
        "- list_actions() returns executable candidate actions. Only use actionIndex values from this list.",
        "- step(action_index, reason) executes exactly one action.",
        "- finish(summary) ends the rollout when you are done or when the action budget is exhausted.",
        "",
        "Rules:",
        "- Do not invent routes or action indices.",
        "- Optimize final TycoonLE score, cargo delivered, operating profit, and debt discipline.",
        "- Prefer short operational reasons for tool calls.",
        "- After using finish(), return a concise JSON summary of what you did.",
        "",
        "Scenario:",
        f"- split: {state['split']}",
        f"- family: {state['family']}",
        f"- seed: {state['seed']}",
        f"- action budget: {state['actionBudget']}",
        f"- visible action limit: {state['candidateLimit']}",
        f"- rankScore visible: {state['showRankScore']}",
        "",
        "Objective:",
        f"- {objective['label']}",
        f"- cargo: {objective['cargo'] or 'all'}",
        f"- delivered target: {objective['deliveredTarget']:.1f}",
        f"- profit target: ${objective['profitTarget']:,.0f}",
        f"- route target: {objective['routeTarget']}",
        f"- max debt ratio: {objective['maxDebtRatio']:.3f}",
        "",
        "Starting state:",
        f"- cash: ${company['cash']:,.0f}",
        f"- loan: ${company['loan']:,.0f}",
        f"- score: {metrics['score']:.3f}",
        f"- cargo delivered: {metrics['cargoDelivered']:.1f}",
        f"- operating profit: ${metrics['operatingProfit']:,.0f}",
        "",
        "Visible executable actions:",
    ]
    for action in state["visibleActions"]:
        lines.append(_format_action(action))
    return "\n".join(lines)


def _format_action(action: dict[str, Any]) -> str:
    estimates = action["estimates"]
    rank = f"; rank {action['rankScore']:.3f}" if "rankScore" in action else ""
    return (
        f"- [{action['actionIndex']}] {action['description']} "
        f"cost ${estimates['totalCost']:,.0f}; monthly profit ${estimates['monthlyProfit']:,.0f}; "
        f"monthly delivered {estimates['monthlyDelivered']:.1f}; loan needed ${action['requiresLoan']:,.0f}; "
        f"terrain {estimates['terrainCost']:.2f}; congestion {estimates['congestion']:.2f}"
        f"{rank}; diagnostics [{', '.join(action['diagnostics']) or 'none'}]"
    )
