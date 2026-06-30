from __future__ import annotations

from typing import Any

from hud import Environment
from hud.capabilities import Capability

from tycoonle_hud.mcp_server import MCP_TOOL_NAME, get_session, start_mcp_server, stop_mcp_server
from tycoonle_hud.prompts import build_task_prompt
from tycoonle_hud.scoring import grade_session


env = Environment(name="tycoonle-hud", version="0.1.0")


@env.initialize
async def _start_tools() -> None:
    url = await start_mcp_server()
    env.add_capability(Capability.mcp(name=MCP_TOOL_NAME, url=url))


@env.shutdown
async def _stop_tools() -> None:
    await stop_mcp_server()


@env.template(id="plan", description="Plan and execute a TycoonLE logistics rollout with MCP tools.")
async def tycoonle_plan(
    split: str = "dev",
    family: str = "chain",
    seed: int = 20_000,
    candidate_limit: int = 12,
    action_budget: int = 6,
    show_rank_score: bool = True,
) -> Any:
    session = get_session()
    session.reset(
        split=split,
        family=family,
        seed=seed,
        candidate_limit=candidate_limit,
        action_budget=action_budget,
        show_rank_score=show_rank_score,
    )
    answer = yield build_task_prompt(session.initial_prompt_state())
    session.record_final_answer(answer)
    yield grade_session(session.snapshot())
