from __future__ import annotations

import asyncio
import os
import socket
from contextlib import closing
from typing import Any

from fastmcp import FastMCP

from tycoonle_hud.session import TycoonLESession


MCP_TOOL_NAME = "tycoonle"
_HOST = "127.0.0.1"
_server = FastMCP(name=MCP_TOOL_NAME)
_session = TycoonLESession()
_server_task: asyncio.Task[Any] | None = None
_server_url: str | None = None


def get_session() -> TycoonLESession:
    return _session


@_server.tool
def observe_world(detail: str = "summary") -> dict[str, Any]:
    """Return the current TycoonLE world. Use detail='full' only when terrain grid detail is needed."""
    return _session.observe(detail=detail)


@_server.tool
def list_actions(limit: int | None = None) -> list[dict[str, Any]]:
    """Return visible executable candidate actions. Use actionIndex values with step()."""
    return _session.list_actions(limit=limit)


@_server.tool
def step(action_index: int, reason: str = "") -> dict[str, Any]:
    """Execute one candidate action by actionIndex and return the resulting observation."""
    return _session.step(action_index=action_index, reason=reason)


@_server.tool
def finish(summary: str = "") -> dict[str, Any]:
    """Mark the rollout finished and return final TycoonLE metrics."""
    return _session.finish(summary=summary)


async def start_mcp_server() -> str:
    global _server_task, _server_url
    if _server_task is not None and not _server_task.done() and _server_url is not None:
        return _server_url

    port = int(os.environ.get("TYCOONLE_HUD_MCP_PORT") or _free_port())
    _server_url = f"http://{_HOST}:{port}/mcp"
    _server_task = asyncio.create_task(_server.run_async(transport="http", host=_HOST, port=port))
    await _wait_for_port(_HOST, port)
    return _server_url


async def stop_mcp_server() -> None:
    global _server_task, _server_url
    if _server_task is not None:
        _server_task.cancel()
        try:
            await _server_task
        except asyncio.CancelledError:
            pass
    _server_task = None
    _server_url = None


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


async def _wait_for_port(host: str, port: int, timeout_seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: Exception | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for MCP server on {host}:{port}") from last_error
