"""MCP client wrapper — the red-teamer's line to the deployed app.

Identical in spirit to `pepper_carrot_eval.client.EvalMCPClient`: a thin async wrapper around a
FastMCP client exposing `search` and `ask`. The difference from the eval isn't here — it's *who
calls these methods*. In the eval, a script does, in fixed order. Here, the agent loop in
`agent.py` decides when and with what arguments (see ADR 0001).

NOTE: this is a near-verbatim port of the eval's client; keep the two in sync, or factor a shared
package later if drift becomes a problem.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

from fastmcp import Client


class RedteamMCPClient:
    """Async context manager around a FastMCP client (search / ask)."""

    def __init__(self, url: str) -> None:
        # FastMCP's client methods are largely unannotated; hold it as Any so mypy --strict
        # stays honest about everything we layer on top.
        self._client: Any = Client(url)

    async def __aenter__(self) -> RedteamMCPClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    async def search(
        self,
        *,
        query: str,
        mode: str,
        k: int = 3,
        current_episode: int | None = None,
        current_page: int | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"query": query, "mode": mode, "k": k}
        if current_episode is not None:
            args["current_episode"] = current_episode
        if current_page is not None:
            args["current_page"] = current_page
        result = await self._client.call_tool("search", args)
        content = result.structured_content
        return content if isinstance(content, dict) else {}

    async def ask(
        self,
        *,
        question: str,
        mode: str,
        episode_slug: str | None = None,
        current_page: int | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"question": question, "mode": mode}
        if episode_slug is not None:
            args["episode_slug"] = episode_slug
        if current_page is not None:
            args["current_page"] = current_page
        result = await self._client.call_tool("ask", args)
        content = result.structured_content
        return content if isinstance(content, dict) else {}
