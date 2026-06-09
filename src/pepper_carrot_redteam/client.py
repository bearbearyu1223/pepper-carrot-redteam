"""MCP client wrapper — the red-teamer's line to the deployed app.

Identical in spirit to `pepper_carrot_eval.client.EvalMCPClient`: a thin async wrapper around a
FastMCP client exposing `search` and `ask`. The difference from the eval isn't here — it's *who
calls these methods*. In the eval, a script does, in fixed order. Here, the agent loop in
`agent.py` decides when and with what arguments (see ADR 0001).

The ONE real divergence from the eval's client: `ask` accepts and returns a `session_id`, so the
agent can run a multi-turn social-engineering conversation against a single server session (ADR
0002). The eval deliberately uses a fresh session per call to keep scoring clean; here, session
continuity *is* the attack, so the agent (via the harness) threads the id across turns. We also
add `episodes()` to resolve an episode number → slug, which the spoiler `ask` path needs.

NOTE: otherwise a near-verbatim port of the eval's client; keep the two in sync, or factor a
shared package later if drift becomes a problem.
"""

from __future__ import annotations

import json
import logging
import time
from types import TracebackType
from typing import Any

from fastmcp import Client

from . import tracing

logger = logging.getLogger(__name__)


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
        t0 = time.perf_counter()
        result = await self._client.call_tool("search", args)
        content = result.structured_content
        content = content if isinstance(content, dict) else {}
        chunks = content.get("chunks", [])
        latency = round(time.perf_counter() - t0, 3)
        logger.debug(
            "search(mode=%s k=%s ep=%s pg=%s) → %d chunks in %.2fs · %r",
            mode, k, current_episode, current_page, len(chunks), latency, query[:60],
        )
        tracing.record(
            component="search", latency_s=latency, input=args,
            output={
                "n_chunks": len(chunks),
                "chunks": [
                    {
                        "table": c.get("source_table"),
                        "ep": (c.get("metadata") or {}).get("episode_number"),
                        "pg": (c.get("metadata") or {}).get("page_number"),
                        "score": round(float(c.get("score", 0.0)), 3),
                    }
                    for c in chunks[:5]
                ],
            },
        )
        return content

    async def ask(
        self,
        *,
        question: str,
        mode: str,
        episode_slug: str | None = None,
        current_page: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ask the companion. Pass `session_id` to continue an existing conversation.

        The server returns the `session_id` it used (a fresh one if none was given), inside the
        structured content — the caller captures it to keep pressing the same conversation across
        turns. This is the lever the spoiler social-engineering and hallucination strategies use.
        """
        args: dict[str, Any] = {"question": question, "mode": mode}
        if episode_slug is not None:
            args["episode_slug"] = episode_slug
        if current_page is not None:
            args["current_page"] = current_page
        if session_id is not None:
            args["session_id"] = session_id
        t0 = time.perf_counter()
        result = await self._client.call_tool("ask", args)
        content = result.structured_content
        content = content if isinstance(content, dict) else {}
        answer = str(content.get("answer", ""))
        latency = round(time.perf_counter() - t0, 3)
        logger.debug(
            "ask(mode=%s ep=%s pg=%s session_in=%s → session_out=%s) → %d chars in %.2fs · %r",
            mode, episode_slug, current_page, session_id, content.get("session_id"),
            len(answer), latency, question[:60],
        )
        tracing.record(
            component="ask", latency_s=latency, input=args,
            output={
                "answer_chars": len(answer),
                "answer_head": answer[:300],
                "session_id": content.get("session_id"),
                "retrieved_doc_ids": content.get("retrieved_doc_ids", []),
            },
        )
        return content

    async def episodes(self) -> list[dict[str, Any]]:
        """Read the `episodes://catalog` resource (slug, number, page_count, …).

        The spoiler `ask` path needs an `episode_slug` to pin the reader's episode; gold positions
        are given as episode *numbers*, so the agent loop resolves number → slug through this.
        """
        contents = await self._client.read_resource("episodes://catalog")
        text = getattr(contents[0], "text", "{}") if contents else "{}"
        data = json.loads(text)
        episodes = data.get("episodes", []) if isinstance(data, dict) else []
        return episodes if isinstance(episodes, list) else []
