"""The agentic core — Claude with tool use, deciding and adapting its own probes.

This is the ONLY place the model gets to drive (ADR 0001). The loop:

    1. Give Claude the strategy mission as the system prompt and the `search`/`ask` tools.
    2. Let it call a tool; execute the call via RedteamMCPClient; feed the result back.
    3. After each `ask`/`search`, run the strategy's oracle on the result to decide pass/fail.
    4. Record a Probe (the agent's intent, the tool calls, the oracle verdict).
    5. Check the Governor; stop on budget / stall / mission success.

The model NEVER computes the verdict — the oracle does. The governor bounds the loop. Everything
the loop produces is a list[Probe] for report.py to turn into a findings report + candidate gold.

STATUS: stub. The control flow and types are sketched; the Anthropic tool-use loop is the TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .client import RedteamMCPClient
from .governor import Governor
from .oracle import Verdict
from .strategies import Strategy


@dataclass
class Probe:
    """One adversarial attempt and its verdict."""

    intent: str                              # the agent's stated goal for this probe
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {tool, args, result_summary}
    verdict: Verdict | None = None           # filled by the oracle
    transcript: list[dict[str, Any]] = field(default_factory=list)  # raw messages, for the report


# Anthropic tool schemas mirroring the MCP tools, so Claude can call them by name.
# (Kept here, not fetched from the server, so the agent's view is explicit and reviewable.)
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": "Return ranked retrieval chunks (with scores) for a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {"type": "string", "enum": ["page", "wiki"]},
                "k": {"type": "integer"},
                "current_episode": {"type": "integer"},
                "current_page": {"type": "integer"},
            },
            "required": ["query", "mode"],
        },
    },
    {
        "name": "ask",
        "description": "Return the real, full-pipeline answer a reader would get.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "mode": {"type": "string", "enum": ["page", "wiki"]},
                "episode_slug": {"type": "string"},
                "current_page": {"type": "integer"},
            },
            "required": ["question", "mode"],
        },
    },
]


async def run_strategy(
    *,
    strategy: Strategy,
    client: RedteamMCPClient,
    governor: Governor,
    episode: int,
    page: int,
) -> list[Probe]:
    """Drive one strategy to completion (or budget) and return the probes it produced.

    TODO (build order step 4):
      - Open an Anthropic client (config.agent_model) with `system=strategy.mission` and
        `tools=TOOL_SCHEMAS`.
      - Loop while governor.should_continue():
          * send the running message list; if the model emits a tool_use, dispatch it through
            `client.search` / `client.ask`, governor.charge(tool), append the tool_result;
          * after a relevant call, run the matching oracle (oracle.spoiler_leaked /
            oracle.judge_hallucination) and attach a Verdict to the current Probe;
          * governor.tick(found_failure=verdict.failed).
      - Return the list[Probe].
    Keep multi-turn social-engineering scoped to a single probe's session (DESIGN §9).
    """
    raise NotImplementedError("agent loop — see DESIGN §8 build order, step 4")
