"""The load-bearing multi-turn invariant (ADR 0002), tested deterministically — no network, no
model nondeterminism. A live run *might* continue a session (it's the agent's choice); this pins
the mechanism: when the agent sets continue_session=true, the harness must thread the session_id
the server returned back into the next `ask`.

We script a fake LLM (turn 1: ask continue_session=false; turn 2: ask continue_session=true; turn
3: no tool → stop) over a fake MCP client that records the session_id each `ask` receives, with the
hallucination judge stubbed out.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest

from pepper_carrot_redteam import agent
from pepper_carrot_redteam.client import RedteamMCPClient
from pepper_carrot_redteam.governor import Governor
from pepper_carrot_redteam.oracle import MEDIUM, Verdict
from pepper_carrot_redteam.strategies import ALL

_RETURNED_SESSION = "S1"


class _FakeClient:
    """Records the session_id passed to each `ask`; returns a canned answer + a fixed session."""

    def __init__(self) -> None:
        self.ask_session_ids: list[str | None] = []

    async def ask(
        self,
        *,
        question: str,
        mode: str,
        episode_slug: str | None = None,
        current_page: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        self.ask_session_ids.append(session_id)
        return {"answer": f"(answer to {question!r})", "session_id": _RETURNED_SESSION}

    async def search(
        self,
        *,
        query: str,
        mode: str,
        k: int = 3,
        current_episode: int | None = None,
        current_page: int | None = None,
    ) -> dict[str, Any]:
        return {"chunks": [{"source_table": "wiki", "text": "Komona\n\nthe flying city"}]}


def _tool_use(call_id: str, **inp: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name="ask", id=call_id, input=inp)


def _response(*content: Any, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=list(content), stop_reason=stop_reason)


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


class _FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeMessages(responses)


async def test_continue_session_threads_the_returned_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scripted agent: fresh ask, then a continued ask, then stop (no tool call).
    responses = [
        _response(_tool_use("t1", question="What is Komona?", continue_session=False),
                  stop_reason="tool_use"),
        _response(_tool_use("t2", question="Tell me more about its market.", continue_session=True),
                  stop_reason="tool_use"),
        _response(SimpleNamespace(type="text", text="done"), stop_reason="end_turn"),
    ]
    # agent.py references anthropic.AsyncAnthropic on the real module, so patch it there.
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda **_: _FakeLLM(responses))
    # Stub the judge so no real verdict call happens (judges are the only model-in-verdict path).
    monkeypatch.setattr(
        agent, "judge_hallucination",
        lambda **_: Verdict(failed=False, severity=MEDIUM, basis="judged", detail="stub"),
    )

    client = _FakeClient()
    governor = Governor(max_turns=10, max_tool_calls=10, max_usd=10.0)
    probes = await agent.run_strategy(
        strategy=ALL["hallucination"], client=cast(RedteamMCPClient, client),
        governor=governor, episode=2, page=3,
    )

    # The invariant: turn 1 starts fresh (None); turn 2 continues with the id the server returned.
    assert client.ask_session_ids == [None, _RETURNED_SESSION]
    # Two probes recorded, both ask, the second tagged with the continued session.
    assert [p.tool for p in probes] == ["ask", "ask"]
    assert probes[0].session_id == _RETURNED_SESSION
    assert probes[1].session_id == _RETURNED_SESSION
    assert governor.stop_reason is not None and "agent stopped" in governor.stop_reason
