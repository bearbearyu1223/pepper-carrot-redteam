"""The blind-spot agent's retrieval feedback, tested deterministically — no network, no model.

`probe_retrieval` runs two wiki searches and a structural verdict; this pins the *observation* the
harness now feeds back so the agent can hill-climb toward a miss: the target's rank/score under its
own name, its rank under the paraphrase (None when the paraphrase dropped it), and the competitors
that crowded it out. The verdict stays with `retrieval_blindspot`; this only checks the summary.
"""

from __future__ import annotations

from typing import Any, cast

from pepper_carrot_redteam import agent
from pepper_carrot_redteam.client import RedteamMCPClient
from pepper_carrot_redteam.governor import Governor
from pepper_carrot_redteam.strategies import ALL


def _wiki(*scored: tuple[str, float]) -> dict[str, Any]:
    return {"chunks": [{"source_table": "wiki", "text": f"{t}\n\nbody", "score": s} for t, s in scored]}


class _FakeClient:
    """Returns a canned wiki result per query: the target's own name vs. the paraphrase."""

    def __init__(self, *, name_query: str, by_name: dict[str, Any], by_paraphrase: dict[str, Any]):
        self._name_query = name_query
        self._by_name = by_name
        self._by_paraphrase = by_paraphrase

    async def search(self, *, query: str, mode: str, k: int = 3, **_: Any) -> dict[str, Any]:
        return self._by_name if query == self._name_query else self._by_paraphrase


async def _run(client: _FakeClient, *, target: str, paraphrase: str) -> agent._Dispatch:
    return await agent._dispatch(
        strategy=ALL["blindspot"], client=cast(RedteamMCPClient, client),
        governor=Governor(max_turns=10, max_tool_calls=10, max_usd=10.0),
        name="probe_retrieval",
        raw={"target_title": target, "paraphrase_query": paraphrase},
        episode=2, page=3, episode_slug=None, current_session_id=None,
    )


async def test_blindspot_feedback_reports_ranks_and_competitors() -> None:
    # Komona is rank 1 by name, but the paraphrase surfaces only other entities → a blind spot.
    client = _FakeClient(
        name_query="Komona",
        by_name=_wiki(("Komona", 0.71), ("Hereva", 0.42)),
        by_paraphrase=_wiki(("Pepper", 0.50), ("Carrot", 0.40), ("Hereva", 0.30), ("Mango", 0.20)),
    )
    d = await _run(client, target="Komona", paraphrase="the floating market town for potions")

    assert d.verdict is not None and d.verdict.failed is True  # the oracle still owns this
    assert d.summary["name_rank"] == {"rank": 1, "score": 0.71}
    assert d.summary["paraphrase_rank"] is None  # the paraphrase dropped it → blind spot
    assert d.summary["competitors"] == [
        {"title": "pepper", "score": 0.5},
        {"title": "carrot", "score": 0.4},
        {"title": "hereva", "score": 0.3},
    ]  # top-3 only — what crowded the target out


async def test_blindspot_feedback_reports_paraphrase_rank_when_still_surfaced() -> None:
    # The paraphrase still finds Komona, just lower down → not a blind spot, but the rank is shown.
    client = _FakeClient(
        name_query="Komona",
        by_name=_wiki(("Komona", 0.71), ("Hereva", 0.42)),
        by_paraphrase=_wiki(("Pepper", 0.50), ("Carrot", 0.40), ("Komona", 0.31)),
    )
    d = await _run(client, target="Komona", paraphrase="the city Pepper visits")

    assert d.verdict is not None and d.verdict.failed is False
    assert d.summary["paraphrase_rank"] == {"rank": 3, "score": 0.31}
