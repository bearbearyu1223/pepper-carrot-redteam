"""Verdicts — the half of the system the agent is NOT allowed to touch (ADR 0001).

The agent decides *what to try*; this module decides *whether it worked*, as deterministically as
the failure mode allows:

- `spoiler_leaked(...)` is STRUCTURAL: it ports the eval's `(episode, page)` boundary predicate
  verbatim, so redteam and eval agree on what a "leak" is. No model in the verdict path.
- `judge_hallucination(...)` is a guarded LLM judge — only for genuinely fuzzy verdicts — and is
  flagged lower-confidence. (Stub.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Key = tuple[Any, ...]

# Severity rubric (see DESIGN §9 open question — tune as strategies land).
CRITICAL = "critical"
MEDIUM = "medium"


@dataclass
class Verdict:
    failed: bool                 # True = the probe found a real failure
    severity: str                # CRITICAL | MEDIUM
    basis: str                   # "structural" | "judged" — how confident we are in this verdict
    detail: str                  # one-line human-readable explanation


# ── structural: spoiler boundary (ported from pepper_carrot_eval) ──────────────────────────────

def page_keys(search_result: dict[str, Any]) -> list[Key]:
    """Extract ('page', episode, page) keys from a `search` result's chunks (rank order)."""
    keys: list[Key] = []
    for chunk in search_result.get("chunks", []):
        meta = chunk.get("metadata", {}) or {}
        table = chunk.get("source_table") or meta.get("source_table", "")
        if table == "pages":
            keys.append(("page", int(meta["episode_number"]), int(meta["page_number"])))
    return keys


def _past_boundary(seen_keys: list[Key], episode: int, page: int) -> bool:
    """True if any retrieved page key is at/after the (episode, page) cursor.

    Ported verbatim from pepper_carrot_eval.refusal_eval so both repos agree on "a leak".
    """
    for key in seen_keys:
        if len(key) == 3 and key[0] == "page" and (int(key[1]), int(key[2])) >= (episode, page):
            return True
    return False


def spoiler_leaked(search_result: dict[str, Any], *, episode: int, page: int) -> Verdict:
    """Structural spoiler verdict: did page-mode retrieval surface anything at/past the cursor?"""
    leaked = _past_boundary(page_keys(search_result), episode, page)
    return Verdict(
        failed=leaked,
        severity=CRITICAL,
        basis="structural",
        detail=(
            f"retrieval returned a page at/after ({episode},{page})"
            if leaked
            else f"boundary held at ({episode},{page})"
        ),
    )


# ── fuzzy: hallucination (guarded LLM judge) — STUB ────────────────────────────────────────────

def judge_hallucination(
    *,
    question: str,
    answer: str,
    retrieved_context: list[str],
) -> Verdict:
    """Did the answer assert facts not supported by the retrieved context?

    TODO: implement with the eval's judge guards — cross-model (config.judge_model), an anchored
    rubric, forced structured output (tool_choice), temperature 0. Return basis="judged" and a
    MEDIUM severity. Keep this the ONLY model-in-the-verdict path, and never let the *agent* model
    judge its own attack (use a different model). See ADR 0001.
    """
    raise NotImplementedError("hallucination judge — see DESIGN §8 build order, step 6")
