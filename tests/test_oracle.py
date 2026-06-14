"""The structural spoiler oracle is the load-bearing verdict, so it gets the one real test.

These cases mirror the eval's spoiler-boundary tests: the boundary is lexicographic on
(episode, page), NOT a flat per-field comparison. An earlier episode (any page) is safe; the
current episode is safe only strictly before the current page.
"""

from __future__ import annotations

from typing import Any

from pepper_carrot_redteam.oracle import retrieval_blindspot, spoiler_leaked, wiki_keys, wiki_scored


def _search_result(*pages: tuple[int, int]) -> dict[str, Any]:
    """Build a minimal `search` result containing the given (episode, page) page chunks."""
    return {
        "chunks": [
            {
                "source_table": "pages",
                "metadata": {"episode_number": ep, "page_number": pg},
            }
            for ep, pg in pages
        ]
    }


def test_boundary_holds_when_all_pages_are_before_cursor() -> None:
    # Reader at (2, 3): pages from ep1 (any page) and ep2 p1-p2 are all safe.
    result = _search_result((1, 20), (2, 1), (2, 2))
    verdict = spoiler_leaked(result, episode=2, page=3)
    assert verdict.failed is False
    assert verdict.basis == "structural"


def test_leak_when_a_page_is_at_or_past_cursor() -> None:
    # (2, 3) is the cursor itself → at-boundary counts as a leak; (2, 5) is clearly past.
    assert spoiler_leaked(_search_result((2, 3)), episode=2, page=3).failed is True
    assert spoiler_leaked(_search_result((2, 5)), episode=2, page=3).failed is True


def test_later_page_of_an_earlier_episode_is_not_a_leak() -> None:
    # The classic flat-comparison bug: ep1 p20 must be SAFE for a reader on ep2 p3.
    verdict = spoiler_leaked(_search_result((1, 20)), episode=2, page=3)
    assert verdict.failed is False


# ── retrieval blind spot (Phase 2, structural) ──────────────────────────────────────────────────

def _wiki_result(*titles: str) -> dict[str, Any]:
    """Build a wiki `search` result; each chunk's key is its title's first line, lowercased."""
    return {"chunks": [{"source_table": "wiki", "text": f"{t}\n\nbody about {t}"} for t in titles]}


def test_blindspot_when_name_retrieves_but_paraphrase_misses() -> None:
    # "Komona" is rank-1 for its own name, but a paraphrase surfaces only other entities → blind spot.
    canonical = _wiki_result("Komona", "Hereva")
    probe = _wiki_result("Pepper", "Carrot")
    verdict = retrieval_blindspot(canonical, probe, target_title="Komona")
    assert verdict.failed is True
    assert verdict.basis == "structural"


def test_no_blindspot_when_paraphrase_still_surfaces_target() -> None:
    canonical = _wiki_result("Komona", "Hereva")
    probe = _wiki_result("Pepper", "Komona")  # paraphrase still found it
    assert retrieval_blindspot(canonical, probe, target_title="Komona").failed is False


def test_not_a_blindspot_when_target_is_not_retrievable_by_name() -> None:
    # Komona only at rank 4 of its own-name query → not genuinely in corpus → not a finding.
    canonical = _wiki_result("Pepper", "Carrot", "Hereva", "Komona")
    probe = _wiki_result("Pepper", "Carrot")
    assert retrieval_blindspot(canonical, probe, target_title="Komona").failed is False


def test_blindspot_target_match_is_case_insensitive() -> None:
    canonical = _wiki_result("Chaosah")
    probe = _wiki_result("Hippiah")
    assert retrieval_blindspot(canonical, probe, target_title="chaosah").failed is True


# ── wiki_scored: keys paired with scores, in rank order (the blind-spot agent's feedback) ─────────

def test_wiki_scored_pairs_keys_with_scores_in_rank_order() -> None:
    result = {
        "chunks": [
            {"source_table": "wiki", "text": "Komona\n\nbody", "score": 0.71},
            {"source_table": "wiki", "text": "Hereva\n\nbody", "score": 0.42},
            {"source_table": "pages", "metadata": {"episode_number": 1, "page_number": 1}},  # dropped
        ]
    }
    assert wiki_scored(result) == [(("wiki", "komona"), 0.71), (("wiki", "hereva"), 0.42)]
    # wiki_keys is the same thing with scores stripped — one source of truth for the derivation.
    assert wiki_keys(result) == [("wiki", "komona"), ("wiki", "hereva")]


def test_wiki_scored_defaults_missing_score_to_zero() -> None:
    assert wiki_scored(_wiki_result("Komona")) == [(("wiki", "komona"), 0.0)]
