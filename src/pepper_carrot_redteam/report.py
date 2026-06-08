"""Outputs — a human-readable findings report and machine-readable candidate gold.

Two artifacts per run (DESIGN §5):
  1. A findings report (Markdown): per probe, the intent, transcript, oracle verdict, severity.
  2. Candidate gold: every CONFIRMED failure written in pepper-carrot-eval's gold schema as
     `*.candidate.yaml`, ready for human review and a move into the eval's data/ dir.

This is how discovery feeds measurement — "find once, guard forever". STATUS: stubs.
"""

from __future__ import annotations

from typing import Any

from .agent import Probe


def build_findings_report(strategy_name: str, probes: list[Probe]) -> str:
    """Render probes as a Markdown triage report.

    TODO: header (strategy, position, run stamp, governor stop_reason), then one section per probe
    with intent, the search/ask transcript, and the verdict (basis + severity). Summarize at top:
    "N probes · M confirmed failures (k critical, j medium)".
    """
    raise NotImplementedError("findings report — see DESIGN §8 build order, step 5")


def to_candidate_gold(probe: Probe) -> dict[str, Any]:
    """Map one confirmed-failure Probe to a pepper-carrot-eval gold entry.

    TODO: shape per the eval's schema —
      - a spoiler leak → a `gold_refusal.yaml` item (kind: spoiler, reader_position, forbidden…),
      - a hallucination → a `gold_qa.yaml` item (question, reference_answer, must_cite_chunk_keys…).
    Tag with `_source: redteam` and `_verify: {...}` and write only CONFIRMED (verdict.failed)
    probes. A human reviews before it moves into the eval.
    """
    raise NotImplementedError("candidate-gold mapping — see DESIGN §8 build order, step 5")


def write_candidate_gold(probes: list[Probe], out_dir: str) -> list[str]:
    """Write each confirmed failure as `<id>.candidate.yaml` under out_dir; return the paths."""
    raise NotImplementedError("candidate-gold writer — see DESIGN §8 build order, step 5")
