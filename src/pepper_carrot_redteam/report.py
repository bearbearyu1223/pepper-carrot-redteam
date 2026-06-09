"""Outputs — a human-readable findings report and machine-readable candidate gold.

Two artifacts per run (DESIGN §5):
  1. A findings report (Markdown): per probe, the intent, result summary, oracle verdict, severity,
     grouped by conversation so social-engineering arcs read as one thread.
  2. Candidate gold: every CONFIRMED failure written in pepper-carrot-eval's gold schema, tagged
     `_source: redteam` + a `_verify` block, ready for human review and a move into the eval's
     data/ dir. Per ADR 0002 / DESIGN §5: a spoiler leak → a `gold_refusal` `kind: spoiler` item;
     a hallucination → a `gold_refusal` `kind: unanswerable` item (decline, don't fabricate).

This is how discovery feeds measurement — "find once, guard forever".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .agent import Probe
from .oracle import CRITICAL

# Map a strategy's oracle to how its confirmed failures land in the eval's refusal gold.
_GOLD_KIND = {"spoiler": "spoiler", "hallucination": "unanswerable"}


def _confirmed(probes: list[Probe]) -> list[Probe]:
    return [p for p in probes if p.verdict is not None and p.verdict.failed]


# ── findings report (Markdown) ──────────────────────────────────────────────────────────────────

def build_findings_report(
    *,
    strategy_name: str,
    episode: int,
    page: int,
    probes: list[Probe],
    stop_reason: str | None,
    run_id: str,
) -> str:
    """Render probes as a Markdown triage report, grouped by conversation (session)."""
    confirmed = _confirmed(probes)
    crit = sum(1 for p in confirmed if p.verdict and p.verdict.severity == CRITICAL)
    med = len(confirmed) - crit

    lines: list[str] = [
        f"# Red-team findings — {strategy_name}",
        "",
        f"- **run**: `{run_id}`",
        f"- **reader position**: episode {episode}, page {page}",
        f"- **stop reason**: {stop_reason or 'completed'}",
        f"- **summary**: {len(probes)} probes · {len(confirmed)} confirmed "
        f"({crit} critical, {med} medium)",
        "",
        "> Agentic runs are not reproducible — this is coverage, not a score (ADR 0001). "
        "Confirmed failures are written as candidate gold for human review.",
        "",
    ]

    # Group by conversation: stateless `search` probes (session_id None) first, then each `ask`
    # session as its own social-engineering arc.
    stateless = [p for p in probes if p.session_id is None]
    sessions: dict[str, list[Probe]] = {}
    for p in probes:
        if p.session_id is not None:
            sessions.setdefault(p.session_id, []).append(p)

    if stateless:
        lines.append("## Stateless probes (`search`)\n")
        lines.extend(_render_probe(p) for p in stateless)
        lines.append("")

    for i, (sid, group) in enumerate(sessions.items(), 1):
        lines.append(f"## Conversation {i} (`ask` session `{sid[:12]}…`)\n")
        lines.extend(_render_probe(p) for p in group)
        lines.append("")

    return "\n".join(lines)


def _render_probe(p: Probe) -> str:
    v = p.verdict
    if v is None:
        verdict = "_unjudged_"
    else:
        flag = "🔴 FAIL" if v.failed else "✅ held"
        verdict = f"{flag} · {v.severity} · _{v.basis}_ — {v.detail}"
    result = ", ".join(f"{k}={val}" for k, val in p.result.items())
    return f"- **turn {p.turn}** (`{p.tool}`): {p.intent!r}\n    - result: {result}\n    - verdict: {verdict}"


# ── candidate gold (eval schema) ─────────────────────────────────────────────────────────────────

def to_candidate_gold(
    probe: Probe,
    *,
    strategy_oracle: str,
    episode: int,
    page: int,
) -> dict[str, Any]:
    """Map one confirmed-failure Probe to a `gold_refusal` entry (the eval derives the episode slug
    from `reader_position`, so we don't need it here). `forbidden_content` is left for the human
    reviewer to fill from the transcript; the judge's rationale is recorded in `_verify`."""
    kind = _GOLD_KIND.get(strategy_oracle, "unanswerable")
    v = probe.verdict
    entry: dict[str, Any] = {
        "id": f"cand-redteam-{strategy_oracle}-ep{episode:02d}-t{probe.turn:02d}",
        "kind": kind,
        "mode": "page" if kind == "spoiler" else "wiki",
        "question": probe.intent,
    }
    if kind == "spoiler":
        entry["reader_position"] = {"episode": episode, "page": page}
    entry["forbidden_content"] = []  # human-review: extract the leaked / fabricated substrings
    entry["_source"] = "redteam"
    entry["_verify"] = {
        "basis": v.basis if v else None,
        "detail": v.detail if v else None,
        "severity": v.severity if v else None,
    }
    return entry


def write_candidate_gold(
    probes: list[Probe],
    out_dir: str,
    *,
    strategy_oracle: str,
    episode: int,
    page: int,
    run_id: str,
) -> list[str]:
    """Write confirmed failures as one `redteam-<oracle>-<run>.candidate.yaml` under out_dir.

    A run-stamped, redteam-namespaced filename so it never clobbers the eval's own
    `gold_*.candidate.yaml`. Returns the paths written (empty if nothing was confirmed)."""
    confirmed = _confirmed(probes)
    if not confirmed:
        return []

    entries = [
        to_candidate_gold(p, strategy_oracle=strategy_oracle, episode=episode, page=page)
        for p in confirmed
    ]
    header = (
        "# AUTO-GENERATED by pepper-carrot-redteam — confirmed failures awaiting human review.\n"
        "# Review, extract `forbidden_content` from the transcript, drop `_verify`/`_source`,\n"
        f"# then move good entries into pepper-carrot-eval/data/gold_refusal.yaml. (run: {run_id})\n"
    )
    body = yaml.safe_dump(entries, sort_keys=False, default_flow_style=False, allow_unicode=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"redteam-{strategy_oracle}-{run_id}.candidate.yaml"
    path.write_text(header + body, encoding="utf-8")
    return [str(path)]
