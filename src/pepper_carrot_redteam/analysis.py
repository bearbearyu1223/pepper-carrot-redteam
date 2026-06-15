"""Analyze saved experiment runs — Break Rate with confidence intervals, per strategy and per
(strategy, position), plus severity, attack efficiency, and cost. Reads one or more
`experiments/<exp-id>/runs.jsonl` files, so you can re-analyze, combine several experiments, or run
an A/B ablation with a significance test — without re-spending anything.

    # one report over one or more experiments (globs ok):
    uv run python -m pepper_carrot_redteam.analysis experiments/exp-2026*/runs.jsonl

    # ablation: does a lever change Break Rate? group A (positional) vs group B (--vs):
    uv run python -m pepper_carrot_redteam.analysis experiments/baseline --vs experiments/multiturn

The metric is **Break Rate** (fraction of runs with a confirmed failure); the run is the unit of
analysis, not the probe. Trust the structural-oracle strategies (spoiler, injection, blindspot) most;
the guarded-judge paths inherit judge error.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

from .experiment import wilson

# ── loading ────────────────────────────────────────────────────────────────────────────────────--

def _resolve(patterns: list[str]) -> list[Path]:
    """Expand globs/paths into runs.jsonl files; a directory resolves to its runs.jsonl."""
    files: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat) or ([pat] if Path(pat).exists() else [])
        for m in matches:
            p = Path(m)
            p = p / "runs.jsonl" if p.is_dir() else p
            if p.is_file():
                files.append(p)
    return files


def load_runs(patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in _resolve(patterns):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── statistics ─────────────────────────────────────────────────────────────────────────────────--

def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test of Break Rate A vs B. Returns (z, p-value)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, 2 * (1 - _normal_cdf(abs(z))))


def _group(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── single-set report ─────────────────────────────────────────────────────────────────────────--

def _strategy_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Break Rate by strategy", "",
             "strategy        runs broke  break_rate (95% CI)     hold   probes  to_break  conf(crit/med)",
             "-" * 92]
    for (strat,), rs in _group(rows, "strategy").items():
        n = len(rs)
        k = sum(1 for r in rs if r.get("broke"))
        lo, hi = wilson(k, n)
        br = k / n if n else 0.0
        broke_turns = [r["first_fail_turn"] for r in rs if r.get("first_fail_turn") is not None]
        to_break = f"{_mean([float(t) for t in broke_turns]):.1f}" if broke_turns else "-"
        crit = sum(int(r.get("n_critical", 0)) for r in rs)
        med = sum(int(r.get("n_confirmed", 0)) for r in rs) - crit
        lines.append(
            f"{strat!s:<14} {n:>4} {k:>5}  {br:>4.2f} ({lo:.2f}-{hi:.2f})       "
            f"{1 - br:>4.2f}  {_mean([float(r['n_probes']) for r in rs]):>6.1f}  {to_break:>8}  "
            f"{sum(int(r.get('n_confirmed', 0)) for r in rs):>3} ({crit}/{med})"
        )
    return lines


def _position_table(rows: list[dict[str, Any]]) -> list[str]:
    positions = {(r["episode"], r["page"]) for r in rows}
    if len(positions) <= 1:
        return []
    lines = ["", "## Break Rate by strategy and position", "",
             "strategy        position   runs broke  break_rate (95% CI)", "-" * 64]
    for (strat, ep, pg), rs in _group(rows, "strategy", "episode", "page").items():
        n = len(rs)
        k = sum(1 for r in rs if r.get("broke"))
        lo, hi = wilson(k, n)
        lines.append(
            f"{strat!s:<14} {f'({ep},{pg})':>8}   {n:>4} {k:>5}  "
            f"{(k / n if n else 0.0):>4.2f} ({lo:.2f}-{hi:.2f})"
        )
    return lines


def _cost_line(rows: list[dict[str, Any]]) -> str:
    client = sum(float(r.get("client_cost_usd", 0.0)) for r in rows)
    companion = sum(float(r.get("companion_cost_usd", 0.0)) for r in rows)
    total = sum(float(r.get("cost_usd", 0.0)) for r in rows)
    n = len(rows)
    mean = total / n if n else 0.0
    return (f"client/metered ${client:.3f} + companion/estimated ${companion:.3f} = ${total:.3f} "
            f"over {n} runs (mean ${mean:.3f}/run)")


def report(rows: list[dict[str, Any]], *, n_files: int) -> str:
    if not rows:
        return "no runs found."
    lines = [f"# Experiment analysis — {len(rows)} runs from {n_files} file(s)", ""]
    lines += _strategy_table(rows)
    lines += _position_table(rows)
    lines += ["", "## Cost", "", _cost_line(rows)]
    return "\n".join(lines)


# ── A/B ablation ─────────────────────────────────────────────────────────────────────────────────

def ablation(a: list[dict[str, Any]], b: list[dict[str, Any]], *, label_a: str, label_b: str) -> str:
    lines = ["# Ablation — A vs B", "",
             f"A = {label_a} ({len(a)} runs)", f"B = {label_b} ({len(b)} runs)", "",
             "strategy        A break (n)     B break (n)     delta     z      p      sig", "-" * 78]
    strategies = list(dict.fromkeys([r["strategy"] for r in a] + [r["strategy"] for r in b]))
    for strat in [*strategies, "OVERALL"]:
        ra = a if strat == "OVERALL" else [r for r in a if r["strategy"] == strat]
        rb = b if strat == "OVERALL" else [r for r in b if r["strategy"] == strat]
        na, nb = len(ra), len(rb)
        ka = sum(1 for r in ra if r.get("broke"))
        kb = sum(1 for r in rb if r.get("broke"))
        pa = ka / na if na else 0.0
        pb = kb / nb if nb else 0.0
        z, p = two_proportion_z(ka, na, kb, nb)
        sig = "*" if p < 0.05 else ""
        lines.append(
            f"{strat:<14} {pa:>5.2f} ({na:>3})    {pb:>5.2f} ({nb:>3})    "
            f"{pb - pa:>+5.2f}    {z:>5.2f}  {p:>5.3f}  {sig}"
        )
    lines += ["", "delta = B minus A break rate; sig=* means p<0.05 (two-proportion z-test)."]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────--

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved experiment runs (Break Rate, CIs, ablation).")
    parser.add_argument("paths", nargs="+", help="runs.jsonl files, experiment dirs, or globs (group A).")
    parser.add_argument("--vs", nargs="+", default=None, metavar="PATH",
                        help="a second group (B) — produces an A-vs-B ablation with a significance test.")
    parser.add_argument("--out", default=None, help="also write the report to this file.")
    args = parser.parse_args()

    a = load_runs(args.paths)
    if not a:
        raise SystemExit(f"no runs found in {args.paths}")
    if args.vs:
        b = load_runs(args.vs)
        if not b:
            raise SystemExit(f"no runs found in {args.vs}")
        out = ablation(a, b, label_a=" ".join(args.paths), label_b=" ".join(args.vs))
    else:
        out = report(a, n_files=len(_resolve(args.paths)))

    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
