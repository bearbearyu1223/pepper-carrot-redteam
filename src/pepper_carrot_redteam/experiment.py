"""Metered experiment harness — run a small grid of strategies x replicates and measure the REAL
token/$ cost, plus the Break Rate per strategy.

Why this exists: the governor's `spent_usd` is a *notional* tool-cost budget. The real bill has two
parts, both on the SAME account:
  1. CLIENT-SIDE calls this harness makes — the Opus agent turns (dominant) plus the Sonnet judges
     and Haiku translates — metered exactly from the SDK `usage` on every call.
  2. SERVER-SIDE companion generation behind every MCP `ask` (and retrieval embeddings behind
     `search`). The MCP server runs on the same account but does not return usage, so the client
     can't meter it — instead we COUNT the calls and apply a per-call estimate (--ask-cost / --search-cost).
Total = metered client cost + estimated companion cost. So a full-grid cost can be extrapolated from
a cheap smoke batch.

    # $0 pipeline check — fakes the model + MCP, no key, no network:
    uv run python -m pepper_carrot_redteam.experiment --mock --reps 2

    # real metered smoke (needs ANTHROPIC_API_KEY; spends real money). The governor caps
    # (MAX_TURNS / MAX_TOOL_CALLS / MAX_USD / STALL_PATIENCE) and TARGET_EPISODE/PAGE come from
    # .env unless overridden by a flag (--max-tool-calls, --episode, --page, --multi-turn):
    uv run python -m pepper_carrot_redteam.experiment --reps 5

Writes one record per run to experiments/<exp-id>/runs.jsonl and prints a Break Rate + cost summary
with a projected full-grid cost. The metric is **Break Rate** (fraction of runs that surface a
confirmed failure); we also report its complement, the Hold Rate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic

from .agent import run_strategy
from .client import RedteamMCPClient
from .config import get_config
from .governor import Governor
from .oracle import CRITICAL
from .strategies import ALL

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"

# ── pricing (USD per 1M tokens) — ESTIMATES; verify against current Anthropic pricing ─────────────
# Token COUNTS in the output are exact (from the SDK); only the $ conversion depends on these.
_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-8":   {"input": 15.0, "output": 75.0, "cache_read": 1.5},
    "claude-sonnet-4-6": {"input": 3.0,  "output": 15.0, "cache_read": 0.3},
    "claude-haiku-4-5":  {"input": 1.0,  "output": 5.0,  "cache_read": 0.1},
}
_DEFAULT_PRICE = {"input": 5.0, "output": 15.0, "cache_read": 0.5}

# Estimated server-side companion cost per MCP call — billed to the SAME account as the agent/judges,
# but invisible to this client (the MCP server doesn't return usage). ESTIMATES; override with
# --ask-cost / --search-cost, or have the server return usage to make these exact.
_COMPANION_COST = {"ask": 0.01, "search": 0.0005}


# ── token metering ───────────────────────────────────────────────────────────────────────────────

@dataclass
class Meter:
    """Accumulates per-model token usage (client-side) and MCP call counts (server-side companion)."""

    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    mcp: dict[str, int] = field(default_factory=lambda: {"ask": 0, "search": 0})

    def count_mcp(self, name: str) -> None:
        self.mcp[name] = self.mcp.get(name, 0) + 1

    def record(self, model: str, usage: Any) -> None:
        if usage is None:
            return
        d = self.by_model.setdefault(
            model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0}
        )
        d["input"] += int(getattr(usage, "input_tokens", 0) or 0)
        d["output"] += int(getattr(usage, "output_tokens", 0) or 0)
        d["cache_read"] += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        d["cache_write"] += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        d["calls"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {m: dict(v) for m, v in self.by_model.items()}


def _diff(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Per-model token delta between two snapshots — attributes usage to a single run."""
    out: dict[str, dict[str, int]] = {}
    for model, tb in after.items():
        ta = before.get(model, {})
        delta = {k: tb[k] - ta.get(k, 0) for k in tb}
        if any(delta.values()):
            out[model] = delta
    return out


def cost_usd(tokens: dict[str, dict[str, int]]) -> float:
    """Convert a per-model token tally to a USD estimate (cache writes ~1.25x input)."""
    total = 0.0
    for model, t in tokens.items():
        p = _PRICES.get(model, _DEFAULT_PRICE)
        total += t.get("input", 0) / 1e6 * p["input"]
        total += t.get("output", 0) / 1e6 * p["output"]
        total += t.get("cache_read", 0) / 1e6 * p["cache_read"]
        total += t.get("cache_write", 0) / 1e6 * p["input"] * 1.25
    return total


def companion_cost_usd(counts: dict[str, int], prices: dict[str, float]) -> float:
    """Estimated server-side cost for the companion calls behind the MCP tools (same account)."""
    return counts.get("ask", 0) * prices["ask"] + counts.get("search", 0) * prices["search"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (robust at small n / extreme p)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ── client wiring: metered real clients, or fakes for --mock ─────────────────────────────────────

_REAL_ASYNC = anthropic.AsyncAnthropic
_REAL_SYNC = anthropic.Anthropic


def _install_metered(meter: Meter) -> None:
    """Swap the Anthropic client classes for thin wrappers that tally `usage` into `meter`."""

    class _AsyncMessages:
        def __init__(self, real: Any) -> None:
            self._real = real

        async def create(self, **kw: Any) -> Any:
            resp = await self._real.create(**kw)
            meter.record(str(kw.get("model", "?")), getattr(resp, "usage", None))
            return resp

    class _SyncMessages:
        def __init__(self, real: Any) -> None:
            self._real = real

        def create(self, **kw: Any) -> Any:
            resp = self._real.create(**kw)
            meter.record(str(kw.get("model", "?")), getattr(resp, "usage", None))
            return resp

    def mk_async(**kw: Any) -> Any:
        real = _REAL_ASYNC(**kw)
        return SimpleNamespace(messages=_AsyncMessages(real.messages))

    def mk_sync(**kw: Any) -> Any:
        real = _REAL_SYNC(**kw)
        return SimpleNamespace(messages=_SyncMessages(real.messages))

    # setattr with a variable name (not a constant) keeps both ruff (B010) and mypy happy while
    # monkeypatching the module-level client classes the agent/judge construct at call time.
    for attr, factory in (("AsyncAnthropic", mk_async), ("Anthropic", mk_sync)):
        setattr(anthropic, attr, factory)


_MOCK_USAGE = SimpleNamespace(
    input_tokens=1200, output_tokens=350, cache_read_input_tokens=200, cache_creation_input_tokens=0
)


def _install_mock(meter: Meter) -> None:
    """Swap in fake model clients that emit one tool call per turn with fake usage — no key, no spend."""

    class _AsyncMessages:
        async def create(self, **kw: Any) -> Any:
            tool = str(kw["tools"][0]["name"])
            if tool == "probe_retrieval":
                inp: dict[str, Any] = {"target_title": "Komona", "paraphrase_query": "the floating city"}
            elif tool == "search":
                inp = {"query": "what happens next"}
            else:
                inp = {"question": "what happens next?"}
            block = SimpleNamespace(type="tool_use", name=tool, id="t", input=inp)
            meter.record(str(kw.get("model", "?")), _MOCK_USAGE)
            return SimpleNamespace(content=[block], stop_reason="tool_use", usage=_MOCK_USAGE)

    class _SyncMessages:
        def create(self, **kw: Any) -> Any:
            # Cover every judge's score key so each caller finds its own.
            out = {"leaked": 0.0, "fabricated": 0.0, "answered_ood": 0.0, "rationale": "mock"}
            block = SimpleNamespace(type="tool_use", input=out)
            meter.record(str(kw.get("model", "?")), _MOCK_USAGE)
            return SimpleNamespace(content=[block], usage=_MOCK_USAGE)

    def mk_async(**_: Any) -> Any:
        return SimpleNamespace(messages=_AsyncMessages())

    def mk_sync(**_: Any) -> Any:
        return SimpleNamespace(messages=_SyncMessages())

    for attr, factory in (("AsyncAnthropic", mk_async), ("Anthropic", mk_sync)):
        setattr(anthropic, attr, factory)


class _MockMCP:
    """A fake MCP client for --mock: canned ask/search, no network."""

    async def ask(self, **_: Any) -> dict[str, Any]:
        return {"answer": "I'd rather not spoil it!", "session_id": "S1"}

    async def search(self, **_: Any) -> dict[str, Any]:
        return {"chunks": [
            {"source_table": "pages", "metadata": {"episode_number": 1, "page_number": 1}},
            {"source_table": "wiki", "text": "Komona\n\nthe floating city", "score": 0.5},
        ]}

    async def episodes(self) -> list[dict[str, Any]]:
        return [{"slug": f"ep{n:02d}", "episode_number": n} for n in range(1, 13)]

    async def __aenter__(self) -> _MockMCP:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False


class _CountingMCP:
    """Wraps an MCP client to count ask/search calls — the server-side, same-account companion cost
    this client can't meter directly."""

    def __init__(self, real: Any, meter: Meter) -> None:
        self._real = real
        self._meter = meter

    async def ask(self, **kw: Any) -> Any:
        self._meter.count_mcp("ask")
        return await self._real.ask(**kw)

    async def search(self, **kw: Any) -> Any:
        self._meter.count_mcp("search")
        return await self._real.search(**kw)

    async def episodes(self) -> Any:
        return await self._real.episodes()


# ── the grid ─────────────────────────────────────────────────────────────────────────────────────

async def _run_one(
    client: Any, strategy_name: str, *, episode: int, page: int,
    max_tool_calls: int, force_multi_turn: bool, meter: Meter, rep: int,
    companion_prices: dict[str, float],
) -> dict[str, Any]:
    cfg = get_config()
    before = meter.snapshot()
    mcp_before = dict(meter.mcp)
    gov = Governor(
        max_turns=cfg.max_turns, max_tool_calls=max_tool_calls, max_usd=cfg.max_usd,
        stall_patience=cfg.stall_patience,
    )
    t0 = time.perf_counter()
    probes = await run_strategy(
        strategy=ALL[strategy_name], client=client, governor=gov,
        episode=episode, page=page, force_multi_turn=force_multi_turn,
    )
    wall = time.perf_counter() - t0
    tokens = _diff(before, meter.snapshot())
    mcp_calls = {k: meter.mcp.get(k, 0) - mcp_before.get(k, 0) for k in meter.mcp}
    client_c = cost_usd(tokens)
    companion_c = companion_cost_usd(mcp_calls, companion_prices)
    confirmed = [p for p in probes if p.verdict and p.verdict.failed]
    return {
        "strategy": strategy_name, "episode": episode, "page": page, "rep": rep,
        "broke": bool(confirmed), "n_probes": len(probes), "n_confirmed": len(confirmed),
        "n_critical": sum(1 for p in confirmed if p.verdict and p.verdict.severity == CRITICAL),
        "stop_reason": gov.stop_reason, "governor_usd": round(gov.spent_usd, 4),
        "wall_s": round(wall, 2), "tokens": tokens,
        "n_ask": mcp_calls.get("ask", 0), "n_search": mcp_calls.get("search", 0),
        "client_cost_usd": round(client_c, 4), "companion_cost_usd": round(companion_c, 4),
        "cost_usd": round(client_c + companion_c, 4),
    }


async def run_grid(
    *, strategies: list[str], episode: int, page: int, reps: int,
    max_tool_calls: int, force_multi_turn: bool, mock: bool, meter: Meter,
    companion_prices: dict[str, float],
) -> list[dict[str, Any]]:
    cfg = get_config()
    records: list[dict[str, Any]] = []
    client_cm: Any = _MockMCP() if mock else RedteamMCPClient(cfg.mcp_server_url)
    async with client_cm as raw_client:
        client = _CountingMCP(raw_client, meter)
        for strategy_name in strategies:
            for rep in range(1, reps + 1):
                rec = await _run_one(
                    client, strategy_name, episode=episode, page=page,
                    max_tool_calls=max_tool_calls, force_multi_turn=force_multi_turn,
                    meter=meter, rep=rep, companion_prices=companion_prices,
                )
                records.append(rec)
                print(
                    f"  {strategy_name:<13} rep {rep}/{reps}: "
                    f"{'BROKE' if rec['broke'] else 'held'} · {rec['n_probes']} probes · "
                    f"${rec['cost_usd']:.3f} · {rec['wall_s']}s"
                )
    return records


# ── reporting ──────────────────────────────────────────────────────────────────────────────────--

def summarize(records: list[dict[str, Any]], *, full_grid_runs: int) -> str:
    lines = ["", "strategy        runs  broke  break_rate (95% CI)        hold   mean$/run   total$"]
    lines.append("-" * 82)
    by_strat: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_strat.setdefault(r["strategy"], []).append(r)
    for strat, rs in by_strat.items():
        n = len(rs)
        k = sum(1 for r in rs if r["broke"])
        lo, hi = wilson(k, n)
        br = k / n if n else 0.0
        mean_cost = sum(r["cost_usd"] for r in rs) / n if n else 0.0
        total = sum(r["cost_usd"] for r in rs)
        lines.append(
            f"{strat:<14} {n:>5} {k:>6}  {br:>4.2f} ({lo:.2f}-{hi:.2f})         "
            f"{1 - br:>4.2f}   {mean_cost:>8.3f}  {total:>8.3f}"
        )
    n_all = len(records)
    total_all = sum(r["cost_usd"] for r in records)
    mean_all = total_all / n_all if n_all else 0.0
    client_all = sum(r["client_cost_usd"] for r in records)
    companion_all = sum(r["companion_cost_usd"] for r in records)
    lines.append("-" * 82)
    lines.append(f"{'TOTAL':<14} {n_all:>5}  {'':>6} {'':>26} {mean_all:>14.3f}  {total_all:>8.3f}")
    lines.append("")
    lines.append(
        f"cost split: client/metered ${client_all:.3f} (agent + judges, exact tokens) + "
        f"companion/estimated ${companion_all:.3f} (server-side ask/search) = ${total_all:.3f}"
    )
    lines.append(f"projected full grid ({full_grid_runs} runs) at this mean $/run: "
                 f"~${mean_all * full_grid_runs:.0f}")
    lines.append("(token COUNTS exact; client $ uses estimated _PRICES, companion $ uses "
                 "--ask-cost/--search-cost)")
    return "\n".join(lines)


def _tokens_by_model(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    agg: dict[str, dict[str, int]] = {}
    for r in records:
        for model, t in r["tokens"].items():
            d = agg.setdefault(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0})
            for kk in d:
                d[kk] += t.get(kk, 0)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Metered Break-Rate smoke experiment.")
    parser.add_argument("--strategies", default=",".join(sorted(ALL)),
                        help="comma-separated strategy names (default: all four).")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--reps", type=int, default=5, help="runs per strategy (default: 5).")
    parser.add_argument("--max-tool-calls", type=int, default=None,
                        help="per-run tool-call budget (default: config MAX_TOOL_CALLS).")
    parser.add_argument("--multi-turn", action="store_true", help="force session continuation.")
    parser.add_argument("--full-grid-runs", type=int, default=720,
                        help="run count to project the full-grid cost to (default: 4x9x20=720).")
    parser.add_argument("--ask-cost", type=float, default=_COMPANION_COST["ask"],
                        help="estimated server-side $ per companion `ask` (same account).")
    parser.add_argument("--search-cost", type=float, default=_COMPANION_COST["search"],
                        help="estimated server-side $ per `search` (retrieval embedding).")
    parser.add_argument("--mock", action="store_true",
                        help="fake the model + MCP — validates the pipeline for $0, no key/network.")
    args = parser.parse_args()

    cfg = get_config()
    if not args.mock and not cfg.agent_enabled:
        raise SystemExit("ANTHROPIC_API_KEY is required for a real run (use --mock to dry-run).")
    episode = args.episode if args.episode is not None else cfg.target_episode
    page = args.page if args.page is not None else cfg.target_page
    # Fall back to the config cap when --max-tool-calls is omitted (mirrors run.py), so the .env
    # MAX_TOOL_CALLS governs the experiment unless explicitly overridden per run.
    tool_cap = args.max_tool_calls if args.max_tool_calls is not None else cfg.max_tool_calls
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in ALL:
            raise SystemExit(f"unknown strategy {s!r}; choose from {sorted(ALL)}")

    meter = Meter()
    _install_mock(meter) if args.mock else _install_metered(meter)

    exp_id = datetime.now().strftime("exp-%Y%m%d-%H%M%S") + ("-mock" if args.mock else "")
    out_dir = _EXPERIMENTS / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {exp_id} · {len(strategies)} strategies x {args.reps} reps @ ({episode},{page})\n"
          f"    caps: tool_calls={tool_cap} turns={cfg.max_turns} usd=${cfg.max_usd} "
          f"stall={cfg.stall_patience}{' · MOCK' if args.mock else ''} ===")

    companion_prices = {"ask": args.ask_cost, "search": args.search_cost}
    records = asyncio.run(run_grid(
        strategies=strategies, episode=episode, page=page, reps=args.reps,
        max_tool_calls=tool_cap, force_multi_turn=args.multi_turn,
        mock=args.mock, meter=meter, companion_prices=companion_prices,
    ))

    (out_dir / "runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    summary = summarize(records, full_grid_runs=args.full_grid_runs)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print("\ntokens by model (client-side, exact):")
    for model, t in _tokens_by_model(records).items():
        print(f"  {model:<22} in={t['input']:>8} out={t['output']:>7} "
              f"cache_read={t['cache_read']:>7} calls={t['calls']:>4}")
    print(f"companion MCP calls (server-side, estimated): "
          f"ask={meter.mcp.get('ask', 0)} search={meter.mcp.get('search', 0)}")
    print(f"\nwrote {out_dir}/runs.jsonl + summary.txt")


if __name__ == "__main__":
    main()
