"""CLI entrypoint — wire config + client + governor + agent + report together.

    uv run pepper-carrot-redteam --strategy spoiler --episode 2 --page 3
    uv run pepper-carrot-redteam --strategy hallucination
    uv run pepper-carrot-redteam --strategy spoiler --dry-run     # 1 probe, no gold written
    uv run pepper-carrot-redteam --strategy spoiler -v            # per-probe progress
    uv run pepper-carrot-redteam --strategy spoiler --multi-turn  # force every ask to continue

Writes a findings report to `findings/<run-id>.md` and, for confirmed failures, candidate gold to
`EVAL_GOLD_DIR` (a pepper-carrot-eval/data checkout) — unless `--dry-run`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import report
from .agent import run_strategy
from .client import RedteamMCPClient
from .config import get_config
from .governor import Governor
from .strategies import ALL
from .tracing import Tracer, set_tracer

logger = logging.getLogger("pepper_carrot_redteam")

_FINDINGS = Path(__file__).resolve().parents[2] / "findings"
_TRACES = Path(__file__).resolve().parents[2] / "traces"


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s · %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    if verbosity < 2:
        for name in ("httpx", "httpcore", "mcp", "mcp.client", "anthropic", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


async def _run(
    strategy_name: str,
    episode: int,
    page: int,
    *,
    max_tool_calls: int | None,
    dry_run: bool,
    trace: bool,
    force_multi_turn: bool,
) -> None:
    cfg = get_config()
    if not cfg.agent_enabled:
        raise SystemExit("ANTHROPIC_API_KEY is required to drive the agent loop and judges.")
    strategy = ALL[strategy_name]

    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    if trace:
        tracer = Tracer(_TRACES / f"{run_id}.jsonl", run_id)
        set_tracer(tracer)
        logger.info("tracing → %s", tracer.path)
    tool_cap = 1 if dry_run else (max_tool_calls if max_tool_calls is not None else cfg.max_tool_calls)
    governor = Governor(
        max_turns=cfg.max_turns, max_tool_calls=tool_cap, max_usd=cfg.max_usd,
        stall_patience=cfg.stall_patience,
    )
    logger.info(
        "run %s · strategy=%s @(%d,%d) · caps: turns=%d tool_calls=%d usd=$%.2f stall=%d%s%s",
        run_id, strategy_name, episode, page, cfg.max_turns, tool_cap, cfg.max_usd,
        cfg.stall_patience,
        " · DRY-RUN" if dry_run else "",
        " · MULTI-TURN" if force_multi_turn else "",
    )

    async with RedteamMCPClient(cfg.mcp_server_url) as client:
        probes = await run_strategy(
            strategy=strategy, client=client, governor=governor, episode=episode, page=page,
            force_multi_turn=force_multi_turn,
        )

    markdown = report.build_findings_report(
        strategy_name=strategy_name, episode=episode, page=page,
        probes=probes, stop_reason=governor.stop_reason, run_id=run_id,
    )
    _FINDINGS.mkdir(parents=True, exist_ok=True)
    findings_path = _FINDINGS / f"{run_id}.md"
    findings_path.write_text(markdown, encoding="utf-8")
    logger.info("wrote findings → %s", findings_path)

    gold_paths: list[str] = []
    if not dry_run:
        gold_paths = report.write_candidate_gold(
            probes, cfg.eval_gold_dir,
            strategy_oracle=strategy.oracle, episode=episode, page=page, run_id=run_id,
        )
        for p in gold_paths:
            logger.info("wrote candidate gold → %s", p)

    print(markdown)
    confirmed = sum(1 for p in probes if p.verdict is not None and p.verdict.failed)
    print(
        f"\n{strategy_name}: {len(probes)} probes · {confirmed} confirmed · "
        f"~${governor.spent_usd:.3f} · stop={governor.stop_reason}"
    )
    if gold_paths:
        print(f"candidate gold: {', '.join(gold_paths)}")
    elif not dry_run:
        print("candidate gold: none (no confirmed failures)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic red-teamer for the Pepper & Carrot companion."
    )
    parser.add_argument("--strategy", choices=sorted(ALL), default="spoiler")
    parser.add_argument("--episode", type=int, default=None, help="target episode (default: config)")
    parser.add_argument("--page", type=int, default=None, help="target page (default: config)")
    parser.add_argument(
        "--max-tool-calls", type=int, default=None, metavar="N",
        help="override the tool-call budget (default: config MAX_TOOL_CALLS).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="cheap smoke test: cap to 1 tool call and do not write candidate gold.",
    )
    parser.add_argument(
        "--multi-turn", action="store_true",
        help="force every `ask` probe to continue the same session (overrides the agent's "
             "continue_session choice). Only affects ask-based strategies: spoiler, hallucination, "
             "injection.",
    )
    parser.add_argument(
        "--no-trace", action="store_true",
        help="disable the JSONL forensic trace (traces/<run-id>.jsonl).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v = per-probe progress; -vv = every HTTP/SDK call (debug).",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = get_config()
    episode = args.episode if args.episode is not None else cfg.target_episode
    page = args.page if args.page is not None else cfg.target_page
    asyncio.run(
        _run(
            args.strategy, episode, page,
            max_tool_calls=args.max_tool_calls, dry_run=args.dry_run, trace=not args.no_trace,
            force_multi_turn=args.multi_turn,
        )
    )


if __name__ == "__main__":
    main()
