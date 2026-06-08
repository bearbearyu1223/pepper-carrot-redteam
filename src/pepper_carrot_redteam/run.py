"""CLI entrypoint — wire config + client + governor + agent + report together.

    uv run pepper-carrot-redteam --strategy spoiler --episode 2 --page 3
    uv run pepper-carrot-redteam --strategy hallucination
    uv run pepper-carrot-redteam --strategy spoiler --max-tool-calls 1   # cheap smoke test

STATUS: arg parsing + the orchestration skeleton are here; the agent loop it calls is the stub in
agent.py. Filling agent.py + report.py makes this runnable end to end.
"""

from __future__ import annotations

import argparse
import asyncio

from .agent import run_strategy
from .client import RedteamMCPClient
from .config import get_config
from .governor import Governor
from .strategies import ALL


async def _run(strategy_name: str, episode: int, page: int) -> None:
    cfg = get_config()
    if not cfg.agent_enabled:
        raise SystemExit("ANTHROPIC_API_KEY is required to drive the agent loop.")
    strategy = ALL[strategy_name]
    governor = Governor(
        max_turns=cfg.max_turns, max_tool_calls=cfg.max_tool_calls, max_usd=cfg.max_usd
    )

    async with RedteamMCPClient(cfg.mcp_server_url) as client:
        probes = await run_strategy(
            strategy=strategy, client=client, governor=governor, episode=episode, page=page
        )

    # TODO: report.build_findings_report(...) → write findings/<run>.md
    #       report.write_candidate_gold(probes, cfg.eval_gold_dir) → *.candidate.yaml
    print(f"{strategy_name}: {len(probes)} probes · stop={governor.stop_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic red-teamer for the Pepper & Carrot companion."
    )
    parser.add_argument("--strategy", choices=sorted(ALL), default="spoiler")
    parser.add_argument("--episode", type=int, default=None, help="target episode (default: config)")
    parser.add_argument("--page", type=int, default=None, help="target page (default: config)")
    args = parser.parse_args()

    cfg = get_config()
    episode = args.episode if args.episode is not None else cfg.target_episode
    page = args.page if args.page is not None else cfg.target_page
    asyncio.run(_run(args.strategy, episode, page))


if __name__ == "__main__":
    main()
