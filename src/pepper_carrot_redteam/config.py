"""Red-teamer configuration from environment / `.env`.

Mirrors the eval's config style. The MCP server URL is the only thing the *client* needs;
the agent loop and the fuzzy judge additionally need `ANTHROPIC_API_KEY`. The budget fields
exist because an agentic loop is unbounded by default — see governor.py.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The deployed pepper-carrot-mcp server (Streamable HTTP) — same one the eval/Claude use.
    mcp_server_url: str = "https://pepper-carrot-mcp.fly.dev/mcp"

    # The agent loop (tool use) and the fuzzy-verdict judge.
    anthropic_api_key: str | None = None
    agent_model: str = "claude-opus-4-8"     # drives the probes
    judge_model: str = "claude-sonnet-4-6"   # cross-model judge for fuzzy verdicts
    translate_model: str = "claude-haiku-4-5"  # cheap transform for the injection `language` option

    # Budget governor — caps on an otherwise-unbounded loop. See governor.py.
    max_turns: int = 12
    max_tool_calls: int = 25
    max_usd: float = 0.50
    stall_patience: int = 3  # stop after this many turns with no new confirmed failure

    # Default reading position the spoiler strategy attacks. The server enforces the real
    # boundary; this is just where we point the probes.
    target_episode: int = 2
    target_page: int = 3

    # Where confirmed failures are written as candidate gold (a pepper-carrot-eval/data dir).
    eval_gold_dir: str = "../pepper-carrot-eval/data"

    @property
    def agent_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


_config: Config | None = None


def get_config() -> Config:
    """Singleton accessor — use everywhere instead of constructing `Config()`."""
    global _config
    if _config is None:
        _config = Config()
    return _config
