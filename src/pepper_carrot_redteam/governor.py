"""Budget governor — caps an otherwise-unbounded agentic loop.

An agent loop will happily run forever and spend real money (`ask` is a live generation — see
Post 17). The governor is a hard stop on turns, tool calls, and estimated USD, plus a "no new
failures in N turns" stall detector. `agent.py` must check `should_continue()` each turn and
record every tool call via `charge()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Rough per-call cost estimates (USD) for the governor's ceiling. Tune against real billing.
# `judge` is a cross-model verdict call (oracle.py); the dual spoiler oracle and the hallucination
# oracle each make one per `ask` probe, so the USD ceiling must account for it.
# `translate` is a cheap Haiku transform for the injection `language` option (agent.py).
_COST = {"ask": 0.01, "search": 0.0005, "judge": 0.005, "translate": 0.001}


@dataclass
class Governor:
    max_turns: int
    max_tool_calls: int
    max_usd: float
    stall_patience: int = 3  # turns with no new confirmed failure before stopping (config: STALL_PATIENCE)

    turns: int = 0
    tool_calls: int = 0
    spent_usd: float = 0.0
    _turns_since_failure: int = field(default=0, repr=False)
    stop_reason: str | None = None

    def charge(self, tool: str) -> None:
        """Record one tool call against the budget."""
        self.tool_calls += 1
        self.spent_usd += _COST.get(tool, 0.0)

    def tick(self, *, found_failure: bool) -> None:
        """Advance one agent turn; reset the stall counter if this turn found a failure."""
        self.turns += 1
        self._turns_since_failure = 0 if found_failure else self._turns_since_failure + 1

    def should_continue(self) -> bool:
        """False (and sets stop_reason) when any budget or stall limit is hit."""
        if self.turns >= self.max_turns:
            self.stop_reason = f"max_turns ({self.max_turns})"
        elif self.tool_calls >= self.max_tool_calls:
            self.stop_reason = f"max_tool_calls ({self.max_tool_calls})"
        elif self.spent_usd >= self.max_usd:
            self.stop_reason = f"max_usd (~${self.spent_usd:.2f})"
        elif self._turns_since_failure >= self.stall_patience:
            self.stop_reason = f"stalled ({self.stall_patience} turns, no new failure)"
        return self.stop_reason is None
