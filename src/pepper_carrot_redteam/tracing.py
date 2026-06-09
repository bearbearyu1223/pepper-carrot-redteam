"""A dependency-free JSONL tracer for the red-teamer's LLM + tool calls (DESIGN §11).

Near-verbatim port of `pepper_carrot_eval.tracing`, with one adaptation: the trace unit is a
**probe**, not a gold item. Every `agent` / `search` / `ask` / judge interaction appends one JSON
record to `traces/<run-id>.jsonl` — a timestamp, the run id, the strategy/turn/session it belongs
to (read from a contextvar the loop sets per turn), the component, and the input/output (including
the agent's reasoning). Greppable with `jq`, diffable, persistent, no external dependency:

    jq 'select(.session_id=="…")' traces/run-*.jsonl   # one social-engineering arc, end to end

The report is human triage; the trace is the full forensic record. Correlation works because
`set_probe()` writes the current strategy/turn/session into a `ContextVar`, and `asyncio.to_thread`
copies that context into the judge thread, so a judge verdict traces back to the probe that
triggered it.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Default is None (not a mutable {}), per ruff B039; readers coalesce to {}.
_ctx: ContextVar[dict[str, Any] | None] = ContextVar("trace_ctx", default=None)


class Tracer:
    """Appends one JSON object per interaction to a run-stamped JSONL file."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.run_id = run_id
        self.path = path
        self._lock = threading.Lock()  # judges run in a worker thread (asyncio.to_thread)
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, component: str, **fields: Any) -> None:
        ctx = _ctx.get() or {}
        record: dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "strategy": ctx.get("strategy"),
            "turn": ctx.get("turn"),
            "session_id": ctx.get("session_id"),
            "component": component,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


_active: Tracer | None = None


def set_tracer(tracer: Tracer | None) -> None:
    """Install (or clear) the process-wide active tracer."""
    global _active
    _active = tracer


def get_tracer() -> Tracer | None:
    return _active


def record(*, component: str, **fields: Any) -> None:
    """Record one interaction — a no-op when tracing is disabled."""
    if _active is not None:
        _active.record(component=component, **fields)


def set_probe(strategy: str, turn: int, session_id: str | None) -> None:
    """Tag subsequent records with the strategy + turn + session (call at each loop top).

    Uses a bare `ContextVar.set` (no reset) — each turn overwrites it, and the value is copied into
    `asyncio.to_thread` workers, so a judge's records inherit the right probe context.
    """
    _ctx.set({"strategy": strategy, "turn": turn, "session_id": session_id})


@contextlib.contextmanager
def trace_context(**fields: Any) -> Iterator[None]:
    """Scoped variant of `set_probe` for nested/explicit blocks."""
    token = _ctx.set({**(_ctx.get() or {}), **fields})
    try:
        yield
    finally:
        _ctx.reset(token)
