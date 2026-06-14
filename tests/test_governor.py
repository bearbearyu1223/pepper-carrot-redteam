"""The stall detector, pinned deterministically. `stall_patience` is the run-tuning knob now wired
from config (STALL_PATIENCE) — these cases lock its behavior: it fires after N quiet turns, a custom
value moves that threshold, and a confirmed failure resets the count.
"""

from __future__ import annotations

from pepper_carrot_redteam.governor import Governor


def _wide(**kw: int) -> Governor:
    # Caps set high so only the stall path can stop the run.
    return Governor(max_turns=100, max_tool_calls=100, max_usd=100.0, **kw)


def test_stall_fires_after_patience_quiet_turns() -> None:
    g = _wide(stall_patience=2)
    assert g.should_continue()           # 0 quiet turns
    g.tick(found_failure=False)
    assert g.should_continue()           # 1 quiet turn
    g.tick(found_failure=False)
    assert not g.should_continue()       # 2 quiet turns → stall
    assert g.stop_reason is not None and "stalled" in g.stop_reason


def test_custom_stall_patience_extends_the_run() -> None:
    g = _wide(stall_patience=5)
    for _ in range(4):
        g.tick(found_failure=False)
    assert g.should_continue()           # 4 < 5, still going
    g.tick(found_failure=False)
    assert not g.should_continue()       # 5th quiet turn → stall


def test_a_found_failure_resets_the_stall_counter() -> None:
    g = _wide(stall_patience=2)
    g.tick(found_failure=False)
    g.tick(found_failure=True)           # resets the counter
    assert g.should_continue()
    g.tick(found_failure=False)
    assert g.should_continue()           # only 1 quiet turn since the failure
    g.tick(found_failure=False)
    assert not g.should_continue()       # now 2 → stall


def test_default_stall_patience_is_three() -> None:
    g = Governor(max_turns=100, max_tool_calls=100, max_usd=100.0)
    assert g.stall_patience == 3
