"""The experiment harness's pure math — cost conversion and the Wilson interval. The grid runner
itself is validated end-to-end offline via `--mock` (no key, no network), not in pytest, since it
monkeypatches the global `anthropic` clients.
"""

from __future__ import annotations

from pepper_carrot_redteam.experiment import companion_cost_usd, cost_usd, wilson


def test_cost_usd_prices_each_token_class() -> None:
    # 1M input + 1M output of Opus at the table's 15/75 → $90; plus a Sonnet cache read at 0.3.
    tokens = {
        "claude-opus-4-8": {"input": 1_000_000, "output": 1_000_000, "cache_read": 0, "cache_write": 0},
        "claude-sonnet-4-6": {"input": 0, "output": 0, "cache_read": 1_000_000, "cache_write": 0},
    }
    assert cost_usd(tokens) == 90.0 + 0.3


def test_cost_usd_charges_cache_writes_above_input() -> None:
    # cache_write is priced at 1.25x the input rate.
    one = cost_usd({"claude-opus-4-8": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 1_000_000}})
    assert one == 15.0 * 1.25


def test_cost_usd_unknown_model_uses_default() -> None:
    tokens = {"mystery-model": {"input": 1_000_000, "output": 0, "cache_read": 0, "cache_write": 0}}
    assert cost_usd(tokens) == 5.0


def test_companion_cost_counts_ask_and_search() -> None:
    # The same-account server-side cost the client can't meter: counted calls x per-call estimate.
    prices = {"ask": 0.01, "search": 0.0005}
    assert companion_cost_usd({"ask": 10, "search": 4}, prices) == 10 * 0.01 + 4 * 0.0005


def test_wilson_interval_brackets_the_point_estimate() -> None:
    lo, hi = wilson(1, 20)  # one break in twenty runs
    assert lo < 0.05 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_handles_zero_and_full() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)  # never broke
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson(10, 10)  # always broke
    assert hi == 1.0 and 0.0 < lo < 1.0
