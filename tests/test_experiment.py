"""The experiment harness's pure math — cost conversion and the Wilson interval. The grid runner
itself is validated end-to-end offline via `--mock` (no key, no network), not in pytest, since it
monkeypatches the global `anthropic` clients.
"""

from __future__ import annotations

import pytest

from pepper_carrot_redteam.experiment import (
    companion_cost_usd,
    cost_usd,
    parse_positions,
    position_matrix,
    wilson,
)


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


def test_parse_positions_parses_cells() -> None:
    assert parse_positions("3:2,9:5,11:4") == [(3, 2), (9, 5), (11, 4)]
    assert parse_positions(" 3:2 , 11:4 ") == [(3, 2), (11, 4)]  # tolerant of whitespace


def test_parse_positions_rejects_bad_input() -> None:
    for bad in ("3-2", "3:2:1", "abc", "3:"):
        with pytest.raises(SystemExit):
            parse_positions(bad)


def test_position_matrix_empty_for_single_position() -> None:
    records = [{"strategy": "spoiler", "episode": 3, "page": 2, "broke": True}]
    assert position_matrix(records) == ""


def test_position_matrix_reports_break_rate_per_cell() -> None:
    records = [
        {"strategy": "spoiler", "episode": 3, "page": 2, "broke": True},
        {"strategy": "spoiler", "episode": 3, "page": 2, "broke": False},  # (3,2): 1/2 = 0.50
        {"strategy": "spoiler", "episode": 9, "page": 5, "broke": False},  # (9,5): 0/1 = 0.00
    ]
    matrix = position_matrix(records)
    assert "(3,2)" in matrix and "(9,5)" in matrix
    assert "0.50" in matrix and "0.00" in matrix


def test_wilson_handles_zero_and_full() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)  # never broke
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson(10, 10)  # always broke
    assert hi == 1.0 and 0.0 < lo < 1.0
