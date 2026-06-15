"""The analysis module's pure pieces: loading saved runs, the two-proportion test, and that the
report/ablation render the expected Break Rates. No network, no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pepper_carrot_redteam.analysis import ablation, load_runs, report, two_proportion_z


def _run(strategy: str, broke: bool, *, ep: int = 3, pg: int = 2, **extra: Any) -> dict[str, Any]:
    base = {
        "strategy": strategy, "episode": ep, "page": pg, "broke": broke,
        "n_probes": 5, "n_confirmed": 1 if broke else 0, "n_critical": 1 if broke else 0,
        "first_fail_turn": 3 if broke else None,
        "client_cost_usd": 0.04, "companion_cost_usd": 0.004, "cost_usd": 0.044,
    }
    base.update(extra)
    return base


def test_load_runs_reads_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "runs.jsonl"
    f.write_text("\n".join(json.dumps(_run("spoiler", b)) for b in (True, False)) + "\n")
    rows = load_runs([str(f)])
    assert len(rows) == 2
    # a directory resolves to its runs.jsonl
    assert len(load_runs([str(tmp_path)])) == 2


def test_two_proportion_z_zero_when_equal() -> None:
    z, p = two_proportion_z(5, 10, 5, 10)
    assert z == 0.0 and p == 1.0


def test_two_proportion_z_flags_a_real_difference() -> None:
    # 2/30 vs 18/30 is a large, significant gap.
    z, p = two_proportion_z(2, 30, 18, 30)
    assert abs(z) > 3 and p < 0.01


def test_report_shows_break_rate_per_strategy() -> None:
    rows = [_run("spoiler", True), _run("spoiler", False), _run("blindspot", False)]
    out = report(rows, n_files=1)
    assert "Break Rate by strategy" in out
    assert "spoiler" in out and "blindspot" in out
    assert "0.50" in out  # spoiler broke 1/2


def test_ablation_reports_delta_and_significance() -> None:
    a = [_run("spoiler", False) for _ in range(30)]
    b = [_run("spoiler", True) for _ in range(30)]  # 0% vs 100%
    out = ablation(a, b, label_a="A", label_b="B")
    assert "Ablation" in out and "OVERALL" in out
    assert "*" in out  # significant
