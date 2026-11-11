"""Checks that the indicators are right, on made-up prices where we know the answer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc import indicators  # noqa: E402

DAYS = pd.bdate_range("2024-01-01", periods=300)


def test_steady_rise_is_uptrend_with_no_drawdown():
    s = pd.Series(np.linspace(100, 160, 300), index=DAYS)
    snap = indicators.snapshot(s)
    assert snap["trend"] == "uptrend"
    assert snap["max_drawdown_12m"] == 0
    assert snap["return_3m"] == pytest.approx(160 / s.iloc[-64] - 1, abs=1e-4)


def test_steady_fall_is_downtrend():
    s = pd.Series(np.linspace(160, 100, 300), index=DAYS)
    assert indicators.trend(s) == "downtrend"


def test_drawdown_of_known_fall():
    s = pd.Series([100, 120, 90, 110], index=DAYS[:4])
    assert indicators.max_drawdown(s) == pytest.approx(90 / 120 - 1)


def test_short_history_is_flagged_not_crashing():
    s = pd.Series(np.linspace(100, 110, 30), index=DAYS[:30])
    assert indicators.snapshot(s)["usable"] is False


def test_benchmark_comparison():
    etf = pd.Series(np.linspace(100, 130, 300), index=DAYS)
    bench = pd.Series(np.linspace(100, 110, 300), index=DAYS)
    assert indicators.snapshot(etf, bench)["return_3m_vs_benchmark"] > 0


def test_ranking_best_first():
    df = pd.DataFrame({"UP": np.linspace(100, 150, 300), "DOWN": np.linspace(150, 100, 300)}, index=DAYS)
    assert list(indicators.momentum_ranking(df).index) == ["UP", "DOWN"]
