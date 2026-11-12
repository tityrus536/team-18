"""Checks the Risk agent on synthetic prices (no internet, no API key, no price file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents import risk  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", "2026-11-06")
RULES = risk.load_rules()  # the real config/rules.yaml
UNIVERSE = pd.DataFrame({
    "deck_ticker": ["WORLD", "BONDS", "QUANTUM", "BITCOIN", "NASDAQ 3X"],
    "theme": ["Global equity (benchmark)", "Fixed income / Bonds", "Frontier tech", "Digital assets",
              "Leveraged & inverse"],
    "exposure_factor": [1, 1, 1, 1, 3],
})


def _prices(vol_by_ticker):
    rng = np.random.default_rng(0)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0.0002, v / np.sqrt(252), len(DAYS))))
                         for t, v in vol_by_ticker.items()}, index=DAYS)


PRICES = _prices({"WORLD": 0.15, "BONDS": 0.05, "QUANTUM": 0.40, "BITCOIN": 0.70, "NASDAQ 3X": 0.80})


def check(weights, prices=PRICES, **kw):
    return risk.check_option({"name": "test", "weights": weights}, RULES, prices, UNIVERSE, **kw)


def test_sensible_portfolio_passes_with_metrics():
    out = check({"WORLD": 0.30, "BONDS": 0.30, "QUANTUM": 0.10, "BITCOIN": 0.10, "CASH": 0.20})
    assert out["breaches"] == []
    assert 0 < out["metrics"]["volatility_12m"] < 0.25 and out["metrics"]["positions"] == 4


def test_position_and_tier_limits_fail():
    out = check({"WORLD": 0.30, "BONDS": 0.30, "QUANTUM": 0.20, "NASDAQ 3X": 0.20})
    text = " ".join(out["breaches"])
    assert out["status"] == "fail"
    assert "QUANTUM: 20.0% is above the 10% limit for High" in text
    assert "NASDAQ 3X: 20.0% is above the 5% limit" in text
    assert "Very high ETFs together" in text


def test_bad_weights_and_unknown_etf_fail():
    out = check({"WORLD": 0.30, "MOONCOIN": 0.30})
    assert any("add up to 60.0%" in b for b in out["breaches"])
    assert any("MOONCOIN: not in the ETF universe" in b for b in out["breaches"])


def test_etf_excluded_by_data_agent_fails():
    out = check({"WORLD": 0.30, "BONDS": 0.30, "CASH": 0.40}, usable=["WORLD"], quality={"BONDS": ["stale"]})
    assert any("BONDS: excluded by the Data agent (stale)" in b for b in out["breaches"])


def test_short_history_and_leverage_are_warnings_not_breaches():
    prices = PRICES.copy()
    prices.loc[: DAYS[-200], "BITCOIN"] = np.nan                      # under 1 year of prices
    out = check({"WORLD": 0.30, "BONDS": 0.30, "BITCOIN": 0.05, "NASDAQ 3X": 0.05, "CASH": 0.30}, prices)
    assert out["breaches"] == [] and out["status"] == "warning"
    w = " ".join(out["warnings"])
    assert "BITCOIN: only 0.8 years" in w and "volatility is understated" in w and "NASDAQ 3X: leveraged" in w


def test_overall_status_is_the_worst_option():
    good = {"name": "ok", "weights": {"WORLD": 0.30, "BONDS": 0.30, "QUANTUM": 0.10, "BITCOIN": 0.10, "CASH": 0.20}}
    bad = {"name": "bad", "weights": {"QUANTUM": 1.0}}
    out = risk.run_risk_check([good, bad], prices=PRICES, universe=UNIVERSE, rules=RULES)
    assert out["status"] == "fail" and all(b.startswith("bad:") for b in out["breaches"])
    assert risk.run_risk_check([], prices=PRICES, universe=UNIVERSE, rules=RULES)["status"] == "fail"
