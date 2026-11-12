"""Checks the Portfolio Options agent on synthetic prices (no internet, no API key, no price file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents import options, risk  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", "2026-11-06")
RULES = risk.load_rules()
# (theme, exposure, yearly drift, volatility)
ETFS = {
    "WORLD": ("Global equity (benchmark)", 1, 0.10, 0.15),
    "S&P 500": ("US equity", 1, 0.12, 0.17),
    "EURO STOXX 50": ("European equity", 1, 0.08, 0.18),
    "EUROPE 600": ("European equity", 1, -0.05, 0.16),
    "EMMKT": ("Emerging markets", 1, 0.15, 0.20),
    "INDIA": ("Emerging markets", 1, 0.20, 0.22),
    "GOLD": ("Precious metals", 1, 0.18, 0.15),
    "QUANTUM": ("Frontier tech", 1, 0.40, 0.40),
    "URANIUM": ("Energy transition", 1, -0.20, 0.35),
    "NASDAQ 3X": ("Leveraged & inverse", 3, 0.45, 0.70),
    "GLOBAL AGG BOND": ("Fixed income / Bonds", 1, 0.02, 0.05),
    "HIGHY BOND": ("Fixed income / Bonds", 1, 0.04, 0.06),
    "OVERNIGHT CASH": ("Money market (cash)", 1, 0.03, 0.005),
}
UNIVERSE = pd.DataFrame([{"deck_ticker": t, "theme": th, "exposure_factor": x, "yahoo_ticker": t.replace(" ", "") + ".DE"}
                         for t, (th, x, _, _) in ETFS.items()])
rng = np.random.default_rng(1)
PRICES = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(mu / 252, v / np.sqrt(252), len(DAYS))))
                       for t, (_, _, mu, v) in ETFS.items()}, index=DAYS)
STATE = {"as_of": "2026-11-06", "portfolio": {"cash_eur": 10_000, "holdings": {}}}


def run(state=STATE, **kw):
    return options.run_options_agent(state, use_llm=False, prices=PRICES, universe=UNIVERSE, rules=RULES, **kw)


def test_one_option_per_profile_all_passing_risk():
    opts = run()
    assert [o["name"] for o in opts] == list(RULES["profiles"])
    for o in opts:
        assert sum(o["weights"].values()) == pytest.approx(1, abs=1e-3)
        check = risk.check_option(o, RULES, PRICES, UNIVERSE)
        assert check["breaches"] == [], (o["name"], check["breaches"])


def test_defensive_is_less_risky_than_dynamic():
    opts = {o["name"]: o for o in run()}
    assert opts["Defensive"]["metrics"]["volatility_12m"] < opts["Dynamic"]["metrics"]["volatility_12m"]


def test_negative_signal_never_bought_in_risky_tiers():
    signals = {"views": [{"deck_ticker": "QUANTUM", "signal": "negative", "confidence": "high"},
                         {"deck_ticker": "INDIA", "signal": "positive", "confidence": "high"}]}
    opts = run({**STATE, "signals": signals})
    for o in opts:
        assert "QUANTUM" not in o["weights"]
        assert "technical agent" in o["signals_used"].get("INDIA", "technical agent")


def test_orders_turn_cash_into_the_target_portfolio():
    o = run()[1]
    assert o["portfolio_value_eur"] == 10_000
    assert all(x["action"] == "buy" and x["amount_eur"] >= RULES["trading"]["min_order_eur"] for x in o["orders"])
    spent = sum(x["amount_eur"] for x in o["orders"])
    assert spent <= 10_000 * (1 - o["weights"]["CASH"]) + 1


def test_existing_holding_not_in_target_is_sold():
    state = {**STATE, "portfolio": {"cash_eur": 5_000, "holdings": {"URANIUM": 10}}}
    o = run(state)[0]
    assert "URANIUM" not in o["weights"]
    assert o["orders"][0] == {**o["orders"][0], "deck_ticker": "URANIUM", "action": "sell", "units": 10}


def test_rebuild_after_risk_fail_drops_rejected_etf_and_cuts_risk():
    first = run()
    dyn = next(o for o in first if o["name"] == "Dynamic")
    victim = next(t for t in dyn["weights"] if t != "CASH")
    risk_check = {"status": "fail", "options": [{"name": "Dynamic", "status": "fail", "breaches": [
        f"{victim}: excluded by the Data agent", "portfolio volatility 30.0% is above the 25% limit"]}]}
    again = {o["name"]: o for o in run({**STATE, "proposals": first, "risk_check": risk_check})}
    assert victim not in again["Dynamic"]["weights"]
    assert again["Dynamic"]["adjustments"]["risk_scale"] == 0.7
    assert again["Dynamic"]["metrics"]["volatility_12m"] <= dyn["metrics"]["volatility_12m"]
    assert again["Balanced"]["adjustments"] == {"exclude": [], "risk_scale": 1.0}   # untouched


def test_llm_used_once_and_failure_falls_back_to_rule_text():
    class FakeLLM:
        calls = 0

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            FakeLLM.calls += 1
            return options.OptionsNarrative(options=[options.OptionText(name="Balanced", rationale="LLM text",
                                                                        key_risks=["x"])], comparison="cmp")

    opts = options.run_options_agent(STATE, llm=FakeLLM(), prices=PRICES, universe=UNIVERSE, rules=RULES)
    assert FakeLLM.calls == 1
    by = {o["name"]: o for o in opts}
    assert by["Balanced"]["rationale"] == "LLM text" and "no LLM" in by["Defensive"]["rationale"]

    class Broken(FakeLLM):
        def invoke(self, messages):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    opts = options.run_options_agent(STATE, llm=Broken(), prices=PRICES, universe=UNIVERSE, rules=RULES)
    assert all("no LLM" in o["rationale"] for o in opts)
