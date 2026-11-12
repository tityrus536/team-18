"""Checks the Challenger agent on synthetic prices (no internet, no API key, no price file)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents import challenger  # noqa: E402

DAYS = pd.bdate_range("2019-01-01", "2026-11-06")
rng = np.random.default_rng(2)
common = rng.normal(0, 0.012, len(DAYS))
PRICES = pd.DataFrame({
    "HEALTH CARE": 100 * np.exp(np.cumsum(common + rng.normal(0.0003, 0.002, len(DAYS)))),
    "BIOTECH": 100 * np.exp(np.cumsum(common + rng.normal(0.0003, 0.002, len(DAYS)))),   # twin of HEALTH CARE
    "BONDS": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, len(DAYS)))),
    "GOLD": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, len(DAYS)))),
}, index=DAYS)
PRICES.loc["2020-02-19":"2020-03-23", ["HEALTH CARE", "BIOTECH"]] *= np.linspace(1, 0.6, len(
    PRICES.loc["2020-02-19":"2020-03-23"]))[:, None]                                    # a crash in the stress window

TWINS = {"name": "Twins", "weights": {"HEALTH CARE": 0.40, "BIOTECH": 0.40, "CASH": 0.20}}
MIXED = {"name": "Mixed", "weights": {"HEALTH CARE": 0.30, "BONDS": 0.40, "GOLD": 0.30}}
TWINS_COPY = {"name": "Copy", "weights": {"HEALTH CARE": 0.40, "BIOTECH": 0.35, "CASH": 0.25}}


def test_evidence_finds_twins_crisis_and_similar_options():
    ev = challenger.gather_evidence([TWINS, MIXED, TWINS_COPY], PRICES)
    twins, mixed = ev["options"][0], ev["options"][1]
    assert twins["correlated_pairs"][0]["pair"] == ["HEALTH CARE", "BIOTECH"]
    assert mixed["correlated_pairs"] == []
    assert mixed["diversification_ratio"] > twins["diversification_ratio"]
    assert twins["stress_test"]["COVID crash (Feb-Mar 2020)"]["return"] < -0.15
    assert ev["similar_options"] == [{"options": ["Twins", "Copy"], "overlap": 0.95}]


def test_stress_window_after_as_of_is_not_used():
    ev = challenger.gather_evidence([TWINS], PRICES.loc[:"2021-12-31"])
    assert list(ev["options"][0]["stress_test"]) == ["COVID crash (Feb-Mar 2020)"]   # 2022 is the future


def test_weak_signals_counted_from_technical_views():
    signals = {"views": [{"deck_ticker": "BONDS", "signal": "neutral", "confidence": "medium"},
                         {"deck_ticker": "GOLD", "signal": "positive", "confidence": "low"},
                         {"deck_ticker": "HEALTH CARE", "signal": "positive", "confidence": "high"}]}
    ev = challenger.gather_evidence([MIXED], PRICES, signals)
    assert ev["options"][0]["weak_signal_weight"] == 0.70


def test_rule_objections_sorted_by_severity_without_llm():
    out = challenger.run_challenger({"proposals": [TWINS, MIXED, TWINS_COPY]}, use_llm=False, prices=PRICES)
    severities = [o["severity"] for o in out]
    assert severities == sorted(severities, key=challenger.SEVERITY_ORDER.get)
    assert out[0]["severity"] == "high" and "COVID" in out[0]["point"]
    assert any(o["option"] == "all options" and "not a real choice" in o["point"] for o in out)
    assert all(set(o) == {"option", "point", "severity", "evidence", "suggestion"} for o in out)


def test_llm_called_once_and_failure_falls_back_to_rules():
    class FakeLLM:
        calls = 0

        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            FakeLLM.calls += 1
            return challenger.ChallengerReport(objections=[
                challenger.Objection(option="Twins", point="one bet", severity="low", evidence="corr", suggestion="x"),
                challenger.Objection(option="Twins", point="crash", severity="high", evidence="-30%", suggestion="y")])

    out = challenger.run_challenger({"proposals": [TWINS]}, llm=FakeLLM(), prices=PRICES)
    assert FakeLLM.calls == 1 and [o["point"] for o in out] == ["crash", "one bet"]

    class Broken(FakeLLM):
        def invoke(self, messages):
            raise RuntimeError("503 UNAVAILABLE")

    out = challenger.run_challenger({"proposals": [TWINS]}, llm=Broken(), prices=PRICES)
    assert any("move almost together" in o["point"] for o in out)


def test_no_options_gives_a_single_note():
    assert challenger.run_challenger({"proposals": []}, use_llm=False, prices=PRICES)[0]["severity"] == "low"
