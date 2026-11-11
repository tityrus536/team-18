"""Checks the Data agent and the quality checks, without internet or API key."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc import quality  # noqa: E402
from qdc.agents import data_agent  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", "2026-11-06")


# ---------------------------------------------------------------- quality checks

def test_clean_series_is_ok():
    s = pd.Series(np.linspace(100, 150, len(DAYS)), index=DAYS)
    assert quality.check_etf(s, DAYS[-1])["status"] == "ok"


def test_short_history_flagged():
    s = pd.Series(np.nan, index=DAYS)
    s.iloc[-200:] = np.linspace(100, 110, 200)
    assert quality.check_etf(s, DAYS[-1])["status"] == "short_history"


def test_flat_prices_flagged_as_stale():
    s = pd.Series(np.linspace(100, 150, len(DAYS)), index=DAYS)
    s.iloc[-10:] = 150.0
    assert "stale" == quality.check_etf(s, DAYS[-1])["status"]


def test_big_move_allowed_for_leveraged_etf_only():
    s = pd.Series(np.linspace(100, 150, len(DAYS)), index=DAYS)
    s.iloc[-50:] *= 0.75  # one-day fall of 25%
    assert quality.check_etf(s, DAYS[-1], exposure_factor=1)["status"] == "check_moves"
    assert quality.check_etf(s, DAYS[-1], exposure_factor=3)["status"] == "ok"


def test_report_covers_etfs_without_ticker():
    prices = pd.DataFrame({"WORLD": np.linspace(100, 150, len(DAYS))}, index=DAYS)
    universe = pd.DataFrame({"deck_ticker": ["WORLD", "SEMIS"], "yahoo_ticker": ["EUNL.DE", None],
                             "exposure_factor": [1, 1]})
    report = quality.quality_report(prices, universe)
    assert report.loc["SEMIS", "status"] == "no_data"
    assert report.loc["WORLD", "status"] == "ok"


# ---------------------------------------------------------------- ticker finder

ETFS = pd.DataFrame([
    {"deck_ticker": "NASDAQ", "isin": "IE00B53SZB19", "theme": "Equity / Tech",
     "name": "iShares NASDAQ 100 UCITS ETF USD (Acc)"},
    {"deck_ticker": "GHOST", "isin": "XX0000000000", "theme": "Test", "name": "Fund with no listing"},
])


class FakeLLM:
    """Acts like Gemini: records each request and returns fixed choices."""
    def __init__(self, choices):
        self.choices, self.requests = choices, []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        return data_agent.TickerChoices(choices=self.choices)


def fake_collect(etf):
    if etf["deck_ticker"] == "GHOST":
        return []
    return [{"symbol": "SXRV.DE", "name": "iShares NASDAQ 100", "currency": "EUR", "days_of_history": 2500}]


def test_finder_uses_one_request_per_batch_and_skips_etfs_without_listing(monkeypatch, tmp_path):
    monkeypatch.setattr(data_agent, "CANDIDATES_FILE", tmp_path / "candidates.json")
    choice = data_agent.TickerChoice(deck_ticker="NASDAQ", isin="IE00B53SZB19", yahoo_ticker="SXRV.DE",
                                     exchange="XETRA", currency="EUR", confidence="high",
                                     reason="Name matches, EUR listing", alternatives=[])
    fake = FakeLLM([choice])
    out = data_agent.run_ticker_finder(ETFS, llm=fake, pause_seconds=0, collect=fake_collect)

    assert len(fake.requests) == 1                      # one request for the whole batch
    assert "GHOST" not in fake.requests[0][1][1]        # nothing to choose: not sent to the LLM
    assert list(out["deck_ticker"]) == ["NASDAQ", "GHOST"]
    assert out.loc[0, "yahoo_ticker"] == "SXRV.DE"
    assert pd.isna(out.loc[1, "yahoo_ticker"]) and "justETF" in out.loc[1, "reason"]
    assert (tmp_path / "candidates.json").exists()      # candidates saved for human review


def test_llm_error_is_recorded_not_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(data_agent, "CANDIDATES_FILE", tmp_path / "candidates.json")

    class Broken(FakeLLM):
        def invoke(self, messages):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    out = data_agent.run_ticker_finder(ETFS, llm=Broken([]), pause_seconds=0, collect=fake_collect)
    row = out[out.deck_ticker == "NASDAQ"].iloc[0]
    assert row["reason"].startswith("ERROR") and "currency" in out.columns


def test_only_approved_proposals_are_applied():
    universe = pd.DataFrame({"deck_ticker": ["NASDAQ", "GOLD"], "isin": ["IE00B53SZB19", "IE00B4ND3602"],
                             "yahoo_ticker": [None, None], "exchange": [None, None], "currency": [None, None]})
    proposals = pd.DataFrame({"deck_ticker": ["NASDAQ", "GOLD"], "isin": ["IE00B53SZB19", "IE00B4ND3602"],
                              "yahoo_ticker": ["SXRV.DE", "PPFB.DE"], "exchange": ["XETRA", "XETRA"],
                              "currency": ["EUR", "EUR"], "approved": ["yes", ""]})
    new, updated = data_agent.apply_approved(proposals, universe)
    assert updated == ["NASDAQ"]
    assert new.loc[new.deck_ticker == "NASDAQ", "yahoo_ticker"].item() == "SXRV.DE"
    assert pd.isna(new.loc[new.deck_ticker == "GOLD", "yahoo_ticker"].item())
