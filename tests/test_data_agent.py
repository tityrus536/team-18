"""Checks the Data agent and the quality checks, without internet or API key."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc import quality  # noqa: E402
from qdc.agents import data_agent  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", "2026-11-06")


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


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

ETF = pd.Series({"deck_ticker": "NASDAQ", "isin": "IE00B53SZB19", "theme": "Equity / Tech",
                 "name": "iShares NASDAQ 100 UCITS ETF USD (Acc)"})


def test_agent_returns_a_ticker_choice(monkeypatch):
    calls = []
    monkeypatch.setattr(data_agent.search_yahoo_listings, "func",
                        lambda query: calls.append(query) or [{"symbol": "SXRV.DE", "name": "iShares NASDAQ 100"}])
    choice = {"deck_ticker": "NASDAQ", "isin": "IE00B53SZB19", "yahoo_ticker": "SXRV.DE", "exchange": "XETRA",
              "currency": "EUR", "confidence": "high", "reason": "Name matches, EUR listing", "alternatives": []}
    fake = FakeToolModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "search_yahoo_listings", "args": {"query": "IE00B53SZB19"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "TickerChoice", "args": choice, "id": "2"}]),
    ]))
    result = data_agent.find_ticker(ETF, fake)
    assert calls == ["IE00B53SZB19"]
    assert result.yahoo_ticker == "SXRV.DE"


def test_one_failing_etf_does_not_stop_the_others(monkeypatch):
    def flaky(etf, llm):
        if etf["deck_ticker"] == "BAD":
            raise RuntimeError("503 high demand")
        return data_agent.TickerChoice(deck_ticker=etf["deck_ticker"], isin=etf["isin"], yahoo_ticker="X.DE",
                                       exchange="XETRA", currency="EUR", confidence="high", reason="ok",
                                       alternatives=[])
    monkeypatch.setattr(data_agent, "find_ticker", flaky)
    etfs = pd.DataFrame([{**ETF, "deck_ticker": "BAD"}, ETF])
    out = data_agent.run_ticker_finder(etfs, llm=object(), pause_seconds=0)
    assert out.loc[0, "reason"].startswith("ERROR") and out.loc[1, "yahoo_ticker"] == "X.DE"


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
