"""Data agent.

Job 1, with AI (one-off): find the best Yahoo ticker for each ETF from its ISIN.
    The LLM searches Yahoo, inspects the listings and proposes one, with a reason.
    Nothing changes in config/etf_universe.csv until a human approves the proposal.

Job 2, without AI (every day): check the quality of the downloaded prices
    and tell the other agents which ETFs are usable (data_node).
"""
from __future__ import annotations

import time
from typing import Literal

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from qdc.data import UNIVERSE_FILE, load_prices, load_universe
from qdc.quality import quality_report

# ---------------------------------------------------------------- Job 1: find tickers


class TickerChoice(BaseModel):
    deck_ticker: str
    isin: str
    yahoo_ticker: str | None = Field(description="Chosen Yahoo symbol, or null if no listing matches")
    exchange: str | None
    currency: str | None = Field(description="Currency exactly as Yahoo reports it, e.g. EUR, USD, GBp")
    confidence: Literal["high", "medium", "low"]
    reason: str = Field(description="Why this listing: name match, currency, history, volume")
    alternatives: list[str] = Field(description="Other matching symbols, if any")


SYSTEM_PROMPT = """You are the Data agent of a student investment team.
Your job: find the best Yahoo Finance listing for one ETF, given its ISIN and name.

Steps:
1. Call search_yahoo_listings with the ISIN. If nothing useful comes back,
   search with the fund name instead.
2. Call inspect_listing on the promising symbols (at most 4).
3. Choose ONE listing:
   - the name must match the fund (same issuer, same index, same share class:
     Acc/Dist, currency-hedged or not). If unsure, say so.
   - prefer a EUR listing on Xetra (.DE) or Euronext (.AS, .PA, .BR, .MI)
   - then prefer the longest price history and the higher trading volume
   - use a USD or GBP listing only if no good EUR listing exists
4. Never invent a symbol: choose only among symbols returned by your tools.
   If no listing matches, return yahoo_ticker = null with confidence low.
Confidence: high = name clearly matches and data looks complete;
medium = small doubt (share class, short history); low = no clear match."""


@tool
def search_yahoo_listings(query: str) -> list[dict]:
    """Search Yahoo Finance by ISIN or fund name. Returns the listings found (symbol, name, exchange, type)."""
    import yfinance as yf

    try:
        quotes = yf.Search(query, max_results=10, news_count=0).quotes
    except Exception as e:  # network or Yahoo error: tell the agent instead of crashing
        return [{"error": str(e)}]
    return [
        {
            "symbol": q.get("symbol"),
            "name": q.get("longname") or q.get("shortname"),
            "exchange": q.get("exchDisp") or q.get("exchange"),
            "type": q.get("quoteType"),
        }
        for q in quotes
    ]


@tool
def inspect_listing(symbol: str) -> dict:
    """Facts about one Yahoo listing: full name, currency, first and last price date, average daily volume."""
    import yfinance as yf

    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="max", auto_adjust=False)
        info = t.info or {}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    if hist.empty:
        return {"symbol": symbol, "error": "no price history"}
    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency"),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
        "first_date": str(hist.index[0].date()),
        "last_date": str(hist.index[-1].date()),
        "days_of_history": len(hist),
        "avg_volume_3m": int(hist["Volume"].tail(63).mean()) if "Volume" in hist else None,
    }


def find_ticker(etf: pd.Series, llm) -> TickerChoice:
    """Run the agent for one ETF."""
    agent = create_agent(
        llm,
        tools=[search_yahoo_listings, inspect_listing],
        system_prompt=SYSTEM_PROMPT,
        response_format=TickerChoice,
    )
    question = (
        f"Find the Yahoo listing for: {etf['name']}\n"
        f"ISIN: {etf['isin']}\nDeck ticker: {etf['deck_ticker']}\nTheme: {etf['theme']}"
    )
    return agent.invoke({"messages": [("user", question)]})["structured_response"]


def run_ticker_finder(etfs: pd.DataFrame, llm=None, pause_seconds: float = 4.0) -> pd.DataFrame:
    """Propose a ticker for each ETF. Errors on one ETF don't stop the others.

    The pause keeps us under the free Gemini rate limit.
    """
    from qdc.llm import get_llm

    llm = llm or get_llm()
    rows = []
    for i, (_, etf) in enumerate(etfs.iterrows()):
        print(f"[{i + 1}/{len(etfs)}] {etf['deck_ticker']} ({etf['isin']})", flush=True)
        try:
            choice = find_ticker(etf, llm).model_dump()
            choice["alternatives"] = ", ".join(choice["alternatives"])
        except Exception as e:
            choice = {"deck_ticker": etf["deck_ticker"], "isin": etf["isin"], "yahoo_ticker": None,
                      "confidence": "low", "reason": f"ERROR: {e}"}
        rows.append({**choice, "approved": ""})
        if i < len(etfs) - 1:
            time.sleep(pause_seconds)
    return pd.DataFrame(rows)


def apply_approved(proposals: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Copy the human-approved proposals into the universe table. Returns (new table, tickers updated)."""
    ok = proposals["approved"].fillna("").astype(str).str.strip().str.lower().isin(["yes", "y", "x", "1", "oui"])
    approved = proposals[ok & proposals["yahoo_ticker"].notna()]
    u = universe.copy()
    for _, p in approved.iterrows():
        mask = (u["deck_ticker"] == p["deck_ticker"]) & (u["isin"] == p["isin"])
        if not mask.any():
            raise ValueError(f"{p['deck_ticker']} / {p['isin']} is not in the universe")
        u.loc[mask, ["yahoo_ticker", "exchange", "currency"]] = [p["yahoo_ticker"], p["exchange"], p["currency"]]
    return u, approved["deck_ticker"].tolist()


# ---------------------------------------------------------------- Job 2: data quality


def data_node(state: dict) -> dict:
    """The Data agent as a node of the team workflow: which ETFs can the other agents use?"""
    prices = load_prices(state.get("as_of"))
    report = quality_report(prices, load_universe(), as_of=state.get("as_of"))
    usable = report.index[report["status"].isin(["ok", "short_history", "check_moves"])].tolist()
    return {
        "data_report": {
            "as_of": str(prices.index[-1].date()),
            "usable_etfs": usable,
            "excluded": report.index[~report.index.isin(usable)].tolist(),
            "issues": {t: r["issues"] for t, r in report.iterrows() if r["issues"]},
        }
    }


__all__ = ["run_ticker_finder", "apply_approved", "data_node", "UNIVERSE_FILE"]
