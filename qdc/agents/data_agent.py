"""Data agent.

Job 1, with AI (one-off): find the best Yahoo ticker for each ETF from its ISIN.
    1. Python searches Yahoo and collects the facts on every candidate listing
       (no LLM, so no quota used).
    2. The LLM compares the candidates and chooses, for 10 ETFs per request.
       45 ETFs = about 5 requests, which fits the free Gemini quota.
    3. Nothing changes in config/etf_universe.csv until a human approves.

Job 2, without AI (every day): check the quality of the downloaded prices
    and tell the other agents which ETFs are usable (data_node).
"""
from __future__ import annotations

import json
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from qdc.data import ROOT, load_prices, load_universe
from qdc.quality import quality_report

CANDIDATES_FILE = ROOT / "outputs" / "ticker_candidates.json"
MAX_CANDIDATES = 5

# ---------------------------------------------------------------- Job 1a: collect candidates (Python)


def search_yahoo(query: str) -> list[dict]:
    """Listings Yahoo returns for an ISIN or a fund name."""
    import yfinance as yf

    quotes = yf.Search(query, max_results=10, news_count=0).quotes
    return [
        {"symbol": q.get("symbol"), "name": q.get("longname") or q.get("shortname"),
         "exchange": q.get("exchDisp") or q.get("exchange"), "type": q.get("quoteType")}
        for q in quotes if q.get("symbol")
    ]


def inspect_listing(symbol: str) -> dict:
    """Facts about one listing: name, currency, price history, volume."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    hist = t.history(period="max", auto_adjust=False)
    if hist.empty:
        return {"symbol": symbol, "error": "no price history"}
    try:
        info = t.info or {}
    except Exception:
        info = {}
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


def collect_candidates(etf: pd.Series, pause_seconds: float = 0.5) -> list[dict]:
    """Search by ISIN (then by name if nothing found) and inspect up to MAX_CANDIDATES listings."""
    try:
        found = search_yahoo(etf["isin"]) or search_yahoo(etf["name"])
    except Exception as e:
        return [{"error": f"Yahoo search failed: {e}"}]
    candidates = []
    for listing in found[:MAX_CANDIDATES]:
        try:
            candidates.append({**listing, **inspect_listing(listing["symbol"])})
        except Exception as e:
            candidates.append({**listing, "error": str(e)})
        time.sleep(pause_seconds)
    return candidates


# ---------------------------------------------------------------- Job 1b: choose (LLM)


class TickerChoice(BaseModel):
    deck_ticker: str
    isin: str
    yahoo_ticker: str | None = Field(description="Chosen symbol from the candidates, or null if none matches")
    exchange: str | None
    currency: str | None = Field(description="Currency exactly as in the candidate facts, e.g. EUR, USD, GBp")
    confidence: Literal["high", "medium", "low"]
    reason: str = Field(description="Why this listing: name match, currency, history, volume")
    alternatives: list[str] = Field(description="Other candidate symbols that also match the fund")


class TickerChoices(BaseModel):
    choices: list[TickerChoice]


SYSTEM_PROMPT = """You are the Data agent of a student investment team.
For each ETF below, choose the best Yahoo Finance listing among its candidates.

Rules:
- The candidate's name must match the fund: same issuer, same index, same share
  class (Acc/Dist, currency-hedged or not). If the candidates' names don't match
  the fund, return yahoo_ticker = null.
- Prefer a EUR listing on Xetra (.DE) or Euronext (.AS, .PA, .BR, .MI).
- Then prefer the longest price history and the higher average volume.
- Use a USD or GBP listing only if no good EUR listing exists.
- Choose ONLY among the candidate symbols given. Never invent a symbol.
- Copy currency and exchange from the chosen candidate's facts.
Confidence: high = name clearly matches and data looks complete;
medium = small doubt (share class, short history, low volume); low = no clear match.
Return exactly one choice per ETF, in the same order."""


def choose_listings(batch: list[dict], llm) -> list[TickerChoice]:
    """One LLM request for a batch of ETFs and their candidates."""
    prompt = json.dumps(batch, indent=1, default=str)
    result = llm.with_structured_output(TickerChoices).invoke(
        [("system", SYSTEM_PROMPT), ("user", f"ETFs and their candidate listings:\n{prompt}")]
    )
    return result.choices


def _error_row(etf: pd.Series, reason: str) -> dict:
    return {"deck_ticker": etf["deck_ticker"], "isin": etf["isin"], "yahoo_ticker": None, "exchange": None,
            "currency": None, "confidence": "low", "reason": reason, "alternatives": ""}


def run_ticker_finder(etfs: pd.DataFrame, llm=None, batch_size: int = 10,
                      pause_seconds: float = 5.0, collect=collect_candidates) -> pd.DataFrame:
    """Propose a ticker for each ETF, using about len(etfs) / batch_size LLM requests."""
    from qdc.llm import get_llm

    # 1. Python: collect candidates for every ETF (no LLM)
    items = []
    for i, (_, etf) in enumerate(etfs.iterrows()):
        print(f"[{i + 1}/{len(etfs)}] searching Yahoo for {etf['deck_ticker']} ({etf['isin']})", flush=True)
        items.append((etf, collect(etf)))
    CANDIDATES_FILE.parent.mkdir(exist_ok=True)
    CANDIDATES_FILE.write_text(json.dumps(
        {etf["deck_ticker"]: c for etf, c in items}, indent=1, default=str))

    # 2. LLM: choose, one request per batch
    llm = llm or get_llm()
    rows = []
    to_ask = []
    for etf, c in items:
        if any("symbol" in x and "error" not in x for x in c):
            to_ask.append((etf, c))
        else:
            rows.append(_error_row(etf, "no usable Yahoo listing found: check justETF by hand"))

    batches = [to_ask[i:i + batch_size] for i in range(0, len(to_ask), batch_size)]
    for b, batch in enumerate(batches):
        print(f"Asking the LLM to choose for batch {b + 1}/{len(batches)} ({len(batch)} ETFs)", flush=True)
        payload = [{"deck_ticker": e["deck_ticker"], "isin": e["isin"], "fund_name": e["name"],
                    "candidates": c} for e, c in batch]
        try:
            choices = {ch.deck_ticker: ch for ch in choose_listings(payload, llm)}
        except Exception as e:
            rows += [_error_row(etf, f"ERROR: {e}") for etf, _ in batch]
        else:
            for etf, _ in batch:
                ch = choices.get(etf["deck_ticker"])
                if ch is None:
                    rows.append(_error_row(etf, "ERROR: the LLM skipped this ETF"))
                else:
                    rows.append({**ch.model_dump(), "alternatives": ", ".join(ch.alternatives)})
        if b < len(batches) - 1:
            time.sleep(pause_seconds)

    order = {t: i for i, t in enumerate(etfs["deck_ticker"])}
    out = pd.DataFrame(rows).sort_values("deck_ticker", key=lambda s: s.map(order)).reset_index(drop=True)
    out["approved"] = ""
    return out


def apply_approved(proposals: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Copy the human-approved proposals into the universe table. Returns (new table, tickers updated)."""
    ok = proposals["approved"].fillna("").astype(str).str.strip().str.lower().isin(["yes", "y", "x", "1", "oui"])
    approved = proposals[ok & proposals["yahoo_ticker"].notna()]
    u = universe.copy()
    for col in ["yahoo_ticker", "exchange", "currency"]:
        u[col] = u[col].astype(object)
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
