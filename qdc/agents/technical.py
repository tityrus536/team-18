"""Technical agent: reads trend, momentum and risk indicators and gives a view per ETF.

How it works:
1. Python calculates the indicators for every ETF (qdc.indicators). No LLM, no quota used.
2. The LLM reads all those numbers in ONE request (or one per batch of 25 ETFs)
   and gives a view per ETF, quoting the numbers as evidence.
3. It must answer in a fixed format (TechnicalReport), so the next agents can read it.

Why not let the LLM call tools one ETF at a time? The free Gemini tier allows
about 20 requests a day: tool calls for 50 ETFs would use it all in one run.
"""
from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from qdc import indicators
from qdc.data import BENCHMARK, available_universe, load_prices

BATCH_SIZE = 25


class TechnicalView(BaseModel):
    deck_ticker: str = Field(description="ETF label from the Belfius deck, e.g. WORLD")
    signal: Literal["positive", "neutral", "negative"]
    confidence: Literal["low", "medium", "high"]
    evidence: list[str] = Field(description="The numbers behind the signal, quoted from the data")
    risks: list[str] = Field(description="What could make this view wrong")


class TechnicalReport(BaseModel):
    as_of: str
    views: list[TechnicalView]
    summary: str = Field(description="3-5 sentences for the team")
    data_issues: list[str] = Field(description="ETFs skipped or with doubtful data, and why")


SYSTEM_PROMPT = """You are the Technical Analysis agent of a student investment team.
You analyse ETFs using ONLY the numbers given to you. Never invent a number.
Returns and volatility are fractions: 0.05 means 5%.

For every ETF given:
- signal: positive (uptrend, positive momentum, beating the benchmark),
  negative (downtrend, negative momentum, deep drawdown), otherwise neutral
- confidence is low when indicators disagree or history is short
- quote the key numbers as evidence, written as percentages
- leveraged or inverse ETFs (exposure_factor not 1): say their daily reset makes
  multi-day returns drift from the stated multiple, and list it as a risk
- ETFs marked usable = false go in data_issues, not in views

You do not decide the portfolio. Humans do. You give views and evidence."""


def build_snapshots(prices: pd.DataFrame, universe: pd.DataFrame) -> list[dict]:
    """All indicators for every ETF with prices, ready to send to the LLM."""
    bench = prices[BENCHMARK] if BENCHMARK in prices else None
    info = universe.set_index("deck_ticker")
    out = []
    for t in prices.columns:
        snap = indicators.snapshot(prices[t], None if t == BENCHMARK else bench)
        meta = {"theme": info.loc[t, "theme"], "exposure_factor": float(info.loc[t, "exposure_factor"])} \
            if t in info.index else {}
        out.append({"deck_ticker": t, **meta, **snap})
    return out


def analyse(snapshots: list[dict], as_of: str, llm) -> TechnicalReport:
    """One LLM request for a list of ETF snapshots."""
    data = json.dumps(snapshots, indent=1, default=str)
    return llm.with_structured_output(TechnicalReport).invoke([
        ("system", SYSTEM_PROMPT),
        ("user", f"Indicators as of {as_of} (benchmark: {BENCHMARK} = MSCI World). "
                 f"Report as_of = {as_of}.\n{data}"),
    ])


def run_technical_agent(as_of: str | None = None, llm=None) -> TechnicalReport:
    """Run the agent on prices up to as_of and return its structured report."""
    from qdc.llm import get_llm

    llm = llm or get_llm()
    prices = load_prices(as_of).dropna(axis=1, how="all")
    as_of = str(prices.index[-1].date())
    snaps = build_snapshots(prices, available_universe())

    reports = [analyse(snaps[i:i + BATCH_SIZE], as_of, llm) for i in range(0, len(snaps), BATCH_SIZE)]
    if len(reports) == 1:
        return reports[0]
    return TechnicalReport(  # merge batches
        as_of=as_of,
        views=[v for r in reports for v in r.views],
        summary=" ".join(r.summary for r in reports),
        data_issues=[d for r in reports for d in r.data_issues],
    )


def technical_node(state: dict) -> dict:
    """The agent as a node of the team workflow (qdc.agents.graph)."""
    report = run_technical_agent(state.get("as_of"))
    return {"signals": report.model_dump()}
