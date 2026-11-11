"""Technical agent: reads trend, momentum and risk indicators and gives a view per ETF.

How it works:
1. Python calculates the indicators (qdc.indicators). The LLM never calculates.
2. The LLM gets tools to look at those numbers and decides which ETFs to inspect.
3. It must answer in a fixed format (TechnicalReport), so the next agents can read it.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from qdc import indicators
from qdc.data import BENCHMARK, available_universe, load_prices


class TechnicalView(BaseModel):
    deck_ticker: str = Field(description="ETF label from the Belfius deck, e.g. WORLD")
    signal: Literal["positive", "neutral", "negative"]
    confidence: Literal["low", "medium", "high"]
    evidence: list[str] = Field(description="The numbers behind the signal, quoted from the tools")
    risks: list[str] = Field(description="What could make this view wrong")


class TechnicalReport(BaseModel):
    as_of: str
    views: list[TechnicalView]
    summary: str = Field(description="3-5 sentences for the team")
    data_issues: list[str] = Field(description="ETFs skipped or with doubtful data, and why")


SYSTEM_PROMPT = """You are the Technical Analysis agent of a student investment team.
You analyse ETFs using ONLY the numbers returned by your tools. Never invent a number.

For every ETF available:
- call get_technical_snapshot and base your view on its numbers
- signal: positive (uptrend, positive momentum, beating the benchmark),
  negative (downtrend, negative momentum, deep drawdown), otherwise neutral
- confidence is low when indicators disagree or history is short
- quote the key numbers as evidence (as percentages)
- leveraged or inverse ETFs (exposure_factor not 1): say their daily reset makes
  multi-day returns drift from the stated multiple, and list it as a risk

You do not decide the portfolio. Humans do. You give views and evidence."""


def build_tools(prices: pd.DataFrame, universe: pd.DataFrame):
    """Tools are built around the prices cut at as_of, so the agent cannot see the future."""
    bench = prices[BENCHMARK] if BENCHMARK in prices else None
    info = universe.set_index("deck_ticker")

    @tool
    def list_available_etfs() -> list[dict]:
        """List the ETFs with price data: deck ticker, theme and exposure factor (1 = normal, 3 = 3x leveraged, -2 = 2x inverse)."""
        return [
            {"deck_ticker": t, "theme": info.loc[t, "theme"], "exposure_factor": float(info.loc[t, "exposure_factor"])}
            for t in prices.columns
        ]

    @tool
    def get_technical_snapshot(deck_ticker: str) -> dict:
        """Returns, volatility, drawdown, trend and performance vs MSCI World for one ETF (fractions: 0.05 = 5%)."""
        if deck_ticker not in prices:
            return {"error": f"{deck_ticker} has no price data. Use list_available_etfs."}
        return indicators.snapshot(prices[deck_ticker], None if deck_ticker == BENCHMARK else bench)

    @tool
    def rank_by_momentum(window: Literal["1m", "3m", "6m", "12m"] = "3m") -> dict:
        """All available ETFs ranked by return over the window, best first (fractions)."""
        return {k: round(v, 4) for k, v in indicators.momentum_ranking(prices, window).items()}

    return [list_available_etfs, get_technical_snapshot, rank_by_momentum]


def run_technical_agent(as_of: str | None = None, llm=None) -> TechnicalReport:
    """Run the agent on prices up to as_of and return its structured report."""
    from qdc.llm import get_llm

    prices = load_prices(as_of).dropna(axis=1, how="all")
    as_of = str(prices.index[-1].date())
    agent = create_agent(
        llm or get_llm(),
        tools=build_tools(prices, available_universe()),
        system_prompt=SYSTEM_PROMPT,
        response_format=TechnicalReport,
    )
    result = agent.invoke(
        {"messages": [("user", f"Analyse every available ETF as of {as_of}. Report as_of = {as_of}.")]}
    )
    return result["structured_response"]


def technical_node(state: dict) -> dict:
    """The agent as a node of the team workflow (qdc.agents.graph)."""
    report = run_technical_agent(state.get("as_of"))
    return {"signals": report.model_dump()}
