"""The shared state: the one object that travels through every agent.

Each agent reads what it needs and returns ONLY its own key.
Agree any change to this file with the whole team: every agent depends on it.
"""
from __future__ import annotations

from typing import TypedDict


class TradingState(TypedDict, total=False):
    as_of: str              # last date the agents may see, "YYYY-MM-DD"
    portfolio: dict         # current holdings and cash
    data_report: dict       # Data agent: which ETFs are usable
    signals: dict           # Technical agent: view per ETF, with evidence
    macro_view: dict        # Macro/News agent: view per theme, with sources
    proposals: list         # Portfolio Options agent: 2-3 options and draft orders
    risk_check: dict        # Risk agent: "pass", "warning" or "fail", and why
    revisions: int          # how many times Risk sent the proposals back
    objections: list        # Challenger agent: weak points of the proposal
    human_decision: dict    # the team's final orders and reasons
