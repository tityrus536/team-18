"""The team workflow: all agents connected with LangGraph.

    Data -> Technical + Macro (in parallel) -> Portfolio Options -> Risk -> Challenger
                                                  ^                  |
                                                  +--- if "fail" ----+   (max MAX_REVISIONS)

To plug in a real agent: replace its placeholder in AGENTS below. Nothing else changes.
The final step is always human: the team reads the result and decides the orders.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from qdc.agents import placeholders as ph
from qdc.agents.state import TradingState

MAX_REVISIONS = 2

AGENTS = {
    "data": ph.data_placeholder,
    "technical": ph.technical_placeholder,  # real one: qdc.agents.technical.technical_node
    "macro": ph.macro_placeholder,
    "options": ph.options_placeholder,
    "risk": ph.risk_placeholder,
    "challenger": ph.challenger_placeholder,
}


def after_risk(state: TradingState) -> str:
    """Send proposals back to Options if Risk fails them, up to MAX_REVISIONS times."""
    failed = state.get("risk_check", {}).get("status") == "fail"
    return "options" if failed and state.get("revisions", 0) < MAX_REVISIONS else "challenger"


def build_graph(agents: dict | None = None):
    nodes = {**AGENTS, **(agents or {})}
    g = StateGraph(TradingState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "data")
    g.add_edge("data", "technical")
    g.add_edge("data", "macro")
    g.add_edge(["technical", "macro"], "options")  # waits for both
    g.add_edge("options", "risk")
    g.add_conditional_edges("risk", after_risk, ["options", "challenger"])
    g.add_edge("challenger", END)
    return g.compile()
