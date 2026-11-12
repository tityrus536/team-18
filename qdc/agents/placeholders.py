"""Placeholder agents: they return fake results in the right format.

They let us run the whole workflow before every agent exists. Each owner replaces
their placeholder with the real agent, keeping the same input (state) and output key.
"""
from __future__ import annotations


def data_placeholder(state: dict) -> dict:
    return {"data_report": {"status": "placeholder", "usable_etfs": [], "issues": []}}


def technical_placeholder(state: dict) -> dict:
    return {"signals": {"status": "placeholder", "as_of": state.get("as_of"), "views": [], "summary": ""}}


def macro_placeholder(state: dict) -> dict:
    return {"macro_view": {"status": "placeholder", "themes": [], "sources": []}}


def options_placeholder(state: dict) -> dict:
    # Format the real agent must follow: target weights (fractions adding up to 1, "CASH" allowed).
    # These weights are an example to exercise the Risk agent, not advice.
    return {
        "proposals": [{"name": "placeholder balanced", "orders": [], "rationale": "",
                       "weights": {"WORLD": 0.30, "S&P 500": 0.15, "EURO STOXX 50": 0.15,
                                   "EMMKT": 0.10, "GLOBAL AGG BOND": 0.20, "CASH": 0.10}}],
        "revisions": state.get("revisions", -1) + 1,  # 0 on the first pass, +1 each time Risk sends it back
    }


def risk_placeholder(state: dict) -> dict:
    return {"risk_check": {"status": "pass", "breaches": [], "note": "placeholder"}}


def challenger_placeholder(state: dict) -> dict:
    return {"objections": [{"point": "placeholder", "severity": "low"}]}
