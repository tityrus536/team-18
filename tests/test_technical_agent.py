"""Runs the Technical agent end to end with a FAKE model (no API key, no internet).

The fake model behaves like an LLM that calls one tool, then gives its final answer.
It checks that tools, prompt and output format are wired correctly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents import technical  # noqa: E402

DAYS = pd.bdate_range("2025-01-01", periods=300)
PRICES = pd.DataFrame({"WORLD": np.linspace(100, 120, 300), "SEMIS": np.linspace(20, 30, 300)}, index=DAYS)
UNIVERSE = pd.DataFrame(
    {"deck_ticker": ["WORLD", "SEMIS"], "theme": ["Global equity", "Equity / Tech"], "exposure_factor": [1, 1]}
)


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_tools_return_real_numbers():
    tools = {t.name: t for t in technical.build_tools(PRICES, UNIVERSE)}
    snap = tools["get_technical_snapshot"].invoke({"deck_ticker": "SEMIS"})
    assert snap["usable"] and snap["trend"] == "uptrend"
    assert "error" in tools["get_technical_snapshot"].invoke({"deck_ticker": "NOPE"})


def test_agent_returns_structured_report(monkeypatch):
    report = {
        "as_of": str(DAYS[-1].date()),
        "views": [{"deck_ticker": "SEMIS", "signal": "positive", "confidence": "medium",
                   "evidence": ["3m return +6%"], "risks": ["tech concentration"]}],
        "summary": "SEMIS is in an uptrend.",
        "data_issues": [],
    }
    fake = FakeToolModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "get_technical_snapshot", "args": {"deck_ticker": "SEMIS"}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "TechnicalReport", "args": report, "id": "2"}]),
    ]))
    monkeypatch.setattr(technical, "load_prices", lambda as_of=None: PRICES)
    monkeypatch.setattr(technical, "available_universe", lambda: UNIVERSE)

    result = technical.run_technical_agent(llm=fake)
    assert isinstance(result, technical.TechnicalReport)
    assert result.views[0].signal == "positive"
