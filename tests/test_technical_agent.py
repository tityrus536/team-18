"""Runs the Technical agent end to end with a FAKE model (no API key, no internet)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents import technical  # noqa: E402

DAYS = pd.bdate_range("2025-01-01", periods=300)
PRICES = pd.DataFrame({"WORLD": np.linspace(100, 120, 300), "SEMIS": np.linspace(20, 30, 300)}, index=DAYS)
UNIVERSE = pd.DataFrame(
    {"deck_ticker": ["WORLD", "SEMIS"], "theme": ["Global equity", "Equity / Tech"], "exposure_factor": [1, 1]}
)


class FakeLLM:
    """Acts like Gemini: records what it was sent and returns a fixed report."""
    def __init__(self, report):
        self.report, self.requests = report, []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        return self.report


def test_snapshots_contain_real_numbers():
    snaps = {s["deck_ticker"]: s for s in technical.build_snapshots(PRICES, UNIVERSE)}
    assert snaps["SEMIS"]["trend"] == "uptrend"
    assert snaps["SEMIS"]["return_3m_vs_benchmark"] is not None
    assert snaps["SEMIS"]["theme"] == "Equity / Tech"


def test_agent_uses_one_request_and_returns_report(monkeypatch):
    report = technical.TechnicalReport(
        as_of=str(DAYS[-1].date()),
        views=[technical.TechnicalView(deck_ticker="SEMIS", signal="positive", confidence="medium",
                                       evidence=["3m return +6%"], risks=["tech concentration"])],
        summary="SEMIS is in an uptrend.", data_issues=[])
    fake = FakeLLM(report)
    monkeypatch.setattr(technical, "load_prices", lambda as_of=None: PRICES)
    monkeypatch.setattr(technical, "available_universe", lambda: UNIVERSE)

    result = technical.run_technical_agent(llm=fake)
    assert len(fake.requests) == 1  # one request for all ETFs: fits the free quota
    assert "SEMIS" in fake.requests[0][1][1]  # the numbers were sent
    assert result.views[0].signal == "positive"
