"""Checks that the agents are connected correctly, using placeholders (no API key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdc.agents.graph import MAX_REVISIONS, build_graph  # noqa: E402


def test_full_workflow_fills_every_agent_output():
    result = build_graph().invoke({"as_of": "2026-11-06"})
    for key in ["data_report", "signals", "macro_view", "proposals", "risk_check", "objections"]:
        assert key in result, f"{key} missing: an agent is not connected"


def test_risk_fail_sends_back_but_stops_after_max_revisions():
    always_fail = lambda state: {"risk_check": {"status": "fail", "breaches": ["test"]}}  # noqa: E731
    result = build_graph({"risk": always_fail}).invoke({"as_of": "2026-11-06"})
    assert result["revisions"] == MAX_REVISIONS
    assert "objections" in result  # still reaches the Challenger
