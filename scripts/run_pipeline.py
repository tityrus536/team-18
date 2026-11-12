"""Run the whole team workflow.

Run from the project root:
    python scripts/run_pipeline.py                     # placeholders only, no API key needed
    python scripts/run_pipeline.py --real technical    # real Technical agent, others placeholders
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdc.agents.graph import build_graph  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--real", nargs="*", default=[], choices=["data", "technical", "risk"],
                        help="agents to run for real")
    args = parser.parse_args()

    real = {}
    if "data" in args.real:
        from qdc.agents.data_agent import data_node
        real["data"] = data_node
    if "technical" in args.real:
        from qdc.agents.technical import technical_node
        real["technical"] = technical_node
    if "risk" in args.real:
        from qdc.agents.risk import risk_node
        real["risk"] = risk_node

    app = build_graph(real)
    result = app.invoke({"as_of": args.as_of, "portfolio": {"cash_eur": 10_000, "holdings": {}}})
    print(json.dumps(result, indent=2, default=str))
