"""Run the whole team workflow.

Run from the project root:
    python scripts/run_pipeline.py                                  # placeholders only, no API key needed
    python scripts/run_pipeline.py --real technical                 # real Technical agent, others placeholders
    python scripts/run_pipeline.py --real data options risk --no-llm  # real agents that work without Gemini

The final state is also saved to outputs/run_<as_of>.json for the Decision Log.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdc.agents.graph import build_graph  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--real", nargs="*", default=[], choices=["data", "technical", "options", "risk"],
                        help="agents to run for real")
    parser.add_argument("--no-llm", action="store_true",
                        help="agents that can work without Gemini do so (Options writes its text by rules)")
    args = parser.parse_args()

    real = {}
    if "data" in args.real:
        from qdc.agents.data_agent import data_node
        real["data"] = data_node
    if "technical" in args.real:
        from qdc.agents.technical import technical_node
        real["technical"] = technical_node
    if "options" in args.real:
        from qdc.agents.options import options_node, options_node_no_llm
        real["options"] = options_node_no_llm if args.no_llm else options_node
    if "risk" in args.real:
        from qdc.agents.risk import risk_node
        real["risk"] = risk_node

    app = build_graph(real)
    result = app.invoke({"as_of": args.as_of, "portfolio": {"cash_eur": 10_000, "holdings": {}}})
    text = json.dumps(result, indent=2, default=str)
    print(text)

    as_of = (result.get("risk_check") or {}).get("as_of") or args.as_of or "latest"
    out = ROOT / "outputs" / f"run_{as_of}.json"
    out.write_text(text, encoding="utf-8")
    print(f"\nSaved to {out}", file=sys.stderr)
