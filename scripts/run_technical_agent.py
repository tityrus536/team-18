"""Run the Technical agent and save its report.

Run from the project root:
    python scripts/run_technical_agent.py                  # latest data
    python scripts/run_technical_agent.py --as-of 2026-10-30
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdc.agents.technical import run_technical_agent  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", help="last date the agent may see, YYYY-MM-DD")
    args = parser.parse_args()

    report = run_technical_agent(args.as_of)
    out = ROOT / "outputs" / f"technical_{report.as_of}.json"
    out.write_text(json.dumps(report.model_dump(), indent=2))

    print(report.summary, "\n")
    for v in report.views:
        print(f"{v.deck_ticker:<16} {v.signal:<9} ({v.confidence})  {'; '.join(v.evidence[:2])}")
    if report.data_issues:
        print("\nData issues:", *report.data_issues, sep="\n- ")
    print(f"\nSaved to {out}")
