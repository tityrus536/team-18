"""Step 1: the Data agent proposes a Yahoo ticker for each ETF that doesn't have one yet.

Run from the project root:
    python scripts/find_tickers.py --limit 3          # test on 3 ETFs first
    python scripts/find_tickers.py                    # all ETFs without a ticker
    python scripts/find_tickers.py --only NASDAQ GOLD # specific ETFs (even if filled)
    python scripts/find_tickers.py --no-llm           # without Gemini: fixed rules choose

Then open outputs/ticker_proposals.csv, check each proposal, write "yes" in the
"approved" column for the ones you accept, and run scripts/apply_tickers.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdc.agents.data_agent import run_ticker_finder  # noqa: E402
from qdc.data import load_universe  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="deck tickers to search, e.g. NASDAQ GOLD")
    parser.add_argument("--limit", type=int, help="stop after this many ETFs (for testing)")
    parser.add_argument("--no-llm", action="store_true",
                        help="choose with fixed rules instead of Gemini (no quota used, less careful on share class)")
    args = parser.parse_args()

    u = load_universe()
    if args.only:
        etfs = u[u["deck_ticker"].isin(args.only)]
    else:
        etfs = u[u["yahoo_ticker"].isna()]
    if args.limit:
        etfs = etfs.head(args.limit)
    then = "rules, no Gemini" if args.no_llm else f"about {-(-len(etfs) // 10)} Gemini requests"
    print(f"Searching tickers for {len(etfs)} ETFs: Yahoo first (no quota used), then {then}.\n")

    proposals = run_ticker_finder(etfs, use_llm=not args.no_llm)
    out = ROOT / "outputs" / "ticker_proposals.csv"
    if out.exists():  # keep earlier proposals, replace the ones searched again
        old = pd.read_csv(out)
        old = old[~old["deck_ticker"].isin(proposals["deck_ticker"])]
        proposals = pd.concat([old, proposals], ignore_index=True)
    proposals.to_csv(out, index=False)

    cols = [c for c in ["deck_ticker", "yahoo_ticker", "currency", "confidence"] if c in proposals]
    print("\n", proposals[cols].to_string(index=False))
    errors = proposals[proposals["reason"].astype(str).str.startswith("ERROR")]
    if len(errors):
        print(f"\n{len(errors)} ETFs failed. First error:\n{errors['reason'].iloc[0][:300]}")
    print(f"\nSaved to {out}")
    print('Next: check each row, write "yes" in the "approved" column, then run scripts/apply_tickers.py')
