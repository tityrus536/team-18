"""Step 1: the Data agent proposes a Yahoo ticker for each ETF that doesn't have one yet.

Run from the project root:
    python scripts/find_tickers.py --limit 3          # test on 3 ETFs first
    python scripts/find_tickers.py                    # all ETFs without a ticker
    python scripts/find_tickers.py --only NASDAQ GOLD # specific ETFs (even if filled)

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
    args = parser.parse_args()

    u = load_universe()
    if args.only:
        etfs = u[u["deck_ticker"].isin(args.only)]
    else:
        etfs = u[u["yahoo_ticker"].isna()]
    if args.limit:
        etfs = etfs.head(args.limit)
    print(f"Searching tickers for {len(etfs)} ETFs. This takes a few seconds per ETF.\n")

    proposals = run_ticker_finder(etfs)
    out = ROOT / "outputs" / "ticker_proposals.csv"
    if out.exists():  # keep earlier proposals, replace the ones searched again
        old = pd.read_csv(out)
        old = old[~old["deck_ticker"].isin(proposals["deck_ticker"])]
        proposals = pd.concat([old, proposals], ignore_index=True)
    proposals.to_csv(out, index=False)

    cols = ["deck_ticker", "yahoo_ticker", "currency", "confidence"]
    print("\n", proposals[cols].to_string(index=False))
    print(f"\nSaved to {out}")
    print('Next: check each row, write "yes" in the "approved" column, then run scripts/apply_tickers.py')
