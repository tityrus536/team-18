"""Step 2: copy the proposals you approved into config/etf_universe.csv.

Run from the project root, after writing "yes" in the "approved" column of
outputs/ticker_proposals.csv:
    python scripts/apply_tickers.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdc.agents.data_agent import apply_approved  # noqa: E402
from qdc.data import UNIVERSE_FILE, load_universe  # noqa: E402

if __name__ == "__main__":
    proposals = pd.read_csv(ROOT / "outputs" / "ticker_proposals.csv")
    universe, updated = apply_approved(proposals, load_universe())
    if not updated:
        print('Nothing approved. Write "yes" in the "approved" column first.')
        sys.exit(0)
    universe.to_csv(UNIVERSE_FILE, index=False)
    print(f"Updated {len(updated)} ETFs: {', '.join(updated)}")
    print("Next: python scripts/update_prices.py, then python scripts/check_data.py")
