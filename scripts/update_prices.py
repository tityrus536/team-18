"""Download prices for every ETF with a Yahoo ticker in config/etf_universe.csv.

Run from the project root:  python scripts/update_prices.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdc.data import PRICES_FILE, available_universe, download_prices  # noqa: E402

if __name__ == "__main__":
    u = available_universe()
    print(f"Downloading {len(u)} ETFs: {', '.join(u['deck_ticker'])}")
    prices = download_prices()
    print(f"Saved {prices.shape[1]} ETFs, {prices.index[0].date()} to {prices.index[-1].date()}, in {PRICES_FILE}")
    print("\nFirst date with data per ETF (check short histories):")
    print(prices.apply(lambda s: s.first_valid_index().date() if s.first_valid_index() is not None else None))
