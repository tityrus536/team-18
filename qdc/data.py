"""Data pipeline: load the ETF universe, download prices, convert to EUR, save and reload.

Rule: agents never download data themselves. They read the saved price file,
cut at an `as_of` date, so every run can be reproduced and never sees the future.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_FILE = ROOT / "config" / "etf_universe.csv"
DATA_DIR = ROOT / "data"
PRICES_FILE = DATA_DIR / "prices_eur.parquet"
BENCHMARK = "WORLD"  # MSCI World, the benchmark named by Belfius

# Yahoo FX tickers: price of 1 EUR in that currency
FX_TICKERS = {"USD": "EURUSD=X", "GBP": "EURGBP=X"}


def load_universe(path: Path = UNIVERSE_FILE) -> pd.DataFrame:
    """The 50 ETFs from the Belfius deck, with the trading tickers filled in so far."""
    return pd.read_csv(path)


def available_universe(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Only the ETFs whose Yahoo ticker has been filled in."""
    u = load_universe() if universe is None else universe
    return u[u["yahoo_ticker"].notna() & (u["yahoo_ticker"].astype(str).str.strip() != "")]


def _field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pick one price field from a yfinance download (columns are field x ticker)."""
    if isinstance(data.columns, pd.MultiIndex):
        return data[field]
    return data[[field]]


def download_prices(start: str = "2015-01-01") -> pd.DataFrame:
    """Download adjusted closes for all filled tickers, convert to EUR and save.

    Returns a DataFrame: one column per deck ticker (e.g. "WORLD"), EUR prices.
    """
    import yfinance as yf

    u = available_universe()
    tickers = u["yahoo_ticker"].tolist()
    DATA_DIR.mkdir(exist_ok=True)

    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    raw.to_parquet(DATA_DIR / f"raw_prices_{pd.Timestamp.today().date()}.parquet")

    adj = _field(raw, "Adj Close").dropna(how="all")
    fx_raw = yf.download(list(FX_TICKERS.values()), start=start, auto_adjust=False, progress=False)
    fx = _field(fx_raw, "Close").reindex(adj.index).ffill(limit=3)  # past rates only

    eur = pd.DataFrame(index=adj.index)
    for _, row in u.iterrows():
        series = adj[row["yahoo_ticker"]]
        ccy = str(row["currency"]).strip()
        if ccy == "GBp":  # London prices in pence
            series, ccy = series / 100, "GBP"
        if ccy in FX_TICKERS:
            series = series / fx[FX_TICKERS[ccy]]
        elif ccy != "EUR":
            raise ValueError(f"{row['deck_ticker']}: unknown currency {ccy!r}")
        eur[row["deck_ticker"]] = series

    eur.to_parquet(PRICES_FILE)
    return eur


def load_prices(as_of: str | None = None, path: Path = PRICES_FILE) -> pd.DataFrame:
    """Saved EUR prices, cut at `as_of` (inclusive) so nothing after that date is visible."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/update_prices.py")
    prices = pd.read_parquet(path).sort_index()
    if as_of is not None:
        prices = prices.loc[: pd.Timestamp(as_of)]
    return prices
