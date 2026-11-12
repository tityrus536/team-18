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


SPLIT_RATIO = 5.0     # a one-day price ratio this large is a share split, not a market move
SPIKE_MOVE = 0.40     # a one-day move this large that reverses the next day is a bad tick
SPIKE_REVERT = 0.15   # "reverses" = next price within 15% of the price before the spike


def clean_prices(s: pd.Series) -> tuple[pd.Series, list[str]]:
    """Fix the two Yahoo errors that break volatility and drawdowns. Returns (series, notes).

    - unadjusted split: the price jumps by SPLIT_RATIO or more and stays there,
      so all earlier prices are rescaled to the new level
    - bad tick: one price far from both its neighbours, so it is removed
    Real crashes (e.g. -64% on a 3x ETF) are below SPLIT_RATIO and are kept.
    """
    s = s.copy()
    notes = []
    valid = s.dropna()
    for i in range(1, len(valid) - 1):
        prev, cur, nxt = valid.iloc[i - 1], valid.iloc[i], valid.iloc[i + 1]
        if abs(cur / prev - 1) > SPIKE_MOVE and abs(nxt / prev - 1) < SPIKE_REVERT:
            s.loc[valid.index[i]] = float("nan")
            notes.append(f"removed bad price {cur:.4g} on {valid.index[i].date()}")
    valid = s.dropna()
    ratio = valid / valid.shift(1)
    for day, r in ratio[(ratio >= SPLIT_RATIO) | (ratio <= 1 / SPLIT_RATIO)].items():
        s.loc[:day - pd.Timedelta(days=1)] *= r
        notes.append(f"adjusted split (x{1 / r:.4g}) on {day.date()}")
    return s, notes


def download_prices(start: str = "2015-01-01") -> pd.DataFrame:
    """Download adjusted closes for all filled tickers, convert to EUR and save.

    Returns a DataFrame: one column per deck ticker (e.g. "WORLD"), EUR prices.
    """
    import yfinance as yf

    u = available_universe()
    tickers = u["yahoo_ticker"].tolist()
    DATA_DIR.mkdir(exist_ok=True)

    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    adj = _field(raw, "Adj Close").copy()
    # Yahoo's batch download sometimes drops a ticker at random: retry those one by one
    for t in [t for t in tickers if t not in adj or adj[t].isna().all()]:
        print(f"{t}: empty in batch download, retrying alone")
        one = yf.download(t, start=start, auto_adjust=False, progress=False, multi_level_index=False)
        if not one.empty:
            adj[t] = one["Adj Close"]
    adj.to_parquet(DATA_DIR / f"raw_prices_{pd.Timestamp.today().date()}.parquet")

    adj = adj.dropna(how="all")
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
        series, notes = clean_prices(series)
        for note in notes:
            print(f"{row['deck_ticker']}: {note}")
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
