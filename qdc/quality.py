"""Data quality checks, in plain Python. Run after every price download.

Each ETF gets a status:
  ok              usable as is
  short_history   fewer years than MIN_YEARS: careful with backtests
  gaps            too many missing days since its first price
  stale           price not updated recently, or stuck at the same value
  check_moves     a daily move larger than expected: verify against another source
"""
from __future__ import annotations

import pandas as pd

MIN_YEARS = 3.0          # team assumption, not a competition rule
MAX_MISSING = 0.05       # exchanges have different holidays, so a few % is normal
MAX_STALE_RUN = 5        # same price 5 days in a row is suspicious...
STALE_LOOKBACK = 63      # ...in the last 3 months (old Yahoo gaps don't make an ETF unusable today)
MAX_DAYS_SINCE_LAST = 5  # calendar days between the as_of date and the last price
BASE_MOVE_LIMIT = 0.15   # daily move limit for a normal ETF, scaled by leverage


def _longest_flat_run(s: pd.Series) -> int:
    """Longest number of consecutive days with exactly the same price."""
    same = s.diff().eq(0)
    runs = same.groupby((~same).cumsum()).sum()
    return int(runs.max()) + 1 if len(runs) and runs.max() > 0 else 1


def check_etf(prices: pd.Series, as_of: pd.Timestamp, exposure_factor: float = 1.0) -> dict:
    """Quality facts and issues for one ETF's price series."""
    first = prices.first_valid_index()
    if first is None:
        return {"status": "no_data", "issues": ["no prices downloaded"]}

    s = prices.loc[first:]
    valid = s.dropna()
    years = (valid.index[-1] - first).days / 365.25
    move_limit = BASE_MOVE_LIMIT * max(1.0, abs(exposure_factor))
    facts = {
        "first_date": str(first.date()),
        "last_date": str(valid.index[-1].date()),
        "years_of_history": round(years, 1),
        "missing_pct": round(float(s.isna().mean()), 4),
        "longest_flat_run_days": _longest_flat_run(valid),
        "recent_flat_run_days": _longest_flat_run(valid.iloc[-STALE_LOOKBACK:]),
        "max_daily_move": round(float(valid.pct_change().abs().max()), 4),
        "days_since_last_price": int((as_of - valid.index[-1]).days),
    }

    issues = []
    if years < MIN_YEARS:
        issues.append(("short_history", f"only {years:.1f} years of prices"))
    if facts["missing_pct"] > MAX_MISSING:
        issues.append(("gaps", f"{facts['missing_pct']:.1%} of days missing"))
    if facts["recent_flat_run_days"] >= MAX_STALE_RUN:
        issues.append(("stale", f"same price {facts['recent_flat_run_days']} days in a row in the last 3 months"))
    if facts["days_since_last_price"] > MAX_DAYS_SINCE_LAST:
        issues.append(("stale", f"last price is {facts['days_since_last_price']} days old"))
    if facts["max_daily_move"] > move_limit:
        issues.append(("check_moves", f"a daily move of {facts['max_daily_move']:.1%}"))

    facts["status"] = issues[0][0] if issues else "ok"
    facts["issues"] = [text for _, text in issues]
    return facts


def quality_report(prices: pd.DataFrame, universe: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """One row per ETF in the universe, including those with no ticker or no data yet."""
    as_of = pd.Timestamp(as_of) if as_of is not None else prices.index[-1]
    rows = []
    for _, etf in universe.iterrows():
        t = etf["deck_ticker"]
        if t not in prices:
            has_ticker = pd.notna(etf.get("yahoo_ticker")) and str(etf.get("yahoo_ticker")).strip()
            reason = "no prices downloaded" if has_ticker else "no Yahoo ticker yet"
            rows.append({"deck_ticker": t, "status": "no_data", "issues": [reason]})
            continue
        rows.append({"deck_ticker": t, **check_etf(prices[t], as_of, float(etf.get("exposure_factor", 1) or 1))})
    return pd.DataFrame(rows).set_index("deck_ticker")
