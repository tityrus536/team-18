"""Technical indicators, in plain Python. The LLM never calculates: it calls these.

Every function only uses the prices it is given, so cutting prices at an `as_of`
date (see qdc.data.load_prices) is enough to avoid look-ahead bias.
"""
from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS = 252
WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def _ret(s: pd.Series, days: int) -> float | None:
    return float(s.iloc[-1] / s.iloc[-1 - days] - 1) if len(s) > days else None


def max_drawdown(s: pd.Series) -> float:
    """Largest fall from a previous peak, as a negative fraction (e.g. -0.25)."""
    return float((s / s.cummax() - 1).min())


def trend(s: pd.Series) -> str:
    """'uptrend' if price > 50-day average > 200-day average, 'downtrend' if the reverse."""
    if len(s) < 200:
        return "not enough history"
    price, sma50, sma200 = s.iloc[-1], s.iloc[-50:].mean(), s.iloc[-200:].mean()
    if price > sma50 > sma200:
        return "uptrend"
    if price < sma50 < sma200:
        return "downtrend"
    return "mixed"


def snapshot(prices: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """All technical facts about one ETF, as a dict the agent can read."""
    s = prices.dropna()
    if len(s) < WINDOWS["3m"] + 1:
        return {"usable": False, "reason": f"only {len(s)} days of prices"}

    daily = s.pct_change().dropna()
    last_year = s.iloc[-TRADING_DAYS:]
    out = {
        "usable": True,
        "as_of": str(s.index[-1].date()),
        "days_of_history": len(s),
        "last_price_eur": round(float(s.iloc[-1]), 4),
        **{f"return_{k}": _round(_ret(s, d)) for k, d in WINDOWS.items()},
        "volatility_3m_annualised": _round(float(daily.iloc[-WINDOWS["3m"]:].std() * math.sqrt(TRADING_DAYS))),
        "max_drawdown_12m": _round(max_drawdown(last_year)),
        "distance_from_12m_high": _round(float(s.iloc[-1] / last_year.max() - 1)),
        "trend": trend(s),
    }
    if benchmark is not None:
        b = benchmark.dropna().loc[: s.index[-1]]
        mine, bench = _ret(s, WINDOWS["3m"]), _ret(b, WINDOWS["3m"])
        out["return_3m_vs_benchmark"] = _round(mine - bench) if None not in (mine, bench) else None
    return out


def momentum_ranking(prices: pd.DataFrame, window: str = "3m") -> pd.Series:
    """ETFs ranked by return over the window, best first. ETFs without enough history are dropped."""
    days = WINDOWS[window]
    rets = {c: _ret(prices[c].dropna(), days) for c in prices.columns}
    return pd.Series({k: v for k, v in rets.items() if v is not None}).sort_values(ascending=False)


def _round(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)
