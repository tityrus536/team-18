"""Risk agent: checks every portfolio option against config/rules.yaml. Plain Python, no LLM.

Input: state["proposals"], a list of options from Portfolio Options, each like
    {"name": "Balanced", "weights": {"WORLD": 0.40, "EURO STOXX 50": 0.20, ..., "CASH": 0.10},
     "orders": [...], "rationale": "..."}
Weights are target fractions of the portfolio; "CASH" is uninvested money.

Output: state["risk_check"]
    status   "fail" if any option breaks a hard limit (the graph sends it back to Options),
             "warning" if any option breaks a soft limit, else "pass"
    options  per option: status, breaches (hard), warnings (soft), metrics
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from qdc import indicators
from qdc.data import ROOT, load_prices, load_universe

RULES_FILE = ROOT / "config" / "rules.yaml"
CASH = "CASH"
LOOKBACK = indicators.TRADING_DAYS  # 12 months of daily prices for volatility and drawdown


def load_rules(path: Path = RULES_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def portfolio_returns(weights: dict[str, float], prices: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Daily returns of the portfolio over the last LOOKBACK days, rebalanced daily.

    ETFs with fewer days of prices count as 0% return before their first price,
    which understates risk: they are returned so the agent can say so.
    """
    held = [t for t, w in weights.items() if t != CASH and w > 0]
    window = prices[held].ffill(limit=5).iloc[-(LOOKBACK + 1):]
    daily = window.pct_change(fill_method=None).iloc[1:]
    short = [t for t in held if daily[t].isna().any()]
    w = pd.Series({t: weights[t] for t in held})
    return daily.fillna(0.0) @ w, short


def check_option(option: dict, rules: dict, prices: pd.DataFrame, universe: pd.DataFrame,
                 usable: list[str] | None = None, quality: dict | None = None) -> dict:
    """Hard breaches, soft warnings and risk metrics for one portfolio option."""
    lim, warn, tiers = rules["limits"], rules["warnings"], rules["theme_tiers"]
    weights = {t: float(w) for t, w in (option.get("weights") or {}).items() if float(w) != 0}
    breaches, warnings = [], []

    if not weights:
        return {"name": option.get("name"), "status": "fail", "breaches": ["no target weights given"],
                "warnings": [], "metrics": {}}

    # 1. weights are valid
    total = sum(weights.values())
    if abs(total - 1) > lim["weight_tolerance"]:
        breaches.append(f"weights add up to {total:.1%}, not 100%")
    for t, w in weights.items():
        if w < 0:
            breaches.append(f"{t}: negative weight {w:.1%} (short selling is done with inverse ETFs only)")
    info = universe.set_index("deck_ticker")
    unknown = [t for t in weights if t != CASH and t not in info.index]
    for t in unknown:
        breaches.append(f"{t}: not in the ETF universe")
    etfs = {t: w for t, w in weights.items() if t != CASH and t in info.index}

    # 2. only ETFs the Data agent says are usable, and with prices
    for t in etfs:
        if usable is not None and t not in usable:
            breaches.append(f"{t}: excluded by the Data agent ({'; '.join((quality or {}).get(t, [])) or 'no usable data'})")
        elif t not in prices or prices[t].dropna().empty:
            breaches.append(f"{t}: no prices")

    # 3. position, tier, theme and cash limits
    tier_of = {t: tiers.get(info.loc[t, "theme"], "Very high") for t in etfs}  # unknown theme: strictest
    for t, w in etfs.items():
        cap = lim["max_position_by_tier"][tier_of[t]]
        if w > cap + 1e-9:
            breaches.append(f"{t}: {w:.1%} is above the {cap:.0%} limit for {tier_of[t]} ETFs")
    for tier, cap in lim["max_total_by_tier"].items():
        tw = sum(w for t, w in etfs.items() if tier_of[t] == tier)
        if tw > cap + 1e-9:
            breaches.append(f"{tier} ETFs together: {tw:.1%}, above the {cap:.0%} limit")
    by_theme = pd.Series({t: w for t, w in etfs.items()}).groupby(lambda t: info.loc[t, "theme"]).sum()
    for theme, tw in by_theme.items():
        if tw > lim["max_theme"] + 1e-9:
            breaches.append(f"theme {theme}: {tw:.1%}, above the {lim['max_theme']:.0%} limit")
    if weights.get(CASH, 0) > lim["max_cash"] + 1e-9:
        breaches.append(f"cash {weights[CASH]:.1%} is above the {lim['max_cash']:.0%} limit")
    max_orders = lim.get("max_orders_per_day")
    if max_orders and len(option.get("orders") or []) > max_orders:
        breaches.append(f"{len(option['orders'])} orders, above the limit of {max_orders} per day")

    # 4. risk metrics on the last 12 months (only ETFs with prices)
    priced = {t: w for t, w in etfs.items() if t in prices and not prices[t].dropna().empty}
    metrics = {
        "positions": len(etfs),
        "cash": round(weights.get(CASH, 0.0), 4),
        "leveraged_or_inverse": round(sum(w for t, w in etfs.items()
                                          if float(info.loc[t, "exposure_factor"]) != 1), 4),
        "gross_exposure": round(sum(w * abs(float(info.loc[t, "exposure_factor"])) for t, w in etfs.items()), 4),
    }
    if priced:
        rets, short = portfolio_returns(priced, prices)
        vol = float(rets.std() * math.sqrt(indicators.TRADING_DAYS))
        dd = indicators.max_drawdown((1 + rets).cumprod())
        metrics.update({"volatility_12m": round(vol, 4), "max_drawdown_12m": round(dd, 4),
                        "return_12m": round(float((1 + rets).prod() - 1), 4)})
        if vol > lim["max_volatility"]:
            breaches.append(f"portfolio volatility {vol:.1%} is above the {lim['max_volatility']:.0%} limit")
        elif vol > warn["volatility_above"]:
            warnings.append(f"portfolio volatility {vol:.1%} is above {warn['volatility_above']:.0%}")
        if dd < warn["drawdown_12m_below"]:
            warnings.append(f"this portfolio would have fallen {dd:.1%} from its peak in the last 12 months")
        if short:
            warnings.append(f"less than 12 months of prices for {', '.join(short)}: volatility is understated")

    # 5. soft warnings
    if len(etfs) < warn["min_positions"]:
        warnings.append(f"only {len(etfs)} ETFs: concentrated portfolio")
    for t in etfs:
        if t in prices:
            s = prices[t].dropna()
            years = (s.index[-1] - s.index[0]).days / 365.25 if len(s) else 0
            if years < warn["min_history_years"]:
                warnings.append(f"{t}: only {years:.1f} years of prices")
        if float(info.loc[t, "exposure_factor"]) != 1:
            warnings.append(f"{t}: leveraged/inverse with daily reset, returns drift from the multiple over time")

    status = "fail" if breaches else "warning" if warnings else "pass"
    return {"name": option.get("name"), "status": status, "breaches": breaches, "warnings": warnings,
            "metrics": metrics}


def run_risk_check(proposals: list[dict], as_of: str | None = None, data_report: dict | None = None,
                   prices: pd.DataFrame | None = None, universe: pd.DataFrame | None = None,
                   rules: dict | None = None) -> dict:
    rules = rules or load_rules()
    prices = load_prices(as_of) if prices is None else prices
    universe = load_universe() if universe is None else universe
    # use the Data agent's list only if it really ran (placeholders return an empty list)
    usable = None
    if data_report and data_report.get("status") != "placeholder" and data_report.get("usable_etfs"):
        usable = data_report["usable_etfs"]
    quality = (data_report or {}).get("issues") or {}
    if not isinstance(quality, dict):
        quality = {}

    options = [check_option(o, rules, prices, universe, usable, quality) for o in proposals or []]
    if not options:
        return {"status": "fail", "as_of": str(prices.index[-1].date()), "options": [],
                "breaches": ["no portfolio options to check"]}
    order = {"pass": 0, "warning": 1, "fail": 2}
    status = max((o["status"] for o in options), key=order.get)
    return {
        "status": status,
        "as_of": str(prices.index[-1].date()),
        "options": options,
        "breaches": [f"{o['name']}: {b}" for o in options for b in o["breaches"]],
    }


def risk_node(state: dict) -> dict:
    """The Risk agent as a node of the team workflow (qdc.agents.graph)."""
    return {"risk_check": run_risk_check(state.get("proposals", []), state.get("as_of"),
                                         state.get("data_report"))}
