"""Portfolio Options agent: turns the signals into 2-3 portfolio options with draft orders.

How it works:
1. Python scores every usable ETF: the Technical agent's view if there is one,
   otherwise a simple rule on trend and 3-month return (no LLM).
2. Python builds one portfolio per profile in config/rules.yaml (Defensive, Balanced,
   Dynamic): each risk tier gets its budget, spent on the best-scored ETFs of that
   tier, within the position, tier and theme limits. Unused money goes to CASH.
3. Python computes the orders that move the current portfolio to each option.
4. The LLM explains and compares the options in ONE request (or rules write a short
   text with use_llm=False, so no quota is used).

When Risk fails an option, the graph sends it back here: the option is rebuilt
without the ETFs Risk rejected, and with less risk if volatility was too high.

You do not decide. Humans choose an option (or none) and log it in the Decision Log.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field

from qdc import indicators
from qdc.agents.risk import CASH, load_rules, portfolio_returns
from qdc.data import BENCHMARK, available_universe, load_prices

TIER_ORDER = ["Defensive", "Core", "Elevated", "High", "Very high"]
SIGNAL_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
CONFIDENCE_VALUE = {"high": 1.0, "medium": 0.6, "low": 0.3}
SAFE_TIERS = ("Defensive", "Core")  # these may hold neutral ETFs; riskier tiers need a positive signal


# ---------------------------------------------------------------- 1. score the ETFs (Python)


def rule_signal(snap: dict) -> tuple[str, str]:
    """Fallback when the Technical agent gave no view: trend + 3-month return, low confidence."""
    r3 = snap.get("return_3m")
    if snap.get("trend") == "uptrend" and r3 is not None and r3 > 0:
        return "positive", "low"
    if snap.get("trend") == "downtrend" or (r3 is not None and r3 < 0 and snap.get("trend") != "uptrend"):
        return "negative", "low"
    return "neutral", "low"


def score_etfs(prices: pd.DataFrame, universe: pd.DataFrame, rules: dict,
               signals: dict | None = None, usable: list[str] | None = None) -> pd.DataFrame:
    """One row per usable ETF: tier, theme, signal, confidence, score, where the view came from."""
    views = {v["deck_ticker"]: v for v in (signals or {}).get("views", [])}
    bench = prices[BENCHMARK] if BENCHMARK in prices else None
    info = universe.set_index("deck_ticker")
    rows = []
    for t in prices.columns:
        if t not in info.index or (usable is not None and t not in usable):
            continue
        snap = indicators.snapshot(prices[t], None if t == BENCHMARK else bench)
        if not snap.get("usable"):
            continue
        if t in views:
            sig, conf, source = views[t]["signal"], views[t]["confidence"], "technical agent"
        else:
            (sig, conf), source = rule_signal(snap), "trend rule"
        rows.append({
            "deck_ticker": t, "theme": info.loc[t, "theme"],
            "tier": rules["theme_tiers"].get(info.loc[t, "theme"], "Very high"),
            "exposure_factor": float(info.loc[t, "exposure_factor"]),
            "signal": sig, "confidence": conf, "source": source,
            "score": SIGNAL_VALUE[sig] * CONFIDENCE_VALUE[conf],
            "return_3m": snap.get("return_3m"), "trend": snap.get("trend"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["score", "return_3m"], ascending=False, na_position="last").set_index("deck_ticker")


# ---------------------------------------------------------------- 2. build the portfolios (Python)


def build_weights(budgets: dict[str, float], table: pd.DataFrame, rules: dict,
                  exclude: set[str] = frozenset(), risk_scale: float = 1.0) -> tuple[dict[str, float], dict]:
    """Target weights for one profile, within the limits of rules.yaml by construction.

    Returns (weights, budget each tier could not use). Within a tier, ETFs are filled to
    half their cap first (to spread the money), then to the full cap. In Core, the benchmark
    comes first when its signal is not negative. Money a risky tier can't use moves down
    to the next safer tier (Very high -> High -> Elevated -> Core), then to CASH, then to
    the bond ETFs if cash is full.
    """
    lim, min_w = rules["limits"], rules["trading"]["min_weight"]
    # less risk after a volatility breach: risky budgets shrink, the rest goes to Defensive
    budgets = dict(budgets)
    if risk_scale < 1:
        moved = sum(b * (1 - risk_scale) for tier, b in budgets.items() if tier not in SAFE_TIERS)
        budgets = {tier: b * risk_scale if tier not in SAFE_TIERS else b for tier, b in budgets.items()}
        budgets["Defensive"] = budgets.get("Defensive", 0) + moved

    weights: dict[str, float] = {}
    theme_used: dict[str, float] = defaultdict(float)
    table = table[~table.index.isin(exclude)]

    def fill(tier: str, amount: float, min_score: float) -> float:
        """Spend `amount` on the best ETFs of `tier`; return what could not be spent."""
        cap = lim["max_position_by_tier"][tier]
        tier_room = lim["max_total_by_tier"].get(tier, 1.0) - sum(
            w for t, w in weights.items() if table.loc[t, "tier"] == tier)
        amount_left = min(amount, max(tier_room, 0))
        unspent = amount - amount_left
        candidates = table[(table["tier"] == tier) & (table["score"] >= min_score)]
        if tier == "Core" and BENCHMARK in candidates.index:
            candidates = pd.concat([candidates.loc[[BENCHMARK]], candidates.drop(BENCHMARK)])
        for step_cap in (cap / 2, cap):  # spread first, then top up
            for t, row in candidates.iterrows():
                if amount_left < min_w:
                    break
                w = min(step_cap - weights.get(t, 0), amount_left, lim["max_theme"] - theme_used[row["theme"]])
                if w < min_w and not (t in weights and w > 1e-9):  # no tiny new position
                    continue
                weights[t] = weights.get(t, 0) + w
                theme_used[row["theme"]] += w
                amount_left -= w
        return unspent + amount_left

    # riskiest tier first: money a tier can't use moves down to the next safer tier
    unused, carry = {}, 0.0
    for tier in reversed(TIER_ORDER[1:]):  # Very high -> High -> Elevated -> Core
        amount = budgets.get(tier, 0) + carry
        if amount <= 1e-9:
            continue
        left = fill(tier, amount, 0.0 if tier in SAFE_TIERS else 1e-9)
        if left > 1e-6:
            unused[tier] = round(left, 4)
        carry = left
    carry += fill("Defensive", budgets.get("Defensive", 0), 0.0)
    leftover = carry
    to_cash = min(leftover, lim["max_cash"])
    fill("Defensive", leftover - to_cash, 0.0)  # cash full: neutral or better bonds take the rest
    weights = {t: round(w, 4) for t, w in weights.items() if w > 0}
    weights[CASH] = round(1 - sum(weights.values()), 4)
    return weights, unused


# ---------------------------------------------------------------- 3. orders (Python)


def make_orders(weights: dict[str, float], portfolio: dict, last_prices: pd.Series,
                universe: pd.DataFrame, min_order_eur: float) -> tuple[list[dict], float]:
    """Whole-unit orders from the current holdings to the target weights. Sells first.

    Returns (orders, total portfolio value in EUR). Prices are the last EUR close:
    the real fill will differ, so these are draft orders for the team to check.
    """
    holdings = (portfolio or {}).get("holdings", {}) or {}
    cash = float((portfolio or {}).get("cash_eur", 0) or 0)
    total = cash + sum(units * float(last_prices[t]) for t, units in holdings.items() if t in last_prices)
    yahoo = universe.set_index("deck_ticker")["yahoo_ticker"]
    orders = []
    for t in sorted((set(weights) | set(holdings)) - {CASH}):
        price = float(last_prices[t])
        diff = math.floor(weights.get(t, 0) * total / price) - holdings.get(t, 0)
        if diff and abs(diff) * price >= min_order_eur:
            orders.append({"deck_ticker": t, "yahoo_ticker": yahoo.get(t), "action": "buy" if diff > 0 else "sell",
                           "units": abs(int(diff)), "price_eur": round(price, 2),
                           "amount_eur": round(abs(diff) * price, 2)})
    orders.sort(key=lambda o: o["action"] != "sell")
    return orders, round(total, 2)


def option_metrics(weights: dict[str, float], prices: pd.DataFrame) -> dict:
    priced = {t: w for t, w in weights.items() if t != CASH}
    if not priced:
        return {}
    rets, _ = portfolio_returns(priced, prices)
    return {"volatility_12m": round(float(rets.std() * math.sqrt(indicators.TRADING_DAYS)), 4),
            "return_12m": round(float((1 + rets).prod() - 1), 4),
            "max_drawdown_12m": round(indicators.max_drawdown((1 + rets).cumprod()), 4)}


# ---------------------------------------------------------------- 4. explain (LLM, one request)


class OptionText(BaseModel):
    name: str
    rationale: str = Field(description="3-5 sentences: what this option bets on and why, quoting signals and numbers")
    key_risks: list[str] = Field(description="2-4 things that could make this option lose money")


class OptionsNarrative(BaseModel):
    options: list[OptionText]
    comparison: str = Field(description="3-5 sentences comparing the options for the team")


SYSTEM_PROMPT = """You are the Portfolio Options agent of a student investment team.
Python has already built the portfolio options below from the agents' signals and the
team's rules. You EXPLAIN them; you never change weights or invent numbers.
Weights, returns and volatility are fractions: 0.05 means 5%. Write them as percentages.

For each option: what it bets on, which signals support it (quote them), and its key risks
(concentration, leveraged ETFs, short histories, weak signals, the macro view if given).
Then compare the options in a few sentences. Past 12-month numbers are not forecasts: say so.
You do not choose. The humans do."""


def rule_text(option: dict, table: pd.DataFrame) -> dict:
    """Short explanation without the LLM."""
    held = [t for t in option["weights"] if t != CASH]
    by_tier = defaultdict(float)
    for t in held:
        by_tier[table.loc[t, "tier"]] += option["weights"][t]
    tiers = ", ".join(f"{tier} {w:.0%}" for tier, w in sorted(by_tier.items(), key=lambda x: -x[1]))
    top = ", ".join(f"{t} ({table.loc[t, 'signal']}, {table.loc[t, 'source']})"
                    for t in sorted(held, key=lambda t: -option["weights"][t])[:4])
    m = option.get("metrics", {})
    unused = option.get("unused_budget") or {}
    note = (" Not enough ETFs with a positive signal in " + ", ".join(f"{k} ({v:.0%})" for k, v in unused.items())
            + ": that money moved to safer tiers (cash last).") if unused else ""
    risks = [f"{t}: weak or neutral signal" for t in held if table.loc[t, "score"] <= 0][:3]
    risks += [f"{t}: leveraged/inverse, daily reset" for t in held if table.loc[t, "exposure_factor"] != 1]
    return {"rationale": f"{option['name']}: {tiers}, cash {option['weights'].get(CASH, 0):.0%}. "
                         f"Largest positions: {top}. Past 12 months (not a forecast): return "
                         f"{m.get('return_12m', 0):.1%}, volatility {m.get('volatility_12m', 0):.1%}, "
                         f"max drawdown {m.get('max_drawdown_12m', 0):.1%}.{note} (Written by rules, no LLM.)",
            "key_risks": risks or ["past 12-month behaviour may not repeat"]}


def explain(options: list[dict], signals: dict | None, macro_view: dict | None, llm) -> OptionsNarrative:
    payload = {
        "options": [{k: o[k] for k in ("name", "weights", "metrics", "signals_used", "unused_budget")}
                    for o in options],
        "technical_summary": (signals or {}).get("summary"),
        "macro_view": macro_view,
    }
    return llm.with_structured_output(OptionsNarrative).invoke([
        ("system", SYSTEM_PROMPT),
        ("user", json.dumps(payload, indent=1, default=str)),
    ])


# ---------------------------------------------------------------- the agent


def _adjustments(state: dict, universe: pd.DataFrame) -> dict[str, dict]:
    """What to change per profile after Risk failed it: ETFs to drop, risk to cut."""
    previous = {p["name"]: p.get("adjustments", {}) for p in state.get("proposals") or [] if isinstance(p, dict)}
    out = {name: {"exclude": list(a.get("exclude", [])), "risk_scale": a.get("risk_scale", 1.0)}
           for name, a in previous.items()}
    known = set(universe["deck_ticker"])
    for opt in (state.get("risk_check") or {}).get("options", []):
        if opt.get("status") != "fail":
            continue
        adj = out.setdefault(opt["name"], {"exclude": [], "risk_scale": 1.0})
        for b in opt.get("breaches", []):
            m = re.match(r"^(.+?): ", b)
            if m and m.group(1) in known and m.group(1) not in adj["exclude"]:
                adj["exclude"].append(m.group(1))
            if "volatility" in b:
                adj["risk_scale"] = round(adj["risk_scale"] * 0.7, 4)
    return out


def run_options_agent(state: dict, use_llm: bool = True, llm=None, prices: pd.DataFrame | None = None,
                      universe: pd.DataFrame | None = None, rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    prices = load_prices(state.get("as_of")) if prices is None else prices
    universe = available_universe() if universe is None else universe
    report = state.get("data_report") or {}
    usable = report.get("usable_etfs") if report.get("status") != "placeholder" and report.get("usable_etfs") else None
    signals = state.get("signals") if (state.get("signals") or {}).get("status") != "placeholder" else None

    table = score_etfs(prices, universe, rules, signals, usable)
    if table.empty:
        return [{"name": "no option", "weights": {CASH: 1.0}, "orders": [], "rationale": "no usable ETF",
                 "key_risks": [], "metrics": {}, "signals_used": {}, "adjustments": {}}]
    adjust = _adjustments(state, universe)
    last = prices.ffill().iloc[-1]

    options = []
    for name, budgets in rules["profiles"].items():
        adj = adjust.get(name, {"exclude": [], "risk_scale": 1.0})
        weights, unused = build_weights(budgets, table, rules, set(adj["exclude"]), adj["risk_scale"])
        orders, total = make_orders(weights, state.get("portfolio") or {}, last, universe,
                                    rules["trading"]["min_order_eur"])
        options.append({
            "name": name, "weights": weights, "orders": orders, "portfolio_value_eur": total,
            "unused_budget": unused,  # per tier, budget moved to safer tiers: no ETF with a good enough signal
            "metrics": option_metrics(weights, prices),
            "signals_used": {t: f"{table.loc[t, 'signal']} ({table.loc[t, 'confidence']}, {table.loc[t, 'source']})"
                             for t in weights if t != CASH},
            "adjustments": adj,
        })

    texts = {}
    if use_llm:
        from qdc.llm import get_llm
        try:
            narrative = explain(options, signals, state.get("macro_view"), llm or get_llm())
            texts = {o.name: {"rationale": o.rationale, "key_risks": o.key_risks} for o in narrative.options}
            for o in options:
                o["comparison"] = narrative.comparison
        except Exception as e:  # quota or network: the options are still valid, only the text is missing
            print(f"Options agent: LLM explanation failed ({str(e)[:120]}), using rule text")
    for o in options:
        o.update(texts.get(o["name"]) or rule_text(o, table))
    return options


def options_node(state: dict) -> dict:
    """The agent as a node of the team workflow (one Gemini request per run)."""
    return {"proposals": run_options_agent(state), "revisions": state.get("revisions", -1) + 1}


def options_node_no_llm(state: dict) -> dict:
    """Same, without Gemini: explanations are written by rules."""
    return {"proposals": run_options_agent(state, use_llm=False), "revisions": state.get("revisions", -1) + 1}
