"""Challenger agent: the devil's advocate. Finds the weak points of each portfolio option.

How it works:
1. Python gathers the evidence against each option (no LLM):
   - holdings that move together (correlation over 12 months), real diversification
   - money placed on weak signals (neutral, or low confidence)
   - ETFs bought after a strong run (reversal risk), leveraged ETFs
   - stress test: what the option would have done in past crises (prices before as_of only)
   - options that look alike (the humans don't get a real choice)
2. The LLM turns the evidence into clear objections in ONE request
   (or rules write them with use_llm=False, so no quota is used).

Output: state["objections"], a list of
    {"option": "Balanced" or "all options", "point": ..., "severity": "high"|"medium"|"low",
     "evidence": ..., "suggestion": ...}
It never changes the options: the humans read the objections and decide.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from qdc import indicators
from qdc.agents.risk import CASH, portfolio_returns
from qdc.data import available_universe, load_prices

# Past crises for the stress test. Only windows that ended before as_of are used.
STRESS_WINDOWS = {
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan-Oct 2022)": ("2022-01-03", "2022-10-12"),
}
HIGH_CORRELATION = 0.85      # two holdings above this move almost as one
MIN_PAIR_WEIGHT = 0.10       # only flag pairs where both holdings are at least 10%
WEAK_SIGNAL_SHARE = 0.30     # flag when more than 30% sits on neutral or low-confidence signals
STRONG_RUN = 0.15            # 3-month return above 15%...
NEAR_HIGH = -0.02            # ...and within 2% of its 12-month high: bought after a strong run
SIMILAR_OPTIONS = 0.70       # two options sharing 70% of their weight are not a real choice
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------- 1. evidence (Python)


def overlap(a: dict[str, float], b: dict[str, float]) -> float:
    """Share of weight two portfolios have in common (1 = identical)."""
    return sum(min(a.get(t, 0), b.get(t, 0)) for t in set(a) | set(b))


def stress_test(weights: dict[str, float], prices: pd.DataFrame) -> dict[str, dict]:
    """Return of the option (rebalanced daily) over each past crisis, using the holdings that existed."""
    out = {}
    held = {t: w for t, w in weights.items() if t != CASH and t in prices}
    for name, (start, end) in STRESS_WINDOWS.items():
        if prices.index[-1] < pd.Timestamp(end):
            continue  # not yet in the past at as_of
        window = prices.loc[start:end, list(held)].ffill(limit=5)
        existing = [t for t in held if window[t].notna().iloc[0]]
        if not existing:
            continue
        daily = window[existing].pct_change(fill_method=None).iloc[1:].fillna(0.0)
        w = pd.Series({t: held[t] for t in existing})
        ret = float((1 + daily @ w).prod() - 1)  # missing holdings and cash count as 0%
        out[name] = {"return": round(ret, 4), "weight_covered": round(float(w.sum()), 4),
                     "missing": [t for t in held if t not in existing]}
    return out


def gather_evidence(options: list[dict], prices: pd.DataFrame, signals: dict | None = None,
                    risk_check: dict | None = None) -> dict:
    """All facts the Challenger can use, per option and across options."""
    views = {v["deck_ticker"]: v for v in (signals or {}).get("views", [])}
    risk_by_name = {o["name"]: o for o in (risk_check or {}).get("options", [])}
    lookback = prices.iloc[-(indicators.TRADING_DAYS + 1):].ffill(limit=5)
    daily = lookback.pct_change(fill_method=None).iloc[1:]

    per_option = []
    for opt in options:
        w = {t: x for t, x in (opt.get("weights") or {}).items() if t != CASH and x > 0 and t in prices}
        held = sorted(w, key=lambda t: -w[t])
        facts = {"name": opt["name"], "weights": opt.get("weights"), "metrics": opt.get("metrics", {}),
                 "risk_status": risk_by_name.get(opt["name"], {}).get("status"),
                 "risk_warnings": risk_by_name.get(opt["name"], {}).get("warnings", [])}

        corr = daily[held].corr() if len(held) > 1 else pd.DataFrame()
        facts["correlated_pairs"] = [
            {"pair": [a, b], "correlation": round(float(corr.loc[a, b]), 2), "combined_weight": round(w[a] + w[b], 4)}
            for a, b in combinations(held, 2)
            if pd.notna(corr.loc[a, b]) and corr.loc[a, b] > HIGH_CORRELATION and min(w[a], w[b]) >= MIN_PAIR_WEIGHT
        ]
        if held:
            rets, _ = portfolio_returns(w, prices)
            port_vol = float(rets.std() * math.sqrt(indicators.TRADING_DAYS))
            solo_vol = sum(w[t] * float(daily[t].std() * math.sqrt(indicators.TRADING_DAYS)) for t in held)
            # 1 = no diversification at all; higher = holdings offset each other
            facts["diversification_ratio"] = round(solo_vol / port_vol, 2) if port_vol > 0 else None
            facts["worst_month_12m"] = round(float((1 + rets).rolling(21).apply(lambda r: r.prod() - 1).min()), 4)

        weak = {t: w[t] for t in held if t in views
                and (views[t]["signal"] == "neutral" or views[t]["confidence"] == "low")}
        facts["weak_signal_weight"] = round(sum(weak.values()), 4)
        facts["weak_signal_holdings"] = {t: f"{views[t]['signal']} ({views[t]['confidence']})" for t in weak}

        facts["after_strong_run"] = []
        for t in held:
            snap = indicators.snapshot(prices[t])
            if (snap.get("return_3m") or 0) > STRONG_RUN and (snap.get("distance_from_12m_high") or -1) > NEAR_HIGH:
                facts["after_strong_run"].append({"etf": t, "weight": w[t], "return_3m": snap["return_3m"]})
        facts["stress_test"] = stress_test(w, prices)
        facts["unused_budget"] = opt.get("unused_budget", {})
        per_option.append(facts)

    similar = [{"options": [a["name"], b["name"]], "overlap": round(overlap(a["weights"], b["weights"]), 2)}
               for a, b in combinations(options, 2)
               if overlap(a.get("weights") or {}, b.get("weights") or {}) >= SIMILAR_OPTIONS]
    return {"options": per_option, "similar_options": similar}


# ---------------------------------------------------------------- 2a. objections by rules (no LLM)


def rule_objections(evidence: dict) -> list[dict]:
    out = []

    def add(option, point, severity, ev, suggestion):
        out.append({"option": option, "point": point, "severity": severity, "evidence": ev,
                    "suggestion": suggestion})

    for f in evidence["options"]:
        name = f["name"]
        for p in f.get("correlated_pairs", []):
            a, b = p["pair"]
            add(name, f"{a} and {b} move almost together: {p['combined_weight']:.0%} is close to one bet",
                "medium", f"correlation {p['correlation']} over 12 months", f"keep one of {a} / {b}, or reduce both")
        dr = f.get("diversification_ratio")
        if dr is not None and dr < 1.15:
            add(name, "little real diversification: the holdings rise and fall together", "medium",
                f"diversification ratio {dr} (1 = none)", "add an asset that moves differently (bonds, gold, another region)")
        if f["weak_signal_weight"] > WEAK_SIGNAL_SHARE:
            add(name, f"{f['weak_signal_weight']:.0%} of the money rests on weak signals", "medium",
                ", ".join(f"{t}: {s}" for t, s in f["weak_signal_holdings"].items()),
                "check these positions by hand, or keep them smaller")
        for r in f["after_strong_run"]:
            add(name, f"{r['etf']} is bought after a strong run, close to its 12-month high", "low",
                f"3-month return {r['return_3m']:.1%}", "consider buying in two steps")
        for crisis, s in f.get("stress_test", {}).items():
            if s["return"] < -0.08:
                sev = "high" if s["return"] < -0.15 else "medium"
                miss = f"; {', '.join(s['missing'])} did not exist yet" if s["missing"] else ""
                add(name, f"would have lost {s['return']:.1%} in the {crisis}", sev,
                    f"stress test on {s['weight_covered']:.0%} of the portfolio{miss}",
                    "make sure the team accepts this loss in a similar crisis")
        if f.get("unused_budget"):
            add(name, "the profile could not use its full risk budget: it is less risky than its name says", "low",
                ", ".join(f"{k} {v:.0%}" for k, v in f["unused_budget"].items()),
                "fine in a weak market; say it in the Decision Log")
    for s in evidence["similar_options"]:
        a, b = s["options"]
        add("all options", f"{a} and {b} share {s['overlap']:.0%} of their weights: not a real choice", "medium",
            "same top holdings in both", "ask Portfolio Options for a more different alternative")
    return sorted(out, key=lambda o: SEVERITY_ORDER[o["severity"]])


# ---------------------------------------------------------------- 2b. objections by the LLM (one request)


class Objection(BaseModel):
    option: str = Field(description="Option name, or 'all options'")
    point: str = Field(description="The weakness, in one sentence")
    severity: Literal["high", "medium", "low"]
    evidence: str = Field(description="The numbers from the evidence that show it")
    suggestion: str = Field(description="What the team could check or change")


class ChallengerReport(BaseModel):
    objections: list[Objection]


SYSTEM_PROMPT = """You are the Challenger agent of a student investment team: the devil's advocate.
Your job is to find the weak points of each portfolio option BEFORE the humans decide.
Use ONLY the evidence given. Never invent a number. Fractions: 0.05 means 5%.

Look for: holdings that are really one bet (high correlation), little diversification,
money resting on weak signals, buying after a strong run, leveraged ETFs, losses in past
crises (stress_test), options that are too similar, and anything in the macro view or
risk warnings that contradicts the option.

Severity: high = could cause a large loss or misleads the team; medium = a real weakness
to discuss; low = worth knowing. Give at most 4 objections per option, the most important
first, plus objections for "all options" if they apply. Be concrete and short.
You do not choose and you do not change the options."""


def challenge(evidence: dict, macro_view: dict | None, llm) -> list[dict]:
    payload = {"evidence": evidence, "macro_view": macro_view}
    report = llm.with_structured_output(ChallengerReport).invoke([
        ("system", SYSTEM_PROMPT),
        ("user", json.dumps(payload, indent=1, default=str)),
    ])
    return sorted([o.model_dump() for o in report.objections], key=lambda o: SEVERITY_ORDER[o["severity"]])


# ---------------------------------------------------------------- the agent


def run_challenger(state: dict, use_llm: bool = True, llm=None, prices: pd.DataFrame | None = None) -> list[dict]:
    options = [o for o in state.get("proposals") or [] if o.get("weights")]
    if not options:
        return [{"option": "all options", "point": "no portfolio option to challenge", "severity": "low",
                 "evidence": "", "suggestion": ""}]
    prices = load_prices(state.get("as_of")) if prices is None else prices
    signals = state.get("signals") if (state.get("signals") or {}).get("status") != "placeholder" else None
    evidence = gather_evidence(options, prices, signals, state.get("risk_check"))

    if use_llm:
        from qdc.llm import get_llm
        try:
            macro = state.get("macro_view") if (state.get("macro_view") or {}).get("status") != "placeholder" else None
            return challenge(evidence, macro, llm or get_llm())
        except Exception as e:  # quota or network: rules still give the main objections
            print(f"Challenger: LLM failed ({str(e)[:120]}), using rule objections")
    return rule_objections(evidence)


def challenger_node(state: dict) -> dict:
    """The agent as a node of the team workflow (one Gemini request per run)."""
    return {"objections": run_challenger(state)}


def challenger_node_no_llm(state: dict) -> dict:
    return {"objections": run_challenger(state, use_llm=False)}


if __name__ == "__main__":  # quick look without the graph: python -m qdc.agents.challenger
    from qdc.agents.options import run_options_agent

    st = {"portfolio": {"cash_eur": 10_000, "holdings": {}}}
    st["proposals"] = run_options_agent(st, use_llm=False, universe=available_universe())
    for o in run_challenger(st, use_llm=False):
        print(f"[{o['severity']:<6}] {o['option']:<11} {o['point']}  ({o['evidence']})")
