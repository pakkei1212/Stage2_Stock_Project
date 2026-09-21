"""Stage I, Layer 3 (initial): stop placement, risk ceiling and position sizing.

Deterministic arithmetic over public daily bars. Nothing here talks to a
broker, reads an account balance or sizes a real order — ``portfolio_value`` is
a number the user puts in ``.env``.

Design rules this module encodes:

* **Structure has priority.** The primary invalidation reference is the final
  contraction's low (the structural low of the base), not a percentage and not
  the market cap.
* **Volatility is the floor, not the rule.** A stop tighter than
  ``trade_atr_stop_multiple`` x ATR% would be inside the stock's ordinary daily
  noise, so the plan widens to that floor when structure is tighter than noise.
* **Market cap and liquidity are secondary modifiers** of the *ceiling* on how
  much risk a plan may carry (and of trailing room later) — never a direct
  "mega cap → 5% stop" rule.
* **A required stop wider than the ceiling is a SKIP, not a wider stop.**
  ``trade_plan_status = SKIP_RISK_TOO_WIDE``; the position is not resized to
  make an unacceptable stop fit.

Every threshold is a provisional heuristic of ours (config keys ``trade_*``),
not a rule prescribed by Minervini, and is meant to be re-evaluated against
recorded trade outcomes.
"""
import logging
import math

import pandas as pd

from .config import CONFIG

logger = logging.getLogger(__name__)

OK = "OK"
SKIP_RISK_TOO_WIDE = "SKIP_RISK_TOO_WIDE"
NO_PRICE_DATA = "NO_PRICE_DATA"

STRUCTURE = "structure"
VOLATILITY_FLOOR = "volatility_floor"
VOLATILITY_ONLY = "volatility_only"


def _r(value, digits=2):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else round(value, digits)


def _float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ── indicators (simple, explicit, no TA dependency) ───────────────────────

def true_range(df):
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev = close.shift(1).fillna(close.iloc[0])
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(df, period=20):
    """Average true range over the last ``period`` bars (simple mean of TR)."""
    if df is None or len(df) < 2:
        return None
    tr = true_range(df).dropna()
    if tr.empty:
        return None
    return float(tr.iloc[-min(period, len(tr)):].mean())


def ema(df, span):
    if df is None or len(df) < 2:
        return None
    value = df["Close"].astype(float).ewm(span=span, adjust=False).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def sma(df, window):
    if df is None or len(df) < window:
        return None
    value = df["Close"].astype(float).rolling(window).mean().iloc[-1]
    return None if pd.isna(value) else float(value)


def recent_swing_low(df, lookback=20):
    """Lowest low of the last ``lookback`` bars — the nearest structural floor."""
    if df is None or df.empty:
        return None
    lows = df["Low"].astype(float).iloc[-lookback:]
    return None if lows.empty else float(lows.min())


# ── market-cap / liquidity modifiers ──────────────────────────────────────

def cap_tier(market_cap, config=CONFIG):
    """(max_risk_pct, trail_atr_multiple, label) for a market cap.

    Secondary modifier only: it moves the ceiling on acceptable risk and the
    trailing allowance, and never by itself decides a stop price.
    """
    cap = _float(market_cap)
    if cap is not None:
        for floor, max_risk_pct, trail in config["trade_market_cap_tiers"]:
            if cap >= floor:
                return float(max_risk_pct), float(trail), f"cap>=${floor / 1e9:.0f}B"
    return (float(config["trade_default_risk_ceiling_pct"]),
            float(config["trade_default_trail_atr_multiple"]),
            "cap unknown/below tiers")


def allowed_risk_pct(market_cap, avg_dollar_volume=None, config=CONFIG):
    """Ceiling on stop distance: market-cap tier, tightened when liquidity is thin,
    and never above the hard ``trade_max_risk_pct``. Returns (pct, notes)."""
    ceiling, _, label = cap_tier(market_cap, config)
    notes = [f"risk ceiling {ceiling}% ({label})"]
    adv = _float(avg_dollar_volume)
    if adv is not None and adv < float(config["trade_liquidity_floor_usd"]):
        factor = float(config["trade_illiquid_ceiling_factor"])
        ceiling *= factor
        notes.append(f"thin liquidity (${adv / 1e6:.1f}M/day): ceiling x{factor}")
    hard = float(config["trade_max_risk_pct"])
    if ceiling > hard:
        ceiling = hard
        notes.append(f"clamped to the hard maximum {hard}%")
    return _r(ceiling), notes


def trail_atr_multiple(market_cap, atr_pct, config=CONFIG):
    """ATR multiple of trailing room. Actual volatility outranks market cap: a
    high-ATR name gets extra room whatever its tier."""
    _, multiple, _ = cap_tier(market_cap, config)
    pct = _float(atr_pct)
    if pct is not None and pct >= float(config["trade_high_volatility_atr_pct"]):
        multiple += float(config["trade_high_volatility_trail_bonus"])
    return multiple


# ── Layer 3: the initial risk plan ────────────────────────────────────────

def build_risk_plan(reference_price, bars, *, structural_low=None, market_cap=None,
                    avg_dollar_volume=None, config=CONFIG):
    """Advisory initial-stop plan for an entry at ``reference_price``.

    ``structural_low`` is the base's invalidation reference (the final
    contraction low); when absent, the recent swing low from ``bars`` is used,
    and failing that the plan is volatility-only.
    """
    price = _float(reference_price)
    plan = {
        "reference_price": _r(price),
        "atr_period": int(config["trade_atr_period"]),
        "atr20": None, "atr_pct": None,
        "structural_low": None, "structural_low_source": None,
        "structural_stop": None, "structural_risk_pct": None,
        "volatility_required_risk_pct": None,
        "allowed_max_risk_pct": None,
        "required_risk_pct": None,
        "suggested_initial_stop": None,
        "risk_basis": None,
        "trade_plan_status": NO_PRICE_DATA,
        "notes": [],
    }
    if price is None or price <= 0 or bars is None or len(bars) < 2:
        plan["notes"].append("no usable price history for the risk plan")
        return plan

    period = int(config["trade_atr_period"])
    atr_value = atr(bars, period)
    atr_pct = (atr_value / price * 100) if atr_value else None
    plan["atr20"] = _r(atr_value, 4)
    plan["atr_pct"] = _r(atr_pct)

    low = _float(structural_low)
    source = "final_contraction_low"
    if low is None:
        low = recent_swing_low(bars, int(config["trade_swing_lookback_days"]))
        source = "recent_swing_low"
    if low is not None and low >= price:
        plan["notes"].append(f"structural low {_r(low)} is not below the entry price — ignored")
        low, source = None, None

    buffer_pct = float(config["trade_structural_stop_buffer_pct"])
    if low is not None:
        structural_stop = low * (1 - buffer_pct / 100)
        plan.update({
            "structural_low": _r(low), "structural_low_source": source,
            "structural_stop": _r(structural_stop),
            "structural_risk_pct": _r((price - structural_stop) / price * 100),
        })

    vol_required = (float(config["trade_atr_stop_multiple"]) * atr_pct) if atr_pct else None
    plan["volatility_required_risk_pct"] = _r(vol_required)

    ceiling, ceiling_notes = allowed_risk_pct(market_cap, avg_dollar_volume, config)
    plan["allowed_max_risk_pct"] = ceiling
    plan["notes"].extend(ceiling_notes)

    structural_risk = plan["structural_risk_pct"]
    if structural_risk is None and vol_required is None:
        plan["notes"].append("neither a structural low nor an ATR could be computed")
        return plan

    if structural_risk is None:
        required, basis = vol_required, VOLATILITY_ONLY
        plan["notes"].append("no structural low available — stop derived from volatility alone")
    elif vol_required is None or structural_risk >= vol_required:
        required, basis = structural_risk, STRUCTURE
    else:
        required, basis = vol_required, VOLATILITY_FLOOR
        plan["notes"].append(f"structural stop ({structural_risk}%) sits inside {config['trade_atr_stop_multiple']}x "
                             f"ATR noise — widened to the {_r(vol_required)}% volatility floor")

    stop = price * (1 - required / 100)
    plan.update({
        "required_risk_pct": _r(required),
        "suggested_initial_stop": _r(stop),
        "risk_basis": basis,
        "trade_plan_status": OK,
    })
    if ceiling is not None and required > ceiling:
        plan["trade_plan_status"] = SKIP_RISK_TOO_WIDE
        plan["notes"].append(f"required stop distance {_r(required)}% exceeds the {ceiling}% ceiling — "
                             f"skip rather than widen the stop or tighten it into the noise")
    return plan


# ── position sizing ───────────────────────────────────────────────────────

def position_size(entry_price, stop_price, config=CONFIG, portfolio_value=None):
    """Risk-based and portfolio-portion sizing; the suggestion is the smaller.

        risk_budget        = portfolio_value * risk_per_trade_pct
        risk_based_shares  = risk_budget / (entry_price - stop_price)
        portion_budget     = portfolio_value * max_position_portion
        portion_based      = portion_budget / entry_price
        suggested_shares   = floor(min(risk_based_shares, portion_based))
    """
    entry, stop = _float(entry_price), _float(stop_price)
    value = _float(portfolio_value if portfolio_value is not None else config["trade_portfolio_value"])
    risk_pct = float(config["trade_risk_per_trade_pct"])
    portion_pct = float(config["trade_max_position_portion_pct"])
    out = {
        "portfolio_value": _r(value),
        "risk_per_trade_pct": risk_pct,
        "max_position_portion_pct": portion_pct,
        "risk_budget": None, "per_share_risk": None,
        "risk_based_shares": None, "portion_budget": None, "portion_based_shares": None,
        "suggested_shares": 0, "binding_constraint": None,
        "position_value": None, "position_portion_pct": None, "position_risk_pct": None,
        "notes": [],
    }
    if not entry or entry <= 0 or value is None or value <= 0:
        out["notes"].append("no entry price or portfolio value — cannot size")
        return out
    if stop is None or stop >= entry:
        out["notes"].append("stop is not below the entry price — cannot size")
        return out

    risk_budget = value * risk_pct / 100
    per_share = entry - stop
    risk_shares = risk_budget / per_share
    portion_budget = value * portion_pct / 100
    portion_shares = portion_budget / entry
    shares = int(math.floor(min(risk_shares, portion_shares)))

    out.update({
        "risk_budget": _r(risk_budget),
        "per_share_risk": _r(per_share),
        "risk_based_shares": _r(risk_shares, 4),
        "portion_budget": _r(portion_budget),
        "portion_based_shares": _r(portion_shares, 4),
        "suggested_shares": max(0, shares),
        "binding_constraint": "risk" if risk_shares <= portion_shares else "portion",
    })
    if shares <= 0:
        out["notes"].append("sizing rounds down to zero shares at this portfolio value")
        return out
    out["position_value"] = _r(shares * entry)
    out["position_portion_pct"] = _r(shares * entry / value * 100)
    out["position_risk_pct"] = _r(shares * per_share / value * 100)
    return out


def size_for_plan(entry_price, risk_plan, config=CONFIG, portfolio_value=None):
    """Sizing for a Layer-3 plan; a SKIP_RISK_TOO_WIDE plan is never sized."""
    if risk_plan.get("trade_plan_status") != OK:
        out = position_size(entry_price, None, config, portfolio_value)
        out["notes"].append(f"not sized: trade_plan_status={risk_plan.get('trade_plan_status')}")
        return out
    return position_size(entry_price, risk_plan.get("suggested_initial_stop"), config, portfolio_value)


def structural_low_from_metrics(metrics):
    """The final contraction's low from stored Stage-G metrics, when present."""
    lows = (metrics or {}).get("contraction_low_prices") or []
    values = [_float(v) for v in lows if _float(v) is not None]
    return values[-1] if values else None
