"""Stage I, Layer 3 (ongoing): daily monitoring of open positions.

Runs as part of the normal Stage-H daily cycle but is **independent of today's
screener**: every open position in the trade journal is monitored, whether or
not the symbol appears in today's Stage-G output. Price history comes from the
existing cached OHLCV choke point (``data_sources.batch_download``).

Everything here is advisory. The monitor computes deterministic numbers,
classifies a state and writes a snapshot; it never sells, never places or
cancels anything at a broker, and there is no brokerage connection to place
anything with.

Corporate actions are processed *before* this monitor runs
(:mod:`pipeline.corporate_actions`), so the share basis a snapshot is computed
on is always current. Price bars are split- and dividend-adjusted by the
source; this module therefore applies **no** price adjustment of its own —
doing so would double-count. See ``corporate_actions.PRICE_BASIS``.

Monitor states (``MONITOR_STATES``), evaluated in this priority order::

    CORPORATE_ACTION_REVIEW  an unresolved corporate action, or market data
                           that stopped — mechanical exit conclusions are
                           suppressed until the user resolves it
    STOP_TRIGGERED         close at/below the current protective level
    EXIT_REVIEW            failed breakout, structure break, or an abnormal
                           high-volume reversal below EMA10
    PARTIAL_PROFIT_REVIEW  large cushion AND a climactic extension from EMA10
    TIGHTEN_PROTECTION     protection just ratcheted up, or a big winner lost EMA10
    WATCH                  below EMA10 but structure intact
    HOLD                   nothing to do

Minervini-style *principles* are the inspiration — cut failures quickly, never
widen a stop, protect profits progressively, give genuine leaders room. The
particular thresholds below are OUR provisional heuristics (``trade_*`` config
keys) and are explicitly not presented as rules from the books. There is no
universal "take profit at +20%".
"""
import logging
from datetime import datetime, timezone

import pandas as pd

from . import corporate_actions as ca
from . import risk_model as rm
from . import trade_store as ts
from .config import CONFIG

logger = logging.getLogger(__name__)

HOLD = "HOLD"
WATCH = "WATCH"
TIGHTEN_PROTECTION = "TIGHTEN_PROTECTION"
PARTIAL_PROFIT_REVIEW = "PARTIAL_PROFIT_REVIEW"
EXIT_REVIEW = "EXIT_REVIEW"
STOP_TRIGGERED = "STOP_TRIGGERED"
CORPORATE_ACTION_REVIEW = "CORPORATE_ACTION_REVIEW"

MONITOR_STATES = (HOLD, WATCH, TIGHTEN_PROTECTION, PARTIAL_PROFIT_REVIEW, EXIT_REVIEW,
                  STOP_TRIGGERED, CORPORATE_ACTION_REVIEW)
ALERT_STATES = (TIGHTEN_PROTECTION, PARTIAL_PROFIT_REVIEW, EXIT_REVIEW, STOP_TRIGGERED,
                CORPORATE_ACTION_REVIEW)


def _r(value, digits=2):
    return None if value is None else round(float(value), digits)


def fetch_bars(symbols, config=CONFIG):
    """Daily bars per symbol through the existing OHLCV cache (Stage-B choke point)."""
    from . import data_sources
    if not symbols:
        return {}
    period = f"{int(config['trade_monitor_lookback_days'])}d"
    return data_sources.batch_download(list(symbols), period=period, config=config)


def _entry_date(trade):
    opened = trade.get("opened_at")
    if not opened:
        return None
    try:
        dt = datetime.fromisoformat(str(opened))
    except ValueError:
        return None
    return (dt.astimezone(timezone.utc) if dt.tzinfo else dt).date()


def structural_swing_low(bars, lookback, exclude_recent=3):
    """Lowest low of the lookback window, ignoring the most recent bars.

    The last few sessions are excluded deliberately: today's own low is always
    at or below today's close, so a swing low that included it could never be
    broken. What matters is whether price has undercut the floor built *before*
    this move.
    """
    if bars is None or len(bars) <= exclude_recent:
        return None
    window = bars["Low"].astype(float).iloc[-(lookback + exclude_recent):-exclude_recent]
    return None if window.empty else float(window.min())


def _since_entry(bars, entry_date):
    if entry_date is None:
        return bars
    index = pd.to_datetime(bars.index)
    slice_ = bars[index.date >= entry_date]
    return slice_ if not slice_.empty else bars.iloc[-1:]


# ── protective level (§11) ────────────────────────────────────────────────

def protective_level(*, close, average_cost, pnl_pct, initial_stop, previous_level, ema10, ema21,
                     swing_low, atr_value, atr_pct, market_cap, config=CONFIG):
    """Progressive protective level: structure + ATR allowance + cap/liquidity
    modifier + profit cushion. Returns ``(level, basis, notes)``.

    Two hard rules: the level never moves *down* (an initial stop is never
    widened, and a level once raised stays raised), and it is never tighter than
    the stock's own ATR-based room unless it was already there.
    """
    notes = []
    multiple = rm.trail_atr_multiple(market_cap, atr_pct, config)
    allowance = (multiple * atr_value) if atr_value else 0.0
    ratchet = [v for v in (initial_stop, previous_level) if v is not None]

    candidates = [("initial_stop", initial_stop), ("previous_level", previous_level)]
    if pnl_pct is not None:
        if pnl_pct >= float(config["trade_breakeven_cushion_pct"]) and average_cost:
            candidates.append(("breakeven", average_cost))
        if pnl_pct >= float(config["trade_swing_trail_cushion_pct"]) and swing_low:
            buffer_pct = float(config["trade_structural_stop_buffer_pct"])
            candidates.append(("swing_low", swing_low * (1 - buffer_pct / 100)))
        if pnl_pct >= float(config["trade_ema21_trail_cushion_pct"]) and ema21:
            candidates.append(("ema21_minus_atr", ema21 - allowance))
        if pnl_pct >= float(config["trade_ema10_trail_cushion_pct"]) and ema10:
            candidates.append(("ema10_minus_atr", ema10 - allowance))

    usable = [(name, float(value)) for name, value in candidates if value is not None]
    if not usable:
        return None, None, ["no stop recorded and no cushion yet — no protective level"]
    basis, level = max(usable, key=lambda pair: pair[1])

    if allowance and close is not None:
        room = close - allowance
        if level > room:
            floor = max(ratchet) if ratchet else None
            clamped = max(room, floor) if floor is not None else room
            if clamped < level:
                notes.append(f"held {_r(multiple, 2)}x ATR below the close rather than tightening into noise")
                level, basis = clamped, f"{basis}+atr_room"
    if ratchet and level < max(ratchet):
        level, basis = max(ratchet), "ratchet"
        notes.append("protection never moves down")
    return _r(level), basis, notes


# ── state classification (§10) ────────────────────────────────────────────

def classify_position(snapshot, config=CONFIG):
    """(state, reasons) from a computed snapshot. Pure and deterministic."""
    review = snapshot.get("corporate_action_review")
    if review:
        # §9: an unresolved corporate action (or market data that stopped)
        # outranks every technical conclusion. Nothing here is a sale.
        return CORPORATE_ACTION_REVIEW, [str(review)]

    close = snapshot.get("close")
    level = snapshot.get("current_protective_level")
    initial_stop = snapshot.get("initial_stop")
    pnl = snapshot.get("pnl_pct")
    ema10, ema21 = snapshot.get("ema10"), snapshot.get("ema21")
    swing_low = snapshot.get("swing_low")
    metrics = snapshot.get("metrics") or {}
    reasons = []

    if close is None:
        return HOLD, ["no price data for this session"]

    if level is not None and close <= level:
        return STOP_TRIGGERED, [f"close {_r(close)} is at/below the protective level {_r(level)}"]
    if initial_stop is not None and close <= initial_stop:
        return STOP_TRIGGERED, [f"close {_r(close)} is at/below the initial stop {_r(initial_stop)}"]

    # §5: judge the move on the same basis the price references live on. With
    # the adjusted bars this project uses, Yahoo has already removed every
    # ex-dividend drop, so the allowance is 0.0 and this changes nothing —
    # adding the dividend back on top would be the double adjustment.
    allowance = float((metrics or {}).get("dividend_price_allowance") or 0.0)
    judged = close + allowance

    pivot = snapshot.get("pivot_price")
    days = snapshot.get("days_since_entry")
    if (pivot and days is not None and days <= int(config["trade_failed_breakout_days"])
            and pnl is not None and pnl < 0
            and judged < float(pivot) * (1 - float(config["trade_failed_breakout_pct"]) / 100)):
        reasons.append(f"failed breakout: back {_r((float(pivot) - judged) / float(pivot) * 100)}% below the "
                       f"{_r(pivot)} pivot {days} session(s) after entry")
    if ema21 and swing_low and judged < ema21 and judged < swing_low:
        reasons.append(f"structure break: close below EMA21 ({_r(ema21)}) and the recent swing low ({_r(swing_low)})")
    if metrics.get("high_volume_reversal") and ema10 and close < ema10:
        reasons.append(f"abnormal reversal volume ({metrics.get('volume_vs_20d')}x 20d avg) with a weak close, "
                       f"below EMA10")
    if reasons:
        return EXIT_REVIEW, reasons

    ext10 = metrics.get("ext_ema10_pct")
    if (pnl is not None and pnl >= float(config["trade_partial_profit_cushion_pct"])
            and ext10 is not None and ext10 >= float(config["trade_climax_ext_ema10_pct"])):
        return PARTIAL_PROFIT_REVIEW, [
            f"+{_r(pnl)}% cushion and price {_r(ext10)}% above EMA10 — climactic extension, review taking "
            f"partial profit (no fixed profit target is applied)"]

    if snapshot.get("protection_raised"):
        reasons.append(f"protective level raised to {_r(level)}")
    if pnl is not None and pnl >= float(config["trade_breakeven_cushion_pct"]) and ema10 and close < ema10:
        reasons.append(f"winner closed below EMA10 ({_r(ema10)}) — tighten protection")
    if reasons:
        return TIGHTEN_PROTECTION, reasons

    if ema10 and close < ema10:
        detail = "close below EMA10"
        intact = []
        if ema21 and close >= ema21:
            intact.append(f"above EMA21 ({_r(ema21)})")
        if swing_low and close >= swing_low:
            intact.append(f"above the recent swing low ({_r(swing_low)})")
        if intact:
            detail += "; still " + " and ".join(intact)
        return WATCH, [detail]
    return HOLD, ["trend structure intact"]


# ── corporate-action context ──────────────────────────────────────────────

def _previous_metrics(previous_snapshot):
    """The stored metrics dict of the previous snapshot (live dict or JSON)."""
    if not previous_snapshot:
        return {}
    metrics = previous_snapshot.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return ts.loads(previous_snapshot.get("metrics_json"), {}) or {}


def _review_note(trade, actions):
    """Why this position is held in review, or None.

    The trade's own flag wins (that is what ``corporate_actions`` set when it
    refused to guess); an outstanding action is the fallback wording.
    """
    if trade.get("needs_review"):
        return trade.get("review_reason") or "an unresolved corporate action needs review"
    outstanding = [a for a in actions or [] if a.get("status") in ca.OPEN_REVIEW_STATUSES]
    if outstanding:
        return ca.review_reason(outstanding[-1])
    return None


# ── snapshot ──────────────────────────────────────────────────────────────

def build_snapshot(trade, bars, trading_date, *, previous_snapshot=None, market_cap=None,
                   actions=(), config=CONFIG):
    """All §9 position numbers for one open trade, plus its monitor state.

    ``actions`` are the trade's recorded corporate actions. They are used two
    ways, neither of which rewrites history:

    * the protective level carried forward from the previous snapshot is
      **rebased** onto today's share basis (a level recorded before a 2-for-1
      split is half as much in today's shares — carrying it forward unchanged
      would fire an instant, entirely fictional stop);
    * dividend income is reported next to price P&L instead of being folded
      into either the average cost or the realised total.
    """
    symbol = trade["symbol"]
    shares = float(trade.get("shares") or 0)
    avg_cost = float(trade.get("average_cost") or 0)
    initial_stop = trade.get("initial_stop")
    actions = list(actions or [])
    prev_date = (previous_snapshot or {}).get("trading_date")
    prev_level = (previous_snapshot or {}).get("current_protective_level")
    if prev_level is not None and prev_date:
        rebased = ca.rebase_price(prev_level, actions, after=prev_date, up_to=trading_date)
        if rebased is not None and abs(rebased - float(prev_level)) > 1e-9:
            logger.info("  %s: protective level %s rebased to %s after a share-basis change",
                        symbol, _r(prev_level), _r(rebased))
        prev_level = rebased
    dividend_income = float(trade.get("dividend_income") or 0)
    realized_pnl = float(trade.get("realized_pnl") or 0)
    review = _review_note(trade, actions)
    snapshot = {
        "trade_id": trade["trade_id"], "symbol": symbol, "trading_date": trading_date,
        "shares": shares, "average_cost": _r(avg_cost), "initial_stop": _r(initial_stop),
        "pivot_price": trade.get("pivot_price"),
        "close": None, "pnl_pct": None, "highest_since_entry": None, "mfe_pct": None, "mae_pct": None,
        "atr20": None, "atr_pct": None, "ema10": None, "ema21": None, "sma50": None, "swing_low": None,
        "current_protective_level": _r(prev_level) if prev_level is not None else _r(initial_stop),
        "distance_to_initial_stop_pct": None, "distance_to_protective_pct": None,
        "days_since_entry": None, "protection_raised": False,
        "previous_monitor_state": (previous_snapshot or {}).get("monitor_state"),
        "price_pnl": None, "dividend_income": _r(dividend_income),
        "total_pnl": _r(realized_pnl + dividend_income),
        "corporate_action_review": review,
        "corporate_actions": [ca.summarize(a) for a in actions],
        "monitor_state": HOLD, "reasons": [], "metrics": {},
    }
    if bars is None or len(bars) < 2 or "Close" not in getattr(bars, "columns", []):
        # §9: no data is NOT a sale. Count the consecutive misses; the caller
        # escalates to a recorded DELISTING review once they pass the threshold.
        missing = int(_previous_metrics(previous_snapshot).get("missing_data_sessions") or 0) + 1
        snapshot["reasons"] = ["no price history available for this symbol"]
        snapshot["metrics"] = {"data": "unavailable", "missing_data_sessions": missing}
        snapshot["monitor_state"] = CORPORATE_ACTION_REVIEW if review else HOLD
        if review:
            snapshot["reasons"] = [str(review)]
        return snapshot

    bars = bars.dropna(subset=["Close"])
    close = float(bars["Close"].iloc[-1])
    prior_close = float(bars["Close"].iloc[-2]) if len(bars) >= 2 else None
    since = _since_entry(bars, _entry_date(trade))
    highest = float(since["High"].max())
    lowest = float(since["Low"].min())
    atr_value = rm.atr(bars, int(config["trade_atr_period"]))
    atr_pct = (atr_value / close * 100) if atr_value and close else None
    ema10, ema21 = rm.ema(bars, 10), rm.ema(bars, 21)
    sma50 = rm.sma(bars, 50)
    swing_low = structural_swing_low(bars, int(config["trade_swing_lookback_days"]),
                                     int(config["trade_swing_exclude_recent_days"]))

    volume = float(bars["Volume"].iloc[-1]) if "Volume" in bars.columns else None
    avg_volume20 = (float(bars["Volume"].iloc[-21:-1].mean())
                    if "Volume" in bars.columns and len(bars) >= 21 else None)
    volume_ratio = (volume / avg_volume20) if volume and avg_volume20 else None
    high, low = float(bars["High"].iloc[-1]), float(bars["Low"].iloc[-1])
    close_location = ((close - low) / (high - low)) if high > low else 0.5
    reversal = bool(volume_ratio is not None
                    and volume_ratio >= float(config["trade_reversal_volume_ratio"])
                    and prior_close is not None and close < prior_close
                    and close_location <= float(config["trade_reversal_close_location"]))

    allowance, allowance_note = ca.dividend_price_allowance(
        actions, latest_date=trading_date, config=config,
        price_basis=config.get("trade_price_basis", ca.PRICE_BASIS))
    price_pnl = (close - avg_cost) * shares if avg_cost else None

    snapshot.update({
        "close": _r(close),
        "pnl_pct": _r((close - avg_cost) / avg_cost * 100) if avg_cost else None,
        "price_pnl": _r(price_pnl),
        "total_pnl": _r((price_pnl or 0) + realized_pnl + dividend_income),
        "highest_since_entry": _r(highest),
        "mfe_pct": _r((highest - avg_cost) / avg_cost * 100) if avg_cost else None,
        "mae_pct": _r((lowest - avg_cost) / avg_cost * 100) if avg_cost else None,
        "atr20": _r(atr_value, 4), "atr_pct": _r(atr_pct),
        "ema10": _r(ema10), "ema21": _r(ema21), "sma50": _r(sma50), "swing_low": _r(swing_low),
        "days_since_entry": max(0, len(since) - 1),
        "metrics": {
            "volume": None if volume is None else round(volume),
            "avg_volume_20d": None if avg_volume20 is None else round(avg_volume20),
            "volume_vs_20d": _r(volume_ratio, 3),
            "close_location_in_range": _r(close_location, 3),
            "high_volume_reversal": reversal,
            "ext_ema10_pct": _r((close - ema10) / ema10 * 100) if ema10 else None,
            "ext_ema21_pct": _r((close - ema21) / ema21 * 100) if ema21 else None,
            "ext_sma50_pct": _r((close - sma50) / sma50 * 100) if sma50 else None,
            "trail_atr_multiple": _r(rm.trail_atr_multiple(market_cap, atr_pct, config), 2),
            "market_cap": market_cap,
            # 0.0 on the adjusted basis this project uses — see §5 in
            # corporate_actions. Never compensate for a dividend twice.
            "dividend_price_allowance": allowance,
            "dividend_price_allowance_note": allowance_note,
            "price_basis": config.get("trade_price_basis", ca.PRICE_BASIS),
            "missing_data_sessions": 0,
        },
    })

    level, basis, notes = protective_level(
        close=close, average_cost=avg_cost, pnl_pct=snapshot["pnl_pct"], initial_stop=initial_stop,
        previous_level=prev_level, ema10=ema10, ema21=ema21, swing_low=swing_low,
        atr_value=atr_value, atr_pct=atr_pct, market_cap=market_cap, config=config)
    baseline = prev_level if prev_level is not None else initial_stop
    snapshot.update({
        "current_protective_level": level,
        "protection_raised": bool(level is not None and baseline is not None and level > float(baseline) + 1e-9),
        "distance_to_initial_stop_pct": (_r((close - float(initial_stop)) / close * 100)
                                         if initial_stop is not None else None),
        "distance_to_protective_pct": _r((close - level) / close * 100) if level is not None else None,
    })
    snapshot["metrics"]["protective_basis"] = basis
    snapshot["metrics"]["protective_notes"] = notes

    state, reasons = classify_position(snapshot, config)
    snapshot["monitor_state"] = state
    snapshot["reasons"] = reasons
    return snapshot


# ── daily run ─────────────────────────────────────────────────────────────

def monitor_open_positions(trading_date, config=CONFIG, *, store=None, fetch=None, market_caps=None):
    """Snapshot every open position for ``trading_date`` and persist the rows.

    Runs whether or not the symbol was screened today — a held stock that has
    dropped out of Stage G is still monitored. Never raises for one bad symbol.

    Corporate actions are expected to have been detected and applied *before*
    this runs (``scheduler.run_trade_layer``), so the trade rows read here are
    already on the current share basis.
    """
    store = store or ts.open_store(config, create=False)
    if store is None:
        return []
    positions = store.open_positions()
    if not positions:
        return []
    fetch = fetch or fetch_bars
    symbols = sorted({p["symbol"] for p in positions})
    logger.info("Position monitor: %d open position(s) for %s (%s).", len(positions), trading_date,
                ", ".join(symbols))
    try:
        bars_by_symbol = fetch(symbols, config) or {}
    except Exception:
        logger.warning("Position monitor: price download failed — positions not updated", exc_info=True)
        return []

    market_caps = market_caps or {}
    review_after = int(config.get("trade_missing_bars_review_sessions", 3))
    snapshots = []
    for trade in positions:
        try:
            previous = store.latest_snapshot(trade["trade_id"], before=trading_date)
            actions = ca.actions_for_trade(store, trade)
            snapshot = build_snapshot(trade, bars_by_symbol.get(trade["symbol"]), trading_date,
                                      previous_snapshot=previous, actions=actions,
                                      market_cap=market_caps.get(trade["symbol"]), config=config)
            missing = int((snapshot.get("metrics") or {}).get("missing_data_sessions") or 0)
            if missing >= review_after and not snapshot.get("corporate_action_review"):
                # §9: market data stopped and stayed stopped. Record it as a
                # corporate action needing review — never as a sale.
                outcome, created = ca.record_delisting(store, trade, trading_date,
                                                       missing_sessions=missing)
                if created:
                    logger.info("  %s: no market data for %d session(s) — recorded for review",
                                trade["symbol"], missing)
                snapshot = build_snapshot(trade, bars_by_symbol.get(trade["symbol"]), trading_date,
                                          previous_snapshot=previous,
                                          actions=ca.actions_for_trade(store, trade),
                                          market_cap=market_caps.get(trade["symbol"]), config=config)
            store.save_snapshot(snapshot)
            if snapshot["monitor_state"] != snapshot.get("previous_monitor_state"):
                store.record_event(trade["trade_id"], ts.EVENT_MONITOR, price=snapshot.get("close"),
                                   shares=snapshot.get("shares"), reason=snapshot["monitor_state"],
                                   detail="; ".join(snapshot.get("reasons") or []) or None)
            snapshots.append(snapshot)
            logger.info("  %s: %s (close %s, P&L %s%%, protective %s)", trade["symbol"],
                        snapshot["monitor_state"], snapshot["close"], snapshot["pnl_pct"],
                        snapshot["current_protective_level"])
        except Exception:
            logger.warning("Position monitor failed for %s — other positions continue",
                           trade["symbol"], exc_info=True)
    return snapshots
