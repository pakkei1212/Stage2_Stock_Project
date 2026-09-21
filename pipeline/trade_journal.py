"""Stage I service layer: setups -> plans -> recorded trades.

The one place that ties the pieces together, shared by the Telegram bot and the
CLI so both behave identically:

* ``generate_setup_plans`` — after a Stage-H run, gate every Stage-G row
  (:mod:`setup_gate`) and, for the plannable ones, compute and persist the
  Layer-2 entry plan, the Layer-3 risk plan and the position size.
* ``preview_*`` / ``apply_*`` — every mutation is a two-step: build a preview
  payload (pure, writes nothing) and apply it only once the user confirms.
  Nothing reaches ``trades`` / ``trade_fills`` without that confirmation.

Advisory only. No broker is contacted at any point: a "fill" here is a number
the user typed in after trading somewhere else entirely.
"""
import logging

from . import corporate_actions as ca
from . import entry_plan as ep
from . import position_monitor as pm
from . import risk_model as rm
from . import setup_gate as gate
from . import trade_store as tstore
from .config import CONFIG
from .telegram_notifier import row_vcp_metrics

logger = logging.getLogger(__name__)


def _r(value, digits=2):
    return None if value is None else round(float(value), digits)


# ── Layer 2 + Layer 3 for stored Stage-G rows ─────────────────────────────

def build_plan_for_row(row, bars, config=CONFIG, setup_state=None, setup_reasons=None):
    """Entry plan + risk plan + sizing for one gated ``stock_results`` row."""
    metrics = row_vcp_metrics(row)
    entry = ep.build_entry_plan(row, config, setup_state=setup_state)
    reference_price = entry.get("entry_trigger_price") or entry.get("current_price")
    risk = rm.build_risk_plan(
        reference_price, bars,
        structural_low=rm.structural_low_from_metrics(metrics),
        market_cap=row.get("market_cap"),
        config=config,
    )
    sizing = rm.size_for_plan(reference_price, risk, config)
    return {
        "run_id": row.get("run_id"),
        "trading_date": row.get("trading_date"),
        "symbol": row["symbol"],
        "setup_state": setup_state,
        "setup_reasons": list(setup_reasons or []),
        "vcp_numeric_quality": metrics.get("vcp_numeric_quality"),
        "pivot_state": metrics.get("pivot_state"),
        "entry_status": entry.get("entry_status"),
        "trade_plan_status": risk.get("trade_plan_status"),
        "pivot_price": entry.get("pivot_price"),
        "current_price": entry.get("current_price"),
        "entry_trigger_price": entry.get("entry_trigger_price"),
        "maximum_chase_price": entry.get("maximum_chase_price"),
        "suggested_initial_stop": risk.get("suggested_initial_stop"),
        "required_risk_pct": risk.get("required_risk_pct"),
        "allowed_max_risk_pct": risk.get("allowed_max_risk_pct"),
        "suggested_shares": sizing.get("suggested_shares"),
        "position_portion_pct": sizing.get("position_portion_pct"),
        "entry_plan": entry,
        "risk_plan": risk,
        "sizing": sizing,
    }


def generate_setup_plans(run_id, trading_date, results, config=CONFIG, *, store=None, fetch=None):
    """Gate a run's Stage-G rows and persist a plan row for each.

    Every Stage-G symbol gets its gate state recorded (including NOT_READY /
    MANUAL_REVIEW / MISSED_EXTENDED, which are useful review context); only
    READY / BREAKOUT_TRIGGERED setups get Layer-2 and Layer-3 plans.
    """
    gated = gate.gate_rows(results)
    if not gated:
        return []
    store = store or tstore.open_store(config, create=True)
    plannable = [(row, state, reasons) for row, state, reasons in gated if gate.is_plannable(state)]

    bars_by_symbol = {}
    if plannable:
        fetch = fetch or pm.fetch_bars
        try:
            bars_by_symbol = fetch(sorted({row["symbol"] for row, _, _ in plannable}), config) or {}
        except Exception:
            logger.warning("Entry/risk plans: price download failed — plans limited to Stage-G metrics",
                           exc_info=True)

    plans = []
    for row, state, reasons in gated:
        try:
            if gate.is_plannable(state):
                plan = build_plan_for_row(row, bars_by_symbol.get(row["symbol"]), config, state, reasons)
            else:
                metrics = row_vcp_metrics(row)
                plan = {
                    "run_id": row.get("run_id"), "trading_date": row.get("trading_date"),
                    "symbol": row["symbol"], "setup_state": state, "setup_reasons": list(reasons),
                    "vcp_numeric_quality": metrics.get("vcp_numeric_quality"),
                    "pivot_state": metrics.get("pivot_state"),
                    "pivot_price": metrics.get("pivot_price_candidate"),
                    "current_price": metrics.get("current_price"),
                }
            store.save_setup_plan(plan)
            plans.append(plan)
        except Exception:
            logger.warning("Setup plan failed for %s — other setups continue", row.get("symbol"),
                           exc_info=True)
    ready = [p for p in plans if gate.is_plannable(p.get("setup_state"))]
    logger.info("Setup gate: %d Stage-G setup(s) evaluated, %d plannable (%s).", len(plans), len(ready),
                ", ".join(f"{p['symbol']}:{p['setup_state']}" for p in ready) or "none")
    return plans


# ── previews (pure) and applies (the only writers) ────────────────────────

def _setup_context(store, symbol):
    """The most recent recorded gate state / plan for a symbol, if any.

    Copied verbatim from the persisted ``setup_plans`` row (nothing recomputed);
    it travels in the pending-action payload, so Confirm applies exactly what
    the user was shown, and becomes the trade's planned-trade record."""
    if store is None:
        return None
    plan = store.latest_setup_plan(symbol)
    if plan is None:
        return None
    return {
        "plan_id": plan.get("plan_id"),
        "run_id": plan.get("run_id"),
        "trading_date": plan.get("trading_date"),
        "setup_state": plan.get("setup_state"),
        "vcp_numeric_quality": plan.get("vcp_numeric_quality"),
        "pivot_state": plan.get("pivot_state"),
        "pivot_price": plan.get("pivot_price"),
        "current_price": plan.get("current_price"),
        "entry_status": plan.get("entry_status"),
        "trade_plan_status": plan.get("trade_plan_status"),
        "entry_trigger_price": plan.get("entry_trigger_price"),
        "maximum_chase_price": plan.get("maximum_chase_price"),
        "suggested_initial_stop": plan.get("suggested_initial_stop"),
        "required_risk_pct": plan.get("required_risk_pct"),
        "allowed_max_risk_pct": plan.get("allowed_max_risk_pct"),
        "suggested_shares": plan.get("suggested_shares"),
        "position_portion_pct": plan.get("position_portion_pct"),
        "entry_plan": plan.get("entry_plan"),
        "risk_plan": plan.get("risk_plan"),
        "sizing": plan.get("sizing"),
    }


#: Tolerance before an actual portfolio portion counts as "above plan"
#: (rounding of the stored plan, not a trading threshold).
_PORTION_TOLERANCE_PCT = 0.05


def compare_to_plan(setup, price, shares, portfolio_portion_pct):
    """Planned vs actual for a new position. Pure; warnings never block a record.

    Reads the persisted plan values only — the plan itself is not recomputed or
    modified. Returns None when there is no plan with an entry trigger.
    """
    if not setup or setup.get("entry_trigger_price") is None:
        return None
    trigger = float(setup["entry_trigger_price"])
    chase = setup.get("maximum_chase_price")
    suggested = setup.get("suggested_shares")
    planned_portion = setup.get("position_portion_pct")
    out = {
        "planned_entry_trigger": _r(trigger),
        "planned_maximum_chase": _r(chase),
        "planned_shares": suggested,
        "planned_portion_pct": _r(planned_portion),
        "actual_price": _r(price),
        "actual_shares": shares,
        "actual_portion_pct": _r(portfolio_portion_pct),
        "slippage_from_trigger_pct": _r((price - trigger) / trigger * 100) if trigger else None,
        "warnings": [],
    }
    if price < trigger:
        out["warnings"].append(f"Fill is below planned breakout trigger (${trigger:,.2f})")
    if chase is not None and price > float(chase):
        out["warnings"].append(f"Fill is above maximum chase price (${float(chase):,.2f})")
    if suggested is not None and shares > float(suggested):
        out["warnings"].append(f"Actual shares exceed suggested position size ({float(suggested):g})")
    if (planned_portion is not None and portfolio_portion_pct is not None
            and portfolio_portion_pct > float(planned_portion) + _PORTION_TOLERANCE_PCT):
        out["warnings"].append(f"Actual portfolio portion exceeds plan "
                               f"({portfolio_portion_pct:.1f}% vs {float(planned_portion):.1f}%)")
    return out


def preview_buy(symbol, price, shares, config=CONFIG, *, store=None, notes=None):
    """Payload describing the buy that WOULD be recorded. Writes nothing."""
    symbol = str(symbol).upper()
    price, shares = float(price), float(shares)
    if price <= 0 or shares <= 0:
        raise tstore.TradeError("price and shares must both be greater than zero")
    setup = _setup_context(store, symbol)
    existing = store.open_trade(symbol) if store else None
    portfolio_value = float(config["trade_portfolio_value"])
    value = price * shares
    payload = {
        "action": "buy", "symbol": symbol, "price": price, "shares": shares, "value": _r(value),
        "notes": notes,
        "portfolio_value": _r(portfolio_value),
        "portfolio_portion_pct": _r(value / portfolio_value * 100) if portfolio_value else None,
        "suggested_initial_stop": (setup or {}).get("suggested_initial_stop"),
        "setup": setup,
        "adds_to_existing": bool(existing),
        "existing_shares": float(existing["shares"]) if existing else 0.0,
        "existing_average_cost": _r(existing["average_cost"]) if existing else None,
        "warnings": [],
    }
    stop = payload["suggested_initial_stop"]
    if stop:
        payload["risk_pct_at_stop"] = _r((price - float(stop)) / price * 100)
        payload["risk_amount"] = _r((price - float(stop)) * shares)
    if setup is None:
        payload["warnings"].append("no recorded Stage-G setup for this symbol — recorded without a plan")
    elif setup.get("setup_state") not in gate.PLANNABLE_STATES:
        payload["warnings"].append(f"latest setup state is {setup.get('setup_state')}, "
                                   f"not {'/'.join(gate.PLANNABLE_STATES)}")
    if setup and setup.get("trade_plan_status") == rm.SKIP_RISK_TOO_WIDE:
        payload["warnings"].append("the planned stop was wider than the configured risk ceiling "
                                   "(SKIP_RISK_TOO_WIDE)")
    if payload["adds_to_existing"]:
        payload["warnings"].append(f"adds to an existing position of {payload['existing_shares']:g} shares")
    else:
        payload["plan_comparison"] = compare_to_plan(setup, price, shares, payload["portfolio_portion_pct"])
    return payload


def apply_buy(payload, config=CONFIG, *, store=None, timestamp=None):
    """Record a confirmed buy. The only path into ``trades`` / ``trade_fills``."""
    store = store or tstore.open_store(config, create=True)
    setup = payload.get("setup") or {}
    trade_id, opened_new = store.record_buy(
        payload["symbol"], payload["price"], payload["shares"], timestamp=timestamp,
        setup={"setup_run_id": setup.get("run_id"), "setup_date": setup.get("trading_date"),
               "setup_state": setup.get("setup_state"), "vcp_quality": setup.get("vcp_numeric_quality"),
               "pivot_price": setup.get("pivot_price")},
        initial_stop=payload.get("suggested_initial_stop"),
        portfolio_portion_pct=payload.get("portfolio_portion_pct"),
        portfolio_value=payload.get("portfolio_value"),
        # The planned trade, kept beside the actual one for later evaluation.
        # Stored on a new trade only; the setup_plans row itself is never modified.
        entry_plan=setup.get("entry_plan"),
        risk_plan=_planned_risk(setup),
        notes=payload.get("notes"),
        event_detail=_plan_vs_actual_record(payload),
    )
    trade = store.get_trade(trade_id)
    return {"trade_id": trade_id, "opened_new": opened_new, "trade": trade}


def _planned_risk(setup):
    """Risk plan + sizing as persisted, for ``trades.risk_plan_json``."""
    if not setup.get("risk_plan") and not setup.get("sizing"):
        return None
    return {**(setup.get("risk_plan") or {}), "sizing": setup.get("sizing")}


def _plan_vs_actual_record(payload):
    """Audit record for the OPEN/BUY event: which plan was shown, what was
    actually recorded, and every warning the user confirmed through."""
    setup = payload.get("setup")
    if not setup:
        return None
    planned = {k: setup.get(k) for k in (
        "plan_id", "run_id", "trading_date", "setup_state", "entry_status", "trade_plan_status",
        "pivot_price", "entry_trigger_price", "maximum_chase_price", "suggested_initial_stop",
        "required_risk_pct", "allowed_max_risk_pct", "suggested_shares", "position_portion_pct")}
    comparison = payload.get("plan_comparison")
    return {
        "planned": planned,
        "actual": {"price": payload.get("price"), "shares": payload.get("shares"),
                   "value": payload.get("value"), "portfolio_portion_pct": payload.get("portfolio_portion_pct")},
        "comparison": comparison,
        "warnings": list(payload.get("warnings") or []) + list((comparison or {}).get("warnings") or []),
    }


def preview_sell(symbol, price, shares=None, config=CONFIG, *, store=None, notes=None):
    """Payload for a partial sell (``shares``) or a full close (``shares=None``)."""
    symbol = str(symbol).upper()
    price = float(price)
    if price <= 0:
        raise tstore.TradeError("price must be greater than zero")
    trade = store.open_trade(symbol) if store else None
    if trade is None:
        raise tstore.TradeError(f"no open position in {symbol}")
    held = float(trade["shares"] or 0)
    qty = held if shares is None else float(shares)
    if qty <= 0:
        raise tstore.TradeError("shares must be greater than zero")
    if qty > held + 1e-9:
        raise tstore.TradeError(f"cannot sell {qty:g} {symbol}: only {held:g} held")
    avg = float(trade["average_cost"] or 0)
    return {
        "action": "close" if shares is None or qty >= held - 1e-9 else "sell",
        "symbol": symbol, "price": price, "shares": qty, "notes": notes,
        "held_shares": held, "average_cost": _r(avg),
        "shares_remaining": _r(held - qty, 6),
        "realized_pnl": _r((price - avg) * qty),
        "realized_pnl_pct": _r((price - avg) / avg * 100) if avg else None,
        "value": _r(price * qty),
        "closes_position": qty >= held - 1e-9,
        "trade_id": trade["trade_id"],
        "warnings": [],
    }


def preview_close(symbol, price, config=CONFIG, *, store=None, notes=None):
    return preview_sell(symbol, price, None, config, store=store, notes=notes)


def apply_sell(payload, config=CONFIG, *, store=None, timestamp=None):
    """Record a confirmed sell / close."""
    store = store or tstore.open_store(config, create=True)
    shares = None if payload.get("closes_position") else payload["shares"]
    result = store.record_sell(payload["symbol"], payload["price"], shares, timestamp=timestamp,
                               reason=payload.get("reason"), notes=payload.get("notes"))
    result["trade"] = store.get_trade(result["trade_id"])
    return result


apply_close = apply_sell


# ── read-only views ───────────────────────────────────────────────────────

def positions_view(config=CONFIG, *, store=None):
    """Open positions joined with their most recent monitor snapshot."""
    store = store or tstore.open_store(config, create=False)
    if store is None:
        return []
    out = []
    for trade in store.open_positions():
        snapshot = store.latest_snapshot(trade["trade_id"])
        out.append({"trade": trade, "snapshot": snapshot})
    return out


def position_detail(symbol, config=CONFIG, *, store=None, snapshot_limit=10):
    store = store or tstore.open_store(config, create=False)
    if store is None:
        return None
    trade = store.open_trade(symbol)
    if trade is None:
        history = store.trades_for_symbol(symbol, limit=1)
        if not history:
            return None
        trade = history[0]
    actions = ca.actions_for_trade(store, trade)
    return {
        "trade": trade,
        "fills": store.fills(trade["trade_id"]),
        "snapshot": store.latest_snapshot(trade["trade_id"]),
        "snapshots": store.snapshots_for_trade(trade["trade_id"], limit=snapshot_limit),
        "events": store.events(trade["trade_id"], limit=10),
        # §13: original fills + corporate actions + sales = the current position.
        "corporate_actions": actions,
        "open_reviews": [a for a in actions if a.get("status") in ca.OPEN_REVIEW_STATUSES],
    }


def actions_view(symbol=None, config=CONFIG, *, store=None, limit=None):
    """Recorded corporate actions — one symbol, or all of them (§13)."""
    store = store or tstore.open_store(config, create=False)
    if store is None:
        return []
    return store.corporate_actions(symbol, limit=limit)


def resolve_corporate_review(symbol, config=CONFIG, *, store=None, notes=None, initial_stop=None,
                             pivot_price=None):
    """Clear a position's corporate-action review after the user has dealt with it.

    Optionally records the stop/pivot the *user* decided on. The app never picks
    those numbers itself for an ambiguous action — that is the whole point of
    the review.
    """
    store = store or tstore.open_store(config, create=False)
    if store is None:
        raise tstore.TradeError("no trade journal recorded yet")
    trade = store.open_trade(symbol)
    if trade is None:
        raise tstore.TradeError(f"no open position in {str(symbol).upper()}")
    fields = {}
    if initial_stop is not None:
        fields["initial_stop"] = float(initial_stop)
    if pivot_price is not None:
        fields["pivot_price"] = float(pivot_price)
    if fields:
        store.update_trade(trade["trade_id"], **fields)
    resolved = ca.resolve_reviews(store, trade, notes=notes)
    return {"trade_id": trade["trade_id"], "symbol": trade["symbol"], "resolved": resolved,
            "updated": fields}


def trades_view(config=CONFIG, *, store=None, limit=20, status=None):
    store = store or tstore.open_store(config, create=False)
    if store is None:
        return []
    return store.recent_trades(limit=limit, status=status)
