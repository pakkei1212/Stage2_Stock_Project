"""Stage I, Layer 2: the deterministic entry plan for a READY setup.

Advisory arithmetic on top of the frozen Stage-G numbers — it places no order,
and there is no brokerage connection anywhere in this project.

Two rules matter more than the formulas:

* **The numeric detector's pivot is authoritative.** ``pivot_price_candidate``
  (the selected base's right-side resistance) is the entry reference. Claude's
  ``pivot_price`` is carried alongside for context only and never becomes the
  trigger — a vision model reading a chart does not set an execution price.
* **Breakout-volume confirmation reuses the Stage-G threshold**
  ``vcp_breakout_volume_ratio``. This layer introduces no second opinion about
  what a valid breakout looks like.

``trade_entry_trigger_buffer_pct`` and ``trade_max_chase_pct`` are OUR
provisional heuristics (see config.py), not rules from any book.
"""
import logging

from .config import CONFIG
from .telegram_notifier import claude_state, row_vcp_metrics

logger = logging.getLogger(__name__)

AWAIT_BREAKOUT = "AWAIT_BREAKOUT"
BREAKOUT_ACTIONABLE = "BREAKOUT_ACTIONABLE"
BREAKOUT_VOLUME_UNCONFIRMED = "BREAKOUT_VOLUME_UNCONFIRMED"
EXTENDED_NO_CHASE = "EXTENDED_NO_CHASE"
NO_PIVOT = "NO_PIVOT"

ENTRY_STATUSES = (AWAIT_BREAKOUT, BREAKOUT_ACTIONABLE, BREAKOUT_VOLUME_UNCONFIRMED,
                  EXTENDED_NO_CHASE, NO_PIVOT)


def _r(value, digits=2):
    return None if value is None else round(float(value), digits)


def claude_summary(row):
    """The stored Claude verdict for a row, or None when Claude did not review it."""
    if claude_state(row) != "reviewed":
        return None
    return {
        "is_vcp_pattern": bool(row.get("vcp_is_pattern")),
        "pattern_stage": row.get("vcp_pattern_stage"),
        "confidence": row.get("vcp_confidence"),
        "entry_recommendation": row.get("vcp_entry_recommendation"),
        # Informational only — never used as the entry or stop price.
        "pivot_price": _r(row.get("vcp_pivot_price")),
        "suggested_stop_loss": _r(row.get("vcp_stop_loss")),
    }


def build_entry_plan(row, config=CONFIG, setup_state=None):
    """Layer-2 entry plan dict for one gated ``stock_results`` row."""
    m = row_vcp_metrics(row)
    pivot = m.get("pivot_price_candidate")
    current = m.get("current_price") or row.get("last_close")
    required_volume = float(config["vcp_breakout_volume_ratio"])
    buffer_pct = float(config["trade_entry_trigger_buffer_pct"])
    chase_pct = float(config["trade_max_chase_pct"])

    plan = {
        "symbol": row["symbol"],
        "setup_run_id": row.get("run_id"),
        "setup_date": row.get("trading_date"),
        "setup_state": setup_state,
        "final_rank": row.get("final_rank"),
        "pivot_price": _r(pivot),
        "pivot_source": "numeric_vcp_detector",
        "current_price": _r(current),
        "distance_to_pivot_pct": None,
        "entry_status": NO_PIVOT,
        "entry_trigger_price": None,
        "maximum_chase_price": None,
        "max_chase_pct": chase_pct,
        "breakout_volume_ratio": _r(m.get("breakout_volume_vs_20d"), 3),
        "current_volume_ratio_20d": _r(m.get("current_volume_vs_20d"), 3),
        "required_breakout_volume_ratio": required_volume,
        "breakout_volume_confirmed": m.get("breakout_volume_confirmed"),
        "sessions_since_breakout": m.get("sessions_since_breakout"),
        "vcp_numeric_quality": m.get("vcp_numeric_quality"),
        "pivot_state": m.get("pivot_state"),
        "contraction_pcts": m.get("contraction_pcts"),
        "claude": claude_summary(row),
        "notes": [],
    }
    if not pivot or not current:
        plan["notes"].append("no numeric pivot in the stored metrics — no entry plan")
        return plan

    pivot, current = float(pivot), float(current)
    trigger = pivot * (1 + buffer_pct / 100)
    max_chase = pivot * (1 + chase_pct / 100)
    plan.update({
        "distance_to_pivot_pct": _r((current - pivot) / pivot * 100),
        "entry_trigger_price": _r(trigger),
        "maximum_chase_price": _r(max_chase),
    })

    if current > max_chase:
        plan["entry_status"] = EXTENDED_NO_CHASE
        plan["notes"].append(f"price is {plan['distance_to_pivot_pct']}% above the pivot, beyond the "
                             f"{chase_pct}% chase limit")
    elif current < trigger:
        plan["entry_status"] = AWAIT_BREAKOUT
        plan["notes"].append(f"coiling below the pivot; trigger at {plan['entry_trigger_price']}")
    elif plan["breakout_volume_confirmed"] is False:
        plan["entry_status"] = BREAKOUT_VOLUME_UNCONFIRMED
        plan["notes"].append(f"breakout volume {plan['breakout_volume_ratio']}x 20d avg is below the "
                             f"required {required_volume}x")
    else:
        plan["entry_status"] = BREAKOUT_ACTIONABLE
        if plan["breakout_volume_confirmed"] is None:
            plan["notes"].append("breakout volume not measurable from the stored metrics")
    return plan


def is_actionable(plan):
    return plan.get("entry_status") in (AWAIT_BREAKOUT, BREAKOUT_ACTIONABLE)
