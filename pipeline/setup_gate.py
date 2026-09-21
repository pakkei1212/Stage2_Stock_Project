"""Stage I, gate: is a stored Stage-G setup ready for an entry plan?

Reads the *stored* Stage-G output of a run (``stock_results`` rows) and answers
one question: should Layer 2 (entry plan) and Layer 3 (risk plan) be generated
for this symbol? It never recomputes, reinterprets or overwrites the VCP
verdict — the numeric detector and Claude's verdict are inputs, and both are
left exactly as Stage G wrote them.

States::

    NOT_READY            numeric quality or pivot position rules it out
    MANUAL_REVIEW        clean numerically, but Claude reviewed it and said "no VCP"
    READY                coiling below the pivot: plan the breakout entry
    BREAKOUT_TRIGGERED   fresh breakout above the pivot: entry window is live
    MISSED_EXTENDED      already extended above the pivot — no chase

The rule (provisional, configurable through the Stage-G keys it reads)::

    numeric quality in {strong, acceptable}
    AND pivot_state in {coiling_below_pivot, fresh_breakout}
    AND NOT far_below_pivot / extended_above_pivot
    AND NOT (Claude actually reviewed it and returned is_vcp_pattern = false)

"Claude not requested" (selective review skipped it, review off, token-free
mode) is *not* a negative verdict and never blocks a clean numeric setup —
only a real review that disagrees does, and then the setup is parked in
MANUAL_REVIEW rather than silently dropped or silently promoted.
"""
import logging

# Stage-H reader helpers for a stock_results row. Importing them keeps a single
# definition of "did Claude actually review this row?" in the project.
from .telegram_notifier import claude_state, row_vcp_metrics

logger = logging.getLogger(__name__)

NOT_READY = "NOT_READY"
MANUAL_REVIEW = "MANUAL_REVIEW"
READY = "READY"
BREAKOUT_TRIGGERED = "BREAKOUT_TRIGGERED"
MISSED_EXTENDED = "MISSED_EXTENDED"

SETUP_STATES = (NOT_READY, MANUAL_REVIEW, READY, BREAKOUT_TRIGGERED, MISSED_EXTENDED)

#: States for which Layer 2 / Layer 3 plans are generated.
PLANNABLE_STATES = (READY, BREAKOUT_TRIGGERED)

READY_QUALITIES = ("strong", "acceptable")
READY_PIVOT_STATES = ("coiling_below_pivot", "fresh_breakout")
BLOCKING_PIVOT_STATES = ("far_below_pivot", "extended_above_pivot")


def evaluate_setup(row):
    """(state, reasons) for one ``stock_results`` row. Pure; no I/O."""
    metrics = row_vcp_metrics(row)
    quality = metrics.get("vcp_numeric_quality")
    pivot_state = metrics.get("pivot_state")
    reviewed = claude_state(row) == "reviewed"
    claude_says_no = reviewed and row.get("vcp_is_pattern") in (0, False)
    reasons = []

    if quality not in READY_QUALITIES:
        reasons.append(f"numeric quality {quality or 'n/a'} not in {'/'.join(READY_QUALITIES)}")
        return NOT_READY, reasons
    if pivot_state == "extended_above_pivot":
        reasons.append(f"extended above pivot ({metrics.get('pct_from_pivot')}%) — no chase")
        return MISSED_EXTENDED, reasons
    if pivot_state not in READY_PIVOT_STATES:
        reasons.append(f"pivot state {pivot_state or 'unknown'} not in {'/'.join(READY_PIVOT_STATES)}")
        return NOT_READY, reasons
    if claude_says_no:
        reasons.append(f"numeric {quality} setup, but Claude reviewed it and returned is_vcp_pattern=false")
        return MANUAL_REVIEW, reasons

    reasons.append(f"numeric {quality} · {pivot_state}")
    if not reviewed:
        reasons.append("Claude review not requested (does not block a clean numeric setup)")
    if pivot_state == "fresh_breakout":
        return BREAKOUT_TRIGGERED, reasons
    return READY, reasons


def is_plannable(state):
    return state in PLANNABLE_STATES


def gate_rows(rows):
    """[(row, state, reasons)] for every Stage-G row in a run, ranked order preserved."""
    out = []
    for row in rows:
        if claude_state(row) is None:        # Stage G never looked at this symbol
            continue
        state, reasons = evaluate_setup(row)
        out.append((row, state, reasons))
    return out


def plannable_rows(rows):
    return [(row, state, reasons) for row, state, reasons in gate_rows(rows) if is_plannable(state)]
