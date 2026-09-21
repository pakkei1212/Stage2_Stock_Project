"""Stage I: auditable corporate-action handling for the trade journal.

Strictly downstream of Stage A-G and strictly advisory. This module never
screens, never scores, never touches VCP metrics, and — like the rest of Stage
I — never connects to a broker: it records events, adjusts *derived* journal
values, and flags anything ambiguous for the user to resolve by hand.

Price-data basis (the thing that makes all of this correct — see README
"Corporate actions"):

* **Technical OHLCV is split- AND dividend-adjusted.** Every bar in this
  project comes from ``data_sources.batch_download`` -> ``yf.download(...,
  auto_adjust=True)``, which back-adjusts Open/High/Low/Close for both splits
  and dividends (Volume for splits only). ``ohlcv_cache.merge`` keeps the cache
  on one basis by rescaling stored history whenever the overlap's median close
  ratio moves more than ``SPLIT_TOL``. The newest bar is therefore always the
  real traded price, and the history behind it is on that same basis.
* **Recorded fills are raw as-traded prices** on the share basis of the day the
  user traded. They are immutable.
* **Stored pivots/stops** were derived from the adjusted series on the run date,
  which for the latest bar equals the as-traded basis — so they live on the same
  basis as the fills.

The consequence, and the rule this module exists to enforce:

    The bars need no correction from us. Only the *journal's* derived values
    (shares, average cost, stops, pivots, protective levels) fall out of step
    when the share basis changes, and those are what we adjust. Applying a
    second price adjustment on top of Yahoo's would double-count.

``PRICE_BASIS`` states that contract in code; :func:`dividend_price_allowance`
is the single place a raw-price basis would be compensated for, and it returns
0.0 for the adjusted basis we actually have.
"""
import hashlib
import json
import logging
from datetime import date, datetime

from .config import CONFIG

logger = logging.getLogger(__name__)

#: What ``data_sources.batch_download`` hands us. Not a knob: it is a statement
#: about yfinance's ``auto_adjust=True``, asserted by the regression tests.
PRICE_BASIS = "split_and_dividend_adjusted"
RAW_PRICE_BASIS = "raw"

# ── action types ──────────────────────────────────────────────────────────

SPLIT = "SPLIT"
REVERSE_SPLIT = "REVERSE_SPLIT"
CASH_DIVIDEND = "CASH_DIVIDEND"
SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
STOCK_DIVIDEND = "STOCK_DIVIDEND"
SYMBOL_CHANGE = "SYMBOL_CHANGE"
MERGER = "MERGER"
SPINOFF = "SPINOFF"
RIGHTS = "RIGHTS"
DELISTING = "DELISTING"
OTHER = "OTHER"

ACTION_TYPES = (SPLIT, REVERSE_SPLIT, CASH_DIVIDEND, SPECIAL_DIVIDEND, STOCK_DIVIDEND,
                SYMBOL_CHANGE, MERGER, SPINOFF, RIGHTS, DELISTING, OTHER)

#: Actions that restate the share basis: ``shares *= ratio``, per-share
#: references ``/= ratio``. Economic position value is unchanged.
SHARE_BASIS_TYPES = (SPLIT, REVERSE_SPLIT, STOCK_DIVIDEND)
DIVIDEND_TYPES = (CASH_DIVIDEND, SPECIAL_DIVIDEND)

#: §14 — the ONLY actions whose arithmetic is unambiguous enough to apply
#: without asking. Everything else becomes a review event.
AUTO_APPLY_TYPES = SHARE_BASIS_TYPES + DIVIDEND_TYPES + (SYMBOL_CHANGE,)

#: §8/§9 — recorded and flagged, never guessed at.
REVIEW_TYPES = (MERGER, SPINOFF, RIGHTS, DELISTING, OTHER)

# ── statuses ──────────────────────────────────────────────────────────────

DETECTED = "DETECTED"
APPLIED = "APPLIED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
NOT_APPLICABLE = "NOT_APPLICABLE"
RESOLVED = "RESOLVED"

STATUSES = (DETECTED, APPLIED, REVIEW_REQUIRED, AWAITING_CONFIRMATION, NOT_APPLICABLE, RESOLVED)

#: Statuses that hold a position in CORPORATE_ACTION_REVIEW.
OPEN_REVIEW_STATUSES = (REVIEW_REQUIRED, AWAITING_CONFIRMATION)

SOURCE_YFINANCE = "yfinance"
SOURCE_MANUAL = "manual"
SOURCE_MONITOR = "monitor"


class CorporateActionError(Exception):
    """A rejected corporate-action operation (bad ratio, unknown type)."""


# ── small helpers ─────────────────────────────────────────────────────────

def to_date_str(value):
    """``YYYY-MM-DD`` for a date/datetime/Timestamp/string."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    text = str(value)
    return text[:10]


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _r(value, digits=6):
    return None if value is None else round(float(value), digits)


def is_share_basis(action_type):
    return action_type in SHARE_BASIS_TYPES


def classify_split(ratio):
    """SPLIT for ratio > 1 (2-for-1 = 2.0), REVERSE_SPLIT for ratio < 1 (1-for-5 = 0.2)."""
    value = _num(ratio)
    if value is None or value <= 0 or value != value:      # NaN-safe
        raise CorporateActionError(f"split ratio must be a positive number, got {ratio!r}")
    if abs(value - 1.0) < 1e-9:
        raise CorporateActionError("a 1:1 split changes nothing")
    return SPLIT if value > 1 else REVERSE_SPLIT


def describe_ratio(ratio):
    """Human wording for a share-basis ratio: 2.0 -> '2-for-1', 0.2 -> '1-for-5'."""
    value = float(ratio)
    if value >= 1:
        whole = round(value)
        if abs(value - whole) < 1e-9:
            return f"{whole}-for-1"
        return f"{value:g}-for-1"
    inverse = 1.0 / value
    whole = round(inverse)
    if abs(inverse - whole) < 1e-9:
        return f"1-for-{whole}"
    return f"1-for-{inverse:g}"


def fingerprint(symbol, effective_date, action_type, value=None, source=None):
    """Stable identity of one event — the idempotency key (§2).

    The same upstream event always hashes the same way, so re-detecting it on
    every daily cycle inserts nothing and applies nothing a second time.
    """
    amount = "" if value is None else f"{float(value):.10g}"
    blob = "|".join([str(symbol).upper(), to_date_str(effective_date) or "", str(action_type),
                     amount, str(source or "")])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def make_action(symbol, effective_date, action_type, *, split_ratio=None, cash_amount=None,
                currency="USD", old_symbol=None, new_symbol=None, source=SOURCE_YFINANCE,
                source_event_id=None, details=None, status=None, notes=None, trade_id=None):
    """A corporate-action row (plain dict) with its fingerprint and default status."""
    if action_type not in ACTION_TYPES:
        raise CorporateActionError(f"unknown action type {action_type!r}")
    symbol = str(symbol).upper()
    effective = to_date_str(effective_date)
    if not effective:
        raise CorporateActionError("an effective date is required")
    try:
        date.fromisoformat(effective)
    except ValueError:
        raise CorporateActionError(f"{effective_date!r} is not a YYYY-MM-DD date")
    if action_type in SHARE_BASIS_TYPES:
        ratio = _num(split_ratio)
        if not ratio or ratio <= 0:
            raise CorporateActionError(f"{action_type} needs a positive share ratio "
                                       f"(2 = 2-for-1, 0.2 = 1-for-5), got {split_ratio!r}")
        if action_type in (SPLIT, REVERSE_SPLIT) and classify_split(ratio) != action_type:
            raise CorporateActionError(
                f"a ratio of {ratio:g} is a {classify_split(ratio)}, not a {action_type}")
    if action_type in DIVIDEND_TYPES and not (_num(cash_amount) or 0) > 0:
        raise CorporateActionError(f"{action_type} needs a positive cash amount per share")
    value = split_ratio if action_type in SHARE_BASIS_TYPES else cash_amount
    if action_type == SYMBOL_CHANGE:
        value = None
    action = {
        "symbol": symbol,
        "effective_date": effective,
        "action_type": action_type,
        "status": status or (DETECTED if action_type in AUTO_APPLY_TYPES else REVIEW_REQUIRED),
        "split_ratio": _num(split_ratio),
        "cash_amount": _num(cash_amount),
        "currency": currency,
        "old_symbol": (str(old_symbol).upper() if old_symbol else None),
        "new_symbol": (str(new_symbol).upper() if new_symbol else None),
        "source": source,
        "source_event_id": source_event_id,
        "details": dict(details or {}),
        "notes": notes,
        "trade_id": trade_id,
    }
    suffix = action["new_symbol"] if action_type == SYMBOL_CHANGE else None
    action["fingerprint"] = fingerprint(symbol, effective, action_type,
                                        value if suffix is None else None,
                                        suffix or source)
    return action


# ── share-basis arithmetic (§3, §6) ───────────────────────────────────────

def basis_factor(actions, *, after=None, up_to=None):
    """Cumulative share-basis ratio of the APPLIED share-basis actions in a window.

    ``after`` / ``up_to`` are inclusive-exclusive on the effective date:
    ``after < effective_date <= up_to``. Returns 1.0 when nothing applies, so
    callers can divide/multiply unconditionally.
    """
    after, up_to = to_date_str(after), to_date_str(up_to)
    factor = 1.0
    for action in actions or []:
        if action.get("action_type") not in SHARE_BASIS_TYPES:
            continue
        if action.get("status") != APPLIED:
            continue
        effective = to_date_str(action.get("effective_date"))
        ratio = _num(action.get("split_ratio"))
        if not effective or not ratio or ratio <= 0:
            continue
        if after is not None and effective <= after:
            continue
        if up_to is not None and effective > up_to:
            continue
        factor *= ratio
    return factor


def rebase_price(price, actions, *, after=None, up_to=None):
    """Restate a per-share price recorded before ``after`` onto today's basis.

    A stored protective level from before a 2-for-1 split is half as much in
    today's shares. Snapshots are historical records and are never rewritten;
    the monitor rebases what it carries forward instead.
    """
    value = _num(price)
    if value is None:
        return None
    factor = basis_factor(actions, after=after, up_to=up_to)
    return value / factor if factor else value


def rebase_shares(shares, actions, *, after=None, up_to=None):
    value = _num(shares)
    if value is None:
        return None
    return value * basis_factor(actions, after=after, up_to=up_to)


def adjusted_trade_fields(trade, ratio):
    """The journal fields a share-basis action restates. Pure — writes nothing.

    Economic value is preserved exactly: ``shares * average_cost`` is invariant
    (up to float rounding), and realised P&L / dividend income already earned
    are deliberately absent — they are money, not share-denominated references.
    """
    ratio = _num(ratio)
    if not ratio or ratio <= 0:
        raise CorporateActionError(f"share-basis ratio must be positive, got {ratio!r}")

    def per_share(value):
        value = _num(value)
        return None if value is None else _r(value / ratio)

    return {
        "shares": _r((_num(trade.get("shares")) or 0.0) * ratio, 10),
        "average_cost": per_share(trade.get("average_cost")),
        "total_bought_shares": _r((_num(trade.get("total_bought_shares")) or 0.0) * ratio, 10),
        "total_sold_shares": _r((_num(trade.get("total_sold_shares")) or 0.0) * ratio, 10),
        "initial_stop": per_share(trade.get("initial_stop")),
        "pivot_price": per_share(trade.get("pivot_price")),
    }


# ── dividends (§4) ────────────────────────────────────────────────────────

def eligible_shares_at(fills, ex_date, actions=()):
    """(shares, certain, reason) held immediately before ``ex_date``.

    Shares must be held *before* the ex-date to be entitled, so only fills
    strictly earlier count. A fill dated on the ex-date itself cannot be
    resolved from what we store (no trade time, no settlement data), so the
    entitlement is reported as uncertain and the caller records it for review
    rather than guessing (§4).

    Fill quantities are raw as-traded shares, so any share-basis action between
    the fill and the ex-date is applied to them first.
    """
    ex_date = to_date_str(ex_date)
    held = 0.0
    on_ex_date = False
    for fill in fills or []:
        fill_date = to_date_str(fill.get("timestamp"))
        if not fill_date:
            continue
        if fill_date == ex_date:
            on_ex_date = True
            continue
        if fill_date > ex_date:
            continue
        shares = _num(fill.get("shares")) or 0.0
        shares *= basis_factor(actions, after=fill_date, up_to=ex_date)
        held += shares if str(fill.get("side")).upper() == "BUY" else -shares

    held = _r(held, 10)
    if on_ex_date:
        return held, False, (f"a fill is dated on the ex-date {ex_date} — entitlement cannot be "
                             f"determined from the recorded fills")
    if held <= 1e-9:
        return 0.0, True, f"no shares were held before the ex-date {ex_date}"
    return held, True, None


def is_special_dividend(amount, *, reference_price=None, regular_amounts=(), config=CONFIG):
    """Heuristic (OURS, not a rule from any book): a payment far outside the
    normal cadence — a big one-off yield, or a large multiple of the recent
    regular dividends. Both thresholds are ``trade_*`` config keys."""
    amount = _num(amount)
    if not amount or amount <= 0:
        return False, None
    yield_pct = float(config.get("trade_special_dividend_yield_pct", 3.0))
    multiple = float(config.get("trade_special_dividend_multiple", 3.0))

    reference_price = _num(reference_price)
    if reference_price and reference_price > 0 and (amount / reference_price * 100) >= yield_pct:
        return True, (f"{amount:g} per share is {amount / reference_price * 100:.1f}% of the "
                      f"{reference_price:g} reference price")
    regular = sorted(a for a in (_num(x) or 0.0 for x in regular_amounts) if a > 0)
    if regular:
        median = regular[len(regular) // 2] if len(regular) % 2 else \
            (regular[len(regular) // 2 - 1] + regular[len(regular) // 2]) / 2
        if median > 0 and amount >= median * multiple:
            return True, (f"{amount:g} per share is {amount / median:.1f}x the {median:g} median of "
                          f"the recent regular dividends")
    return False, None


def dividend_price_allowance(actions, *, latest_date, config=CONFIG, price_basis=PRICE_BASIS):
    """Price to add back before judging a move, given the OHLCV basis (§5).

    With the adjusted basis this project actually uses, Yahoo has already
    removed the ex-dividend drop from the series, so the answer is **0.0** —
    compensating again would be the double adjustment. The raw branch exists so
    that a future raw-price source cannot silently produce fake "gap down /
    failed breakout / structure break" conclusions.
    """
    if price_basis != RAW_PRICE_BASIS:
        return 0.0, None
    window = int(config.get("trade_dividend_signal_window_sessions", 3))
    latest_date = to_date_str(latest_date)
    total, parts = 0.0, []
    for action in actions or []:
        if action.get("action_type") not in DIVIDEND_TYPES or action.get("status") != APPLIED:
            continue
        effective = to_date_str(action.get("effective_date"))
        amount = _num(action.get("cash_amount"))
        if not effective or not amount or not latest_date or effective > latest_date:
            continue
        if _sessions_between(effective, latest_date) > window:
            continue
        total += amount
        parts.append(f"{amount:g} ex-{effective}")
    if not total:
        return 0.0, None
    return total, "raw prices: added back " + ", ".join(parts)


def _sessions_between(start, end):
    """Rough calendar-day distance; only used to bound a short alert window."""
    try:
        return (date.fromisoformat(end) - date.fromisoformat(start)).days
    except (TypeError, ValueError):
        return 10 ** 6


def recent_special_dividend(actions, *, latest_date, config=CONFIG):
    """The most recent applied SPECIAL_DIVIDEND inside the alert window, if any."""
    window = int(config.get("trade_dividend_signal_window_sessions", 3))
    latest_date = to_date_str(latest_date)
    best = None
    for action in actions or []:
        if action.get("action_type") != SPECIAL_DIVIDEND:
            continue
        effective = to_date_str(action.get("effective_date"))
        if not effective or not latest_date or effective > latest_date:
            continue
        if _sessions_between(effective, latest_date) > window:
            continue
        if best is None or effective > to_date_str(best.get("effective_date")):
            best = action
    return best


# ── detection (§10) ───────────────────────────────────────────────────────

def yfinance_actions(symbol, *, start=None, config=CONFIG):
    """Splits and dividends for one symbol, straight from yfinance.

    The default detection source. It is injected everywhere it is used
    (``Deps.fetch_corporate_actions``), so tests never reach the network.
    Returns ``{"splits": {date: ratio}, "dividends": {date: amount}}``.
    """
    import yfinance as yf

    ticker = yf.Ticker(str(symbol).upper())
    out = {"splits": {}, "dividends": {}}
    start = to_date_str(start)
    for name, attr in (("splits", "splits"), ("dividends", "dividends")):
        try:
            series = getattr(ticker, attr)
        except Exception:
            logger.debug("%s: %s lookup failed", symbol, attr, exc_info=True)
            continue
        if series is None or len(series) == 0:
            continue
        for stamp, value in series.items():
            day = to_date_str(stamp)
            if not day or (start and day < start):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                out[name][day] = number
    return out


def detect_for_trade(trade, raw, *, as_of, config=CONFIG, known_fingerprints=()):
    """Corporate actions for one open trade that are new since it was opened.

    ``raw`` is whatever ``fetch_corporate_actions`` returned. Nothing here
    writes: it returns candidate action dicts, already de-duplicated against
    ``known_fingerprints`` (the idempotency guard, §2).
    """
    symbol = str(trade.get("symbol") or "").upper()
    opened = to_date_str(trade.get("opened_at"))
    as_of = to_date_str(as_of)
    known = set(known_fingerprints or ())
    out = []

    splits = dict((raw or {}).get("splits") or {})
    dividends = dict((raw or {}).get("dividends") or {})
    reference_price = _num(trade.get("average_cost"))
    regular = [amount for day, amount in sorted(dividends.items()) if not opened or day < opened]

    for day, ratio in sorted(splits.items()):
        if opened and day <= opened:
            continue                                    # before we held it — already in the fills
        if as_of and day > as_of:
            continue
        try:
            action_type = classify_split(ratio)
        except CorporateActionError:
            continue
        action = make_action(symbol, day, action_type, split_ratio=float(ratio),
                             source=SOURCE_YFINANCE, trade_id=trade.get("trade_id"),
                             details={"ratio_text": describe_ratio(ratio)})
        if action["fingerprint"] not in known:
            out.append(action)

    for day, amount in sorted(dividends.items()):
        if opened and day <= opened:
            continue
        if as_of and day > as_of:
            continue
        special, why = is_special_dividend(amount, reference_price=reference_price,
                                           regular_amounts=regular, config=config)
        action = make_action(symbol, day, SPECIAL_DIVIDEND if special else CASH_DIVIDEND,
                             cash_amount=float(amount), source=SOURCE_YFINANCE,
                             trade_id=trade.get("trade_id"),
                             details={"per_share": float(amount), "special_reason": why} if special
                             else {"per_share": float(amount)})
        if action["fingerprint"] not in known:
            out.append(action)

    for event in (raw or {}).get("events") or []:
        action = _event_to_action(symbol, event, trade, as_of)
        if action and action["fingerprint"] not in known:
            out.append(action)

    out.sort(key=lambda a: (a["effective_date"], a["action_type"]))
    return out


def _event_to_action(symbol, event, trade, as_of):
    """A pre-shaped event dict (symbol change, merger, …) from any source."""
    action_type = str(event.get("action_type") or event.get("type") or "").upper()
    if action_type not in ACTION_TYPES:
        logger.debug("%s: ignoring corporate-action event of unknown type %r", symbol, action_type)
        return None
    effective = to_date_str(event.get("effective_date") or event.get("date"))
    if not effective or (as_of and effective > to_date_str(as_of)):
        return None
    return make_action(
        symbol, effective, action_type,
        split_ratio=event.get("split_ratio"), cash_amount=event.get("cash_amount"),
        currency=event.get("currency") or "USD",
        old_symbol=event.get("old_symbol") or symbol, new_symbol=event.get("new_symbol"),
        source=event.get("source") or SOURCE_MANUAL, source_event_id=event.get("source_event_id"),
        details=event.get("details"), notes=event.get("notes"), trade_id=trade.get("trade_id"),
    )


# ── review reasons (§8, §9) ───────────────────────────────────────────────

REVIEW_MESSAGES = {
    MERGER: "merger/acquisition detected — exchange ratios and cost-basis allocation were not guessed",
    SPINOFF: "spin-off detected — automatic cost-basis allocation was not attempted",
    RIGHTS: "rights/warrant issue detected — no share or basis allocation was attempted",
    DELISTING: "market data for this position stopped — this is not treated as a sale",
    OTHER: "an unclassified corporate action was recorded",
    SPECIAL_DIVIDEND: ("special dividend — the recorded stop, pivot and protective level were NOT "
                       "changed automatically; review them against the post-distribution price"),
}


def review_reason(action):
    """Why a position is held in CORPORATE_ACTION_REVIEW."""
    action_type = action.get("action_type")
    base = REVIEW_MESSAGES.get(action_type, REVIEW_MESSAGES[OTHER])
    return f"{action_type} {to_date_str(action.get('effective_date'))}: {base}"


def is_unambiguous_cash_merger(action):
    """A cash acquisition we can *propose* a close for (§8) — never apply on our own."""
    return (action.get("action_type") == MERGER
            and (_num(action.get("cash_amount")) or 0) > 0
            and not action.get("details", {}).get("stock_component"))


def summarize(action):
    """One-line description used in CLI output, events and Telegram."""
    action_type = action.get("action_type")
    day = to_date_str(action.get("effective_date"))
    if action_type in SHARE_BASIS_TYPES:
        ratio = _num(action.get("split_ratio"))
        label = "stock dividend" if action_type == STOCK_DIVIDEND else describe_ratio(ratio or 1)
        if action_type == STOCK_DIVIDEND and ratio:
            label = f"stock dividend {(ratio - 1) * 100:g}%"
        return f"{day} {action_type} ({label})"
    if action_type in DIVIDEND_TYPES:
        return f"{day} {action_type} ({_num(action.get('cash_amount')):g}/share)"
    if action_type == SYMBOL_CHANGE:
        return f"{day} {action_type} ({action.get('old_symbol')} -> {action.get('new_symbol')})"
    if action_type == MERGER and _num(action.get("cash_amount")):
        return f"{day} {action_type} ({_num(action.get('cash_amount')):g} cash/share)"
    return f"{day} {action_type}"


def details_json(action):
    return json.dumps(action.get("details") or {}, sort_keys=True, default=str)


# ── orchestration: detect, then safely apply (§10) ────────────────────────
# Order inside the daily Stage-H/I cycle:
#     screening -> setup plans -> DETECT -> APPLY -> position monitor -> report
# Every open position is checked, whether or not the screener saw it today.

from . import trade_store as tstore                                # noqa: E402  (no import cycle)


def _trade_summary(trade):
    return {"trade_id": trade.get("trade_id"), "symbol": trade.get("symbol"),
            "shares": _num(trade.get("shares")), "average_cost": _num(trade.get("average_cost")),
            "initial_stop": _num(trade.get("initial_stop")),
            "pivot_price": _num(trade.get("pivot_price"))}


def apply_share_basis(store, action, trade, *, timestamp=None):
    """Split / reverse split / stock dividend: restate the share basis (§3, §6).

    Economic value is preserved (shares x average cost is invariant), realised
    P&L and dividend income already earned are untouched, and the original fills
    stay exactly as the user reported them.
    """
    ratio = _num(action.get("split_ratio"))
    before = _trade_summary(trade)
    fields = adjusted_trade_fields(trade, ratio)
    detail = json.dumps({"ratio": ratio, "before": before, "after": fields},
                        sort_keys=True, default=str)
    applied = store.apply_action_to_trade(
        action["action_id"], trade["trade_id"], expected_status=action["status"],
        trade_fields=fields, timestamp=timestamp,
        event_reason="%s %s effective %s" % (action["action_type"], describe_ratio(ratio),
                                             to_date_str(action["effective_date"])),
        event_detail=detail)
    return {**action, "status": APPLIED if applied else action["status"],
            "applied": applied, "before": before, "after": fields, "ratio": ratio}


def apply_cash_dividend(store, action, trade, *, prior_actions=(), timestamp=None):
    """Cash / special dividend (§4): book income, change nothing else.

    Shares, the original fill prices and the average cost are deliberately
    untouched — a dividend is cash received, not a change to what the shares
    cost. When entitlement cannot be determined from the recorded fills the
    action is stored for review instead of guessed at.
    """
    per_share = _num(action.get("cash_amount")) or 0.0
    fills = store.fills(trade["trade_id"])
    shares, certain, why = eligible_shares_at(fills, action["effective_date"], prior_actions)

    if not certain:
        store.set_action_status(action["action_id"], REVIEW_REQUIRED,
                                expected_status=action["status"], notes=why,
                                trade_id=trade["trade_id"])
        store.set_trade_review(trade["trade_id"], "%s %s: %s" % (
            action["action_type"], to_date_str(action["effective_date"]), why))
        return {**action, "status": REVIEW_REQUIRED, "applied": False, "needs_review": True,
                "eligible_shares": shares, "uncertain_reason": why, "per_share": per_share}

    if shares <= 0:
        store.set_action_status(action["action_id"], NOT_APPLICABLE,
                                expected_status=action["status"], notes=why,
                                trade_id=trade["trade_id"])
        return {**action, "status": NOT_APPLICABLE, "applied": False, "eligible_shares": 0.0,
                "income": 0.0, "per_share": per_share}

    income = _r(per_share * shares, 4)
    detail = json.dumps({"per_share": per_share, "eligible_shares": shares, "income": income},
                        sort_keys=True, default=str)
    applied = store.apply_action_to_trade(
        action["action_id"], trade["trade_id"], expected_status=action["status"],
        dividend_income_delta=income, timestamp=timestamp,
        event_reason="%s %g/share on %g shares (ex-%s)" % (
            action["action_type"], per_share, shares, to_date_str(action["effective_date"])),
        event_detail=detail)
    result = {**action, "status": APPLIED if applied else action["status"], "applied": applied,
              "eligible_shares": shares, "income": income, "per_share": per_share}

    # §5: an ordinary dividend is small, and the bars are already adjusted for
    # it, so nothing else is affected. A SPECIAL dividend materially rebases the
    # price series against a stop/pivot that was recorded on the
    # pre-distribution price — flag that for the user rather than guess a new
    # stop (and never let it fire a mechanical exit).
    if applied and action["action_type"] == SPECIAL_DIVIDEND:
        store.set_trade_review(trade["trade_id"], review_reason(action))
        result["needs_review"] = True
    return result


def apply_symbol_change(store, action, trade, *, timestamp=None):
    """Ticker change (§7): same trade, same fills, same P&L — new symbol.

    The trade is never closed and reopened. ``symbol_history_json`` keeps the
    whole chain, so the position's past stays readable.
    """
    new_symbol = (action.get("new_symbol") or "").upper()
    old_symbol = (action.get("old_symbol") or trade["symbol"]).upper()
    if not new_symbol or new_symbol == old_symbol:
        store.set_action_status(action["action_id"], REVIEW_REQUIRED,
                                expected_status=action["status"],
                                notes="no unambiguous new symbol recorded",
                                trade_id=trade["trade_id"])
        return {**action, "status": REVIEW_REQUIRED, "applied": False, "needs_review": True}

    clash = store.open_trade(new_symbol)
    if clash and clash["trade_id"] != trade["trade_id"]:
        note = "an open position in %s already exists — merge them by hand" % new_symbol
        store.set_action_status(action["action_id"], REVIEW_REQUIRED,
                                expected_status=action["status"], notes=note,
                                trade_id=trade["trade_id"])
        store.set_trade_review(trade["trade_id"],
                               "SYMBOL_CHANGE %s -> %s: %s" % (old_symbol, new_symbol, note))
        return {**action, "status": REVIEW_REQUIRED, "applied": False, "needs_review": True,
                "notes": note}

    history = json.loads(trade["symbol_history_json"]) if trade.get("symbol_history_json") else []
    if not history:
        history = [{"symbol": old_symbol, "from": to_date_str(trade.get("opened_at"))}]
    history.append({"symbol": new_symbol, "from": to_date_str(action["effective_date"]),
                    "previous": old_symbol})
    applied = store.apply_action_to_trade(
        action["action_id"], trade["trade_id"], expected_status=action["status"],
        trade_fields={"symbol": new_symbol}, symbol_history=history, timestamp=timestamp,
        event_reason="SYMBOL_CHANGE %s -> %s effective %s" % (
            old_symbol, new_symbol, to_date_str(action["effective_date"])),
        event_detail=json.dumps({"symbol_history": history}, sort_keys=True, default=str))
    return {**action, "status": APPLIED if applied else action["status"], "applied": applied,
            "old_symbol": old_symbol, "new_symbol": new_symbol, "symbol_history": history}


def flag_for_review(store, action, trade, *, status=None, timestamp=None):
    """§8/§9: record the event, hold the position, invent nothing.

    No cost-basis allocation, no exchange ratio and no new share quantity is
    guessed. A cash acquisition with unambiguous terms becomes
    AWAITING_CONFIRMATION so the user can confirm the close explicitly;
    everything else is REVIEW_REQUIRED.
    """
    status = status or (AWAITING_CONFIRMATION if is_unambiguous_cash_merger(action)
                        else REVIEW_REQUIRED)
    reason = review_reason(action)
    store.set_action_status(action["action_id"], status, expected_status=action["status"],
                            notes=reason, trade_id=trade["trade_id"])
    store.set_trade_review(trade["trade_id"], reason)
    store.record_event(trade["trade_id"], tstore.EVENT_CORPORATE_ACTION, timestamp=timestamp,
                       reason="%s %s - review required" % (action["action_type"],
                                                           to_date_str(action["effective_date"])),
                       detail=reason)
    return {**action, "status": status, "applied": False, "needs_review": True,
            "review_reason": reason}


def apply_action(store, action, trade, *, prior_actions=(), timestamp=None):
    """Route one recorded action to its handler. Returns the outcome dict."""
    action_type = action["action_type"]
    if action_type in SHARE_BASIS_TYPES:
        return apply_share_basis(store, action, trade, timestamp=timestamp)
    if action_type in DIVIDEND_TYPES:
        return apply_cash_dividend(store, action, trade, prior_actions=prior_actions,
                                   timestamp=timestamp)
    if action_type == SYMBOL_CHANGE:
        return apply_symbol_change(store, action, trade, timestamp=timestamp)
    return flag_for_review(store, action, trade, timestamp=timestamp)


def record_delisting(store, trade, trading_date, *, missing_sessions=None, timestamp=None):
    """Market data for an open position stopped (§9).

    Disappearing data is never read as a sale. The event is recorded once and
    the position goes into review, which suppresses every technical exit
    conclusion until the user resolves it. Returns ``(outcome, created)``.
    """
    action = make_action(trade["symbol"], trading_date, DELISTING, source=SOURCE_MONITOR,
                         details={"missing_sessions": missing_sessions},
                         trade_id=trade.get("trade_id"),
                         notes="no market data for this symbol")
    row, created = store.record_corporate_action(action)
    if not created:
        return row, False
    return flag_for_review(store, row, trade, timestamp=timestamp), True


def process_open_positions(trading_date, config=CONFIG, *, store=None, fetch_actions=None,
                           now=None, force=False):
    """Detect and safely apply corporate actions for every open position.

    Runs **before** the position monitor, so the monitor always sees an
    up-to-date share basis. One probe per symbol per session
    (``corporate_action_checks``), so repeating the daily cycle costs no extra
    Yahoo calls. Never raises for one bad symbol, and never touches Stage A-G.
    """
    if not config.get("trade_corporate_actions_enabled", True):
        return []
    store = store or tstore.open_store(config, create=False)
    if store is None:
        return []
    positions = store.open_positions()
    if not positions:
        return []
    fetch_actions = fetch_actions or yfinance_actions
    now = now or tstore.utc_now()
    trading_date = to_date_str(trading_date)
    lookback = int(config.get("trade_corporate_action_lookback_days", 400))

    results = []
    for trade in positions:
        symbol = trade["symbol"]
        if not force and store.symbol_checked(symbol, trading_date):
            logger.debug("%s: corporate actions already checked for %s", symbol, trading_date)
            continue
        try:
            start = lookback_start(trade.get("opened_at"), trading_date, lookback)
            raw = fetch_actions(symbol, start=start, config=config) or {}
            store.mark_symbol_checked(symbol, trading_date, timestamp=now)
        except Exception:
            logger.warning("%s: corporate-action lookup failed — other positions continue",
                           symbol, exc_info=True)
            continue

        try:
            candidates = detect_for_trade(trade, raw, as_of=trading_date, config=config,
                                          known_fingerprints=store.known_action_fingerprints(symbol))
        except Exception:
            logger.warning("%s: corporate-action detection failed", symbol, exc_info=True)
            continue

        for candidate in candidates:
            try:
                row, created = store.record_corporate_action(candidate)
                if not created:
                    continue                                  # idempotency: already seen
                current = store.get_trade(trade["trade_id"])
                prior = store.corporate_actions(trade_id=trade["trade_id"], statuses=(APPLIED,))
                outcome = apply_action(store, row, current, prior_actions=prior, timestamp=now)
                results.append(outcome)
                logger.info("  %s: %s -> %s", symbol, summarize(row), outcome.get("status"))
            except Exception:
                logger.warning("%s: could not process %s — other actions continue", symbol,
                               summarize(candidate), exc_info=True)
    if results:
        logger.info("Corporate actions: %d event(s) processed for session %s.",
                    len(results), trading_date)
    return results


def lookback_start(opened_at, trading_date, lookback_days):
    """Earliest date worth asking about: the entry date, bounded by the lookback."""
    opened = to_date_str(opened_at)
    try:
        floor = date.fromisoformat(to_date_str(trading_date)).toordinal() - int(lookback_days)
        floor = date.fromordinal(max(floor, 1)).isoformat()
    except (TypeError, ValueError):
        return opened
    return max(opened, floor) if opened else floor


def actions_for_trade(store, trade):
    """Every recorded action for one trade, oldest first (the §12/§13 audit view)."""
    if store is None or not trade:
        return []
    return store.corporate_actions(trade_id=trade.get("trade_id"))


def open_reviews(store, trade):
    """The actions still holding this position in review."""
    return [a for a in actions_for_trade(store, trade) if a.get("status") in OPEN_REVIEW_STATUSES]


def resolve_reviews(store, trade, *, timestamp=None, notes=None):
    """Mark this position's outstanding review actions resolved and release it.

    The user, not the app, decides that a merger/spin-off/special dividend has
    been dealt with. Nothing about the recorded history changes: the actions
    keep their type, date and terms and simply move to RESOLVED.
    """
    resolved = []
    for action in open_reviews(store, trade):
        if store.set_action_status(action["action_id"], RESOLVED,
                                   expected_status=action["status"], notes=notes):
            resolved.append(action)
    store.clear_trade_review(trade["trade_id"], timestamp=timestamp)
    if resolved:
        store.record_event(trade["trade_id"], tstore.EVENT_CORPORATE_ACTION, timestamp=timestamp,
                           reason="corporate-action review resolved by the user",
                           detail="; ".join(summarize(a) for a in resolved))
    return resolved
