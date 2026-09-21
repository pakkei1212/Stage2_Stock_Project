"""Stage I: the Telegram trade-journal interface (pure logic, no network).

This module turns a raw Telegram update into replies and, when the user
confirms, into journal writes. It performs no I/O of its own:
:mod:`pipeline.trade_bot` owns the polling loop and the sending.

Two safety rules are enforced here, not in the transport:

1. **Authorization.** Mutating commands (and every inline-button callback) are
   accepted only from an explicit allowlist — ``TELEGRAM_ALLOWED_CHAT_ID``
   (comma-separated; falls back to ``TELEGRAM_CHAT_ID``) and, optionally,
   ``TELEGRAM_ALLOWED_USER_ID``. Updates from anywhere else are ignored
   entirely: nothing is written and no reply is sent. Credentials are never
   logged — only the offending chat id is.
2. **Confirmation.** ``/buy``, ``/sell`` and ``/close`` never write to SQLite.
   They create a ``pending_actions`` row and an inline Confirm/Cancel keyboard;
   only a Confirm callback applies the change, and only once — the claim is a
   conditional UPDATE, so a second tap on the same button (a duplicate callback,
   a retried delivery) writes nothing.

The daily dashboard's buttons land here too. ``🟢 Open Position`` (``op:SYM``)
asks for the actual fill price, then the share count, then shows the same
Confirm/Cancel preview as ``/buy`` with the persisted Stage-I plan beside the
actual fill (planned-vs-actual warnings never block a record). ``💼 View
Position`` (``vp:SYM``) is ``/position SYM``. Neither creates a second OPEN
trade for a symbol.

No command here can reach a broker: there is no brokerage API, no account
access and no order placement anywhere in this project. ``/buy`` and Open
Position record a fill the user already made somewhere else.
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from . import trade_journal as tj
from . import trade_store as tstore
from .config import CONFIG
from .telegram_notifier import OPEN_POSITION_PREFIX, VIEW_POSITION_PREFIX, esc, position_keyboard
from .trade_store import TradeError

logger = logging.getLogger(__name__)

CONFIRM = "confirm"
CANCEL = "cancel"
CALLBACK_PREFIX = "ta"          # ta:<token>:confirm | ta:<token>:cancel
# op:<SYMBOL> "Open Position" / vp:<SYMBOL> "View Position" — the dashboard buttons.
# rb:<SYMBOL> is the older "Record Buy" button, still answered for messages sent
# before the dashboard existed; it now starts the same Open Position flow.
RECORD_BUY_PREFIX = "rb"
POSITION_INPUT = "position_input"   # pending kind: collecting fill price, then shares

JOURNAL_ONLY = "Journal entry only — no brokerage order is placed."

MUTATION_COMMANDS = ("buy", "sell", "close")
READ_COMMANDS = ("positions", "position", "trades", "help", "start")

_NUMBER = r"[-+]?\d+(?:\.\d+)?"


@dataclass
class Reply:
    text: str
    reply_markup: Optional[dict] = None
    pending_token: Optional[str] = None      # attach the sent message id to this action


@dataclass
class Response:
    replies: list = field(default_factory=list)
    callback_answer: Optional[str] = None
    retire_markup: Optional[tuple] = None    # (chat_id, message_id) whose keyboard to drop
    authorized: bool = True
    denial_reason: Optional[str] = None
    applied: Optional[dict] = None           # what was written, for logging/tests


# ── authorization ─────────────────────────────────────────────────────────

def _id_set(*env_vars):
    out = set()
    for var in env_vars:
        for part in (os.environ.get(var) or "").split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out


def allowed_chat_ids():
    """Allowlisted chats. ``TELEGRAM_ALLOWED_CHAT_ID`` wins; otherwise the chat
    the reports go to. Empty means "no chat may mutate anything"."""
    explicit = _id_set("TELEGRAM_ALLOWED_CHAT_ID")
    return explicit or _id_set("TELEGRAM_CHAT_ID")


def allowed_user_ids():
    return _id_set("TELEGRAM_ALLOWED_USER_ID")


def authorize(chat_id, user_id=None):
    """(allowed, reason). Applies to every command and callback, not just mutations."""
    chats = allowed_chat_ids()
    if not chats:
        return False, "no allowlist configured (set TELEGRAM_ALLOWED_CHAT_ID)"
    if str(chat_id) not in chats:
        return False, f"chat {chat_id} is not allowlisted"
    users = allowed_user_ids()
    if users and str(user_id) not in users:
        return False, f"user {user_id} is not allowlisted"
    return True, None


# ── parsing ───────────────────────────────────────────────────────────────

def parse_command(text):
    """(command, args) for a slash command, or (None, []) — bot suffixes stripped."""
    if not text:
        return None, []
    parts = text.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None, []
    command = parts[0][1:].split("@", 1)[0].lower()
    return command, parts[1:]


def _price_shares(args, *, need_shares=True):
    if len(args) < (2 if need_shares else 1):
        raise TradeError("not enough arguments")
    symbol = args[0].upper().lstrip("$")
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol):
        raise TradeError(f"{args[0]!r} is not a symbol")
    if not re.fullmatch(_NUMBER, args[1]):
        raise TradeError(f"{args[1]!r} is not a price")
    price = float(args[1])
    shares = None
    rest = args[2:]
    if need_shares:
        if len(args) < 3 or not re.fullmatch(_NUMBER, args[2]):
            raise TradeError("shares missing")
        shares = float(args[2])
        rest = args[3:]
    notes = " ".join(rest) or None
    return symbol, price, shares, notes


# ── formatting ────────────────────────────────────────────────────────────

def _money(value, digits=2):
    return "–" if value is None else f"${float(value):,.{digits}f}"


def _qty(value):
    return "–" if value is None else f"{float(value):g}"


def _pct(value, digits=1):
    return "–" if value is None else f"{float(value):+.{digits}f}%"


def confirm_keyboard(token):
    return {"inline_keyboard": [[
        {"text": "✅ Confirm", "callback_data": f"{CALLBACK_PREFIX}:{token}:{CONFIRM}"},
        {"text": "✖ Cancel", "callback_data": f"{CALLBACK_PREFIX}:{token}:{CANCEL}"},
    ]]}


def open_position_keyboard(symbol):
    """"Open Position" = record MY actual fill in the local journal. Pressing it
    only starts a conversation — it never assumes the user bought anything."""
    return position_keyboard(symbol, held=False)


def view_position_keyboard(symbol):
    return position_keyboard(symbol, held=True)


def format_buy_confirmation(payload):
    """Confirmation for a buy — the persisted Stage-I plan beside the actual fill."""
    setup = payload.get("setup") or {}
    adds = payload.get("adds_to_existing")
    lines = ["🟢 <b>Add to existing position?</b>" if adds else "🟢 <b>Record position?</b>", "",
             f"<b>{esc(payload['symbol'])}</b>",
             f"Buy: {_qty(payload['shares'])} @ {_money(payload['price'])}",
             f"Position value: {_money(payload['value'])}"]
    if adds:
        lines.append(f"Already held: {_qty(payload['existing_shares'])} @ "
                     f"{_money(payload.get('existing_average_cost'))}")
    lines.append("")
    if setup:
        quality = (setup.get("vcp_numeric_quality") or "").upper()
        lines.append(f"Setup: {esc(quality or 'n/a')} VCP · {esc(setup.get('setup_state') or 'n/a')}"
                     + (f" ({esc(setup.get('trading_date'))})" if setup.get("trading_date") else ""))
        if setup.get("pivot_price") is not None:
            lines.append(f"Pivot: {_money(setup['pivot_price'])}")

    plan_lines = []
    if setup.get("entry_trigger_price") is not None:
        plan_lines.append(f"Entry trigger: {_money(setup['entry_trigger_price'])}")
    if setup.get("maximum_chase_price") is not None:
        plan_lines.append(f"Max chase: {_money(setup['maximum_chase_price'])}")
    if payload.get("suggested_initial_stop") is not None:
        plan_lines.append(f"Suggested stop: {_money(payload['suggested_initial_stop'])}")
    if setup.get("suggested_shares") is not None:
        plan_lines.append(f"Suggested size: {_qty(setup['suggested_shares'])} shares")
    if setup.get("position_portion_pct") is not None:
        plan_lines.append(f"Planned portion: {float(setup['position_portion_pct']):.1f}%")
    if plan_lines:
        lines += ["", "<b>Stage-I plan</b>"] + plan_lines
    if payload.get("suggested_initial_stop") is None:
        lines.append("Suggested initial stop: none recorded — set one yourself")

    comparison = payload.get("plan_comparison") or {}
    actual = ["", "<b>Actual vs plan</b>" if comparison else "<b>Actual</b>",
              f"Actual fill: {_money(payload['price'])}"]
    if comparison.get("slippage_from_trigger_pct") is not None:
        actual.append(f"Slippage from trigger: {_pct(comparison['slippage_from_trigger_pct'], 2)}")
    if payload.get("risk_pct_at_stop") is not None:
        actual.append(f"Risk to suggested stop: {float(payload['risk_pct_at_stop']):.1f}%")
    if payload.get("portfolio_portion_pct") is not None:
        actual.append(f"Actual portion: {float(payload['portfolio_portion_pct']):.1f}%")
    lines += actual
    warnings = list(comparison.get("warnings") or []) + list(payload.get("warnings") or [])
    if warnings:
        lines.append("")
        lines += [f"⚠ {esc(w)}" for w in warnings]
    lines += ["", f"<i>{JOURNAL_ONLY}</i>"]
    return "\n".join(lines)


def format_sell_confirmation(payload):
    closing = payload.get("closes_position")
    lines = ["🔴 <b>Close position?</b>" if closing else "🟠 <b>Record partial sell?</b>", "",
             f"<b>{esc(payload['symbol'])}</b>",
             f"Sell: {_qty(payload['shares'])} @ {_money(payload['price'])}",
             f"Value: {_money(payload['value'])}",
             f"Average cost: {_money(payload['average_cost'])}",
             f"Realised P&L: {_money(payload['realized_pnl'])} ({_pct(payload.get('realized_pnl_pct'))})"]
    if not closing:
        lines.append(f"Remaining after sell: {_qty(payload['shares_remaining'])}")
    lines += ["", f"<i>{JOURNAL_ONLY}</i>"]
    return "\n".join(lines)


def format_applied_buy(payload, result):
    trade = result.get("trade") or {}
    verb = "Position opened" if result.get("opened_new") else "Added to position"
    return "\n".join([
        f"✅ <b>{esc(verb)}: {esc(payload['symbol'])}</b>",
        f"Recorded: {_qty(payload['shares'])} @ {_money(payload['price'])}",
        f"Now holding: {_qty(trade.get('shares'))} @ {_money(trade.get('average_cost'))}",
        f"Initial stop: {_money(trade.get('initial_stop'))}",
    ])


def format_applied_sell(payload, result):
    trade = result.get("trade") or {}
    head = "✅ <b>Position closed</b>" if result.get("closed") else "✅ <b>Partial sell recorded</b>"
    lines = [f"{head}: {esc(payload['symbol'])}",
             f"Sold: {_qty(result.get('shares_sold'))} @ {_money(payload['price'])}",
             f"Realised P&L: {_money(result.get('realized_pnl'))}"]
    if not result.get("closed"):
        lines.append(f"Still holding: {_qty(trade.get('shares'))} @ {_money(trade.get('average_cost'))}")
    else:
        lines.append(f"Total realised on this trade: {_money(result.get('total_realized_pnl'))}")
    return "\n".join(lines)


def format_positions(views):
    if not views:
        return "💼 No open positions recorded."
    lines = ["💼 <b>Open positions</b>"]
    for view in views:
        trade, snap = view["trade"], view.get("snapshot") or {}
        line = (f"\n<b>{esc(trade['symbol'])}</b> · {_qty(trade.get('shares'))} @ "
                f"{_money(trade.get('average_cost'))}")
        lines.append(line)
        if snap.get("close") is not None:
            lines.append(f"Close: {_money(snap['close'])} · {_pct(snap.get('pnl_pct'))}"
                         f" · {esc(snap.get('monitor_state') or '')}".rstrip(" ·"))
        stop = snap.get("current_protective_level", trade.get("initial_stop"))
        if stop is not None:
            lines.append(f"Protective level: {_money(stop)} (initial {_money(trade.get('initial_stop'))})")
    lines.append("\n<i>Advisory only — no orders are placed.</i>")
    return "\n".join(lines)


def format_position_detail(detail):
    trade, snap = detail["trade"], detail.get("snapshot") or {}
    lines = [f"💼 <b>{esc(trade['symbol'])}</b> · {esc(trade['status'])}",
             f"Shares: {_qty(trade.get('shares'))} @ {_money(trade.get('average_cost'))}",
             f"Opened: {esc(str(trade.get('opened_at'))[:19])}"]
    if trade.get("setup_date"):
        lines.append(f"Setup: {esc(trade.get('vcp_quality') or 'n/a')} · {esc(trade.get('setup_state') or '')}"
                     f" ({esc(trade['setup_date'])})")
    if snap:
        atr = snap.get("atr_pct")
        lines += ["",
                  f"Close: {_money(snap.get('close'))} · {_pct(snap.get('pnl_pct'))}",
                  f"High since entry: {_money(snap.get('highest_since_entry'))} · "
                  f"MFE {_pct(snap.get('mfe_pct'))} · MAE {_pct(snap.get('mae_pct'))}",
                  "",
                  f"Initial stop: {_money(trade.get('initial_stop'))}",
                  f"Protective level: {_money(snap.get('current_protective_level'))}",
                  "",
                  f"ATR: {'–' if atr is None else f'{float(atr):.1f}%'}",
                  f"EMA10: {_money(snap.get('ema10'))} · EMA21: {_money(snap.get('ema21'))}",
                  f"SMA50: {_money(snap.get('sma50'))}",
                  "",
                  f"Status: <b>{esc(snap.get('monitor_state') or '')}</b> "
                  f"({esc(snap.get('trading_date') or '')})"]
        reasons = tstore.loads(snap.get("reasons_json"), []) if isinstance(snap.get("reasons_json"), str) else []
        lines += [f"• {esc(r)}" for r in reasons[:4]]
    elif trade.get("status") == tstore.OPEN:
        lines += [f"Initial stop: {_money(trade.get('initial_stop'))}",
                  "<i>Not monitored yet — the next daily run adds close, protective level and status.</i>"]
    lines += _corporate_action_detail(trade, snap, detail.get("corporate_actions"))

    fills = detail.get("fills") or []
    if fills:
        lines += ["", "<b>Fills</b>"]
        lines += [f"{esc(f['side'])} {_qty(f['shares'])} @ {_money(f['price'])} "
                  f"({esc(str(f['timestamp'])[:10])})" for f in fills]
        if detail.get("corporate_actions"):
            lines.append("<i>Fills are the original as-traded prices and quantities and are never "
                         "rewritten; corporate actions adjust the derived position values only.</i>")
    lines += ["", "<i>Advisory only — no orders are placed.</i>"]
    return "\n".join(lines)


def _corporate_action_detail(trade, snap, actions):
    """§12: price vs dividend vs total P&L, and the actions since entry.

    Nothing is added when the position has seen no corporate action, so the
    ordinary detail view is unchanged.
    """
    actions = list(actions or [])
    dividends = float(trade.get("dividend_income") or 0)
    if not actions and not dividends:
        return []
    lines = ["", "<b>P&L</b>",
             f"Price P&L: {_money((snap or {}).get('price_pnl'))}",
             f"Dividend income: {_money(dividends)}",
             f"Total P&L: {_money((snap or {}).get('total_pnl'))}"]
    if trade.get("realized_pnl"):
        lines.append(f"(realised {_money(trade.get('realized_pnl'))} of that is already booked)")
    if actions:
        lines += ["", "<b>Corporate actions since entry</b>"]
        for action in actions:
            lines.append(f"{esc(action.get('effective_date'))} · {esc(action.get('action_type'))} · "
                         f"{esc(action.get('status'))}")
        open_reviews = [a for a in actions if a.get("status") in ("REVIEW_REQUIRED",
                                                                  "AWAITING_CONFIRMATION")]
        if open_reviews:
            lines.append("⚠ Unresolved — technical exit conclusions are suppressed for this "
                         "position until you resolve it.")
    history = trade.get("symbol_history_json")
    if history:
        try:
            chain = [str(h.get("symbol")) for h in json.loads(history)]
        except (TypeError, ValueError):
            chain = []
        if len(chain) > 1:
            lines.append(f"Ticker history: {esc(' → '.join(chain))}")
    return lines


def format_trades(trades):
    if not trades:
        return "No trades recorded yet."
    lines = ["📒 <b>Trades</b>"]
    for t in trades:
        status = t["status"]
        line = (f"{esc(t['symbol'])} · {esc(status)} · {_qty(t.get('shares'))} @ "
                f"{_money(t.get('average_cost'))}")
        if t.get("realized_pnl"):
            line += f" · realised {_money(t['realized_pnl'])}"
        line += f" · opened {esc(str(t.get('opened_at'))[:10])}"
        lines.append(line)
    return "\n".join(lines)


HELP_TEXT = "\n".join([
    "📒 <b>Trade journal</b> (advisory only — no broker is connected)", "",
    "<b>Record fills you already made</b>",
    "/buy SYMBOL PRICE SHARES — e.g. <code>/buy CACC 622.50 20</code>",
    "/sell SYMBOL PRICE SHARES — partial sell",
    "/close SYMBOL PRICE — sell everything still held",
    "Each of these asks for Confirm before anything is written.",
    "🟢 <b>Open Position</b> on a READY setup asks for your fill price, then shares, "
    "then shows the plan vs your fill before Confirm.", "",
    "<b>Read-only</b>",
    "/positions — open positions and their monitor state",
    "/position SYMBOL — one position in detail",
    "/trades — recent trades",
    "/help — this message",
])


# ── update handling ───────────────────────────────────────────────────────

def _expiry(config, now):
    minutes = int(config.get("trade_confirm_ttl_minutes", 30))
    return now + timedelta(minutes=minutes) if now is not None else None


def _pending_reply(store, kind, chat_id, user_id, payload, text, config, now):
    token = store.create_pending(kind, chat_id, payload, user_id=user_id, expires_at=_expiry(config, now))
    return Response(replies=[Reply(text, confirm_keyboard(token), pending_token=token)])


def handle_update(update, config=CONFIG, *, store=None, now=None):
    """Turn one Telegram update into a :class:`Response`. Writes only on Confirm."""
    now = now or tstore.utc_now()
    if "callback_query" in update:
        return _handle_callback(update["callback_query"], config, store, now)
    message = update.get("message") or update.get("edited_message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    user_id = (message.get("from") or {}).get("id")
    text = (message.get("text") or "").strip()
    if chat_id is None or not text:
        return Response()

    allowed, reason = authorize(chat_id, user_id)
    if not allowed:
        logger.warning("Ignoring Telegram update: %s", reason)
        return Response(authorized=False, denial_reason=reason)

    command, args = parse_command(text)
    if command is None:
        return _handle_free_text(text, chat_id, user_id, config, store, now)
    return _handle_command(command, args, chat_id, user_id, config, store, now)


def _handle_command(command, args, chat_id, user_id, config, store, now):
    if command in ("help", "start"):
        return Response(replies=[Reply(HELP_TEXT)])
    if command == "positions":
        return Response(replies=[Reply(format_positions(tj.positions_view(config, store=store)))])
    if command == "position":
        if not args:
            return Response(replies=[Reply("Usage: <code>/position SYMBOL</code>")])
        return _view_position(args[0].upper(), config, store)
    if command == "trades":
        limit = int(args[0]) if args and args[0].isdigit() else 20
        return Response(replies=[Reply(format_trades(tj.trades_view(config, store=store, limit=limit)))])
    if command == "cancel":
        pending = store.pending_for_chat(chat_id) if store else []
        for item in pending:
            store.claim_pending(item["token"], tstore.CANCELLED, now=now)
        return Response(replies=[Reply(f"Cancelled {len(pending)} pending action(s).")])

    if command in MUTATION_COMMANDS:
        return _handle_mutation(command, args, chat_id, user_id, config, store, now)
    return Response(replies=[Reply(f"Unknown command <code>/{esc(command)}</code>.\n\n{HELP_TEXT}")])


def _handle_mutation(command, args, chat_id, user_id, config, store, now):
    usage = {"buy": "/buy SYMBOL PRICE SHARES", "sell": "/sell SYMBOL PRICE SHARES",
             "close": "/close SYMBOL PRICE"}[command]
    try:
        symbol, price, shares, notes = _price_shares(args, need_shares=command != "close")
        if command == "buy":
            payload = tj.preview_buy(symbol, price, shares, config, store=store, notes=notes)
            return _pending_reply(store, "buy", chat_id, user_id, payload,
                                  format_buy_confirmation(payload), config, now)
        if command == "sell":
            payload = tj.preview_sell(symbol, price, shares, config, store=store, notes=notes)
        else:
            payload = tj.preview_close(symbol, price, config, store=store, notes=notes)
        return _pending_reply(store, payload["action"], chat_id, user_id, payload,
                              format_sell_confirmation(payload), config, now)
    except TradeError as e:
        return Response(replies=[Reply(f"⚠ {esc(str(e))}\n\nUsage: <code>{esc(usage)}</code>")])


def _expired(item, now):
    expires = item.get("expires_at_utc")
    return bool(expires) and now is not None and now.isoformat() > expires


def _handle_free_text(text, chat_id, user_id, config, store, now):
    """A plain number completes the next step of an Open Position conversation:
    fill price first, then shares. ``PRICE SHARES`` in one reply also works.
    Nothing is written to trades/fills here — the last step only builds the
    usual Confirm/Cancel preview."""
    if store is None:
        return Response()
    waiting = store.pending_for_chat(chat_id, kind=POSITION_INPUT)
    if not waiting:
        return Response()
    item = waiting[0]
    if _expired(item, now):
        store.claim_pending(item["token"], tstore.EXPIRED, now=now)
        return Response(replies=[Reply("⌛ That Open Position request expired — tap the button again.")])
    state = item["payload"] or {}
    symbol = state.get("symbol")
    text = text.strip().lstrip("$")

    if state.get("step") == "shares":
        if not re.fullmatch(_NUMBER, text) or float(text) <= 0:
            return Response(replies=[Reply(f"Number of shares for {esc(symbol)}? Reply with a number, "
                                           "e.g. <code>20</code> — or /cancel.")])
        return _finish_open_position(item, symbol, float(state["price"]), float(text),
                                     chat_id, user_id, config, store, now)

    both = re.fullmatch(rf"({_NUMBER})[\s,]+({_NUMBER})", text)
    if both and float(both.group(1)) > 0 and float(both.group(2)) > 0:
        return _finish_open_position(item, symbol, float(both.group(1)), float(both.group(2)),
                                     chat_id, user_id, config, store, now)
    if not re.fullmatch(_NUMBER, text) or float(text) <= 0:
        return Response(replies=[Reply(f"Actual fill price for {esc(symbol)}? Reply with a number, "
                                       "e.g. <code>631.00</code> — or /cancel.")])
    if store.claim_pending(item["token"], tstore.CONFIRMED, now=now) is None:
        return Response()                        # a duplicate delivery of the same reply
    price = float(text)
    token = store.create_pending(POSITION_INPUT, chat_id, {"symbol": symbol, "step": "shares", "price": price},
                                 user_id=user_id, expires_at=_expiry(config, now))
    plan = store.latest_setup_plan(symbol) or {}
    lines = [f"🟢 <b>{esc(symbol)}</b> · fill {_money(price)}", "", "<b>Number of shares?</b>"]
    if plan.get("suggested_shares") is not None:
        lines.append(f"Plan suggests {_qty(plan['suggested_shares'])} shares.")
    lines += ["<i>Reply with a number, e.g. <code>20</code> — or /cancel.</i>"]
    return Response(replies=[Reply("\n".join(lines), pending_token=token)])


def _finish_open_position(item, symbol, price, shares, chat_id, user_id, config, store, now):
    """Last input step: claim the conversation and show the normal Confirm preview."""
    try:
        payload = tj.preview_buy(symbol, price, shares, config, store=store)
    except TradeError as e:
        return Response(replies=[Reply(f"⚠ {esc(str(e))}")])
    if store.claim_pending(item["token"], tstore.CONFIRMED, now=now) is None:
        return Response()
    return _pending_reply(store, "buy", chat_id, user_id, payload,
                          format_buy_confirmation(payload), config, now)


def _start_open_position(symbol, chat_id, user_id, config, store, now):
    """The 🟢 Open Position button: record MY fill in the local journal.

    If an OPEN trade already exists this shows it instead (one OPEN trade per
    symbol — additional buys go through /buy). Otherwise it asks for the fill
    price; nothing is written to trades/fills until the final Confirm."""
    if store.open_trade(symbol) is not None:
        response = _view_position(symbol, config, store)
        response.replies[0].text += (f"\n\nAlready held — to add shares use "
                                     f"<code>/buy {esc(symbol)} PRICE SHARES</code>.")
        return response
    for stale in store.pending_for_chat(chat_id, kind=POSITION_INPUT):
        store.claim_pending(stale["token"], tstore.CANCELLED, now=now)
    token = store.create_pending(POSITION_INPUT, chat_id, {"symbol": symbol, "step": "price"},
                                 user_id=user_id, expires_at=_expiry(config, now))
    plan = store.latest_setup_plan(symbol) or {}
    lines = [f"🟢 <b>Open Position — {esc(symbol)}</b>", "",
             "Record the position you actually took in the local journal.", "",
             "<b>Actual fill price?</b>"]
    if plan.get("entry_trigger_price") is not None:
        lines.append(f"Plan: trigger {_money(plan['entry_trigger_price'])} · "
                     f"max chase {_money(plan.get('maximum_chase_price'))}")
    lines += ["<i>Reply with a number, e.g. <code>631.00</code> — or /cancel.</i>", "",
              f"<i>{JOURNAL_ONLY}</i>"]
    return Response(replies=[Reply("\n".join(lines), pending_token=token)], callback_answer="Fill price?")


def _view_position(symbol, config, store):
    """The 💼 View Position button — exactly what /position SYMBOL returns."""
    detail = tj.position_detail(symbol, config, store=store)
    return Response(replies=[Reply(format_position_detail(detail) if detail
                                   else f"No recorded trade for {esc(symbol)}.")])


def _handle_callback(callback, config, store, now):
    data = callback.get("data") or ""
    message = callback.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    user_id = (callback.get("from") or {}).get("id")
    allowed, reason = authorize(chat_id, user_id)
    if not allowed:
        logger.warning("Ignoring Telegram callback: %s", reason)
        return Response(authorized=False, denial_reason=reason,
                        callback_answer="Not authorized for this chat.")

    prefix, _, rest = data.partition(":")
    if prefix in (OPEN_POSITION_PREFIX, RECORD_BUY_PREFIX, VIEW_POSITION_PREFIX):
        symbol = rest.upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol):
            return Response(callback_answer="Unknown symbol.")
        if prefix == VIEW_POSITION_PREFIX:
            return _view_position(symbol, config, store)
        return _start_open_position(symbol, chat_id, user_id, config, store, now)

    if not data.startswith(f"{CALLBACK_PREFIX}:"):
        return Response(callback_answer="Unknown action.")
    _, token, decision = (data.split(":") + ["", ""])[:3]

    if decision == CANCEL:
        claimed = store.claim_pending(token, tstore.CANCELLED, now=now)
        if claimed is None:
            return Response(callback_answer="Already handled.",
                            retire_markup=(chat_id, message.get("message_id")))
        return Response(replies=[Reply("✖ Cancelled — nothing was recorded.")],
                        callback_answer="Cancelled.",
                        retire_markup=(chat_id, message.get("message_id")))
    if decision != CONFIRM:
        return Response(callback_answer="Unknown action.")

    # Duplicate-callback protection: only the tap that flips the row out of
    # 'pending' applies anything.
    claimed = store.claim_pending(token, tstore.CONFIRMED, now=now)
    if claimed is None:
        return Response(callback_answer="Already handled.",
                        retire_markup=(chat_id, message.get("message_id")))
    payload = claimed["payload"] or {}
    expires = claimed.get("expires_at_utc")
    if expires and str(now.isoformat()) > expires:
        return Response(replies=[Reply("⌛ That confirmation expired — send the command again.")],
                        callback_answer="Expired.", retire_markup=(chat_id, message.get("message_id")))
    try:
        if payload.get("action") == "buy":
            result = tj.apply_buy(payload, config, store=store, timestamp=now)
            text = format_applied_buy(payload, result)
        else:
            result = tj.apply_sell(payload, config, store=store, timestamp=now)
            text = format_applied_sell(payload, result)
    except TradeError as e:
        return Response(replies=[Reply(f"⚠ Not recorded: {esc(str(e))}")],
                        callback_answer="Rejected.", retire_markup=(chat_id, message.get("message_id")))
    return Response(replies=[Reply(text)], callback_answer="Recorded.",
                    retire_markup=(chat_id, message.get("message_id")), applied=result)
