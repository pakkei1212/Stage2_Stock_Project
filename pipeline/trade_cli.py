"""Stage I: local trade-journal CLI — the fallback when Telegram is unavailable.

Same service layer as the bot (:mod:`pipeline.trade_journal`), so both produce
identical rows::

  python -m pipeline.trade_cli add CACC --price 622.50 --shares 20
  python -m pipeline.trade_cli sell CACC --price 680 --shares 10
  python -m pipeline.trade_cli close CACC --price 705
  python -m pipeline.trade_cli positions
  python -m pipeline.trade_cli show CACC
  python -m pipeline.trade_cli trades [--all]
  python -m pipeline.trade_cli plan [--date YYYY-MM-DD]     # recorded setup gate + plans
  python -m pipeline.trade_cli monitor [--date YYYY-MM-DD]  # re-run the position monitor
  python -m pipeline.trade_cli actions CACC                 # corporate actions for one symbol
  python -m pipeline.trade_cli actions --all                # every recorded corporate action
  python -m pipeline.trade_cli record-action CACC --type SPLIT --date 2026-10-01 --ratio 2
  python -m pipeline.trade_cli resolve CACC [--stop 295]    # clear a corporate-action review

Mutating commands print the same preview the Telegram flow shows and ask for a
confirmation (``--yes`` skips the prompt for scripted use). Nothing is written
before that confirmation.

Advisory only: this records fills the user made elsewhere. No brokerage API, no
account access, no orders.
"""
import argparse
import sys

from . import corporate_actions as ca
from . import position_monitor as pm
from . import trade_journal as tj
from . import trade_store as tstore
from . import trading_calendar as tc
from .config import CONFIG
from .trade_store import TradeError


def _money(value, digits=2):
    return "-" if value is None else f"{float(value):,.{digits}f}"


def _qty(value):
    return "-" if value is None else f"{float(value):g}"


def _confirm(prompt, assume_yes, out, stream=None):
    if assume_yes:
        return True
    print(f"{prompt} [y/N]: ", end="", file=out, flush=True)
    answer = (stream or sys.stdin).readline().strip().lower()
    return answer in ("y", "yes")


def _print_preview(payload, out):
    for line in _preview_lines(payload):
        print(line, file=out)


def _preview_lines(payload):
    if payload["action"] == "buy":
        setup = payload.get("setup") or {}
        lines = [f"Record BUY  {payload['symbol']}  {_qty(payload['shares'])} @ {_money(payload['price'])}"
                 f"   value {_money(payload['value'])}"]
        if setup:
            lines.append(f"  Setup: {(setup.get('vcp_numeric_quality') or 'n/a').upper()} VCP · "
                         f"{setup.get('setup_state')} ({setup.get('trading_date')})")
            lines.append(f"  Pivot: {_money(setup.get('pivot_price'))}")
        lines.append(f"  Suggested initial stop: {_money(payload.get('suggested_initial_stop'))}")
        lines.append(f"  Planned portfolio portion: {_money(payload.get('portfolio_portion_pct'), 1)}%")
    else:
        verb = "CLOSE" if payload["closes_position"] else "SELL"
        lines = [f"Record {verb} {payload['symbol']}  {_qty(payload['shares'])} @ "
                 f"{_money(payload['price'])}   value {_money(payload['value'])}",
                 f"  Average cost: {_money(payload['average_cost'])}   "
                 f"realised P&L: {_money(payload['realized_pnl'])}",
                 f"  Remaining after this fill: {_qty(payload['shares_remaining'])}"]
    for warning in payload.get("warnings") or []:
        lines.append(f"  ! {warning}")
    lines.append("  (journal entry only — no order is placed)")
    return lines


def cmd_add(store, args, config, out):
    payload = tj.preview_buy(args.symbol, args.price, args.shares, config, store=store, notes=args.note)
    _print_preview(payload, out)
    if not _confirm("Record this buy?", args.yes, out):
        print("Cancelled — nothing was recorded.", file=out)
        return 1
    result = tj.apply_buy(payload, config, store=store)
    trade = result["trade"]
    print(f"Recorded. {trade['symbol']}: {_qty(trade['shares'])} @ {_money(trade['average_cost'])} "
          f"(initial stop {_money(trade['initial_stop'])})", file=out)
    return 0


def cmd_sell(store, args, config, out):
    shares = None if getattr(args, "shares", None) is None else args.shares
    payload = tj.preview_sell(args.symbol, args.price, shares, config, store=store, notes=args.note)
    _print_preview(payload, out)
    if not _confirm("Record this sell?", args.yes, out):
        print("Cancelled — nothing was recorded.", file=out)
        return 1
    result = tj.apply_sell(payload, config, store=store)
    print(f"Recorded. Sold {_qty(result['shares_sold'])} {result['symbol']} · realised "
          f"{_money(result['realized_pnl'])} · remaining {_qty(result['shares_remaining'])}", file=out)
    return 0


def cmd_positions(store, args, config, out):
    views = tj.positions_view(config, store=store)
    if not views:
        print("No open positions recorded.", file=out)
        return 0
    print(f"{'symbol':<8}{'shares':>9}{'avg cost':>11}{'close':>10}{'P&L %':>9}{'stop':>10}"
          f"{'protect':>10}  state", file=out)
    for view in views:
        trade, snap = view["trade"], view.get("snapshot") or {}
        print(f"{trade['symbol']:<8}{_qty(trade['shares']):>9}{_money(trade['average_cost']):>11}"
              f"{_money(snap.get('close')):>10}{_money(snap.get('pnl_pct'), 1):>9}"
              f"{_money(trade.get('initial_stop')):>10}"
              f"{_money(snap.get('current_protective_level')):>10}  "
              f"{snap.get('monitor_state') or '-'}", file=out)
    print("\nAdvisory only — no orders are placed.", file=out)
    return 0


def cmd_show(store, args, config, out):
    detail = tj.position_detail(args.symbol, config, store=store)
    if detail is None:
        print(f"No recorded trade for {args.symbol.upper()}.", file=out)
        return 1
    trade, snap = detail["trade"], detail.get("snapshot") or {}
    print(f"{trade['symbol']} · {trade['status']} · {_qty(trade['shares'])} @ "
          f"{_money(trade['average_cost'])}", file=out)
    print(f"  opened {str(trade['opened_at'])[:19]}   setup {trade.get('vcp_quality') or '-'} "
          f"{trade.get('setup_state') or ''} {trade.get('setup_date') or ''}".rstrip(), file=out)
    print(f"  initial stop {_money(trade.get('initial_stop'))}   pivot {_money(trade.get('pivot_price'))}",
          file=out)
    if snap:
        print(f"  close {_money(snap.get('close'))} ({_money(snap.get('pnl_pct'), 1)}%)   "
              f"high since entry {_money(snap.get('highest_since_entry'))}   "
              f"MFE {_money(snap.get('mfe_pct'), 1)}%  MAE {_money(snap.get('mae_pct'), 1)}%", file=out)
        print(f"  ATR {_money(snap.get('atr_pct'), 1)}%   EMA10 {_money(snap.get('ema10'))}   "
              f"EMA21 {_money(snap.get('ema21'))}   SMA50 {_money(snap.get('sma50'))}", file=out)
        print(f"  protective level {_money(snap.get('current_protective_level'))}   "
              f"state {snap.get('monitor_state')} ({snap.get('trading_date')})", file=out)
        for reason in (snap.get("reasons_json") and tstore.loads(snap["reasons_json"], [])) or []:
            print(f"    - {reason}", file=out)
    actions = detail.get("corporate_actions") or []
    if actions or trade.get("dividend_income"):
        print(f"  price P&L {_money(snap.get('price_pnl'))}   "
              f"dividend income {_money(trade.get('dividend_income'))}   "
              f"total P&L {_money(snap.get('total_pnl'))}", file=out)
    if trade.get("needs_review"):
        print(f"  ! corporate-action review: {trade.get('review_reason')}", file=out)
    print("  fills:", file=out)
    for fill in detail["fills"]:
        print(f"    {str(fill['timestamp'])[:10]}  {fill['side']:<4} {_qty(fill['shares']):>8} @ "
              f"{_money(fill['price'])}", file=out)
    if actions:
        print("  corporate actions (fills above are never rewritten):", file=out)
        for action in actions:
            print(f"    {action['effective_date']}  {action['action_type']:<16} "
                  f"{action['status']}", file=out)
    return 0


def cmd_trades(store, args, config, out):
    trades = tj.trades_view(config, store=store, limit=args.limit,
                            status=None if args.all else tstore.OPEN)
    if not trades:
        print("No trades recorded yet.", file=out)
        return 0
    print(f"{'symbol':<8}{'status':<8}{'shares':>9}{'avg cost':>11}{'realised':>11}  opened", file=out)
    for t in trades:
        print(f"{t['symbol']:<8}{t['status']:<8}{_qty(t['shares']):>9}{_money(t['average_cost']):>11}"
              f"{_money(t['realized_pnl']):>11}  {str(t['opened_at'])[:19]}", file=out)
    return 0


def cmd_plan(store, args, config, out):
    trading_date = args.date or store.latest_plan_date()
    plans = store.setup_plans_for_date(trading_date) if trading_date else []
    if not plans:
        print("No setup plans recorded yet (the Stage-H daily run generates them).", file=out)
        return 0
    print(f"Setup gate / advisory plans for {trading_date}", file=out)
    print(f"{'symbol':<8}{'setup state':<20}{'entry status':<28}{'pivot':>10}{'trigger':>10}"
          f"{'stop':>10}{'risk%':>8}{'shares':>8}  plan status", file=out)
    for p in plans:
        print(f"{p['symbol']:<8}{p['setup_state']:<20}{(p.get('entry_status') or '-'):<28}"
              f"{_money(p.get('pivot_price')):>10}{_money(p.get('entry_trigger_price')):>10}"
              f"{_money(p.get('suggested_initial_stop')):>10}"
              f"{_money(p.get('required_risk_pct'), 1):>8}{_qty(p.get('suggested_shares')):>8}  "
              f"{p.get('trade_plan_status') or '-'}", file=out)
    print("\nAdvisory only — no orders are placed.", file=out)
    return 0


def cmd_monitor(store, args, config, out):
    trading_date = tc.to_date(args.date) if args.date else tc.latest_completed_session(
        tstore.utc_now(), config["session_close_buffer_minutes"])
    snapshots = pm.monitor_open_positions(trading_date, config, store=store)
    if not snapshots:
        print("No open positions to monitor.", file=out)
        return 0
    for snap in snapshots:
        print(f"{snap['symbol']:<8}{snap['monitor_state']:<22}close {_money(snap['close'])}  "
              f"P&L {_money(snap['pnl_pct'], 1)}%  protective {_money(snap['current_protective_level'])}",
              file=out)
        for reason in snap.get("reasons") or []:
            print(f"    - {reason}", file=out)
    return 0


def cmd_actions(store, args, config, out):
    """§13: the audit trail — original fills + corporate actions + sales."""
    symbol = None if getattr(args, "all", False) else getattr(args, "symbol", None)
    actions = tj.actions_view(symbol, config, store=store)
    if not actions:
        scope = "any symbol" if symbol is None else symbol.upper()
        print(f"No corporate actions recorded for {scope}.", file=out)
        return 0
    print(f"{'date':<12}{'symbol':<8}{'type':<18}{'status':<22}{'ratio':>8}{'cash':>10}  detail",
          file=out)
    for a in actions:
        detail = a.get("notes") or ""
        if a.get("new_symbol"):
            detail = f"{a.get('old_symbol') or '?'} -> {a['new_symbol']}  {detail}".strip()
        print(f"{str(a['effective_date']):<12}{a['symbol']:<8}{a['action_type']:<18}"
              f"{a['status']:<22}{_qty(a.get('split_ratio')):>8}{_money(a.get('cash_amount')):>10}  "
              f"{detail}", file=out)
    print("\nFills are never rewritten: these actions adjust the derived position values only.",
          file=out)
    return 0


def cmd_record_action(store, args, config, out):
    """Record a corporate action by hand (a ticker change or merger Yahoo does
    not expose). Mutating, so it previews and asks first, like every other
    journal write."""
    action_type = str(args.type).upper()
    try:
        action = ca.make_action(args.symbol, args.date, action_type, split_ratio=args.ratio,
                                cash_amount=args.amount, new_symbol=args.new_symbol,
                                old_symbol=args.symbol, source=ca.SOURCE_MANUAL,
                                notes=args.note)
    except ca.CorporateActionError as e:
        print(f"Error: {e}", file=out)
        return 2

    trade = store.open_trade(args.symbol)
    print(f"Record corporate action: {ca.summarize(action)}", file=out)
    if trade is None:
        print(f"  ! no open position in {args.symbol.upper()} — the event is recorded but nothing "
              f"is adjusted", file=out)
    elif action_type in ca.SHARE_BASIS_TYPES:
        after = ca.adjusted_trade_fields(trade, action["split_ratio"])
        print(f"  shares {_qty(trade['shares'])} -> {_qty(after['shares'])}   "
              f"average cost {_money(trade['average_cost'])} -> {_money(after['average_cost'])}   "
              f"initial stop {_money(trade.get('initial_stop'))} -> "
              f"{_money(after['initial_stop'])}", file=out)
    elif action_type in ca.REVIEW_TYPES:
        print("  this action is recorded for manual review — no cost basis, exchange ratio or "
              "share quantity is guessed", file=out)
    print("  (journal entry only — no order is placed)", file=out)
    if not _confirm("Record this corporate action?", args.yes, out):
        print("Cancelled — nothing was recorded.", file=out)
        return 1

    row, created = store.record_corporate_action(action)
    if not created:
        print(f"Already recorded ({row['status']}) — nothing changed.", file=out)
        return 0
    if trade is None:
        print(f"Recorded. {ca.summarize(row)} (status {row['status']}).", file=out)
        return 0
    outcome = ca.apply_action(store, row, trade,
                              prior_actions=store.corporate_actions(trade_id=trade["trade_id"],
                                                                    statuses=(ca.APPLIED,)))
    updated = store.get_trade(trade["trade_id"])
    print(f"Recorded. {ca.summarize(row)} -> {outcome.get('status')}", file=out)
    print(f"  {updated['symbol']}: {_qty(updated['shares'])} @ {_money(updated['average_cost'])}   "
          f"initial stop {_money(updated.get('initial_stop'))}   "
          f"dividend income {_money(updated.get('dividend_income'))}", file=out)
    if updated.get("needs_review"):
        print(f"  ! held for review: {updated.get('review_reason')}", file=out)
    return 0


def cmd_resolve(store, args, config, out):
    """Clear a corporate-action review once the user has dealt with it."""
    try:
        result = tj.resolve_corporate_review(args.symbol, config, store=store, notes=args.note,
                                             initial_stop=args.stop, pivot_price=args.pivot)
    except TradeError as e:
        print(f"Error: {e}", file=out)
        return 2
    if not result["resolved"] and not result["updated"]:
        print(f"{result['symbol']} had no outstanding corporate-action review.", file=out)
        return 0
    for action in result["resolved"]:
        print(f"Resolved: {ca.summarize(action)}", file=out)
    for field, value in result["updated"].items():
        print(f"Updated {field} to {_money(value)} (your number, not a computed one).", file=out)
    print(f"{result['symbol']} is monitored normally again.", file=out)
    return 0


def main(argv=None, config=CONFIG, out=sys.stdout):
    parser = argparse.ArgumentParser(prog="python -m pipeline.trade_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=config["trade_db_path"], help="trade journal SQLite path")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="record a buy fill")
    p.add_argument("symbol")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--shares", type=float, required=True)
    p.add_argument("--note")

    p = sub.add_parser("sell", help="record a partial sell")
    p.add_argument("symbol")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--shares", type=float, required=True)
    p.add_argument("--note")

    p = sub.add_parser("close", help="sell everything still held")
    p.add_argument("symbol")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--note")

    sub.add_parser("positions", help="open positions with their latest monitor state")

    p = sub.add_parser("show", help="one position in detail")
    p.add_argument("symbol")

    p = sub.add_parser("trades", help="recorded trades")
    p.add_argument("--all", action="store_true", help="include closed trades")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("plan", help="recorded setup gate states and entry plans")
    p.add_argument("--date", help="trading session (default: the most recent recorded)")

    p = sub.add_parser("monitor", help="re-run the position monitor now")
    p.add_argument("--date", help="trading session to label the snapshots with")

    p = sub.add_parser("actions", help="recorded corporate actions")
    p.add_argument("symbol", nargs="?", help="one symbol (omit with --all)")
    p.add_argument("--all", action="store_true", help="every symbol")

    p = sub.add_parser("record-action", help="record a corporate action by hand")
    p.add_argument("symbol")
    p.add_argument("--type", required=True, choices=list(ca.ACTION_TYPES),
                   help="corporate action type")
    p.add_argument("--date", required=True, help="effective / ex-date (YYYY-MM-DD)")
    p.add_argument("--ratio", type=float, help="new shares per old share (2 = 2-for-1, 0.2 = 1-for-5)")
    p.add_argument("--amount", type=float, help="cash per share")
    p.add_argument("--new-symbol", dest="new_symbol", help="for a ticker change")
    p.add_argument("--note")

    p = sub.add_parser("resolve", help="clear a corporate-action review on a position")
    p.add_argument("symbol")
    p.add_argument("--stop", type=float, help="the initial stop YOU decided on")
    p.add_argument("--pivot", type=float, help="the pivot reference YOU decided on")
    p.add_argument("--note")

    args = parser.parse_args(argv)
    store = tstore.TradeStore(args.db)
    handlers = {"add": cmd_add, "sell": cmd_sell, "close": cmd_sell, "positions": cmd_positions,
                "show": cmd_show, "trades": cmd_trades, "plan": cmd_plan, "monitor": cmd_monitor,
                "actions": cmd_actions, "record-action": cmd_record_action, "resolve": cmd_resolve}
    if args.command == "close":
        args.shares = None
    try:
        return handlers[args.command](store, args, config, out)
    except TradeError as e:
        print(f"Error: {e}", file=out)
        return 2


if __name__ == "__main__":
    sys.exit(main())
