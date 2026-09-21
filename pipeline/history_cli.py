"""Inspect the Stage H SQLite history without opening SQLite by hand.

  python -m pipeline.history_cli runs [--limit 15]
  python -m pipeline.history_cli top [--sessions 10] [--top-n 10]
  python -m pipeline.history_cli symbol NVDA [--sessions 10]
  python -m pipeline.history_cli status
"""
import argparse
import os
import sys
from datetime import datetime, timezone

from . import trading_calendar as tc
from .config import CONFIG
from .history_store import HistoryStore


def _fmt(v, width=0, digits=None):
    if v is None:
        s = "–"
    elif digits is not None:
        s = f"{float(v):.{digits}f}"
    else:
        s = str(v)
    return s.ljust(width) if width else s


def cmd_runs(store, args, out):
    rows = store.recent_runs(args.limit)
    if not rows:
        print("No runs recorded yet.", file=out)
        return
    print(f"{'trading_date':<13}{'status':<21}{'canon':<6}{'trigger':<10}{'ranked':>7}{'8/8':>5}"
          f"{'vcp':>5}  {'started (UTC)':<20}{'config':<17}commit", file=out)
    for r in rows:
        print(f"{r['trading_date']:<13}{r['status']:<21}{'yes' if r['is_canonical'] else '':<6}"
              f"{r['trigger']:<10}{_fmt(r.get('count_ranked')):>7}{_fmt(r.get('count_trend_template_pass')):>5}"
              f"{_fmt(r.get('count_vcp_analyzed')):>5}  {(r['started_at_utc'] or '')[:19]:<20}"
              f"{_fmt(r.get('config_hash'), 17)}{(r.get('git_commit') or '–')[:10]}", file=out)
        if r.get("error_summary") and not r["is_canonical"]:
            print(f"{'':<13}-> {r.get('failed_stage') or ''} {r['error_summary'][:110]}", file=out)


def cmd_top(store, args, out):
    sessions = store.canonical_sessions()[-args.sessions:]
    if not sessions:
        print("No successful runs recorded yet.", file=out)
        return
    rows = store.results_for_sessions(sessions)
    by_session = {}
    for r in rows:
        by_session.setdefault(r["trading_date"], {})[r["symbol"]] = r["final_rank"]
    latest = by_session.get(str(sessions[-1]), {})
    symbols = {s for sess in by_session.values() for s, rank in sess.items() if rank <= args.top_n}
    order = sorted(symbols, key=lambda s: (latest.get(s, 10**9), s))
    labels = [str(s)[5:] for s in sessions]   # MM-DD
    print(f"Top {args.top_n} rank history over {len(sessions)} recorded session(s) "
          f"({sessions[0]} -> {sessions[-1]}); '–' = not ranked that session", file=out)
    print(f"{'symbol':<8}" + "".join(f"{l:>7}" for l in labels), file=out)
    for sym in order:
        cells = []
        for s in sessions:
            rank = by_session.get(str(s), {}).get(sym)
            cells.append(f"{('#' + str(rank)) if rank else '–':>7}")
        print(f"{sym:<8}" + "".join(cells), file=out)


def cmd_symbol(store, args, out):
    rows = store.symbol_history(args.symbol, limit=args.sessions)
    if not rows:
        print(f"No recorded results for {args.symbol.upper()}.", file=out)
        return
    print(f"{args.symbol.upper()} — last {len(rows)} recorded session(s)", file=out)
    print(f"{'trading_date':<13}{'rank':>5}{'composite':>10}{'tech':>6}{'fund':>6}{'close':>10}  vcp", file=out)
    for r in reversed(rows):
        vcp = r.get("vcp_entry_recommendation") if r.get("vcp_status") == "analyzed" else r.get("vcp_status")
        if vcp and r.get("vcp_confidence") and r.get("vcp_status") == "analyzed":
            vcp = f"{vcp} ({r['vcp_confidence']})"
        print(f"{r['trading_date']:<13}{r['final_rank']:>5}{_fmt(r['composite_score'], digits=1):>10}"
              f"{_fmt(r['technical_score'], digits=0):>6}{_fmt(r['fundamentals_score'], digits=0):>6}"
              f"{_fmt(r['last_close'], digits=2):>10}  {vcp or '–'}", file=out)
    span = [tc.to_date(r["trading_date"]) for r in rows]
    canonical = set(store.canonical_sessions())
    gaps = [s for s in tc.sessions_between(min(span), max(span)) if s not in canonical]
    if gaps:
        print(f"(sessions with no successful run in this span: {', '.join(map(str, gaps))})", file=out)


def cmd_status(db_path, config, out, now=None):
    """Read-only; never creates the database if it does not exist yet."""
    from .scheduler import build_status, format_status
    now = now or datetime.now(timezone.utc)
    store, db_line = None, "not created yet (no runs recorded)"
    if os.path.exists(db_path):
        try:
            store = HistoryStore(db_path)
            db_line = f"OK ({db_path}, {len(store.canonical_sessions())} successful session(s))"
        except Exception as e:
            db_line = f"ERROR: {type(e).__name__}: {e}"
            store = None
    print("Scheduler / Stage-H Status", file=out)
    print("", file=out)
    st = build_status(config, now, store)
    lines = format_status(st, config)
    width = lines[0].index(" : ")
    for line in lines:
        print(line, file=out)
    print(f"{'History database':<{width}} : {db_line}", file=out)
    print(f"{'Deferred retries':<{width}} : held in the running scheduler's memory only; "
          f"see data/logs/scheduler_*.log", file=out)
    return 2 if st.get("schedule_error") else 0


def main(argv=None, config=CONFIG, out=sys.stdout):
    parser = argparse.ArgumentParser(prog="python -m pipeline.history_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=config["history_db_path"], help="SQLite path (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("runs", help="recent pipeline runs and their status")
    p.add_argument("--limit", type=int, default=15)
    p = sub.add_parser("top", help="Top-N rank history across recent sessions")
    p.add_argument("--sessions", type=int, default=config["trend_lookback_sessions"])
    p.add_argument("--top-n", type=int, default=config["trend_top_n"])
    p = sub.add_parser("symbol", help="one symbol's rank/score/VCP history")
    p.add_argument("symbol")
    p.add_argument("--sessions", type=int, default=config["trend_lookback_sessions"])
    sub.add_parser("status", help="latest session vs latest successful run; would a check run now")
    args = parser.parse_args(argv)

    if args.command == "status":
        return cmd_status(args.db, config, out)
    store = HistoryStore(args.db)
    {"runs": cmd_runs, "top": cmd_top, "symbol": cmd_symbol}[args.command](store, args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
