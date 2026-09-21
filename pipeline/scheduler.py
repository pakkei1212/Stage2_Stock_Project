"""Stage H orchestration: session-aware scheduling, idempotent runs, history,
trends and Telegram — wrapped around the unchanged Stage A-G ``run_screen``.

Two independent questions, deliberately kept apart:

* WHEN to check  — a local wall-clock trigger (default 09:15 Asia/Singapore,
  every calendar day incl. weekends) plus a check at startup.
* WHICH session  — always the latest *completed* NASDAQ session, resolved from
  the exchange calendar in trading_calendar.py. A Singapore weekday means
  nothing here: Friday's session is processed on Saturday morning; a Sunday
  startup still finds an unprocessed Friday.

Every check (startup / scheduled / manual) runs the same ``check_and_run``:

    resolve latest completed session ─ none → skip
    canonical result already stored ─ yes → skip (unless --force)
    US session in progress ─ yes → DEFERRED_MARKET_OPEN (session stays pending)
    acquire process lock ─ held → skip
    claim session in SQLite ─ race lost → skip
    wait for data readiness ─ not ready → DATA_NOT_READY (session stays pending)
    Stage A-G ─ exception / empty result → PIPELINE_FAILED (+ ops alert)
    persist full ranked population (atomic) → RESULTS_PERSISTED
    manifest → trends → Telegram → NOTIFIED | NOTIFICATION_FAILED

Three "did not produce a result" outcomes are deliberately distinct:
DATA_NOT_READY (provider late — retried), DEFERRED_MARKET_OPEN (unsafe to
screen partial bars — reconsidered at the next safe check) and PIPELINE_FAILED
(Stage A-G broke — not auto-retried the same day, to avoid repeated A-G runs).

Deferred retries (long-running loop only): when a check ends DATA_NOT_READY,
the session is kept pending and re-probed every ``deferred_retry_minutes``
(one readiness download per probe; Stage A-G only once it is ready) until the
local ``deferred_retry_cutoff``. Retries missed while the machine slept collapse
into a single catch-up probe on wake. After the cutoff the session is left for
the next startup / daily check.

Automatic catch-up is structurally limited to ONE session: Stage A-G always
screens the latest bars upstream exposes, so only the latest completed session
can be screened correctly. Older unrecorded sessions are reported as gaps and
never replayed (there is no backfill, because it would mislabel today's data).

Main check frequency (check_schedule.py, SCHEDULE_MODE): daily (default) |
times | interval. However often the loop checks, each check is the same
check_and_run, so Stage A-G still runs at most once per completed session.
While a deferred retry is pending, a main check for the same session is a
one-download probe (no second retry stream, no extra DATA_NOT_READY row), and
after PIPELINE_FAILED or the deferred cutoff, main checks leave that session
alone for the rest of the local day.

Usage:
  python -m pipeline.scheduler                      # long-running: startup check + scheduled checks
  python -m pipeline.scheduler --run-once           # one check now
  python -m pipeline.scheduler --run-once --force   # re-run even if the session already succeeded
  python -m pipeline.scheduler --dry-run            # show what a check would decide; runs nothing
  python -m pipeline.scheduler --resend [DATE]      # re-send a stored report (no Stage A-G)
  python -m pipeline.scheduler --test-telegram      # one connectivity-test message; nothing else
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from . import check_schedule as cs
from . import corporate_actions as ca
from . import history_store as hs
from . import keep_awake as ka
from . import position_monitor as pm
from . import readiness
from . import run_metadata
from . import telegram_notifier as tg
from . import trade_journal as tj
from . import trade_store as tstore
from . import trading_calendar as tc
from . import trend_analysis as ta
from .config import CONFIG
from .logging_config import run_log_file, setup_logging
from .run_lock import FileLock, LockHeld

# Explicit name: under `python -m pipeline.scheduler` __name__ is "__main__",
# which would detach this logger from the configured "pipeline" tree.
logger = logging.getLogger("pipeline.scheduler")

# Check outcomes (decisions, not stored run states — skips create no rows)
COMPLETED = "COMPLETED"
WOULD_RUN = "WOULD_RUN"
SKIPPED_NO_SESSION = "SKIPPED_NO_SESSION"
DEFERRED_MARKET_OPEN = hs.DEFERRED_MARKET_OPEN
SKIPPED_ALREADY_COMPLETED = "SKIPPED_ALREADY_COMPLETED"
SKIPPED_LOCKED = "SKIPPED_LOCKED"
SKIPPED_CLAIMED_ELSEWHERE = "SKIPPED_CLAIMED_ELSEWHERE"
SKIPPED_ON_HOLD = "SKIPPED_ON_HOLD"          # loop only: same-day hold after failure / cutoff
DATA_NOT_READY = hs.DATA_NOT_READY
PIPELINE_FAILED = hs.PIPELINE_FAILED

RUN_LOCK_NAME = "pipeline_run.lock"
INSTANCE_LOCK_NAME = "scheduler_instance.lock"


def _default_run_screen(config, run_ts, stats):
    from .run_pipeline import run_screen
    return run_screen(config, run_ts, stats)


@dataclass
class Deps:
    """Injectable collaborators — tests replace all of these; nothing sleeps or hits the network."""
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    sleep: Callable[[float], None] = time.sleep
    fetch_latest_bar: Callable = readiness.fetch_latest_bar_date
    run_screen: Callable = _default_run_screen
    telegram_factory: Callable = lambda config: tg.TelegramClient.from_env(config)
    collect_metadata: Callable = run_metadata.collect
    keep_awake: Callable = ka.keep_awake
    store: Optional[hs.HistoryStore] = None
    # Stage I (advisory trade journal / position monitor): daily bars for held
    # symbols and setup plans, corporate-action lookups, plus the journal
    # database. Tests replace all of them; none of them reaches the network.
    fetch_price_history: Callable = pm.fetch_bars
    fetch_corporate_actions: Callable = ca.yfinance_actions
    trade_store: Optional[object] = None


@dataclass
class CheckOutcome:
    decision: str
    trigger: str
    trading_date: Optional[object] = None
    run_id: Optional[str] = None
    detail: str = ""
    notification_status: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _store(config, deps):
    if deps.store is None:
        deps.store = hs.HistoryStore(config["history_db_path"])
    return deps.store


def _safe_error(exc, limit=500):
    """Exception summary with any secret env value scrubbed and length-capped."""
    text = f"{type(exc).__name__}: {exc}"
    for key in run_metadata.SECRET_ENV_VARS:
        value = os.environ.get(key)
        if value and len(value) >= 6:
            text = text.replace(value, "***")
    return text[:limit]


def _log_clock(now, config):
    local_tz = ZoneInfo(config["scheduler_timezone"])
    logger.info("Clock: local %s (%s) | New York %s | UTC %s",
                now.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %a"), config["scheduler_timezone"],
                now.astimezone(ZoneInfo(tc.EXCHANGE_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %a"),
                now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))


# ── the check ─────────────────────────────────────────────────────────────

def check_and_run(trigger, config=CONFIG, deps=None, *, force=False, dry_run=False,
                  alert_on_not_ready=True, probe_first=False, record_market_open=True):
    """The single orchestration path for startup, scheduled, deferred and manual checks.

    ``trigger="deferred"`` (or ``probe_first=True``) probes readiness once *before*
    claiming the session, so a still-late provider costs one download and writes
    no run row.
    ``alert_on_not_ready=False`` lets the loop send one DATA_NOT_READY alert at
    the cutoff instead of one per attempt.
    ``record_market_open=False`` skips the audit row when the loop has already
    recorded a DEFERRED_MARKET_OPEN for this session during the same US session.
    """
    deps = deps or Deps()
    store = _store(config, deps)
    now = deps.now()
    if now.tzinfo is None:
        raise ValueError("Deps.now() must return a timezone-aware datetime")
    buffer = config["session_close_buffer_minutes"]

    logger.info("== %s check%s ==", trigger, " (force)" if force else "")
    _log_clock(now, config)

    target = tc.latest_completed_session(now, buffer)
    if target is None:
        logger.warning("No completed NASDAQ session could be resolved — skipping.")
        return CheckOutcome(SKIPPED_NO_SESSION, trigger)
    logger.info("Target session: %s (latest completed NASDAQ session%s)",
                target, ", early close" if tc.is_early_close(target) else "")

    if store.has_successful_run(target) and not force:
        if trigger == "deferred":
            logger.info("Deferred target session %s already completed; no action required.", target)
        else:
            logger.info("Session %s already has a successful result — nothing to do.", target)
        return CheckOutcome(SKIPPED_ALREADY_COMPLETED, trigger, target)

    in_progress = tc.session_in_progress(now, buffer)
    if in_progress is not None:
        logger.info("US market currently open (session %s in progress or within %d min of close); "
                    "screening of %s deferred to next safe check.", in_progress, buffer, target)
        if not dry_run and record_market_open:
            local_tz = ZoneInfo(config["scheduler_timezone"])
            store.record_deferral(target, trigger, hs.DEFERRED_MARKET_OPEN,
                                  f"US session {in_progress} in progress; daily bars would be partial",
                                  local_timezone=config["scheduler_timezone"],
                                  local_started_at=now.astimezone(local_tz), now=now)
        return CheckOutcome(DEFERRED_MARKET_OPEN, trigger, target, detail=f"session {in_progress} in progress",
                            extra={"in_progress": in_progress})

    recorded = store.canonical_sessions(up_to=target)
    gaps = [s for s in tc.unrecorded_sessions(recorded, target, config["trend_lookback_sessions"])
            if s != target and recorded and s > recorded[0]]
    if gaps:
        logger.info("%d earlier session(s) in the lookback window were never recorded (%s). "
                    "They will not be backfilled — only the latest completed session can be screened.",
                    len(gaps), ", ".join(map(str, gaps)))

    if dry_run:
        logger.info("Dry run: would process %s now.", target)
        return CheckOutcome(WOULD_RUN, trigger, target)

    pre_verified = None
    if trigger == "deferred" or probe_first:
        ok, latest, reason = readiness.check_once(target, config["data_ready_symbol"], deps.fetch_latest_bar)
        logger.info("%s readiness probe %s: target=%s latest=%s — %s",
                    "Deferred" if trigger == "deferred" else "Pending-episode", config["data_ready_symbol"],
                    target, latest, reason)
        if not ok:
            return CheckOutcome(DATA_NOT_READY, trigger, target, detail=reason,
                                extra={"latest_bar": latest, "attempts": 1})
        pre_verified = readiness.ReadinessResult(True, target, latest, 1, reason)

    lock = FileLock(os.path.join(config["history_lock_dir"], RUN_LOCK_NAME))
    try:
        lock.acquire()
    except LockHeld:
        logger.info("Another process holds the run lock — skipping to avoid duplicate work.")
        return CheckOutcome(SKIPPED_LOCKED, trigger, target)
    try:
        return _run_session(trigger, target, now, config, deps, store, force,
                            pre_verified=pre_verified, alert_on_not_ready=alert_on_not_ready)
    finally:
        lock.release()


def _run_session(trigger, target, now, config, deps, store, force, *, pre_verified=None,
                 alert_on_not_ready=True):
    local_tz = ZoneInfo(config["scheduler_timezone"])
    try:
        run_id = store.claim_session(
            target, trigger, forced=force, local_timezone=config["scheduler_timezone"],
            local_started_at=now.astimezone(local_tz), stale_after_hours=config["run_stale_after_hours"], now=now,
        )
    except hs.AlreadyCompleted:
        logger.info("Session %s completed by another process — skipping.", target)
        return CheckOutcome(SKIPPED_ALREADY_COMPLETED, trigger, target)
    except hs.SessionInProgress:
        logger.info("Session %s is already claimed by an active run — skipping.", target)
        return CheckOutcome(SKIPPED_CLAIMED_ELSEWHERE, trigger, target)
    logger.info("Claimed session %s as run %s.", target, run_id)

    ready = pre_verified or readiness.wait_for_data(target, config, fetch=deps.fetch_latest_bar,
                                                    sleep=deps.sleep)
    if not ready.ready:
        logger.warning("Market data not ready for %s after %d attempt(s) (latest bar: %s) — "
                       "not generating a stale screen; the session stays pending.",
                       target, ready.attempts, ready.latest_bar)
        store.mark_failed(run_id, hs.DATA_NOT_READY, error_summary=ready.reason,
                          latest_bar=ready.latest_bar, attempts=ready.attempts)
        run_metadata.write_manifest(config, store.get_run(run_id))
        if alert_on_not_ready:
            send_ops_alert(run_id, tg.build_data_not_ready_alert(target, ready.latest_bar, ready.attempts),
                           config, deps, store)
        return CheckOutcome(DATA_NOT_READY, trigger, target, run_id, ready.reason,
                            extra={"latest_bar": ready.latest_bar, "attempts": ready.attempts})

    run_ts = datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    try:
        metadata = deps.collect_metadata(config)
    except Exception:
        logger.warning("Run metadata collection failed — continuing without it", exc_info=True)
        metadata = {}
    store.mark_running(run_id, latest_bar=ready.latest_bar, attempts=ready.attempts,
                       run_timestamp=run_ts, metadata=metadata)

    stats = {}
    # Idle sleep mid-run would fail every in-flight fetch and still yield a canonical result.
    with run_log_file(config, run_ts) as log_path, deps.keep_awake():
        logger.info("Stage A-G starting for session %s (run %s, config %s, commit %s).", target, run_id,
                    metadata.get("config_hash"), (metadata.get("git_commit") or "unknown")[:12])
        try:
            result = deps.run_screen(config, run_ts, stats)
        except Exception as e:
            stage = stats.get("current_stage")
            logger.error("Stage A-G failed at stage %s", stage, exc_info=True)
            return _fail_pipeline(run_id, target, trigger, stage, _safe_error(e), stats, log_path,
                                  config, deps, store)

        if result.final_watchlist is None or result.final_watchlist.empty:
            stage = result.stopped_after or stats.get("current_stage")
            msg = f"no ranked results (stopped after Stage {stage})"
            logger.error("Stage A-G produced %s — treating as a failed run so it can be retried.", msg)
            return _fail_pipeline(run_id, target, trigger, stage, msg, stats, log_path, config, deps, store)
        logger.info("Stage A-G finished: %d ranked, %d VCP rows.", len(result.final_watchlist), len(result.vcp_df))

        artifacts = {
            "run_timestamp": run_ts,
            "log": log_path,
            "watchlist_csv": result.watchlist_path,
            "vcp_report_csv": result.vcp_report_path,
            "chart_dir": result.chart_dir,
            "vcp_debug_dir": result.vcp_debug_dir,
        }
        counts = {k: v for k, v in stats.items() if k != "current_stage"}
        try:
            n = store.persist_results(run_id, target, result.final_watchlist, result.vcp_df,
                                      chart_paths=result.chart_paths, debug_chart_dir=result.vcp_debug_dir,
                                      stage_counts=counts, artifacts=artifacts, forced=force)
        except Exception as e:
            logger.error("Persisting results failed — run is NOT canonical", exc_info=True)
            return _fail_pipeline(run_id, target, trigger, "H:persist", _safe_error(e), stats, log_path,
                                  config, deps, store)
        logger.info("Persisted %d ranked results for %s to %s.", n, target, config["history_db_path"])
        logger.info("Stage counts: %s", counts)

        manifest = run_metadata.write_manifest(config, store.get_run(run_id))
        if manifest:
            logger.info("Run manifest: %s", manifest)

        positions, corporate = run_trade_layer(run_id, target, config, deps, store)

        trend = None
        try:
            trend = build_trend_report(store, target, config)
            logger.info("Trends vs %s: %d new, %d dropped, %d rising, %d falling, %d persistent.",
                        trend.previous_session, len(trend.new_entries), len(trend.dropped),
                        len(trend.by_category(ta.RISING)), len(trend.by_category(ta.FALLING)),
                        len(trend.by_category(ta.PERSISTENT)))
        except Exception:
            logger.warning("Trend calculation failed — notification will retry it", exc_info=True)

        notification_status = notify_run(run_id, config, deps, store, trend=trend, positions=positions,
                                         corporate_actions=corporate)
        run_metadata.write_manifest(config, store.get_run(run_id), extra={"telegram_status": notification_status})

    return CheckOutcome(COMPLETED, trigger, target, run_id, notification_status=notification_status)


def _fail_pipeline(run_id, target, trigger, stage, error, stats, log_path, config, deps, store):
    counts = {k: v for k, v in stats.items() if k != "current_stage"}
    store.mark_failed(run_id, hs.PIPELINE_FAILED, error_summary=error, failed_stage=stage, stage_counts=counts)
    run_metadata.write_manifest(config, store.get_run(run_id), extra={"log": log_path})
    send_ops_alert(run_id, tg.build_pipeline_failed_alert(target, stage, log_path, error), config, deps, store)
    return CheckOutcome(PIPELINE_FAILED, trigger, target, run_id, error)


# ── Stage I: advisory trade journal / position monitor ───────────────────
# Strictly downstream of a canonical run and strictly advisory: it reads the
# stored Stage-G output, records gate states and plans, and snapshots open
# positions. It never changes screening, scoring, ranking or VCP results, never
# fails a run, and never contacts a broker — this project has no brokerage API,
# no account access and no order placement.

def run_trade_layer(run_id, trading_date, config, deps, store):
    """Gate setups, apply corporate actions, then monitor open positions.

    The order matters and is fixed (§10)::

        screening -> setup plans -> corporate-action detection
                  -> safe corporate-action adjustment -> position monitor -> report

    Corporate actions come first so the monitor never reads a stale share basis
    (a position that just split 2-for-1 would otherwise look like an instant
    50% loss against its recorded stop). Every open position is checked, whether
    or not the screener saw it today.

    Returns ``(snapshots, actions)`` — both empty on any problem. A Stage-I
    failure never changes the run's canonical outcome, and nothing here
    contacts a broker.
    """
    if not config.get("trade_journal_enabled", True):
        return [], []
    try:
        journal = deps.trade_store or tstore.open_store(config, create=True)
        results = store.results_for_run(run_id)
        tj.generate_setup_plans(run_id, trading_date, results, config, store=journal,
                                fetch=deps.fetch_price_history)
        actions = run_corporate_actions(trading_date, config, deps, journal)
        snapshots = pm.monitor_open_positions(trading_date, config, store=journal,
                                              fetch=deps.fetch_price_history)
        return snapshots, actions
    except Exception:
        logger.warning("Trade journal layer failed — the screening result is unaffected", exc_info=True)
        return [], []


def run_corporate_actions(trading_date, config, deps, journal):
    """Detect and safely apply corporate actions for every open position.

    Its own try/except: a corporate-action problem must not stop the position
    monitor, let alone the run.
    """
    if not config.get("trade_corporate_actions_enabled", True):
        return []
    try:
        return ca.process_open_positions(trading_date, config, store=journal,
                                         fetch_actions=deps.fetch_corporate_actions,
                                         now=deps.now())
    except Exception:
        logger.warning("Corporate-action processing failed — positions are still monitored",
                       exc_info=True)
        return []


def stored_position_snapshots(config, trading_date):
    """Snapshots recorded for a session (used by --resend too). Never creates the DB."""
    if not (config.get("trade_journal_enabled", True) and config.get("trade_positions_in_report", True)):
        return []
    try:
        journal = tstore.open_store(config, create=False)
        return journal.snapshots_for_date(trading_date) if journal else []
    except Exception:
        logger.warning("Could not read position snapshots for the report", exc_info=True)
        return []


def position_buttons_enabled(config):
    """Open/View Position buttons need the journal AND the separate trade-bot
    process that answers them — otherwise a tap would go nowhere."""
    return bool(config.get("trade_journal_enabled", True) and config.get("trade_bot_enabled", False))


@dataclass
class JournalContext:
    """What the report reads back from the trade journal (never recomputed)."""
    plans: dict = field(default_factory=dict)          # {symbol: setup_plans row} for this run
    open_trades: list = field(default_factory=list)    # OPEN trades right now


def report_journal_context(config, trading_date, run_id=None, deps=None):
    """The persisted Stage-I setup plans for a run and the OPEN trades. Never
    creates the journal; any problem degrades to an empty context."""
    if not config.get("trade_journal_enabled", True):
        return JournalContext()
    try:
        journal = (deps.trade_store if deps is not None else None) or tstore.open_store(config, create=False)
        if journal is None:
            return JournalContext()
        rows = journal.setup_plans_for_date(trading_date)
        # A --force rerun can leave a superseded run's plans on the same date.
        own = [p for p in rows if run_id and p.get("run_id") == run_id]
        plans = {p["symbol"]: p for p in (own or rows)}
        return JournalContext(plans=plans, open_trades=journal.open_positions())
    except Exception:
        logger.warning("Could not read setup plans / open trades for the report", exc_info=True)
        return JournalContext()


# ── trends + notification ─────────────────────────────────────────────────

def build_trend_report(store, trading_date, config):
    trading_date = tc.to_date(trading_date)
    window = tc.recent_trading_sessions(config["trend_lookback_sessions"], trading_date)
    recorded = [s for s in store.canonical_sessions(up_to=trading_date) if window and s >= window[0]]
    if trading_date not in recorded:
        recorded.append(trading_date)
    rows = store.results_for_sessions(recorded)
    return ta.analyze(rows, recorded, top_n=config["trend_top_n"],
                      rank_move_min=config["trend_rank_move_min"],
                      persistent_min_streak=config["trend_persistent_min_streak"],
                      expected_sessions=window)


def _chart_rows(results, config):
    """The Stage-G verification charts that will actually be sent (on disk)."""
    if not config.get("telegram_send_charts"):
        return []
    rows = tg.select_vcp_chart_rows(results, config["telegram_vcp_chart_max"])
    return [r for r in rows if r.get("vcp_debug_chart_path") and os.path.exists(r["vcp_debug_chart_path"])]


def notify_run(run_id, config, deps, store, trend=None, positions=None, corporate_actions=None):
    """Send the stored reports for a canonical run. Never raises; returns
    SENT / FAILED / SKIPPED and moves the run to NOTIFIED / NOTIFICATION_FAILED.

    Deliveries, each logically separate: the main dashboard (its status is the
    run's notification status), corporate-action alerts, the VCP verification
    charts and the optional debug summary. Only the main report can fail the
    notification; the others are logged and recorded, never escalated.

    ``positions`` are this session's Stage-I snapshots; when not supplied (a
    --resend) they are read back from the trade journal. ``corporate_actions``
    are this session's processed events, sent as their own short messages after
    the report (§11) — read-only detection needs no confirmation."""
    if not config.get("telegram_enabled"):
        logger.info("Telegram disabled (TELEGRAM_ENABLED=0) — no notification sent.")
        store.record_notification(run_id, "report", "SKIPPED", detail="TELEGRAM_ENABLED=0")
        return "SKIPPED"
    try:
        run = store.get_run(run_id)
        client = deps.telegram_factory(config)
        trend = trend or build_trend_report(store, run["trading_date"], config)
        results = store.results_for_run(run_id)
        counts = json.loads(run["stage_counts_json"]) if run.get("stage_counts_json") else {}
        if positions is None:
            positions = stored_position_snapshots(config, run["trading_date"])
        journal = report_journal_context(config, run["trading_date"], run_id, deps)
        buttons = position_buttons_enabled(config)
        chart_rows = _chart_rows(results, config)
        chunks = tg.build_report(run["trading_date"], counts, results, trend,
                                 max_candidates=config["telegram_report_candidates"],
                                 positions=positions, plans=journal.plans, open_trades=journal.open_trades)
        keyboard = None
        if buttons:
            # A READY setup whose chart is sent gets its button on the chart instead.
            keyboard = tg.build_report_keyboard(results, journal.plans, journal.open_trades, positions,
                                                exclude={r["symbol"] for r in chart_rows})
        logger.info("Sending Telegram report for %s (%d message(s))...", run["trading_date"], len(chunks))
        sent = tg.send_text_chunks(client, chunks, reply_markup=keyboard)
        store.record_notification(run_id, "report", "SENT" if sent.ok else "FAILED", attempts=sent.attempts,
                                  error_summary=sent.error,
                                  provider_message_id=",".join(str(m) for m in sent.message_ids if m is not None) or None,
                                  detail=f"{len(sent.message_ids)}/{len(chunks)} chunks")
        if not sent.ok:
            logger.error("Telegram report failed: %s", sent.error)
            store.set_notification_status(run_id, hs.NOTIFICATION_FAILED)
            return "FAILED"
        logger.info("Telegram report sent.")

        _send_corporate_action_alerts(client, corporate_actions)

        if chart_rows:
            _send_charts(client, run_id, chart_rows, store, plans=journal.plans,
                         open_trades=journal.open_trades, buttons=buttons)
        if config.get("telegram_debug_summary", True):
            _send_debug_summary(client, run_id, run, counts, results, config, store, journal=journal,
                                positions=positions, corporate_actions=corporate_actions, buttons=buttons,
                                report_status=f"SENT ({len(chunks)} message(s))")
        store.set_notification_status(run_id, hs.NOTIFIED)
        return "SENT"
    except Exception as e:
        error = _safe_error(e)
        logger.error("Telegram notification failed: %s", error)
        try:
            store.record_notification(run_id, "report", "FAILED", error_summary=error)
            store.set_notification_status(run_id, hs.NOTIFICATION_FAILED)
        except Exception:
            logger.warning("Could not record notification failure", exc_info=True)
        return "FAILED"


def _send_debug_summary(client, run_id, run, counts, results, config, store, *, journal, positions,
                        corporate_actions, buttons, report_status):
    """The separate 🛠 debug message. Non-critical: a failure is logged and
    recorded as a ``debug`` notification, and the run stays successful."""
    try:
        chunks = tg.build_debug_report(run["trading_date"], counts, results, run=run, plans=journal.plans,
                                       positions=positions, open_trade_count=len(journal.open_trades),
                                       corporate_actions=corporate_actions, config=config,
                                       buttons_enabled=buttons, report_status=report_status)
        sent = tg.send_text_chunks(client, chunks)
        store.record_notification(run_id, "debug", "SENT" if sent.ok else "FAILED", attempts=sent.attempts,
                                  error_summary=sent.error,
                                  detail=f"{len(sent.message_ids)}/{len(chunks)} chunks")
        if not sent.ok:
            logger.warning("Telegram debug summary not delivered (report already sent): %s", sent.error)
    except Exception as e:
        error = _safe_error(e)
        logger.warning("Telegram debug summary failed (report already sent): %s", error)
        try:
            store.record_notification(run_id, "debug", "FAILED", error_summary=error)
        except Exception:
            logger.debug("could not record debug summary failure", exc_info=True)


def _send_corporate_action_alerts(client, actions):
    """One short message per corporate action that touched an open position.

    Non-critical, like every other notification here: a send failure is logged
    and the run stays successful. Detection and safe adjustment are read-only
    for the user — no confirmation is asked for them; only an action that would
    need an uncertain journal mutation says so and waits for the user.
    """
    for action in actions or []:
        try:
            client.send_message(tg.build_corporate_action_alert(action))
        except Exception:
            logger.warning("Corporate-action alert for %s could not be sent",
                           action.get("symbol"), exc_info=True)


def _send_charts(client, run_id, rows, store, plans=None, open_trades=(), buttons=False):
    """Sends the existing Stage-G VCP verification charts (never re-rendered).

    A READY chart's caption carries its persisted entry/stop/size, and — when the
    trade bot is on — an Open Position button (View Position if already held)."""
    plans = plans or {}
    held = {t["symbol"]: t for t in open_trades or []}
    ready = set(tg.button_symbols(rows, plans, open_trades)) if buttons else set()
    for row in rows:
        path = row.get("vcp_debug_chart_path")
        if not path or not os.path.exists(path):
            logger.info("  %s: no VCP verification chart on disk — not sent.", row["symbol"])
            continue
        symbol = row["symbol"]
        markup = tg.position_keyboard(symbol, symbol in held) if symbol in ready else None
        try:
            mid, attempts = client.send_photo(path, caption=tg.build_chart_caption(row, plans.get(symbol),
                                                                                 held.get(symbol)),
                                              reply_markup=markup)
            store.record_notification(run_id, "chart", "SENT", attempts=attempts,
                                      provider_message_id=str(mid) if mid is not None else None,
                                      detail=symbol)
        except Exception as e:
            error = _safe_error(e)
            logger.warning("  %s: chart send failed (report already delivered): %s", symbol, error)
            store.record_notification(run_id, "chart", "FAILED", error_summary=error, detail=symbol)


def send_ops_alert(run_id, text, config, deps, store):
    """Best-effort operational alert. Never raises and never masks the original failure."""
    if not (config.get("telegram_enabled") and config.get("telegram_ops_alerts")):
        return None
    try:
        client = deps.telegram_factory(config)
        mid, attempts = client.send_message(text)
        store.record_notification(run_id, "ops_alert", "SENT", attempts=attempts,
                                  provider_message_id=str(mid) if mid is not None else None)
        return "SENT"
    except Exception as e:
        error = _safe_error(e)
        logger.warning("Operational alert could not be sent: %s", error)
        try:
            store.record_notification(run_id, "ops_alert", "FAILED", error_summary=error)
        except Exception:
            logger.debug("could not record ops alert failure", exc_info=True)
        return "FAILED"


def resend_notification(config=CONFIG, deps=None, trading_date=None):
    deps = deps or Deps()
    store = _store(config, deps)
    run = store.canonical_run(tc.to_date(trading_date)) if trading_date else store.latest_canonical_run()
    if run is None:
        logger.error("No successful run found%s.", f" for {trading_date}" if trading_date else "")
        return None
    if not config.get("telegram_enabled"):
        logger.error("TELEGRAM_ENABLED=0 — enable it to resend.")
        return "SKIPPED"
    logger.info("Resending stored report for %s (run %s) — Stage A-G is not re-run.",
                run["trading_date"], run["run_id"])
    status = notify_run(run["run_id"], config, deps, store)
    run_metadata.write_manifest(config, store.get_run(run["run_id"]), extra={"telegram_status": status})
    return status


TEST_MESSAGE = "✅ NASDAQ Stage 2 Telegram connection test successful"


def send_test_message(config=CONFIG, deps=None):
    """Send one connectivity-test message. No Stage A-G, no SQLite access.
    Returns True on delivery. Errors are redacted before logging."""
    deps = deps or Deps()
    if not config.get("telegram_enabled"):
        logger.warning("TELEGRAM_ENABLED is off: this test message is sent anyway, but scheduled "
                       "reports and ops alerts will NOT be sent until TELEGRAM_ENABLED=1.")
    try:
        client = deps.telegram_factory(config)
        local = deps.now().astimezone(ZoneInfo(config["scheduler_timezone"]))
        mid, attempts = client.send_message(
            f"{TEST_MESSAGE}\n<i>{local.strftime('%Y-%m-%d %H:%M')} {tg.esc(config['scheduler_timezone'])}"
            f" · no pipeline run</i>")
    except Exception as e:
        logger.error("Telegram connection test FAILED: %s", _safe_error(e))
        return False
    logger.info("Telegram connection test OK (message id %s, %d attempt(s)).", mid, attempts)
    return True


# ── status (dry-run summary + history_cli status) ────────────────────────

def build_status(config, now, store=None):
    """Read-only operational snapshot. Mirrors check_and_run's decision order
    (no session -> already successful -> market open -> would run) without
    locking, claiming, downloading or writing anything."""
    buffer = config["session_close_buffer_minutes"]
    local_tz = ZoneInfo(config["scheduler_timezone"])
    target = tc.latest_completed_session(now, buffer)
    in_progress = tc.session_in_progress(now, buffer)
    st = {
        "now_local": now.astimezone(local_tz), "now_new_york": now.astimezone(ZoneInfo(tc.EXCHANGE_TIMEZONE)),
        "us_session_in_progress": in_progress, "latest_completed_session": target,
        "latest_successful": None, "latest_attempt_for_target": None, "telegram_status": None,
        "schedule": None, "schedule_error": None, "next_scheduled_check": None,
    }
    try:
        st["schedule"] = cs.from_config(config)
        if st["schedule"].mode != cs.INTERVAL:
            # Wall-clock modes are fully determined by config. Interval mode depends
            # on when the running loop last checked, which this process cannot see.
            st["next_scheduled_check"] = st["schedule"].next_after(now)
    except cs.ScheduleConfigError as e:
        st["schedule_error"] = str(e)
    if store is not None:
        st["latest_successful"] = store.latest_canonical_run()
        attempts = [r for r in store.recent_runs(50) if target and r["trading_date"] == str(target)]
        st["latest_attempt_for_target"] = attempts[0] if attempts else None
        if st["latest_successful"]:
            reports = [n for n in store.notifications_for_run(st["latest_successful"]["run_id"])
                       if n["kind"] == "report"]
            st["telegram_status"] = reports[-1]["status"] if reports else None

    success = store is not None and target is not None and store.has_successful_run(target)
    if target is None:
        st["decision"], st["reason"] = SKIPPED_NO_SESSION, "no completed NASDAQ session could be resolved"
    elif success:
        st["decision"], st["reason"] = SKIPPED_ALREADY_COMPLETED, f"session {target} already has a successful result"
    elif in_progress is not None:
        st["decision"] = DEFERRED_MARKET_OPEN
        st["reason"] = (f"US session {in_progress} in progress (or within {buffer} min of close); "
                        f"{target} would be recorded DEFERRED_MARKET_OPEN and left for the next safe check")
    else:
        st["decision"] = WOULD_RUN
        st["reason"] = (f"session {target} has no successful result and the US market is closed; "
                        f"Stage A-G would start once {config['data_ready_symbol']}'s latest bar == {target}")
    st["missing_latest_session"] = target is not None and not success
    return st


def format_status(st, config):
    succ = st["latest_successful"]
    attempt = st["latest_attempt_for_target"]
    yes_no = lambda b: "Yes" if b else "No"
    lines = [
        ("Local time", f"{st['now_local']:%Y-%m-%d %H:%M:%S %a} ({config['scheduler_timezone']})"),
        ("New York time", f"{st['now_new_york']:%Y-%m-%d %H:%M:%S %a}"),
        ("US session in progress", f"Yes ({st['us_session_in_progress']})" if st["us_session_in_progress"] else "No"),
        ("Latest completed XNAS session", st["latest_completed_session"] or "none"),
        ("Latest successful run", f"{succ['trading_date']} ({succ['trigger']}, run {succ['run_id'][:8]})"
                                  if succ else "none"),
        ("Latest run state", succ["status"] if succ else "-"),
        ("Latest Telegram status", st["telegram_status"] or ("-" if not succ else "not recorded")),
        ("Missing latest session", yes_no(st["missing_latest_session"])),
    ]
    if attempt and st["missing_latest_session"]:
        lines.append(("Last attempt for that session", f"{attempt['status']} at {(attempt['started_at_utc'] or '')[:19]} UTC"))
    lines += [
        ("Would a check run now", f"{yes_no(st['decision'] == WOULD_RUN)} ({st['decision']})"),
        ("Reason", st["reason"]),
    ]
    schedule = st.get("schedule")
    if st.get("schedule_error"):
        lines.append(("Schedule", f"INVALID: {st['schedule_error']}"))
    elif schedule is not None:
        lines += schedule.describe()
        if st["next_scheduled_check"] is not None:
            lines.append(("Next scheduled check", f"{st['next_scheduled_check']:%Y-%m-%d %H:%M %a} "
                                                  f"(if a scheduler loop is running)"))
        else:
            lines.append(("Next scheduled check", f"{schedule.interval_minutes} min after the running loop's "
                                                  f"previous check (loop state is not visible here)"))
    width = max(len(k) for k, _ in lines)
    return [f"{k:<{width}} : {v}" for k, v in lines]


# ── long-running loop ─────────────────────────────────────────────────────

def parse_local_time(value):
    hh, mm = value.strip().split(":")
    return dtime(int(hh), int(mm))


def next_trigger_after(now, local_time, tz_name):
    """Next wall-clock occurrence of ``local_time`` in ``tz_name`` strictly after ``now``."""
    tz = ZoneInfo(tz_name)
    local_now = now.astimezone(tz)
    candidate = datetime.combine(local_now.date(), local_time, tzinfo=tz)
    if candidate <= local_now:
        candidate = datetime.combine(local_now.date() + timedelta(days=1), local_time, tzinfo=tz)
    return candidate


@dataclass
class PendingSession:
    """A session whose readiness check failed and is waiting for a deferred retry."""
    trading_date: object
    due: datetime
    run_id: Optional[str]        # the DATA_NOT_READY attempt row (for the cutoff alert)
    attempts: int
    latest_bar: object


def _deferred_cutoff(day_of, config):
    tz = ZoneInfo(config["scheduler_timezone"])
    local = day_of.astimezone(tz)
    return datetime.combine(local.date(), parse_local_time(config["deferred_retry_cutoff"]), tzinfo=tz)


def _hhmm(dt, config):
    return dt.astimezone(ZoneInfo(config["scheduler_timezone"])).strftime("%H:%M")


def _give_up_for_today(pending, config, deps):
    tz_label = "SGT" if config["scheduler_timezone"] == "Asia/Singapore" else config["scheduler_timezone"]
    logger.warning("Target session %s still not ready at %s %s. Deferring until the next safe "
                   "scheduler/startup check.", pending.trading_date, config["deferred_retry_cutoff"], tz_label)
    if pending.run_id:
        send_ops_alert(pending.run_id, tg.build_data_not_ready_alert(
            pending.trading_date, pending.latest_bar, pending.attempts,
            note=f"Deferred retries stopped at the {config['deferred_retry_cutoff']} cutoff; "
                 "the next startup/daily check will try again."), config, deps, _store(config, deps))


def plan_deferred_retry(outcome, now, pending, config, deps):
    """Next pending state after a check. Returns a PendingSession or None."""
    if outcome is None:                      # unexpected error inside the check: keep the plan
        return pending
    if outcome.decision != DATA_NOT_READY:
        if pending is not None:
            if outcome.decision in (COMPLETED, SKIPPED_ALREADY_COMPLETED):
                logger.info("Deferred retries for %s cancelled: session completed.", pending.trading_date)
            else:
                logger.info("Deferred retries for %s cancelled (%s).", pending.trading_date, outcome.decision)
        return None
    if not config.get("deferred_retry_enabled", True):
        return None

    same = pending is not None and pending.trading_date == outcome.trading_date
    extra = outcome.extra or {}
    state = PendingSession(
        trading_date=outcome.trading_date, due=now,
        run_id=outcome.run_id or (pending.run_id if same else None),
        attempts=(pending.attempts if same else 0) + int(extra.get("attempts") or 1),
        latest_bar=extra.get("latest_bar"),
    )
    cutoff = _deferred_cutoff(now, config)
    if now >= cutoff:
        _give_up_for_today(state, config, deps)
        return None
    state.due = min(now + timedelta(minutes=float(config["deferred_retry_minutes"])), cutoff)
    if same:
        logger.info("Target session %s still not ready. Next deferred retry at %s.",
                    state.trading_date, _hhmm(state.due, config))
    else:
        logger.info("Target session %s still not ready after initial retries. Data not ready; entering "
                    "deferred retry mode. Deferred retry scheduled for %s (cutoff %s).",
                    state.trading_date, _hhmm(state.due, config), config["deferred_retry_cutoff"])
    return state


@dataclass
class SameDayHold:
    """Main checks leave ``trading_date`` alone for the rest of ``local_date``:
    set after PIPELINE_FAILED (never repeat A-G automatically the same day) or
    when deferred retries reach the cutoff. Startup/manual checks ignore it."""
    trading_date: object
    local_date: object
    reason: str


def _local_date(dt, config):
    return dt.astimezone(ZoneInfo(config["scheduler_timezone"])).date()


def _update_hold(outcome, pending, hold, now, config):
    if outcome is None:
        return hold
    if outcome.decision in (COMPLETED, SKIPPED_ALREADY_COMPLETED):
        return None
    if outcome.decision == PIPELINE_FAILED:
        return SameDayHold(outcome.trading_date, _local_date(now, config), "Stage A-G failed (PIPELINE_FAILED)")
    if outcome.decision == DATA_NOT_READY and pending is None and config.get("deferred_retry_enabled", True):
        # plan_deferred_retry returned nothing although retries are on: the cutoff was reached.
        return SameDayHold(outcome.trading_date, _local_date(now, config),
                           f"data still not ready at the {config['deferred_retry_cutoff']} cutoff")
    return hold


def _market_open_key(outcome, previous):
    if outcome is not None and outcome.decision == DEFERRED_MARKET_OPEN:
        return (outcome.trading_date, (outcome.extra or {}).get("in_progress"))
    return previous


def _scheduled_check(config, deps, pending, hold, market_open_seen, alert_each):
    """One main (scheduled) check, reconciled with the loop's episode state.

    Returns (outcome, market_open_seen)."""
    now = deps.now()
    buffer = config["session_close_buffer_minutes"]
    target = tc.latest_completed_session(now, buffer)
    if (hold is not None and target is not None and hold.trading_date == target
            and hold.local_date == _local_date(now, config)
            and not _store(config, deps).has_successful_run(target)):
        logger.info("Scheduled check: session %s is on hold for today (%s); no automatic attempt until "
                    "the next local day or a scheduler restart (manual: --run-once).", target, hold.reason)
        return CheckOutcome(SKIPPED_ON_HOLD, "scheduled", target, detail=hold.reason), market_open_seen
    probe_first = pending is not None and target is not None and pending.trading_date == target
    if probe_first:
        logger.info("Scheduled check joins the pending readiness episode for %s (one probe, no new claim).", target)
    in_progress = tc.session_in_progress(now, buffer)
    outcome = _guarded_check("scheduled", config, deps, alert_on_not_ready=alert_each, probe_first=probe_first,
                             record_market_open=market_open_seen != (target, in_progress))
    return outcome, _market_open_key(outcome, market_open_seen)


def run_forever(config=CONFIG, deps=None, max_iterations=None):
    """Startup check, then main checks on the configured schedule (daily /
    times / interval, see check_schedule.py), plus bounded same-day deferred
    retries while a session's data is late.

    Wakes at least every ``scheduler_poll_seconds`` and compares the wall clock
    with the next due times, so a laptop that slept through one or more
    scheduled checks (or deferred retries) runs ONE catch-up check on resume.
    Raises check_schedule.ScheduleConfigError on an invalid schedule.
    """
    deps = deps or Deps()
    if not config.get("scheduler_enabled", True):
        logger.warning("SCHEDULE_ENABLED=0 — scheduler loop not started (use --run-once for a manual check).")
        return 0
    schedule = cs.from_config(config)        # fail fast, before taking the instance lock

    instance_lock = FileLock(os.path.join(config["history_lock_dir"], INSTANCE_LOCK_NAME))
    try:
        instance_lock.acquire()
    except LockHeld:
        # The normal outcome for the Windows 09:05 watchdog trigger: the loop is
        # already up, so there is nothing to do. Exit 0 so Task Scheduler does not
        # treat it as a failure and start its restart-on-failure retries.
        logger.info("Scheduler already running (instance lock held) — nothing to do.")
        return 0

    try:
        logger.info("Scheduler started: %s checks %s, startup check %s, exchange %s, history %s. "
                    "Stage A-G runs at most once per completed session.", schedule.mode, schedule.summary(),
                    "on" if config["scheduler_run_on_startup"] else "off", tc.EXCHANGE, config["history_db_path"])

        poll = float(config["scheduler_poll_seconds"])
        # With deferred retries on, a DATA_NOT_READY alert is sent once at the
        # cutoff rather than after every attempt.
        alert_each = not config.get("deferred_retry_enabled", True)

        # Loop state. `pending` is the single readiness episode (deferred retries);
        # `hold` stops main checks re-attempting a session later the same local day;
        # `market_open_seen` records one DEFERRED_MARKET_OPEN row per US session.
        pending = hold = market_open_seen = None
        if config["scheduler_run_on_startup"]:
            outcome = _guarded_check("startup", config, deps, alert_on_not_ready=alert_each)
            pending = plan_deferred_retry(outcome, deps.now(), pending, config, deps)
            hold = _update_hold(outcome, pending, hold, deps.now(), config)
            market_open_seen = _market_open_key(outcome, market_open_seen)
        # Computed after the startup check: an occurrence that came due while it was
        # running is covered by it (no second check for the same moment). A late-data
        # or failed outcome is handled by the pending retry / hold above instead.
        next_due = schedule.next_after(deps.now())
        logger.info("Next scheduled check: %s", next_due.isoformat())

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            now = deps.now()
            if now >= next_due:
                late = (now - next_due).total_seconds()
                if late > 2 * poll:
                    missed = schedule.occurrences_between(next_due, now)
                    logger.info("Missed scheduled check due %s (%.0f min late — system was likely asleep "
                                "or suspended); running it now%s.", next_due.isoformat(), late / 60,
                                f" ({missed} missed occurrences collapsed into one check)" if missed > 1 else "")
                outcome, market_open_seen = _scheduled_check(config, deps, pending, hold, market_open_seen,
                                                             alert_each)
                if outcome is None or outcome.decision != SKIPPED_ON_HOLD:
                    pending = plan_deferred_retry(outcome, deps.now(), pending, config, deps)
                    hold = _update_hold(outcome, pending, hold, deps.now(), config)
                next_due = schedule.next_after(deps.now())
                logger.info("Next scheduled check: %s", next_due.isoformat())
                continue
            if pending is not None and now >= pending.due:
                if now > _deferred_cutoff(pending.due, config):
                    logger.info("Deferred retry due %s was missed and its cutoff has passed.", pending.due.isoformat())
                    _give_up_for_today(pending, config, deps)
                    hold = SameDayHold(pending.trading_date, _local_date(now, config),
                                       f"data still not ready at the {config['deferred_retry_cutoff']} cutoff")
                    pending = None
                    continue
                late = (now - pending.due).total_seconds()
                if late > 2 * poll:
                    logger.info("Missed deferred retry due %s (%.0f min late — system was likely asleep); "
                                "running one catch-up check now.", pending.due.isoformat(), late / 60)
                outcome = _guarded_check("deferred", config, deps, alert_on_not_ready=False)
                pending = plan_deferred_retry(outcome, deps.now(), pending, config, deps)
                hold = _update_hold(outcome, pending, hold, deps.now(), config)
                market_open_seen = _market_open_key(outcome, market_open_seen)
                continue
            wake = next_due if pending is None else min(next_due, pending.due)
            deps.sleep(max(1.0, min(poll, (wake - now).total_seconds())))
        return 0
    finally:
        instance_lock.release()


def _guarded_check(trigger, config, deps, **kwargs):
    """A bug in one check must not kill the long-running loop."""
    try:
        outcome = check_and_run(trigger, config, deps, **kwargs)
        logger.info("%s check outcome: %s %s", trigger, outcome.decision, outcome.trading_date or "")
        return outcome
    except Exception:
        logger.exception("Unexpected error during %s check — scheduler keeps running", trigger)
        return None


# ── CLI ───────────────────────────────────────────────────────────────────

_EXIT_CODES = {PIPELINE_FAILED: 1, DATA_NOT_READY: 1}


def main(argv=None, config=CONFIG):
    parser = argparse.ArgumentParser(prog="python -m pipeline.scheduler",
                                     description="Session-aware Stage 2 scheduler (Stage H).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-once", action="store_true", help="Run a single check now and exit.")
    mode.add_argument("--dry-run", action="store_true", help="Report what a check would do; run nothing.")
    mode.add_argument("--resend", nargs="?", const="latest", metavar="DATE",
                      help="Re-send the stored Telegram report for DATE (default: latest successful run).")
    mode.add_argument("--test-telegram", action="store_true",
                      help="Send one Telegram connectivity-test message; runs no pipeline, touches no history.")
    parser.add_argument("--force", action="store_true",
                        help="With --run-once: re-run the latest session even if it already succeeded.")
    args = parser.parse_args(argv)

    setup_logging(config, log_prefix="scheduler")

    if args.test_telegram:
        return 0 if send_test_message(config) else 1
    if args.resend:
        status = resend_notification(config, trading_date=None if args.resend == "latest" else args.resend)
        return 0 if status == "SENT" else 1
    if args.dry_run:
        deps = Deps()
        try:
            cs.from_config(config)
        except cs.ScheduleConfigError as e:
            logger.error("Invalid scheduler configuration: %s", e)
            return 2
        outcome = check_and_run("manual", config, deps, dry_run=True)
        logger.info("-- Dry-run summary (nothing was run, sent or recorded) --")
        for line in format_status(build_status(config, deps.now(), _store(config, deps)), config):
            logger.info("  %s", line)
        logger.info("Dry-run decision: %s %s", outcome.decision, outcome.trading_date or "")
        return 0
    if args.run_once or args.force:
        outcome = check_and_run("manual", config, force=args.force)
        logger.info("Outcome: %s %s %s", outcome.decision, outcome.trading_date or "", outcome.detail)
        return _EXIT_CODES.get(outcome.decision, 0)
    try:
        return run_forever(config)
    except cs.ScheduleConfigError as e:
        logger.error("Invalid scheduler configuration: %s — scheduler not started.", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
