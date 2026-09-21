"""SQLite canonical history (Stage H).

SQLite is the longitudinal source of truth; CSV/PNG/JSON artifacts remain the
human-readable per-run outputs.

Idempotency is enforced by the schema, not just by application checks:

* ``ux_runs_canonical`` — at most one canonical (successful) run per
  ``trading_date``. A run only becomes canonical inside the same transaction
  that writes its complete ``stock_results``; a failure anywhere in that
  transaction rolls back to "no canonical result", never to a partial one.
* ``ux_runs_active`` — at most one WAITING_FOR_DATA/RUNNING claim per
  ``trading_date``, so a second process cannot start duplicate expensive work
  even if the file lock is bypassed.

Run status model::

    WAITING_FOR_DATA -> RUNNING -> RESULTS_PERSISTED -> NOTIFIED (or NOTIFICATION_FAILED)
    not run (session stays pending, retried later): DATA_NOT_READY, DEFERRED_MARKET_OPEN
    failure/terminal: PIPELINE_FAILED, ABANDONED

A run is "successful" iff ``is_canonical = 1``; notification outcome never
changes that.
"""
import json
import logging
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Status values
WAITING_FOR_DATA = "WAITING_FOR_DATA"
RUNNING = "RUNNING"
RESULTS_PERSISTED = "RESULTS_PERSISTED"
NOTIFIED = "NOTIFIED"
NOTIFICATION_FAILED = "NOTIFICATION_FAILED"
DATA_NOT_READY = "DATA_NOT_READY"
DEFERRED_MARKET_OPEN = "DEFERRED_MARKET_OPEN"
PIPELINE_FAILED = "PIPELINE_FAILED"
ABANDONED = "ABANDONED"

ACTIVE_STATUSES = (WAITING_FOR_DATA, RUNNING)
SUCCESS_STATUSES = (RESULTS_PERSISTED, NOTIFIED, NOTIFICATION_FAILED)

# Stage-count columns persisted on pipeline_runs (all also kept in stage_counts_json).
STAGE_COUNT_COLUMNS = (
    "universe", "liquidity", "market_cap", "sector_classified", "stage_c",
    "stage_d", "trend_template_pass", "fundamentals_scored", "ranked",
    "charts", "vcp_analyzed", "vcp_pattern", "vcp_actionable",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id                    TEXT PRIMARY KEY,
    trading_date              TEXT NOT NULL,          -- NASDAQ session, YYYY-MM-DD
    trigger                   TEXT NOT NULL,          -- startup | scheduled | manual
    forced                    INTEGER NOT NULL DEFAULT 0,
    status                    TEXT NOT NULL,
    is_canonical              INTEGER NOT NULL DEFAULT 0,
    superseded_by_run_id      TEXT,
    started_at_utc            TEXT NOT NULL,
    pipeline_started_at_utc   TEXT,
    completed_at_utc          TEXT,
    local_execution_timezone  TEXT,
    local_started_at          TEXT,
    run_timestamp             TEXT,                   -- names data/charts/<ts>/ and the run log
    data_ready_latest_bar     TEXT,
    data_ready_attempts       INTEGER,
    git_commit                TEXT,
    git_dirty                 INTEGER,
    config_hash               TEXT,
    config_json               TEXT,                   -- secrets excluded
    environment_json          TEXT,
    {", ".join(f"count_{c} INTEGER" for c in STAGE_COUNT_COLUMNS)},
    stage_counts_json         TEXT,
    artifacts_json            TEXT,
    failed_stage              TEXT,
    error_summary             TEXT,
    created_at_utc            TEXT NOT NULL,
    updated_at_utc            TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_canonical
    ON pipeline_runs(trading_date) WHERE is_canonical = 1;
CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_active
    ON pipeline_runs(trading_date) WHERE status IN ('WAITING_FOR_DATA', 'RUNNING');
CREATE INDEX IF NOT EXISTS ix_runs_trading_date ON pipeline_runs(trading_date);

CREATE TABLE IF NOT EXISTS stock_results (
    run_id                    TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    trading_date              TEXT NOT NULL,
    symbol                    TEXT NOT NULL,
    final_rank                INTEGER NOT NULL,       -- 1-based position in the ranked watchlist
    composite_score           REAL,
    max_score                 REAL,
    technical_score           REAL,
    fundamentals_score        REAL,
    sector                    TEXT,
    industry                  TEXT,
    market_cap                REAL,
    last_close                REAL,                   -- reference close for outcome tracking
    rs_3mo                    REAL,
    rs_6mo                    REAL,
    vcp_status                TEXT,                   -- analyzed | skipped | error | NULL (not analysed)
    vcp_is_pattern            INTEGER,
    vcp_pattern_stage         TEXT,
    vcp_confidence            TEXT,
    vcp_entry_recommendation  TEXT,
    vcp_pivot_price           REAL,
    vcp_stop_loss             REAL,
    vcp_metrics_json          TEXT,
    vcp_verdict_json          TEXT,
    chart_path                TEXT,
    vcp_debug_chart_path      TEXT,
    ranking_json              TEXT,                   -- the complete Stage E row
    PRIMARY KEY (run_id, symbol),
    UNIQUE (run_id, final_rank)
);

CREATE INDEX IF NOT EXISTS ix_results_symbol_date ON stock_results(symbol, trading_date);
CREATE INDEX IF NOT EXISTS ix_results_date_rank ON stock_results(trading_date, final_rank);

CREATE TABLE IF NOT EXISTS notification_delivery (
    notification_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                    TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    channel                   TEXT NOT NULL,          -- telegram
    kind                      TEXT NOT NULL,          -- report | chart | ops_alert
    attempted_at_utc          TEXT NOT NULL,
    status                    TEXT NOT NULL,          -- SENT | FAILED | SKIPPED
    attempts                  INTEGER,
    error_summary             TEXT,
    provider_message_id       TEXT,
    detail                    TEXT
);

CREATE INDEX IF NOT EXISTS ix_notification_run ON notification_delivery(run_id);
"""

# Stage-G VCP verdict fields (VCP_SCHEMA) mapped onto first-class columns.
_VCP_VERDICT_KEYS = (
    "is_vcp_pattern", "pattern_stage", "contraction_count_observed", "volume_dry_up_confirmed",
    "pivot_price", "suggested_stop_loss", "confidence", "entry_recommendation", "rationale",
)
# Every other key of a Stage-G row is a numeric metric / detector assessment and
# goes into vcp_metrics_json as-is. The metric set evolves (see
# stage_vcp_analysis.VCP_METRICS_VERSION), so it is not whitelisted: older rows
# simply carry fewer keys and never need migrating.
_VCP_NON_METRIC_KEYS = (frozenset(_VCP_VERDICT_KEYS)
                        | {"Symbol", "skipped", "error", "detail", "stop_reason", "llm_review"})


# Columns needed for trend analysis (skips the bulky JSON payloads).
_TREND_COLUMNS = ", ".join(f"r.{c}" for c in (
    "run_id", "trading_date", "symbol", "final_rank", "composite_score", "max_score",
    "technical_score", "fundamentals_score", "sector", "last_close", "vcp_status",
    "vcp_is_pattern", "vcp_pattern_stage", "vcp_confidence", "vcp_entry_recommendation",
    "vcp_pivot_price", "chart_path",
))


class AlreadyCompleted(Exception):
    """A canonical result already exists for the trading date."""


class SessionInProgress(Exception):
    """Another active claim exists for the trading date."""


def utc_now():
    return datetime.now(timezone.utc)


def _iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            raise ValueError("naive datetime passed to history_store")
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def _date_str(d):
    return d.isoformat() if isinstance(d, (date, datetime)) else str(d)


def to_jsonable(value):
    """NaN/NaT -> None, numpy/pandas scalars -> Python, recursively."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _dumps(value):
    return json.dumps(to_jsonable(value), sort_keys=True)


def _num(value):
    v = to_jsonable(value)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class HistoryStore:
    def __init__(self, path):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                         (str(SCHEMA_VERSION),))

    @contextmanager
    def _connect(self):
        # Default rollback journal (not WAL): safer on Docker bind mounts.
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self):
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # ── run lifecycle ─────────────────────────────────────────────────────

    def claim_session(self, trading_date, trigger, *, forced=False, local_timezone=None,
                      local_started_at=None, stale_after_hours=12, now=None):
        """Create a WAITING_FOR_DATA run for ``trading_date``.

        Raises AlreadyCompleted (unless ``forced``) or SessionInProgress.
        Stale active claims (a crashed process) are marked ABANDONED first.
        """
        now = now or utc_now()
        td = _date_str(trading_date)
        run_id = uuid.uuid4().hex
        with self._transaction() as conn:
            if not forced and self._canonical_row(conn, td) is not None:
                raise AlreadyCompleted(td)

            stale_cutoff = _iso(now - timedelta(hours=stale_after_hours))
            abandoned = conn.execute(
                "UPDATE pipeline_runs SET status = ?, error_summary = COALESCE(error_summary, ?), "
                "updated_at_utc = ? WHERE trading_date = ? AND status IN (?, ?) AND started_at_utc < ?",
                (ABANDONED, "claim went stale (process presumed crashed)", _iso(now), td,
                 *ACTIVE_STATUSES, stale_cutoff),
            ).rowcount
            if abandoned:
                logger.warning("Marked %d stale active run(s) for %s as ABANDONED.", abandoned, td)

            try:
                conn.execute(
                    "INSERT INTO pipeline_runs (run_id, trading_date, trigger, forced, status, "
                    "started_at_utc, local_execution_timezone, local_started_at, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, td, trigger, int(bool(forced)), WAITING_FOR_DATA, _iso(now),
                     local_timezone, local_started_at.isoformat() if local_started_at else None,
                     _iso(now), _iso(now)),
                )
            except sqlite3.IntegrityError:
                raise SessionInProgress(td)
        return run_id

    def record_deferral(self, trading_date, trigger, status, reason, *, local_timezone=None,
                        local_started_at=None, now=None):
        """Audit row for a check that deliberately did not run (never active, never
        canonical), so the session stays pending and a later check reconsiders it."""
        now = now or utc_now()
        run_id = uuid.uuid4().hex
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, trading_date, trigger, status, started_at_utc, "
                "completed_at_utc, local_execution_timezone, local_started_at, error_summary, "
                "created_at_utc, updated_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, _date_str(trading_date), trigger, status, _iso(now), _iso(now), local_timezone,
                 local_started_at.isoformat() if local_started_at else None, reason, _iso(now), _iso(now)),
            )
        return run_id

    def update_run(self, run_id, **fields):
        if not fields:
            return
        fields["updated_at_utc"] = _iso(utc_now())
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [_iso(v) if isinstance(v, datetime) else v for v in fields.values()]
        with self._transaction() as conn:
            conn.execute(f"UPDATE pipeline_runs SET {cols} WHERE run_id = ?", (*values, run_id))

    def mark_running(self, run_id, *, latest_bar, attempts, run_timestamp, metadata):
        self.update_run(
            run_id, status=RUNNING, pipeline_started_at_utc=utc_now(),
            data_ready_latest_bar=_date_str(latest_bar) if latest_bar else None,
            data_ready_attempts=attempts, run_timestamp=run_timestamp,
            git_commit=metadata.get("git_commit"), git_dirty=metadata.get("git_dirty"),
            config_hash=metadata.get("config_hash"), config_json=metadata.get("config_json"),
            environment_json=metadata.get("environment_json"),
        )

    def mark_failed(self, run_id, status, *, error_summary=None, failed_stage=None,
                    latest_bar=None, attempts=None, stage_counts=None):
        fields = dict(status=status, completed_at_utc=utc_now(),
                      error_summary=error_summary, failed_stage=failed_stage)
        if latest_bar is not None:
            fields["data_ready_latest_bar"] = _date_str(latest_bar)
        if attempts is not None:
            fields["data_ready_attempts"] = attempts
        if stage_counts:
            fields.update(_count_fields(stage_counts))
        self.update_run(run_id, **fields)

    def persist_results(self, run_id, trading_date, ranked_df, vcp_df=None, *, chart_paths=None,
                        debug_chart_dir=None, stage_counts=None, artifacts=None, forced=False):
        """Atomically write every ranked row and make ``run_id`` canonical.

        Either all rows + canonical status are committed, or nothing is.
        """
        td = _date_str(trading_date)
        rows = build_result_rows(run_id, td, ranked_df, vcp_df, chart_paths or {}, debug_chart_dir)
        now = _iso(utc_now())
        with self._transaction() as conn:
            run = conn.execute("SELECT status, trading_date FROM pipeline_runs WHERE run_id = ?",
                               (run_id,)).fetchone()
            if run is None or run["status"] != RUNNING or run["trading_date"] != td:
                raise RuntimeError(f"run {run_id} is not RUNNING for {td}")

            existing = self._canonical_row(conn, td)
            if existing is not None and existing["run_id"] != run_id:
                if not forced:
                    raise AlreadyCompleted(td)
                conn.execute("UPDATE pipeline_runs SET is_canonical = 0, superseded_by_run_id = ?, "
                             "updated_at_utc = ? WHERE run_id = ?", (run_id, now, existing["run_id"]))
                logger.info("Force rerun: run %s supersedes %s for %s.", run_id, existing["run_id"], td)

            if rows:
                cols = list(rows[0].keys())
                conn.executemany(
                    f"INSERT INTO stock_results ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    [tuple(r[c] for c in cols) for r in rows],
                )

            fields = {
                "status": RESULTS_PERSISTED, "is_canonical": 1, "completed_at_utc": now,
                "stage_counts_json": _dumps(stage_counts or {}),
                "artifacts_json": _dumps(artifacts or {}),
                "updated_at_utc": now,
                **_count_fields(stage_counts or {}),
            }
            conn.execute(f"UPDATE pipeline_runs SET {', '.join(f'{k} = ?' for k in fields)} WHERE run_id = ?",
                         (*fields.values(), run_id))
        return len(rows)

    def record_notification(self, run_id, kind, status, *, attempts=None, error_summary=None,
                            provider_message_id=None, detail=None, channel="telegram"):
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO notification_delivery (run_id, channel, kind, attempted_at_utc, status, "
                "attempts, error_summary, provider_message_id, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, channel, kind, _iso(utc_now()), status, attempts, error_summary,
                 provider_message_id, detail),
            )

    def set_notification_status(self, run_id, status):
        """Only moves a successful run between RESULTS_PERSISTED/NOTIFIED/NOTIFICATION_FAILED."""
        if status not in SUCCESS_STATUSES:
            raise ValueError(status)
        with self._transaction() as conn:
            conn.execute("UPDATE pipeline_runs SET status = ?, updated_at_utc = ? "
                         "WHERE run_id = ? AND is_canonical = 1", (status, _iso(utc_now()), run_id))

    # ── queries ───────────────────────────────────────────────────────────

    @staticmethod
    def _canonical_row(conn, td):
        return conn.execute("SELECT * FROM pipeline_runs WHERE trading_date = ? AND is_canonical = 1",
                            (td,)).fetchone()

    def has_successful_run(self, trading_date):
        return self.canonical_run(trading_date) is not None

    def canonical_run(self, trading_date):
        with self._connect() as conn:
            row = self._canonical_row(conn, _date_str(trading_date))
        return dict(row) if row else None

    def get_run(self, run_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def recent_runs(self, limit=20):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pipeline_runs ORDER BY started_at_utc DESC LIMIT ?",
                                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def latest_canonical_run(self):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pipeline_runs WHERE is_canonical = 1 "
                               "ORDER BY trading_date DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def canonical_sessions(self, up_to=None):
        """Trading dates with a canonical result, ascending (optionally <= up_to)."""
        sql = "SELECT trading_date FROM pipeline_runs WHERE is_canonical = 1"
        params = ()
        if up_to is not None:
            sql += " AND trading_date <= ?"
            params = (_date_str(up_to),)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY trading_date", params).fetchall()
        return [date.fromisoformat(r["trading_date"]) for r in rows]

    def results_for_sessions(self, sessions):
        """Canonical stock_results rows for the given trading dates."""
        dates = [_date_str(s) for s in sessions]
        if not dates:
            return []
        placeholders = ", ".join("?" for _ in dates)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_TREND_COLUMNS} FROM stock_results r JOIN pipeline_runs p ON p.run_id = r.run_id "
                f"WHERE p.is_canonical = 1 AND r.trading_date IN ({placeholders}) "
                f"ORDER BY r.trading_date, r.final_rank", dates,
            ).fetchall()
        return [dict(r) for r in rows]

    def results_for_run(self, run_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM stock_results WHERE run_id = ? ORDER BY final_rank",
                                (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def symbol_history(self, symbol, limit=None):
        sql = ("SELECT r.* FROM stock_results r JOIN pipeline_runs p ON p.run_id = r.run_id "
               "WHERE p.is_canonical = 1 AND r.symbol = ? ORDER BY r.trading_date DESC")
        params = [symbol.upper()]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def notifications_for_run(self, run_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notification_delivery WHERE run_id = ? "
                                "ORDER BY notification_id", (run_id,)).fetchall()
        return [dict(r) for r in rows]


def _count_fields(stage_counts):
    return {f"count_{k}": int(stage_counts[k]) for k in STAGE_COUNT_COLUMNS
            if stage_counts.get(k) is not None}


def build_result_rows(run_id, trading_date, ranked_df, vcp_df, chart_paths, debug_chart_dir):
    """One row per ranked symbol; VCP columns filled only where Stage G produced output."""
    if ranked_df is None or ranked_df.empty:
        return []
    vcp_by_symbol = {}
    if vcp_df is not None and not vcp_df.empty and "Symbol" in vcp_df.columns:
        for rec in vcp_df.to_dict("records"):
            vcp_by_symbol[rec["Symbol"]] = rec

    rows = []
    for rank, rec in enumerate(ranked_df.to_dict("records"), start=1):
        symbol = rec["Symbol"]
        vcp = vcp_by_symbol.get(symbol)
        verdict = {k: vcp.get(k) for k in _VCP_VERDICT_KEYS if k in vcp} if vcp else {}
        metrics = {k: v for k, v in vcp.items() if k not in _VCP_NON_METRIC_KEYS} if vcp else {}
        if vcp is None:
            vcp_status = None
        elif to_jsonable(vcp.get("skipped")):
            vcp_status = "skipped"
            verdict["skipped"] = vcp.get("skipped")
        elif to_jsonable(vcp.get("error")):
            vcp_status = "error"
            verdict["error"] = str(vcp.get("error"))[:500]
            if to_jsonable(vcp.get("detail")):
                verdict["error_detail"] = str(vcp.get("detail"))[:1500]   # e.g. the contradictory raw verdict
        else:
            vcp_status = "analyzed"
        # Stage-G review routing (mode / requested / reasons / agreement) belongs with
        # the Claude side, never with the numeric metrics.
        if vcp is not None and isinstance(vcp.get("llm_review"), dict):
            verdict["review"] = vcp["llm_review"]

        is_pattern = to_jsonable(verdict.get("is_vcp_pattern"))
        debug_path = None
        if vcp is not None and debug_chart_dir:
            candidate = os.path.join(debug_chart_dir, f"{symbol}.png")
            debug_path = candidate if os.path.exists(candidate) else None

        rows.append({
            "run_id": run_id,
            "trading_date": trading_date,
            "symbol": symbol,
            "final_rank": rank,
            "composite_score": _num(rec.get("Composite Score")),
            "max_score": _num(rec.get("Max Score")),
            "technical_score": _num(rec.get("Technical Score")),
            "fundamentals_score": _num(rec.get("Fundamentals Score")),
            "sector": to_jsonable(rec.get("Sector")),
            "industry": to_jsonable(rec.get("Industry")),
            "market_cap": _num(rec.get("Market Cap")),
            "last_close": _num(rec.get("Last Close")),
            "rs_3mo": _num(rec.get("RS vs NASDAQ (3mo)")),
            "rs_6mo": _num(rec.get("RS vs NASDAQ (6mo)")),
            "vcp_status": vcp_status,
            "vcp_is_pattern": None if is_pattern is None else int(bool(is_pattern)),
            "vcp_pattern_stage": to_jsonable(verdict.get("pattern_stage")),
            "vcp_confidence": to_jsonable(verdict.get("confidence")),
            "vcp_entry_recommendation": to_jsonable(verdict.get("entry_recommendation")),
            "vcp_pivot_price": _num(verdict.get("pivot_price")),
            "vcp_stop_loss": _num(verdict.get("suggested_stop_loss")),
            "vcp_metrics_json": _dumps(metrics) if vcp is not None else None,
            "vcp_verdict_json": _dumps(verdict) if vcp is not None else None,
            "chart_path": chart_paths.get(symbol),
            "vcp_debug_chart_path": debug_path,
            "ranking_json": _dumps(rec),
        })
    return rows
