"""Stage I: the actual-trade ledger (SQLite).

A **separate database file** from the canonical screening history
(``data/history/stage2_history.sqlite3``): real money never shares tables — or
a file — with screening results. The only link is the plain text
``setup_run_id`` / ``setup_date`` recorded on a trade, so a fill can be traced
back to the Stage-G setup that suggested it.

Every row here comes from a fill the *user* typed in (Telegram or the CLI).
Nothing in this module — or anywhere else in this project — connects to a
broker, reads an account, or places, modifies or cancels an order.

Tables::

    setup_plans         the gate state + advisory Layer-2/Layer-3 plan for each
                        Stage-G setup of a run (never a trade)
    trades              one per position (OPEN -> CLOSED), with running
                        remaining shares, average cost and realised P&L
    trade_fills         every partial buy / sell, in order
    position_snapshots  one row per open trade per trading session (monitor)
    trade_events        audit trail: OPEN, BUY, SELL, CLOSE, CORPORATE_ACTION,
                        monitor states
    pending_actions     Telegram confirmations awaiting Confirm/Cancel — the
                        ONLY place an unconfirmed mutation lives
    corporate_actions   the auditable corporate-action ledger (splits,
                        dividends, symbol changes, review events), one row per
                        upstream event, de-duplicated by fingerprint
    corporate_action_checks  which symbols were already probed for a session,
                        so a repeated daily cycle makes no extra Yahoo calls

Accounting rules:

* ``average_cost`` is the running average cost of the shares still held. A buy
  re-averages it; a sell never changes it (realised P&L is booked instead).
* ``shares`` is what is still held; selling everything closes the trade.
* At most one OPEN trade per symbol (partial unique index), so "sell CACC"
  is never ambiguous.
* ``trade_fills`` is **immutable**. A corporate action never edits a fill: it
  restates the trade's *derived* values (shares, average cost, stops, pivot)
  and leaves the original audit trail exactly as the user reported it.
* ``dividend_income`` accumulates cash received while holding. It never touches
  ``average_cost`` or ``realized_pnl``; total P&L is realised + unrealised +
  dividend income.
"""
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

OPEN = "OPEN"
CLOSED = "CLOSED"

BUY = "BUY"
SELL = "SELL"

# trade_events.event_type
EVENT_OPEN = "OPEN"
EVENT_BUY = "BUY"
EVENT_SELL = "SELL"
EVENT_CLOSE = "CLOSE"
EVENT_MONITOR = "MONITOR_STATE"
EVENT_CORPORATE_ACTION = "CORPORATE_ACTION"

PENDING = "pending"
CONFIRMED = "confirmed"
CANCELLED = "cancelled"
EXPIRED = "expired"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS setup_plans (
    plan_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 TEXT NOT NULL,            -- the Stage-H run (screening DB) this came from
    trading_date           TEXT NOT NULL,
    symbol                 TEXT NOT NULL,
    setup_state            TEXT NOT NULL,            -- setup_gate.SETUP_STATES
    setup_reasons_json     TEXT,
    vcp_numeric_quality    TEXT,
    pivot_state            TEXT,
    entry_status           TEXT,
    trade_plan_status      TEXT,
    pivot_price            REAL,
    current_price          REAL,
    entry_trigger_price    REAL,
    maximum_chase_price    REAL,
    suggested_initial_stop REAL,
    required_risk_pct      REAL,
    allowed_max_risk_pct   REAL,
    suggested_shares       REAL,
    position_portion_pct   REAL,
    entry_plan_json        TEXT,
    risk_plan_json         TEXT,
    sizing_json            TEXT,
    created_at_utc         TEXT NOT NULL,
    UNIQUE (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS ix_plans_symbol ON setup_plans(symbol, trading_date);
CREATE INDEX IF NOT EXISTS ix_plans_date ON setup_plans(trading_date);

CREATE TABLE IF NOT EXISTS trades (
    trade_id                 TEXT PRIMARY KEY,
    symbol                   TEXT NOT NULL,
    status                   TEXT NOT NULL,            -- OPEN | CLOSED
    opened_at                TEXT NOT NULL,
    closed_at                TEXT,
    shares                   REAL NOT NULL DEFAULT 0,  -- still held
    average_cost             REAL,                     -- of the shares still held
    total_bought_shares      REAL NOT NULL DEFAULT 0,
    total_sold_shares        REAL NOT NULL DEFAULT 0,
    realized_pnl             REAL NOT NULL DEFAULT 0,
    initial_stop             REAL,
    portfolio_portion_pct    REAL,
    portfolio_value_at_entry REAL,
    setup_run_id             TEXT,
    setup_date               TEXT,
    setup_state              TEXT,
    vcp_quality              TEXT,
    pivot_price              REAL,
    entry_plan_json          TEXT,
    risk_plan_json           TEXT,
    notes                    TEXT,
    -- corporate actions (Stage I). Derived values only: fills stay immutable.
    dividend_income          REAL NOT NULL DEFAULT 0,  -- cash received while holding
    symbol_history_json      TEXT,                     -- every symbol this trade has traded under
    needs_review             INTEGER NOT NULL DEFAULT 0,
    review_reason            TEXT,
    created_at_utc           TEXT NOT NULL,
    updated_at_utc           TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_trades_open_symbol ON trades(symbol) WHERE status = 'OPEN';
CREATE INDEX IF NOT EXISTS ix_trades_symbol ON trades(symbol);

CREATE TABLE IF NOT EXISTS trade_fills (
    fill_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id       TEXT NOT NULL REFERENCES trades(trade_id),
    timestamp      TEXT NOT NULL,
    side           TEXT NOT NULL,            -- BUY | SELL
    price          REAL NOT NULL,
    shares         REAL NOT NULL,
    realized_pnl   REAL,                     -- SELL fills only
    notes          TEXT,
    created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_fills_trade ON trade_fills(trade_id, fill_id);

CREATE TABLE IF NOT EXISTS position_snapshots (
    snapshot_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id                    TEXT NOT NULL REFERENCES trades(trade_id),
    trading_date                TEXT NOT NULL,
    symbol                      TEXT NOT NULL,
    shares                      REAL,
    average_cost                REAL,
    close                       REAL,
    pnl_pct                     REAL,
    highest_since_entry         REAL,
    mfe_pct                     REAL,
    mae_pct                     REAL,
    atr20                       REAL,
    atr_pct                     REAL,
    ema10                       REAL,
    ema21                       REAL,
    sma50                       REAL,
    swing_low                   REAL,
    initial_stop                REAL,
    current_protective_level    REAL,
    distance_to_initial_stop_pct REAL,
    distance_to_protective_pct  REAL,
    days_since_entry            INTEGER,
    monitor_state               TEXT,
    price_pnl                   REAL,     -- (close - average cost) x shares held
    dividend_income             REAL,     -- cash booked on this trade so far
    total_pnl                   REAL,     -- realised + unrealised + dividend income
    reasons_json                TEXT,
    metrics_json                TEXT,
    corporate_actions_json      TEXT,     -- actions affecting this trade since entry
    created_at_utc              TEXT NOT NULL,
    UNIQUE (trade_id, trading_date)
);

CREATE INDEX IF NOT EXISTS ix_snapshots_date ON position_snapshots(trading_date);

CREATE TABLE IF NOT EXISTS trade_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id       TEXT NOT NULL REFERENCES trades(trade_id),
    timestamp      TEXT NOT NULL,
    event_type     TEXT NOT NULL,
    price          REAL,
    shares         REAL,
    reason         TEXT,
    detail         TEXT
);

CREATE INDEX IF NOT EXISTS ix_events_trade ON trade_events(trade_id, event_id);

CREATE TABLE IF NOT EXISTS pending_actions (
    token           TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,           -- buy | sell | close | position_input
    chat_id         TEXT NOT NULL,
    user_id         TEXT,
    payload_json    TEXT NOT NULL,
    status          TEXT NOT NULL,           -- pending | confirmed | cancelled | expired
    created_at_utc  TEXT NOT NULL,
    expires_at_utc  TEXT,
    resolved_at_utc TEXT,
    resolution      TEXT,
    message_id      TEXT
);

CREATE INDEX IF NOT EXISTS ix_pending_chat ON pending_actions(chat_id, status);

-- The auditable corporate-action ledger. One row per upstream event; the
-- fingerprint is the idempotency key, so re-detecting the same split or
-- dividend on every daily cycle inserts nothing and applies nothing twice.
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    effective_date  TEXT NOT NULL,            -- ex-date / effective date (YYYY-MM-DD)
    action_type     TEXT NOT NULL,            -- corporate_actions.ACTION_TYPES
    status          TEXT NOT NULL,            -- DETECTED | APPLIED | REVIEW_REQUIRED | ...
    split_ratio     REAL,                     -- new shares per old share (2.0 = 2-for-1)
    cash_amount     REAL,                     -- per share
    currency        TEXT,
    old_symbol      TEXT,
    new_symbol      TEXT,
    source          TEXT,                     -- yfinance | manual | monitor
    source_event_id TEXT,
    fingerprint     TEXT NOT NULL,
    details_json    TEXT,
    trade_id        TEXT,                     -- the journal trade it was applied to, if any
    detected_at     TEXT NOT NULL,
    applied_at      TEXT,
    notes           TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_corporate_actions_fingerprint
    ON corporate_actions(fingerprint);
CREATE INDEX IF NOT EXISTS ix_corporate_actions_symbol
    ON corporate_actions(symbol, effective_date);
CREATE INDEX IF NOT EXISTS ix_corporate_actions_status ON corporate_actions(status);
CREATE INDEX IF NOT EXISTS ix_corporate_actions_trade ON corporate_actions(trade_id);

-- One probe per symbol per session: a repeated daily cycle re-reads this
-- instead of asking Yahoo again.
CREATE TABLE IF NOT EXISTS corporate_action_checks (
    symbol         TEXT NOT NULL,
    trading_date   TEXT NOT NULL,
    checked_at_utc TEXT NOT NULL,
    PRIMARY KEY (symbol, trading_date)
);
"""

#: Columns added after SCHEMA_VERSION 1. Existing journals are migrated in
#: place (ALTER TABLE ADD COLUMN); no row is ever rewritten.
_ADDED_COLUMNS = {
    "trades": (
        ("dividend_income", "REAL NOT NULL DEFAULT 0"),
        ("symbol_history_json", "TEXT"),
        ("needs_review", "INTEGER NOT NULL DEFAULT 0"),
        ("review_reason", "TEXT"),
    ),
    "position_snapshots": (
        ("price_pnl", "REAL"),
        ("dividend_income", "REAL"),
        ("total_pnl", "REAL"),
        ("corporate_actions_json", "TEXT"),
    ),
}


class TradeError(Exception):
    """A rejected journal operation (unknown position, oversell, bad quantity)."""


def utc_now():
    return datetime.now(timezone.utc)


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("naive datetime passed to trade_store")
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _date_str(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else str(value)


def _dumps(value):
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def loads(value, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class TradeStore:
    def __init__(self, path):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                         (str(SCHEMA_VERSION),))
            conn.execute("UPDATE schema_meta SET value = ? WHERE key = 'schema_version' AND value < ?",
                         (str(SCHEMA_VERSION), str(SCHEMA_VERSION)))

    @contextmanager
    def _connect(self):
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

    # ── recording fills ───────────────────────────────────────────────────

    def record_buy(self, symbol, price, shares, *, timestamp=None, setup=None, initial_stop=None,
                   portfolio_portion_pct=None, portfolio_value=None, entry_plan=None, risk_plan=None,
                   notes=None, reason=None, event_detail=None):
        """Record a BUY fill: opens a position, or adds to the open one.

        Returns ``(trade_id, opened_new)``. Average cost is re-averaged over all
        buy fills; a second buy never resets the trade's setup linkage (or the
        plan snapshot stored when the trade was opened). ``event_detail`` is
        stored on this fill's OPEN/BUY event, e.g. the planned-vs-actual record.
        """
        symbol = str(symbol).upper()
        price, shares = _positive(price, "price"), _positive(shares, "shares")
        ts = _iso(timestamp or utc_now())
        setup = setup or {}
        now = _iso(utc_now())
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = ?",
                               (symbol, OPEN)).fetchone()
            if row is None:
                trade_id = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO trades (trade_id, symbol, status, opened_at, shares, average_cost, "
                    "total_bought_shares, initial_stop, portfolio_portion_pct, portfolio_value_at_entry, "
                    "setup_run_id, setup_date, setup_state, vcp_quality, pivot_price, entry_plan_json, "
                    "risk_plan_json, notes, created_at_utc, updated_at_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (trade_id, symbol, OPEN, ts, shares, price, shares, _num(initial_stop),
                     _num(portfolio_portion_pct), _num(portfolio_value), setup.get("setup_run_id"),
                     _date_str(setup["setup_date"]) if setup.get("setup_date") else None,
                     setup.get("setup_state"), setup.get("vcp_quality"), _num(setup.get("pivot_price")),
                     _dumps(entry_plan), _dumps(risk_plan), notes, now, now),
                )
                opened_new = True
            else:
                trade_id = row["trade_id"]
                held = float(row["shares"] or 0)
                avg = float(row["average_cost"] or 0)
                new_shares = held + shares
                new_avg = (held * avg + shares * price) / new_shares if new_shares else price
                conn.execute(
                    "UPDATE trades SET shares = ?, average_cost = ?, total_bought_shares = ?, "
                    "initial_stop = COALESCE(?, initial_stop), notes = COALESCE(?, notes), "
                    "updated_at_utc = ? WHERE trade_id = ?",
                    (new_shares, new_avg, float(row["total_bought_shares"] or 0) + shares,
                     _num(initial_stop), notes, now, trade_id),
                )
                opened_new = False

            conn.execute("INSERT INTO trade_fills (trade_id, timestamp, side, price, shares, notes, "
                         "created_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (trade_id, ts, BUY, price, shares, notes, now))
            _event(conn, trade_id, ts, EVENT_OPEN if opened_new else EVENT_BUY, price, shares,
                   reason or ("position opened" if opened_new else "added to position"),
                   _dumps(event_detail) if event_detail is not None else None)
        return trade_id, opened_new

    def record_sell(self, symbol, price, shares=None, *, timestamp=None, reason=None, notes=None):
        """Record a SELL fill. ``shares=None`` sells everything still held (a close).

        Realised P&L is booked against the running average cost; the average cost
        of the remaining shares is unchanged. Returns a summary dict.
        """
        symbol = str(symbol).upper()
        price = _positive(price, "price")
        ts = _iso(timestamp or utc_now())
        now = _iso(utc_now())
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = ?",
                               (symbol, OPEN)).fetchone()
            if row is None:
                raise TradeError(f"no open position in {symbol}")
            held = float(row["shares"] or 0)
            qty = held if shares is None else _positive(shares, "shares")
            if qty > held + 1e-9:
                raise TradeError(f"cannot sell {_fmt_qty(qty)} {symbol}: only {_fmt_qty(held)} held")
            qty = min(qty, held)
            avg = float(row["average_cost"] or 0)
            realized = (price - avg) * qty
            remaining = held - qty
            closed = remaining <= 1e-9
            conn.execute(
                "UPDATE trades SET shares = ?, total_sold_shares = ?, realized_pnl = ?, status = ?, "
                "closed_at = ?, updated_at_utc = ? WHERE trade_id = ?",
                (0.0 if closed else remaining, float(row["total_sold_shares"] or 0) + qty,
                 float(row["realized_pnl"] or 0) + realized, CLOSED if closed else OPEN,
                 ts if closed else None, now, row["trade_id"]),
            )
            conn.execute("INSERT INTO trade_fills (trade_id, timestamp, side, price, shares, realized_pnl, "
                         "notes, created_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (row["trade_id"], ts, SELL, price, qty, realized, notes, now))
            _event(conn, row["trade_id"], ts, EVENT_CLOSE if closed else EVENT_SELL, price, qty,
                   reason or ("position closed" if closed else "partial sell"))
        return {
            "trade_id": row["trade_id"], "symbol": symbol, "shares_sold": qty,
            "shares_remaining": 0.0 if closed else remaining, "average_cost": avg,
            "realized_pnl": realized, "closed": closed,
            "total_realized_pnl": float(row["realized_pnl"] or 0) + realized,
        }

    def close_position(self, symbol, price, *, timestamp=None, reason=None, notes=None):
        return self.record_sell(symbol, price, None, timestamp=timestamp,
                                reason=reason or "position closed", notes=notes)

    def update_trade(self, trade_id, **fields):
        if not fields:
            return
        fields["updated_at_utc"] = _iso(utc_now())
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._transaction() as conn:
            conn.execute(f"UPDATE trades SET {cols} WHERE trade_id = ?", (*fields.values(), trade_id))

    # ── queries ───────────────────────────────────────────────────────────

    def open_positions(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE status = ? ORDER BY symbol", (OPEN,)).fetchall()
        return [dict(r) for r in rows]

    def open_trade(self, symbol):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = ?",
                               (str(symbol).upper(), OPEN)).fetchone()
        return dict(row) if row else None

    def get_trade(self, trade_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
        return dict(row) if row else None

    def trades_for_symbol(self, symbol, limit=None):
        sql = "SELECT * FROM trades WHERE symbol = ? ORDER BY opened_at DESC"
        params = [str(symbol).upper()]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def recent_trades(self, limit=20, status=None):
        sql = "SELECT * FROM trades"
        params = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def fills(self, trade_id):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM trade_fills WHERE trade_id = ? ORDER BY fill_id", (trade_id,)).fetchall()]

    def events(self, trade_id, limit=50):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM trade_events WHERE trade_id = ? ORDER BY event_id DESC LIMIT ?",
                (trade_id, limit)).fetchall()]

    def record_event(self, trade_id, event_type, *, timestamp=None, price=None, shares=None,
                     reason=None, detail=None):
        with self._transaction() as conn:
            _event(conn, trade_id, _iso(timestamp or utc_now()), event_type, _num(price), _num(shares),
                   reason, detail)

    # ── monitor snapshots ─────────────────────────────────────────────────

    SNAPSHOT_COLUMNS = (
        "trade_id", "trading_date", "symbol", "shares", "average_cost", "close", "pnl_pct",
        "highest_since_entry", "mfe_pct", "mae_pct", "atr20", "atr_pct", "ema10", "ema21", "sma50",
        "swing_low", "initial_stop", "current_protective_level", "distance_to_initial_stop_pct",
        "distance_to_protective_pct", "days_since_entry", "monitor_state",
        "price_pnl", "dividend_income", "total_pnl",
    )

    def save_snapshot(self, snapshot):
        """Insert or replace one trade's snapshot for a trading session."""
        data = {k: snapshot.get(k) for k in self.SNAPSHOT_COLUMNS}
        data["trading_date"] = _date_str(data["trading_date"])
        data["reasons_json"] = _dumps(snapshot.get("reasons") or [])
        data["metrics_json"] = _dumps(snapshot.get("metrics") or {})
        data["corporate_actions_json"] = _dumps(snapshot.get("corporate_actions") or [])
        data["created_at_utc"] = _iso(utc_now())
        cols = list(data)
        with self._transaction() as conn:
            conn.execute(
                f"INSERT INTO position_snapshots ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)}) "
                f"ON CONFLICT(trade_id, trading_date) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("trade_id", "trading_date")),
                tuple(data[c] for c in cols),
            )

    def snapshots_for_date(self, trading_date):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM position_snapshots WHERE trading_date = ? ORDER BY symbol",
                                (_date_str(trading_date),)).fetchall()
        return [dict(r) for r in rows]

    def latest_snapshot(self, trade_id, before=None):
        sql = "SELECT * FROM position_snapshots WHERE trade_id = ?"
        params = [trade_id]
        if before is not None:
            sql += " AND trading_date < ?"
            params.append(_date_str(before))
        sql += " ORDER BY trading_date DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def snapshots_for_trade(self, trade_id, limit=30):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM position_snapshots WHERE trade_id = ? "
                                "ORDER BY trading_date DESC LIMIT ?", (trade_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ── setup plans (Layer 2 / Layer 3, advisory) ─────────────────────────

    PLAN_COLUMNS = (
        "run_id", "trading_date", "symbol", "setup_state", "vcp_numeric_quality", "pivot_state",
        "entry_status", "trade_plan_status", "pivot_price", "current_price", "entry_trigger_price",
        "maximum_chase_price", "suggested_initial_stop", "required_risk_pct", "allowed_max_risk_pct",
        "suggested_shares", "position_portion_pct",
    )

    def save_setup_plan(self, plan):
        """Insert or replace the gate state (+ plan, when one was generated) for
        one symbol of one run."""
        data = {k: plan.get(k) for k in self.PLAN_COLUMNS}
        data["trading_date"] = _date_str(data["trading_date"])
        data["symbol"] = str(data["symbol"]).upper()
        data["setup_reasons_json"] = _dumps(plan.get("setup_reasons") or [])
        data["entry_plan_json"] = _dumps(plan.get("entry_plan"))
        data["risk_plan_json"] = _dumps(plan.get("risk_plan"))
        data["sizing_json"] = _dumps(plan.get("sizing"))
        data["created_at_utc"] = _iso(utc_now())
        cols = list(data)
        with self._transaction() as conn:
            conn.execute(
                f"INSERT INTO setup_plans ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
                f"ON CONFLICT(run_id, symbol) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in cols if c not in ("run_id", "symbol")),
                tuple(data[c] for c in cols),
            )

    def setup_plans_for_date(self, trading_date, states=None):
        sql = "SELECT * FROM setup_plans WHERE trading_date = ?"
        params = [_date_str(trading_date)]
        if states:
            sql += f" AND setup_state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        sql += " ORDER BY symbol"
        with self._connect() as conn:
            return [_plan_row(r) for r in conn.execute(sql, params).fetchall()]

    def latest_plan_date(self):
        """The most recent session for which setup plans were recorded."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(trading_date) AS d FROM setup_plans").fetchone()
        return row["d"] if row and row["d"] else None

    def latest_setup_plan(self, symbol, states=None):
        """The most recent recorded gate state / plan for a symbol."""
        sql = "SELECT * FROM setup_plans WHERE symbol = ?"
        params = [str(symbol).upper()]
        if states:
            sql += f" AND setup_state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        sql += " ORDER BY trading_date DESC, plan_id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return _plan_row(row) if row else None

    # ── corporate actions (auditable ledger, §2) ──────────────────────────
    # Every write here is idempotent: an action is inserted once (unique
    # fingerprint) and applied once (a conditional status UPDATE inside the
    # same transaction that mutates the trade). Fills are never touched.

    ACTION_COLUMNS = ("symbol", "effective_date", "action_type", "status", "split_ratio",
                      "cash_amount", "currency", "old_symbol", "new_symbol", "source",
                      "source_event_id", "fingerprint", "details_json", "trade_id", "notes")

    def record_corporate_action(self, action):
        """Insert one detected action. Returns ``(row, created)``.

        A second detection of the same event finds the existing fingerprint and
        creates nothing — the stored row (with whatever status it has reached)
        is returned instead.
        """
        data = {k: action.get(k) for k in self.ACTION_COLUMNS}
        data["symbol"] = str(data["symbol"]).upper()
        data["effective_date"] = _date_str(data["effective_date"])
        if data.get("details_json") is None:
            data["details_json"] = _dumps(action.get("details") or {})
        data["detected_at"] = _iso(action.get("detected_at") or utc_now())
        cols = list(data)
        placeholders = ", ".join("?" for _ in cols)
        with self._transaction() as conn:
            changed = conn.execute(
                "INSERT OR IGNORE INTO corporate_actions (%s) VALUES (%s)"
                % (", ".join(cols), placeholders), tuple(data[c] for c in cols)).rowcount
            row = conn.execute("SELECT * FROM corporate_actions WHERE fingerprint = ?",
                               (data["fingerprint"],)).fetchone()
        return _action_row(row), bool(changed)

    def get_corporate_action(self, action_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM corporate_actions WHERE action_id = ?",
                               (action_id,)).fetchone()
        return _action_row(row) if row else None

    def corporate_actions(self, symbol=None, *, trade_id=None, statuses=None, action_types=None,
                          since=None, limit=None):
        sql = "SELECT * FROM corporate_actions WHERE 1 = 1"
        params = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(str(symbol).upper())
        if trade_id:
            sql += " AND trade_id = ?"
            params.append(trade_id)
        if statuses:
            sql += " AND status IN (%s)" % ", ".join("?" for _ in statuses)
            params.extend(statuses)
        if action_types:
            sql += " AND action_type IN (%s)" % ", ".join("?" for _ in action_types)
            params.extend(action_types)
        if since:
            sql += " AND effective_date >= ?"
            params.append(_date_str(since))
        sql += " ORDER BY effective_date, action_id"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            return [_action_row(r) for r in conn.execute(sql, params).fetchall()]

    def known_action_fingerprints(self, symbol):
        with self._connect() as conn:
            rows = conn.execute("SELECT fingerprint FROM corporate_actions WHERE symbol = ?",
                                (str(symbol).upper(),)).fetchall()
        return {r["fingerprint"] for r in rows}

    def set_action_status(self, action_id, status, *, expected_status=None, notes=None,
                          applied_at=None, trade_id=None):
        """Conditional status change. False means somebody got there first."""
        sets = ["status = ?"]
        params = [status]
        if notes is not None:
            sets.append("notes = ?")
            params.append(notes)
        if applied_at is not None:
            sets.append("applied_at = ?")
            params.append(_iso(applied_at))
        if trade_id is not None:
            sets.append("trade_id = ?")
            params.append(trade_id)
        sql = "UPDATE corporate_actions SET %s WHERE action_id = ?" % ", ".join(sets)
        params.append(action_id)
        if expected_status is not None:
            sql += " AND status = ?"
            params.append(expected_status)
        with self._transaction() as conn:
            return bool(conn.execute(sql, params).rowcount)

    def apply_action_to_trade(self, action_id, trade_id, *, expected_status, trade_fields=None,
                              dividend_income_delta=None, symbol_history=None, event_reason=None,
                              event_detail=None, notes=None, timestamp=None, status="APPLIED"):
        """Apply one corporate action to one trade, atomically and exactly once.

        The status flip and the trade mutation share a transaction, and the flip
        is conditional on ``expected_status`` — so a duplicate apply (a retried
        cycle, two processes) changes nothing at all. ``trade_fields`` are the
        already-computed derived values; this method does no arithmetic of its
        own and never edits ``trade_fills``.
        """
        ts = _iso(timestamp or utc_now())
        with self._transaction() as conn:
            claimed = conn.execute(
                "UPDATE corporate_actions SET status = ?, applied_at = ?, trade_id = ?, "
                "notes = COALESCE(?, notes) WHERE action_id = ? AND status = ?",
                (status, ts, trade_id, notes, action_id, expected_status)).rowcount
            if not claimed:
                return False
            fields = dict(trade_fields or {})
            if symbol_history is not None:
                fields["symbol_history_json"] = _dumps(symbol_history)
            if fields:
                fields["updated_at_utc"] = ts
                cols = ", ".join("%s = ?" % k for k in fields)
                conn.execute("UPDATE trades SET %s WHERE trade_id = ?" % cols,
                             (*fields.values(), trade_id))
            if dividend_income_delta:
                conn.execute("UPDATE trades SET dividend_income = COALESCE(dividend_income, 0) + ?, "
                             "updated_at_utc = ? WHERE trade_id = ?",
                             (float(dividend_income_delta), ts, trade_id))
            _event(conn, trade_id, ts, EVENT_CORPORATE_ACTION, None, fields.get("shares"),
                   event_reason, event_detail)
        return True

    def set_trade_review(self, trade_id, reason, *, timestamp=None):
        """Hold a position in review. The monitor then suppresses mechanical exit
        conclusions for it until the user resolves it."""
        with self._transaction() as conn:
            conn.execute("UPDATE trades SET needs_review = 1, review_reason = ?, updated_at_utc = ? "
                         "WHERE trade_id = ?", (reason, _iso(timestamp or utc_now()), trade_id))

    def clear_trade_review(self, trade_id, *, timestamp=None):
        with self._transaction() as conn:
            conn.execute("UPDATE trades SET needs_review = 0, review_reason = NULL, "
                         "updated_at_utc = ? WHERE trade_id = ?",
                         (_iso(timestamp or utc_now()), trade_id))

    def mark_symbol_checked(self, symbol, trading_date, *, timestamp=None):
        with self._transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO corporate_action_checks "
                         "(symbol, trading_date, checked_at_utc) VALUES (?, ?, ?)",
                         (str(symbol).upper(), _date_str(trading_date),
                          _iso(timestamp or utc_now())))

    def symbol_checked(self, symbol, trading_date):
        """True when this symbol was already probed for this session — the guard
        that keeps a repeated daily cycle from making the same Yahoo calls."""
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM corporate_action_checks WHERE symbol = ? "
                               "AND trading_date = ?",
                               (str(symbol).upper(), _date_str(trading_date))).fetchone()
        return row is not None

    # ── pending Telegram confirmations ────────────────────────────────────

    def create_pending(self, kind, chat_id, payload, *, user_id=None, expires_at=None, token=None):
        token = token or uuid.uuid4().hex[:12]
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO pending_actions (token, kind, chat_id, user_id, payload_json, status, "
                "created_at_utc, expires_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (token, kind, str(chat_id), None if user_id is None else str(user_id), _dumps(payload),
                 PENDING, _iso(utc_now()), _iso(expires_at)),
            )
        return token

    def get_pending(self, token):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pending_actions WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = loads(out.get("payload_json"), {})
        return out

    def claim_pending(self, token, resolution, *, now=None):
        """Atomically move a pending action out of ``pending``.

        Returns the claimed action, or None if it was already resolved (the
        duplicate-callback guard: a second Confirm tap writes nothing).
        """
        now = now or utc_now()
        with self._transaction() as conn:
            changed = conn.execute(
                "UPDATE pending_actions SET status = ?, resolution = ?, resolved_at_utc = ? "
                "WHERE token = ? AND status = ?",
                (resolution, resolution, _iso(now), token, PENDING),
            ).rowcount
            if not changed:
                return None
            row = conn.execute("SELECT * FROM pending_actions WHERE token = ?", (token,)).fetchone()
        out = dict(row)
        out["payload"] = loads(out.get("payload_json"), {})
        return out

    def set_pending_message_id(self, token, message_id):
        with self._transaction() as conn:
            conn.execute("UPDATE pending_actions SET message_id = ? WHERE token = ?",
                         (None if message_id is None else str(message_id), token))

    def pending_for_chat(self, chat_id, kind=None):
        sql = "SELECT * FROM pending_actions WHERE chat_id = ? AND status = ?"
        params = [str(chat_id), PENDING]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at_utc DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["payload"] = loads(item.get("payload_json"), {})
            out.append(item)
        return out

    def expire_pending(self, now=None):
        """Mark timed-out confirmations expired. Returns the number expired."""
        now = _iso(now or utc_now())
        with self._transaction() as conn:
            return conn.execute(
                "UPDATE pending_actions SET status = ?, resolution = ?, resolved_at_utc = ? "
                "WHERE status = ? AND expires_at_utc IS NOT NULL AND expires_at_utc < ?",
                (EXPIRED, EXPIRED, now, PENDING, now),
            ).rowcount


def _action_row(row):
    if row is None:
        return None
    out = dict(row)
    out["details"] = loads(out.get("details_json"), {})
    return out


def _migrate(conn):
    """Add post-v1 columns to an existing journal. Purely additive: no row is
    rewritten and no historical meaning changes."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(%s)" % table).fetchall()}
        for name, ddl in columns:
            if name not in existing:
                conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, name, ddl))
                logger.info("Trade journal migrated: %s.%s added", table, name)


def _plan_row(row):
    out = dict(row)
    out["setup_reasons"] = loads(out.get("setup_reasons_json"), [])
    out["entry_plan"] = loads(out.get("entry_plan_json"))
    out["risk_plan"] = loads(out.get("risk_plan_json"))
    out["sizing"] = loads(out.get("sizing_json"))
    return out


def _event(conn, trade_id, ts, event_type, price, shares, reason, detail=None):
    conn.execute("INSERT INTO trade_events (trade_id, timestamp, event_type, price, shares, reason, detail) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)", (trade_id, ts, event_type, price, shares, reason, detail))


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TradeError(f"{label} must be a number, got {value!r}")
    if number <= 0:
        raise TradeError(f"{label} must be greater than zero, got {value!r}")
    return number


def _fmt_qty(value):
    return f"{value:g}"


def open_store(config, create=True):
    """Open the trade journal. With ``create=False`` returns None when the
    database file does not exist yet (nothing has ever been recorded)."""
    path = config["trade_db_path"]
    if not create and not os.path.exists(path):
        return None
    return TradeStore(path)
