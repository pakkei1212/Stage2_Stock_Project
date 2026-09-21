"""Corporate actions inside the daily cycle: monitor, scheduler order, reporting.

The point of these tests is that a corporate action must never *look* like a
trading signal. A split must not fire a stop, an ex-dividend must not read as a
failed breakout, and a symbol that stopped reporting prices must not be treated
as a sale. Everything is synthetic and injected — no network, no broker.
"""
import json
from datetime import date

import pytest

from pipeline import corporate_actions as ca
from pipeline import position_monitor as pm
from pipeline import scheduler as sch
from pipeline import telegram_notifier as tg
from pipeline import trade_store as ts
from stage_h_fakes import FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt
from stage_i_fakes import actions_source, make_bars, trade_config

SESSION = "2026-09-15"
TUE = date(2026, 9, 15)


@pytest.fixture
def store(tmp_path):
    return ts.TradeStore(str(tmp_path / "trades" / "journal.sqlite3"))


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


def open_position(store, symbol="CACC", price=100.0, shares=100, stop=92.0, pivot=105.0,
                  timestamp="2026-08-03T13:30:00+00:00"):
    trade_id, _ = store.record_buy(symbol, price, shares, timestamp=timestamp, initial_stop=stop,
                                   setup={"pivot_price": pivot})
    return store.get_trade(trade_id)


# ── §3: a split must not fire a fictional stop ────────────────────────────

def test_a_split_does_not_fire_a_stop(store, config):
    """Bought at 100 with a 92 stop; after a 2-for-1 the price is ~51 and the
    journal is on the new basis, so this is an ordinary winner, not a stop-out."""
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    bars = make_bars([50 + i * 0.05 for i in range(60)])            # post-split prices
    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, bars, SESSION, actions=ca.actions_for_trade(store, trade),
                                 config=config)

    assert snapshot["monitor_state"] != pm.STOP_TRIGGERED
    assert snapshot["pnl_pct"] > 0
    assert snapshot["initial_stop"] == pytest.approx(46.0)


def test_a_protective_level_recorded_before_a_split_is_rebased_not_carried(store, config):
    """The trap this guards: yesterday's 96.0 level against today's ~51 close
    would read as an instant stop-out on a stock that has not moved at all."""
    trade = open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    previous = {"trading_date": "2026-09-09", "current_protective_level": 96.0,
                "monitor_state": pm.HOLD}
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    trade = store.open_trade("CACC")
    actions = ca.actions_for_trade(store, trade)
    snapshot = pm.build_snapshot(trade, make_bars([51.0] * 60), SESSION, previous_snapshot=previous,
                                 actions=actions, config=config)

    assert snapshot["current_protective_level"] == pytest.approx(48.0)
    assert snapshot["monitor_state"] != pm.STOP_TRIGGERED


def test_a_level_recorded_after_the_split_is_left_alone(store, config):
    trade = open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0}}))
    previous = {"trading_date": "2026-09-12", "current_protective_level": 48.0,
                "monitor_state": pm.HOLD}

    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, make_bars([51.0] * 60), SESSION, previous_snapshot=previous,
                                 actions=ca.actions_for_trade(store, trade), config=config)

    assert snapshot["current_protective_level"] == pytest.approx(48.0)


# ── §5: an ex-dividend must not read as an exit signal ────────────────────

def test_an_ordinary_ex_dividend_produces_no_exit_signal(store, config):
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0, pivot=105.0)
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"dividends": {"2026-09-14": 0.30}}))

    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, make_bars([100 + i * 0.2 for i in range(60)]), SESSION,
                                 actions=ca.actions_for_trade(store, trade), config=config)

    assert snapshot["monitor_state"] in (pm.HOLD, pm.WATCH, pm.TIGHTEN_PROTECTION)
    assert snapshot["metrics"]["dividend_price_allowance"] == 0.0
    assert trade["needs_review"] == 0


def test_a_special_dividend_suppresses_mechanical_exit_conclusions(store, config):
    """The post-distribution price legitimately sits below the recorded stop.
    That is a review, not a stop-out — and never an automatic sale."""
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"dividends": {"2026-09-14": 12.0}}))

    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, make_bars([88.0] * 60), SESSION,
                                 actions=ca.actions_for_trade(store, trade), config=config)

    assert snapshot["monitor_state"] == pm.CORPORATE_ACTION_REVIEW
    assert "special dividend" in snapshot["reasons"][0].lower()
    assert store.open_trade("CACC")["shares"] == 100                # nothing was sold


def test_dividend_income_appears_in_total_pnl_but_not_in_average_cost(store, config):
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"dividends": {"2026-09-10": 1.50}}))

    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, make_bars([110.0] * 60), SESSION,
                                 actions=ca.actions_for_trade(store, trade), config=config)

    assert trade["average_cost"] == pytest.approx(100.0)
    assert snapshot["price_pnl"] == pytest.approx(1000.0)
    assert snapshot["dividend_income"] == pytest.approx(150.0)
    assert snapshot["total_pnl"] == pytest.approx(1150.0)


def test_total_pnl_includes_realised_profit_too(store, config):
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    store.record_sell("CACC", 120.0, 50, timestamp="2026-09-08T13:30:00+00:00")
    ca.process_open_positions(SESSION, config, store=store,
                              fetch_actions=actions_source(CACC={"dividends": {"2026-09-10": 1.0}}))

    trade = store.open_trade("CACC")
    snapshot = pm.build_snapshot(trade, make_bars([110.0] * 60), SESSION,
                                 actions=ca.actions_for_trade(store, trade), config=config)

    assert trade["realized_pnl"] == pytest.approx(1000.0)
    assert snapshot["price_pnl"] == pytest.approx(500.0)             # 50 shares x 10
    assert snapshot["dividend_income"] == pytest.approx(50.0)
    assert snapshot["total_pnl"] == pytest.approx(1550.0)


# ── §9: delisting / missing data ──────────────────────────────────────────

def test_missing_data_is_recorded_for_review_not_sold(store, config):
    open_position(store, "GONE", price=100.0, shares=100, stop=92.0)
    config = dict(config, trade_missing_bars_review_sessions=1)

    snapshots = pm.monitor_open_positions(SESSION, config, store=store,
                                          fetch=lambda symbols, cfg=None: {})

    assert snapshots[0]["monitor_state"] == pm.CORPORATE_ACTION_REVIEW
    assert store.open_trade("GONE")["shares"] == 100                # never a fake sale
    assert store.open_trade("GONE")["status"] == ts.OPEN
    action = store.corporate_actions("GONE")[0]
    assert action["action_type"] == ca.DELISTING
    assert action["status"] == ca.REVIEW_REQUIRED


def test_a_single_missing_session_is_not_yet_a_delisting(store, config):
    open_position(store, "QUIET", price=100.0, shares=100)
    config = dict(config, trade_missing_bars_review_sessions=3)

    snapshots = pm.monitor_open_positions(SESSION, config, store=store,
                                          fetch=lambda symbols, cfg=None: {})

    assert snapshots[0]["monitor_state"] == pm.HOLD
    assert store.corporate_actions("QUIET") == []


def test_consecutive_missing_sessions_escalate(store, config):
    open_position(store, "QUIET", price=100.0, shares=100)
    config = dict(config, trade_missing_bars_review_sessions=3)
    empty = lambda symbols, cfg=None: {}

    for session in ("2026-09-11", "2026-09-14", "2026-09-15"):
        snapshots = pm.monitor_open_positions(session, config, store=store, fetch=empty)

    assert snapshots[0]["monitor_state"] == pm.CORPORATE_ACTION_REVIEW
    assert [a["action_type"] for a in store.corporate_actions("QUIET")] == [ca.DELISTING]


def test_an_unresolved_review_outranks_every_technical_state(config):
    """Even a genuine stop-out reads as a review while the action is open."""
    trade = {"trade_id": "t1", "symbol": "CACC", "shares": 100, "average_cost": 100.0,
             "initial_stop": 92.0, "needs_review": 1, "review_reason": "MERGER 2026-09-10: review",
             "opened_at": "2026-08-03T13:30:00+00:00", "dividend_income": 0, "realized_pnl": 0}
    snapshot = pm.build_snapshot(trade, make_bars([80.0] * 60), SESSION, config=config)

    assert snapshot["monitor_state"] == pm.CORPORATE_ACTION_REVIEW
    assert snapshot["reasons"] == ["MERGER 2026-09-10: review"]


# ── §10: ordering inside the daily cycle ──────────────────────────────────

def bars_for(symbols, config=None):
    return {s: make_bars([100 + i * 0.4 for i in range(60)]) for s in symbols}


def test_corporate_actions_run_before_the_position_monitor(tmp_path):
    """The recorded snapshot must already be on the post-split basis."""
    config = make_config(tmp_path)
    journal = ts.open_store(config, create=True)
    journal.record_buy("HALO", 100.0, 100, timestamp="2026-08-03T13:30:00+00:00", initial_stop=92.0)

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     price_history=lambda symbols, cfg=None: {s: make_bars([51.0] * 60)
                                                              for s in symbols},
                     corporate_actions={"HALO": {"splits": {"2026-09-10": 2.0}}})
    outcome = sch.check_and_run("scheduled", config, deps)
    assert outcome.decision == sch.COMPLETED

    snapshot = ts.TradeStore(config["trade_db_path"]).snapshots_for_date(TUE)[0]
    assert snapshot["shares"] == 200                                 # adjusted before snapshotting
    assert snapshot["average_cost"] == pytest.approx(50.0)
    assert snapshot["initial_stop"] == pytest.approx(46.0)
    assert snapshot["monitor_state"] != pm.STOP_TRIGGERED


def test_a_held_position_absent_from_todays_screen_is_still_checked(tmp_path):
    config = make_config(tmp_path)
    journal = ts.open_store(config, create=True)
    journal.record_buy("HALO", 100.0, 100, timestamp="2026-08-03T13:30:00+00:00")

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA", "BBB"]),
                     latest_bar=TUE, price_history=bars_for,
                     corporate_actions={"HALO": {"dividends": {"2026-09-10": 2.0}}})
    sch.check_and_run("scheduled", config, deps)

    assert [c[0] for c in deps.action_calls] == ["HALO"]             # only what is held
    assert ts.TradeStore(config["trade_db_path"]).open_trade("HALO")["dividend_income"] == 200.0


def test_a_corporate_action_failure_never_fails_the_run(tmp_path):
    config = make_config(tmp_path)
    journal = ts.open_store(config, create=True)
    journal.record_buy("HALO", 100.0, 100, timestamp="2026-08-03T13:30:00+00:00")

    def explode(symbol, start=None, config=None):
        raise RuntimeError("provider exploded")

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     price_history=bars_for)
    deps.fetch_corporate_actions = explode

    outcome = sch.check_and_run("scheduled", config, deps)

    assert outcome.decision == sch.COMPLETED
    assert ts.TradeStore(config["trade_db_path"]).snapshots_for_date(TUE)                # still monitored


def test_the_daily_cycle_probes_each_symbol_once(tmp_path):
    config = make_config(tmp_path)
    journal = ts.open_store(config, create=True)
    journal.record_buy("HALO", 100.0, 100, timestamp="2026-08-03T13:30:00+00:00")
    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     price_history=bars_for, corporate_actions={"HALO": {}})

    sch.check_and_run("scheduled", config, deps)
    sch.check_and_run("manual", config, deps)                        # already canonical: skipped

    assert len(deps.action_calls) == 1


# ── §11 / §12: notification and reporting ─────────────────────────────────

def test_a_split_alert_shows_the_before_and_after(tmp_path):
    config = make_config(tmp_path, telegram_enabled=True)
    journal = ts.open_store(config, create=True)
    journal.record_buy("CACC", 620.0, 20, timestamp="2026-08-03T13:30:00+00:00", initial_stop=590.0)
    telegram = FakeTelegram()

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     telegram=telegram, price_history=bars_for,
                     corporate_actions={"CACC": {"splits": {"2026-09-10": 2.0}}})
    sch.check_and_run("scheduled", config, deps)

    alert = next(m for m in telegram.messages if "Stock split" in m)
    assert "2-for-1 split effective 2026-09-10" in alert
    assert "Shares: 20 → 40" in alert
    assert "Average cost: 620.00 → 310.00" in alert
    assert "Initial stop: 590.00 → 295.00" in alert
    assert "Economic position value unchanged." in alert
    assert "no brokerage action performed" in alert


def test_a_dividend_alert_states_what_did_not_change(tmp_path):
    config = make_config(tmp_path, telegram_enabled=True)
    journal = ts.open_store(config, create=True)
    journal.record_buy("CACC", 620.0, 40, timestamp="2026-08-03T13:30:00+00:00")
    telegram = FakeTelegram()

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     telegram=telegram, price_history=bars_for,
                     corporate_actions={"CACC": {"dividends": {"2026-09-10": 1.20}}})
    sch.check_and_run("scheduled", config, deps)

    alert = next(m for m in telegram.messages if "Cash dividend" in m)
    assert "Ex-date: 2026-09-10" in alert
    assert "Dividend: 1.20/share" in alert
    assert "Eligible shares: 40" in alert
    assert "Expected dividend income: 48.00" in alert
    assert "Shares and average cost unchanged." in alert


def test_a_spinoff_alert_says_nothing_was_allocated(tmp_path):
    config = make_config(tmp_path, telegram_enabled=True)
    journal = ts.open_store(config, create=True)
    journal.record_buy("XYZ", 100.0, 50, timestamp="2026-08-03T13:30:00+00:00")
    telegram = FakeTelegram()

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     telegram=telegram, price_history=bars_for,
                     corporate_actions={"XYZ": {"events": [
                         {"action_type": "SPINOFF", "effective_date": "2026-09-10"}]}})
    sch.check_and_run("scheduled", config, deps)

    alert = next(m for m in telegram.messages if "Corporate action review" in m)
    assert "Spin-off detected." in alert
    assert "Automatic cost-basis allocation was not attempted." in alert
    assert "Review required." in alert


def test_a_failed_alert_send_does_not_fail_the_run(tmp_path):
    config = make_config(tmp_path, telegram_enabled=True)
    journal = ts.open_store(config, create=True)
    journal.record_buy("CACC", 620.0, 20, timestamp="2026-08-03T13:30:00+00:00")

    class OneGoodMessage(FakeTelegram):
        def send_message(self, text, reply_markup=None, chat_id=None):
            if "Stock split" in text:
                raise tg.TelegramError("nope", attempts=1)
            return super().send_message(text, reply_markup=reply_markup, chat_id=chat_id)

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     telegram=OneGoodMessage(), price_history=bars_for,
                     corporate_actions={"CACC": {"splits": {"2026-09-10": 2.0}}})
    outcome = sch.check_and_run("scheduled", config, deps)

    assert outcome.decision == sch.COMPLETED
    assert outcome.notification_status == "SENT"


def test_the_report_stays_uncluttered_without_corporate_actions():
    snapshot = {"symbol": "CACC", "monitor_state": "HOLD", "shares": 20, "average_cost": 620.0,
                "close": 640.0, "pnl_pct": 3.2, "dividend_income": 0, "corporate_actions": []}
    block = tg._position_block(snapshot)
    assert "Dividend income" not in block
    assert "Corporate actions since entry" not in block


def test_the_report_shows_the_pnl_split_once_there_are_actions():
    snapshot = {"symbol": "CACC", "monitor_state": "HOLD", "shares": 40, "average_cost": 310.0,
                "close": 320.0, "pnl_pct": 3.2, "price_pnl": 400.0, "dividend_income": 48.0,
                "total_pnl": 448.0,
                "corporate_actions_json": json.dumps(["2026-09-10 SPLIT (2-for-1)",
                                                      "2026-09-12 CASH_DIVIDEND (1.2/share)"])}
    block = tg._position_block(snapshot)
    assert "Price P&L: 400.00" in block
    assert "Dividend income: 48.00" in block
    assert "Total P&L: 448.00" in block
    assert "2026-09-10 SPLIT (2-for-1)" in block


def test_the_position_report_never_claims_an_order_was_placed(tmp_path):
    config = make_config(tmp_path, telegram_enabled=True)
    journal = ts.open_store(config, create=True)
    journal.record_buy("CACC", 620.0, 20, timestamp="2026-08-03T13:30:00+00:00")
    telegram = FakeTelegram()

    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA"]), latest_bar=TUE,
                     telegram=telegram, price_history=bars_for,
                     corporate_actions={"CACC": {"splits": {"2026-09-10": 2.0}}})
    sch.check_and_run("scheduled", config, deps)

    text = "\n".join(telegram.messages).lower()
    for phrase in ("order placed", "order submitted", "order filled", "executed at broker",
                   "shares were sold", "position was liquidated"):
        assert phrase not in text
