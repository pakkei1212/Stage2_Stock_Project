"""Stage I inside the Stage-H daily cycle, plus the advisory-only boundary.

A scheduled check runs the (faked) Stage A-G screen, then: gates the Stage-G
rows into persisted plans, monitors every open position — including one the
screener never saw — and appends the "💼 OPEN POSITIONS" section to the
Telegram report. No network, no broker, no Claude.
"""
import pathlib
import re
from datetime import date

import pytest

from pipeline import history_store as hs
from pipeline import scheduler as sch
from pipeline import setup_gate as gate
from pipeline import telegram_trade as tt
from pipeline import trade_store as ts
from stage_h_fakes import FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt
from stage_i_fakes import make_bars

TUE = date(2026, 9, 15)
RISE = [100 + i * 0.8 for i in range(60)]

NUMERIC_READY = {
    "vcp_numeric_quality": "strong",
    "pivot_state": "coiling_below_pivot",
    "pct_from_pivot": -1.7,
    "contraction_low_prices": [95.0, 104.0, 110.0],
    "breakout_volume_confirmed": None,
    "current_volume_vs_20d": 0.7,
    "sessions_since_breakout": None,
}
# Same setup with a tighter final contraction: the structural stop fits the risk
# ceiling, so the persisted plan is usable (NUMERIC_READY is SKIP_RISK_TOO_WIDE).
NUMERIC_USABLE = {**NUMERIC_READY, "contraction_low_prices": [95.0, 104.0, 114.0]}


def bars_for(symbols, config=None):
    return {s: make_bars(RISE) for s in symbols}


def run_check(tmp_path, *, telegram=None, screen=None, **config_overrides):
    config = make_config(tmp_path, telegram_enabled=telegram is not None,
                         telegram_send_charts=telegram is not None, **config_overrides)
    clock = FakeClock(sgt("2026-09-16 09:15"))
    deps = make_deps(clock, screen=screen or FakeScreen(["AAA", "BBB"], vcp_numeric=NUMERIC_READY),
                     latest_bar=TUE, telegram=telegram, price_history=bars_for)
    outcome = sch.check_and_run("scheduled", config, deps)
    return config, outcome, deps


def journal(config):
    return ts.TradeStore(config["trade_db_path"])


def seed_open_position(tmp_path, symbol="HALO", price=100.0, shares=10, stop=90.0):
    """A position held before today's run — the screener never returns this symbol."""
    config = make_config(tmp_path)
    store = ts.open_store(config, create=True)
    store.record_buy(symbol, price, shares, initial_stop=stop)
    return store


def test_run_persists_setup_gate_states_and_plans(tmp_path):
    config, outcome, _ = run_check(tmp_path)
    assert outcome.decision == sch.COMPLETED

    plans = {p["symbol"]: p for p in journal(config).setup_plans_for_date(TUE)}
    assert set(plans) == {"AAA", "BBB"}
    assert plans["AAA"]["setup_state"] == gate.READY
    assert plans["AAA"]["entry_status"] == "AWAIT_BREAKOUT"
    assert plans["AAA"]["suggested_initial_stop"] is not None
    assert plans["AAA"]["suggested_shares"] >= 0


def test_held_position_is_monitored_even_though_it_is_not_in_todays_screen(tmp_path):
    seed_open_position(tmp_path, "HALO")
    config, outcome, _ = run_check(tmp_path)

    store = journal(config)
    snapshots = store.snapshots_for_date(TUE)
    assert [s["symbol"] for s in snapshots] == ["HALO"]
    assert snapshots[0]["monitor_state"] in ("HOLD", "TIGHTEN_PROTECTION", "WATCH")
    assert snapshots[0]["close"] is not None

    screened = hs.HistoryStore(config["history_db_path"]).results_for_run(outcome.run_id)
    assert "HALO" not in {r["symbol"] for r in screened}         # truly independent of the screen


def test_telegram_report_gets_an_open_positions_section(tmp_path):
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    config, _, _ = run_check(tmp_path, telegram=telegram)

    report = "\n".join(telegram.messages)
    assert "💼 <b>OPEN POSITIONS</b>" in report and "💼 Open positions: 1" in report
    assert "HALO" in report
    assert "never places, modifies or cancels an order" in report


def test_no_positions_means_no_positions_section(tmp_path):
    telegram = FakeTelegram()
    run_check(tmp_path, telegram=telegram)
    report = "\n".join(telegram.messages)
    assert "💼 <b>OPEN POSITIONS</b>" not in report and "💼 Open positions: 0" in report


def test_resend_reuses_the_stored_snapshots(tmp_path):
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    config, _, deps = run_check(tmp_path, telegram=telegram)

    telegram.messages.clear()
    sch.resend_notification(config, deps)
    assert "💼 <b>OPEN POSITIONS</b>" in "\n".join(telegram.messages)


def _callbacks(markup):
    return [b["callback_data"] for row in (markup or {}).get("inline_keyboard", []) for b in row]


def test_ready_charts_carry_the_open_position_button(tmp_path):
    telegram = FakeTelegram()
    config, _, _ = run_check(tmp_path, telegram=telegram, trade_bot_enabled=True,
                             screen=FakeScreen(["AAA", "BBB"], vcp_numeric=NUMERIC_USABLE))

    plans = {p["symbol"]: p for p in journal(config).setup_plans_for_date(TUE)}
    assert all(p["setup_state"] == gate.READY and p["trade_plan_status"] == "OK" for p in plans.values())
    assert telegram.photos, "VCP verification charts should have been sent"
    assert [_callbacks(m) for m in telegram.markups] == [["op:AAA"], ["op:BBB"]]
    assert all("Entry trigger:" in caption and "Suggested size:" in caption for _, caption in telegram.photos)
    # the charts carry the buttons, so the main report does not repeat them
    assert telegram.message_markups[0] is None


def test_ready_setup_already_held_gets_view_position_not_open(tmp_path):
    seed_open_position(tmp_path, "AAA", price=118.0, shares=10, stop=110.0)
    telegram = FakeTelegram()
    run_check(tmp_path, telegram=telegram, trade_bot_enabled=True,
              screen=FakeScreen(["AAA", "BBB"], vcp_numeric=NUMERIC_USABLE))

    assert [_callbacks(m) for m in telegram.markups] == [["vp:AAA"], ["op:BBB"]]
    assert "Already held" in telegram.photos[0][1]


def test_main_report_keyboard_holds_buttons_when_no_chart_is_sent(tmp_path):
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    config = make_config(tmp_path, telegram_enabled=True, telegram_send_charts=False, trade_bot_enabled=True)
    deps = make_deps(FakeClock(sgt("2026-09-16 09:15")), screen=FakeScreen(["AAA", "BBB"], vcp_numeric=NUMERIC_USABLE),
                     latest_bar=TUE, telegram=telegram, price_history=bars_for)
    sch.check_and_run("scheduled", config, deps)

    assert telegram.photos == []
    assert _callbacks(telegram.message_markups[0]) == ["op:AAA", "op:BBB", "vp:HALO"]
    assert telegram.message_markups[1] is None                     # the debug message has no buttons


def test_no_position_buttons_when_the_trade_bot_is_off(tmp_path):
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    run_check(tmp_path, telegram=telegram, trade_bot_enabled=False)
    assert telegram.markups and all(m is None for m in telegram.markups)
    assert all(m is None for m in telegram.message_markups)
    assert "READY setup buttons: off" in telegram.messages[-1]


def test_weak_setups_get_no_plan_and_no_button(tmp_path):
    telegram = FakeTelegram()
    screen = FakeScreen(["AAA", "BBB"], vcp_numeric={**NUMERIC_READY, "vcp_numeric_quality": "weak"})
    config, _, _ = run_check(tmp_path, telegram=telegram, screen=screen, trade_bot_enabled=True)

    plans = journal(config).setup_plans_for_date(TUE)
    assert {p["setup_state"] for p in plans} == {gate.NOT_READY}
    assert all(p["suggested_initial_stop"] is None for p in plans)
    assert all(m is None for m in telegram.markups + telegram.message_markups)


def test_risk_too_wide_setup_is_watch_with_no_button(tmp_path):
    telegram = FakeTelegram()
    config, _, _ = run_check(tmp_path, telegram=telegram, trade_bot_enabled=True)   # NUMERIC_READY

    plans = journal(config).setup_plans_for_date(TUE)
    assert {(p["setup_state"], p["trade_plan_status"]) for p in plans} == {(gate.READY, "SKIP_RISK_TOO_WIDE")}
    report = telegram.messages[0]
    assert "🟢 Ready: none" in report and "🟡 Watch: AAA, BBB" in report
    assert "SKIP_RISK_TOO_WIDE" in report and "NO TRADE — STOP TOO WIDE" in report
    assert all(m is None for m in telegram.markups + telegram.message_markups)
    assert "READY: 2" in telegram.messages[-1] and "SKIP_RISK_TOO_WIDE: 2" in telegram.messages[-1]


def test_debug_summary_explains_how_many_setups_became_ready(tmp_path):
    telegram = FakeTelegram()
    run_check(tmp_path, telegram=telegram, trade_bot_enabled=True,
              screen=FakeScreen(["AAA", "BBB"], vcp_numeric=NUMERIC_USABLE))
    debug = telegram.messages[-1]
    assert debug.startswith("🛠")
    assert "Setup plans: 2" in debug and "READY: 2" in debug and "MANUAL_REVIEW: 0" in debug
    assert "Trade plan OK: 2" in debug and "SKIP_RISK_TOO_WIDE: 0" in debug
    assert "Dashboard: 🟢 2 ready" in debug and "READY setup buttons: 2" in debug
    assert "Stage D screened" not in telegram.messages[0]          # main report: decisions only


def test_trade_layer_failure_never_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sch.tj, "generate_setup_plans",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    _, outcome, _ = run_check(tmp_path)
    assert outcome.decision == sch.COMPLETED                      # screening is unaffected


def test_trade_layer_can_be_switched_off(tmp_path):
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    config, _, _ = run_check(tmp_path, telegram=telegram, trade_journal_enabled=False)
    assert journal(config).snapshots_for_date(TUE) == []
    assert "💼 <b>OPEN POSITIONS</b>" not in "\n".join(telegram.messages)


# ── the advisory-only boundary (§13) ──────────────────────────────────────

BROKER_IMPORTS = ("ib_insync", "ib_async", "ibapi", "alpaca", "ccxt", "tda.client", "schwab",
                  "interactive_brokers", "robin_stocks", "questrade")
ORDER_CALLS = ("place_order", "placeOrder", "submit_order", "submitOrder", "cancel_order",
               "cancelOrder", "modify_order", "reqAccountSummary", "account_balance",
               "broker_client", "order_ticket")


def _sources():
    root = pathlib.Path(__file__).resolve().parents[1]
    return {p: p.read_text(encoding="utf-8") for p in (root / "pipeline").glob("*.py")}


def test_no_module_imports_a_brokerage_library():
    for path, text in _sources().items():
        for name in BROKER_IMPORTS:
            assert not re.search(rf"^\s*(import|from)\s+{re.escape(name)}\b", text, re.MULTILINE), \
                f"{path.name} imports {name}"


def test_no_order_placement_code_path_exists():
    for path, text in _sources().items():
        for call in ORDER_CALLS:
            assert call not in text, f"{path.name} contains {call}"


def test_requirements_contain_no_brokerage_client():
    root = pathlib.Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in BROKER_IMPORTS:
        assert name.split(".")[0] not in requirements


def test_the_journal_only_records_user_supplied_fills(tmp_path):
    """Every write path into `trades` starts from a price/quantity the user typed."""
    store = seed_open_position(tmp_path, "CACC", price=622.50, shares=20)
    fills = store.fills(store.open_trade("CACC")["trade_id"])
    assert [(f["side"], f["price"], f["shares"]) for f in fills] == [("BUY", 622.50, 20.0)]


def test_position_and_report_wording_never_claims_an_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_ID", "1")
    seed_open_position(tmp_path, "HALO")
    telegram = FakeTelegram()
    run_check(tmp_path, telegram=telegram)
    text = "\n".join(telegram.messages).lower()
    for phrase in ("order placed", "order submitted", "order filled", "executed at broker"):
        assert phrase not in text


def test_help_text_states_the_boundary():
    assert "no broker is connected" in tt.HELP_TEXT
