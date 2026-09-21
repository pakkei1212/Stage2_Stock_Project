"""Stage I CLI fallback: same service layer as Telegram, same confirmation rule."""
import io

import pytest

from pipeline import trade_cli
from pipeline import trade_journal as tj
from pipeline import trade_store as ts
from stage_i_fakes import RUN_ID, SESSION, flat_bars, stage_g_row, trade_config


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


@pytest.fixture
def db(config):
    ts.open_store(config, create=True)
    return config["trade_db_path"]


def run(db, config, argv):
    out = io.StringIO()
    code = trade_cli.main(["--db", db] + argv, config, out)
    return code, out.getvalue()


def test_add_records_a_buy_after_confirmation(db, config):
    code, text = run(db, config, ["--yes", "add", "CACC", "--price", "622.50", "--shares", "20"])
    assert code == 0
    assert "Record BUY  CACC" in text and "no order is placed" in text
    assert "Recorded." in text
    store = ts.TradeStore(db)
    trade = store.open_trade("CACC")
    assert trade["shares"] == 20.0 and trade["average_cost"] == 622.50


def test_declining_the_prompt_writes_nothing(db, config, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    code, text = run(db, config, ["add", "CACC", "--price", "622.50", "--shares", "20"])
    assert code == 1 and "Cancelled" in text
    assert ts.TradeStore(db).open_positions() == []


def test_accepting_the_prompt_records_the_buy(db, config, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    code, _ = run(db, config, ["add", "CACC", "--price", "622.50", "--shares", "20"])
    assert code == 0 and ts.TradeStore(db).open_trade("CACC")["shares"] == 20.0


def test_partial_sell_then_close(db, config):
    run(db, config, ["--yes", "add", "CACC", "--price", "600", "--shares", "20"])
    code, text = run(db, config, ["--yes", "sell", "CACC", "--price", "680", "--shares", "10"])
    assert code == 0 and "Recorded. Sold 10 CACC" in text
    store = ts.TradeStore(db)
    assert store.open_trade("CACC")["shares"] == 10.0

    code, text = run(db, config, ["--yes", "close", "CACC", "--price", "705"])
    assert code == 0 and "Record CLOSE CACC" in text
    assert store.open_trade("CACC") is None
    assert store.trades_for_symbol("CACC")[0]["status"] == ts.CLOSED


def test_selling_more_than_held_is_an_error(db, config):
    run(db, config, ["--yes", "add", "CACC", "--price", "600", "--shares", "5"])
    code, text = run(db, config, ["--yes", "sell", "CACC", "--price", "680", "--shares", "10"])
    assert code == 2 and "only 5 held" in text


def test_positions_and_show(db, config):
    run(db, config, ["--yes", "add", "CACC", "--price", "600", "--shares", "20"])
    store = ts.TradeStore(db)
    trade_id = store.open_trade("CACC")["trade_id"]
    store.save_snapshot({"trade_id": trade_id, "symbol": "CACC", "trading_date": SESSION,
                         "close": 671.4, "pnl_pct": 11.9, "monitor_state": "HOLD",
                         "current_protective_level": 620.0, "reasons": ["trend structure intact"]})

    _, positions = run(db, config, ["positions"])
    assert "CACC" in positions and "671.40" in positions and "HOLD" in positions
    assert "no orders are placed" in positions

    _, detail = run(db, config, ["show", "CACC"])
    assert "protective level 620.00" in detail and "trend structure intact" in detail
    assert "BUY" in detail


def test_positions_with_nothing_held(db, config):
    assert "No open positions" in run(db, config, ["positions"])[1]


def test_trades_listing(db, config):
    run(db, config, ["--yes", "add", "CACC", "--price", "600", "--shares", "20"])
    run(db, config, ["--yes", "close", "CACC", "--price", "700"])
    assert "No trades recorded yet." in run(db, config, ["trades"])[1]        # open only, by default
    assert "CLOSED" in run(db, config, ["trades", "--all"])[1]


def test_plan_lists_the_recorded_gate_states(db, config):
    store = ts.TradeStore(db)
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("AAA", market_cap=5e9),
                                              stage_g_row("BBB", quality="weak", rank=2)],
                            config, store=store,
                            fetch=lambda symbols, cfg=None: {s: flat_bars(60, 98.0) for s in symbols})
    _, text = run(db, config, ["plan"])
    assert "AAA" in text and "READY" in text and "AWAIT_BREAKOUT" in text
    assert "BBB" in text and "NOT_READY" in text


def test_plan_without_any_recorded_plans(db, config):
    assert "No setup plans recorded yet" in run(db, config, ["plan"])[1]


def test_show_for_an_unknown_symbol(db, config):
    code, text = run(db, config, ["show", "NVDA"])
    assert code == 1 and "No recorded trade for NVDA." in text
