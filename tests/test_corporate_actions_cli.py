"""The §13 audit surface: `trade_cli actions`, `record-action`, `resolve`, and
the `/position` detail. Synthetic journal only — no network, no broker.

The reconstruction these tests protect is the one the user needs to trust:

    original fills + corporate actions + sales = the current position
"""
import io

import pytest

from pipeline import corporate_actions as ca
from pipeline import telegram_trade as tt
from pipeline import trade_cli
from pipeline import trade_journal as tj
from pipeline import trade_store as ts
from stage_i_fakes import actions_source, trade_config


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


@pytest.fixture
def store(config):
    return ts.TradeStore(config["trade_db_path"])


def run_cli(config, *argv):
    out = io.StringIO()
    code = trade_cli.main(["--db", config["trade_db_path"], *argv], config=config, out=out)
    return code, out.getvalue()


def open_position(store, symbol="CACC", price=620.0, shares=20, stop=590.0,
                  timestamp="2026-08-03T13:30:00+00:00"):
    trade_id, _ = store.record_buy(symbol, price, shares, timestamp=timestamp, initial_stop=stop)
    return store.get_trade(trade_id)


# ── listing ───────────────────────────────────────────────────────────────

def test_actions_lists_one_symbol(store, config):
    open_position(store)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0},
                                                                 "dividends": {"2026-09-12": 1.0}}))

    code, text = run_cli(config, "actions", "CACC")

    assert code == 0
    assert "2026-09-10" in text and "SPLIT" in text and "APPLIED" in text
    assert "CASH_DIVIDEND" in text
    assert "never rewritten" in text


def test_actions_all_covers_every_symbol(store, config):
    open_position(store, "AAA", price=100.0, shares=10)
    open_position(store, "BBB", price=100.0, shares=10)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(AAA={"splits": {"2026-09-10": 2.0}},
                                                           BBB={"dividends": {"2026-09-11": 1.0}}))

    code, text = run_cli(config, "actions", "--all")

    assert code == 0
    assert "AAA" in text and "BBB" in text


def test_actions_says_so_when_there_is_nothing(store, config):
    open_position(store)
    code, text = run_cli(config, "actions", "CACC")
    assert code == 0 and "No corporate actions recorded" in text


# ── recording by hand ─────────────────────────────────────────────────────

def test_record_action_previews_before_writing(store, config, monkeypatch):
    open_position(store, "CACC", price=620.0, shares=20, stop=590.0)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))

    code, text = run_cli(config, "record-action", "CACC", "--type", "SPLIT",
                         "--date", "2026-10-01", "--ratio", "2")

    assert code == 1                                          # declined at the prompt
    assert "shares 20 -> 40" in text
    assert "average cost 620.00 -> 310.00" in text
    assert "no order is placed" in text
    assert store.open_trade("CACC")["shares"] == 20            # nothing written
    assert store.corporate_actions("CACC") == []


def test_record_action_applies_once_confirmed(store, config):
    open_position(store, "CACC", price=620.0, shares=20, stop=590.0)

    code, text = run_cli(config, "--yes", "record-action", "CACC", "--type", "SPLIT",
                         "--date", "2026-10-01", "--ratio", "2")

    assert code == 0 and "APPLIED" in text
    after = store.open_trade("CACC")
    assert after["shares"] == 40 and after["average_cost"] == pytest.approx(310.0)


def test_recording_the_same_action_again_changes_nothing(store, config):
    open_position(store, "CACC", price=620.0, shares=20)
    args = ("--yes", "record-action", "CACC", "--type", "SPLIT", "--date", "2026-10-01",
            "--ratio", "2")

    run_cli(config, *args)
    code, text = run_cli(config, *args)

    assert code == 0 and "Already recorded" in text
    assert store.open_trade("CACC")["shares"] == 40
    assert len(store.corporate_actions("CACC")) == 1


def test_record_action_can_change_a_ticker(store, config):
    trade = open_position(store, "ABC", price=50.0, shares=100)

    code, _ = run_cli(config, "--yes", "record-action", "ABC", "--type", "SYMBOL_CHANGE",
                      "--date", "2026-10-01", "--new-symbol", "XYZ")

    assert code == 0
    assert store.get_trade(trade["trade_id"])["symbol"] == "XYZ"


@pytest.mark.parametrize("argv, expected", [
    (("--type", "SPLIT", "--date", "not-a-date", "--ratio", "2"), "not a YYYY-MM-DD date"),
    (("--type", "SPLIT", "--date", "2026-10-01", "--ratio", "0.2"), "is a REVERSE_SPLIT"),
    (("--type", "SPLIT", "--date", "2026-10-01"), "needs a positive share ratio"),
    (("--type", "CASH_DIVIDEND", "--date", "2026-10-01"), "needs a positive cash amount"),
])
def test_record_action_refuses_nonsense_rather_than_recording_it(store, config, argv, expected):
    open_position(store, "CACC")

    code, text = run_cli(config, "--yes", "record-action", "CACC", *argv)

    assert code == 2 and expected in text
    assert store.corporate_actions("CACC") == []


def test_a_complex_action_recorded_by_hand_is_review_only(store, config):
    open_position(store, "CACC", price=100.0, shares=50, stop=92.0)

    code, text = run_cli(config, "--yes", "record-action", "CACC", "--type", "MERGER",
                         "--date", "2026-10-01")

    assert code == 0
    assert "no cost basis, exchange ratio or share quantity is guessed" in text
    after = store.open_trade("CACC")
    assert after["shares"] == 50 and after["needs_review"] == 1


# ── resolving ─────────────────────────────────────────────────────────────

def test_resolve_releases_the_position_and_records_the_users_stop(store, config):
    open_position(store, "CACC", price=100.0, shares=50, stop=92.0)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"events": [
                                  {"action_type": "SPINOFF", "effective_date": "2026-09-10"}]}))
    assert store.open_trade("CACC")["needs_review"] == 1

    code, text = run_cli(config, "resolve", "CACC", "--stop", "85")

    assert code == 0
    assert "Resolved" in text and "your number" in text
    after = store.open_trade("CACC")
    assert after["needs_review"] == 0
    assert after["initial_stop"] == pytest.approx(85.0)


def test_resolve_on_a_clean_position_says_so(store, config):
    open_position(store, "CACC")
    code, text = run_cli(config, "resolve", "CACC")
    assert code == 0 and "no outstanding corporate-action review" in text


def test_resolve_rejects_an_unknown_symbol(store, config):
    code, text = run_cli(config, "resolve", "NOPE")
    assert code == 2 and "no open position" in text


# ── the reconstruction (§13) ──────────────────────────────────────────────

def test_show_reconstructs_fills_plus_actions_plus_sales(store, config):
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0},
                                                                 "dividends": {"2026-09-12": 0.50}}))
    store.record_sell("CACC", 60.0, 50, timestamp="2026-09-14T13:30:00+00:00")

    code, text = run_cli(config, "show", "CACC")

    assert code == 0
    assert "BUY" in text and "100 @ 100.00" in text             # the ORIGINAL fill, unchanged
    assert "SELL" in text and "50 @ 60.00" in text
    assert "2026-09-10  SPLIT" in text
    assert "dividend income" in text
    # 100 shares -> x2 = 200 -> sold 50 -> 150 still held, at half the cost.
    after = store.open_trade("CACC")
    assert after["shares"] == 150 and after["average_cost"] == pytest.approx(50.0)


def test_position_detail_reports_the_pnl_split_and_the_actions(store, config):
    open_position(store, "CACC", price=100.0, shares=100, stop=92.0)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"dividends": {"2026-09-12": 0.50}}))

    detail = tj.position_detail("CACC", config, store=store)
    text = tt.format_position_detail(detail)

    assert "Dividend income: $50.00" in text
    assert "Corporate actions since entry" in text
    assert "CASH_DIVIDEND" in text
    assert "never rewritten" in text
    assert "no orders are placed" in text


def test_position_detail_is_unchanged_without_corporate_actions(store, config):
    open_position(store, "CACC")
    text = tt.format_position_detail(tj.position_detail("CACC", config, store=store))
    assert "Corporate actions since entry" not in text
    assert "Dividend income" not in text


def test_position_detail_shows_a_ticker_history(store, config):
    open_position(store, "ABC", price=50.0, shares=100)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(ABC={"events": [
                                  {"action_type": "SYMBOL_CHANGE", "effective_date": "2026-09-10",
                                   "new_symbol": "XYZ"}]}))

    text = tt.format_position_detail(tj.position_detail("XYZ", config, store=store))

    assert "Ticker history: ABC → XYZ" in text


def test_an_open_review_is_stated_in_the_detail(store, config):
    open_position(store, "CACC", price=100.0, shares=50)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"events": [
                                  {"action_type": "MERGER", "effective_date": "2026-09-10"}]}))

    text = tt.format_position_detail(tj.position_detail("CACC", config, store=store))

    assert "Unresolved" in text
    assert "suppressed" in text


# ── the advisory-only boundary still holds ────────────────────────────────

def test_no_corporate_action_path_can_place_an_order(store, config):
    """Every corporate-action outcome leaves the ledger a record of what the
    USER did, plus adjustments — never an execution."""
    open_position(store, "CACC", price=100.0, shares=50)
    ca.process_open_positions("2026-09-15", config, store=store,
                              fetch_actions=actions_source(CACC={"splits": {"2026-09-10": 2.0},
                                                                 "events": [
                                                                     {"action_type": "MERGER",
                                                                      "effective_date": "2026-09-11",
                                                                      "cash_amount": 200.0}]}))

    trade = store.open_trade("CACC")
    assert trade["status"] == ts.OPEN                          # nothing closed itself
    assert [(f["side"], f["price"], f["shares"]) for f in store.fills(trade["trade_id"])] == \
        [("BUY", 100.0, 50.0)]                                 # the only fill is the user's
