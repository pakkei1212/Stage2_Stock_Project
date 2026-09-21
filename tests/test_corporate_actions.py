"""Stage I corporate actions: splits, dividends, ticker changes and review events.

Synthetic data only — every corporate-action lookup goes through
``stage_i_fakes.actions_source``, so nothing here asks Yahoo what happened to a
ticker and every number below is exact. No broker, no Telegram, no Claude.

The two invariants most of these tests exist to protect:

* a share-basis change preserves economic value and leaves realised P&L alone;
* a dividend is cash, not a change to what the shares cost — and it must never
  be adjusted for twice, because the bars are already adjusted.
"""
import pytest

from pipeline import corporate_actions as ca
from pipeline import trade_store as ts
from stage_i_fakes import actions_source, trade_config

SESSION = "2026-09-15"


@pytest.fixture
def store(tmp_path):
    return ts.TradeStore(str(tmp_path / "trades" / "journal.sqlite3"))


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


def open_position(store, symbol="CACC", price=620.0, shares=20, stop=590.0, pivot=630.0,
                  timestamp="2026-09-01T13:30:00+00:00"):
    trade_id, _ = store.record_buy(symbol, price, shares, timestamp=timestamp, initial_stop=stop,
                                   setup={"pivot_price": pivot})
    return store.get_trade(trade_id)


def run(store, config, fetch, trading_date=SESSION, **kwargs):
    return ca.process_open_positions(trading_date, config, store=store, fetch_actions=fetch,
                                     **kwargs)


# ── §1 price-data semantics: never adjust twice ───────────────────────────

def test_the_declared_price_basis_is_split_and_dividend_adjusted(config):
    """yfinance runs with auto_adjust=True everywhere in this project."""
    import inspect

    from pipeline import data_sources

    source = inspect.getsource(data_sources)
    assert "auto_adjust=True" in source
    assert "auto_adjust=False" not in source
    assert ca.PRICE_BASIS == "split_and_dividend_adjusted"
    assert config["trade_price_basis"] == ca.PRICE_BASIS


def test_adjusted_bars_get_no_second_dividend_adjustment(config):
    """The regression guard: with adjusted prices the allowance is exactly zero."""
    dividend = ca.make_action("CACC", "2026-09-14", ca.CASH_DIVIDEND, cash_amount=1.20)
    dividend["status"] = ca.APPLIED
    allowance, note = ca.dividend_price_allowance([dividend], latest_date=SESSION, config=config)
    assert allowance == 0.0
    assert note is None


def test_a_raw_price_source_would_be_compensated_once(config):
    """The other half of the guard: the raw branch adds the dividend back once,
    so the compensation lives in exactly one place."""
    dividend = ca.make_action("CACC", "2026-09-14", ca.CASH_DIVIDEND, cash_amount=1.20)
    dividend["status"] = ca.APPLIED
    allowance, note = ca.dividend_price_allowance([dividend], latest_date=SESSION, config=config,
                                                  price_basis=ca.RAW_PRICE_BASIS)
    assert allowance == pytest.approx(1.20)
    assert "added back" in note


def test_a_dividend_outside_the_window_is_not_compensated(config):
    dividend = ca.make_action("CACC", "2026-01-02", ca.CASH_DIVIDEND, cash_amount=1.20)
    dividend["status"] = ca.APPLIED
    allowance, _ = ca.dividend_price_allowance([dividend], latest_date=SESSION, config=config,
                                               price_basis=ca.RAW_PRICE_BASIS)
    assert allowance == 0.0


# ── §3 splits and reverse splits ──────────────────────────────────────────

def test_two_for_one_split_preserves_economic_value(store, config):
    trade = open_position(store)                       # 20 @ 620, stop 590, pivot 630
    before_value = trade["shares"] * trade["average_cost"]

    run(store, config, actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    after = store.open_trade("CACC")
    assert after["shares"] == 40
    assert after["average_cost"] == pytest.approx(310.0)
    assert after["initial_stop"] == pytest.approx(295.0)
    assert after["pivot_price"] == pytest.approx(315.0)
    assert after["shares"] * after["average_cost"] == pytest.approx(before_value)


def test_one_for_five_reverse_split(store, config):
    open_position(store, "HALO", price=10.0, shares=500, stop=9.0, pivot=11.0)

    run(store, config, actions_source(HALO={"splits": {"2026-09-10": 0.2}}))

    after = store.open_trade("HALO")
    assert after["shares"] == 100
    assert after["average_cost"] == pytest.approx(50.0)
    assert after["initial_stop"] == pytest.approx(45.0)
    assert after["shares"] * after["average_cost"] == pytest.approx(5000.0)
    recorded = store.corporate_actions("HALO")[0]
    assert recorded["action_type"] == ca.REVERSE_SPLIT


def test_fractional_split_ratio(store, config):
    open_position(store, "MIDC", price=90.0, shares=100, stop=81.0, pivot=95.0)

    run(store, config, actions_source(MIDC={"splits": {"2026-09-10": 1.5}}))    # 3-for-2

    after = store.open_trade("MIDC")
    assert after["shares"] == pytest.approx(150.0)
    assert after["average_cost"] == pytest.approx(60.0)
    assert after["initial_stop"] == pytest.approx(54.0)
    assert after["shares"] * after["average_cost"] == pytest.approx(9000.0)


def test_realized_pnl_earned_before_a_split_does_not_change(store, config):
    open_position(store, "CACC", price=600.0, shares=20, stop=550.0)
    store.record_sell("CACC", 700.0, 10, timestamp="2026-09-05T13:30:00+00:00")
    booked = store.open_trade("CACC")["realized_pnl"]
    assert booked == pytest.approx(1000.0)

    run(store, config, actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    after = store.open_trade("CACC")
    assert after["realized_pnl"] == pytest.approx(booked)          # money already earned
    assert after["shares"] == 20                                   # 10 remaining, doubled
    assert after["average_cost"] == pytest.approx(300.0)


def test_a_split_never_rewrites_the_fills(store, config):
    trade = open_position(store)
    before = [(f["side"], f["price"], f["shares"]) for f in store.fills(trade["trade_id"])]

    run(store, config, actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    assert [(f["side"], f["price"], f["shares"])
            for f in store.fills(trade["trade_id"])] == before == [("BUY", 620.0, 20.0)]


def test_a_split_records_a_corporate_action_event(store, config):
    trade = open_position(store)
    run(store, config, actions_source(CACC={"splits": {"2026-09-10": 2.0}}))

    events = [e for e in store.events(trade["trade_id"])
              if e["event_type"] == ts.EVENT_CORPORATE_ACTION]
    assert len(events) == 1
    assert "SPLIT 2-for-1 effective 2026-09-10" in events[0]["reason"]
    assert '"before"' in events[0]["detail"] and '"after"' in events[0]["detail"]


# ── §6 stock dividends ────────────────────────────────────────────────────

def test_ten_percent_stock_dividend_is_a_share_adjustment(store, config):
    trade = open_position(store, "DIVX", price=100.0, shares=100, stop=92.0, pivot=105.0)
    action = ca.make_action("DIVX", "2026-09-10", ca.STOCK_DIVIDEND, split_ratio=1.10)
    row, _ = store.record_corporate_action(action)

    ca.apply_action(store, row, trade)

    after = store.open_trade("DIVX")
    assert after["shares"] == pytest.approx(110.0)
    assert after["average_cost"] == pytest.approx(100 / 1.1)
    assert after["initial_stop"] == pytest.approx(92 / 1.1)
    assert after["shares"] * after["average_cost"] == pytest.approx(10_000.0)
    assert store.corporate_actions("DIVX")[0]["status"] == ca.APPLIED


# ── §4 cash dividends ─────────────────────────────────────────────────────

def test_ordinary_cash_dividend_books_income_and_changes_nothing_else(store, config):
    open_position(store, "CACC", price=620.0, shares=40, stop=590.0)

    run(store, config, actions_source(CACC={"dividends": {"2026-09-10": 1.20}}))

    after = store.open_trade("CACC")
    assert after["dividend_income"] == pytest.approx(48.0)          # 1.20 x 40
    assert after["shares"] == 40                                   # unchanged
    assert after["average_cost"] == pytest.approx(620.0)            # unchanged
    assert after["realized_pnl"] == 0
    assert after["needs_review"] == 0
    action = store.corporate_actions("CACC")[0]
    assert action["action_type"] == ca.CASH_DIVIDEND and action["status"] == ca.APPLIED


def test_multiple_dividend_payments_accumulate(store, config):
    open_position(store, "CACC", price=100.0, shares=100)

    run(store, config, actions_source(CACC={"dividends": {"2026-09-03": 0.50,
                                                          "2026-09-10": 0.60}}))

    assert store.open_trade("CACC")["dividend_income"] == pytest.approx(110.0)
    assert len(store.corporate_actions("CACC")) == 2


def test_entitlement_uses_the_shares_held_before_the_ex_date(store, config):
    """A partial sell before the ex-date reduces the entitlement."""
    open_position(store, "CACC", price=100.0, shares=100, timestamp="2026-08-01T13:30:00+00:00")
    store.record_sell("CACC", 120.0, 40, timestamp="2026-09-05T13:30:00+00:00")

    run(store, config, actions_source(CACC={"dividends": {"2026-09-10": 1.0}}))

    assert store.open_trade("CACC")["dividend_income"] == pytest.approx(60.0)


def test_a_fill_on_the_ex_date_is_reviewed_not_guessed(store, config):
    open_position(store, "CACC", price=100.0, shares=100, timestamp="2026-08-01T13:30:00+00:00")
    store.record_buy("CACC", 105.0, 50, timestamp="2026-09-10T13:30:00+00:00")

    run(store, config, actions_source(CACC={"dividends": {"2026-09-10": 1.0}}))

    after = store.open_trade("CACC")
    assert after["dividend_income"] == 0                           # nothing was guessed
    assert after["needs_review"] == 1
    action = store.corporate_actions("CACC")[0]
    assert action["status"] == ca.REVIEW_REQUIRED
    assert "ex-date" in action["notes"]


def test_a_dividend_with_no_shares_held_before_the_ex_date_is_not_applicable(store, config):
    """Bought after the ex-date: nothing was entitled, so nothing is booked."""
    open_position(store, "CACC", price=100.0, shares=10, timestamp="2026-09-13T13:30:00+00:00")

    action = ca.make_action("CACC", "2026-09-12", ca.CASH_DIVIDEND, cash_amount=1.0)
    row, _ = store.record_corporate_action(action)
    ca.apply_cash_dividend(store, row, store.open_trade("CACC"))

    assert store.open_trade("CACC")["dividend_income"] == 0
    assert store.corporate_actions("CACC")[0]["status"] == ca.NOT_APPLICABLE


def test_entitlement_rebases_fills_across_an_intervening_split(store, config):
    """100 shares bought, then a 2-for-1 split, then a dividend: 200 shares are
    entitled even though the fill still says 100."""
    open_position(store, "CACC", price=100.0, shares=100, timestamp="2026-08-01T13:30:00+00:00")
    run(store, config, actions_source(CACC={"splits": {"2026-09-01": 2.0}}), trading_date="2026-09-02")
    run(store, config, actions_source(CACC={"splits": {"2026-09-01": 2.0},
                                            "dividends": {"2026-09-10": 0.25}}))

    assert store.open_trade("CACC")["shares"] == 200
    assert store.open_trade("CACC")["dividend_income"] == pytest.approx(50.0)


# ── §5 special dividends ──────────────────────────────────────────────────

def test_a_large_one_off_payment_is_classified_special(store, config):
    open_position(store, "CACC", price=100.0, shares=50)

    run(store, config, actions_source(CACC={"dividends": {"2026-09-10": 10.0}}))

    action = store.corporate_actions("CACC")[0]
    assert action["action_type"] == ca.SPECIAL_DIVIDEND
    assert action["status"] == ca.APPLIED
    assert store.open_trade("CACC")["dividend_income"] == pytest.approx(500.0)


def test_a_special_dividend_flags_the_position_instead_of_guessing_a_new_stop(store, config):
    open_position(store, "CACC", price=100.0, shares=50, stop=92.0, pivot=105.0)

    run(store, config, actions_source(CACC={"dividends": {"2026-09-10": 10.0}}))

    after = store.open_trade("CACC")
    assert after["needs_review"] == 1
    assert "special dividend" in after["review_reason"].lower()
    assert after["initial_stop"] == pytest.approx(92.0)             # NOT silently moved
    assert after["pivot_price"] == pytest.approx(105.0)
    assert after["average_cost"] == pytest.approx(100.0)


def test_a_payment_far_above_the_regular_cadence_is_special(config):
    special, why = ca.is_special_dividend(3.0, reference_price=1000.0,
                                          regular_amounts=[0.5, 0.5, 0.6], config=config)
    assert special and "median" in why
    ordinary, _ = ca.is_special_dividend(0.6, reference_price=1000.0,
                                         regular_amounts=[0.5, 0.5, 0.6], config=config)
    assert not ordinary


# ── §7 ticker changes ─────────────────────────────────────────────────────

def test_symbol_change_preserves_trade_identity(store, config):
    trade = open_position(store, "ABC", price=50.0, shares=100, stop=45.0)
    store.record_sell("ABC", 60.0, 20, timestamp="2026-09-05T13:30:00+00:00")
    events = {"events": [{"action_type": "SYMBOL_CHANGE", "effective_date": "2026-09-10",
                          "old_symbol": "ABC", "new_symbol": "XYZ"}]}

    run(store, config, actions_source(ABC=events))

    after = store.get_trade(trade["trade_id"])
    assert after["symbol"] == "XYZ"
    assert after["status"] == ts.OPEN                               # not closed and reopened
    assert after["trade_id"] == trade["trade_id"]
    assert after["opened_at"] == trade["opened_at"]
    assert after["shares"] == 80 and after["average_cost"] == pytest.approx(50.0)
    assert after["realized_pnl"] == pytest.approx(200.0)
    assert len(store.fills(trade["trade_id"])) == 2
    assert store.open_trade("XYZ")["trade_id"] == trade["trade_id"]
    assert store.open_trade("ABC") is None


def test_symbol_change_keeps_an_auditable_history(store, config):
    import json

    trade = open_position(store, "ABC", price=50.0, shares=100)
    run(store, config, actions_source(ABC={"events": [
        {"action_type": "SYMBOL_CHANGE", "effective_date": "2026-09-10", "new_symbol": "XYZ"}]}))

    history = json.loads(store.get_trade(trade["trade_id"])["symbol_history_json"])
    assert [h["symbol"] for h in history] == ["ABC", "XYZ"]


def test_symbol_change_into_an_existing_position_is_reviewed_not_merged(store, config):
    open_position(store, "ABC", price=50.0, shares=100)
    open_position(store, "XYZ", price=30.0, shares=10)

    run(store, config, actions_source(ABC={"events": [
        {"action_type": "SYMBOL_CHANGE", "effective_date": "2026-09-10", "new_symbol": "XYZ"}]}))

    assert store.open_trade("ABC") is not None                      # untouched
    assert store.open_trade("ABC")["needs_review"] == 1
    assert store.corporate_actions("ABC")[0]["status"] == ca.REVIEW_REQUIRED


# ── §8 mergers, spin-offs, rights ─────────────────────────────────────────

@pytest.mark.parametrize("action_type", ["SPINOFF", "RIGHTS", "MERGER"])
def test_complex_actions_become_manual_review(store, config, action_type):
    trade = open_position(store, "CACC", price=100.0, shares=50, stop=92.0)

    run(store, config, actions_source(CACC={"events": [
        {"action_type": action_type, "effective_date": "2026-09-10"}]}))

    after = store.get_trade(trade["trade_id"])
    assert after["needs_review"] == 1
    assert after["shares"] == 50                                    # nothing invented
    assert after["average_cost"] == pytest.approx(100.0)
    assert after["initial_stop"] == pytest.approx(92.0)
    assert store.corporate_actions("CACC")[0]["status"] == ca.REVIEW_REQUIRED


def test_a_stock_for_stock_merger_never_produces_an_exchange_ratio(store, config):
    open_position(store, "CACC", price=100.0, shares=50)

    run(store, config, actions_source(CACC={"events": [
        {"action_type": "MERGER", "effective_date": "2026-09-10",
         "details": {"stock_component": "0.75 ACME per CACC"}}]}))

    action = store.corporate_actions("CACC")[0]
    assert action["status"] == ca.REVIEW_REQUIRED                   # not AWAITING_CONFIRMATION
    assert store.open_trade("CACC")["shares"] == 50


def test_a_cash_acquisition_with_final_terms_waits_for_the_user(store, config):
    open_position(store, "CACC", price=100.0, shares=50)

    run(store, config, actions_source(CACC={"events": [
        {"action_type": "MERGER", "effective_date": "2026-09-10", "cash_amount": 125.0}]}))

    action = store.corporate_actions("CACC")[0]
    assert action["status"] == ca.AWAITING_CONFIRMATION
    assert action["cash_amount"] == pytest.approx(125.0)
    assert store.open_trade("CACC")["status"] == ts.OPEN            # never auto-closed
    assert store.open_trade("CACC")["needs_review"] == 1


# ── §2 idempotency ────────────────────────────────────────────────────────

def test_the_same_event_is_never_applied_twice(store, config):
    open_position(store, "CACC", price=620.0, shares=20, stop=590.0)
    fetch = actions_source(CACC={"splits": {"2026-09-10": 2.0},
                                 "dividends": {"2026-09-11": 1.0}})

    run(store, config, fetch)
    first = store.open_trade("CACC")
    for _ in range(3):
        run(store, config, fetch, force=True)                       # re-probe every time

    after = store.open_trade("CACC")
    assert after["shares"] == first["shares"] == 40
    assert after["average_cost"] == pytest.approx(first["average_cost"])
    assert after["dividend_income"] == pytest.approx(first["dividend_income"])
    assert len(store.corporate_actions("CACC")) == 2


def test_recording_the_same_action_twice_creates_one_row(store):
    action = ca.make_action("CACC", "2026-09-10", ca.SPLIT, split_ratio=2.0)
    first, created_first = store.record_corporate_action(action)
    second, created_second = store.record_corporate_action(dict(action))

    assert created_first and not created_second
    assert first["action_id"] == second["action_id"]
    assert len(store.corporate_actions("CACC")) == 1


def test_a_second_apply_of_a_claimed_action_writes_nothing(store):
    trade = open_position(store, "CACC", price=620.0, shares=20)
    row, _ = store.record_corporate_action(
        ca.make_action("CACC", "2026-09-10", ca.SPLIT, split_ratio=2.0))

    assert ca.apply_share_basis(store, row, trade)["applied"] is True
    # Replaying the identical call (the stale DETECTED status) must be a no-op.
    assert ca.apply_share_basis(store, row, store.open_trade("CACC"))["applied"] is False
    assert store.open_trade("CACC")["shares"] == 40


def test_one_probe_per_symbol_per_session(store, config):
    open_position(store, "CACC")
    fetch = actions_source(CACC={"splits": {}})

    run(store, config, fetch)
    run(store, config, fetch)
    run(store, config, fetch)

    assert len(fetch.calls) == 1
    assert store.symbol_checked("CACC", SESSION)
    run(store, config, fetch, trading_date="2026-09-16")
    assert len(fetch.calls) == 2


def test_a_fingerprint_is_stable_and_type_specific():
    a = ca.fingerprint("CACC", "2026-09-10", ca.SPLIT, 2.0)
    assert a == ca.fingerprint("cacc", "2026-09-10", ca.SPLIT, 2.0)
    assert a != ca.fingerprint("CACC", "2026-09-10", ca.SPLIT, 3.0)
    assert a != ca.fingerprint("CACC", "2026-09-11", ca.SPLIT, 2.0)
    assert a != ca.fingerprint("CACC", "2026-09-10", ca.STOCK_DIVIDEND, 2.0)


# ── detection boundaries ──────────────────────────────────────────────────

def test_actions_before_the_entry_date_are_ignored(store, config):
    """They are already inside the price the user paid."""
    open_position(store, "CACC", price=620.0, shares=20, timestamp="2026-09-01T13:30:00+00:00")

    run(store, config, actions_source(CACC={"splits": {"2026-08-01": 2.0},
                                            "dividends": {"2026-08-15": 1.0}}))

    assert store.corporate_actions("CACC") == []
    assert store.open_trade("CACC")["shares"] == 20


def test_actions_after_the_session_are_ignored(store, config):
    open_position(store, "CACC", price=620.0, shares=20)
    run(store, config, actions_source(CACC={"splits": {"2026-12-01": 2.0}}))
    assert store.corporate_actions("CACC") == []


def test_a_closed_position_is_not_probed(store, config):
    open_position(store, "CACC", price=620.0, shares=20)
    store.close_position("CACC", 700.0)
    fetch = actions_source(CACC={"splits": {"2026-09-10": 2.0}})

    assert run(store, config, fetch) == []
    assert fetch.calls == []


def test_a_broken_lookup_never_stops_the_other_positions(store, config):
    open_position(store, "AAA", price=100.0, shares=10)
    open_position(store, "BBB", price=100.0, shares=10)

    def fetch(symbol, start=None, config=None):
        if symbol == "AAA":
            raise RuntimeError("provider exploded")
        return {"splits": {"2026-09-10": 2.0}}

    run(store, config, fetch)

    assert store.open_trade("AAA")["shares"] == 10                 # untouched
    assert store.open_trade("BBB")["shares"] == 20                 # still processed


def test_processing_can_be_switched_off(store, tmp_path):
    open_position(store, "CACC")
    config = trade_config(tmp_path, trade_corporate_actions_enabled=False)
    fetch = actions_source(CACC={"splits": {"2026-09-10": 2.0}})

    assert run(store, config, fetch) == []
    assert fetch.calls == []


# ── pure arithmetic helpers ───────────────────────────────────────────────

def test_classify_split_and_describe_ratio():
    assert ca.classify_split(2.0) == ca.SPLIT
    assert ca.classify_split(0.2) == ca.REVERSE_SPLIT
    assert ca.describe_ratio(2.0) == "2-for-1"
    assert ca.describe_ratio(0.2) == "1-for-5"
    for bad in (0, -1, None):
        with pytest.raises(ca.CorporateActionError):
            ca.classify_split(bad)


def test_basis_factor_composes_and_is_window_bounded():
    actions = [
        dict(ca.make_action("X", "2026-03-01", ca.SPLIT, split_ratio=2.0), status=ca.APPLIED),
        dict(ca.make_action("X", "2026-06-01", ca.STOCK_DIVIDEND, split_ratio=1.1), status=ca.APPLIED),
        dict(ca.make_action("X", "2026-07-01", ca.CASH_DIVIDEND, cash_amount=1.0), status=ca.APPLIED),
    ]
    assert ca.basis_factor(actions) == pytest.approx(2.2)
    assert ca.basis_factor(actions, after="2026-03-01") == pytest.approx(1.1)
    assert ca.basis_factor(actions, after="2026-06-01") == 1.0
    assert ca.rebase_price(220.0, actions, after="2026-01-01") == pytest.approx(100.0)


def test_only_applied_share_basis_actions_rebase_anything():
    pending = [dict(ca.make_action("X", "2026-03-01", ca.SPLIT, split_ratio=2.0),
                    status=ca.REVIEW_REQUIRED)]
    assert ca.basis_factor(pending) == 1.0


def test_review_resolution_releases_the_position(store, config):
    open_position(store, "CACC", price=100.0, shares=50)
    run(store, config, actions_source(CACC={"events": [
        {"action_type": "SPINOFF", "effective_date": "2026-09-10"}]}))
    trade = store.open_trade("CACC")
    assert trade["needs_review"] == 1

    resolved = ca.resolve_reviews(store, trade, notes="allocated by hand")

    assert [a["action_type"] for a in resolved] == ["SPINOFF"]
    assert store.open_trade("CACC")["needs_review"] == 0
    assert store.corporate_actions("CACC")[0]["status"] == ca.RESOLVED
