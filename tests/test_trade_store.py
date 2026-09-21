"""Stage I ledger: partial fills, average cost, realised P&L, snapshots and the
pending-confirmation guard. Temporary SQLite files only; no network anywhere."""
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import trade_store as ts
from pipeline.trade_store import TradeError

T0 = datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return ts.TradeStore(str(tmp_path / "trades.sqlite3"))


def test_a_buy_opens_a_position(store):
    trade_id, opened_new = store.record_buy("cacc", 622.50, 20, timestamp=T0,
                                            setup={"setup_run_id": "run1", "setup_date": "2026-09-15",
                                                   "setup_state": "READY", "vcp_quality": "strong",
                                                   "pivot_price": 621.52},
                                            initial_stop=594.20, portfolio_portion_pct=8.0)
    trade = store.get_trade(trade_id)
    assert opened_new is True
    assert (trade["symbol"], trade["status"], trade["shares"], trade["average_cost"]) == \
        ("CACC", ts.OPEN, 20.0, 622.50)
    assert trade["initial_stop"] == 594.20 and trade["pivot_price"] == 621.52
    assert trade["setup_run_id"] == "run1" and trade["setup_state"] == "READY"
    assert [f["side"] for f in store.fills(trade_id)] == ["BUY"]
    assert [e["event_type"] for e in store.events(trade_id)] == [ts.EVENT_OPEN]


def test_multiple_buy_fills_re_average_the_cost(store):
    store.record_buy("CACC", 100.0, 10, timestamp=T0)
    trade_id, opened_new = store.record_buy("CACC", 120.0, 10, timestamp=T0 + timedelta(days=1))
    trade = store.get_trade(trade_id)
    assert opened_new is False
    assert trade["shares"] == 20.0
    assert trade["average_cost"] == 110.0
    assert trade["total_bought_shares"] == 20.0
    assert len(store.fills(trade_id)) == 2
    assert len(store.open_positions()) == 1


def test_uneven_fills_average_by_share_count(store):
    store.record_buy("HALO", 50.0, 30, timestamp=T0)
    trade_id, _ = store.record_buy("HALO", 60.0, 10, timestamp=T0)
    assert store.get_trade(trade_id)["average_cost"] == pytest.approx((30 * 50 + 10 * 60) / 40)


def test_partial_sell_books_realised_pnl_and_keeps_the_average_cost(store):
    trade_id, _ = store.record_buy("CACC", 100.0, 20, timestamp=T0)
    result = store.record_sell("CACC", 130.0, 8, timestamp=T0 + timedelta(days=5))
    trade = store.get_trade(trade_id)
    assert result["shares_sold"] == 8.0 and result["shares_remaining"] == 12.0
    assert result["realized_pnl"] == pytest.approx(240.0)
    assert result["closed"] is False
    assert trade["status"] == ts.OPEN
    assert trade["shares"] == 12.0
    assert trade["average_cost"] == 100.0            # unchanged by a sell
    assert trade["realized_pnl"] == pytest.approx(240.0)
    assert [e["event_type"] for e in store.events(trade_id)][0] == ts.EVENT_SELL


def test_selling_the_rest_closes_the_position(store):
    trade_id, _ = store.record_buy("CACC", 100.0, 20, timestamp=T0)
    store.record_sell("CACC", 130.0, 8, timestamp=T0)
    result = store.record_sell("CACC", 140.0, 12, timestamp=T0 + timedelta(days=6))
    trade = store.get_trade(trade_id)
    assert result["closed"] is True
    assert trade["status"] == ts.CLOSED and trade["shares"] == 0.0
    assert trade["closed_at"] is not None
    assert trade["realized_pnl"] == pytest.approx(240.0 + 480.0)
    assert store.open_positions() == []


def test_close_position_sells_everything_held(store):
    store.record_buy("CACC", 100.0, 15, timestamp=T0)
    result = store.close_position("CACC", 105.0, timestamp=T0 + timedelta(days=2))
    assert result["shares_sold"] == 15.0 and result["closed"] is True
    assert store.get_trade(result["trade_id"])["status"] == ts.CLOSED


def test_overselling_is_rejected_and_writes_nothing(store):
    trade_id, _ = store.record_buy("CACC", 100.0, 10, timestamp=T0)
    with pytest.raises(TradeError, match="only 10 held"):
        store.record_sell("CACC", 120.0, 11, timestamp=T0)
    assert store.get_trade(trade_id)["shares"] == 10.0
    assert len(store.fills(trade_id)) == 1


def test_selling_without_an_open_position_is_rejected(store):
    with pytest.raises(TradeError, match="no open position"):
        store.record_sell("NVDA", 100.0, 1, timestamp=T0)


def test_non_positive_quantities_are_rejected(store):
    with pytest.raises(TradeError):
        store.record_buy("CACC", 100.0, 0, timestamp=T0)
    with pytest.raises(TradeError):
        store.record_buy("CACC", -1.0, 5, timestamp=T0)


def test_a_symbol_can_be_traded_again_after_closing(store):
    first, _ = store.record_buy("CACC", 100.0, 10, timestamp=T0)
    store.close_position("CACC", 110.0, timestamp=T0)
    second, opened_new = store.record_buy("CACC", 120.0, 5, timestamp=T0 + timedelta(days=30))
    assert opened_new is True and second != first
    assert len(store.trades_for_symbol("CACC")) == 2
    assert len(store.open_positions()) == 1


def test_snapshots_are_upserted_per_session(store):
    trade_id, _ = store.record_buy("CACC", 100.0, 10, timestamp=T0)
    base = {"trade_id": trade_id, "symbol": "CACC", "trading_date": "2026-09-15", "close": 105.0,
            "monitor_state": "HOLD", "reasons": ["trend structure intact"]}
    store.save_snapshot(base)
    store.save_snapshot({**base, "close": 106.0, "monitor_state": "WATCH"})
    rows = store.snapshots_for_date("2026-09-15")
    assert len(rows) == 1
    assert (rows[0]["close"], rows[0]["monitor_state"]) == (106.0, "WATCH")
    store.save_snapshot({**base, "trading_date": "2026-09-16", "close": 99.0})
    assert store.latest_snapshot(trade_id)["trading_date"] == "2026-09-16"
    assert store.latest_snapshot(trade_id, before="2026-09-16")["trading_date"] == "2026-09-15"


def test_pending_action_can_only_be_claimed_once(store):
    token = store.create_pending("buy", "12345", {"symbol": "CACC", "price": 10.0})
    first = store.claim_pending(token, ts.CONFIRMED)
    second = store.claim_pending(token, ts.CONFIRMED)
    assert first is not None and first["payload"]["symbol"] == "CACC"
    assert second is None                                   # duplicate callback protection
    assert store.get_pending(token)["status"] == ts.CONFIRMED


def test_cancelled_pending_action_cannot_later_be_confirmed(store):
    token = store.create_pending("buy", "12345", {"symbol": "CACC"})
    assert store.claim_pending(token, ts.CANCELLED) is not None
    assert store.claim_pending(token, ts.CONFIRMED) is None


def test_expired_pending_actions_are_swept(store):
    now = T0
    store.create_pending("buy", "1", {"symbol": "A"}, expires_at=now - timedelta(minutes=1))
    live = store.create_pending("buy", "1", {"symbol": "B"}, expires_at=now + timedelta(minutes=30))
    assert store.expire_pending(now) == 1
    assert [p["token"] for p in store.pending_for_chat("1")] == [live]


def test_open_store_does_not_create_the_database_when_asked_not_to(tmp_path):
    config = {"trade_db_path": str(tmp_path / "nothing" / "trades.sqlite3")}
    assert ts.open_store(config, create=False) is None
    assert ts.open_store(config, create=True) is not None
