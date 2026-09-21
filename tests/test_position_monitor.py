"""Stage I Layer 3 (ongoing): position snapshots, protective levels and the
winning-exit state machine.

Synthetic close paths only — the monitor's price fetch is injected, so nothing
here touches yfinance, a broker or Telegram.
"""
import pytest

from pipeline import position_monitor as pm
from pipeline import risk_model as rm
from pipeline import trade_store as ts
from stage_i_fakes import make_bars, trade_config

SESSION = "2026-09-15"
RISE = [100 + i * 0.8 for i in range(40)]           # steady advance to 131.20


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


@pytest.fixture
def store(config):
    return ts.open_store(config, create=True)


def trade(**overrides):
    base = {"trade_id": "t1", "symbol": "CACC", "shares": 10.0, "average_cost": 128.0,
            "initial_stop": 115.0, "pivot_price": None, "opened_at": "2026-01-02T00:00:00+00:00"}
    base.update(overrides)
    return base


def snapshot(closes, config, *, volumes=None, previous=None, market_cap=5e9, **trade_overrides):
    bars = make_bars(closes, volumes=volumes)
    return pm.build_snapshot(trade(**trade_overrides), bars, SESSION, previous_snapshot=previous,
                             market_cap=market_cap, config=config)


# ── the numbers (§9) ──────────────────────────────────────────────────────

def test_snapshot_computes_every_monitored_number(config):
    snap = snapshot(RISE, config, average_cost=100.0, initial_stop=90.0)
    assert snap["close"] == 131.2
    assert snap["pnl_pct"] == 31.2
    assert snap["highest_since_entry"] == pytest.approx(132.51, abs=0.01)
    assert snap["mfe_pct"] > snap["pnl_pct"] > 0
    assert snap["mae_pct"] < 0                                   # the entry bar's low
    assert snap["atr20"] is not None and snap["atr_pct"] is not None
    assert snap["ema10"] and snap["ema21"] and snap["sma50"] is None   # only 40 bars
    assert snap["days_since_entry"] == 39
    assert snap["distance_to_initial_stop_pct"] > 0
    assert snap["metrics"]["avg_volume_20d"] is not None


def test_mfe_and_mae_are_measured_from_the_entry_date_only(config):
    bars = make_bars([50.0] * 20 + RISE)                          # a cheap pre-entry stretch
    entry = bars.index[21].strftime("%Y-%m-%dT00:00:00+00:00")
    snap = pm.build_snapshot(trade(average_cost=100.0, opened_at=entry), bars, SESSION,
                             market_cap=5e9, config=config)
    assert snap["mae_pct"] > -10                                  # the 50.0 stretch is excluded
    assert snap["days_since_entry"] == 38


def test_swing_low_ignores_the_most_recent_sessions(config):
    """Today's own low is always below today's close, so a swing low that
    included it could never be broken."""
    bars = make_bars(RISE + [110.0, 108.0])
    swing = pm.structural_swing_low(bars, 20, 3)
    assert swing > float(bars["Close"].iloc[-1])                  # today's drop breaks it
    assert rm.recent_swing_low(bars, 20) <= float(bars["Low"].iloc[-1])


# ── protective level (§11) ────────────────────────────────────────────────

def test_without_a_cushion_the_protective_level_is_the_initial_stop(config):
    snap = snapshot(RISE, config, average_cost=128.0, initial_stop=115.0)
    assert snap["current_protective_level"] == 115.0
    assert snap["protection_raised"] is False


def test_a_profit_cushion_ratchets_protection_up(config):
    snap = snapshot(RISE, config, average_cost=100.0, initial_stop=95.0)
    assert snap["current_protective_level"] > 95.0
    assert snap["protection_raised"] is True


def test_protection_never_moves_down(config):
    snap = snapshot(RISE, config, average_cost=100.0, initial_stop=95.0,
                    previous={"current_protective_level": 130.0, "monitor_state": "HOLD"})
    assert snap["current_protective_level"] == 130.0
    assert snap["protection_raised"] is False


def test_protection_is_never_tighter_than_the_atr_allowance(config):
    """A big winner's level stays a volatility-sized distance below the close."""
    snap = snapshot(RISE, config, average_cost=100.0, initial_stop=95.0)
    atr_room = snap["metrics"]["trail_atr_multiple"] * snap["atr20"]
    assert snap["current_protective_level"] <= snap["close"] - atr_room + 1e-6


def test_higher_volatility_leaves_more_room_than_a_calm_large_cap(config):
    calm = snapshot(RISE, config, average_cost=100.0, initial_stop=95.0, market_cap=300e9)
    wild = snapshot(RISE, config, average_cost=100.0, initial_stop=95.0, market_cap=5e9)
    assert wild["metrics"]["trail_atr_multiple"] > calm["metrics"]["trail_atr_multiple"]
    assert wild["current_protective_level"] < calm["current_protective_level"]


# ── states (§10) ──────────────────────────────────────────────────────────

def test_hold_while_the_trend_is_intact(config):
    assert snapshot(RISE, config)["monitor_state"] == pm.HOLD


def test_watch_below_ema10_with_structure_intact(config):
    snap = snapshot(RISE + [126.0], config, average_cost=124.0, initial_stop=110.0)
    assert snap["monitor_state"] == pm.WATCH
    assert "below EMA10" in snap["reasons"][0]
    assert "above EMA21" in snap["reasons"][0]


def test_tighten_protection_when_a_winner_loses_ema10(config):
    snap = snapshot(RISE + [126.0], config, average_cost=100.0, initial_stop=110.0)
    assert snap["monitor_state"] == pm.TIGHTEN_PROTECTION
    assert any("tighten protection" in r for r in snap["reasons"])


def test_partial_profit_review_only_on_a_climactic_extension(config):
    climax = snapshot(RISE + [160.0], config, average_cost=100.0, initial_stop=95.0)
    assert climax["monitor_state"] == pm.PARTIAL_PROFIT_REVIEW
    assert "no fixed profit target" in climax["reasons"][0]
    # The same cushion without the extension is NOT a profit-taking signal:
    # there is deliberately no universal "+20% -> sell" rule.
    assert snapshot(RISE, config, average_cost=100.0, initial_stop=95.0)["monitor_state"] != \
        pm.PARTIAL_PROFIT_REVIEW


def test_structure_break_is_an_exit_review(config):
    snap = snapshot(RISE + [110.0, 108.0], config, average_cost=106.0, initial_stop=95.0)
    assert snap["monitor_state"] == pm.EXIT_REVIEW
    assert "structure break" in snap["reasons"][0]


def test_failed_breakout_shortly_after_entry_is_an_exit_review(config):
    closes = RISE + [96.0]
    bars = make_bars(closes)
    entry = bars.index[-6].strftime("%Y-%m-%dT00:00:00+00:00")     # bought 5 sessions ago
    snap = pm.build_snapshot(trade(average_cost=131.0, initial_stop=90.0, pivot_price=130.0,
                                   opened_at=entry), bars, SESSION, market_cap=5e9, config=config)
    assert snap["monitor_state"] == pm.EXIT_REVIEW
    assert any("failed breakout" in r for r in snap["reasons"])


def test_abnormal_high_volume_reversal_below_ema10_is_an_exit_review(config):
    volumes = [1e6] * len(RISE) + [4e6]
    snap = snapshot(RISE + [126.0], config, volumes=volumes, average_cost=124.0, initial_stop=110.0)
    assert snap["monitor_state"] == pm.EXIT_REVIEW
    assert "reversal volume" in snap["reasons"][0]
    assert snap["metrics"]["high_volume_reversal"] is True


def test_the_same_reversal_volume_on_an_ordinary_day_is_not_an_exit(config):
    volumes = [1e6] * len(RISE) + [4e6]
    quiet = snapshot(RISE + [132.0], config, volumes=volumes, average_cost=124.0, initial_stop=110.0)
    assert quiet["monitor_state"] != pm.EXIT_REVIEW


def test_stop_triggered_when_the_close_breaches_the_protective_level(config):
    snap = snapshot([100.0] * 30 + [94.0], config, average_cost=100.0, initial_stop=95.0)
    assert snap["monitor_state"] == pm.STOP_TRIGGERED
    assert "at/below the protective level" in snap["reasons"][0]


def test_stop_triggered_beats_every_other_state(config):
    snap = snapshot(RISE + [80.0], config, average_cost=100.0, initial_stop=95.0,
                    volumes=[1e6] * len(RISE) + [9e6])
    assert snap["monitor_state"] == pm.STOP_TRIGGERED


def test_no_price_history_degrades_without_raising(config):
    snap = pm.build_snapshot(trade(), None, SESSION, config=config)
    assert snap["monitor_state"] == pm.HOLD
    assert snap["close"] is None
    assert "no price history" in snap["reasons"][0]


# ── the daily run ─────────────────────────────────────────────────────────

def test_open_positions_are_monitored_even_when_absent_from_todays_screen(config, store):
    """The screener found nothing today; the held stock is still monitored."""
    store.record_buy("HALO", 50.0, 10, initial_stop=45.0)
    asked = []

    def fetch(symbols, cfg=None):
        asked.extend(symbols)
        return {s: make_bars(RISE) for s in symbols}

    snapshots = pm.monitor_open_positions(SESSION, config, store=store, fetch=fetch)
    assert asked == ["HALO"]                    # driven by the ledger, not by today's results
    assert [s["symbol"] for s in snapshots] == ["HALO"]
    stored = store.snapshots_for_date(SESSION)
    assert len(stored) == 1 and stored[0]["monitor_state"] == snapshots[0]["monitor_state"]


def test_monitor_records_an_event_only_when_the_state_changes(config, store):
    trade_id, _ = store.record_buy("CACC", 128.0, 10, initial_stop=115.0)
    fetch = lambda symbols, cfg=None: {s: make_bars(RISE) for s in symbols}

    pm.monitor_open_positions(SESSION, config, store=store, fetch=fetch)
    pm.monitor_open_positions("2026-09-16", config, store=store, fetch=fetch)
    monitor_events = [e for e in store.events(trade_id) if e["event_type"] == ts.EVENT_MONITOR]
    assert len(monitor_events) == 1 and monitor_events[0]["reason"] == pm.HOLD


def test_monitor_carries_the_protective_level_forward(config, store):
    store.record_buy("CACC", 100.0, 10, initial_stop=95.0)
    fetch_high = lambda symbols, cfg=None: {s: make_bars(RISE) for s in symbols}
    first = pm.monitor_open_positions(SESSION, config, store=store, fetch=fetch_high)[0]

    fetch_low = lambda symbols, cfg=None: {s: make_bars(RISE + [126.0]) for s in symbols}
    second = pm.monitor_open_positions("2026-09-16", config, store=store, fetch=fetch_low)[0]
    assert second["current_protective_level"] >= first["current_protective_level"]


def test_nothing_held_means_nothing_fetched(config, store):
    def fetch(symbols, cfg=None):
        raise AssertionError("must not download anything with no open positions")
    assert pm.monitor_open_positions(SESSION, config, store=store, fetch=fetch) == []


def test_a_download_failure_never_raises(config, store):
    store.record_buy("CACC", 100.0, 10)

    def boom(symbols, cfg=None):
        raise RuntimeError("provider down")
    assert pm.monitor_open_positions(SESSION, config, store=store, fetch=boom) == []


def test_one_bad_symbol_does_not_stop_the_others(config, store):
    store.record_buy("AAA", 100.0, 10, initial_stop=90.0)
    store.record_buy("BBB", 100.0, 10, initial_stop=90.0)
    fetch = lambda symbols, cfg=None: {"AAA": make_bars(RISE), "BBB": None}
    snapshots = pm.monitor_open_positions(SESSION, config, store=store, fetch=fetch)
    assert {s["symbol"] for s in snapshots} == {"AAA", "BBB"}
    assert [s for s in snapshots if s["symbol"] == "BBB"][0]["close"] is None
