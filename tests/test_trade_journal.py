"""Stage I service layer: gating a run into persisted plans, and the
preview/apply split that keeps unconfirmed mutations out of SQLite."""
import pytest

from pipeline import risk_model as rm
from pipeline import setup_gate as gate
from pipeline import trade_journal as tj
from pipeline import trade_store as ts
from stage_i_fakes import SESSION, RUN_ID, flat_bars, non_stage_g_row, stage_g_row, trade_config


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


@pytest.fixture
def store(config):
    return ts.open_store(config, create=True)


def bars_fetch(recorded=None, price=100.0):
    def fetch(symbols, cfg=None):
        if recorded is not None:
            recorded.extend(symbols)
        return {s: flat_bars(60, price) for s in symbols}
    return fetch


def test_generate_setup_plans_records_every_stage_g_symbol(config, store):
    results = [
        stage_g_row("AAA", quality="strong", pivot_state="coiling_below_pivot", rank=1),
        stage_g_row("BBB", quality="weak", rank=2),
        stage_g_row("CCC", claude=False, rank=3),
        non_stage_g_row("ZZZZ", rank=4),
    ]
    fetched = []
    tj.generate_setup_plans(RUN_ID, SESSION, results, config, store=store, fetch=bars_fetch(fetched))

    rows = {p["symbol"]: p for p in store.setup_plans_for_date(SESSION)}
    assert set(rows) == {"AAA", "BBB", "CCC"}                    # ZZZZ was never analysed by Stage G
    assert rows["AAA"]["setup_state"] == gate.READY
    assert rows["BBB"]["setup_state"] == gate.NOT_READY
    assert rows["CCC"]["setup_state"] == gate.MANUAL_REVIEW
    assert fetched == ["AAA"]                                    # bars only for plannable setups


def test_only_plannable_setups_get_layer_2_and_layer_3_plans(config, store):
    results = [stage_g_row("AAA", pivot=100.0, price=98.0, market_cap=5e9),
               stage_g_row("BBB", quality="weak", rank=2)]
    tj.generate_setup_plans(RUN_ID, SESSION, results, config, store=store, fetch=bars_fetch())

    ready = store.latest_setup_plan("AAA")
    blocked = store.latest_setup_plan("BBB")
    assert ready["entry_status"] == "AWAIT_BREAKOUT"
    assert ready["entry_trigger_price"] == 100.1
    assert ready["suggested_initial_stop"] is not None
    assert ready["suggested_shares"] > 0
    assert ready["entry_plan"]["pivot_source"] == "numeric_vcp_detector"
    assert blocked["entry_status"] is None and blocked["suggested_initial_stop"] is None


def test_plan_persists_the_risk_and_sizing_payloads(config, store):
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("AAA", market_cap=5e9)], config,
                            store=store, fetch=bars_fetch())
    plan = store.latest_setup_plan("AAA")
    assert plan["risk_plan"]["risk_basis"] in (rm.STRUCTURE, rm.VOLATILITY_FLOOR, rm.VOLATILITY_ONLY)
    assert plan["sizing"]["binding_constraint"] in ("risk", "portion")
    assert plan["trade_plan_status"] in (rm.OK, rm.SKIP_RISK_TOO_WIDE)


def test_a_failed_price_download_still_records_the_gate_states(config, store):
    def boom(symbols, cfg=None):
        raise RuntimeError("provider down")
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("AAA")], config, store=store, fetch=boom)
    plan = store.latest_setup_plan("AAA")
    assert plan["setup_state"] == gate.READY
    assert plan["trade_plan_status"] == rm.NO_PRICE_DATA


def test_preview_buy_attaches_the_latest_setup_and_writes_nothing(config, store):
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("CACC", pivot=621.52, price=610.0,
                                                          market_cap=5e9)],
                            config, store=store, fetch=bars_fetch(price=610.0))
    payload = tj.preview_buy("cacc", 622.50, 20, config, store=store)

    assert payload["symbol"] == "CACC" and payload["value"] == 12450.0
    assert payload["setup"]["setup_state"] == gate.READY
    assert payload["setup"]["pivot_price"] == 621.52
    assert payload["suggested_initial_stop"] == store.latest_setup_plan("CACC")["suggested_initial_stop"]
    assert payload["portfolio_portion_pct"] == pytest.approx(12.45)
    assert store.open_positions() == []                          # preview never writes


def test_preview_buy_warns_when_there_is_no_recorded_setup(config, store):
    payload = tj.preview_buy("NVDA", 100.0, 10, config, store=store)
    assert payload["setup"] is None
    assert any("no recorded Stage-G setup" in w for w in payload["warnings"])


def test_preview_buy_warns_when_the_setup_is_not_plannable(config, store):
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("BBB", quality="weak")], config,
                            store=store, fetch=bars_fetch())
    payload = tj.preview_buy("BBB", 50.0, 10, config, store=store)
    assert any("NOT_READY" in w for w in payload["warnings"])


def test_apply_buy_links_the_trade_to_its_setup(config, store):
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row("CACC", pivot=621.52, price=610.0,
                                                          market_cap=5e9)],
                            config, store=store, fetch=bars_fetch(price=610.0))
    payload = tj.preview_buy("CACC", 622.50, 20, config, store=store)
    result = tj.apply_buy(payload, config, store=store)

    trade = result["trade"]
    assert result["opened_new"] is True
    assert trade["setup_run_id"] == RUN_ID and trade["setup_date"] == SESSION
    assert trade["vcp_quality"] == "strong" and trade["setup_state"] == gate.READY
    assert trade["initial_stop"] == payload["suggested_initial_stop"]
    assert trade["portfolio_portion_pct"] == pytest.approx(12.45)


def test_preview_buy_flags_an_addition_to_an_existing_position(config, store):
    store.record_buy("CACC", 600.0, 10)
    payload = tj.preview_buy("CACC", 622.50, 5, config, store=store)
    assert payload["adds_to_existing"] is True and payload["existing_shares"] == 10.0
    assert any("adds to an existing position" in w for w in payload["warnings"])


def test_preview_sell_computes_the_realised_result_without_writing(config, store):
    store.record_buy("CACC", 600.0, 20)
    payload = tj.preview_sell("CACC", 680.0, 10, config, store=store)
    assert payload["action"] == "sell" and payload["closes_position"] is False
    assert payload["realized_pnl"] == 800.0
    assert payload["shares_remaining"] == 10.0
    assert store.open_trade("CACC")["shares"] == 20.0            # untouched


def test_preview_close_covers_everything_held(config, store):
    store.record_buy("CACC", 600.0, 20)
    payload = tj.preview_close("CACC", 705.0, config, store=store)
    assert payload["action"] == "close" and payload["shares"] == 20.0
    assert payload["closes_position"] is True


def test_apply_sell_records_the_fill(config, store):
    store.record_buy("CACC", 600.0, 20)
    payload = tj.preview_sell("CACC", 680.0, 10, config, store=store)
    result = tj.apply_sell(payload, config, store=store)
    assert result["shares_remaining"] == 10.0
    assert store.open_trade("CACC")["shares"] == 10.0


def test_positions_view_joins_the_latest_snapshot(config, store):
    trade_id, _ = store.record_buy("CACC", 600.0, 20)
    store.save_snapshot({"trade_id": trade_id, "symbol": "CACC", "trading_date": SESSION,
                         "close": 650.0, "monitor_state": "HOLD"})
    view = tj.positions_view(config, store=store)[0]
    assert view["trade"]["symbol"] == "CACC"
    assert view["snapshot"]["close"] == 650.0


def test_position_detail_returns_fills_and_events(config, store):
    store.record_buy("CACC", 600.0, 20)
    detail = tj.position_detail("cacc", config, store=store)
    assert detail["trade"]["symbol"] == "CACC"
    assert len(detail["fills"]) == 1 and detail["events"]
