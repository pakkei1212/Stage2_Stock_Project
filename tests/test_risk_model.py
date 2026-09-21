"""Stage I Layer 3: ATR, structural stops, the risk ceiling and position sizing.

All thresholds asserted here are OUR configurable heuristics, not rules from
any book — the tests pin the arithmetic and the precedence, not the values'
authority.
"""
from pipeline import risk_model as rm
from stage_i_fakes import flat_bars, make_bars, trade_config


def cfg(tmp_path, **overrides):
    return trade_config(tmp_path, **overrides)


# ── indicators ────────────────────────────────────────────────────────────

def test_atr_of_a_flat_2pct_range_series():
    bars = flat_bars(60, 100.0, wick_pct=1.0)          # High 101, Low 99 every day
    assert rm.atr(bars, 20) == 2.0
    assert rm.recent_swing_low(bars, 20) == 99.0


def test_ema_and_sma_track_the_close_path():
    bars = flat_bars(60, 50.0)
    assert round(rm.ema(bars, 10), 6) == 50.0
    assert round(rm.sma(bars, 50), 6) == 50.0
    assert rm.sma(flat_bars(20, 50.0), 50) is None      # not enough history


# ── market cap / liquidity as secondary modifiers ─────────────────────────

def test_cap_tiers_are_ordered_and_have_an_unknown_cap_default(tmp_path):
    config = cfg(tmp_path)
    assert rm.cap_tier(250e9, config)[:2] == (7.0, 2.0)
    assert rm.cap_tier(50e9, config)[:2] == (8.0, 2.5)
    assert rm.cap_tier(5e9, config)[:2] == (10.0, 3.0)
    assert rm.cap_tier(None, config)[:2] == (config["trade_default_risk_ceiling_pct"],
                                             config["trade_default_trail_atr_multiple"])


def test_thin_liquidity_tightens_the_ceiling(tmp_path):
    config = cfg(tmp_path, trade_liquidity_floor_usd=20e6, trade_illiquid_ceiling_factor=0.8)
    liquid, _ = rm.allowed_risk_pct(50e9, 100e6, config)
    thin, notes = rm.allowed_risk_pct(50e9, 5e6, config)
    assert liquid == 8.0 and thin == 6.4
    assert any("thin liquidity" in n for n in notes)


def test_hard_maximum_clamps_every_tier(tmp_path):
    ceiling, notes = rm.allowed_risk_pct(5e9, None, cfg(tmp_path, trade_max_risk_pct=6.0))
    assert ceiling == 6.0
    assert any("hard maximum" in n for n in notes)


def test_volatility_outranks_market_cap_for_trailing_room(tmp_path):
    config = cfg(tmp_path, trade_high_volatility_atr_pct=5.0, trade_high_volatility_trail_bonus=0.5)
    assert rm.trail_atr_multiple(250e9, 2.0, config) == 2.0      # calm mega cap: tightest
    assert rm.trail_atr_multiple(250e9, 6.0, config) == 2.5      # same cap, high ATR: more room
    assert rm.trail_atr_multiple(5e9, 2.0, config) == 3.0        # mid cap: more room than mega


# ── the initial risk plan ─────────────────────────────────────────────────

def test_structure_has_priority_when_it_is_wider_than_the_volatility_floor(tmp_path):
    config = cfg(tmp_path, trade_atr_stop_multiple=2.0, trade_structural_stop_buffer_pct=1.0)
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=92.0, market_cap=5e9,
                              config=config)
    assert plan["atr20"] == 2.0 and plan["atr_pct"] == 2.0
    assert plan["volatility_required_risk_pct"] == 4.0
    assert plan["structural_stop"] == 91.08                      # 92 less the 1% buffer
    assert plan["structural_risk_pct"] == 8.92
    assert plan["risk_basis"] == rm.STRUCTURE
    assert plan["suggested_initial_stop"] == 91.08
    assert plan["trade_plan_status"] == rm.OK


def test_a_structural_stop_inside_the_noise_is_widened_to_the_volatility_floor(tmp_path):
    config = cfg(tmp_path, trade_atr_stop_multiple=2.0)
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=99.0, market_cap=5e9,
                              config=config)
    assert plan["structural_risk_pct"] == 1.99
    assert plan["risk_basis"] == rm.VOLATILITY_FLOOR
    assert plan["required_risk_pct"] == 4.0
    assert plan["suggested_initial_stop"] == 96.0
    assert any("volatility floor" in n for n in plan["notes"])


def test_required_stop_wider_than_the_ceiling_is_a_skip_not_a_wider_stop(tmp_path):
    config = cfg(tmp_path)
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=92.0, market_cap=300e9,
                              config=config)
    assert plan["allowed_max_risk_pct"] == 7.0                   # mega-cap tier
    assert plan["required_risk_pct"] == 8.92
    assert plan["trade_plan_status"] == rm.SKIP_RISK_TOO_WIDE
    assert plan["suggested_initial_stop"] == 91.08               # unchanged: never widened or tightened
    assert any("skip rather than widen" in n for n in plan["notes"])


def test_the_same_setup_passes_for_a_mid_cap_and_skips_for_a_mega_cap(tmp_path):
    config = cfg(tmp_path)
    mid = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=92.0, market_cap=5e9,
                             config=config)
    mega = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=92.0, market_cap=300e9,
                              config=config)
    assert (mid["trade_plan_status"], mega["trade_plan_status"]) == (rm.OK, rm.SKIP_RISK_TOO_WIDE)
    assert mid["suggested_initial_stop"] == mega["suggested_initial_stop"]


def test_without_a_structural_low_the_swing_low_is_used_then_volatility(tmp_path):
    config = cfg(tmp_path)
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), market_cap=5e9, config=config)
    assert plan["structural_low_source"] == "recent_swing_low"
    assert plan["structural_low"] == 99.0
    assert plan["risk_basis"] == rm.VOLATILITY_FLOOR


def test_a_structural_low_above_the_entry_is_ignored(tmp_path):
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=140.0, market_cap=5e9,
                              config=cfg(tmp_path))
    assert plan["structural_low"] is None
    assert any("not below the entry price" in n for n in plan["notes"])


def test_no_price_history_means_no_plan(tmp_path):
    plan = rm.build_risk_plan(100.0, None, config=cfg(tmp_path))
    assert plan["trade_plan_status"] == rm.NO_PRICE_DATA
    assert plan["suggested_initial_stop"] is None


def test_atr_reflects_actual_volatility(tmp_path):
    calm = rm.build_risk_plan(100.0, flat_bars(60, 100.0, wick_pct=0.5), market_cap=5e9,
                              config=cfg(tmp_path))
    wild = rm.build_risk_plan(100.0, flat_bars(60, 100.0, wick_pct=4.0), market_cap=5e9,
                              config=cfg(tmp_path))
    assert wild["atr_pct"] > calm["atr_pct"]
    assert wild["required_risk_pct"] > calm["required_risk_pct"]


# ── position sizing ───────────────────────────────────────────────────────

def test_risk_based_sizing_when_risk_is_the_binding_constraint(tmp_path):
    config = cfg(tmp_path, trade_portfolio_value=100_000, trade_risk_per_trade_pct=0.5,
                 trade_max_position_portion_pct=25.0)
    sizing = rm.position_size(100.0, 96.0, config)
    assert sizing["risk_budget"] == 500.0 and sizing["per_share_risk"] == 4.0
    assert sizing["risk_based_shares"] == 125.0
    assert sizing["portion_based_shares"] == 250.0
    assert sizing["suggested_shares"] == 125
    assert sizing["binding_constraint"] == "risk"
    assert sizing["position_value"] == 12_500.0


def test_portion_based_sizing_when_the_portion_cap_is_the_binding_constraint(tmp_path):
    config = cfg(tmp_path, trade_portfolio_value=100_000, trade_risk_per_trade_pct=1.0,
                 trade_max_position_portion_pct=10.0)
    sizing = rm.position_size(100.0, 96.0, config)
    assert sizing["risk_based_shares"] == 250.0
    assert sizing["portion_based_shares"] == 100.0
    assert sizing["suggested_shares"] == 100
    assert sizing["binding_constraint"] == "portion"
    assert sizing["position_portion_pct"] == 10.0


def test_the_smaller_of_the_two_constraints_always_wins(tmp_path):
    config = cfg(tmp_path, trade_portfolio_value=50_000, trade_risk_per_trade_pct=2.0,
                 trade_max_position_portion_pct=20.0)
    sizing = rm.position_size(25.0, 24.0, config)
    assert sizing["risk_based_shares"] == 1000.0 and sizing["portion_based_shares"] == 400.0
    assert sizing["suggested_shares"] == min(1000, 400)


def test_shares_are_floored_never_rounded_up(tmp_path):
    config = cfg(tmp_path, trade_portfolio_value=100_000, trade_risk_per_trade_pct=1.0,
                 trade_max_position_portion_pct=90.0)
    sizing = rm.position_size(100.0, 97.0, config)
    assert sizing["risk_based_shares"] == 333.3333
    assert sizing["suggested_shares"] == 333


def test_a_stop_at_or_above_the_entry_cannot_be_sized(tmp_path):
    sizing = rm.position_size(100.0, 100.0, cfg(tmp_path))
    assert sizing["suggested_shares"] == 0
    assert any("not below the entry" in n for n in sizing["notes"])


def test_a_skip_risk_too_wide_plan_is_never_sized(tmp_path):
    config = cfg(tmp_path)
    plan = rm.build_risk_plan(100.0, flat_bars(60, 100.0), structural_low=92.0, market_cap=300e9,
                              config=config)
    sizing = rm.size_for_plan(100.0, plan, config)
    assert sizing["suggested_shares"] == 0
    assert any("SKIP_RISK_TOO_WIDE" in n for n in sizing["notes"])


def test_portfolio_value_is_a_plain_config_number(tmp_path):
    small = rm.position_size(100.0, 96.0, cfg(tmp_path, trade_portfolio_value=10_000))
    large = rm.position_size(100.0, 96.0, cfg(tmp_path, trade_portfolio_value=1_000_000))
    assert small["suggested_shares"] * 100 == large["suggested_shares"]


def test_structural_low_comes_from_the_final_contraction():
    assert rm.structural_low_from_metrics({"contraction_low_prices": [80.0, 88.0, 92.5]}) == 92.5
    assert rm.structural_low_from_metrics({}) is None


def test_rising_bars_keep_the_swing_low_below_the_entry():
    bars = make_bars([90 + i * 0.5 for i in range(40)])
    assert rm.recent_swing_low(bars, 20) < float(bars["Close"].iloc[-1])
