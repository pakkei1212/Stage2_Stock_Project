"""Stage I Layer 2: the deterministic entry plan.

The numeric detector's pivot is the only entry reference, and breakout-volume
confirmation reuses the Stage-G threshold.
"""
from pipeline import entry_plan as ep
from pipeline import setup_gate as gate
from stage_i_fakes import stage_g_row, trade_config


def cfg(tmp_path, **overrides):
    return trade_config(tmp_path, **overrides)


def test_coiling_setup_awaits_the_breakout_with_trigger_and_chase_limit(tmp_path):
    config = cfg(tmp_path, trade_entry_trigger_buffer_pct=0.1, trade_max_chase_pct=5.0)
    plan = ep.build_entry_plan(stage_g_row(pivot=100.0, price=98.0), config, setup_state=gate.READY)

    assert plan["entry_status"] == ep.AWAIT_BREAKOUT
    assert plan["pivot_price"] == 100.0
    assert plan["entry_trigger_price"] == 100.1
    assert plan["maximum_chase_price"] == 105.0
    assert plan["distance_to_pivot_pct"] == -2.0
    assert plan["required_breakout_volume_ratio"] == config["vcp_breakout_volume_ratio"]
    assert plan["setup_state"] == gate.READY
    assert plan["setup_run_id"] == "run-0001" and plan["setup_date"] == "2026-09-15"


def test_fresh_breakout_with_confirmed_volume_is_actionable(tmp_path):
    row = stage_g_row(pivot=100.0, price=102.0, pivot_state="fresh_breakout",
                      metrics={"breakout_volume_vs_20d": 2.1, "breakout_volume_confirmed": True,
                               "sessions_since_breakout": 1})
    plan = ep.build_entry_plan(row, cfg(tmp_path))
    assert plan["entry_status"] == ep.BREAKOUT_ACTIONABLE
    assert plan["breakout_volume_ratio"] == 2.1
    assert ep.is_actionable(plan)


def test_breakout_without_the_required_volume_is_flagged(tmp_path):
    config = cfg(tmp_path)
    row = stage_g_row(pivot=100.0, price=101.0, pivot_state="fresh_breakout",
                      metrics={"breakout_volume_vs_20d": 1.1, "breakout_volume_confirmed": False,
                               "sessions_since_breakout": 1})
    plan = ep.build_entry_plan(row, config)
    assert plan["entry_status"] == ep.BREAKOUT_VOLUME_UNCONFIRMED
    assert f"required {config['vcp_breakout_volume_ratio']}" in plan["notes"][0]
    assert not ep.is_actionable(plan)


def test_price_beyond_the_chase_limit_is_not_chased(tmp_path):
    config = cfg(tmp_path, trade_max_chase_pct=5.0)
    plan = ep.build_entry_plan(stage_g_row(pivot=100.0, price=106.0), config)
    assert plan["entry_status"] == ep.EXTENDED_NO_CHASE
    assert plan["maximum_chase_price"] == 105.0


def test_chase_limit_is_configurable(tmp_path):
    row = stage_g_row(pivot=100.0, price=106.0)
    assert ep.build_entry_plan(row, cfg(tmp_path, trade_max_chase_pct=10.0))["entry_status"] != \
        ep.EXTENDED_NO_CHASE


def test_claude_never_sets_the_entry_price(tmp_path):
    row = stage_g_row(pivot=100.0, price=98.0, claude=True,
                      claude_fields={"pivot_price": 250.0, "suggested_stop_loss": 200.0})
    plan = ep.build_entry_plan(row, cfg(tmp_path))
    assert plan["pivot_price"] == 100.0
    assert plan["pivot_source"] == "numeric_vcp_detector"
    assert plan["entry_trigger_price"] == 100.1
    assert plan["claude"]["pivot_price"] == 250.0          # carried for context only


def test_claude_summary_is_absent_when_claude_did_not_review(tmp_path):
    assert ep.build_entry_plan(stage_g_row(claude=None), cfg(tmp_path))["claude"] is None


def test_missing_pivot_yields_no_entry_plan(tmp_path):
    row = stage_g_row(metrics={"pivot_price_candidate": None})
    plan = ep.build_entry_plan(row, cfg(tmp_path))
    assert plan["entry_status"] == ep.NO_PIVOT
    assert plan["entry_trigger_price"] is None
