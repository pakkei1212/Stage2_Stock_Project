"""Stage G v2 detector: coherent base selection, confirmed vs forming (right-edge)
contractions, High/Low depths, tightening / higher-low / volume / final-zone /
breakout metrics, base-specific pivot and the deterministic quality assessment.

All fixtures are deterministic piecewise-linear paths (conftest.make_path_df):
a steady lead-in rally, then (bars, target_close, volume) segments. Thresholds
asserted here are the detector's own heuristics, not book rules.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_path_df
from pipeline import stage_vcp_analysis as g
from pipeline.config import CONFIG

M = 1_000_000


def metrics(segments, config=CONFIG, **kw):
    m = g.compute_vcp_metrics(make_path_df(segments, **kw), config)
    assert m is not None
    return m


def vcp(v=(10 * M, 7 * M, 4 * M, 2 * M), coil=(6, 96.8, 1.5 * M)):
    """24% -> 14% -> 8% -> 4% under ~98 resistance, rising lows; per-leg volumes ``v``."""
    return [(12, 76.0, v[0]), (12, 98.0, 6 * M), (9, 84.3, v[1]), (9, 97.5, 5 * M),
            (7, 89.7, v[2]), (6, 97.5, 3.5 * M), (5, 93.6, v[3]), coil]


# ── price structure ──────────────────────────────────────────────────────

def test_excellent_vcp_is_one_coherent_strong_base():
    m = metrics(vcp())
    assert m["contraction_count"] == 4 and m["ignored_contraction_count"] == 0
    assert [round(p) for p in m["contraction_pcts"]] == [24, 14, 9, 5]       # High/Low: a bit deeper than Close
    assert m["contraction_confirmed"] == [True, True, True, True]
    assert m["contraction_shrink_fraction"] == 1.0 and m["largest_contraction_expansion_ratio"] < 1
    assert m["tightening_quality"] == "strong" and m["contractions_decreasing"] is True
    assert m["higher_low_flags"] == [True, True, True] and m["base_coherence"] == "strong"
    assert m["resistance_dispersion_pct"] < 3
    assert all(r > 0.8 for r in m["recovery_ratios"])                       # each drop mostly recovered
    assert m["volume_dryup_quality"] == "strong" and m["final_tightness"] == "tight"
    assert m["pivot_state"] == "coiling_below_pivot"
    assert m["vcp_numeric_quality"] == "strong" and m["vcp_quality_notes"] == []
    assert m["vcp_metrics_version"] == g.VCP_METRICS_VERSION


def test_depth_is_measured_on_high_low_excursion_not_close():
    df = make_path_df(vcp())
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    t1 = d["contraction_legs"][0]
    close_depth = (t1["val_high"] - t1["val_low"]) / t1["val_high"]
    base = d["base"]
    high = base["High"].iloc[t1["idx_high"] - 1:t1["idx_low"]].max()
    low = base["Low"].iloc[t1["measured_high_idx"] + 1:t1["idx_low"] + 2].min()
    assert t1["measured_high_price"] == pytest.approx(high) and t1["measured_low_price"] == pytest.approx(low)
    assert t1["depth"] == pytest.approx((high - low) / high)
    assert t1["depth"] > close_depth
    assert m["contraction_high_prices"][0] == pytest.approx(high, abs=0.01)


def test_one_mild_violation_is_acceptable_not_rejected():
    m = metrics([(12, 78.0, 10 * M), (12, 98.0, 6 * M), (9, 85.3, 7 * M), (9, 97.0, 5 * M),
                 (9, 83.4, 6 * M), (9, 96.0, 4 * M), (6, 89.3, 3 * M), (6, 95.0, 2 * M)])   # 22 -> 13 -> 14 -> 7
    assert m["contraction_count"] == 4
    assert 1.0 < m["largest_contraction_expansion_ratio"] <= CONFIG["vcp_max_expansion_ratio"]
    assert m["tightening_quality"] == "acceptable" and m["contractions_decreasing"] is True
    assert m["vcp_numeric_quality"] == "acceptable"


def test_large_later_expansion_is_penalized():
    m = metrics([(12, 75.0, 10 * M), (12, 98.0, 6 * M), (7, 88.2, 7 * M), (7, 97.0, 5 * M),
                 (10, 77.6, 8 * M), (10, 96.0, 5 * M), (6, 88.3, 4 * M), (6, 95.0, 3 * M)])  # 25 -> 10 -> 20 -> 8
    assert m["largest_contraction_expansion_ratio"] > 1.8
    assert m["first_to_last_depth_ratio"] < 0.5                  # last/first alone would look fine
    assert m["tightening_quality"] == "weak" and m["contractions_decreasing"] is False
    assert m["vcp_numeric_quality"] == "weak"
    assert any("expanded" in n for n in m["vcp_quality_notes"])


def test_shrinking_swings_of_an_advancing_trend_are_not_one_base():
    segments = [(10, 85.0, 8 * M), (10, 118.0, 6 * M), (8, 106.2, 6 * M), (10, 138.0, 5 * M),
                (7, 129.7, 5 * M), (10, 160.0, 4 * M), (6, 155.2, 3 * M), (5, 158.0, 3 * M)]
    df = make_path_df(segments)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    # every swing in the window does shrink (15% -> 10% -> 6% -> 3%) ...
    all_depths = [leg["depth"] for leg in d["ignored_legs"] + d["contraction_legs"]]
    assert len(all_depths) == 4 and all(b < a for a, b in zip(all_depths, all_depths[1:]))
    # ... but the highs climb far outside one resistance band, so only the last leg is the base
    assert m["contraction_count"] == 1 and m["ignored_contraction_count"] == 3
    assert m["tightening_quality"] == "insufficient" and m["contractions_decreasing"] is False
    assert m["vcp_numeric_quality"] == "absent"


def test_falling_lows_are_weak_even_when_depths_shrink():
    m = metrics([(12, 80.0, 8 * M), (10, 92.0, 6 * M), (9, 78.0, 6 * M), (8, 86.0, 5 * M),
                 (7, 77.0, 5 * M), (6, 80.0, 4 * M)])
    assert m["contraction_count"] >= 2
    assert all(b < a for a, b in zip(m["contraction_pcts"], m["contraction_pcts"][1:]))
    assert m["higher_low_fraction"] == 0.0 and m["base_coherence"] == "weak"
    assert m["vcp_numeric_quality"] not in ("strong", "acceptable")


def test_tightening_classification_boundaries():
    q = g._tightening_quality
    assert q([24, 14, 8, 4]) == "strong"
    assert q([22, 13, 14, 7]) == "acceptable"
    assert q([25, 10, 20, 8]) == "weak"
    assert q([10, 9, 8, 1, 15]) == "not_contracting"             # ends on its widest leg
    assert q([20, 22, 24, 5]) == "not_contracting"               # most legs expand
    assert q([20, 18]) == "weak"                                 # shrinks, but not meaningfully
    assert q([20]) == "insufficient"


# ── right edge / no lookahead ────────────────────────────────────────────

RIGHT_EDGE = [(12, 80.0, 10 * M), (12, 98.0, 6 * M), (9, 87.2, 6 * M), (9, 97.5, 4 * M),
              (7, 91.6, 3 * M), (8, 97.0, 2.5 * M), (3, 94.1, 1.5 * M)]   # 20 -> 11 -> 6, then a 3% pullback now


def test_forming_final_contraction_inside_the_pivot_window_is_reported():
    window = CONFIG["vcp_pivot_window_days"]
    df = make_path_df(RIGHT_EDGE)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)

    assert m["contraction_count"] == 4
    assert m["contraction_confirmed"] == [True, True, True, False]
    assert m["final_contraction_confirmed"] is False and 3 <= m["final_contraction_pct"] <= 4.5
    assert d["contraction_legs"][-1]["idx_high"] >= len(d["base"]) - window    # inside the blind spot
    assert "final contraction still forming at the right edge" in m["vcp_quality_notes"]

    # The symmetric (v1) pivot finder alone cannot see that leg at all.
    confirmed_only = g._collapse_alternating(
        [(i, k, v, True) for i, k, v in g._find_pivots(d["base"]["Close"], window)])
    confirmed_legs = [a for a, b in zip(confirmed_only, confirmed_only[1:]) if a[1] == "high" and b[1] == "low"]
    assert len(confirmed_legs) == 3


def test_confirmed_pivots_never_use_bars_beyond_the_data():
    close = make_path_df(RIGHT_EDGE)["Close"]
    window = CONFIG["vcp_pivot_window_days"]
    assert all(i <= len(close) - 1 - window for i, _, _ in g._find_pivots(close, window))
    assert all(i >= len(close) - window for i, _, _ in g._find_forming_pivots(close, window))


def test_metrics_as_of_a_date_ignore_every_later_bar():
    df = make_path_df(RIGHT_EDGE)
    future = make_path_df(RIGHT_EDGE + [(10, 60.0, 30 * M)])    # a crash that has not happened yet
    as_of = g.compute_vcp_metrics(future.iloc[:len(df)], CONFIG)
    assert as_of == g.compute_vcp_metrics(df, CONFIG)


def test_forming_leg_becomes_confirmed_later_without_changing_its_high():
    early = make_path_df(RIGHT_EDGE)
    later = make_path_df(RIGHT_EDGE[:-1] + [(4, 93.9, 1.5 * M), (6, 96.5, 1.5 * M)])
    _, d_early = g.compute_vcp_metrics(early, CONFIG, return_details=True)
    _, d_later = g.compute_vcp_metrics(later, CONFIG, return_details=True)
    forming, confirmed = d_early["contraction_legs"][-1], d_later["contraction_legs"][-1]
    assert forming["confirmed"] is False and confirmed["confirmed"] is True
    assert forming["val_high"] == confirmed["val_high"]                   # same swing high, now confirmed
    assert d_early["base"].index[forming["idx_high"]] == d_later["base"].index[confirmed["idx_high"]]


def test_noise_wiggles_below_min_swing_are_not_contractions():
    df = make_path_df(vcp())
    rng = np.random.default_rng(3)
    wiggle = 1 + rng.uniform(-0.003, 0.003, len(df))
    for col in ("Open", "High", "Low", "Close"):
        df[col] = df[col] * wiggle
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
    m = g.compute_vcp_metrics(df, CONFIG)
    assert m["contraction_count"] == 4 and m["tightening_quality"] == "strong"


# ── volume ───────────────────────────────────────────────────────────────

def test_clean_volume_dry_up():
    m = metrics(vcp(v=(10 * M, 7 * M, 4 * M, 2 * M), coil=(6, 96.8, 5 * M)))
    assert m["volume_shrink_fraction"] == 1.0 and m["volume_monotonicity"] == 1.0
    assert m["largest_volume_expansion_ratio"] < 1 and m["volume_dryup_quality"] == "strong"


def test_misleading_first_to_last_volume_ratio_is_caught():
    m = metrics(vcp(v=(10 * M, 5 * M, 12 * M, 4 * M), coil=(6, 96.8, 5 * M)))
    assert m["volume_dryup_ratio"] < 0.5                          # the v1 summary looks great
    assert m["largest_volume_expansion_ratio"] > 2
    assert m["volume_shrink_fraction"] == pytest.approx(0.667, abs=1e-3)
    assert m["volume_dryup_quality"] == "weak"
    assert any("contraction volume expanded" in n for n in m["vcp_quality_notes"])


def test_rising_contraction_volume_is_no_dry_up():
    m = metrics([(12, 80.0, 5 * M), (12, 98.0, 5 * M), (9, 87.2, 6 * M), (9, 97.5, 5 * M),
                 (7, 91.6, 7 * M), (6, 96.8, 5 * M)])
    assert m["volume_contraction_ratios"] and all(r > 1 for r in m["volume_contraction_ratios"])
    assert m["volume_dryup_quality"] == "none"
    assert m["vcp_numeric_quality"] in ("weak", "absent")


def test_final_zone_dryness_vs_20_and_50_day_baselines():
    m = metrics(vcp(coil=(6, 96.8, 1 * M)))
    assert m["final_zone_days"] == CONFIG["vcp_final_zone_days"] and m["final_zone_pre_breakout"] is False
    assert m["final_zone_volume_ratio_20d"] < 0.5 and m["final_zone_volume_ratio_50d"] < 0.3


BREAKOUT = [(12, 76.0, 10 * M), (12, 98.0, 5 * M), (9, 84.3, 7 * M), (9, 97.5, 5 * M),
            (7, 89.7, 4 * M), (6, 97.0, 3 * M), (5, 93.6, 2 * M), (25, 96.5, 2 * M), (1, 99.8, 3.6 * M)]


def test_breakout_on_1_8x_volume_is_numerically_confirmed():
    m = metrics(BREAKOUT)
    assert m["pivot_state"] == "fresh_breakout" and m["sessions_since_breakout"] == 0
    assert m["current_volume_vs_20d"] == pytest.approx(1.8) and m["breakout_volume_vs_20d"] == pytest.approx(1.8)
    assert m["breakout_volume_confirmed"] is True
    # the pre-breakout dry-up is measured before the breakout bar: the 3.6M spike is excluded
    assert m["final_zone_pre_breakout"] is True and m["final_zone_volume_ratio_20d"] == pytest.approx(1.0)


def test_breakout_on_light_volume_is_flagged():
    m = metrics(BREAKOUT[:-1] + [(1, 99.8, 2.2 * M)])
    assert m["pivot_state"] == "fresh_breakout" and m["breakout_volume_confirmed"] is False
    assert m["vcp_numeric_quality"] != "strong"


# ── pivot ────────────────────────────────────────────────────────────────

def test_repeated_resistance_sets_the_pivot():
    m = metrics([(12, 80.0, 8 * M), (12, 99.5, 5 * M), (9, 88.0, 6 * M), (9, 99.8, 4 * M),
                 (7, 93.0, 3 * M), (6, 98.5, 2 * M)])
    assert m["pivot_price_candidate"] == pytest.approx(max(m["contraction_high_prices"][1:]))
    assert 99.8 <= m["pivot_price_candidate"] <= 100.5
    assert m["pivot_state"] == "coiling_below_pivot"


def test_minor_bounce_high_below_resistance_is_not_the_pivot():
    segments = [(12, 78.0, 10 * M), (12, 99.0, 6 * M), (9, 88.0, 6 * M),
                (8, 93.0, 4 * M), (5, 90.0, 3 * M), (3, 91.0, 2 * M)]
    df = make_path_df(segments)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    last_high = [p for p in d["pivots"] if p[1] == "high"][-1][2]      # what v1 would have used
    assert last_high == pytest.approx(93.0)
    assert m["pivot_price_candidate"] == pytest.approx(99.3, abs=0.1)
    assert m["pct_below_pivot"] > 5 and m["pivot_state"] == "far_below_pivot"


def test_price_far_above_a_stale_pivot_is_extended_not_a_breakout():
    m = metrics([(12, 76.0, 10 * M), (12, 98.0, 5 * M), (9, 84.3, 7 * M), (9, 97.5, 5 * M),
                 (7, 89.7, 4 * M), (6, 97.0, 3 * M), (5, 93.6, 2 * M), (6, 99.5, 4 * M), (25, 120.0, 4 * M)])
    assert m["pct_from_pivot"] > 15 and m["sessions_since_breakout"] > 20
    assert m["pivot_state"] == "extended_above_pivot" and m["breakout_volume_confirmed"] is None
    assert m["pct_below_pivot"] == pytest.approx(-m["pct_from_pivot"])


def test_mid_correction_far_below_pivot():
    m = metrics([(12, 76.0, 10 * M), (12, 98.0, 5 * M), (9, 84.3, 7 * M), (9, 97.5, 5 * M),
                 (7, 89.7, 4 * M), (6, 97.0, 3 * M), (4, 82.0, 6 * M)])
    assert m["pivot_state"] == "far_below_pivot" and m["final_contraction_confirmed"] is False
    assert m["final_tightness"] == "loose" and m["vcp_numeric_quality"] == "weak"


# ── compatibility ────────────────────────────────────────────────────────

V1_KEYS = ("contraction_pcts", "contraction_count", "contraction_ratios", "contraction_monotonicity",
           "contractions_decreasing", "volume_dryup_ratio", "volume_leg_avgs", "pivot_price_candidate",
           "current_price", "pct_below_pivot", "pct_ext_ema10", "pct_ext_ema21", "pct_ext_ma50")


def test_v1_metric_keys_are_still_present_and_json_friendly():
    import json
    m = metrics(vcp())
    assert all(k in m for k in V1_KEYS)
    assert m["contraction_monotonicity"] == m["contraction_shrink_fraction"]
    json.dumps(m)                                                 # plain Python types only


def test_prompt_carries_numeric_ground_truth_and_forming_flag():
    m = metrics(RIGHT_EDGE)
    prompt = g._build_prompt("TEST", m)
    assert "still FORMING at the right edge" in prompt
    assert str(m["contraction_pcts"]) in prompt and str(m["final_zone_volume_ratio_20d"]) in prompt
    assert "not a buy signal" in prompt and "do not re-estimate" in prompt


def test_render_verification_chart_for_base_with_forming_leg(tmp_path):
    path = g.render_vcp_annotated_chart("EDGE", make_path_df(RIGHT_EDGE), CONFIG, out_dir=str(tmp_path))
    assert path and (tmp_path / "EDGE.png").stat().st_size > 10_000


def test_flat_series_without_legs_still_returns_metrics():
    dates = pd.bdate_range("2025-01-02", periods=200)
    df = pd.DataFrame({"Open": 100.0, "High": 100.2, "Low": 99.8, "Close": 100.0, "Volume": 1e6}, index=dates)
    m = g.compute_vcp_metrics(df, CONFIG)
    assert m["contraction_count"] == 0 and m["vcp_numeric_quality"] == "absent"
    assert m["pivot_state"] == "coiling_below_pivot" and m["base_start"] is None


def test_gap_up_peak_does_not_leak_the_pre_gap_low_into_the_first_contraction():
    """Regression (seen live on HALO): a leg's Low must come from after its swing
    high. The bar before a gap-up peak has a Low far below the peak; counting it
    turned a ~6% pullback into a 21% 'T1' and a false strong tightening read."""
    segments = [(40, 86.0, 2 * M), (1, 103.0, 8 * M),                       # gap-up session
                (5, 97.0, 4 * M), (8, 108.0, 2 * M), (4, 103.0, 1.6 * M),
                (6, 110.0, 1.8 * M), (4, 105.0, 1.5 * M), (4, 107.5, 1.3 * M)]
    df = make_path_df(segments, lead_in_top=80.0, wick=0.003)                 # rally 50 -> 80 -> 86, then the gap
    gap = 150 + 40                                                            # index of the gap-up bar
    df.iloc[gap, df.columns.get_loc("Open")] = 100.0
    df.iloc[gap, df.columns.get_loc("Low")] = 99.5                            # gap day trades 99.5-103.3
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    first = next(leg for leg in d["contraction_legs"] if leg["idx_high"] == gap - (len(df) - len(d["base"])))
    pre_gap_low = df["Low"].iloc[gap - 1]
    assert first["low_price"] > pre_gap_low + 5                               # the ~85.7 pre-gap Low is excluded
    assert first["depth"] * 100 < 8                                           # ~6%, not ~21%
    assert m["contraction_count"] == 3
    assert m["tightening_quality"] != "strong"                                # ~6.4 -> 5.2 -> 5.1% is not a strong VCP



# ── chronological High -> later Low measurement ──────────────────────────

def _bars(rows, start="2025-06-02"):
    """rows: (high, low, close, volume); Open = previous close."""
    high, low, close, volume = (np.array(c, dtype=float) for c in zip(*rows))
    return pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=pd.bdate_range(start, periods=len(rows)))


def _assert_chronological(leg):
    """Debug invariant (tests only; production degrades instead of asserting)."""
    assert leg["measured_low_idx"] > leg["measured_high_idx"]
    assert leg["measured_low_idx"] > leg["idx_high"]
    v0, v1 = leg["volume_window"]
    assert v0 > leg["measured_high_idx"] and v0 > leg["idx_high"] and v1 == leg["measured_low_idx"]


def test_wide_range_pivot_bar_measures_high_to_the_later_pullback_low():
    #              high   low   close  volume
    base = _bars([(86.5, 85.5, 86.0, 1e6), (86.8, 85.8, 86.5, 1e6),
                  (103.0, 82.0, 102.5, 9e6),                    # pivot bar: range 82 -> 103, closes near the high
                  (102.8, 100.5, 101.0, 2e6), (101.2, 99.0, 99.5, 2e6), (99.8, 98.0, 98.5, 2e6),
                  (100.5, 98.4, 100.0, 1e6)])
    leg = g._measure_leg(base, ih=2, il=5)
    _assert_chronological(leg)
    assert leg["measured_high_price"] == 103.0 and leg["measured_high_date"] == str(base.index[2].date())
    assert leg["measured_low_price"] == 98.0 and leg["measured_low_idx"] == 5
    assert leg["measured_low_date"] == str(base.index[5].date())
    assert leg["depth"] == pytest.approx((103 - 98) / 103)       # ~4.9%, not (103 - 82) / 103 ~20%


def test_same_bar_can_never_supply_both_high_and_low():
    base = _bars([(100.5, 99.5, 100.0, 1e6), (105.0, 90.0, 104.8, 5e6),   # huge single-bar range
                  (104.9, 104.0, 104.5, 1e6), (104.6, 104.2, 104.3, 1e6)])
    leg = g._measure_leg(base, ih=1, il=3)
    _assert_chronological(leg)
    assert leg["measured_low_price"] == 104.0 and leg["depth"] < 0.01
    # the High comes from a later bar and nothing afterwards trades below it: no contraction is invented
    no_pullback = _bars([(100.5, 99.5, 100.0, 1e6), (101.0, 90.0, 100.8, 5e6),
                         (107.0, 100.5, 101.0, 1e6), (107.5, 107.2, 107.4, 1e6)])
    assert g._measure_leg(no_pullback, ih=1, il=3) is None


def test_a_lower_low_before_the_measured_high_is_never_used():
    base = _bars([(95.0, 70.0, 94.0, 1e6),                      # far lower Low, before the High
                  (101.0, 95.0, 100.8, 1e6),                     # structural (Close) swing high, wide range
                  (104.0, 100.0, 100.5, 1e6),                    # measured High comes one bar LATER (intraday)
                  (100.6, 99.0, 99.2, 1e6), (99.5, 98.0, 98.2, 1e6), (99.0, 98.1, 98.9, 1e6)])
    leg = g._measure_leg(base, ih=1, il=4)
    _assert_chronological(leg)
    assert leg["measured_high_idx"] == 2 and leg["measured_high_price"] == 104.0
    assert leg["measured_low_price"] == 98.0                     # not 70 (bar 0) nor 95/100 (at or before the High)


def test_pivot_bar_volume_is_not_counted_as_pullback_selling():
    base = _bars([(86.5, 85.5, 86.0, 1e6), (103.0, 82.0, 102.5, 50e6),   # 50M breakout / gap bar
                  (102.8, 100.5, 101.0, 2e6), (101.2, 99.0, 99.5, 3e6), (99.8, 98.0, 98.5, 4e6),
                  (100.5, 98.4, 100.0, 9e6)])
    leg = g._measure_leg(base, ih=1, il=4)
    _assert_chronological(leg)
    assert leg["volume_window"] == (2, 4)
    assert leg["vol_avg"] == pytest.approx(3e6)                  # (2M + 3M + 4M) / 3; the 50M bar is excluded


def test_wide_range_gap_bar_end_to_end_depth_and_volume():
    segments = [(40, 86.0, 2 * M), (1, 102.5, 40 * M),          # 40M wide-range gap day
                (5, 98.5, 3 * M), (8, 106.0, 2 * M), (4, 102.0, 1.5 * M), (5, 104.0, 1.5 * M)]
    df = make_path_df(segments, lead_in_top=80.0)
    gap = 150 + 40
    df.iloc[gap, df.columns.get_loc("Low")] = 82.0
    df.iloc[gap, df.columns.get_loc("High")] = 103.0
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    off = len(df) - len(d["base"])
    first = next(leg for leg in d["contraction_legs"] if off + leg["measured_high_idx"] == gap)
    _assert_chronological(first)
    assert first["measured_high_price"] == 103.0
    assert first["measured_low_price"] > 97 and first["depth"] < 0.06         # ~103 -> ~98, not -> 82
    assert first["vol_avg"] == pytest.approx(3 * M)                           # gap-day 40M excluded
    assert m["contraction_pcts"][0] < 6 and m["volume_leg_avgs"][0] == round(3 * M)


def test_forming_leg_is_also_measured_chronologically():
    df = make_path_df(RIGHT_EDGE)
    _, d0 = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    off = len(df) - len(d0["base"])
    high_bar = off + d0["contraction_legs"][-1]["idx_high"]
    df.iloc[high_bar, df.columns.get_loc("Low")] = df["Close"].iloc[high_bar] * 0.9   # 10% wick on the forming high bar
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    forming = d["contraction_legs"][-1]
    assert forming["confirmed"] is False
    _assert_chronological(forming)
    assert forming["measured_low_idx"] <= len(d["base"]) - 1                  # never past the last bar
    assert m["final_contraction_pct"] < 5                                      # the wick is not the pullback


def test_measurement_as_of_a_date_ignores_appended_future_bars():
    segments = [(40, 86.0, 2 * M), (1, 102.5, 40 * M), (5, 98.5, 3 * M), (3, 100.0, 2 * M)]
    df = make_path_df(segments, lead_in_top=80.0)
    df.iloc[190, df.columns.get_loc("Low")] = 82.0
    future = make_path_df(segments + [(6, 90.0, 9 * M), (6, 110.0, 9 * M)], lead_in_top=80.0)
    future.iloc[190, future.columns.get_loc("Low")] = 82.0
    m1, d1 = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    m2, d2 = g.compute_vcp_metrics(future.iloc[:len(df)], CONFIG, return_details=True)
    assert m1 == m2 and d1["contraction_legs"] == d2["contraction_legs"]


def test_every_leg_in_every_fixture_is_chronological():
    fixtures = [vcp(), RIGHT_EDGE, BREAKOUT,
                [(12, 75.0, 10 * M), (12, 98.0, 6 * M), (7, 88.2, 7 * M), (7, 97.0, 5 * M),
                 (10, 77.6, 8 * M), (10, 96.0, 5 * M), (6, 88.3, 4 * M), (6, 95.0, 3 * M)],
                [(10, 85.0, 8 * M), (10, 118.0, 6 * M), (8, 106.2, 6 * M), (10, 138.0, 5 * M),
                 (7, 129.7, 5 * M), (10, 160.0, 4 * M), (6, 155.2, 3 * M), (5, 158.0, 3 * M)]]
    rng = np.random.default_rng(11)
    checked = 0
    for segments in fixtures:
        for noise in (0.0, 0.03):
            df = make_path_df(segments)
            if noise:                                                           # random wide intraday ranges
                wick = rng.uniform(0, noise, len(df))
                df["High"] = df["High"] * (1 + wick)
                df["Low"] = df["Low"] * (1 - rng.uniform(0, noise, len(df)))
            _, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
            for leg in d["ignored_legs"] + d["contraction_legs"]:
                _assert_chronological(leg)
                checked += 1
    assert checked >= 20
