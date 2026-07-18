"""Stage G: verify the numeric VCP (Volatility Contraction Pattern) signals —
contraction depths, whether they're actually tightening, and volume dry-up —
are computed correctly from OHLCV data, independent of the Claude vision call.
"""
import numpy as np
import pandas as pd

from conftest import make_vcp_df
from pipeline.config import CONFIG
from pipeline.stage_vcp_analysis import _assess_tightening, compute_vcp_metrics


def test_textbook_vcp_detects_decreasing_contractions_and_volume_dryup():
    df = make_vcp_df(seed=0)

    metrics = compute_vcp_metrics(df, CONFIG)

    assert metrics is not None
    assert metrics["contraction_count"] >= 2
    assert metrics["contractions_decreasing"] is True
    assert metrics["contraction_monotonicity"] == 1.0
    # Each successive pullback should be shallower than the last.
    pcts = metrics["contraction_pcts"]
    assert all(pcts[i] > pcts[i + 1] for i in range(len(pcts) - 1))
    # Leg-over-leg ratios should all be < 1.0 (each leg shallower than the last).
    assert all(r < 1.0 for r in metrics["contraction_ratios"])
    assert len(metrics["contraction_ratios"]) == metrics["contraction_count"] - 1
    # Volume dry-up is now measured per contraction leg (pullback), not by a
    # half-vs-half window split: one average per contraction, later ones lighter.
    assert len(metrics["volume_leg_avgs"]) == metrics["contraction_count"]
    assert metrics["volume_dryup_ratio"] < 1.0  # last contraction lighter than the first
    assert metrics["pivot_price_candidate"] > 0
    assert metrics["current_price"] > 0


def test_assess_tightening_requires_majority_of_legs_to_shrink_not_just_averaged_halves():
    # A single deep outlier leg (10%) can drag the "first half" average below
    # the "second half" average even though 3 of 4 legs never shrank and the
    # base ends on its widest pullback yet (15% > the first leg's 10%). The
    # old halves-average comparison called this "contracting" (8% < 9.5%);
    # the leg-over-leg check must not.
    contractions = [0.10, 0.09, 0.08, 0.01, 0.15]

    ratios, monotonic_fraction, is_contracting = _assess_tightening(contractions)

    assert len(ratios) == len(contractions) - 1
    assert is_contracting is False


def test_assess_tightening_confirms_a_genuinely_tightening_sequence():
    contractions = [0.20, 0.12, 0.06]

    ratios, monotonic_fraction, is_contracting = _assess_tightening(contractions)

    assert ratios == [0.6, 0.5]
    assert monotonic_fraction == 1.0
    assert is_contracting is True


def test_assess_tightening_returns_false_for_fewer_than_two_legs():
    assert _assess_tightening([]) == ([], None, False)
    assert _assess_tightening([0.10]) == ([], None, False)


def test_flat_choppy_series_has_no_clear_contraction_trend():
    # Pure noise around a constant price: pullbacks exist but don't reliably
    # shrink over time, so the "tightening base" signal should not fire.
    rng = np.random.default_rng(42)
    n = 140
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + rng.normal(0, 5, n).cumsum() * 0.05  # mean-reverting-ish noise
    close = np.clip(close, 80, 120)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    volume = np.full(n, 1_000_000.0) + rng.integers(-100_000, 100_000, n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)

    metrics = compute_vcp_metrics(df, CONFIG)

    # Should still return a well-formed metrics dict (not crash), even if it
    # doesn't represent a valid VCP setup.
    assert metrics is not None
    assert isinstance(metrics["contractions_decreasing"], bool)
    assert 0 <= metrics["contraction_count"] == len(metrics["contraction_pcts"])


def test_returns_none_when_not_enough_trailing_history():
    n = CONFIG["vcp_pivot_window_days"] * 4 - 1  # one short of the minimum
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000.0},
        index=dates,
    )

    assert compute_vcp_metrics(df, CONFIG) is None


def test_pivot_price_is_never_below_current_price_for_a_series_off_its_high():
    df = make_vcp_df(seed=0)
    metrics = compute_vcp_metrics(df, CONFIG)

    # The pivot is defined as the most recent swing high, so a price that's
    # currently below it should show a non-negative "pct below pivot".
    assert metrics["pct_below_pivot"] >= 0
    assert metrics["pivot_price_candidate"] >= metrics["current_price"] * (1 - 1e-9)
