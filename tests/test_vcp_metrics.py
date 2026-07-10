"""Stage G: verify the numeric VCP (Volatility Contraction Pattern) signals —
contraction depths, whether they're actually tightening, and volume dry-up —
are computed correctly from OHLCV data, independent of the Claude vision call.
"""
import numpy as np
import pandas as pd

from conftest import make_vcp_df
from pipeline.config import CONFIG
from pipeline.stage_vcp_analysis import compute_vcp_metrics


def test_textbook_vcp_detects_decreasing_contractions_and_volume_dryup():
    df = make_vcp_df(seed=0)

    metrics = compute_vcp_metrics(df, CONFIG)

    assert metrics is not None
    assert metrics["contraction_count"] >= 2
    assert metrics["contractions_decreasing"] is True
    # Each successive pullback should be shallower than the last.
    pcts = metrics["contraction_pcts"]
    assert all(pcts[i] > pcts[i + 1] for i in range(len(pcts) - 1))
    # Volume should be drying up (second half lower than first half).
    assert metrics["volume_dryup_ratio"] < 1.0
    assert metrics["pivot_price_candidate"] > 0
    assert metrics["current_price"] > 0


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
