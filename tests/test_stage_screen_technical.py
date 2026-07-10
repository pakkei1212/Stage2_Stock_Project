"""Stage D: verify the 8 Minervini Trend Template booleans are each computed
correctly (not just "some score comes out"), using synthetic price series with
known trend shape instead of live yfinance data.
"""
import pandas as pd

from conftest import make_ohlcv
from pipeline.config import CONFIG
from pipeline.stage_screen import compute_stage2_metrics

TECH_KEYS = [
    "Price Above MAs", "MA150 Above MA200", "200d MA Rising", "MAs Stacked Bullish",
    "Above 52w Low (25%+)", "Near 52w High", "RS Positive", "Volume Confirms Uptrend",
]


def _flags(metrics):
    return {k: bool(metrics[k]) for k in TECH_KEYS}


def test_strong_uptrend_passes_all_eight_criteria():
    stock = make_ohlcv(daily_drift=0.15, seed=1, volume_bias="bullish")
    benchmark = make_ohlcv(daily_drift=0.01, start=100.0, seed=3, volume_bias="neutral")["Close"]

    metrics = compute_stage2_metrics(stock, benchmark, CONFIG)

    assert metrics is not None
    assert _flags(metrics) == {k: True for k in TECH_KEYS}
    # Sanity on the underlying numbers, not just the derived booleans.
    assert metrics["MA50"] > metrics["MA150"] > metrics["MA200"]
    assert metrics["Last Close"] > metrics["MA200"]
    assert metrics["200d MA Slope %"] > 0
    assert metrics["Pct Below 52w High"] == 0  # series is monotonically rising


def test_sustained_downtrend_fails_most_criteria():
    stock = make_ohlcv(daily_drift=-0.15, start=200.0, seed=2, volume_bias="bearish")
    benchmark = make_ohlcv(daily_drift=0.01, start=100.0, seed=3, volume_bias="neutral")["Close"]

    metrics = compute_stage2_metrics(stock, benchmark, CONFIG)

    assert metrics is not None
    flags = _flags(metrics)
    # A price that only ever falls should never be "above its MAs", have a
    # rising 200d MA, stacked MAs, be 25%+ above its low, or show positive RS.
    assert flags["Price Above MAs"] is False
    assert flags["MA150 Above MA200"] is False
    assert flags["200d MA Rising"] is False
    assert flags["MAs Stacked Bullish"] is False
    assert flags["Above 52w Low (25%+)"] is False
    assert flags["RS Positive"] is False
    assert flags["Volume Confirms Uptrend"] is False
    assert metrics["200d MA Slope %"] < 0
    # Fewer than half the criteria should ever pass for a pure downtrend.
    assert sum(flags.values()) <= 2


def test_returns_none_when_history_too_short():
    stock = make_ohlcv(n=200, seed=1)  # < 260 rows required
    benchmark = make_ohlcv(n=200, seed=3)["Close"]

    assert compute_stage2_metrics(stock, benchmark, CONFIG) is None
