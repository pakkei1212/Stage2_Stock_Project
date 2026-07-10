"""Shared synthetic OHLCV fixtures for stage correctness tests.

None of these tests hit the network — every yfinance/requests/Anthropic call is
monkeypatched with deterministic fake data so the assertions are exact and
reproducible, not "some real ticker happened to pass".
"""
import numpy as np
import pandas as pd

HI_VOL = 2_000_000
LO_VOL = 800_000


def make_ohlcv(n=400, start=50.0, daily_drift=0.15, noise_scale=0.3, seed=0, volume_bias="neutral"):
    """Synthetic daily OHLCV series with a controllable linear drift + noise.

    volume_bias:
      "bullish" -> up (close-over-close) days get more volume than down days
      "bearish" -> down days get more volume than up days
      "neutral" -> volume is unrelated to direction
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    steps = daily_drift + rng.normal(0, noise_scale, n)
    close = np.maximum(start + np.cumsum(steps), 1.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, n)

    day_over_day_up = np.r_[False, np.diff(close) > 0]
    if volume_bias == "bullish":
        base_vol = np.where(day_over_day_up, HI_VOL, LO_VOL)
    elif volume_bias == "bearish":
        base_vol = np.where(day_over_day_up, LO_VOL, HI_VOL)
    else:
        base_vol = np.full(n, (HI_VOL + LO_VOL) / 2)
    volume = base_vol + rng.integers(-50_000, 50_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_vcp_df(n=140, seed=0):
    """Textbook VCP: three pullbacks of decreasing depth (20% -> 12% -> 6%),
    each followed by a rally, ending in a tight base with volume drying up
    in the second half."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)

    price = 100.0
    seg1 = price + np.cumsum(np.full(20, 1.0))
    price = seg1[-1]
    seg2 = price - np.linspace(0, price * 0.20, 15)
    price = seg2[-1]
    seg3 = price + np.linspace(0, (seg1[-1] - price) * 0.9, 15)
    price = seg3[-1]
    seg4 = price - np.linspace(0, price * 0.12, 15)
    price = seg4[-1]
    seg5 = price + np.linspace(0, (seg3[-1] - price) * 0.9, 15)
    price = seg5[-1]
    seg6 = price - np.linspace(0, price * 0.06, 15)
    price = seg6[-1]
    tail_len = n - (20 + 15 + 15 + 15 + 15 + 15)
    seg7 = price + rng.normal(0, 0.3, tail_len)

    close = np.concatenate([seg1, seg2, seg3, seg4, seg5, seg6, seg7])[:n]
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2

    half = n // 2
    volume = np.r_[np.full(half, 3_000_000), np.full(n - half, 900_000)].astype(float)
    volume += rng.integers(-50_000, 50_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
