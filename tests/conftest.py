"""Shared synthetic OHLCV fixtures for stage correctness tests.

None of these tests hit the network — every yfinance/requests/Anthropic call is
monkeypatched with deterministic fake data so the assertions are exact and
reproducible, not "some real ticker happened to pass".
"""
import socket

import numpy as np
import pandas as pd
import pytest

HI_VOL = 2_000_000


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any attempt to open a network connection fails the test loudly."""
    def _blocked(*args, **kwargs):
        raise RuntimeError("tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(key, raising=False)
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


def make_path_df(segments, start_price=50.0, lead_in_bars=150, lead_in_top=100.0,
                 lead_in_volume=5_000_000.0, wick=0.003):
    """Deterministic piecewise-linear daily OHLCV (no noise) for VCP structure tests.

    A steady lead-in rally (Stage 2 context + MA history) climbs from
    ``start_price`` to ``lead_in_top``; then each ``(bars, target_close, volume)``
    segment moves Close linearly to its target at a constant daily volume.
    Open = prior close; High/Low add a small ``wick`` around the body, so
    High/Low depths run slightly deeper than Close-to-Close moves.
    """
    closes = list(np.linspace(start_price, lead_in_top, lead_in_bars))
    volumes = [float(lead_in_volume)] * lead_in_bars
    price = lead_in_top
    for bars, target, volume in segments:
        closes.extend(np.linspace(price, target, bars + 1)[1:])
        volumes.extend([float(volume)] * bars)
        price = target
    close = np.array(closes)
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) * (1 + wick),
         "Low": np.minimum(open_, close) * (1 - wick), "Close": close, "Volume": np.array(volumes)},
        index=pd.bdate_range("2025-01-02", periods=len(close)),
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
