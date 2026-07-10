"""Stage B/C: verify the liquidity/price filter, market-cap filter, and
sector relative-strength ranking pick the right survivors and compute the
right numbers, with yfinance fully monkeypatched out.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import data_sources as ds


def test_cheap_liquidity_filter_applies_price_and_dollar_volume_thresholds(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=30)

    def fake_batch_download(tickers, period=None, config=None, **kw):
        return {
            "GOOD": pd.DataFrame({"Close": np.full(30, 50.0), "Volume": np.full(30, 200_000.0)}, index=dates),
            "PENNY": pd.DataFrame({"Close": np.full(30, 1.0), "Volume": np.full(30, 5_000_000.0)}, index=dates),
            "THIN": pd.DataFrame({"Close": np.full(30, 50.0), "Volume": np.full(30, 100.0)}, index=dates),
            "EXPENSIVE": pd.DataFrame({"Close": np.full(30, 3000.0), "Volume": np.full(30, 200_000.0)}, index=dates),
        }

    monkeypatch.setattr(ds, "batch_download", fake_batch_download)

    config = {
        "min_price": 5.0, "max_price": 2000.0, "min_avg_dollar_volume": 5_000_000,
        "bulk_lookback_days": 30, "batch_size": 50, "batch_sleep_sec": 0, "max_retries": 1,
    }
    universe = pd.DataFrame({"Symbol": ["GOOD", "PENNY", "THIN", "EXPENSIVE"]})
    survivors, rows_df = ds.cheap_liquidity_filter(universe, config)

    assert survivors == ["GOOD"]
    good_row = rows_df[rows_df["Symbol"] == "GOOD"].iloc[0]
    assert good_row["Avg Dollar Volume (1mo)"] == 50.0 * 200_000.0
    assert good_row["Passed"]
    assert not rows_df[rows_df["Symbol"] == "PENNY"].iloc[0]["Passed"]
    assert not rows_df[rows_df["Symbol"] == "THIN"].iloc[0]["Passed"]
    assert not rows_df[rows_df["Symbol"] == "EXPENSIVE"].iloc[0]["Passed"]


def test_market_cap_filter_keeps_only_mid_and_large_cap(monkeypatch):
    caps = {"BIG": 50_000_000_000, "SMALL": 500_000_000, "MEGA": 3_000_000_000_000}
    monkeypatch.setattr(ds, "_get_market_cap_timeout", lambda t, timeout_sec=8: caps[t])

    config = {"min_market_cap": 2_000_000_000, "max_market_cap": None}
    survivors, cap_df = ds.market_cap_filter(list(caps.keys()), config)

    assert set(survivors) == {"BIG", "MEGA"}
    assert cap_df.set_index("Symbol").loc["SMALL", "Passed"] == False  # noqa: E712 (explicit bool check)


def test_market_cap_filter_drops_unresolvable_tickers(monkeypatch):
    monkeypatch.setattr(ds, "_get_market_cap_timeout", lambda t, timeout_sec=8: np.nan)

    config = {"min_market_cap": 2_000_000_000, "max_market_cap": None}
    survivors, cap_df = ds.market_cap_filter(["UNKNOWN"], config)

    assert survivors == []
    assert pd.isna(cap_df.iloc[0]["Market Cap"])


def test_rank_sector_strength_orders_by_relative_strength_and_is_arithmetically_correct(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=100)

    def fake_batch_download(tickers, period=None, interval=None, config=None, **kw):
        return {
            "TECH1": pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates),   # +50%
            "TECH2": pd.DataFrame({"Close": np.linspace(100, 140, 100)}, index=dates),   # +40%
            "ENERGY1": pd.DataFrame({"Close": np.linspace(100, 105, 100)}, index=dates), # +5%
        }

    def fake_yf_download(ticker, period=None, interval=None, auto_adjust=None, progress=None):
        return pd.DataFrame({"Close": np.linspace(100, 110, 100)}, index=dates)  # benchmark +10%

    monkeypatch.setattr(ds, "batch_download", fake_batch_download)
    monkeypatch.setattr(ds.yf, "download", fake_yf_download)

    sector_df = pd.DataFrame({
        "Symbol": ["TECH1", "TECH2", "ENERGY1"],
        "Sector": ["Technology", "Technology", "Energy"],
    })
    config = {"sector_rs_lookback_days": 90, "rs_benchmark": "^IXIC"}
    strength, merged = ds.rank_sector_strength(sector_df, config)

    assert strength["Sector"].tolist() == ["Technology", "Energy"]  # sorted desc by RS
    tech_row = strength[strength["Sector"] == "Technology"].iloc[0]
    energy_row = strength[strength["Sector"] == "Energy"].iloc[0]
    # Technology's median stock return is the median of +50% and +40% = +45%.
    assert tech_row["Median Stock Return"] == pytest.approx(0.45)
    assert tech_row["Benchmark Return"] == pytest.approx(0.10)
    assert tech_row["Relative Strength"] == pytest.approx(0.35)
    assert energy_row["Relative Strength"] == pytest.approx(-0.05)
