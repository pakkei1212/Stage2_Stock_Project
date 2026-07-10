"""Stage D2: verify the 6 CANSLIM-style fundamentals booleans and the
Fundamentals Score sum, using a monkeypatched yf.Ticker so no network call is
made and every input value is known ahead of time.
"""
import pandas as pd
import pytest

from pipeline import stage_screen
from pipeline.config import CONFIG


class FakeTicker:
    def __init__(self, info, income_stmt, balance_sheet):
        self.info = info
        self.income_stmt = income_stmt
        self.balance_sheet = balance_sheet


def _patch_tickers(monkeypatch, tickers_data):
    monkeypatch.setattr(
        stage_screen.yf, "Ticker", lambda sym: tickers_data[sym],
    )


def test_strong_fundamentals_scores_six_of_six(monkeypatch):
    # EPS, revenue and net income all grow every year; equity grows slower
    # than income so ROE stays healthy; margins are solidly positive.
    income = pd.DataFrame(
        {
            "FY2021": [1.0, 100.0, 15.0],
            "FY2022": [1.5, 130.0, 20.0],
            "FY2023": [2.2, 170.0, 28.0],
            "FY2024": [3.0, 220.0, 38.0],
        },
        index=["Diluted EPS", "Total Revenue", "Net Income"],
    )
    balance = pd.DataFrame(
        {"FY2021": [80.0], "FY2022": [95.0], "FY2023": [110.0], "FY2024": [130.0]},
        index=["Stockholders Equity"],
    )
    info = {"earningsQuarterlyGrowth": 0.35, "revenueGrowth": 0.22}
    _patch_tickers(monkeypatch, {"STRONG": FakeTicker(info, income, balance)})

    fund_df = stage_screen.fundamentals_screen(["STRONG"], CONFIG, timeout_sec=5)
    row = fund_df.iloc[0]

    assert row["EPS Growth OK"] and row["Sales Growth OK"]
    assert row["EPS Trend OK"] and row["Sales Trend OK"]
    assert row["ROE OK"] and row["Profitable"]
    assert row["Fundamentals Score"] == 6
    assert row["Avg ROE"] == pytest.approx(0.23622, abs=1e-4)


def test_weak_fundamentals_scores_zero_of_six(monkeypatch):
    # EPS and revenue shrink every year, net income is negative (unprofitable),
    # and quarterly growth is below the required thresholds.
    income = pd.DataFrame(
        {
            "FY2021": [3.0, 220.0, -5.0],
            "FY2022": [2.2, 190.0, -8.0],
            "FY2023": [1.5, 160.0, -12.0],
            "FY2024": [1.0, 130.0, -18.0],
        },
        index=["Diluted EPS", "Total Revenue", "Net Income"],
    )
    balance = pd.DataFrame(
        {"FY2021": [80.0], "FY2022": [78.0], "FY2023": [74.0], "FY2024": [68.0]},
        index=["Stockholders Equity"],
    )
    info = {"earningsQuarterlyGrowth": -0.10, "revenueGrowth": 0.02}
    _patch_tickers(monkeypatch, {"WEAK": FakeTicker(info, income, balance)})

    fund_df = stage_screen.fundamentals_screen(["WEAK"], CONFIG, timeout_sec=5)
    row = fund_df.iloc[0]

    assert not row["EPS Growth OK"] and not row["Sales Growth OK"]
    assert not row["EPS Trend OK"] and not row["Sales Trend OK"]
    assert not row["ROE OK"] and not row["Profitable"]
    assert row["Fundamentals Score"] == 0


def test_missing_data_degrades_gracefully_instead_of_crashing(monkeypatch):
    # yfinance sometimes returns an empty info dict / empty statements for a
    # thinly-covered ticker — the screen should score it 0, not raise.
    _patch_tickers(
        monkeypatch, {"NODATA": FakeTicker({}, pd.DataFrame(), pd.DataFrame())},
    )

    fund_df = stage_screen.fundamentals_screen(["NODATA"], CONFIG, timeout_sec=5)
    row = fund_df.iloc[0]

    assert row["Fundamentals Score"] == 0
    assert bool(row["EPS Growth OK"]) is False
    assert row["Years of Annual Data"] == 0
    assert pd.isna(row["Avg ROE"])
