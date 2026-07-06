"""Stage D/D2: Minervini Trend Template + fundamentals screen.

Ported from notebooks/nasdaq_stage2_screener.ipynb, unchanged logic.
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import yfinance as yf

from .config import CONFIG
from .data_sources import batch_download


def compute_stage2_metrics(df, benchmark_series, config=CONFIG):
    """Minervini Trend Template — 8 technical criteria. Returns dict or None."""
    df = df.dropna(subset=["Close", "Volume"]).copy()
    if len(df) < 260:
        return None

    close = df["Close"]
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    last_close = close.iloc[-1]
    lma50, lma150, lma200 = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]
    if pd.isna(lma200):
        return None

    price_above_mas = last_close > lma150 and last_close > lma200
    ma150_above_ma200 = lma150 > lma200

    ma200_then = ma200.iloc[-config["ma_slope_window"]]
    ma200_slope = (lma200 / ma200_then - 1) if pd.notna(ma200_then) and ma200_then != 0 else np.nan
    ma200_rising = ma200_slope > 0 if pd.notna(ma200_slope) else False

    stacked = lma50 > lma150 and lma50 > lma200

    low_52w = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    pct_above_low = (last_close - low_52w) / low_52w if low_52w > 0 else np.nan
    above_low_ok = pct_above_low >= config["above_low_pct"] if pd.notna(pct_above_low) else False

    high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    pct_below_high = (high_52w - last_close) / high_52w if high_52w > 0 else np.nan
    near_high = pct_below_high <= config["near_high_pct"] if pd.notna(pct_below_high) else False

    def ret_over(series, days):
        if len(series) <= days:
            return np.nan
        v1 = series.iloc[-1]; v1 = v1.item() if hasattr(v1, "item") else float(v1)
        v0 = series.iloc[-days]; v0 = v0.item() if hasattr(v0, "item") else float(v0)
        return (v1 / v0 - 1) if v0 != 0 else np.nan

    rs_3m = ret_over(close, 63) - ret_over(benchmark_series, 63)
    rs_6m = ret_over(close, 126) - ret_over(benchmark_series, 126)
    rs_positive = (rs_3m > 0 and rs_6m > 0) if pd.notna(rs_3m) and pd.notna(rs_6m) else False

    rec = df.iloc[-60:]
    avg_up = rec[rec["Close"].diff() > 0]["Volume"].mean()
    avg_down = rec[rec["Close"].diff() < 0]["Volume"].mean()
    vol_ok = (avg_up > avg_down) if pd.notna(avg_up) and pd.notna(avg_down) else False

    return {
        "Last Close": last_close, "MA50": lma50, "MA150": lma150, "MA200": lma200,
        "Price Above MAs": price_above_mas, "MA150 Above MA200": ma150_above_ma200,
        "200d MA Rising": ma200_rising, "200d MA Slope %": ma200_slope,
        "MAs Stacked Bullish": stacked,
        "Above 52w Low (25%+)": above_low_ok, "Pct Above 52w Low": pct_above_low,
        "Near 52w High": near_high, "Pct Below 52w High": pct_below_high,
        "RS Positive": rs_positive, "RS vs NASDAQ (3mo)": rs_3m, "RS vs NASDAQ (6mo)": rs_6m,
        "Volume Confirms Uptrend": vol_ok, "52w High": high_52w, "52w Low": low_52w,
    }


def run_stage2_screen(tickers, config=CONFIG):
    print(f"Pulling {config['full_lookback_days']}d history for {len(tickers)} survivors...")
    price_data = batch_download(tickers, period=f"{config['full_lookback_days']}d", config=config)

    bench_raw = yf.download(config["rs_benchmark"], period=f"{config['full_lookback_days']}d",
                             interval="1d", auto_adjust=True, progress=False)["Close"]
    bench = bench_raw.iloc[:, 0] if isinstance(bench_raw, pd.DataFrame) else bench_raw

    rows = []
    for t, df in price_data.items():
        m = compute_stage2_metrics(df, bench, config)
        if m:
            m["Symbol"] = t
            rows.append(m)

    print(f"Stage D: metrics computed for {len(rows)} tickers.")
    return pd.DataFrame(rows)


def _annual_series(stmt, row_names):
    row = next((r for r in row_names if r in stmt.index), None)
    return [stmt.loc[row, c] for c in sorted(stmt.columns)] if row else []


def _yoy_growth_rates(values):
    vals = [v for v in values if pd.notna(v)]
    return [(vals[i + 1] - vals[i]) / abs(vals[i])
            for i in range(len(vals) - 1) if vals[i]]


def fundamentals_screen(tickers, config=CONFIG, timeout_sec=15):
    """Minervini/CANSLIM fundamentals — runs only on the small Stage D survivor set.
    Combines current-quarter YoY momentum with a multi-year annual trend check that
    uses every fiscal year yfinance exposes (no fixed cap — usually ~4 annual periods).
    """
    rows = []
    print(f"Fetching fundamentals for {len(tickers)} Stage D survivors...")

    for i, t in enumerate(tickers):
        info, income_stmt, balance_sheet = {}, pd.DataFrame(), pd.DataFrame()

        def fetch(_t=t):
            tk = yf.Ticker(_t)
            return tk.info, tk.income_stmt, tk.balance_sheet

        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(fetch)
        ex.shutdown(wait=False)
        try:
            info, income_stmt, balance_sheet = future.result(timeout=timeout_sec)
        except (FuturesTimeoutError, Exception):
            print(f"  [{i+1}/{len(tickers)}] {t} timed out")

        eps = info.get("earningsQuarterlyGrowth", np.nan)
        sales = info.get("revenueGrowth", np.nan)
        eps_ok = pd.notna(eps) and eps >= config["min_quarterly_eps_growth"]
        sales_ok = pd.notna(sales) and sales >= config["min_quarterly_sales_growth"]

        eps_years = _annual_series(income_stmt, ["Diluted EPS", "Basic EPS"])
        sales_years = _annual_series(income_stmt, ["Total Revenue"])
        ni_years = _annual_series(income_stmt, ["Net Income"])
        equity_years = _annual_series(balance_sheet, ["Stockholders Equity", "Total Stockholders Equity"])
        n_years = max(len(eps_years), len(sales_years))

        eps_yoy = _yoy_growth_rates(eps_years)
        sales_yoy = _yoy_growth_rates(sales_years)
        eps_trend_ok = bool(eps_yoy) and all(g > 0 for g in eps_yoy)
        sales_trend_ok = bool(sales_yoy) and all(g > 0 for g in sales_yoy)

        margins = [ni / rev for ni, rev in zip(ni_years, sales_years) if pd.notna(ni) and pd.notna(rev) and rev]
        roes = [ni / eq for ni, eq in zip(ni_years, equity_years) if pd.notna(ni) and pd.notna(eq) and eq]
        avg_roe = np.nanmean(roes) if roes else np.nan
        avg_margin = np.nanmean(margins) if margins else np.nan
        roe_ok = pd.notna(avg_roe) and avg_roe >= config["min_roe"]
        margin_ok = pd.notna(avg_margin) and avg_margin > config["min_profit_margin"]

        rows.append({
            "Symbol": t,
            "Quarterly EPS Growth YoY": eps, "EPS Growth OK": eps_ok,
            "Quarterly Sales Growth YoY": sales, "Sales Growth OK": sales_ok,
            "Years of Annual Data": n_years,
            "EPS Trend OK": eps_trend_ok,
            "Sales Trend OK": sales_trend_ok,
            "Avg ROE": avg_roe, "ROE OK": roe_ok,
            "Avg Profit Margin": avg_margin, "Profitable": margin_ok,
        })
        time.sleep(0.3)

    fund_df = pd.DataFrame(rows)
    fund_df["Fundamentals Score"] = fund_df[
        ["EPS Growth OK", "Sales Growth OK", "EPS Trend OK", "Sales Trend OK", "ROE OK", "Profitable"]
    ].sum(axis=1)
    print(f"Stage D2: fundamentals scored for {len(fund_df)} tickers.")
    return fund_df
