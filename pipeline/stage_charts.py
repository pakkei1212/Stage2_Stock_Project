"""Stage F: daily price chart generation for VCP visual analysis.

Renders a candlestick price panel (OHLC + MA50/150/200) and a volume panel
(colored by up/down day) so both the LLM vision pass and a human reviewer can
read volume dry-up alongside price contraction. Candlesticks are used
deliberately: intraday range is the direct visual signature of the
volatility contraction VCP is judged on, which a close-only line discards.
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless — no display available in the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .config import CONFIG

logger = logging.getLogger(__name__)


def fetch_chart_data(ticker, config=CONFIG):
    """Pulls enough history to compute MA200 plus the chart's trailing window."""
    lookback = config["full_lookback_days"]
    df = yf.download(ticker, period=f"{lookback}d", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def render_chart(ticker, df, config=CONFIG):
    """Renders the price+volume PNG for one ticker. Returns the saved file path, or None."""
    if df is None or df.empty:
        logger.warning("No data for %s — skipping chart", ticker)
        return None

    close = df["Close"]
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    window = config["chart_lookback_days"]
    plot_df = df.iloc[-window:]
    plot_ma50 = ma50.iloc[-window:]
    plot_ma150 = ma150.iloc[-window:]
    plot_ma200 = ma200.iloc[-window:]

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10.24, 7.68), dpi=250, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Integer x-positions so trading days are evenly spaced (no weekend gaps),
    # the standard layout for candlestick charts.
    xs = np.arange(len(plot_df))
    o = plot_df["Open"].to_numpy()
    h = plot_df["High"].to_numpy()
    low = plot_df["Low"].to_numpy()
    c = plot_df["Close"].to_numpy()
    up = c >= o
    candle_colors = np.where(up, "tab:green", "tab:red").tolist()

    # Wicks (low->high) and bodies (open<->close), colored by up/down day.
    ax_price.vlines(xs, low, h, color=candle_colors, linewidth=0.8, zorder=2)
    ax_price.bar(xs, height=np.abs(c - o), bottom=np.minimum(o, c),
                 width=0.7, color=candle_colors, linewidth=0, zorder=2)

    ax_price.plot(xs, plot_ma50.to_numpy(), label="MA50", linewidth=1, color="tab:orange", zorder=3)
    ax_price.plot(xs, plot_ma150.to_numpy(), label="MA150", linewidth=1, color="tab:blue", zorder=3)
    ax_price.plot(xs, plot_ma200.to_numpy(), label="MA200", linewidth=1, color="tab:purple", zorder=3)
    ax_price.set_title(f"{ticker} — Daily Candlesticks with 50/150/200-day MAs")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_vol.bar(xs, plot_df["Volume"].to_numpy(), color=candle_colors, width=0.7)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.3)

    # Month-boundary x-ticks with date labels (shared across both panels).
    tick_pos, tick_lab, last_month = [], [], None
    for i, ts in enumerate(plot_df.index):
        month = (ts.year, ts.month)
        if month != last_month:
            tick_pos.append(i)
            tick_lab.append(ts.strftime("%Y-%m"))
            last_month = month
    ax_vol.set_xticks(tick_pos)
    ax_vol.set_xticklabels(tick_lab)
    ax_vol.set_xlim(-0.5, len(plot_df) - 0.5)

    plt.tight_layout()

    os.makedirs(config["chart_dir"], exist_ok=True)
    out_path = os.path.join(config["chart_dir"], f"{ticker}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_charts(tickers, config=CONFIG):
    """Fetches data and renders a chart for each ticker. Returns {symbol: path}."""
    paths = {}
    for i, t in enumerate(tickers):
        logger.debug("  [%d/%d] rendering chart for %s", i + 1, len(tickers), t)
        df = fetch_chart_data(t, config)
        path = render_chart(t, df, config)
        if path:
            paths[t] = path
    logger.info("Stage F: %d charts rendered to %s", len(paths), config["chart_dir"])
    return paths
