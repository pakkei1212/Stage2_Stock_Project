"""Stage F: daily price chart generation for VCP visual analysis.

Renders a price panel (close + MA50/150/200) and a volume panel (colored by
up/down day) so both the LLM vision pass and a human reviewer can read
volume dry-up alongside price contraction.
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless — no display available in the container
import matplotlib.pyplot as plt
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
    plot_close = close.iloc[-window:]
    plot_ma50 = ma50.iloc[-window:]
    plot_ma150 = ma150.iloc[-window:]
    plot_ma200 = ma200.iloc[-window:]

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10.24, 7.68), dpi=100, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(plot_close.index, plot_close, label="Close", linewidth=1.4, color="black")
    ax_price.plot(plot_ma50.index, plot_ma50, label="MA50", linewidth=1, color="tab:orange")
    ax_price.plot(plot_ma150.index, plot_ma150, label="MA150", linewidth=1, color="tab:blue")
    ax_price.plot(plot_ma200.index, plot_ma200, label="MA200", linewidth=1, color="tab:red")
    ax_price.set_title(f"{ticker} — Daily Close with 50/150/200-day MAs")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    colors = ["tab:green" if c >= o else "tab:red"
              for o, c in zip(plot_df["Open"], plot_df["Close"])]
    ax_vol.bar(plot_df.index, plot_df["Volume"], color=colors, width=1.0)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.3)

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
