"""Parquet-backed OHLCV cache (roadmap item 3).

One Parquet file per ticker under ``cache_dir``, indexed by date with
Open/High/Low/Close/Volume columns. This lets the weekly pipeline fetch only
the *delta* since the last run instead of re-downloading full history for the
whole ~4,300-ticker universe every time — cutting runtime and yfinance
rate-limit exposure, and accumulating the price history that backtesting
(roadmap item 1) will need.

Correctness note: yfinance is called with ``auto_adjust=True``, so a split or
dividend retroactively rescales *all* historical adjusted prices. When
appending a delta we therefore compare the overlap between cached and
freshly-fetched bars; a uniform price ratio far from 1.0 signals a corporate
action, and the cached history is rescaled onto the new adjustment basis before
concatenation so the series stays continuous.
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLS = ["Open", "High", "Low", "Close"]
_OHLCV_COLS = _PRICE_COLS + ["Volume"]

# A uniform >2% shift across the whole overlap is a corporate action (split /
# sizeable dividend re-adjustment), not day-to-day noise.
SPLIT_TOL = 0.02


def cache_path(ticker, cache_dir):
    return os.path.join(cache_dir, f"{ticker}.parquet")


def load(ticker, cache_dir):
    """Return the cached frame for ``ticker`` (DatetimeIndex, sorted), or None."""
    path = cache_path(ticker, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.warning("OHLCV cache read failed for %s — ignoring cached copy", ticker, exc_info=True)
        return None
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def save(ticker, df, cache_dir):
    """Persist ``df`` (OHLCV columns only) to the ticker's Parquet file."""
    if df is None or df.empty:
        return
    os.makedirs(cache_dir, exist_ok=True)
    out = df[[c for c in _OHLCV_COLS if c in df.columns]].copy()
    out.index = pd.to_datetime(out.index)
    try:
        out.to_parquet(cache_path(ticker, cache_dir))
    except Exception:
        logger.warning("OHLCV cache write failed for %s", ticker, exc_info=True)


def merge(old, new):
    """Merge freshly-fetched ``new`` bars into cached ``old``.

    ``new`` wins on any overlapping dates. If the overlap reveals a uniform
    price re-scaling (split/dividend re-adjustment), the older cached bars are
    rescaled onto the new basis first so the combined series stays continuous.
    Returns a sorted, de-duplicated frame.
    """
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old

    old = old.sort_index()
    new = new.sort_index()

    overlap = old.index.intersection(new.index)
    if len(overlap) and "Close" in old.columns and "Close" in new.columns:
        old_close = old.loc[overlap, "Close"].replace(0, np.nan)
        ratio = (new.loc[overlap, "Close"] / old_close).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            med = float(ratio.median())
            if np.isfinite(med) and abs(med - 1.0) > SPLIT_TOL:
                for col in _PRICE_COLS:
                    if col in old.columns:
                        old[col] = old[col] * med
                if "Volume" in old.columns and med != 0:
                    old["Volume"] = old["Volume"] / med
                logger.info(
                    "Corporate action detected (overlap close ratio %.4f) — rescaled cached history", med,
                )

    combined = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def is_fresh(df, today, max_age_days):
    """True if the newest cached bar is within ``max_age_days`` of ``today``."""
    if df is None or df.empty:
        return False
    last = pd.to_datetime(df.index.max())
    return (today - last).days <= max_age_days


def covers(df, start_needed):
    """True if the cache reaches back at least as far as ``start_needed``."""
    if df is None or df.empty:
        return False
    return pd.to_datetime(df.index.min()) <= start_needed
