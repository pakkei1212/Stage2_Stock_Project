"""Market-data readiness guard (Stage H).

Stage A-G always screens the *latest* bars upstream exposes, so the only way
to label a run with the right trading session is to confirm, before launching,
that the latest available daily bar IS that session. One lightweight download
of a reference index (``^IXIC`` by default — already the pipeline's RS
benchmark) answers that.

Deliberately not routed through the OHLCV cache: the question is "what does
the provider expose right now", which a cached copy cannot answer.
"""
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReadinessResult:
    ready: bool
    target_session: object
    latest_bar: object      # date or None
    attempts: int
    reason: str


def fetch_latest_bar_date(symbol):
    """Date of the newest daily bar upstream has for ``symbol``, or None."""
    import yfinance as yf
    df = yf.download(symbol, period="10d", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]) if "Close" in df.columns else df
    if df.empty:
        return None
    return pd.Timestamp(df.index.max()).date()


def check_once(target_session, symbol, fetch=fetch_latest_bar_date):
    try:
        latest = fetch(symbol)
    except Exception as e:
        return False, None, f"fetch failed: {type(e).__name__}"
    if latest is None:
        return False, None, "no bars returned"
    if latest == target_session:
        return True, latest, "latest bar matches target session"
    if latest > target_session:
        # A newer (likely in-progress) session is already visible; screening now
        # would mix a partial bar into data labelled as the target session.
        return False, latest, "a newer bar than the target session is present"
    return False, latest, "latest bar is older than the target session"


def wait_for_data(target_session, config, fetch=fetch_latest_bar_date, sleep=None, on_retry=None):
    """Poll up to ``data_ready_max_attempts`` times, sleeping
    ``data_ready_retry_minutes`` between attempts. ``sleep`` and ``fetch`` are
    injectable so tests never wait or touch the network."""
    import time
    sleep = sleep or time.sleep
    symbol = config["data_ready_symbol"]
    max_attempts = max(1, int(config["data_ready_max_attempts"]))
    delay = float(config["data_ready_retry_minutes"]) * 60

    latest, reason = None, ""
    for attempt in range(1, max_attempts + 1):
        ok, latest, reason = check_once(target_session, symbol, fetch)
        logger.info("Data readiness [%d/%d] %s: target=%s latest=%s — %s",
                    attempt, max_attempts, symbol, target_session, latest, reason)
        if ok:
            return ReadinessResult(True, target_session, latest, attempt, reason)
        if latest is not None and latest > target_session:
            break  # waiting will not make a newer bar go away
        if attempt < max_attempts:
            if on_retry:
                on_retry(attempt)
            logger.info("Data not ready — retrying in %.0f minute(s).", delay / 60)
            sleep(delay)
    return ReadinessResult(False, target_session, latest, attempt, reason)
