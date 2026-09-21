"""NASDAQ trading-session resolution (Stage H).

The single place that knows about exchange sessions. Everything else asks this
module "which NASDAQ session is that?" instead of doing date arithmetic —
there is deliberately no ``today - 1 day`` anywhere in the scheduler.

Backed by ``exchange_calendars``' XNAS calendar, which maintains holidays and
early closes (e.g. the day after Thanksgiving closes at 13:00 ET). Session
boundaries come back as UTC timestamps, so a Singapore wall-clock time and a
New York close are compared as instants, never as naive local dates.

A *session date* is the exchange's own label for the trading day (a plain
``datetime.date``). It is independent of where the code runs: Monday's session
is still Monday when it is processed at 09:15 Tuesday in Singapore.
"""
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd

EXCHANGE = "XNAS"
EXCHANGE_TIMEZONE = "America/New_York"

# Fixed bounds keep the calendar (and therefore every test) independent of the
# real current date; exchange_calendars' default window is relative to "now".
_CAL_START = "2016-01-01"
_CAL_END = "2036-12-31"


@lru_cache(maxsize=1)
def _calendar():
    import exchange_calendars as xcals
    return xcals.get_calendar(EXCHANGE, start=_CAL_START, end=_CAL_END)


def _as_date(d):
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    return d


def _ts(d):
    return pd.Timestamp(_as_date(d))


def _require_aware(now):
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("trading_calendar requires a timezone-aware datetime")
    return now


def _in_bounds(d):
    cal = _calendar()
    ts = _ts(d)
    return cal.first_session <= ts <= cal.last_session


def is_trading_session(d):
    """True if ``d`` (a date) is a NASDAQ trading session."""
    if not _in_bounds(d):
        return False
    return bool(_calendar().is_session(_ts(d)))


def session_open_utc(session):
    return _calendar().session_open(_ts(session)).to_pydatetime()


def session_close_utc(session):
    """The session's actual close as an aware UTC datetime (early closes honoured)."""
    return _calendar().session_close(_ts(session)).to_pydatetime()


def is_early_close(session):
    close_ny = pd.Timestamp(session_close_utc(session)).tz_convert(EXCHANGE_TIMEZONE)
    return (close_ny.hour, close_ny.minute) < (16, 0)


def previous_trading_session(d):
    """The latest session strictly before ``d``."""
    cal = _calendar()
    return cal.date_to_session(_ts(d) - pd.Timedelta(days=1), direction="previous").date()


def next_trading_session(d):
    """The earliest session strictly after ``d``."""
    cal = _calendar()
    return cal.date_to_session(_ts(d) + pd.Timedelta(days=1), direction="next").date()


def sessions_between(start, end):
    """All sessions in [start, end], inclusive, as dates."""
    return [ts.date() for ts in _calendar().sessions_in_range(_ts(start), _ts(end))]


def recent_trading_sessions(n, ending_at):
    """The last ``n`` sessions ending at (and including, if a session) ``ending_at``."""
    if n <= 0:
        return []
    cal = _calendar()
    end = cal.date_to_session(_ts(ending_at), direction="previous")
    return [ts.date() for ts in cal.sessions_window(end, -n)]


def latest_completed_session(now, close_buffer_minutes=0):
    """The most recent session whose close (+ buffer) is at or before ``now``.

    ``now`` must be timezone-aware; its zone is irrelevant (Singapore, New York
    and UTC inputs for the same instant resolve identically). Returns a
    ``date`` or None if nothing in the calendar's range qualifies.
    """
    now_utc = _require_aware(now).astimezone(timezone.utc)
    buffer = timedelta(minutes=close_buffer_minutes)
    # The exchange-local date is the latest session that could possibly be done.
    candidate = pd.Timestamp(now_utc).tz_convert(EXCHANGE_TIMEZONE).date()
    cal = _calendar()
    if _ts(candidate) > cal.last_session:
        candidate = cal.last_session.date()
    for _ in range(15):  # never more than ~5 non-session days in a row
        if _ts(candidate) < cal.first_session:
            return None
        if is_trading_session(candidate) and session_close_utc(candidate) + buffer <= now_utc:
            return candidate
        candidate = candidate - timedelta(days=1)
    return None


def session_in_progress(now, close_buffer_minutes=0):
    """The session that has opened but is not yet completed at ``now``, or None.

    While a session is in progress, upstream daily data includes a *partial* bar
    for it, so screening then would mislabel intraday data as the prior session.
    """
    now_utc = _require_aware(now).astimezone(timezone.utc)
    buffer = timedelta(minutes=close_buffer_minutes)
    d = pd.Timestamp(now_utc).tz_convert(EXCHANGE_TIMEZONE).date()
    for candidate in (d, d - timedelta(days=1)):
        if is_trading_session(candidate):
            if session_open_utc(candidate) <= now_utc < session_close_utc(candidate) + buffer:
                return candidate
    return None


def unrecorded_sessions(recorded, ending_at, lookback):
    """Sessions in the recent ``lookback`` window ending at ``ending_at`` that are
    not in ``recorded`` — used to report gaps, never to trigger backfill."""
    recorded = {_as_date(r) for r in recorded}
    return [s for s in recent_trading_sessions(lookback, ending_at) if s not in recorded]


def to_date(value):
    """Parse an ISO string / date / datetime into a date."""
    if isinstance(value, str):
        return date.fromisoformat(value)
    return _as_date(value)
