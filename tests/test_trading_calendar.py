"""Stage H: NASDAQ session resolution across weekends, holidays, early closes,
U.S. DST transitions and the Singapore/New York calendar-date offset."""
from datetime import date, datetime, timezone

import pytest

from pipeline import trading_calendar as tc
from stage_h_fakes import NY, SGT, sgt


def test_normal_weekday_is_a_session_and_weekend_is_not():
    assert tc.is_trading_session(date(2026, 9, 14))       # Monday
    assert not tc.is_trading_session(date(2026, 9, 12))   # Saturday
    assert not tc.is_trading_session(date(2026, 9, 13))   # Sunday


def test_us_market_holiday_is_not_a_session():
    assert not tc.is_trading_session(date(2026, 9, 7))    # Labor Day
    assert not tc.is_trading_session(date(2026, 11, 26))  # Thanksgiving
    assert tc.previous_trading_session(date(2026, 9, 8)) == date(2026, 9, 4)


def test_early_close_session_is_detected_and_its_close_honoured():
    day_after_thanksgiving = date(2026, 11, 27)
    assert tc.is_trading_session(day_after_thanksgiving)
    assert tc.is_early_close(day_after_thanksgiving)
    close_ny = tc.session_close_utc(day_after_thanksgiving).astimezone(NY)
    assert (close_ny.hour, close_ny.minute) == (13, 0)
    assert not tc.is_early_close(date(2026, 11, 25))


def test_tuesday_morning_singapore_resolves_monday_us_session():
    assert tc.latest_completed_session(sgt("2026-09-15 09:15"), 60) == date(2026, 9, 14)


def test_saturday_morning_singapore_resolves_friday_us_session():
    assert tc.latest_completed_session(sgt("2026-09-19 09:15"), 60) == date(2026, 9, 18)


def test_sunday_startup_still_resolves_friday_session():
    assert tc.latest_completed_session(sgt("2026-09-20 11:37"), 60) == date(2026, 9, 18)


def test_monday_morning_singapore_still_resolves_friday():
    # Monday's U.S. session hasn't started yet at 09:15 Monday in Singapore.
    assert tc.latest_completed_session(sgt("2026-09-21 09:15"), 60) == date(2026, 9, 18)


def test_morning_after_holiday_resolves_last_session_before_it():
    # Tuesday 09:15 SGT after Labor Day Monday -> previous Friday.
    assert tc.latest_completed_session(sgt("2026-09-08 09:15"), 60) == date(2026, 9, 4)


def test_early_close_completes_hours_earlier_than_a_normal_close():
    # 2026-11-27 closes 13:00 ET = 02:00 SGT Sat; with a 60-minute buffer it's
    # complete by 03:00 SGT — long before a normal 16:00 close would be.
    assert tc.latest_completed_session(sgt("2026-11-28 03:30"), 60) == date(2026, 11, 27)
    # 01:30 SGT = 12:30 ET: still trading; latest completed skips Thanksgiving to the 25th.
    assert tc.latest_completed_session(sgt("2026-11-28 01:30"), 60) == date(2026, 11, 25)
    assert tc.session_in_progress(sgt("2026-11-28 01:30"), 60) == date(2026, 11, 27)


def test_dst_start_shifts_close_by_one_hour_in_singapore_time():
    # Friday 2026-03-06 (EST): close 16:00 ET = 05:00 SGT Saturday.
    assert tc.latest_completed_session(sgt("2026-03-07 05:30"), 0) == date(2026, 3, 6)
    assert tc.latest_completed_session(sgt("2026-03-07 04:30"), 0) == date(2026, 3, 5)
    # Monday 2026-03-09 (EDT after the 03-08 switch): close 16:00 ET = 04:00 SGT Tuesday.
    assert tc.latest_completed_session(sgt("2026-03-10 04:30"), 0) == date(2026, 3, 9)
    assert tc.latest_completed_session(sgt("2026-03-10 03:30"), 0) == date(2026, 3, 6)


def test_dst_end_shifts_close_back():
    # Monday 2026-11-02 (EST after the 11-01 switch): close = 21:00 UTC = 05:00 SGT Tuesday.
    assert tc.latest_completed_session(sgt("2026-11-03 04:30"), 0) == date(2026, 10, 30)
    assert tc.latest_completed_session(sgt("2026-11-03 05:30"), 0) == date(2026, 11, 2)


def test_same_instant_in_any_timezone_resolves_identically():
    instant = sgt("2026-09-15 09:15")
    assert (tc.latest_completed_session(instant, 60)
            == tc.latest_completed_session(instant.astimezone(NY), 60)
            == tc.latest_completed_session(instant.astimezone(timezone.utc), 60))


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        tc.latest_completed_session(datetime(2026, 9, 15, 9, 15))


def test_session_in_progress_during_us_hours_on_singapore_evening():
    # 22:00 SGT Tuesday = 10:00 ET Tuesday: Tuesday's session is live.
    assert tc.session_in_progress(sgt("2026-09-15 22:00"), 60) == date(2026, 9, 15)
    assert tc.session_in_progress(sgt("2026-09-15 09:15"), 60) is None
    assert tc.session_in_progress(sgt("2026-09-19 09:15"), 60) is None   # Saturday


def test_recent_sessions_skip_weekends_and_holidays():
    assert tc.recent_trading_sessions(4, date(2026, 9, 8)) == [
        date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 8)]
    # A non-session end date anchors to the previous session.
    assert tc.recent_trading_sessions(2, date(2026, 9, 13)) == [date(2026, 9, 10), date(2026, 9, 11)]


def test_unrecorded_sessions_reports_gaps_only():
    recorded = [date(2026, 9, 10), date(2026, 9, 14)]
    assert tc.unrecorded_sessions(recorded, date(2026, 9, 14), 3) == [date(2026, 9, 11)]
