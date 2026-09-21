"""check_schedule: parsing/validation of the main check frequency and the
next-occurrence arithmetic for daily / times / interval modes. Pure; no loop."""
from datetime import time as dtime

import pytest

from pipeline import check_schedule as cs
from stage_h_fakes import sgt


def _cfg(**kw):
    base = {"scheduler_mode": "daily", "scheduler_local_time": "09:15", "scheduler_times": "",
            "scheduler_interval_minutes": "60", "scheduler_timezone": "Asia/Singapore"}
    base.update(kw)
    return base


# ── modes / defaults ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", [None, "", "  ", "daily", "DAILY", " Daily "])
def test_missing_or_blank_mode_is_daily(mode):
    s = cs.from_config(_cfg(scheduler_mode=mode))
    assert s.mode == cs.DAILY and s.times == (dtime(9, 15),)


def test_daily_next_occurrence_is_every_calendar_day():
    s = cs.from_config(_cfg())
    assert s.next_after(sgt("2026-09-18 10:00")) == sgt("2026-09-19 09:15")      # Fri -> Sat
    assert s.next_after(sgt("2026-09-19 09:15")) == sgt("2026-09-20 09:15")      # strictly after
    assert s.next_after(sgt("2026-09-20 08:00")) == sgt("2026-09-20 09:15")


def test_times_are_trimmed_sorted_and_each_becomes_due_in_order():
    s = cs.from_config(_cfg(scheduler_mode="times", scheduler_times=" 18:00, 09:15 ,13:00"))
    assert s.times == (dtime(9, 15), dtime(13, 0), dtime(18, 0))
    t, seen = sgt("2026-09-15 08:00"), []
    for _ in range(4):
        t = s.next_after(t)
        seen.append(t)
    assert seen == [sgt("2026-09-15 09:15"), sgt("2026-09-15 13:00"), sgt("2026-09-15 18:00"),
                    sgt("2026-09-16 09:15")]
    assert s.next_after(sgt("2026-09-15 11:00")) == sgt("2026-09-15 13:00")


def test_interval_next_is_relative_to_the_given_moment():
    s = cs.from_config(_cfg(scheduler_mode="interval", scheduler_interval_minutes="60"))
    assert s.next_after(sgt("2026-09-15 11:00")) == sgt("2026-09-15 12:00")
    assert cs.from_config(_cfg(scheduler_mode="interval", scheduler_interval_minutes=15)).interval_minutes == 15


def test_only_the_active_modes_variables_are_validated():
    # daily ignores a broken SCHEDULE_TIMES / interval left in .env for another mode
    assert cs.from_config(_cfg(scheduler_times="banana", scheduler_interval_minutes="0")).mode == cs.DAILY


def test_missed_occurrences_are_counted_for_logging():
    times = cs.from_config(_cfg(scheduler_mode="times", scheduler_times="09:15,13:00,18:00"))
    assert times.occurrences_between(sgt("2026-09-15 13:00"), sgt("2026-09-15 17:00")) == 1
    assert times.occurrences_between(sgt("2026-09-15 13:00"), sgt("2026-09-15 19:00")) == 2
    interval = cs.from_config(_cfg(scheduler_mode="interval"))
    assert interval.occurrences_between(sgt("2026-09-15 13:00"), sgt("2026-09-15 17:00")) == 5


def test_describe_rows_per_mode():
    assert ("Daily check", "09:15") in cs.from_config(_cfg()).describe()
    assert ("Scheduled checks", "09:15, 13:00, 18:00") in cs.from_config(
        _cfg(scheduler_mode="times", scheduler_times="09:15,13:00,18:00")).describe()
    assert ("Interval", "60 minutes") in cs.from_config(_cfg(scheduler_mode="interval")).describe()


# ── invalid configuration fails fast ─────────────────────────────────────

@pytest.mark.parametrize("overrides, message", [
    ({"scheduler_mode": "banana"}, "SCHEDULE_MODE='banana' is not one of daily, times, interval"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": "0"}, "outside the supported range 15-1440"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": "5"}, "outside the supported range 15-1440"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": "1441"}, "outside the supported range"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": "60.5"}, "whole number of minutes"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": "-30"}, "whole number of minutes"),
    ({"scheduler_mode": "interval", "scheduler_interval_minutes": ""}, "whole number of minutes"),
    ({"scheduler_mode": "times", "scheduler_times": "25:00"}, "'25:00' is not a valid HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": "abc"}, "'abc' is not a valid HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": "09:60"}, "not a valid HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": "09:15,"}, "not a valid HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": "09:15;13:00"}, "not a valid HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": ""}, "must list at least one HH:MM time"),
    ({"scheduler_mode": "times", "scheduler_times": "09:15,13:00,9:15"}, "lists 09:15 more than once"),
    ({"scheduler_local_time": "9.15"}, "LOCAL_SCHEDULER_TIME='9.15' is not a valid HH:MM time"),
    ({"scheduler_timezone": "Mars/Olympus"}, "not a known IANA time zone"),
])
def test_invalid_configuration_raises_clear_error(overrides, message):
    with pytest.raises(cs.ScheduleConfigError) as exc:
        cs.from_config(_cfg(**overrides))
    assert message in str(exc.value)


def test_next_after_requires_aware_datetime():
    with pytest.raises(ValueError):
        cs.from_config(_cfg()).next_after(sgt("2026-09-15 09:00").replace(tzinfo=None))
