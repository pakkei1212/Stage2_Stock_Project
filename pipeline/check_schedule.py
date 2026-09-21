"""Main scheduler check frequency (Stage H).

Decides only WHEN the long-running scheduler runs a normal check. It never
decides WHICH session is processed or WHETHER Stage A-G runs: every check still
goes through ``scheduler.check_and_run``, so Stage A-G runs at most once per
completed XNAS session however often the scheduler checks.

Modes (``SCHEDULE_MODE``, default ``daily``):

    daily     one check per local day at LOCAL_SCHEDULER_TIME (the original behaviour)
    times     a check at each of SCHEDULE_TIMES, e.g. "09:15,13:00,18:00"
    interval  a check every SCHEDULE_INTERVAL_MINUTES after the previous check

Only the active mode's variables are validated; malformed values raise
ScheduleConfigError instead of being reinterpreted.
"""
import re
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DAILY, TIMES, INTERVAL = "daily", "times", "interval"
MODES = (DAILY, TIMES, INTERVAL)
MIN_INTERVAL_MINUTES = 15
MAX_INTERVAL_MINUTES = 24 * 60

_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


class ScheduleConfigError(ValueError):
    """The scheduler frequency configuration is invalid."""


def parse_hhmm(value, name):
    text = str(value).strip()
    m = _HHMM.match(text)
    if not m:
        raise ScheduleConfigError(f"{name}={value!r} is not a valid HH:MM time between 00:00 and 23:59")
    return dtime(int(m.group(1)), int(m.group(2)))


def parse_times(value, name="SCHEDULE_TIMES"):
    text = "" if value is None else str(value).strip()
    if not text:
        raise ScheduleConfigError(f"{name} must list at least one HH:MM time when SCHEDULE_MODE=times "
                                  "(e.g. 09:15,13:00,18:00)")
    times = [parse_hhmm(part, f"{name} entry") for part in text.split(",")]
    duplicates = sorted({t.strftime("%H:%M") for t in times if times.count(t) > 1})
    if duplicates:
        raise ScheduleConfigError(f"{name}={value!r} lists {', '.join(duplicates)} more than once")
    return tuple(sorted(times))


def parse_interval(value, name="SCHEDULE_INTERVAL_MINUTES"):
    text = "" if value is None or isinstance(value, bool) else str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ScheduleConfigError(f"{name}={value!r} must be a whole number of minutes")
    minutes = int(text)
    if not MIN_INTERVAL_MINUTES <= minutes <= MAX_INTERVAL_MINUTES:
        raise ScheduleConfigError(f"{name}={minutes} is outside the supported range "
                                  f"{MIN_INTERVAL_MINUTES}-{MAX_INTERVAL_MINUTES} minutes")
    return minutes


@dataclass(frozen=True)
class CheckSchedule:
    mode: str
    timezone: str
    times: tuple = ()                        # daily / times: sorted local wall-clock times
    interval_minutes: Optional[int] = None   # interval

    @property
    def tz(self):
        return ZoneInfo(self.timezone)

    def next_after(self, now):
        """The next main check strictly after ``now`` (aware datetime).

        daily/times: the next configured local wall-clock time, every calendar day.
        interval: ``now + interval``; the loop passes the end of the previous
        check, so the cadence re-bases after each check and after a wake-up.
        """
        if now.tzinfo is None:
            raise ValueError("next_after requires a timezone-aware datetime")
        if self.mode == INTERVAL:
            return now + timedelta(minutes=self.interval_minutes)
        local = now.astimezone(self.tz)
        for day in (local.date(), local.date() + timedelta(days=1)):
            for t in self.times:
                candidate = datetime.combine(day, t, tzinfo=self.tz)
                if candidate > local:
                    return candidate
        raise AssertionError("unreachable: a daily/times schedule always has a time tomorrow")

    def occurrences_between(self, first_due, now, limit=10_000):
        """How many scheduled occurrences fell in [first_due, now] (for missed-check logging)."""
        count, t = 0, first_due
        while t <= now and count < limit:
            count += 1
            t = self.next_after(t)
        return count

    def summary(self):
        if self.mode == INTERVAL:
            return f"every {self.interval_minutes} min ({self.timezone})"
        label = "daily at" if self.mode == DAILY else "at"
        return f"{label} {', '.join(t.strftime('%H:%M') for t in self.times)} {self.timezone}, every calendar day"

    def describe(self):
        """(label, value) rows for --dry-run and history_cli status."""
        rows = [("Schedule mode", self.mode), ("Timezone", self.timezone)]
        if self.mode == DAILY:
            rows.append(("Daily check", self.times[0].strftime("%H:%M")))
        elif self.mode == TIMES:
            rows.append(("Scheduled checks", ", ".join(t.strftime("%H:%M") for t in self.times)))
        else:
            rows.append(("Interval", f"{self.interval_minutes} minutes"))
        return rows


def from_config(config):
    """Build and validate the schedule from CONFIG. Raises ScheduleConfigError."""
    mode = str(config.get("scheduler_mode") or DAILY).strip().lower() or DAILY
    if mode not in MODES:
        raise ScheduleConfigError(f"SCHEDULE_MODE={config.get('scheduler_mode')!r} is not one of "
                                  f"{', '.join(MODES)}")
    tz_name = str(config.get("scheduler_timezone") or "").strip()
    try:
        ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ScheduleConfigError(f"LOCAL_SCHEDULER_TIMEZONE={tz_name!r} is not a known IANA time zone")
    if mode == DAILY:
        return CheckSchedule(DAILY, tz_name, times=(parse_hhmm(config.get("scheduler_local_time"),
                                                               "LOCAL_SCHEDULER_TIME"),))
    if mode == TIMES:
        return CheckSchedule(TIMES, tz_name, times=parse_times(config.get("scheduler_times")))
    return CheckSchedule(INTERVAL, tz_name, interval_minutes=parse_interval(config.get("scheduler_interval_minutes")))
