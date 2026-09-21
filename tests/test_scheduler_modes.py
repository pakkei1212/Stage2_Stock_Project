"""Stage H main check frequency in the long-running loop: daily / times /
interval modes, their interaction with the startup check, sleep/resume,
idempotency, the single readiness episode, the same-day hold and the
market-open guard.

Every check is recorded by wrapping the real ``check_and_run``; the fake clock
advances instantly, Stage A-G is a recorder, Telegram is in-memory. No network."""
import logging
import os
from datetime import date

import pytest

from pipeline import check_schedule as cs
from pipeline import history_store as hs
from pipeline import scheduler as sch
from stage_h_fakes import FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt

SAT_FRI = date(2026, 9, 18)
MON = date(2026, 9, 14)
FRI_BEFORE = date(2026, 9, 11)


class _Stop(Exception):
    pass


def _store(cfg):
    return hs.HistoryStore(cfg["history_db_path"])


def _record_checks(monkeypatch, clock):
    """Wrap the real check_and_run; returns [(trigger, HH:MM, decision)]."""
    real = sch.check_and_run
    seen = []

    def wrapper(trigger, config, deps, **kw):
        started = clock.current
        outcome = real(trigger, config, deps, **kw)
        seen.append((trigger, started.strftime("%H:%M"), outcome.decision))
        return outcome
    monkeypatch.setattr(sch, "check_and_run", wrapper)
    return seen


def _run_until(cfg, deps, clock, end, jump=None):
    """Run the loop until the fake clock reaches ``end``. ``jump=(at, to)`` simulates one suspend."""
    real_sleep = clock.sleep
    state = {"jumped": False}

    def sleep(seconds):
        if jump and not state["jumped"] and clock.current >= jump[0]:
            state["jumped"] = True
            clock.current = jump[1]
            return
        real_sleep(seconds)
        if clock.current >= end:
            raise _Stop
    deps.sleep = sleep
    try:
        sch.run_forever(cfg, deps)
    except _Stop:
        pass


def _fetch(clock, ready_from=None, ready_bar=MON, stale_bar=FRI_BEFORE):
    calls = []

    def fetch(symbol):
        calls.append(clock.current.strftime("%H:%M"))
        return ready_bar if ready_from is not None and clock.current >= ready_from else stale_bar
    fetch.calls = calls
    return fetch


def _seed_success(cfg, when, session=MON):
    clock = FakeClock(sgt(when))
    assert sch.check_and_run("manual", cfg, make_deps(clock, FakeScreen(["AAA"]), latest_bar=session)).decision \
        == sch.COMPLETED


TIMES = dict(scheduler_mode="times", scheduler_times="09:15,13:00,18:00")
INTERVAL = dict(scheduler_mode="interval", scheduler_interval_minutes="60")


# ── daily (default) ──────────────────────────────────────────────────────

def test_daily_mode_one_check_at_0915_and_nothing_else_that_day(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False)
    clock = FakeClock(sgt("2026-09-15 08:00"))
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(["AAA"])
    _run_until(cfg, make_deps(clock, screen, latest_bar=MON), clock, sgt("2026-09-15 21:00"))
    assert seen == [("scheduled", "09:15", sch.COMPLETED)]
    assert len(screen.calls) == 1


def test_config_without_schedule_mode_behaves_as_daily(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False)
    for key in ("scheduler_mode", "scheduler_times", "scheduler_interval_minutes"):
        cfg.pop(key)                                    # an old CONFIG / .env with no new keys
    clock = FakeClock(sgt("2026-09-15 08:00"))
    seen = _record_checks(monkeypatch, clock)
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-16 10:00"))
    assert [(t, hm) for t, hm, _ in seen] == [("scheduled", "09:15"), ("scheduled", "09:15")]


# ── times ────────────────────────────────────────────────────────────────

def test_times_mode_each_time_is_a_check_and_stage_a_to_g_runs_once(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, telegram_enabled=True, **TIMES)
    clock = FakeClock(sgt("2026-09-15 08:00"))
    seen = _record_checks(monkeypatch, clock)
    screen, tele = FakeScreen(["AAA", "BBB"]), FakeTelegram()
    _run_until(cfg, make_deps(clock, screen, latest_bar=MON, telegram=tele), clock, sgt("2026-09-15 21:00"))

    assert seen == [("scheduled", "09:15", sch.COMPLETED),
                    ("scheduled", "13:00", sch.SKIPPED_ALREADY_COMPLETED),
                    ("scheduled", "18:00", sch.SKIPPED_ALREADY_COMPLETED)]
    assert len(screen.calls) == 1
    main = [m for m in tele.messages if not m.startswith("🛠")]
    assert len(main) == 1                                   # one daily report (+ its debug summary)
    assert len(tele.messages) == 2
    runs = _store(cfg).recent_runs()
    assert [(r["status"], r["is_canonical"]) for r in runs] == [(hs.NOTIFIED, 1)]
    run_dir = os.path.join(cfg["chart_dir"], screen.calls[0])
    charts = [f for f in os.listdir(run_dir) if f.endswith(".png")]
    assert sorted(charts) == ["AAA.png", "BBB.png"]          # generated once, by the one run
    assert sorted(os.listdir(os.path.join(run_dir, "vcp_debug"))) == ["AAA.png", "BBB.png"]
    assert len(os.listdir(cfg["chart_dir"])) == 1


# ── interval ─────────────────────────────────────────────────────────────

def test_interval_mode_checks_hourly_but_runs_stage_a_to_g_once(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, **INTERVAL)
    clock = FakeClock(sgt("2026-09-15 11:00"))
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(["AAA"])
    _run_until(cfg, make_deps(clock, screen, latest_bar=MON), clock, sgt("2026-09-15 14:30"))

    assert seen == [("startup", "11:00", sch.COMPLETED),
                    ("scheduled", "12:00", sch.SKIPPED_ALREADY_COMPLETED),
                    ("scheduled", "13:00", sch.SKIPPED_ALREADY_COMPLETED),
                    ("scheduled", "14:00", sch.SKIPPED_ALREADY_COMPLETED)]
    assert len(screen.calls) == 1 and len(_store(cfg).recent_runs()) == 1


# ── startup interaction ──────────────────────────────────────────────────

@pytest.mark.parametrize("mode, expected_next", [
    ({}, "2026-09-16T09:15:00+08:00"),
    (TIMES, "2026-09-15T13:00:00+08:00"),
    (INTERVAL, "2026-09-15T12:00:00+08:00"),
])
def test_startup_check_then_next_scheduled_check_per_mode(tmp_path, monkeypatch, caplog, mode, expected_next):
    cfg = make_config(tmp_path, **mode)
    clock = FakeClock(sgt("2026-09-15 11:00"))
    seen = _record_checks(monkeypatch, clock)
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-15 11:30"))
    assert [(t, hm) for t, hm, _ in seen] == [("startup", "11:00")]
    assert f"Next scheduled check: {expected_next}" in caplog.text


def test_occurrence_due_while_the_startup_check_runs_is_not_checked_again(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, **TIMES)
    _seed_success(cfg, "2026-09-15 08:00")
    clock = FakeClock(sgt("2026-09-15 09:14"))
    real = sch.check_and_run

    def slow_check(trigger, config, deps, **kw):                # the startup check straddles 09:15
        outcome = real(trigger, config, deps, **kw)
        if trigger == "startup":
            clock.current = sgt("2026-09-15 09:16")
        return outcome
    monkeypatch.setattr(sch, "check_and_run", slow_check)
    seen = _record_checks(monkeypatch, clock)
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-15 13:30"))
    assert [(t, hm) for t, hm, _ in seen] == [("startup", "09:14"), ("scheduled", "13:00")]


# ── sleep / resume ───────────────────────────────────────────────────────

def test_times_mode_sleep_through_one_time_runs_one_catch_up(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, **TIMES)
    _seed_success(cfg, "2026-09-15 08:00")
    clock = FakeClock(sgt("2026-09-15 09:00"))
    seen = _record_checks(monkeypatch, clock)
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-15 18:30"),
               jump=(sgt("2026-09-15 12:00"), sgt("2026-09-15 17:00")))
    assert [(t, hm) for t, hm, _ in seen] == [("scheduled", "09:15"), ("scheduled", "17:00"),
                                             ("scheduled", "18:00")]
    assert "Missed scheduled check due 2026-09-15T13:00:00+08:00" in caplog.text


def test_times_mode_sleep_through_several_times_collapses_to_one_check(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, **TIMES)
    _seed_success(cfg, "2026-09-15 08:00")
    clock = FakeClock(sgt("2026-09-15 09:00"))
    seen = _record_checks(monkeypatch, clock)
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-15 20:00"),
               jump=(sgt("2026-09-15 12:00"), sgt("2026-09-15 19:00")))
    assert [(t, hm) for t, hm, _ in seen] == [("scheduled", "09:15"), ("scheduled", "19:00")]
    assert "2 missed occurrences collapsed into one check" in caplog.text


def test_daily_mode_sleep_through_0915_runs_one_catch_up(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False)
    clock = FakeClock(sgt("2026-09-15 08:00"))
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(["AAA"])
    _run_until(cfg, make_deps(clock, screen, latest_bar=MON), clock, sgt("2026-09-15 20:00"),
               jump=(sgt("2026-09-15 09:00"), sgt("2026-09-15 15:20")))
    assert seen == [("scheduled", "15:20", sch.COMPLETED)]
    assert len(screen.calls) == 1


def test_interval_mode_sleep_runs_one_catch_up_then_rebases(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path, **INTERVAL)
    _seed_success(cfg, "2026-09-15 08:00")
    clock = FakeClock(sgt("2026-09-15 11:00"))
    seen = _record_checks(monkeypatch, clock)
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, latest_bar=MON), clock, sgt("2026-09-15 18:10"),
               jump=(sgt("2026-09-15 12:30"), sgt("2026-09-15 17:05")))
    assert [(t, hm) for t, hm, _ in seen] == [("startup", "11:00"), ("scheduled", "12:00"),
                                             ("scheduled", "17:05"), ("scheduled", "18:05")]
    assert "5 missed occurrences collapsed into one check" in caplog.text


# ── readiness episode: one retry stream, one DATA_NOT_READY row ──────────

def test_times_check_during_pending_retry_joins_the_episode(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, **TIMES)
    clock = FakeClock(sgt("2026-09-15 09:00"))
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(["AAA"])
    fetch = _fetch(clock, ready_from=sgt("2026-09-15 13:30"))
    _run_until(cfg, make_deps(clock, screen, latest_bar=fetch), clock, sgt("2026-09-15 19:00"))

    # short retries, deferred probes, the 13:00 main check as a probe (which re-bases the
    # pending retry to 14:00), then ready -> Stage A-G once; 18:00 just skips.
    assert fetch.calls == ["09:15", "09:25", "09:35", "10:35", "11:35", "12:35", "13:00", "14:00"]
    assert len(screen.calls) == 1
    assert [(t, hm) for t, hm, _ in seen] == [("scheduled", "09:15"), ("deferred", "10:35"), ("deferred", "11:35"),
                                             ("deferred", "12:35"), ("scheduled", "13:00"), ("deferred", "14:00"),
                                             ("scheduled", "18:00")]
    store = _store(cfg)
    assert sorted(r["status"] for r in store.recent_runs()) == [hs.DATA_NOT_READY, hs.RESULTS_PERSISTED]
    assert store.canonical_run(MON)["trigger"] == "deferred"


def test_interval_never_ready_is_one_stream_to_cutoff_then_on_hold(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path, telegram_enabled=True, **INTERVAL)
    clock = FakeClock(sgt("2026-09-15 09:15"))
    seen = _record_checks(monkeypatch, clock)
    screen, tele = FakeScreen(["AAA"]), FakeTelegram()
    fetch = _fetch(clock)                                         # never ready
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, screen, latest_bar=fetch, telegram=tele), clock, sgt("2026-09-16 04:50"))

    # Identical probe times to the daily-mode pattern: hourly main checks merge into
    # the deferred stream instead of adding a second one.
    assert fetch.calls == ["09:15", "09:25", "09:35"] + [f"{h}:35" for h in range(10, 20)] + ["20:30"]
    assert screen.calls == []
    assert len(tele.messages) == 1 and "20:30 cutoff" in tele.messages[0]    # one alert
    statuses = [r["status"] for r in _store(cfg).recent_runs()]
    # One DATA_NOT_READY for the episode; during Tuesday's US session (after local
    # midnight ends the hold) exactly one DEFERRED_MARKET_OPEN row, not one per hour.
    assert sorted(statuses) == [hs.DATA_NOT_READY, hs.DEFERRED_MARKET_OPEN]
    # After the cutoff, main checks 20:35-23:35 are held (no claim, no short retries, no alert) ...
    assert caplog.text.count("is on hold for today (data still not ready at the 20:30 cutoff)") == 4
    # ... and after local midnight they only meet the market-open guard.
    assert [hm for t, hm, d in seen if d == sch.DEFERRED_MARKET_OPEN] == ["00:35", "01:35", "02:35", "03:35", "04:35"]


def test_pipeline_failure_is_not_retried_by_later_main_checks_same_day(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, scheduler_mode="times",
                      scheduler_times="09:15,13:00,18:00")
    clock = FakeClock(sgt("2026-09-19 09:00"))                   # Saturday; Friday's session pending
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(RuntimeError("boom"), ["AAA"])
    caplog.set_level(logging.INFO, logger="pipeline")
    _run_until(cfg, make_deps(clock, screen, latest_bar=SAT_FRI), clock, sgt("2026-09-20 10:00"))

    # 13:00 and 18:00 are held without calling check_and_run; Sunday is a new local day.
    assert [(t, hm, d) for t, hm, d in seen] == [("scheduled", "09:15", sch.PIPELINE_FAILED),
                                                ("scheduled", "09:15", sch.COMPLETED)]
    assert caplog.text.count("session 2026-09-18 is on hold for today (Stage A-G failed (PIPELINE_FAILED))") == 2
    assert len(screen.calls) == 2
    assert _store(cfg).canonical_run(SAT_FRI) is not None


def test_hold_is_ignored_once_the_session_completed_elsewhere(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, **TIMES)
    clock = FakeClock(sgt("2026-09-19 09:00"))
    seen = _record_checks(monkeypatch, clock)
    deps = make_deps(clock, FakeScreen(RuntimeError("boom")), latest_bar=SAT_FRI)
    real_sleep = clock.sleep
    manual = {"done": False}

    def sleep(seconds):
        real_sleep(seconds)
        if not manual["done"] and clock.current >= sgt("2026-09-19 10:00"):
            manual["done"] = True
            _seed_success(cfg, "2026-09-19 10:00", session=SAT_FRI)
        if clock.current >= sgt("2026-09-19 13:30"):
            raise _Stop
    deps.sleep = sleep
    with pytest.raises(_Stop):
        sch.run_forever(cfg, deps)
    assert [d for t, hm, d in seen if t == "scheduled"] == [sch.PIPELINE_FAILED, sch.SKIPPED_ALREADY_COMPLETED]


# ── market-open guard in every mode ──────────────────────────────────────

@pytest.mark.parametrize("mode", [
    dict(scheduler_mode="daily", scheduler_local_time="22:00"),
    dict(scheduler_mode="times", scheduler_times="22:00,23:00"),
    dict(scheduler_mode="interval", scheduler_interval_minutes="30"),
])
def test_all_modes_defer_during_us_session_with_one_audit_row(tmp_path, monkeypatch, mode):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, **mode)
    clock = FakeClock(sgt("2026-09-15 21:45"))                   # Tue US session opens 21:30 SGT; Mon pending
    seen = _record_checks(monkeypatch, clock)
    screen = FakeScreen(["AAA"])
    fetch = _fetch(clock, ready_from=clock.current)
    _run_until(cfg, make_deps(clock, screen, latest_bar=fetch), clock, sgt("2026-09-15 23:50"))

    assert seen and all(d == sch.DEFERRED_MARKET_OPEN for _, _, d in seen)
    assert screen.calls == [] and fetch.calls == []               # no readiness download, no Stage A-G
    assert [r["status"] for r in _store(cfg).recent_runs()] == [hs.DEFERRED_MARKET_OPEN]


# ── invalid configuration ────────────────────────────────────────────────

@pytest.mark.parametrize("overrides", [
    dict(scheduler_mode="banana"),
    dict(scheduler_mode="interval", scheduler_interval_minutes="0"),
    dict(scheduler_mode="interval", scheduler_interval_minutes="5"),
    dict(scheduler_mode="times", scheduler_times="25:00"),
    dict(scheduler_mode="times", scheduler_times="abc"),
])
def test_invalid_schedule_fails_fast_before_any_check_or_lock(tmp_path, monkeypatch, overrides):
    cfg = make_config(tmp_path, **overrides)
    clock = FakeClock(sgt("2026-09-15 09:00"))
    seen = _record_checks(monkeypatch, clock)
    with pytest.raises(cs.ScheduleConfigError):
        sch.run_forever(cfg, make_deps(clock, latest_bar=MON), max_iterations=5)
    assert seen == []
    assert not os.path.exists(os.path.join(cfg["history_lock_dir"], sch.INSTANCE_LOCK_NAME))


def test_cli_loop_and_dry_run_exit_2_on_invalid_schedule(tmp_path):
    import io
    cfg = make_config(tmp_path, scheduler_mode="times", scheduler_times="09:15,abc")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.addHandler(handler)       # setup_logging disables propagation to caplog's root handler
    try:
        assert sch.main([], config=cfg) == 2
        assert sch.main(["--dry-run"], config=cfg) == 2
    finally:
        pipeline_logger.removeHandler(handler)
    assert stream.getvalue().count("Invalid scheduler configuration: SCHEDULE_TIMES entry='abc' is not a "
                                   "valid HH:MM time") == 2


def test_dry_run_and_status_report_the_schedule(tmp_path, caplog):
    cfg = make_config(tmp_path, **TIMES)
    st = sch.build_status(cfg, sgt("2026-09-15 11:00"), None)
    text = "\n".join(sch.format_status(st, cfg))
    assert "Schedule mode" in text and ": times" in text
    assert "Scheduled checks              : 09:15, 13:00, 18:00" in text
    assert "Next scheduled check          : 2026-09-15 13:00 Tue" in text

    icfg = make_config(tmp_path, **INTERVAL)
    text = "\n".join(sch.format_status(sch.build_status(icfg, sgt("2026-09-15 11:00"), None), icfg))
    assert "Interval                      : 60 minutes" in text
    assert "loop state is not visible here" in text

    bad = make_config(tmp_path, scheduler_mode="banana")
    text = "\n".join(sch.format_status(sch.build_status(bad, sgt("2026-09-15 11:00"), None), bad))
    assert "Schedule                      : INVALID: SCHEDULE_MODE='banana'" in text


def test_history_cli_status_shows_schedule_and_exits_2_when_invalid(tmp_path):
    import io
    from pipeline import history_cli
    cfg = make_config(tmp_path, **TIMES)
    out = io.StringIO()
    assert history_cli.main(["--db", cfg["history_db_path"], "status"], config=cfg, out=out) == 0
    assert "Schedule mode" in out.getvalue() and "09:15, 13:00, 18:00" in out.getvalue()
    bad = make_config(tmp_path, scheduler_mode="interval", scheduler_interval_minutes="5")
    out = io.StringIO()
    assert history_cli.main(["--db", bad["history_db_path"], "status"], config=bad, out=out) == 2
    assert "INVALID: SCHEDULE_INTERVAL_MINUTES=5 is outside the supported range 15-1440" in out.getvalue()
