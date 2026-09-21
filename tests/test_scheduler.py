"""Stage H: scheduler decisions — startup catch-up, weekend handling, the
one-session automatic catch-up limit, data readiness (no real sleeps),
locking/idempotency, --force, and Telegram-failure isolation."""
import json
import logging
import os
from datetime import date, timedelta

import pytest

from pipeline import history_store as hs
from pipeline import readiness
from pipeline import scheduler as sch
from pipeline.run_lock import FileLock
from stage_h_fakes import (FAKE_TOKEN, FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt)

MON = date(2026, 9, 14)
FRI = date(2026, 9, 18)


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def _store(cfg):
    return hs.HistoryStore(cfg["history_db_path"])


# ── startup catch-up ─────────────────────────────────────────────────────

def test_startup_with_latest_session_already_successful_runs_nothing(cfg):
    clock = FakeClock(sgt("2026-09-15 09:00"))
    screen = FakeScreen(["AAA", "BBB"])
    deps = make_deps(clock, screen, latest_bar=MON)
    assert sch.check_and_run("startup", cfg, deps).decision == sch.COMPLETED

    again = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=MON))
    assert again.decision == sch.SKIPPED_ALREADY_COMPLETED
    assert len(screen.calls) == 1


def test_late_startup_catches_up_missing_session_once(cfg):
    clock = FakeClock(sgt("2026-09-15 11:37"))
    screen = FakeScreen(["AAA", "BBB", "CCC"])
    outcome = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=MON))
    assert outcome.decision == sch.COMPLETED and outcome.trading_date == MON
    assert len(screen.calls) == 1
    assert _store(cfg).has_successful_run(MON)


def test_scheduled_check_later_same_day_does_not_duplicate(cfg):
    clock = FakeClock(sgt("2026-09-15 09:00"))
    screen = FakeScreen(["AAA"])
    sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=MON))
    clock.current = sgt("2026-09-15 09:15")
    assert sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON)).decision \
        == sch.SKIPPED_ALREADY_COMPLETED
    assert len(screen.calls) == 1
    assert len(_store(cfg).recent_runs()) == 1


def test_saturday_morning_processes_friday(cfg):
    clock = FakeClock(sgt("2026-09-19 09:15"))
    outcome = sch.check_and_run("scheduled", cfg, make_deps(clock, latest_bar=FRI))
    assert outcome.decision == sch.COMPLETED and outcome.trading_date == FRI


def test_sunday_startup_processes_unprocessed_friday_once(cfg):
    screen = FakeScreen(["AAA"])
    clock = FakeClock(sgt("2026-09-20 14:02"))
    assert sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=FRI)).trading_date == FRI
    clock.current = sgt("2026-09-21 09:15")   # Monday SGT: U.S. Monday hasn't traded yet
    assert sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=FRI)).decision \
        == sch.SKIPPED_ALREADY_COMPLETED
    assert len(screen.calls) == 1


def test_laptop_absent_one_day_processes_latest_missing_session(cfg):
    screen = FakeScreen(["AAA"])
    clock = FakeClock(sgt("2026-09-15 09:15"))
    sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON))
    clock.current = sgt("2026-09-17 10:05")   # Wednesday SGT missed; Thursday SGT startup
    outcome = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=date(2026, 9, 16)))
    assert outcome.trading_date == date(2026, 9, 16)
    assert _store(cfg).canonical_sessions() == [MON, date(2026, 9, 16)]   # 09-15 not backfilled


def test_laptop_absent_one_week_runs_only_the_latest_session(cfg, caplog):
    screen = FakeScreen(["AAA"])
    clock = FakeClock(sgt("2026-09-08 09:15"))
    sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=date(2026, 9, 4)))
    clock.current = sgt("2026-09-19 09:30")
    caplog.set_level(logging.INFO, logger="pipeline")
    outcome = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=FRI))
    assert outcome.trading_date == FRI
    assert len(screen.calls) == 2                     # one per check, never a replay of the backlog
    assert _store(cfg).canonical_sessions() == [date(2026, 9, 4), FRI]
    assert "will not be backfilled" in caplog.text


def test_session_in_progress_is_skipped(cfg):
    clock = FakeClock(sgt("2026-09-15 22:00"))      # 10:00 ET Tuesday
    screen = FakeScreen(["AAA"])
    outcome = sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON))
    assert outcome.decision == sch.DEFERRED_MARKET_OPEN
    assert screen.calls == []
    assert [r["status"] for r in _store(cfg).recent_runs()] == [hs.DEFERRED_MARKET_OPEN]


def test_dry_run_decides_but_runs_nothing(cfg):
    screen = FakeScreen(["AAA"])
    outcome = sch.check_and_run("manual", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), screen,
                                                        latest_bar=MON), dry_run=True)
    assert outcome.decision == sch.WOULD_RUN and screen.calls == []


# ── data readiness ───────────────────────────────────────────────────────

def test_readiness_ready_when_latest_bar_matches(cfg):
    sleeps = []
    r = readiness.wait_for_data(MON, cfg, fetch=lambda s: MON, sleep=sleeps.append)
    assert r.ready and r.attempts == 1 and sleeps == []


def test_readiness_stale_then_ready_on_retry_without_real_sleep(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA"])
    outcome = sch.check_and_run("scheduled", cfg, make_deps(
        clock, screen, latest_bar=[date(2026, 9, 11), date(2026, 9, 11), MON]))
    assert outcome.decision == sch.COMPLETED
    assert clock.sleeps == [600.0, 600.0]
    run = _store(cfg).get_run(outcome.run_id)
    assert run["data_ready_attempts"] == 3 and run["data_ready_latest_bar"] == "2026-09-14"


def test_never_ready_executes_no_stage_a_to_g_and_records_state(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA"])
    outcome = sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=date(2026, 9, 11)))
    assert outcome.decision == sch.DATA_NOT_READY
    assert screen.calls == []
    run = _store(cfg).get_run(outcome.run_id)
    assert run["status"] == hs.DATA_NOT_READY and run["is_canonical"] == 0
    assert run["data_ready_attempts"] == 3
    assert not _store(cfg).has_successful_run(MON)


def test_newer_bar_than_target_stops_waiting_immediately(cfg):
    sleeps = []
    r = readiness.wait_for_data(MON, cfg, fetch=lambda s: date(2026, 9, 15), sleep=sleeps.append)
    assert not r.ready and r.attempts == 1 and sleeps == []


def test_readiness_fetch_exception_is_not_ready_not_a_crash(cfg):
    def boom(symbol):
        raise ConnectionError("down")
    r = readiness.wait_for_data(MON, {**cfg, "data_ready_max_attempts": 2}, fetch=boom, sleep=lambda s: None)
    assert not r.ready and "fetch failed" in r.reason


def test_data_not_ready_sends_ops_alert_when_enabled(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    tele = FakeTelegram()
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")),
                                                           latest_bar=date(2026, 9, 11), telegram=tele))
    assert outcome.decision == sch.DATA_NOT_READY
    assert len(tele.messages) == 1 and "Market data not ready" in tele.messages[0]
    assert "Latest available bar: 2026-09-11" in tele.messages[0]


# ── locking / idempotency ────────────────────────────────────────────────

def test_second_instance_holding_run_lock_prevents_duplicate_work(cfg):
    held = FileLock(os.path.join(cfg["history_lock_dir"], sch.RUN_LOCK_NAME)).acquire()
    try:
        screen = FakeScreen(["AAA"])
        outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), screen,
                                                               latest_bar=MON))
        assert outcome.decision == sch.SKIPPED_LOCKED and screen.calls == []
    finally:
        held.release()


def test_active_claim_in_database_blocks_second_process(cfg):
    _store(cfg).claim_session(MON, "startup", now=sgt("2026-09-15 09:10"))
    screen = FakeScreen(["AAA"])
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), screen,
                                                           latest_bar=MON))
    assert outcome.decision == sch.SKIPPED_CLAIMED_ELSEWHERE and screen.calls == []


def test_second_scheduler_loop_exits_cleanly_when_one_is_already_running(cfg):
    held = FileLock(os.path.join(cfg["history_lock_dir"], sch.INSTANCE_LOCK_NAME)).acquire()
    try:
        screen = FakeScreen(["AAA"])
        rc = sch.run_forever(cfg, make_deps(FakeClock(sgt("2026-09-15 09:00")), screen, latest_bar=MON),
                             max_iterations=5)
        assert rc == 0 and screen.calls == []       # watchdog "already running" is success
    finally:
        held.release()


def test_force_reruns_a_completed_session_and_supersedes_it(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA", "BBB"], ["BBB", "AAA"])
    first = sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON))
    forced = sch.check_and_run("manual", cfg, make_deps(clock, screen, latest_bar=MON), force=True)
    assert forced.decision == sch.COMPLETED and len(screen.calls) == 2
    store = _store(cfg)
    assert store.canonical_run(MON)["run_id"] == forced.run_id
    assert store.get_run(first.run_id)["superseded_by_run_id"] == forced.run_id
    assert [r["symbol"] for r in store.results_for_sessions([MON])] == ["BBB", "AAA"]


def test_force_does_not_bypass_the_data_readiness_guard(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA"])
    sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON))
    outcome = sch.check_and_run("manual", cfg, make_deps(clock, screen, latest_bar=date(2026, 9, 11)), force=True)
    assert outcome.decision == sch.DATA_NOT_READY and len(screen.calls) == 1
    assert _store(cfg).has_successful_run(MON)       # original canonical result untouched


# ── pipeline failure handling ────────────────────────────────────────────

def test_pipeline_exception_is_recorded_with_stage_and_not_canonical(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    tele = FakeTelegram()
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-never-leak"
    try:
        screen = FakeScreen(RuntimeError("yfinance exploded with sk-ant-should-never-leak"))
        outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), screen,
                                                               latest_bar=MON, telegram=tele))
    finally:
        del os.environ["ANTHROPIC_API_KEY"]
    assert outcome.decision == sch.PIPELINE_FAILED
    run = _store(cfg).get_run(outcome.run_id)
    assert run["status"] == hs.PIPELINE_FAILED and run["failed_stage"] == "D" and run["is_canonical"] == 0
    assert "sk-ant" not in run["error_summary"] and "***" in run["error_summary"]
    assert "Stage 2 Pipeline Failed" in tele.messages[0] and "sk-ant" not in tele.messages[0]
    manifest = json.loads(open(os.path.join(cfg["run_manifest_dir"], "2026-09-14", "run.json")).read())
    assert manifest["status"] == hs.PIPELINE_FAILED


def test_empty_ranked_result_is_a_retryable_failure(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(None, ["AAA"])
    assert sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON)).decision == sch.PIPELINE_FAILED
    assert sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON)).decision == sch.COMPLETED


def test_ops_alert_failure_does_not_mask_pipeline_failure(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    outcome = sch.check_and_run("scheduled", cfg, make_deps(
        FakeClock(sgt("2026-09-15 09:15")), FakeScreen(ValueError("bad")), latest_bar=MON,
        telegram=FakeTelegram(permanent_failure=True)))
    assert outcome.decision == sch.PIPELINE_FAILED
    notes = _store(cfg).notifications_for_run(outcome.run_id)
    assert notes[0]["kind"] == "ops_alert" and notes[0]["status"] == "FAILED"


# ── Telegram isolation ───────────────────────────────────────────────────

def test_telegram_disabled_leaves_run_successful_and_records_skip(cfg):
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), latest_bar=MON))
    store = _store(cfg)
    assert outcome.notification_status == "SKIPPED"
    assert store.get_run(outcome.run_id)["status"] == hs.RESULTS_PERSISTED
    assert store.notifications_for_run(outcome.run_id)[0]["status"] == "SKIPPED"


def test_telegram_failure_keeps_pipeline_and_history_successful(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    outcome = sch.check_and_run("scheduled", cfg, make_deps(
        FakeClock(sgt("2026-09-15 09:15")), latest_bar=MON, telegram=FakeTelegram(permanent_failure=True)))
    store = _store(cfg)
    assert outcome.decision == sch.COMPLETED and outcome.notification_status == "FAILED"
    run = store.get_run(outcome.run_id)
    assert run["status"] == hs.NOTIFICATION_FAILED and run["is_canonical"] == 1
    assert store.has_successful_run(MON)
    assert len(store.results_for_run(outcome.run_id)) == 4


def test_missing_telegram_credentials_is_notification_failure_not_crash(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), latest_bar=MON,
                                                           telegram=None))
    assert outcome.decision == sch.COMPLETED and outcome.notification_status == "FAILED"


def test_resend_uses_stored_results_without_rerunning_stage_a_to_g(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True)
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA", "BBB"])
    first = sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON,
                                                         telegram=FakeTelegram(permanent_failure=True)))
    assert first.notification_status == "FAILED"

    tele = FakeTelegram()
    status = sch.resend_notification(cfg, make_deps(clock, screen, latest_bar=MON, telegram=tele), "2026-09-14")
    assert status == "SENT" and len(screen.calls) == 1
    assert "Trading session: <b>2026-09-14</b>" in tele.messages[0]
    assert _store(cfg).get_run(first.run_id)["status"] == hs.NOTIFIED


def test_chart_send_failure_does_not_fail_text_notification(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True, telegram_send_charts=True, telegram_vcp_chart_max=2)
    tele = FakeTelegram(fail_photos=True)
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")), latest_bar=MON,
                                                           telegram=tele))
    assert outcome.notification_status == "SENT"
    kinds = [(n["kind"], n["status"]) for n in _store(cfg).notifications_for_run(outcome.run_id)]
    assert ("report", "SENT") in kinds and kinds.count(("chart", "FAILED")) == 2


# ── loop ─────────────────────────────────────────────────────────────────

def test_next_trigger_is_every_calendar_day_including_weekends():
    t = sch.parse_local_time("09:15")
    fri_after = sch.next_trigger_after(sgt("2026-09-18 10:00"), t, "Asia/Singapore")
    assert fri_after == sgt("2026-09-19 09:15")          # Saturday
    assert sch.next_trigger_after(sgt("2026-09-19 09:15"), t, "Asia/Singapore") == sgt("2026-09-20 09:15")
    assert sch.next_trigger_after(sgt("2026-09-20 08:00"), t, "Asia/Singapore") == sgt("2026-09-20 09:15")


def test_loop_startup_check_then_scheduled_check_after_sleep(cfg):
    clock = FakeClock(sgt("2026-09-18 20:00"))           # Friday evening SGT = 08:00 ET, before the open
    screen = FakeScreen(["AAA"])
    latest = {"bar": date(2026, 9, 17)}
    deps = make_deps(clock, screen, latest_bar=lambda s: latest["bar"])

    original_sleep = clock.sleep

    def sleep_and_advance_market(seconds):
        original_sleep(seconds)
        if clock.current >= sgt("2026-09-19 05:00"):
            latest["bar"] = FRI
    deps.sleep = sleep_and_advance_market

    sch.run_forever(cfg, deps, max_iterations=2000)
    store = _store(cfg)
    # Startup processed Thursday; the Saturday 09:15 check processed Friday.
    assert store.canonical_sessions() == [date(2026, 9, 17), FRI]
    assert len(screen.calls) == 2
    friday_run = store.canonical_run(FRI)
    assert friday_run["trigger"] == "scheduled"
    assert friday_run["local_started_at"].startswith("2026-09-19T09:15")


def test_loop_disabled_does_not_run(tmp_path):
    cfg = make_config(tmp_path, scheduler_enabled=False)
    screen = FakeScreen(["AAA"])
    assert sch.run_forever(cfg, make_deps(FakeClock(sgt("2026-09-15 09:00")), screen, latest_bar=MON)) == 0
    assert screen.calls == []


def test_unexpected_error_in_a_check_does_not_kill_the_loop(cfg, monkeypatch):
    calls = []

    def flaky(trigger, config, deps, **kw):
        calls.append(trigger)
        raise RuntimeError("bug")

    monkeypatch.setattr(sch, "check_and_run", flaky)
    clock = FakeClock(sgt("2026-09-15 09:00"))
    sch.run_forever(cfg, make_deps(clock, latest_bar=MON), max_iterations=100)
    assert calls[:2] == ["startup", "scheduled"]


def test_resume_after_sleeping_through_trigger_runs_missed_check_and_logs_it(cfg, caplog):
    """Laptop asleep 08:50-10:40 SGT across the 09:15 trigger (Tue; Monday session unprocessed)."""
    clock = FakeClock(sgt("2026-09-15 08:50"))
    screen = FakeScreen(["AAA"])
    deps = make_deps(clock, screen, latest_bar=MON)
    cfg = {**cfg, "scheduler_run_on_startup": False}
    real_sleep = clock.sleep
    slept = {"done": False}

    def sleep(seconds):
        if not slept["done"] and clock.current >= sgt("2026-09-15 08:52"):
            slept["done"] = True
            clock.current = sgt("2026-09-15 10:40")      # system suspend: no wake-up at 09:15
            return
        real_sleep(seconds)
    deps.sleep = sleep

    caplog.set_level(logging.INFO, logger="pipeline")
    sch.run_forever(cfg, deps, max_iterations=10)
    assert len(screen.calls) == 1
    run = _store(cfg).canonical_run(MON)
    assert run["trigger"] == "scheduled" and run["local_started_at"].startswith("2026-09-15T10:40")
    assert "Missed scheduled check due 2026-09-15T09:15:00+08:00" in caplog.text
