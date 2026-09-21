"""Stage H: late-provider deferred retries (bounded by a local cutoff), sleep
through a pending retry, US-market-open deferral and next-morning recovery.

All loop tests drive scheduler.run_forever with a fake clock: sleeping advances
time instantly, readiness answers depend on the fake time, and Stage A-G is a
recorder. Nothing waits or touches the network."""
import logging
from datetime import date

import pytest

from pipeline import history_store as hs
from pipeline import scheduler as sch
from stage_h_fakes import FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt

FRI_BEFORE = date(2026, 9, 11)
MON = date(2026, 9, 14)
TUE = date(2026, 9, 15)


def _store(cfg):
    return hs.HistoryStore(cfg["history_db_path"])


def _fetch(clock, ready_from=None, ready_bar=MON, stale_bar=FRI_BEFORE):
    """Readiness source whose answer depends on fake time; records call times."""
    calls = []

    def fetch(symbol):
        calls.append(clock.current)
        if ready_from is not None and clock.current >= ready_from:
            return ready_bar
        return stale_bar
    fetch.calls = calls
    return fetch


def _hhmm(times):
    return [t.strftime("%H:%M") for t in times]


def _sleep_jump(clock, at, to):
    """Replace clock.sleep so the machine 'suspends' once: at `at` it jumps to `to`."""
    real = clock.sleep
    state = {"done": False}

    def sleep(seconds):
        if not state["done"] and clock.current >= at:
            state["done"] = True
            clock.current = to
            return
        real(seconds)
    return sleep


# ── 1. late provider ──────────────────────────────────────────────────────

def test_late_provider_ready_on_deferred_retry_runs_pipeline_exactly_once(tmp_path, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False)
    clock = FakeClock(sgt("2026-09-15 09:14"))
    screen = FakeScreen(["AAA", "BBB"])
    fetch = _fetch(clock, ready_from=sgt("2026-09-15 11:00"))
    caplog.set_level(logging.INFO, logger="pipeline")

    sch.run_forever(cfg, make_deps(clock, screen, latest_bar=fetch), max_iterations=400)

    # 09:15/09:25/09:35 short retries, then hourly deferred probes; ready at the first probe after 11:00.
    assert _hhmm(fetch.calls) == ["09:15", "09:25", "09:35", "10:35", "11:35"]
    assert len(screen.calls) == 1
    store = _store(cfg)
    run = store.canonical_run(MON)
    assert run["trigger"] == "deferred" and run["local_started_at"].startswith("2026-09-15T11:35")
    assert sorted(r["status"] for r in store.recent_runs()) == [hs.DATA_NOT_READY, hs.RESULTS_PERSISTED]
    assert "Target session 2026-09-14 still not ready after initial retries" in caplog.text
    assert "entering deferred retry mode. Deferred retry scheduled for 10:35" in caplog.text
    assert "Deferred retries for 2026-09-14 cancelled: session completed." in caplog.text


# ── 2. never ready before cutoff ──────────────────────────────────────────

def test_never_ready_stops_at_cutoff_without_running_and_session_stays_pending(tmp_path, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False, telegram_enabled=True)
    clock = FakeClock(sgt("2026-09-15 09:14"))
    screen = FakeScreen(["AAA"])
    tele = FakeTelegram()
    fetch = _fetch(clock)                                   # never ready
    caplog.set_level(logging.INFO, logger="pipeline")

    deps = make_deps(clock, screen, latest_bar=fetch, telegram=tele)
    sch.run_forever(cfg, deps, max_iterations=760)          # runs to ~21:00 SGT, before the US open

    assert screen.calls == []
    assert _hhmm(fetch.calls) == ["09:15", "09:25", "09:35"] + [f"{h}:35" for h in range(10, 20)] + ["20:30"]
    assert "Target session 2026-09-14 still not ready at 20:30 SGT. Deferring until the next safe " \
           "scheduler/startup check." in caplog.text
    # exactly one ops alert, at the cutoff — not one per attempt
    assert len(tele.messages) == 1 and "Market data not ready" in tele.messages[0]
    assert "20:30 cutoff" in tele.messages[0]
    store = _store(cfg)
    assert not store.has_successful_run(MON)
    assert [r["status"] for r in store.recent_runs()] == [hs.DATA_NOT_READY]   # probes write no rows

    # The session is not permanently failed: a later safe startup check picks it up.
    clock.current = sgt("2026-09-15 21:10")                 # 09:10 ET, before the open
    later = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=MON))
    assert later.decision == sch.COMPLETED and later.trading_date == MON
    assert len(screen.calls) == 1


# ── 3. sleep through a deferred retry ─────────────────────────────────────

def test_sleep_through_deferred_retry_runs_one_catch_up_probe_not_every_missed_one(tmp_path, caplog):
    cfg = make_config(tmp_path, data_ready_max_attempts=1)
    clock = FakeClock(sgt("2026-09-15 12:00"))
    screen = FakeScreen(["AAA"])
    fetch = _fetch(clock)
    deps = make_deps(clock, screen, latest_bar=fetch)
    deps.sleep = _sleep_jump(clock, at=sgt("2026-09-15 12:30"), to=sgt("2026-09-15 15:20"))
    caplog.set_level(logging.INFO, logger="pipeline")

    sch.run_forever(cfg, deps, max_iterations=50)           # stops well before the 16:20 retry

    assert _hhmm(fetch.calls) == ["12:00", "15:20"]          # nothing replayed for 13:00 / 14:00 / 15:00
    assert "Missed deferred retry due 2026-09-15T13:00:00+08:00" in caplog.text
    assert "Next deferred retry at 16:20" in caplog.text
    assert screen.calls == []


def test_waking_after_cutoff_with_a_stale_pending_retry_does_not_probe(tmp_path, caplog):
    cfg = make_config(tmp_path, data_ready_max_attempts=1)
    clock = FakeClock(sgt("2026-09-15 19:00"))
    fetch = _fetch(clock)
    deps = make_deps(clock, FakeScreen(["AAA"]), latest_bar=fetch)
    deps.sleep = _sleep_jump(clock, at=sgt("2026-09-15 19:10"), to=sgt("2026-09-15 20:50"))
    caplog.set_level(logging.INFO, logger="pipeline")

    sch.run_forever(cfg, deps, max_iterations=30)

    assert _hhmm(fetch.calls) == ["19:00"]
    assert "cutoff has passed" in caplog.text


# ── 4. wake during U.S. market hours ──────────────────────────────────────

def test_wake_during_us_market_hours_defers_without_stage_a_to_g(tmp_path, caplog):
    cfg = make_config(tmp_path)
    clock = FakeClock(sgt("2026-09-15 22:00"))              # 10:00 ET Tuesday; Monday unprocessed
    screen = FakeScreen(["AAA"])
    fetch = _fetch(clock, ready_from=clock.current)
    caplog.set_level(logging.INFO, logger="pipeline")

    sch.run_forever(cfg, make_deps(clock, screen, latest_bar=fetch), max_iterations=5)

    assert screen.calls == [] and fetch.calls == []          # not even a readiness download
    rows = _store(cfg).recent_runs()
    assert [(r["trading_date"], r["status"], r["is_canonical"]) for r in rows] == \
        [("2026-09-14", hs.DEFERRED_MARKET_OPEN, 0)]
    assert "US market currently open" in caplog.text
    assert "screening of 2026-09-14 deferred to next safe check" in caplog.text
    assert "deferred retry mode" not in caplog.text          # market-open is not a readiness retry


def test_market_open_on_an_already_completed_session_is_just_a_skip(tmp_path):
    cfg = make_config(tmp_path)
    clock = FakeClock(sgt("2026-09-15 09:15"))
    screen = FakeScreen(["AAA"])
    sch.check_and_run("scheduled", cfg, make_deps(clock, screen, latest_bar=MON))
    clock.current = sgt("2026-09-15 22:00")
    outcome = sch.check_and_run("startup", cfg, make_deps(clock, screen, latest_bar=MON))
    assert outcome.decision == sch.SKIPPED_ALREADY_COMPLETED
    assert all(r["status"] != hs.DEFERRED_MARKET_OPEN for r in _store(cfg).recent_runs())


# ── 5. next-morning recovery ──────────────────────────────────────────────

def test_next_morning_after_market_open_deferral_runs_latest_session_once(tmp_path, caplog):
    cfg = make_config(tmp_path, scheduler_run_on_startup=False)
    screen = FakeScreen(["AAA"])

    evening = FakeClock(sgt("2026-09-15 22:00"))            # Monday unresolved, US Tuesday trading
    assert sch.check_and_run("startup", cfg, make_deps(evening, screen, latest_bar=MON)).decision \
        == sch.DEFERRED_MARKET_OPEN

    morning = FakeClock(sgt("2026-09-16 09:14"))
    fetch = _fetch(morning, ready_from=morning.current, ready_bar=TUE)
    caplog.set_level(logging.INFO, logger="pipeline")
    sch.run_forever(cfg, make_deps(morning, screen, latest_bar=fetch), max_iterations=120)

    store = _store(cfg)
    # The latest completed session is now Tuesday; it runs exactly once. Monday is
    # not replayed (one-session catch-up) — its deferral row documents why it's missing.
    assert len(screen.calls) == 1
    assert store.canonical_sessions() == [TUE]
    assert store.canonical_run(TUE)["trigger"] == "scheduled"
    assert _hhmm(fetch.calls) == ["09:15"]
    assert [r["status"] for r in store.recent_runs() if r["trading_date"] == "2026-09-14"] == \
        [hs.DEFERRED_MARKET_OPEN]


# ── 6. already successful ─────────────────────────────────────────────────

def test_pending_retry_after_session_completed_elsewhere_does_nothing(tmp_path, caplog):
    cfg = make_config(tmp_path, data_ready_max_attempts=1)
    clock = FakeClock(sgt("2026-09-15 12:00"))
    loop_screen = FakeScreen(["AAA"])
    manual_screen = FakeScreen(["AAA"])
    fetch = _fetch(clock)
    deps = make_deps(clock, loop_screen, latest_bar=fetch)

    real_sleep = clock.sleep
    done = {"manual": False}

    def sleep(seconds):
        real_sleep(seconds)
        if not done["manual"] and clock.current >= sgt("2026-09-15 12:30"):
            done["manual"] = True           # e.g. a manual --run-once from another terminal succeeds
            outcome = sch.check_and_run("manual", cfg, make_deps(FakeClock(clock.current), manual_screen,
                                                                 latest_bar=MON))
            assert outcome.decision == sch.COMPLETED
    deps.sleep = sleep
    caplog.set_level(logging.INFO, logger="pipeline")

    sch.run_forever(cfg, deps, max_iterations=150)

    assert loop_screen.calls == [] and len(manual_screen.calls) == 1
    assert _hhmm(fetch.calls) == ["12:00"]                  # the 13:00 retry stopped before probing
    assert "Deferred target session 2026-09-14 already completed; no action required." in caplog.text
    assert "Deferred retries for 2026-09-14 cancelled: session completed." in caplog.text


def test_startup_and_daily_checks_after_success_never_rerun(tmp_path):
    cfg = make_config(tmp_path)
    clock = FakeClock(sgt("2026-09-15 09:00"))
    screen = FakeScreen(["AAA"])
    fetch = _fetch(clock, ready_from=clock.current)
    sch.run_forever(cfg, make_deps(clock, screen, latest_bar=fetch), max_iterations=300)   # through 09:15 and beyond
    assert len(screen.calls) == 1 and _hhmm(fetch.calls) == ["09:00"]


# ── planning rules ────────────────────────────────────────────────────────

@pytest.mark.parametrize("decision", [sch.PIPELINE_FAILED, sch.DEFERRED_MARKET_OPEN, sch.SKIPPED_LOCKED])
def test_only_data_not_ready_schedules_deferred_retries(tmp_path, decision):
    cfg = make_config(tmp_path)
    outcome = sch.CheckOutcome(decision, "scheduled", MON)
    assert sch.plan_deferred_retry(outcome, sgt("2026-09-15 10:00"), None, cfg, None) is None


def test_deferred_retries_can_be_disabled(tmp_path):
    cfg = make_config(tmp_path, deferred_retry_enabled=False)
    outcome = sch.CheckOutcome(sch.DATA_NOT_READY, "scheduled", MON, extra={"attempts": 3})
    assert sch.plan_deferred_retry(outcome, sgt("2026-09-15 10:00"), None, cfg, None) is None


def test_retry_is_clamped_to_cutoff(tmp_path):
    cfg = make_config(tmp_path, deferred_retry_cutoff="20:30")
    outcome = sch.CheckOutcome(sch.DATA_NOT_READY, "deferred", MON, extra={"attempts": 1})
    pending = sch.plan_deferred_retry(outcome, sgt("2026-09-15 19:50"), None, cfg, None)
    assert pending.due == sgt("2026-09-15 20:30")
