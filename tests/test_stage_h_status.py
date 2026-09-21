"""Stage H operator commands: --test-telegram and the read-only status summary
(shared by --dry-run and `history_cli status`). No network, no real sleeps."""
import io
import logging
import os
from datetime import date

import pytest

from pipeline import history_cli
from pipeline import history_store as hs
from pipeline import scheduler as sch
from stage_h_fakes import FAKE_TOKEN, FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt

MON = date(2026, 9, 14)


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


# ── --test-telegram ──────────────────────────────────────────────────────

def test_test_telegram_sends_one_message_and_touches_no_history(cfg):
    tel = FakeTelegram()
    screen = FakeScreen(["AAA"])
    assert sch.send_test_message(cfg, make_deps(FakeClock(sgt("2026-09-15 08:50")), screen, telegram=tel))
    assert len(tel.messages) == 1 and sch.TEST_MESSAGE in tel.messages[0]
    assert screen.calls == [] and tel.photos == []
    assert not os.path.exists(cfg["history_db_path"])


def test_test_telegram_failure_returns_false_and_redacts_token(cfg, monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    deps = make_deps(FakeClock(sgt("2026-09-15 08:50")))

    def factory(config):
        raise ConnectionError(f"HTTPSConnectionPool: /bot{FAKE_TOKEN}/sendMessage timed out")
    deps.telegram_factory = factory
    with caplog.at_level(logging.INFO, logger="pipeline"):
        assert sch.send_test_message(cfg, deps) is False
    assert "FAILED" in caplog.text and FAKE_TOKEN not in caplog.text


def test_test_telegram_warns_when_reports_are_disabled(cfg, caplog):
    with caplog.at_level(logging.INFO, logger="pipeline"):
        sch.send_test_message(dict(cfg, telegram_enabled=False),
                              make_deps(FakeClock(sgt("2026-09-15 08:50")), telegram=FakeTelegram()))
    assert "TELEGRAM_ENABLED is off" in caplog.text


def test_cli_test_telegram_without_credentials_exits_non_zero(cfg):
    assert sch.main(["--test-telegram"], config=cfg) == 1       # conftest unsets credentials
    assert not os.path.exists(cfg["history_db_path"])


# ── status ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("when, expected", [
    ("2026-09-15 09:15", sch.WOULD_RUN),                 # Mon session done, market closed
    ("2026-09-15 22:00", sch.DEFERRED_MARKET_OPEN),      # Tue session open; Mon still unprocessed
])
def test_status_decision_matches_dry_run_check(cfg, when, expected):
    clock = FakeClock(sgt(when))
    deps = make_deps(clock, latest_bar=MON)
    outcome = sch.check_and_run("manual", cfg, deps, dry_run=True)
    st = sch.build_status(cfg, clock.now(), deps.store)
    assert outcome.decision == st["decision"] == expected
    assert st["latest_completed_session"] == MON and st["missing_latest_session"]


def test_status_after_successful_run_reports_skip_and_telegram(cfg):
    clock = FakeClock(sgt("2026-09-15 09:15"))
    deps = make_deps(clock, FakeScreen(["AAA", "BBB"]), latest_bar=MON, telegram=FakeTelegram())
    assert sch.check_and_run("scheduled", dict(cfg, telegram_enabled=True), deps).decision == sch.COMPLETED

    st = sch.build_status(cfg, clock.now(), deps.store)
    assert st["decision"] == sch.SKIPPED_ALREADY_COMPLETED == sch.check_and_run(
        "manual", cfg, make_deps(clock, latest_bar=MON), dry_run=True).decision
    assert not st["missing_latest_session"]
    assert st["latest_successful"]["status"] == hs.NOTIFIED and st["telegram_status"] == "SENT"
    text = "\n".join(sch.format_status(st, cfg))
    assert "Latest successful run" in text and "2026-09-14" in text and "Next scheduled check" in text and "Schedule mode" in text
    assert "2026-09-16 09:15" in text


def test_history_cli_status_does_not_create_missing_database(cfg):
    out = io.StringIO()
    history_cli.main(["--db", cfg["history_db_path"], "status"], config=cfg, out=out)
    assert "Scheduler / Stage-H Status" in out.getvalue()
    assert "not created yet" in out.getvalue()
    assert not os.path.exists(cfg["history_db_path"])


def test_cli_dry_run_prints_summary_and_records_nothing(cfg, caplog):
    with caplog.at_level(logging.INFO, logger="pipeline"):
        assert sch.main(["--dry-run"], config=cfg) == 0
    assert "Dry-run summary" in caplog.text and "US session in progress" in caplog.text
    assert "Would a check run now" in caplog.text
    assert hs.HistoryStore(cfg["history_db_path"]).recent_runs() == []


# ── keep-awake during Stage A-G ──────────────────────────────────────────

def test_keep_awake_blocks_idle_sleep_then_releases():
    from pipeline import keep_awake as ka
    calls = []
    with ka.keep_awake(setter=lambda flags: calls.append(flags) or 0x80000000) as held:
        assert held and calls == [ka.ES_CONTINUOUS | ka.ES_SYSTEM_REQUIRED]
    assert calls[-1] == ka.ES_CONTINUOUS


def test_keep_awake_failure_never_raises_and_does_not_release():
    from pipeline import keep_awake as ka
    calls = []

    def broken(flags):
        calls.append(flags)
        raise OSError("no kernel32")
    with ka.keep_awake(setter=broken) as held:
        assert held is False
    assert len(calls) == 1


def test_stage_a_to_g_runs_inside_keep_awake(cfg):
    from contextlib import contextmanager
    events = []

    @contextmanager
    def fake_keep_awake():
        events.append("acquire")
        yield True
        events.append("release")

    def screen(config, run_ts, stats):
        events.append("screen")
        return FakeScreen(["AAA"])(config, run_ts, stats)

    deps = make_deps(FakeClock(sgt("2026-09-15 09:15")), screen, latest_bar=MON)
    deps.keep_awake = fake_keep_awake
    assert sch.check_and_run("scheduled", cfg, deps).decision == sch.COMPLETED
    assert events == ["acquire", "screen", "release"]


def test_readiness_failure_does_not_hold_keep_awake(cfg):
    events = []
    deps = make_deps(FakeClock(sgt("2026-09-15 09:15")), latest_bar=date(2026, 9, 11))
    deps.keep_awake = lambda: events.append("acquire")
    assert sch.check_and_run("scheduled", cfg, deps).decision == sch.DATA_NOT_READY
    assert events == []
