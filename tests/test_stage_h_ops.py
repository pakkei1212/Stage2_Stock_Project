"""Stage H operational pieces: secret redaction in logs, the file lock, the
history CLI and the per-run log file."""
import io
import logging
import os
from datetime import date

from pipeline import history_cli
from pipeline import history_store as hs
from pipeline.logging_config import SecretRedactingFilter, run_log_file
from pipeline.run_lock import FileLock, LockHeld
from stage_h_fakes import FAKE_TOKEN, ranked_df, vcp_df


def test_log_filter_redacts_secret_values_in_messages_and_tracebacks(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactingFilter())
    log = logging.getLogger("pipeline.test_redaction")
    log.addHandler(handler)
    log.propagate = False
    try:
        log.warning("POST https://api.telegram.org/bot%s/sendMessage", FAKE_TOKEN)
        try:
            raise ConnectionError(f"url: /bot{FAKE_TOKEN}/sendPhoto")
        except ConnectionError:
            log.exception("send failed")
    finally:
        log.removeHandler(handler)
    out = stream.getvalue()
    assert FAKE_TOKEN not in out and out.count("***") >= 2


def test_file_lock_is_exclusive_and_released(tmp_path):
    path = str(tmp_path / "x.lock")
    first = FileLock(path).acquire()
    try:
        try:
            FileLock(path).acquire()
            raise AssertionError("second acquire should fail")
        except LockHeld:
            pass
    finally:
        first.release()
    with FileLock(path) as again:
        assert again.held


def test_run_log_file_captures_run_and_detaches(tmp_path):
    cfg = {"log_dir": str(tmp_path)}
    log = logging.getLogger("pipeline.run_log_test")
    with run_log_file(cfg, "20260915_091500") as path:
        logging.getLogger("pipeline").setLevel(logging.INFO)
        log.info("inside run")
    log.info("after run")
    text = open(path, encoding="utf-8").read()
    assert os.path.basename(path) == "pipeline_20260915_091500.log"
    assert "inside run" in text and "after run" not in text


def _seed(db):
    store = hs.HistoryStore(db)
    for d, order in ((date(2026, 9, 11), ["AAA", "BBB", "NVDA"]), (date(2026, 9, 14), ["NVDA", "AAA", "BBB"])):
        run_id = store.claim_session(d, "scheduled")
        store.mark_running(run_id, latest_bar=d, attempts=1, run_timestamp="t", metadata={})
        store.persist_results(run_id, d, ranked_df(order), vcp_df(order[:1]), stage_counts={"ranked": 3})
    return store


def test_history_cli_runs_top_and_symbol(tmp_path):
    db = str(tmp_path / "h.sqlite3")
    _seed(db)
    out = io.StringIO()
    history_cli.main(["--db", db, "runs"], out=out)
    assert "2026-09-14" in out.getvalue() and "RESULTS_PERSISTED" in out.getvalue()

    out = io.StringIO()
    history_cli.main(["--db", db, "top", "--sessions", "5", "--top-n", "3"], out=out)
    lines = out.getvalue().splitlines()
    nvda = next(l for l in lines if l.startswith("NVDA"))
    assert "#3" in nvda and nvda.rstrip().endswith("#1")

    out = io.StringIO()
    history_cli.main(["--db", db, "symbol", "nvda"], out=out)
    text = out.getvalue()
    assert "NVDA" in text and "2026-09-11" in text and "wait_for_breakout (high)" in text
    assert "sessions with no successful run" not in text          # 09-11 -> 09-14 has no gap


def test_scheduler_module_entrypoint_logs_its_decision(tmp_path):
    """Runs the real `python -m pipeline.scheduler --dry-run` in a subprocess.
    Guards against the __name__ == "__main__" logger silently dropping INFO logs.
    --dry-run resolves the session and reads SQLite only: no data fetch, no network."""
    import subprocess
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": repo, "PYTHONIOENCODING": "utf-8",
           "HISTORY_DB_PATH": str(tmp_path / "h.sqlite3"), "TELEGRAM_ENABLED": "0", "LOG_LEVEL": "INFO"}
    proc = subprocess.run([sys.executable, "-m", "pipeline.scheduler", "--dry-run"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "== manual check ==" in proc.stdout
    assert "Clock: local" in proc.stdout
    logs = list((tmp_path / "data" / "logs").glob("scheduler_*.log"))
    assert logs and "== manual check ==" in logs[0].read_text(encoding="utf-8")
