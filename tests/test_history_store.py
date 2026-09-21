"""Stage H: SQLite persistence — claims, canonical uniqueness, atomic result
writes, JSON round-trips, metadata and notification bookkeeping."""
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pipeline import history_store as hs
from pipeline import run_metadata
from stage_h_fakes import ranked_df, vcp_df

D = date(2026, 9, 14)


@pytest.fixture
def store(tmp_path):
    return hs.HistoryStore(str(tmp_path / "h.sqlite3"))


def _running(store, trading_date=D, **kw):
    run_id = store.claim_session(trading_date, "scheduled", **kw)
    store.mark_running(run_id, latest_bar=trading_date, attempts=1, run_timestamp="20260915_091500",
                       metadata={"git_commit": "abc", "git_dirty": 0, "config_hash": "h1",
                                 "config_json": "{}", "environment_json": "{}"})
    return run_id


def test_run_insertion_and_full_ranked_population_persisted(store, tmp_path):
    run_id = _running(store)
    symbols = [f"S{i:03d}" for i in range(120)]   # far more than any Top-N
    chart = tmp_path / "S000.png"
    chart.write_bytes(b"x")
    n = store.persist_results(run_id, D, ranked_df(symbols, [100 - i * 0.1 for i in range(120)]),
                              vcp_df(["S000", "S001"]), chart_paths={"S000": str(chart)},
                              stage_counts={"universe": 4312, "stage_d": 120, "ranked": 120, "vcp_analyzed": 2},
                              artifacts={"watchlist_csv": "data/reports/w.csv"})
    assert n == 120
    run = store.get_run(run_id)
    assert run["status"] == hs.RESULTS_PERSISTED and run["is_canonical"] == 1
    assert run["count_universe"] == 4312 and run["count_ranked"] == 120
    assert json.loads(run["stage_counts_json"])["vcp_analyzed"] == 2
    assert json.loads(run["artifacts_json"])["watchlist_csv"] == "data/reports/w.csv"
    assert run["git_commit"] == "abc" and run["config_hash"] == "h1"

    results = store.results_for_run(run_id)
    assert [r["final_rank"] for r in results] == list(range(1, 121))
    assert results[119]["symbol"] == "S119" and results[119]["vcp_status"] is None
    assert results[0]["chart_path"] == str(chart)
    assert store.has_successful_run(D)


def test_vcp_metrics_and_verdict_json_round_trip(store):
    run_id = _running(store)
    store.persist_results(run_id, D, ranked_df(["AAA"]), vcp_df(["AAA"]))
    row = store.results_for_run(run_id)[0]
    metrics = json.loads(row["vcp_metrics_json"])
    verdict = json.loads(row["vcp_verdict_json"])
    assert metrics["contraction_pcts"] == [20.0, 12.0, 6.0]
    assert metrics["volume_leg_avgs"] == [3000000, 1500000, 900000]
    assert verdict["rationale"] == "tight <base> & dry-up"
    assert row["vcp_status"] == "analyzed" and row["vcp_is_pattern"] == 1
    assert row["vcp_entry_recommendation"] == "wait_for_breakout"
    assert row["vcp_pivot_price"] == 120.5 and row["vcp_stop_loss"] == 110.0
    ranking = json.loads(row["ranking_json"])
    assert ranking["Symbol"] == "AAA" and ranking["Composite Score"] == 14


def test_token_free_and_error_verdicts_are_labelled_not_invented(store):
    run_id = _running(store)
    vcp = pd.DataFrame([
        {"Symbol": "AAA", "contraction_count": 2, "skipped": "token_free_mode"},
        {"Symbol": "BBB", "contraction_count": 1, "error": "boom"},
    ])
    store.persist_results(run_id, D, ranked_df(["AAA", "BBB", "CCC"]), vcp)
    rows = {r["symbol"]: r for r in store.results_for_run(run_id)}
    assert rows["AAA"]["vcp_status"] == "skipped" and rows["AAA"]["vcp_entry_recommendation"] is None
    assert rows["BBB"]["vcp_status"] == "error"
    assert rows["CCC"]["vcp_status"] is None


def test_nan_and_numpy_values_are_stored_as_null_and_python_types(store):
    run_id = _running(store)
    df = ranked_df(["AAA"])
    df["Sector"] = [np.nan]
    df["Market Cap"] = [np.float64("nan")]
    store.persist_results(run_id, D, df, None)
    row = store.results_for_run(run_id)[0]
    assert row["sector"] is None and row["market_cap"] is None
    assert json.loads(row["ranking_json"])["Sector"] is None


def test_failed_transaction_rolls_back_and_leaves_no_canonical_result(store, monkeypatch):
    run_id = _running(store)
    original = hs.build_result_rows

    def rows_with_duplicate_rank(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[-1]["final_rank"] = 1            # violates UNIQUE(run_id, final_rank) mid-insert
        return rows

    monkeypatch.setattr(hs, "build_result_rows", rows_with_duplicate_rank)
    with pytest.raises(sqlite3.IntegrityError):
        store.persist_results(run_id, D, ranked_df(["AAA", "BBB", "CCC"]), None)

    assert not store.has_successful_run(D)
    assert store.results_for_run(run_id) == []
    assert store.get_run(run_id)["status"] == hs.RUNNING


def test_same_trading_date_cannot_be_claimed_twice_while_active(store):
    store.claim_session(D, "startup")
    with pytest.raises(hs.SessionInProgress):
        store.claim_session(D, "scheduled")


def test_completed_session_cannot_be_claimed_without_force(store):
    run_id = _running(store)
    store.persist_results(run_id, D, ranked_df(["AAA"]), None)
    with pytest.raises(hs.AlreadyCompleted):
        store.claim_session(D, "scheduled")


def test_schema_rejects_two_canonical_rows_for_one_date(store):
    a = _running(store)
    store.persist_results(a, D, ranked_df(["AAA"]), None)
    b = store.claim_session(D, "manual", forced=True)
    with store._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE pipeline_runs SET is_canonical = 1 WHERE run_id = ?", (b,))


def test_force_rerun_supersedes_previous_canonical_atomically(store):
    first = _running(store)
    store.persist_results(first, D, ranked_df(["AAA", "BBB"]), None)
    second = _running(store, forced=True)
    store.persist_results(second, D, ranked_df(["BBB", "AAA"]), None, forced=True)

    assert store.canonical_run(D)["run_id"] == second
    old = store.get_run(first)
    assert old["is_canonical"] == 0 and old["superseded_by_run_id"] == second
    assert [r["symbol"] for r in store.results_for_sessions([D])] == ["BBB", "AAA"]


def test_non_forced_persist_refuses_to_overwrite_existing_canonical(store):
    first = _running(store)
    store.persist_results(first, D, ranked_df(["AAA"]), None)
    second = _running(store, forced=True)
    with pytest.raises(hs.AlreadyCompleted):
        store.persist_results(second, D, ranked_df(["ZZZ"]), None, forced=False)
    assert store.canonical_run(D)["run_id"] == first
    assert store.results_for_run(second) == []


def test_forced_claim_is_still_blocked_by_an_active_claim(store):
    store.claim_session(D, "startup")
    with pytest.raises(hs.SessionInProgress):
        store.claim_session(D, "manual", forced=True)


def test_stale_active_claim_is_abandoned(store):
    old_now = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)
    stale = store.claim_session(D, "startup", now=old_now)
    fresh = store.claim_session(D, "scheduled", now=old_now + timedelta(hours=13), stale_after_hours=12)
    assert store.get_run(stale)["status"] == hs.ABANDONED
    assert store.get_run(fresh)["status"] == hs.WAITING_FOR_DATA


def test_notification_status_never_touches_non_canonical_runs(store):
    run_id = store.claim_session(D, "startup")
    store.mark_failed(run_id, hs.DATA_NOT_READY, error_summary="stale")
    store.set_notification_status(run_id, hs.NOTIFIED)
    assert store.get_run(run_id)["status"] == hs.DATA_NOT_READY
    store.record_notification(run_id, "ops_alert", "SENT", attempts=1, provider_message_id="7")
    assert store.notifications_for_run(run_id)[0]["kind"] == "ops_alert"


def test_naive_datetimes_are_rejected(store):
    with pytest.raises(ValueError):
        store.claim_session(D, "startup", now=datetime(2026, 9, 15, 9, 15))


def test_config_snapshot_excludes_secrets_and_hash_ignores_operational_keys():
    cfg = {"min_price": 5.0, "anthropic_api_key": "sk-ant-secret", "telegram_bot_token": "123:abc",
           "telegram_enabled": True, "history_db_path": "x", "fundamentals_weight": 1.0}
    snap = run_metadata.redacted_config(cfg)
    assert "anthropic_api_key" not in snap and "telegram_bot_token" not in snap
    assert "sk-ant-secret" not in json.dumps(snap)

    toggled = {**cfg, "telegram_enabled": False, "history_db_path": "elsewhere"}
    assert run_metadata.config_hash(cfg) == run_metadata.config_hash(toggled)
    assert run_metadata.config_hash(cfg) != run_metadata.config_hash({**cfg, "min_price": 6.0})


def test_manifest_written_from_run_row(store, tmp_path):
    run_id = _running(store)
    store.persist_results(run_id, D, ranked_df(["AAA"]), None, stage_counts={"ranked": 1},
                          artifacts={"log": "data/logs/pipeline_x.log"})
    cfg = {"run_manifest_dir": str(tmp_path / "runs"), "history_db_path": store.path}
    path = run_metadata.write_manifest(cfg, store.get_run(run_id), extra={"telegram_status": "SENT"})
    doc = json.loads(open(path, encoding="utf-8").read())
    assert path.replace("\\", "/").endswith("runs/2026-09-14/run.json")
    assert doc["run_id"] == run_id and doc["stage_counts"] == {"ranked": 1}
    assert doc["telegram_status"] == "SENT" and doc["is_canonical"] is True


def test_v2_vcp_metrics_are_persisted_without_a_whitelist_and_old_shape_still_reads(store):
    """Stage G metric sets grow over time: new keys land in vcp_metrics_json as-is,
    verdict/bookkeeping keys never do, and a v1-shaped row still round-trips."""
    run_id = _running(store)
    vcp = vcp_df(["AAA", "BBB"])
    vcp["vcp_metrics_version"] = [2, None]
    vcp["contraction_confirmed"] = [[True, True, False], None]
    vcp["vcp_quality_notes"] = [["final contraction still forming at the right edge"], None]
    vcp["pivot_state"] = ["coiling_below_pivot", None]
    store.persist_results(run_id, D, ranked_df(["AAA", "BBB"]), vcp)
    rows = {r["symbol"]: r for r in store.results_for_run(run_id)}
    new = json.loads(rows["AAA"]["vcp_metrics_json"])
    assert new["vcp_metrics_version"] == 2 and new["contraction_confirmed"] == [True, True, False]
    assert new["pivot_state"] == "coiling_below_pivot" and new["contraction_pcts"] == [20.0, 12.0, 6.0]
    assert not {"Symbol", "rationale", "pivot_price", "entry_recommendation"} & set(new)
    old = json.loads(rows["BBB"]["vcp_metrics_json"])
    assert old["contraction_pcts"] == [20.0, 12.0, 6.0] and old["vcp_metrics_version"] is None
