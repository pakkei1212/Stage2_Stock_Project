"""Roadmap item 3: verify the OHLCV Parquet cache serves fresh data without
re-fetching, delta-fetches only the missing tail when stale, full-fetches cold
or too-shallow tickers, and that market caps are reused within their TTL.

All network is monkeypatched via `_download_batches` / `_get_market_cap_timeout`
so these tests are deterministic and offline.
"""
import os

import numpy as np
import pandas as pd
import pytest

from pipeline import data_sources as ds
from pipeline import ohlcv_cache


def _ohlcv(index, start_price=100.0):
    n = len(index)
    close = np.linspace(start_price, start_price + n, n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000.0},
        index=pd.DatetimeIndex(index),
    )


def _cache_config(tmp_path, **overrides):
    cfg = {
        "ohlcv_cache_enabled": True,
        "ohlcv_cache_dir": str(tmp_path / "ohlcv"),
        "ohlcv_cache_max_age_days": 1,
        "ohlcv_cache_overlap_days": 5,
        "batch_size": 50, "batch_sleep_sec": 0, "max_retries": 1,
    }
    cfg.update(overrides)
    return cfg


def _today():
    return pd.Timestamp(pd.Timestamp.now("UTC").date())


def _set_written_at(cfg, ticker, when_utc):
    """Pin the cache file's write time (the cache's record of when bars were fetched)."""
    ts = pd.Timestamp(when_utc).timestamp()
    os.utime(ohlcv_cache.cache_path(ticker, cfg["ohlcv_cache_dir"]), (ts, ts))


def test_fresh_deep_cache_is_served_without_any_fetch(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    # 300 daily bars ending today -> fresh and deeper than a 30d request.
    idx = pd.date_range(end=_today(), periods=300, freq="D")
    ohlcv_cache.save("AAA", _ohlcv(idx), cfg["ohlcv_cache_dir"])
    # Written well after today's session closed, so the last bar is complete
    # regardless of when the test suite runs.
    _set_written_at(cfg, "AAA", pd.Timestamp.now("UTC") + pd.Timedelta(days=2))

    def _boom(*a, **k):
        raise AssertionError("_download_batches must not be called on a cache hit")

    monkeypatch.setattr(ds, "_download_batches", _boom)

    result = ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=True)

    assert "AAA" in result
    assert result["AAA"].index.max() == _today()
    # Only the requested ~30d window is returned, not all 300 cached bars.
    assert len(result["AAA"]) <= 40


def test_stale_but_deep_cache_triggers_delta_fetch_and_appends(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    last_cached = _today() - pd.Timedelta(days=10)
    idx = pd.date_range(end=last_cached, periods=300, freq="D")
    ohlcv_cache.save("AAA", _ohlcv(idx), cfg["ohlcv_cache_dir"])

    calls = {}

    def _fake(tickers, config, interval="1d", **dl_kwargs):
        calls["kwargs"] = dl_kwargs
        new_idx = pd.date_range(start=last_cached - pd.Timedelta(days=2), end=_today(), freq="D")
        return {"AAA": _ohlcv(new_idx, start_price=400.0)}

    monkeypatch.setattr(ds, "_download_batches", _fake)

    result = ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=True)

    # A delta fetch uses start/end, never period.
    assert "start" in calls["kwargs"] and "end" in calls["kwargs"]
    assert "period" not in calls["kwargs"]
    # Result now reaches today, and the cache on disk was updated.
    assert result["AAA"].index.max() == _today()
    assert ohlcv_cache.load("AAA", cfg["ohlcv_cache_dir"]).index.max() == _today()


def test_cold_ticker_is_full_fetched_and_written_to_cache(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    calls = {}

    def _fake(tickers, config, interval="1d", **dl_kwargs):
        calls["kwargs"] = dl_kwargs
        idx = pd.date_range(end=_today(), periods=300, freq="D")
        return {"NEW": _ohlcv(idx)}

    monkeypatch.setattr(ds, "_download_batches", _fake)

    result = ds.batch_download(["NEW"], period="400d", config=cfg, use_cache=True)

    assert calls["kwargs"].get("period") == "400d"      # full-window fetch
    assert "NEW" in result
    assert ohlcv_cache.load("NEW", cfg["ohlcv_cache_dir"]) is not None   # persisted


def test_fresh_but_too_shallow_cache_forces_full_fetch(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    # Only 10 bars ending today: fresh, but nowhere near a 400d request.
    idx = pd.date_range(end=_today(), periods=10, freq="D")
    ohlcv_cache.save("AAA", _ohlcv(idx), cfg["ohlcv_cache_dir"])

    calls = {}

    def _fake(tickers, config, interval="1d", **dl_kwargs):
        calls["kwargs"] = dl_kwargs
        deep = pd.date_range(end=_today(), periods=300, freq="D")
        return {"AAA": _ohlcv(deep)}

    monkeypatch.setattr(ds, "_download_batches", _fake)

    ds.batch_download(["AAA"], period="400d", config=cfg, use_cache=True)

    assert calls["kwargs"].get("period") == "400d"


def test_use_cache_false_bypasses_cache_entirely(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    calls = {}

    def _fake(tickers, config, interval="1d", **dl_kwargs):
        calls["kwargs"] = dl_kwargs
        return {}

    monkeypatch.setattr(ds, "_download_batches", _fake)

    ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=False)

    assert calls["kwargs"].get("period") == "30d"       # went straight to the network path


def test_delta_fetch_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    last_cached = _today() - pd.Timedelta(days=10)
    idx = pd.date_range(end=last_cached, periods=300, freq="D")
    ohlcv_cache.save("AAA", _ohlcv(idx), cfg["ohlcv_cache_dir"])

    monkeypatch.setattr(ds, "_download_batches", lambda *a, **k: {})   # delta returns nothing

    result = ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=True)

    # Stale data beats no data — the ticker is still served from cache.
    assert "AAA" in result
    assert result["AAA"].index.max() == last_cached


# ─────────────────────── market-cap cache ─────────────────────────────────

def _cap_config(tmp_path, **overrides):
    cfg = {
        "min_market_cap": 2_000_000_000, "max_market_cap": None,
        "market_cap_cache_path": str(tmp_path / "market_cap_cache.csv"),
        "market_cap_cache_ttl_days": 7,
    }
    cfg.update(overrides)
    return cfg


def test_market_cap_reused_within_ttl_without_refetch(tmp_path, monkeypatch):
    cfg = _cap_config(tmp_path)
    today = pd.Timestamp(pd.Timestamp.now("UTC").date())
    pd.DataFrame([
        {"Symbol": "BIG", "Market Cap": 50_000_000_000, "AsOf": today},
        {"Symbol": "SMALL", "Market Cap": 500_000_000, "AsOf": today},
    ]).to_csv(cfg["market_cap_cache_path"], index=False)

    monkeypatch.setattr(
        ds, "_get_market_cap_timeout",
        lambda t, timeout_sec=8: (_ for _ in ()).throw(AssertionError(f"refetched {t}")),
    )

    survivors, cap_df = ds.market_cap_filter(["BIG", "SMALL"], cfg)

    assert survivors == ["BIG"]   # BIG passes the $2B floor, SMALL doesn't
    assert cap_df.set_index("Symbol").loc["BIG", "Market Cap"] == 50_000_000_000


def test_market_cap_refetched_after_ttl_expires(tmp_path, monkeypatch):
    cfg = _cap_config(tmp_path)
    stale = pd.Timestamp(pd.Timestamp.now("UTC").date()) - pd.Timedelta(days=30)
    pd.DataFrame([{"Symbol": "BIG", "Market Cap": 1.0, "AsOf": stale}]).to_csv(
        cfg["market_cap_cache_path"], index=False,
    )

    monkeypatch.setattr(ds, "_get_market_cap_timeout", lambda t, timeout_sec=8: 50_000_000_000)

    survivors, cap_df = ds.market_cap_filter(["BIG"], cfg)

    # Stale entry ignored, fresh value fetched, and the cache updated.
    assert survivors == ["BIG"]
    assert cap_df.iloc[0]["Market Cap"] == 50_000_000_000
    reloaded = pd.read_csv(cfg["market_cap_cache_path"]).set_index("Symbol")
    assert reloaded.loc["BIG", "Market Cap"] == 50_000_000_000


# ─────────────────────── split-aware merge (ohlcv_cache.merge) ────────────
# Regression coverage for the corporate-action rescaling branch, which the
# scheduled daily runs now exercise every session. Production code unchanged.

def test_merge_rescales_cached_history_after_a_split():
    old_idx = pd.bdate_range("2026-08-03", periods=20)
    old = _ohlcv(old_idx, start_price=200.0)                  # pre-split basis
    overlap = old_idx[-5:]
    tail = pd.bdate_range(overlap[-1] + pd.Timedelta(days=1), periods=3)
    new_idx = overlap.append(tail)
    # A 2-for-1 split: yfinance auto_adjust re-bases everything, overlap included, to half.
    new = old.loc[overlap].copy()
    new[["Open", "High", "Low", "Close"]] = new[["Open", "High", "Low", "Close"]] * 0.5
    new["Volume"] = new["Volume"] * 2
    new = pd.concat([new, _ohlcv(tail, start_price=float(new["Close"].iloc[-1]) + 0.5)])

    merged = ohlcv_cache.merge(old.copy(), new)

    assert list(merged.index) == list(old_idx.append(tail))
    first_day = old_idx[0]
    assert merged.loc[first_day, "Close"] == pytest.approx(old.loc[first_day, "Close"] * 0.5)
    assert merged.loc[first_day, "Volume"] == pytest.approx(old.loc[first_day, "Volume"] * 2)
    # continuity across the seam: no 2x jump where cached history meets fresh bars
    seam = merged["Close"].pct_change().abs().max()
    assert seam < 0.05


def test_merge_leaves_history_alone_when_overlap_matches_within_tolerance():
    old_idx = pd.bdate_range("2026-08-03", periods=20)
    old = _ohlcv(old_idx)
    new = old.iloc[-5:].copy()
    new["Close"] = new["Close"] * 1.01                         # 1% drift < SPLIT_TOL
    merged = ohlcv_cache.merge(old.copy(), new)
    assert merged.loc[old_idx[0], "Close"] == old.loc[old_idx[0], "Close"]
    assert merged.loc[old_idx[-1], "Close"] == new.loc[old_idx[-1], "Close"]   # new wins on overlap
    assert not merged.index.duplicated().any()


def test_merge_without_overlap_just_concatenates():
    old = _ohlcv(pd.bdate_range("2026-08-03", periods=5), start_price=100)
    new = _ohlcv(pd.bdate_range("2026-08-17", periods=5), start_price=300)
    merged = ohlcv_cache.merge(old.copy(), new)
    assert len(merged) == 10 and merged["Close"].iloc[0] == old["Close"].iloc[0]


# ─────────────── partial-day protection (last_bar_complete) ────────────────
# A bar cached while its US session was still open is intraday data. The next
# morning's scheduled run for that session must refetch it, not serve it.
# Mon 2026-09-14 closes 16:00 ET = 20:00 UTC; the default buffer is 60 min.

MON_SESSION = pd.Timestamp("2026-09-14")
DURING_MON_SESSION = pd.Timestamp("2026-09-14 18:30", tz="UTC")     # 14:30 ET
TUE_0915_SGT = pd.Timestamp("2026-09-15 01:15", tz="UTC")           # scheduled check


def _bars_ending(last_day, periods=300):
    return _ohlcv(pd.bdate_range(end=last_day, periods=periods))


def test_last_bar_written_during_its_session_is_incomplete():
    df = _bars_ending(MON_SESSION)
    assert not ohlcv_cache.last_bar_complete(df, DURING_MON_SESSION, 60)
    # after the close but inside the buffer: still not trusted
    assert not ohlcv_cache.last_bar_complete(df, pd.Timestamp("2026-09-14 20:30", tz="UTC"), 60)


def test_last_bar_written_after_close_plus_buffer_is_complete():
    df = _bars_ending(MON_SESSION)
    assert ohlcv_cache.last_bar_complete(df, pd.Timestamp("2026-09-14 21:00", tz="UTC"), 60)
    assert ohlcv_cache.last_bar_complete(df, TUE_0915_SGT, 60)


def test_early_close_session_uses_its_actual_close():
    # Fri 2026-11-27 (day after Thanksgiving) closes 13:00 ET = 18:00 UTC.
    df = _bars_ending(pd.Timestamp("2026-11-27"))
    assert ohlcv_cache.last_bar_complete(df, pd.Timestamp("2026-11-27 19:00", tz="UTC"), 60)
    assert not ohlcv_cache.last_bar_complete(df, pd.Timestamp("2026-11-27 17:00", tz="UTC"), 60)


def test_unknown_write_time_is_treated_as_incomplete():
    assert not ohlcv_cache.last_bar_complete(_bars_ending(MON_SESSION), None, 60)


def _serve_scenario(tmp_path, monkeypatch, written_at):
    """Cache ends on Monday's session and passes the date-based freshness rule."""
    cfg = _cache_config(tmp_path, session_close_buffer_minutes=60)
    ohlcv_cache.save("AAA", _bars_ending(MON_SESSION), cfg["ohlcv_cache_dir"])
    _set_written_at(cfg, "AAA", written_at)
    monkeypatch.setattr(ohlcv_cache, "is_fresh", lambda df, today, max_age: True)
    return cfg


def test_partial_bar_cached_during_session_is_refetched_not_served(tmp_path, monkeypatch):
    cfg = _serve_scenario(tmp_path, monkeypatch, DURING_MON_SESSION)
    partial_close = float(ohlcv_cache.load("AAA", cfg["ohlcv_cache_dir"]).loc[MON_SESSION, "Close"])
    calls = []

    def _fake(tickers, config, interval="1d", **dl_kwargs):
        calls.append(dl_kwargs)
        final = _ohlcv(pd.bdate_range(end=MON_SESSION, periods=6),
                       start_price=partial_close - 5 + 0.5)    # same basis, final Monday close differs
        final.loc[MON_SESSION, "Close"] = partial_close * 1.01
        return {"AAA": final}

    monkeypatch.setattr(ds, "_download_batches", _fake)
    ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=True)

    assert len(calls) == 1 and "start" in calls[0] and "period" not in calls[0]   # delta, not full
    assert pd.Timestamp(calls[0]["start"]) <= MON_SESSION                        # overlap covers Monday
    cached = ohlcv_cache.load("AAA", cfg["ohlcv_cache_dir"])
    assert cached.loc[MON_SESSION, "Close"] == pytest.approx(partial_close * 1.01)  # partial bar replaced
    assert len(cached) == 300 and not cached.index.duplicated().any()


def test_bar_cached_after_close_is_served_from_disk(tmp_path, monkeypatch):
    cfg = _serve_scenario(tmp_path, monkeypatch, TUE_0915_SGT)

    def _boom(*a, **k):
        raise AssertionError("a complete, fresh cache must not be refetched")

    monkeypatch.setattr(ds, "_download_batches", _boom)
    result = ds.batch_download(["AAA"], period="30d", config=cfg, use_cache=True)
    assert "AAA" in result
