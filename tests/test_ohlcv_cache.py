"""Roadmap item 3: verify the OHLCV Parquet cache serves fresh data without
re-fetching, delta-fetches only the missing tail when stale, full-fetches cold
or too-shallow tickers, and that market caps are reused within their TTL.

All network is monkeypatched via `_download_batches` / `_get_market_cap_timeout`
so these tests are deterministic and offline.
"""
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


def test_fresh_deep_cache_is_served_without_any_fetch(tmp_path, monkeypatch):
    cfg = _cache_config(tmp_path)
    # 300 daily bars ending today -> fresh and deeper than a 30d request.
    idx = pd.date_range(end=_today(), periods=300, freq="D")
    ohlcv_cache.save("AAA", _ohlcv(idx), cfg["ohlcv_cache_dir"])

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
