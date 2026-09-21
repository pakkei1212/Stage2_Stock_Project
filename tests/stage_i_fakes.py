"""Synthetic fixtures for Stage I (setup gate, plans, trade journal, monitor).

Everything is built by hand so assertions are exact: no network, no broker, no
Telegram and no Claude. ``stage_g_row`` mimics a stored ``stock_results`` row
(the gate's only input), and ``make_bars`` builds daily OHLCV with a chosen
close path.
"""
import json

import pandas as pd

from pipeline.config import CONFIG

RUN_ID = "run-0001"
SESSION = "2026-09-15"


def stage_g_row(symbol="CACC", *, quality="strong", pivot_state="coiling_below_pivot",
                pivot=100.0, price=98.0, claude=None, rank=1, run_id=RUN_ID, trading_date=SESSION,
                market_cap=5e9, metrics=None, claude_fields=None):
    """A ``stock_results`` row as history_store writes it.

    ``claude``: None = Stage G never asked Claude (``skipped/not_requested``),
    True/False = Claude reviewed it and said yes/no.
    """
    m = {
        "vcp_numeric_quality": quality,
        "pivot_state": pivot_state,
        "pivot_price_candidate": pivot,
        "current_price": price,
        "pct_from_pivot": round((price - pivot) / pivot * 100, 2) if pivot else None,
        "contraction_pcts": [20.0, 12.0, 6.0],
        "contraction_low_prices": [80.0, 88.0, 92.0],
        "contraction_high_prices": [100.0, 99.0, 98.5],
        "volume_dryup_quality": "strong",
        "breakout_volume_vs_20d": None,
        "current_volume_vs_20d": 0.8,
        "breakout_volume_confirmed": None,
        "sessions_since_breakout": None,
        "vcp_metrics_version": 2,
    }
    m.update(metrics or {})

    if claude is None:
        vcp_status, is_pattern = "skipped", None
        verdict = {"skipped": "not_requested"}
    else:
        vcp_status, is_pattern = "analyzed", int(bool(claude))
        verdict = {"is_vcp_pattern": bool(claude), "pattern_stage": "mature",
                   "confidence": "high", "entry_recommendation": "wait_for_breakout"}
    verdict.update(claude_fields or {})

    return {
        "run_id": run_id, "trading_date": trading_date, "symbol": symbol, "final_rank": rank,
        "composite_score": 12.0, "max_score": 14.0, "technical_score": 8.0, "fundamentals_score": 4.0,
        "sector": "Technology", "industry": "Software", "market_cap": market_cap,
        "last_close": price, "rs_3mo": 0.2, "rs_6mo": 0.3,
        "vcp_status": vcp_status, "vcp_is_pattern": is_pattern,
        "vcp_pattern_stage": verdict.get("pattern_stage"), "vcp_confidence": verdict.get("confidence"),
        "vcp_entry_recommendation": verdict.get("entry_recommendation"),
        "vcp_pivot_price": verdict.get("pivot_price"), "vcp_stop_loss": verdict.get("suggested_stop_loss"),
        "vcp_metrics_json": json.dumps(m, sort_keys=True),
        "vcp_verdict_json": json.dumps(verdict, sort_keys=True),
        "chart_path": None, "vcp_debug_chart_path": None, "ranking_json": "{}",
    }


def non_stage_g_row(symbol="ZZZZ", rank=9):
    """A ranked row Stage G never analysed (vcp_status NULL)."""
    row = stage_g_row(symbol, rank=rank)
    row.update({"vcp_status": None, "vcp_is_pattern": None, "vcp_metrics_json": None,
                "vcp_verdict_json": None})
    return row


def make_bars(closes, *, wick_pct=1.0, volume=1_000_000, volumes=None, start="2026-01-02"):
    """Daily OHLCV from a close path. Open = prior close; High/Low add ``wick_pct``."""
    closes = [float(c) for c in closes]
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * (1 + wick_pct / 100) for o, c in zip(opens, closes)]
    lows = [min(o, c) * (1 - wick_pct / 100) for o, c in zip(opens, closes)]
    vols = [float(v) for v in (volumes if volumes is not None else [volume] * len(closes))]
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
                        index=pd.bdate_range(start, periods=len(closes)))


def flat_bars(n=80, price=100.0, **kwargs):
    return make_bars([price] * n, **kwargs)


def actions_source(**by_symbol):
    """A ``fetch_corporate_actions`` stand-in built from plain dicts.

        actions_source(CACC={"splits": {"2026-10-01": 2.0}})

    Every corporate-action test goes through this: nothing asks Yahoo what
    happened to a ticker, so the arithmetic assertions are exact.
    """
    calls = []

    def fetch(symbol, start=None, config=None):
        calls.append((symbol, start))
        return dict(by_symbol.get(str(symbol).upper()) or {})

    fetch.calls = calls
    return fetch


def trade_config(tmp_path, **overrides):
    cfg = dict(CONFIG)
    cfg.update({
        "trade_db_path": str(tmp_path / "trades" / "trade_journal.sqlite3"),
        "trade_journal_enabled": True,
        "trade_portfolio_value": 100_000.0,
        "trade_risk_per_trade_pct": 1.0,
        "trade_max_position_portion_pct": 25.0,
    })
    cfg.update(overrides)
    return cfg
