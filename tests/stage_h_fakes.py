"""Deterministic fakes for Stage H (scheduler / history / Telegram) tests.

Nothing here sleeps or touches the network: the clock advances when "slept",
Stage A-G is replaced by a synthetic ScreenResult, and Telegram is an
in-memory recorder.
"""
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.config import CONFIG
from pipeline.run_pipeline import ScreenResult
from pipeline.scheduler import Deps
from pipeline.telegram_notifier import TelegramError

SGT = ZoneInfo("Asia/Singapore")
NY = ZoneInfo("America/New_York")
FAKE_TOKEN = "123456:TEST-token-not-real-abcdef"


def sgt(text):
    return datetime.fromisoformat(text).replace(tzinfo=SGT)


def make_config(tmp_path, **overrides):
    cfg = dict(CONFIG)
    cfg.update({
        "history_db_path": str(tmp_path / "history" / "stage2_history.sqlite3"),
        "history_lock_dir": str(tmp_path / "history"),
        "run_manifest_dir": str(tmp_path / "runs"),
        "log_dir": str(tmp_path / "logs"),
        "report_dir": str(tmp_path / "reports"),
        "chart_dir": str(tmp_path / "charts"),
        "scheduler_mode": "daily",
        "scheduler_local_time": "09:15",
        "scheduler_times": "",
        "scheduler_interval_minutes": "60",
        "scheduler_timezone": "Asia/Singapore",
        "scheduler_run_on_startup": True,
        "scheduler_enabled": True,
        "scheduler_poll_seconds": 60,
        "session_close_buffer_minutes": 60,
        "data_ready_retry_minutes": 10,
        "data_ready_max_attempts": 3,
        "telegram_enabled": False,
        "telegram_send_charts": False,
        "telegram_vcp_chart_max": 10,
        "telegram_ops_alerts": True,
        "telegram_max_attempts": 3,
        "trend_top_n": 3,
        "trend_lookback_sessions": 10,
        "trend_rank_move_min": 3,
        "trend_persistent_min_streak": 3,
        # Stage I (trade journal) — always inside tmp_path, never the repo's data/.
        "trade_db_path": str(tmp_path / "trades" / "trade_journal.sqlite3"),
        "trade_journal_enabled": True,
        "trade_bot_enabled": False,
        "trade_positions_in_report": True,
        "trade_portfolio_value": 100_000.0,
    })
    cfg.update(overrides)
    return cfg


class FakeClock:
    def __init__(self, start):
        self.current = start
        self.sleeps = []

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current = self.current + timedelta(seconds=seconds)


def ranked_df(symbols, scores=None):
    scores = scores or [14 - i * 0.5 for i in range(len(symbols))]
    return pd.DataFrame({
        "Symbol": symbols,
        "Sector": ["Technology"] * len(symbols),
        "Industry": ["Semiconductors"] * len(symbols),
        "Market Cap": [1e10] * len(symbols),
        "Composite Score": scores,
        "Max Score": [14.0] * len(symbols),
        "Technical Score": [8] * len(symbols),
        "Fundamentals Score": [s - 8 for s in scores],
        "Last Close": [100.0 + i for i in range(len(symbols))],
        "RS vs NASDAQ (3mo)": [0.1] * len(symbols),
    })


def vcp_df(symbols, recommendation="wait_for_breakout", numeric=None):
    """Stage-G rows. ``numeric`` adds/overrides detector assessment keys (e.g.
    vcp_numeric_quality / pivot_state) so Stage-I gating can be exercised."""
    rows = []
    for s in symbols:
        rows.append({
            "Symbol": s, "contraction_pcts": [20.0, 12.0, 6.0], "contraction_count": 3,
            "contraction_ratios": [0.6, 0.5], "contraction_monotonicity": 1.0,
            "contractions_decreasing": True, "volume_dryup_ratio": 0.4,
            "volume_leg_avgs": [3000000, 1500000, 900000], "pivot_price_candidate": 120.0,
            "current_price": 118.0, "pct_below_pivot": 1.67, "pct_ext_ema10": 1.0,
            "pct_ext_ema21": 2.0, "pct_ext_ma50": 5.0,
            "is_vcp_pattern": True, "pattern_stage": "mature", "contraction_count_observed": 3,
            "volume_dry_up_confirmed": True, "pivot_price": 120.5, "suggested_stop_loss": 110.0,
            "confidence": "high", "entry_recommendation": recommendation, "rationale": "tight <base> & dry-up",
            **(numeric or {}),
        })
    return pd.DataFrame(rows)


class FakeScreen:
    """Stands in for run_pipeline.run_screen. ``plans`` is a list consumed per call:
    a list of symbols, an Exception to raise, or None for an empty result."""

    def __init__(self, *plans, vcp_top=2, write_charts=True, vcp_numeric=None):
        self.plans = list(plans)
        self.calls = []
        self.vcp_top = vcp_top
        self.write_charts = write_charts
        self.vcp_numeric = vcp_numeric

    def __call__(self, config, run_ts, stats):
        self.calls.append(run_ts)
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        stats.update({"universe": 4300, "liquidity": 1500, "market_cap": 700, "sector_classified": 650,
                      "stage_c": 300, "current_stage": "D"})
        chart_dir = os.path.join(config["chart_dir"], run_ts)
        result = ScreenResult(run_timestamp=run_ts, chart_dir=chart_dir, stage_counts=stats)
        if isinstance(plan, Exception):
            raise plan
        if plan is None:
            result.stopped_after = "D"
            return result
        symbols = plan
        result.final_watchlist = ranked_df(symbols)
        top = symbols[:self.vcp_top]
        result.vcp_df = vcp_df(top, numeric=self.vcp_numeric)
        if self.write_charts:
            os.makedirs(result.vcp_debug_dir, exist_ok=True)
            for s in top:
                p = os.path.join(chart_dir, f"{s}.png")
                with open(p, "wb") as f:
                    f.write(b"\x89PNG fake")
                result.chart_paths[s] = p
                with open(os.path.join(result.vcp_debug_dir, f"{s}.png"), "wb") as f:   # Stage-G verification chart
                    f.write(b"\x89PNG debug")
        stats.update({"stage_d": len(symbols), "trend_template_pass": len(symbols), "fundamentals_scored": len(symbols),
                      "ranked": len(symbols), "charts": len(top), "vcp_analyzed": len(top),
                      "vcp_pattern": len(top), "vcp_actionable": len(top), "current_stage": None})
        return result


class FakeTelegram:
    def __init__(self, fail_messages=0, fail_photos=False, permanent_failure=False):
        self.messages = []
        self.message_markups = []      # reply_markup per sent text message (None = no buttons)
        self.photos = []
        self.markups = []              # reply_markup per sent photo
        self.fail_messages = fail_messages
        self.fail_photos = fail_photos
        self.permanent_failure = permanent_failure
        self.max_attempts = 3
        self._next_id = 100

    def send_message(self, text, reply_markup=None, chat_id=None):
        if self.permanent_failure or self.fail_messages > 0:
            self.fail_messages -= 1
            raise TelegramError("Telegram sendMessage failed: HTTP 502: Bad Gateway", attempts=3)
        self.messages.append(text)
        self.message_markups.append(reply_markup)
        self._next_id += 1
        return self._next_id, 1

    def send_photo(self, path, caption=None, reply_markup=None, chat_id=None):
        if self.fail_photos:
            raise TelegramError("Telegram sendPhoto failed: HTTP 400: bad photo", attempts=1)
        self.markups.append(reply_markup)
        self.photos.append((path, caption))
        self._next_id += 1
        return self._next_id, 1


def make_deps(clock, screen=None, latest_bar=None, telegram=None, store=None, price_history=None,
              trade_store=None, corporate_actions=None):
    """latest_bar: a date, a callable(symbol)->date, or a list consumed per call."""
    if callable(latest_bar):
        fetch = latest_bar
    elif isinstance(latest_bar, list):
        seq = list(latest_bar)
        fetch = lambda symbol: seq.pop(0) if len(seq) > 1 else seq[0]
    else:
        fetch = lambda symbol: latest_bar
    fake_meta = lambda config: {"git_commit": "abc123def456", "git_dirty": 0, "config_hash": "cafebabe",
                                "config_json": "{}", "environment_json": "{}"}

    def telegram_factory(config):
        if telegram is None:
            raise TelegramError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set")
        return telegram

    # Stage I corporate actions: a plain {symbol: {"splits": ..., "dividends": ...,
    # "events": [...]}} map, so no test ever asks Yahoo what happened to a ticker.
    action_calls = []

    def fetch_actions(symbol, start=None, config=None):
        action_calls.append((symbol, start))
        return dict((corporate_actions or {}).get(symbol) or {})

    deps = Deps(now=clock.now, sleep=clock.sleep, fetch_latest_bar=fetch,
                run_screen=screen or FakeScreen(["AAA", "BBB", "CCC", "DDD"]),
                telegram_factory=telegram_factory, collect_metadata=fake_meta, store=store,
                # Stage I: no network for setup plans / the position monitor either.
                fetch_price_history=price_history or (lambda symbols, config=None: {}),
                fetch_corporate_actions=fetch_actions,
                trade_store=trade_store)
    deps.action_calls = action_calls
    return deps
