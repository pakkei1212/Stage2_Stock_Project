import os

CONFIG = {
    # Stage B: liquidity filter
    "min_price":               5.0,
    "max_price":               2000.0,
    "min_avg_dollar_volume":   5_000_000,
    "bulk_lookback_days":      30,

    # Stage B: market cap (mid+large only)
    "min_market_cap":          2_000_000_000,   # $2B floor
    "max_market_cap":          None,             # no ceiling

    # Stage C: sector strength
    "top_n_sectors":           5,
    "sector_rs_lookback_days": 90,

    # Stage D: Minervini Trend Template (daily bars)
    "full_lookback_days":      400,
    "ma_slope_window":         20,
    "near_high_pct":           0.25,   # within 25% of 52w high
    "above_low_pct":           0.25,   # at least 25% above 52w low
    "rs_benchmark":            "^IXIC",

    # Stage D2: fundamentals thresholds
    "min_quarterly_eps_growth":   0.20,
    "min_quarterly_sales_growth": 0.10,
    "min_roe":                    0.17,
    "min_profit_margin":          0.0,
    "fundamentals_weight":        1.0,

    # Batching / rate-limit politeness
    "batch_size":       50,
    "batch_sleep_sec":  1.0,
    "max_retries":      3,

    # OHLCV cache (pipeline/ohlcv_cache.py) — persist downloaded price history
    # between runs and fetch only the delta each week. Big runtime + rate-limit
    # win, and accumulates history for backtesting.
    "ohlcv_cache_enabled":       os.environ.get("OHLCV_CACHE", "1") != "0",
    "ohlcv_cache_dir":           os.path.join("data", "ohlcv_cache"),
    "ohlcv_cache_max_age_days":  1,   # cache newer than this (vs today) is served as-is
    "ohlcv_cache_overlap_days":  5,   # re-fetch this much overlap to catch splits on append

    # Market-cap cache — caps barely move week to week; reuse within the TTL
    # instead of the slow per-ticker throttled fetch.
    "market_cap_cache_path":     os.path.join("data", "market_cap_cache.csv"),
    "market_cap_cache_ttl_days": 7,

    # Set to e.g. 300 for a fast test run; None = full universe
    "max_universe_for_testing": None,

    # Stage F: chart generation
    "chart_lookback_days": 130,     # trailing window rendered in the VCP chart
    "chart_dir":            os.path.join("data", "charts"),

    # Stage G: VCP vision analysis
    "vcp_top_n":            int(os.environ.get("VCP_TOP_N", 20)),
    "vcp_pivot_window_days": 5,       # local-high/low detection window for contraction analysis
    # Quantitative VCP detector heuristics (pipeline/stage_vcp_analysis.py). These
    # are OUR detector's defaults, chosen on synthetic structure — not rules
    # prescribed by Minervini — and should be evaluated against stored history.
    "vcp_min_swing_pct":        2.0,   # Close swings smaller than this are noise, not contractions
    "vcp_resistance_band_pct":  10.0,  # one base: all contraction highs within this % of the base high
    "vcp_max_expansion_ratio":  1.25,  # a later leg (depth or volume) above this x the prior = large expansion
    "vcp_final_zone_days":      5,     # sessions forming the final tight zone near the pivot
    "vcp_pivot_proximity_pct":  5.0,   # "near" the pivot: coiling below / fresh breakout above within this %
    "vcp_breakout_volume_ratio": 1.5,  # breakout-bar volume vs prior 20-day avg needed for volume confirmation
    "anthropic_model":       os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
    # Send charts + metrics to Claude for the VCP verdict (spends tokens). Set
    # VCP_LIVE_ANALYSIS=0 for token-free runs that compute metrics and render
    # the verification charts only, skipping the paid vision call.
    "vcp_live_analysis":     os.environ.get("VCP_LIVE_ANALYSIS", "1") != "0",
    # Which candidates Claude reviews when live analysis is on (stage_vcp_analysis.
    # llm_review_mode): off | selective (numeric first; Claude only for ambiguous /
    # visually unusual setups) | all (every Stage-G candidate, for research).
    # VCP_LIVE_ANALYSIS=0 still forces off.
    "vcp_llm_review_mode":   os.environ.get("VCP_LLM_REVIEW_MODE", "selective"),
    # Render a metrics-verification chart per analyzed symbol (pivots,
    # contraction depths, pivot/current price, per-contraction volume overlaid)
    # into chart_dir/vcp_debug/ so the computed numbers can be eyeballed vs the chart.
    "vcp_debug_charts":      os.environ.get("VCP_DEBUG_CHARTS", "1") != "0",

    # Output
    "report_dir": os.path.join("data", "reports"),

    # Logging (see pipeline/logging_config.py) — console + a timestamped,
    # per-run audit log file under log_dir.
    "log_dir":   os.path.join("data", "logs"),
    "log_level": os.environ.get("LOG_LEVEL", "INFO"),

    # ── Stage H: scheduling, history, trends, notification ──────────────────
    # Everything below is post-Stage-G and operational: none of it feeds Stage
    # A-G screening, so it is excluded from the run's config_hash (see
    # run_metadata.OPERATIONAL_KEY_PREFIXES).

    # WHEN to check (local trigger) is independent of WHICH session is processed
    # (always the latest completed NASDAQ session, via trading_calendar.py). The
    # check runs every local calendar day, weekends included.
    # Main check frequency (pipeline/check_schedule.py): daily | times | interval.
    # More frequent checks never mean more Stage A-G runs — at most one per
    # completed XNAS session. Kept as raw strings; validated when the scheduler
    # starts (a bad value is a clear ScheduleConfigError, not an import crash).
    "scheduler_enabled":         os.environ.get("SCHEDULE_ENABLED", "1") != "0",
    "scheduler_mode":            os.environ.get("SCHEDULE_MODE", "daily"),
    "scheduler_local_time":      os.environ.get("LOCAL_SCHEDULER_TIME", "09:15"),       # daily
    "scheduler_times":           os.environ.get("SCHEDULE_TIMES", ""),                  # times
    "scheduler_interval_minutes": os.environ.get("SCHEDULE_INTERVAL_MINUTES", "60"),    # interval
    "scheduler_timezone":        os.environ.get("LOCAL_SCHEDULER_TIMEZONE", "Asia/Singapore"),
    "scheduler_run_on_startup":  os.environ.get("RUN_ON_STARTUP_CHECK", "1") != "0",
    "scheduler_poll_seconds":    int(os.environ.get("SCHEDULER_POLL_SECONDS", 60)),
    # A session counts as completed this long after its (possibly early) close.
    "session_close_buffer_minutes": int(os.environ.get("SESSION_CLOSE_BUFFER_MINUTES", 60)),
    # An active (WAITING_FOR_DATA/RUNNING) claim older than this is presumed to
    # belong to a crashed process and is marked ABANDONED. Claims are only made
    # while holding the exclusive run lock, so a leftover active row on this
    # machine is already dead; keep this comfortably above a full A-G run.
    "run_stale_after_hours":     float(os.environ.get("RUN_STALE_AFTER_HOURS", 6)),

    # Data-readiness guard: the reference symbol's latest daily bar must equal
    # the target session before Stage A-G is launched.
    "data_ready_symbol":         os.environ.get("DATA_READY_SYMBOL", "^IXIC"),
    "data_ready_retry_minutes":  float(os.environ.get("DATA_READY_RETRY_MINUTES", 10)),
    "data_ready_max_attempts":   int(os.environ.get("DATA_READY_MAX_ATTEMPTS", 3)),
    # If the session is still not ready after those short retries, the
    # long-running loop keeps it pending and re-probes readiness (one cheap
    # download, no Stage A-G) every deferred_retry_minutes until the local
    # cutoff; after that it waits for the next startup / daily check.
    "deferred_retry_enabled":    os.environ.get("DEFERRED_RETRY_ENABLED", "1") != "0",
    "deferred_retry_minutes":    float(os.environ.get("DEFERRED_RETRY_MINUTES", 60)),
    "deferred_retry_cutoff":     os.environ.get("DEFERRED_RETRY_CUTOFF", "20:30"),

    # Canonical longitudinal store + per-session run manifests.
    "history_db_path":           os.environ.get("HISTORY_DB_PATH", os.path.join("data", "history", "stage2_history.sqlite3")),
    "history_lock_dir":          os.path.join("data", "history"),
    "run_manifest_dir":          os.path.join("data", "runs"),

    # Top-N trend monitoring (reporting only — full ranked results are stored).
    "trend_top_n":               int(os.environ.get("TREND_TOP_N", 10)),
    "trend_lookback_sessions":   int(os.environ.get("TREND_LOOKBACK_SESSIONS", 10)),
    "trend_rank_move_min":       int(os.environ.get("TREND_RANK_MOVE_MIN", 3)),
    "trend_persistent_min_streak": int(os.environ.get("TREND_PERSISTENT_MIN_STREAK", 3)),

    # Telegram (credentials are read from the environment at send time and are
    # never stored in CONFIG, logs, SQLite or manifests).
    "telegram_enabled":          os.environ.get("TELEGRAM_ENABLED", "0") != "0",
    "telegram_send_charts":      os.environ.get("TELEGRAM_SEND_CHARTS", "1") != "0",
    # VCP verification (debug) charts for the VCP review set, VCP-priority order
    # (replaces the old overall-rank TELEGRAM_CHART_TOP_N).
    "telegram_vcp_chart_max":    int(os.environ.get("TELEGRAM_VCP_CHART_MAX", 10)),
    "telegram_ops_alerts":       os.environ.get("TELEGRAM_OPS_ALERTS", "1") != "0",
    "telegram_max_attempts":     int(os.environ.get("TELEGRAM_MAX_ATTEMPTS", 3)),
    "telegram_report_candidates": int(os.environ.get("TELEGRAM_REPORT_CANDIDATES", 10)),
    # Separate 🛠 debug message (stage counts, Stage-G/I, monitor, corporate
    # actions, run facts) after the main dashboard. Reporting only.
    "telegram_debug_summary":    os.environ.get("TELEGRAM_DEBUG_SUMMARY", "1") != "0",

    # ── Stage I: trade journal + position monitor (advisory only) ───────────
    # Strictly post-Stage-G and advisory: nothing below feeds Stage A-G
    # screening, scoring, ranking or the VCP detector, and none of it ever
    # places, modifies or cancels a broker order — there is no brokerage API,
    # no account access and no execution anywhere in this project. The app only
    # records fills the user supplies, computes plans, monitors public market
    # data and sends Telegram advisories.
    #
    # Every threshold here is OUR provisional strategy heuristic, chosen from
    # structure/volatility reasoning rather than measured outcomes. They are NOT
    # rules prescribed by Minervini, and they should be re-evaluated against
    # recorded trade history before being trusted. All are configurable.
    "trade_journal_enabled":     os.environ.get("TRADE_JOURNAL_ENABLED", "1") != "0",
    # Separate database file: real trades never share tables (or a file) with
    # the canonical screening history.
    "trade_db_path":             os.environ.get("TRADE_DB_PATH", os.path.join("data", "trades", "trade_journal.sqlite3")),

    # Planning parameters (§4). No brokerage balance is read or required — this
    # is a number the user types into .env.
    "trade_portfolio_value":     float(os.environ.get("PORTFOLIO_VALUE", 100_000)),
    "trade_risk_per_trade_pct":  float(os.environ.get("RISK_PER_TRADE_PCT", 1.0)),
    "trade_max_position_portion_pct": float(os.environ.get("MAX_POSITION_PORTION_PCT", 25.0)),

    # Layer 2 — entry plan. The authoritative pivot is always the numeric
    # detector's pivot; Claude never sets an entry price. Breakout-volume
    # confirmation reuses the Stage-G threshold "vcp_breakout_volume_ratio".
    "trade_entry_trigger_buffer_pct": float(os.environ.get("ENTRY_TRIGGER_BUFFER_PCT", 0.1)),
    "trade_max_chase_pct":       float(os.environ.get("MAX_CHASE_PCT", 5.0)),

    # Layer 3 — initial risk plan. Structure first (the final contraction low),
    # with volatility as the floor on how tight a stop may sensibly be.
    "trade_atr_period":          int(os.environ.get("ATR_PERIOD", 20)),
    "trade_atr_stop_multiple":   float(os.environ.get("ATR_STOP_MULTIPLE", 2.0)),
    "trade_structural_stop_buffer_pct": float(os.environ.get("STRUCTURAL_STOP_BUFFER_PCT", 1.0)),
    "trade_swing_lookback_days": int(os.environ.get("SWING_LOOKBACK_DAYS", 20)),
    # The monitor's swing low ignores this many recent sessions, so "price broke
    # the swing low" cannot be satisfied by today's own low (see
    # position_monitor.structural_swing_low).
    "trade_swing_exclude_recent_days": int(os.environ.get("SWING_EXCLUDE_RECENT_DAYS", 3)),
    # Hard ceiling on stop distance, whatever structure/volatility ask for. A
    # required stop wider than the applicable ceiling is SKIP_RISK_TOO_WIDE —
    # the stop is never simply widened to fit.
    "trade_max_risk_pct":        float(os.environ.get("MAX_RISK_PCT", 10.0)),
    # Market cap is a SECONDARY modifier of that ceiling (and of trailing room),
    # never a direct stop rule: (min_market_cap, max_risk_pct, trail_atr_multiple).
    # Matched top-down; the first tier whose floor the cap meets applies.
    "trade_market_cap_tiers": (
        (200_000_000_000, 7.0, 2.0),     # mega cap: typically calmer, tighter ceiling
        (10_000_000_000,  8.0, 2.5),     # large cap
        (2_000_000_000,  10.0, 3.0),     # mid cap (Stage B floor)
    ),
    "trade_default_risk_ceiling_pct": float(os.environ.get("DEFAULT_RISK_CEILING_PCT", 8.0)),
    "trade_default_trail_atr_multiple": float(os.environ.get("DEFAULT_TRAIL_ATR_MULTIPLE", 2.5)),
    # Liquidity modifier: a thin mover gets a tighter ceiling (harder to exit).
    "trade_liquidity_floor_usd": float(os.environ.get("LIQUIDITY_FLOOR_USD", 20_000_000)),
    "trade_illiquid_ceiling_factor": float(os.environ.get("ILLIQUID_CEILING_FACTOR", 0.8)),
    # Actual volatility outranks market cap: a high-ATR name gets extra trailing
    # room regardless of its tier.
    "trade_high_volatility_atr_pct": float(os.environ.get("HIGH_VOLATILITY_ATR_PCT", 5.0)),
    "trade_high_volatility_trail_bonus": float(os.environ.get("HIGH_VOLATILITY_TRAIL_BONUS", 0.5)),

    # Layer 3 — winning-exit monitoring (§10/§11). Progressive protection: each
    # cushion (unrealised P&L %) unlocks a tighter protective reference. There
    # is deliberately no universal "take profit at +20%" rule.
    "trade_breakeven_cushion_pct":   float(os.environ.get("BREAKEVEN_CUSHION_PCT", 5.0)),
    "trade_swing_trail_cushion_pct": float(os.environ.get("SWING_TRAIL_CUSHION_PCT", 8.0)),
    "trade_ema21_trail_cushion_pct": float(os.environ.get("EMA21_TRAIL_CUSHION_PCT", 12.0)),
    "trade_ema10_trail_cushion_pct": float(os.environ.get("EMA10_TRAIL_CUSHION_PCT", 25.0)),
    "trade_partial_profit_cushion_pct": float(os.environ.get("PARTIAL_PROFIT_CUSHION_PCT", 20.0)),
    # Climax/extension check for PARTIAL_PROFIT_REVIEW.
    "trade_climax_ext_ema10_pct":    float(os.environ.get("CLIMAX_EXT_EMA10_PCT", 15.0)),
    # Failed-breakout window and give-back below the setup pivot.
    "trade_failed_breakout_days":    int(os.environ.get("FAILED_BREAKOUT_DAYS", 15)),
    "trade_failed_breakout_pct":     float(os.environ.get("FAILED_BREAKOUT_PCT", 3.0)),
    # Abnormal high-volume reversal: volume vs 20d average, plus a close in the
    # bottom fraction of the session's range.
    "trade_reversal_volume_ratio":   float(os.environ.get("REVERSAL_VOLUME_RATIO", 2.0)),
    "trade_reversal_close_location": float(os.environ.get("REVERSAL_CLOSE_LOCATION", 0.35)),
    # Bars pulled per held symbol for the daily monitor (through the OHLCV cache).
    "trade_monitor_lookback_days":   int(os.environ.get("TRADE_MONITOR_LOOKBACK_DAYS", 260)),

    # Corporate actions (Stage I, advisory). Splits, dividends and ticker
    # changes are applied to the journal's DERIVED values only; mergers,
    # spin-offs, rights and delistings are recorded and flagged, never guessed.
    #
    # "trade_price_basis" is a statement of fact about our data source, not a
    # preference: yfinance runs with auto_adjust=True, so every bar is split-
    # AND dividend-adjusted and must never be adjusted a second time by us.
    # Change it only if the OHLCV source itself changes.
    "trade_price_basis":             "split_and_dividend_adjusted",
    "trade_corporate_actions_enabled": os.environ.get("CORPORATE_ACTIONS_ENABLED", "1") != "0",
    "trade_corporate_action_lookback_days": int(os.environ.get("CORPORATE_ACTION_LOOKBACK_DAYS", 400)),
    # Special-dividend heuristics (OURS, provisional): a one-off payment worth
    # at least this much of the reference price, or this many times the recent
    # regular dividends, is treated as special and flagged for review.
    "trade_special_dividend_yield_pct": float(os.environ.get("SPECIAL_DIVIDEND_YIELD_PCT", 3.0)),
    "trade_special_dividend_multiple": float(os.environ.get("SPECIAL_DIVIDEND_MULTIPLE", 3.0)),
    # How long after an ex-date a dividend is still considered when judging a
    # price move (only used when prices are raw; with adjusted bars it is a no-op).
    "trade_dividend_signal_window_sessions": int(os.environ.get("DIVIDEND_SIGNAL_WINDOW_SESSIONS", 3)),
    # Consecutive sessions without market data before an open position is put
    # into CORPORATE_ACTION_REVIEW. Missing data is never read as a sale.
    "trade_missing_bars_review_sessions": int(os.environ.get("MISSING_BARS_REVIEW_SESSIONS", 3)),

    # Telegram trade journal (pipeline/trade_bot.py — a separate long-poll
    # process). Mutating commands are accepted only from this allowlist; the
    # values are read from the environment at send/receive time and are never
    # stored in CONFIG, SQLite, manifests or logs.
    "trade_bot_enabled":         os.environ.get("TRADE_BOT_ENABLED", "0") != "0",
    "trade_bot_poll_seconds":    int(os.environ.get("TRADE_BOT_POLL_SECONDS", 20)),
    "trade_confirm_ttl_minutes": int(os.environ.get("TRADE_CONFIRM_TTL_MINUTES", 30)),
    "trade_positions_in_report": os.environ.get("TRADE_POSITIONS_IN_REPORT", "1") != "0",
}


def get_sector_cache_path():
    return os.path.join("data", "sector_cache.csv")
