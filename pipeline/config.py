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

    # Set to e.g. 300 for a fast test run; None = full universe
    "max_universe_for_testing": None,

    # Stage F: chart generation
    "chart_lookback_days": 130,     # trailing window rendered in the VCP chart
    "chart_dir":            os.path.join("data", "charts"),

    # Stage G: VCP vision analysis
    "vcp_top_n":            int(os.environ.get("VCP_TOP_N", 20)),
    "vcp_pivot_window_days": 5,       # local-high/low detection window for contraction analysis
    "anthropic_model":       os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),

    # Output
    "report_dir": os.path.join("data", "reports"),
}


def get_sector_cache_path():
    return os.path.join("data", "sector_cache.csv")
