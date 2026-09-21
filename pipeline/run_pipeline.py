"""Weekly Stage 2 screener pipeline (native Python — no notebook execution).

Stages:
  A-C  Universe, liquidity/cap filter, sector strength     (data_sources.py)
  D-D2 Minervini Trend Template + fundamentals screen       (stage_screen.py)
  E    Composite technical+fundamentals ranking             (stage_rank.py)
  F    Daily price/volume chart generation                  (stage_charts.py)
  G    Claude vision VCP entry-point analysis (top N only)  (stage_vcp_analysis.py)

Stage H (persistence, trends, Telegram) lives in pipeline/scheduler.py and
calls ``run_screen`` — the one canonical Stage A-G implementation.

Run with: python -m pipeline.run_pipeline
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import CONFIG
from .logging_config import setup_logging
from .data_sources import run_stage_abc
from .stage_screen import run_stage2_screen, fundamentals_screen
from .stage_rank import score_and_rank, TECH_COLS
from .stage_charts import generate_charts
from .stage_vcp_analysis import run_vcp_analysis

# Explicit name: under `python -m pipeline.run_pipeline` __name__ is "__main__",
# which would detach this logger from the configured "pipeline" tree.
logger = logging.getLogger("pipeline.run_pipeline")


@dataclass
class ScreenResult:
    """Everything Stage A-G produced, for Stage H to persist."""
    run_timestamp: str
    chart_dir: str
    final_watchlist: pd.DataFrame = field(default_factory=pd.DataFrame)
    vcp_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart_paths: dict = field(default_factory=dict)
    watchlist_path: Optional[str] = None
    vcp_report_path: Optional[str] = None
    stopped_after: Optional[str] = None     # "C" / "D" when the run ended early
    stage_counts: dict = field(default_factory=dict)

    @property
    def vcp_debug_dir(self):
        return os.path.join(self.chart_dir, "vcp_debug")


def run_screen(config=CONFIG, run_ts=None, stats=None):
    """Stage A-G. Screening logic is exactly what ``main`` has always run.

    ``stats`` (optional dict) receives per-stage counts plus ``current_stage``
    as the run progresses, so a caller can report where a failure happened.
    """
    run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    stats = {} if stats is None else stats

    # Isolate each run's charts in their own subdirectory (data/charts/<run_ts>/)
    # so successive runs don't overwrite one another; the folder name matches the
    # run's audit log (pipeline_<run_ts>.log). vcp_debug charts nest under it too.
    config = {**config, "chart_dir": os.path.join(config["chart_dir"], run_ts)}
    result = ScreenResult(run_timestamp=run_ts, chart_dir=config["chart_dir"], stage_counts=stats)

    logger.info("Stage 2 Screener — weekly pipeline run starting, %s", datetime.now().isoformat())
    logger.debug("Config: %s", config)

    stage_c_survivors, sector_df, market_cap_stats = run_stage_abc(config, stats=stats)
    if not stage_c_survivors:
        logger.warning("No survivors after Stage C — stopping.")
        result.stopped_after = "C"
        return result

    stats["current_stage"] = "D"
    stage_d_results = run_stage2_screen(stage_c_survivors, config)
    stats["stage_d"] = len(stage_d_results)
    if stage_d_results.empty:
        logger.warning("No survivors after Stage D — stopping.")
        result.stopped_after = "D"
        return result
    stats["trend_template_pass"] = int((stage_d_results[TECH_COLS].sum(axis=1) == len(TECH_COLS)).sum())

    stats["current_stage"] = "D2"
    fundamentals_df = fundamentals_screen(stage_d_results["Symbol"].tolist(), config)
    stats["fundamentals_scored"] = len(fundamentals_df)
    stats["current_stage"] = "E"
    final_watchlist = score_and_rank(stage_d_results, fundamentals_df, sector_df, market_cap_stats, config)
    stats["ranked"] = len(final_watchlist)
    result.final_watchlist = final_watchlist

    os.makedirs(config["report_dir"], exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    watchlist_path = os.path.join(config["report_dir"], f"watchlist_{date_str}.csv")
    final_watchlist.to_csv(watchlist_path, index=False)
    result.watchlist_path = watchlist_path
    logger.info("Saved ranked watchlist: %s (%d tickers)", watchlist_path, len(final_watchlist))

    vcp_candidates = final_watchlist.head(config["vcp_top_n"])
    if vcp_candidates.empty:
        logger.info("No candidates to chart/analyze — done.")
        return result

    stats["current_stage"] = "F"
    logger.info("Generating charts for top %d candidates...", len(vcp_candidates))
    chart_paths = generate_charts(vcp_candidates["Symbol"].tolist(), config)
    stats["charts"] = len(chart_paths)
    result.chart_paths = chart_paths

    stats["current_stage"] = "G"
    logger.info("Running VCP vision analysis on %d charts...", len(chart_paths))
    vcp_df = run_vcp_analysis(vcp_candidates, chart_paths, config)
    result.vcp_df = vcp_df
    stats["vcp_analyzed"] = len(vcp_df)

    if not vcp_df.empty:
        merged = vcp_candidates.merge(vcp_df, on="Symbol", how="left")
        vcp_report_path = os.path.join(config["report_dir"], f"vcp_analysis_{date_str}.csv")
        merged.to_csv(vcp_report_path, index=False)
        result.vcp_report_path = vcp_report_path
        logger.info("Saved VCP analysis report: %s", vcp_report_path)

        if "is_vcp_pattern" in vcp_df.columns:
            stats["vcp_pattern"] = int((vcp_df["is_vcp_pattern"] == True).sum())  # noqa: E712 (NaN-safe)
        actionable = merged[merged.get("entry_recommendation").isin(["buy_now", "wait_for_breakout"])] \
            if "entry_recommendation" in merged.columns else merged.iloc[0:0]
        stats["vcp_actionable"] = len(actionable)
        logger.info("%d candidate(s) flagged buy_now / wait_for_breakout", len(actionable))
        if not actionable.empty:
            logger.info("\n%s", actionable[["Symbol", "pattern_stage", "confidence", "entry_recommendation"]]
                         .to_string(index=False))

    stats["current_stage"] = None
    logger.info("Pipeline run complete.")
    return result


def main(config=CONFIG):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(config, run_timestamp=run_ts)
    run_screen(config, run_ts)


if __name__ == "__main__":
    main()
