"""Weekly Stage 2 screener pipeline (native Python — no notebook execution).

Stages:
  A-C  Universe, liquidity/cap filter, sector strength     (data_sources.py)
  D-D2 Minervini Trend Template + fundamentals screen       (stage_screen.py)
  E    Composite technical+fundamentals ranking             (stage_rank.py)
  F    Daily price/volume chart generation                  (stage_charts.py)
  G    Claude vision VCP entry-point analysis (top N only)  (stage_vcp_analysis.py)

Run with: python -m pipeline.run_pipeline
"""
import logging
import os
from datetime import datetime

from .config import CONFIG
from .logging_config import setup_logging
from .data_sources import run_stage_abc
from .stage_screen import run_stage2_screen, fundamentals_screen
from .stage_rank import score_and_rank
from .stage_charts import generate_charts
from .stage_vcp_analysis import run_vcp_analysis

logger = logging.getLogger(__name__)


def main(config=CONFIG):
    setup_logging(config)

    logger.info("Stage 2 Screener — weekly pipeline run starting, %s", datetime.now().isoformat())
    logger.debug("Config: %s", config)

    stage_c_survivors, sector_df, market_cap_stats = run_stage_abc(config)
    if not stage_c_survivors:
        logger.warning("No survivors after Stage C — stopping.")
        return

    stage_d_results = run_stage2_screen(stage_c_survivors, config)
    if stage_d_results.empty:
        logger.warning("No survivors after Stage D — stopping.")
        return

    fundamentals_df = fundamentals_screen(stage_d_results["Symbol"].tolist(), config)
    final_watchlist = score_and_rank(stage_d_results, fundamentals_df, sector_df, market_cap_stats, config)

    os.makedirs(config["report_dir"], exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    watchlist_path = os.path.join(config["report_dir"], f"watchlist_{date_str}.csv")
    final_watchlist.to_csv(watchlist_path, index=False)
    logger.info("Saved ranked watchlist: %s (%d tickers)", watchlist_path, len(final_watchlist))

    vcp_candidates = final_watchlist.head(config["vcp_top_n"])
    if vcp_candidates.empty:
        logger.info("No candidates to chart/analyze — done.")
        return

    logger.info("Generating charts for top %d candidates...", len(vcp_candidates))
    chart_paths = generate_charts(vcp_candidates["Symbol"].tolist(), config)

    logger.info("Running VCP vision analysis on %d charts...", len(chart_paths))
    vcp_df = run_vcp_analysis(vcp_candidates, chart_paths, config)

    if not vcp_df.empty:
        merged = vcp_candidates.merge(vcp_df, on="Symbol", how="left")
        vcp_report_path = os.path.join(config["report_dir"], f"vcp_analysis_{date_str}.csv")
        merged.to_csv(vcp_report_path, index=False)
        logger.info("Saved VCP analysis report: %s", vcp_report_path)

        actionable = merged[merged.get("entry_recommendation").isin(["buy_now", "wait_for_breakout"])] \
            if "entry_recommendation" in merged.columns else merged.iloc[0:0]
        logger.info("%d candidate(s) flagged buy_now / wait_for_breakout", len(actionable))
        if not actionable.empty:
            logger.info("\n%s", actionable[["Symbol", "pattern_stage", "confidence", "entry_recommendation"]]
                         .to_string(index=False))

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
