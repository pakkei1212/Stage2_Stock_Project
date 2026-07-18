"""Re-run Stage G (Claude vision VCP entry analysis) on its own, reusing an
already-saved watchlist CSV and already-rendered charts from data/charts/ —
no re-screening, re-ranking, or re-charting.

Useful for iterating on the VCP prompt/schema, retrying after an API error,
or re-analyzing with a different --top-n without paying for the rest of the
pipeline again.

Run with: python -m pipeline.run_vcp_only [--watchlist PATH] [--top-n N]
"""
import argparse
import glob
import logging
import os
from datetime import datetime

from .config import CONFIG
from .logging_config import setup_logging
from .stage_vcp_analysis import run_vcp_analysis

logger = logging.getLogger(__name__)


def _latest_watchlist(report_dir):
    candidates = sorted(glob.glob(os.path.join(report_dir, "watchlist_*.csv")))
    return candidates[-1] if candidates else None


def main(config=CONFIG):
    setup_logging(config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", help="Path to a watchlist_*.csv (default: latest in data/reports/)")
    parser.add_argument("--top-n", type=int, default=config["vcp_top_n"],
                         help="How many top-ranked rows to analyze (default: %(default)s)")
    args = parser.parse_args()

    import pandas as pd

    watchlist_path = args.watchlist or _latest_watchlist(config["report_dir"])
    if not watchlist_path or not os.path.exists(watchlist_path):
        logger.error("No watchlist CSV found (looked in %s). Run the full pipeline first.", config["report_dir"])
        return

    logger.info("Loading watchlist: %s", watchlist_path)
    watchlist = pd.read_csv(watchlist_path)
    vcp_candidates = watchlist.head(args.top_n)
    if vcp_candidates.empty:
        logger.info("Watchlist is empty — nothing to analyze.")
        return

    chart_paths = {}
    for symbol in vcp_candidates["Symbol"]:
        path = os.path.join(config["chart_dir"], f"{symbol}.png")
        if os.path.exists(path):
            chart_paths[symbol] = path
        else:
            logger.warning("  %s: no chart found at %s, will be skipped", symbol, path)

    logger.info("Running VCP vision analysis on %d candidates (%d charts found)...",
                len(vcp_candidates), len(chart_paths))
    vcp_df = run_vcp_analysis(vcp_candidates, chart_paths, config)

    if vcp_df.empty:
        logger.info("No VCP verdicts produced — done.")
        return

    os.makedirs(config["report_dir"], exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
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


if __name__ == "__main__":
    main()
