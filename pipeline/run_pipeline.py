"""Weekly Stage 2 screener pipeline (native Python — no notebook execution).

Stages:
  A-C  Universe, liquidity/cap filter, sector strength     (data_sources.py)
  D-D2 Minervini Trend Template + fundamentals screen       (stage_screen.py)
  E    Composite technical+fundamentals ranking             (stage_rank.py)
  F    Daily price/volume chart generation                  (stage_charts.py)
  G    Claude vision VCP entry-point analysis (top N only)  (stage_vcp_analysis.py)

Run with: python -m pipeline.run_pipeline
"""
import os
from datetime import datetime

from .config import CONFIG
from .data_sources import run_stage_abc
from .stage_screen import run_stage2_screen, fundamentals_screen
from .stage_rank import score_and_rank
from .stage_charts import generate_charts
from .stage_vcp_analysis import run_vcp_analysis


def main(config=CONFIG):
    print("=" * 70)
    print(f"Stage 2 Screener — weekly pipeline run, {datetime.now().isoformat()}")
    print("=" * 70)

    stage_c_survivors, sector_df, market_cap_stats = run_stage_abc(config)
    if not stage_c_survivors:
        print("No survivors after Stage C — stopping.")
        return

    stage_d_results = run_stage2_screen(stage_c_survivors, config)
    if stage_d_results.empty:
        print("No survivors after Stage D — stopping.")
        return

    fundamentals_df = fundamentals_screen(stage_d_results["Symbol"].tolist(), config)
    final_watchlist = score_and_rank(stage_d_results, fundamentals_df, sector_df, market_cap_stats, config)

    os.makedirs(config["report_dir"], exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    watchlist_path = os.path.join(config["report_dir"], f"watchlist_{date_str}.csv")
    final_watchlist.to_csv(watchlist_path, index=False)
    print(f"Saved ranked watchlist: {watchlist_path} ({len(final_watchlist)} tickers)")

    vcp_candidates = final_watchlist.head(config["vcp_top_n"])
    if vcp_candidates.empty:
        print("No candidates to chart/analyze — done.")
        return

    print(f"\nGenerating charts for top {len(vcp_candidates)} candidates...")
    chart_paths = generate_charts(vcp_candidates["Symbol"].tolist(), config)

    print(f"\nRunning VCP vision analysis on {len(chart_paths)} charts...")
    vcp_df = run_vcp_analysis(vcp_candidates, chart_paths, config)

    if not vcp_df.empty:
        merged = vcp_candidates.merge(vcp_df, on="Symbol", how="left")
        vcp_report_path = os.path.join(config["report_dir"], f"vcp_analysis_{date_str}.csv")
        merged.to_csv(vcp_report_path, index=False)
        print(f"Saved VCP analysis report: {vcp_report_path}")

        actionable = merged[merged.get("entry_recommendation").isin(["buy_now", "wait_for_breakout"])] \
            if "entry_recommendation" in merged.columns else merged.iloc[0:0]
        print(f"\n{len(actionable)} candidate(s) flagged buy_now / wait_for_breakout:")
        if not actionable.empty:
            print(actionable[["Symbol", "pattern_stage", "confidence", "entry_recommendation"]].to_string(index=False))

    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()
