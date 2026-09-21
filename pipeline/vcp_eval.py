"""Non-persistent Claude VCP evaluation: repeat the live Stage-G call a few times
for a handful of symbols to check verdict stability.

  python -m pipeline.vcp_eval CACC HALO CORT IBKR SLDE --repeats 3

Uses exactly the Stage F chart renderer, the frozen numeric detector and the
production ``analyze_chart`` call, but writes ONLY to
``data/reports/vcp_eval_<ts>/`` (charts + results.json). It never touches the
SQLite history, run manifests, the canonical daily result or Telegram.
Makes real (billed) Claude API calls: symbols x repeats.
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime

from .config import CONFIG
from .logging_config import setup_logging
from .stage_vcp_analysis import (analyze_chart, compute_vcp_metrics, llm_review_decision, new_claude_stats,
                                 prompt_adherence_warnings, rationale_warnings, vcp_agreement)

# Explicit name: under `python -m pipeline.vcp_eval` __name__ is "__main__".
logger = logging.getLogger("pipeline.vcp_eval")

STABILITY_FIELDS = ("is_vcp_pattern", "pattern_stage", "entry_recommendation", "confidence")
NUMERIC_FIELDS = ("vcp_numeric_quality", "tightening_quality", "volume_dryup_quality", "pivot_state",
                  "final_contraction_confirmed", "contraction_pcts", "pivot_price_candidate", "pct_from_pivot")
MAX_REPEATS = 5


def summarize_stability(runs):
    """Per-field distinct values across successful runs, plus pivot spread."""
    ok = [r for r in runs if "error" not in r]
    summary = {"successful_runs": len(ok), "failed_runs": len(runs) - len(ok)}
    for f in STABILITY_FIELDS:
        counts = Counter(str(r.get(f)) for r in ok)
        summary[f] = {"values": dict(counts), "stable": len(counts) <= 1}
    pivots = [r["pivot_price"] for r in ok if isinstance(r.get("pivot_price"), (int, float))]
    if pivots:
        lo, hi = min(pivots), max(pivots)
        summary["pivot_price"] = {"min": lo, "max": hi, "spread_pct": round((hi - lo) / lo * 100, 2) if lo else None}
    else:
        summary["pivot_price"] = {"min": None, "max": None, "spread_pct": None}
    summary["all_stable"] = all(summary[f]["stable"] for f in STABILITY_FIELDS)
    return summary


def evaluate(symbols, repeats, config=CONFIG, client=None, fetch=None, render=None, out_dir=None):
    """Returns (results, stats). ``client``/``fetch``/``render`` are injectable for tests."""
    repeats = max(1, min(int(repeats), MAX_REPEATS))
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    if fetch is None or render is None:
        from .stage_charts import fetch_chart_data, render_chart
        fetch, render = fetch or fetch_chart_data, render or render_chart
    out_dir = out_dir or os.path.join(config["report_dir"], f"vcp_eval_{datetime.now():%Y%m%d_%H%M%S}")
    cfg = {**config, "chart_dir": os.path.join(out_dir, "charts")}
    stats = new_claude_stats()
    results = []
    for symbol in symbols:
        df = fetch(symbol, cfg)
        metrics, details = compute_vcp_metrics(df, cfg, return_details=True)
        entry = {"symbol": symbol, "last_bar": str(df.dropna(subset=["Close"]).index[-1].date()) if len(df) else None}
        if metrics is None:
            entry["skipped"] = "not enough history"
            results.append(entry)
            continue
        chart = render(symbol, df, cfg)
        # Same prompt as production: the selective routing reasons are always passed,
        # even for a symbol selective mode would not route (reasons are then empty).
        routed, reasons = llm_review_decision(metrics, details, {**cfg, "vcp_llm_review_mode": "selective",
                                                                 "vcp_live_analysis": True})
        runs = []
        for i in range(repeats):
            try:
                verdict = analyze_chart(client, symbol, chart, metrics, cfg, legs=details["contraction_legs"],
                                        stats=stats, review_reasons=reasons)
                if "error" not in verdict:
                    verdict["review"] = {"agreement": vcp_agreement(metrics["vcp_numeric_quality"],
                                                                    verdict.get("is_vcp_pattern")),
                                         "rationale_warnings": rationale_warnings(verdict),
                                         "prompt_adherence_warnings": prompt_adherence_warnings(verdict, reasons)}
            except Exception as e:                           # one failed call never stops the evaluation
                stats["errors"] += 1
                verdict = {"error": f"{type(e).__name__}: {e}"[:300]}
            logger.info("  %s run %d/%d: %s", symbol, i + 1, repeats,
                        verdict.get("error") or f"{verdict.get('is_vcp_pattern')} {verdict.get('pattern_stage')} "
                                                f"{verdict.get('entry_recommendation')} {verdict.get('confidence')} "
                                                f"pivot {verdict.get('pivot_price')}")
            for w in (verdict.get("review") or {}).get("prompt_adherence_warnings", []):
                logger.warning("  %s run %d/%d: prompt adherence: %s", symbol, i + 1, repeats, w)
            runs.append(verdict)
        entry.update({"numeric": {k: metrics.get(k) for k in NUMERIC_FIELDS},
                      "selective_routing": {"requested": routed, "reasons": reasons}, "chart": chart, "runs": runs,
                      "stability": summarize_stability(runs)})
        results.append(entry)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"model": config["anthropic_model"], "repeats": repeats, "stats": stats, "results": results},
                  f, indent=2, default=str)
    return results, stats, out_dir


def main(argv=None, config=CONFIG):
    parser = argparse.ArgumentParser(prog="python -m pipeline.vcp_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--repeats", type=int, default=3, help=f"calls per symbol (1-{MAX_REPEATS})")
    args = parser.parse_args(argv)
    setup_logging(config, log_prefix="vcp_eval")
    symbols = [s.upper() for s in args.symbols]
    logger.info("VCP evaluation (non-persistent): %s x %d with %s", ",".join(symbols), args.repeats,
                config["anthropic_model"])
    results, stats, out_dir = evaluate(symbols, args.repeats, config)
    for r in results:
        s = r.get("stability")
        if not s:
            logger.info("%s: %s", r["symbol"], r.get("skipped"))
            continue
        logger.info("%s (numeric %s): stable=%s %s pivot spread %s%%", r["symbol"], r["numeric"]["vcp_numeric_quality"],
                    s["all_stable"], {f: s[f]["values"] for f in STABILITY_FIELDS}, s["pivot_price"]["spread_pct"])
    logger.info("Claude calls: %s. Results: %s", stats, os.path.join(out_dir, "results.json"))
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
