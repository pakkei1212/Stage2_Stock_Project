"""Stage G: quantitative contraction metrics + Claude vision VCP entry analysis.

Design: VCP (Volatility Contraction Pattern) detection is done numerically
first from OHLCV data (contraction depths, volume dry-up) since that's exact
and cheap; those numbers are then handed to Claude *alongside* the chart
image so the model is confirming/narrating a pattern with numeric grounding
rather than eyeballing contractions from pixels alone.
"""
import base64
import json
import logging
import os

import numpy as np
import pandas as pd

from .config import CONFIG

logger = logging.getLogger(__name__)

VCP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_vcp_pattern": {"type": "boolean"},
        "pattern_stage": {
            "type": "string",
            "enum": ["forming", "mature", "breaking_out", "failed", "not_present"],
        },
        "contraction_count_observed": {"type": "integer"},
        "volume_dry_up_confirmed": {"type": "boolean"},
        "pivot_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "suggested_stop_loss": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "entry_recommendation": {
            "type": "string",
            "enum": ["buy_now", "wait_for_breakout", "wait_for_better_setup", "avoid"],
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "is_vcp_pattern", "pattern_stage", "contraction_count_observed",
        "volume_dry_up_confirmed", "pivot_price", "suggested_stop_loss",
        "confidence", "entry_recommendation", "rationale",
    ],
    "additionalProperties": False,
}


def _find_pivots(close, window):
    """A point is a pivot high/low if it's the max/min within +/- window days."""
    pivots = []
    n = len(close)
    for i in range(window, n - window):
        seg = close.iloc[i - window:i + window + 1]
        val = close.iloc[i]
        if val == seg.max():
            pivots.append((i, "high", val))
        elif val == seg.min():
            pivots.append((i, "low", val))
    return pivots


def _collapse_alternating(pivots):
    """Collapse consecutive same-type pivots to the most extreme one, so the
    sequence strictly alternates high/low."""
    collapsed = []
    for idx, kind, val in pivots:
        if collapsed and collapsed[-1][1] == kind:
            keep_prev = (kind == "high" and collapsed[-1][2] >= val) or \
                        (kind == "low" and collapsed[-1][2] <= val)
            if keep_prev:
                continue
            collapsed[-1] = (idx, kind, val)
        else:
            collapsed.append((idx, kind, val))
    return collapsed


def compute_vcp_metrics(df, config=CONFIG):
    """Computes numeric VCP signals from the trailing chart window.

    Returns a dict of metrics, or None if there isn't enough trailing history.
    """
    window = config["chart_lookback_days"]
    pivot_window = config["vcp_pivot_window_days"]
    base = df.iloc[-window:].dropna(subset=["Close", "Volume"])
    if len(base) < pivot_window * 4:
        return None

    close = base["Close"]
    pivots = _collapse_alternating(_find_pivots(close, pivot_window))

    contractions = []
    for (idx_h, kind_h, val_h), (idx_l, kind_l, val_l) in zip(pivots, pivots[1:]):
        if kind_h == "high" and kind_l == "low" and val_h > 0:
            contractions.append((val_h - val_l) / val_h)

    is_contracting = (
        len(contractions) >= 2
        and contractions[-1] < contractions[0]
    )

    half = len(base) // 2
    vol_first_half = base["Volume"].iloc[:half].mean()
    vol_second_half = base["Volume"].iloc[half:].mean()
    volume_dryup_ratio = (
        vol_second_half / vol_first_half if vol_first_half else np.nan
    )

    last_high_pivots = [v for _, k, v in pivots if k == "high"]
    pivot_price_candidate = last_high_pivots[-1] if last_high_pivots else float(close.max())
    current_price = float(close.iloc[-1])
    pct_below_pivot = (pivot_price_candidate - current_price) / pivot_price_candidate \
        if pivot_price_candidate else np.nan

    return {
        "contraction_pcts": [round(c * 100, 2) for c in contractions],
        "contraction_count": len(contractions),
        "contractions_decreasing": bool(is_contracting),
        "volume_dryup_ratio": round(float(volume_dryup_ratio), 3) if pd.notna(volume_dryup_ratio) else None,
        "pivot_price_candidate": round(float(pivot_price_candidate), 2),
        "current_price": round(current_price, 2),
        "pct_below_pivot": round(float(pct_below_pivot) * 100, 2) if pd.notna(pct_below_pivot) else None,
    }


def _build_prompt(ticker, metrics):
    return f"""You are analyzing {ticker} for a Minervini-style Volatility Contraction
Pattern (VCP) entry setup. VCP is a base-building pattern where each pullback
in a consolidation is shallower than the last (decreasing volatility), ideally
accompanied by declining volume as the base tightens, ahead of a breakout
above the pivot (the most recent swing high) on rising volume.

Computed metrics from the trailing {CONFIG['chart_lookback_days']}-day price/volume
history (use these as ground truth for the numbers; use the attached chart image
to judge the visual shape, base structure, and quality of the pattern):

- Contraction depths (successive pullback %, in chronological order): {metrics['contraction_pcts']}
- Number of contractions detected: {metrics['contraction_count']}
- Contractions decreasing over time (tightening base): {metrics['contractions_decreasing']}
- Volume ratio, second half of base vs first half (lower = drying up): {metrics['volume_dryup_ratio']}
- Candidate pivot price (most recent swing high): {metrics['pivot_price_candidate']}
- Current price: {metrics['current_price']}
- Current price is {metrics['pct_below_pivot']}% below the pivot

Judge whether this is a genuine, tradeable VCP setup or not, using both the
numbers above and the chart image. Give a specific pivot price and stop-loss
suggestion if the pattern is valid. Be skeptical — a few pullbacks alone is
not automatically a VCP; the base should show clear tightening and the chart
should look like an actual consolidation, not a sustained downtrend."""


def analyze_chart(client, ticker, chart_path, metrics, config=CONFIG):
    """Sends the chart image + computed metrics to Claude, returns the parsed
    structured VCP verdict as a dict."""
    with open(chart_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=config["anthropic_model"],
        max_tokens=1024,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": VCP_SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                },
                {"type": "text", "text": _build_prompt(ticker, metrics)},
            ],
        }],
    )

    if response.stop_reason == "refusal":
        return {"error": "refusal", "detail": str(getattr(response, "stop_details", None))}

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        return {"error": "no_text_block", "stop_reason": response.stop_reason}

    return json.loads(text_block.text)


def run_vcp_analysis(candidates_df, chart_paths, config=CONFIG):
    """candidates_df: ranked DataFrame (top N already sliced by caller).
    chart_paths: {symbol: png_path} from stage_charts.generate_charts.
    Returns a DataFrame of VCP verdicts merged with the original ranking columns.
    """
    import anthropic
    client = anthropic.Anthropic()

    rows = []
    for _, row in candidates_df.iterrows():
        symbol = row["Symbol"]
        chart_path = chart_paths.get(symbol)
        if not chart_path or not os.path.exists(chart_path):
            logger.warning("  %s: no chart available, skipping VCP analysis", symbol)
            continue

        from .stage_charts import fetch_chart_data
        df = fetch_chart_data(symbol, config)
        metrics = compute_vcp_metrics(df, config)
        if metrics is None:
            logger.warning("  %s: not enough history for contraction analysis, skipping", symbol)
            continue

        logger.info("  Analyzing %s with %s...", symbol, config["anthropic_model"])
        try:
            verdict = analyze_chart(client, symbol, chart_path, metrics, config)
            if "error" in verdict:
                logger.warning("  %s: VCP analysis returned an error verdict: %s", symbol, verdict["error"])
            else:
                logger.info(
                    "  %s: verdict=%s stage=%s confidence=%s",
                    symbol, verdict.get("entry_recommendation"),
                    verdict.get("pattern_stage"), verdict.get("confidence"),
                )
        except Exception as e:
            logger.error("  %s: VCP analysis failed", symbol, exc_info=True)
            verdict = {"error": str(e)}

        rows.append({"Symbol": symbol, **metrics, **verdict})

    logger.info("Stage G: VCP analysis complete for %d candidates.", len(rows))
    return pd.DataFrame(rows)
