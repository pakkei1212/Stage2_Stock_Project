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


def _assess_tightening(contractions):
    """Leg-over-leg ratios (each leg's depth vs. the prior leg's) match how VCP
    theory actually describes tightening — e.g. Minervini's "2, 1, 3/4, 1/2"
    pattern where each pullback is roughly half the last — rather than
    comparing averaged halves of the depth sequence, which can pass on as few
    as one shrinking leg out of many (a single outlier leg can drag one half's
    average below the other's while most legs never actually shrank).

    Requires a majority of consecutive legs to actually shrink, not just an
    averaged trend, and the most recent leg (closest to the pivot, where it
    matters most for entry timing) to be shallower than the first.

    Returns (contraction_ratios, monotonic_fraction, is_contracting).
    """
    if len(contractions) < 2:
        return [], None, False
    ratios = [contractions[i + 1] / contractions[i] for i in range(len(contractions) - 1)]
    monotonic_fraction = sum(r < 1.0 for r in ratios) / len(ratios)
    is_contracting = monotonic_fraction >= 0.5 and contractions[-1] < contractions[0]
    return ratios, monotonic_fraction, bool(is_contracting)


def compute_vcp_metrics(df, config=CONFIG, return_details=False):
    """Computes numeric VCP signals from the trailing chart window.

    Returns a dict of metrics, or None if there isn't enough trailing history.
    If ``return_details`` is True, returns ``(metrics, details)`` (or
    ``(None, None)``), where ``details`` carries the intermediate pivots,
    contraction legs and volume halves used to derive the metrics — so a
    verification chart can be annotated from the exact same computation.
    """
    window = config["chart_lookback_days"]
    pivot_window = config["vcp_pivot_window_days"]
    base = df.iloc[-window:].dropna(subset=["Close", "Volume"])
    if len(base) < pivot_window * 4:
        return (None, None) if return_details else None

    close = base["Close"]
    pivots = _collapse_alternating(_find_pivots(close, pivot_window))

    contractions = []
    contraction_legs = []
    for (idx_h, kind_h, val_h), (idx_l, kind_l, val_l) in zip(pivots, pivots[1:]):
        if kind_h == "high" and kind_l == "low" and val_h > 0 and val_l < val_h:
            depth = (val_h - val_l) / val_h
            contractions.append(depth)
            contraction_legs.append({
                "idx_high": idx_h, "val_high": val_h,
                "idx_low": idx_l, "val_low": val_l, "depth": depth,
            })

    contraction_ratios, monotonic_fraction, is_contracting = _assess_tightening(contractions)

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

    metrics = {
        "contraction_pcts": [round(float(c) * 100, 2) for c in contractions],
        "contraction_count": len(contractions),
        "contraction_ratios": [round(float(r), 3) for r in contraction_ratios],
        "contraction_monotonicity": round(float(monotonic_fraction), 3) if monotonic_fraction is not None else None,
        "contractions_decreasing": bool(is_contracting),
        "volume_dryup_ratio": round(float(volume_dryup_ratio), 3) if pd.notna(volume_dryup_ratio) else None,
        "pivot_price_candidate": round(float(pivot_price_candidate), 2),
        "current_price": round(current_price, 2),
        "pct_below_pivot": round(float(pct_below_pivot) * 100, 2) if pd.notna(pct_below_pivot) else None,
    }

    if return_details:
        details = {
            "base": base,
            "pivots": pivots,
            "contraction_legs": contraction_legs,
            "half": half,
            "vol_first_half": float(vol_first_half) if pd.notna(vol_first_half) else None,
            "vol_second_half": float(vol_second_half) if pd.notna(vol_second_half) else None,
        }
        return metrics, details
    return metrics


def render_vcp_annotated_chart(ticker, df, config=CONFIG, out_dir=None):
    """Renders a verification chart that overlays the *computed* VCP metrics on
    the price/volume panels — detected pivot highs/lows, each contraction leg
    with its depth %, the candidate pivot price and current price, and the
    first-/second-half average volume used for the dry-up ratio — plus a text
    box of the raw metric values. Lets a human eyeball whether the numbers
    handed to Claude actually match the chart. Returns the PNG path, or None.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless — no display available in the container
    import matplotlib.pyplot as plt

    metrics, details = compute_vcp_metrics(df, config, return_details=True)
    if metrics is None:
        logger.warning("  %s: not enough history to render annotated VCP chart", ticker)
        return None

    base = details["base"]
    idx = base.index
    close = base["Close"]

    full_close = df["Close"]
    ma50 = full_close.rolling(50).mean().reindex(idx)
    ma150 = full_close.rolling(150).mean().reindex(idx)
    ma200 = full_close.rolling(200).mean().reindex(idx)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10.24, 7.68), dpi=250, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(idx, close, label="Close", linewidth=1.4, color="black")
    ax_price.plot(idx, ma50, label="MA50", linewidth=1, color="tab:orange")
    ax_price.plot(idx, ma150, label="MA150", linewidth=1, color="tab:blue")
    ax_price.plot(idx, ma200, label="MA200", linewidth=1, color="tab:red")

    # Detected pivots: red ▼ at swing highs, green ▲ at swing lows.
    for i, kind, val in details["pivots"]:
        marker, color = ("v", "tab:red") if kind == "high" else ("^", "tab:green")
        ax_price.scatter([idx[i]], [val], marker=marker, color=color, s=45, zorder=5)

    # Each contraction leg (high -> low) drawn with its depth %.
    for leg in details["contraction_legs"]:
        dh, dl = idx[leg["idx_high"]], idx[leg["idx_low"]]
        ax_price.plot([dh, dl], [leg["val_high"], leg["val_low"]],
                      color="tab:purple", linewidth=1, linestyle="--", alpha=0.75, zorder=4)
        ax_price.annotate(f"-{leg['depth'] * 100:.1f}%", xy=(dl, leg["val_low"]),
                          xytext=(0, -12), textcoords="offset points",
                          ha="center", fontsize=8, color="tab:purple")

    # Candidate pivot price (breakout trigger) and current price.
    ax_price.axhline(metrics["pivot_price_candidate"], color="tab:red", linestyle=":", linewidth=1)
    ax_price.text(idx[0], metrics["pivot_price_candidate"],
                  f" pivot {metrics['pivot_price_candidate']}",
                  va="bottom", ha="left", fontsize=8, color="tab:red")
    ax_price.axhline(metrics["current_price"], color="gray", linestyle=":", linewidth=0.8)
    ax_price.text(idx[-1], metrics["current_price"],
                  f"{metrics['current_price']} ", va="bottom", ha="right", fontsize=8, color="gray")

    # Metrics rendered as a footer below the volume panel (see after
    # tight_layout) rather than as an in-plot box, so the numbers never overlap
    # the price action they describe — in particular the pivot swing high.
    # Each variable-length list (contractions, leg ratios) gets its own line so
    # a high-contraction name can't widen the footer past the figure edge.
    metrics_text = "\n".join([
        "   |   ".join([
            f"contractions %: {metrics['contraction_pcts']}",
            f"count: {metrics['contraction_count']}",
            f"decreasing: {metrics['contractions_decreasing']}",
        ]),
        "   |   ".join([
            f"leg ratios: {metrics['contraction_ratios']}",
            f"monotonic: {metrics['contraction_monotonicity']}",
        ]),
        "   |   ".join([
            f"vol dry-up ratio: {metrics['volume_dryup_ratio']}",
            f"pivot: {metrics['pivot_price_candidate']}",
            f"current: {metrics['current_price']}",
            f"pct below pivot: {metrics['pct_below_pivot']}%",
        ]),
    ])

    ax_price.set_title(f"{ticker} — VCP metrics verification")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    colors = ["tab:green" if c >= o else "tab:red"
              for o, c in zip(base["Open"], base["Close"])]
    ax_vol.bar(idx, base["Volume"], color=colors, width=1.0)
    half = details["half"]
    if details["vol_first_half"] is not None:
        ax_vol.hlines(details["vol_first_half"], idx[0], idx[half - 1],
                      color="tab:blue", linewidth=1.4, label="1st-half avg vol")
    if details["vol_second_half"] is not None:
        ax_vol.hlines(details["vol_second_half"], idx[half], idx[-1],
                      color="tab:orange", linewidth=1.4, label="2nd-half avg vol")
    ax_vol.axvline(idx[half], color="gray", linestyle=":", linewidth=0.8)
    ax_vol.set_ylabel("Volume")
    ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.grid(alpha=0.3)

    # Reserve the bottom strip for the metrics footer so it sits clear of both
    # panels and the date axis.
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.text(0.5, 0.05, metrics_text, ha="center", va="center", fontsize=7.5,
             family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95))

    out_dir = out_dir or os.path.join(config["chart_dir"], "vcp_debug")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


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
- Leg-over-leg contraction ratios (each pullback vs. the prior one; <1.0 = shrinking): {metrics['contraction_ratios']}
- Number of contractions detected: {metrics['contraction_count']}
- Contractions decreasing over time (majority of legs shrinking, most recent leg shallower than the first): {metrics['contractions_decreasing']}
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
    # --- TEMP: token-free chart verification (API call disabled) ---
    # Re-enable the two lines below (and the analyze_chart block further down)
    # to restore live Claude VCP analysis.
    # import anthropic
    # client = anthropic.Anthropic()
    client = None
    # --- END TEMP ---

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

        if config.get("vcp_debug_charts", True):
            try:
                debug_path = render_vcp_annotated_chart(symbol, df, config)
                if debug_path:
                    logger.info("  %s: metrics-verification chart -> %s", symbol, debug_path)
            except Exception:
                logger.warning("  %s: failed to render metrics-verification chart", symbol, exc_info=True)

        # --- TEMP: token-free chart verification (API call disabled) ---
        # The live Claude call is commented out so we can inspect the
        # metrics-verification charts without spending tokens. Restore the
        # block below to re-enable it.
        logger.info("  %s: skipping Claude analysis (token-free mode) — verify chart only", symbol)
        verdict = {"skipped": "token_free_mode"}
        # logger.info("  Analyzing %s with %s...", symbol, config["anthropic_model"])
        # try:
        #     verdict = analyze_chart(client, symbol, chart_path, metrics, config)
        #     if "error" in verdict:
        #         logger.warning("  %s: VCP analysis returned an error verdict: %s", symbol, verdict["error"])
        #     else:
        #         logger.info(
        #             "  %s: verdict=%s stage=%s confidence=%s",
        #             symbol, verdict.get("entry_recommendation"),
        #             verdict.get("pattern_stage"), verdict.get("confidence"),
        #         )
        # except Exception as e:
        #     logger.error("  %s: VCP analysis failed", symbol, exc_info=True)
        #     verdict = {"error": str(e)}
        # --- END TEMP ---

        rows.append({"Symbol": symbol, **metrics, **verdict})

    logger.info("Stage G: VCP analysis complete for %d candidates.", len(rows))
    return pd.DataFrame(rows)
