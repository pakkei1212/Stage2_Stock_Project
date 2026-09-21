"""Stage G: quantitative VCP detection + Claude vision VCP entry analysis.

Metrics first, vision second:

    swing structure (Close)  ->  one coherent base  ->  High/Low contraction metrics,
    volume / final-zone / breakout metrics  ->  deterministic quality assessment
    ->  verification chart  ->  Claude confirms / interprets (numbers are ground truth)

Core VCP concept (Minervini): inside a Stage 2 uptrend, price builds ONE base
under a common resistance area through successive, shallower pullbacks
(volatility contraction), lows tend to rise, supply (volume) dries up, and the
base finishes in a tight area just below a pivot that a valid breakout clears on
expanding volume.

Everything numeric below — the 2% swing filter, the 10% resistance band, the
tightening / dry-up classifications, "tight" <= 10%, the 5% pivot proximity,
1.5x breakout volume — is OUR detector heuristic chosen on synthetic structure,
not a rule prescribed by the book. The thresholds live in CONFIG
(``vcp_*``) or in the documented constants below.

No lookahead: confirmed pivots use a symmetric window only where the window
lies inside the data; the last ``vcp_pivot_window_days`` bars use a truncated
window over data that exists (a "forming" pivot). Every metric uses bars up to
and including the last one only.
"""
import base64
import json
import logging
import os
import re

import numpy as np
import pandas as pd

from .config import CONFIG

logger = logging.getLogger(__name__)

# v2: base segmentation, High/Low depths, forming right-edge leg, quality assessment.
# v1 rows (before 2026-09) hold only the 13 original keys, computed over every swing.
VCP_METRICS_VERSION = 2

# Detector heuristics used by assess_vcp_quality (ours, not book rules).
_TIGHTENING_STRONG_LAST_FIRST = 0.50     # final depth <= half of the first, every leg shrinking
_TIGHTENING_ACCEPTABLE_LAST_FIRST = 0.75  # final depth meaningfully below the first
_VOLUME_STRONG_LAST_FIRST = 0.60
_VOLUME_ACCEPTABLE_LAST_FIRST = 0.85
_MAX_BASE_CONTRACTIONS = 6               # more swings than this under one band reads as a choppy range
_TIGHT_FINAL_CONTRACTION_PCT = 10.0      # final contraction depth regarded as "tight"
_BASELINE_SHORT, _BASELINE_LONG = 20, 50  # sessions for average-volume / range baselines

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


# ── swing structure ───────────────────────────────────────────────────────
# Pivots are tuples (index, "high"|"low", close_value, confirmed).

def _find_pivots(close, window):
    """Confirmed pivots: max/min within +/- window bars. Only bars that have
    ``window`` bars on both sides inside the data can qualify."""
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


def _find_forming_pivots(close, window):
    """Right-edge pivots in the last ``window`` bars, where a symmetric window
    would need bars that do not exist yet. Bar i qualifies against the truncated
    span [i - window, last bar]: a forming high is the highest close of that span
    (never the last bar — nothing has pulled back from it yet), a forming low the
    lowest. Uses only data up to the last bar."""
    pivots = []
    n = len(close)
    for i in range(max(window, n - window), n):
        seg = close.iloc[i - window:]
        val = close.iloc[i]
        if val == seg.max() and i < n - 1:
            pivots.append((i, "high", val))
        elif val == seg.min():
            pivots.append((i, "low", val))
    return pivots


def _collapse_alternating(pivots):
    """Collapse consecutive same-type pivots to the most extreme one, so the
    sequence strictly alternates high/low."""
    collapsed = []
    for p in pivots:
        if collapsed and collapsed[-1][1] == p[1]:
            prev = collapsed[-1]
            keep_prev = (p[1] == "high" and prev[2] >= p[2]) or (p[1] == "low" and prev[2] <= p[2])
            if keep_prev:
                continue
            collapsed[-1] = p
        else:
            collapsed.append(p)
    return collapsed


def _prune_small_swings(pivots, min_pct):
    """Repeatedly drop the smallest swing below ``min_pct`` (removing its later
    pivot and re-collapsing, which keeps the more extreme neighbour), so noise
    wiggles never become 'contractions'."""
    piv = list(pivots)
    while len(piv) >= 2:
        moves = [abs(b[2] - a[2]) / max(a[2], b[2]) * 100 for a, b in zip(piv, piv[1:])]
        j = int(np.argmin(moves))
        if moves[j] >= min_pct:
            break
        del piv[j + 1]
        piv = _collapse_alternating(piv)
    return piv


def _measure_leg(base, ih, il):
    """Chronological High -> later Low measurement of one structural leg.

    ``ih`` / ``il`` are the Close-based swing high / swing low indices.

    1. Measured High: the highest High from the bar before the swing high up to the
       bar before the swing low (earliest bar on ties).
    2. Contraction Low: the lowest Low strictly AFTER both the measured-High bar and
       the swing-high bar, up to the bar after the swing low (clipped to the last
       bar, so a forming leg never looks forward). The pivot bar's own intraday
       range, and any Low earlier than the measured High, can never be the low.
    3. Pullback volume: mean volume over the bars after that same start point
       through the measured Low, inclusive. The high/pivot bar (often a breakout or
       gap day) is not selling evidence and is excluded.

    Returns None (no contraction is invented) if no later bar exists or the later
    Low is not below the High.
    """
    n = len(base)
    highs = base["High"].to_numpy(dtype=float)
    lows = base["Low"].to_numpy(dtype=float)
    volume = base["Volume"].to_numpy(dtype=float)
    hi_start, hi_end = max(ih - 1, 0), il                   # [hi_start, hi_end)
    if hi_end <= hi_start:
        return None
    mh = hi_start + int(np.argmax(highs[hi_start:hi_end]))
    lo_start, lo_end = max(mh, ih) + 1, min(il + 1, n - 1) + 1
    if lo_end <= lo_start:
        return None
    ml = lo_start + int(np.argmin(lows[lo_start:lo_end]))
    hi, lo = float(highs[mh]), float(lows[ml])
    if hi <= 0 or lo >= hi:
        return None
    dates = base.index
    return {
        "idx_high": ih, "idx_low": il,
        "measured_high_idx": mh, "measured_high_date": str(pd.Timestamp(dates[mh]).date()), "measured_high_price": hi,
        "measured_low_idx": ml, "measured_low_date": str(pd.Timestamp(dates[ml]).date()), "measured_low_price": lo,
        "high_price": hi, "low_price": lo, "depth": (hi - lo) / hi, "duration": il - ih,
        "volume_window": (lo_start, ml), "vol_avg": float(volume[lo_start:ml + 1].mean()),
    }


def detect_vcp_structure(base, config=CONFIG):
    """Swing structure -> contraction legs -> the coherent base.

    Structure is detected on Close (robust to single-bar wicks); each leg is then
    measured on actual price excursion, strictly in time order (``_measure_leg``):
    first the measured High, then the contraction Low searched only in LATER bars,
    then the pullback volume over the bars after the High through the Low.

    Base segmentation: starting from the most recent leg, earlier legs are added
    while the highs of all included legs stay within ``vcp_resistance_band_pct``
    of the base's highest high. The first leg whose high breaks that band (an
    earlier, lower step of an advancing trend, or a much higher prior peak) ends
    the base; it and everything before it are ignored.
    """
    window = config["vcp_pivot_window_days"]
    close = base["Close"]
    confirmed = [(i, k, float(v), True) for i, k, v in _find_pivots(close, window)]
    forming = [(i, k, float(v), False) for i, k, v in _find_forming_pivots(close, window)]
    pivots = _prune_small_swings(_collapse_alternating(confirmed + forming), config["vcp_min_swing_pct"])

    legs = []
    for a, b in zip(pivots, pivots[1:]):
        if a[1] != "high" or b[1] != "low":
            continue
        leg = _measure_leg(base, a[0], b[0])
        if leg is None:
            continue
        leg.update({"val_high": a[2], "val_low": b[2], "high_confirmed": a[3], "low_confirmed": b[3],
                    "confirmed": bool(a[3] and b[3])})
        legs.append(leg)

    band = config["vcp_resistance_band_pct"] / 100
    start = len(legs)
    for i in range(len(legs) - 1, -1, -1):
        leg_highs = [leg["high_price"] for leg in legs[i:]]
        if (max(leg_highs) - min(leg_highs)) / max(leg_highs) > band:
            break
        start = i
    return {"pivots": pivots, "legs": legs, "base_start": start,
            "base_legs": legs[start:], "ignored_legs": legs[:start]}


# ── metrics ───────────────────────────────────────────────────────────────

def _r(value, digits=2):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def _mean_before(values, end, length):
    """Mean of the ``length`` values ending just before index ``end``; None if short."""
    start = end - length
    if start < 0 or end > len(values):
        return None
    return float(np.mean(values[start:end]))


def _leg_ratios(values):
    return [values[i + 1] / values[i] for i in range(len(values) - 1) if values[i]]


def compute_vcp_metrics(df, config=CONFIG, return_details=False):
    """Numeric VCP metrics as of the last bar of ``df``.

    Returns a dict (raw metrics + the deterministic assessment from
    ``assess_vcp_quality``), or None without enough history. With
    ``return_details`` returns ``(metrics, details)`` where details carry the
    pivots, all legs, the selected base and the final-zone span, so the
    verification chart is drawn from the exact same computation.

    Backward compatibility: the 13 v1 keys are kept. Their meaning is now
    base-specific — ``contraction_pcts`` / ``volume_leg_avgs`` cover the selected
    base only with High/Low depths, ``contractions_decreasing`` is True only for
    strong/acceptable tightening, and ``pivot_price_candidate`` is the base's
    right-side resistance.
    """
    window = config["vcp_pivot_window_days"]
    full = df.dropna(subset=["High", "Low", "Close", "Volume"])
    base = full.iloc[-config["chart_lookback_days"]:]
    if len(base) < window * 4:
        return (None, None) if return_details else None

    structure = detect_vcp_structure(base, config)
    legs = structure["base_legs"]
    off = len(full) - len(base)                      # base index i == full index off + i
    last = len(full) - 1
    close_f = full["Close"].to_numpy(dtype=float)
    high_f = full["High"].to_numpy(dtype=float)
    low_f = full["Low"].to_numpy(dtype=float)
    vol_f = full["Volume"].to_numpy(dtype=float)
    current_price = float(close_f[-1])

    # price contraction
    depths = [leg["depth"] for leg in legs]
    ratios = _leg_ratios(depths)
    shrink_fraction = sum(r < 1 for r in ratios) / len(ratios) if ratios else None
    leg_highs = [leg["high_price"] for leg in legs]
    leg_lows = [leg["low_price"] for leg in legs]
    higher_low_flags = [bool(leg_lows[i + 1] > leg_lows[i]) for i in range(len(leg_lows) - 1)]

    recovery_ratios = []
    for k, leg in enumerate(legs):
        if k + 1 < len(legs):
            recovery_high = legs[k + 1]["high_price"]
        elif leg["measured_low_idx"] < len(base) - 1:
            # only bars after the measured Low can be the recovery from it
            recovery_high = float(base["High"].iloc[leg["measured_low_idx"] + 1:].max())
        else:
            recovery_high = None                     # the final low is the last bar
        drop = leg["high_price"] - leg["low_price"]
        recovery_ratios.append(None if recovery_high is None else (recovery_high - leg["low_price"]) / drop)

    # candidate pivot: right-side resistance of this base
    if legs:
        pivot = max(leg_highs[1:] or leg_highs[:1])
    else:
        pivot = float(base["High"].max())
    pct_from_pivot = (current_price - pivot) / pivot * 100 if pivot else None

    breakout_idx = None
    if legs:
        j0 = off + legs[-1]["idx_low"] + 1
        above = np.nonzero(close_f[j0:] > pivot)[0]
        breakout_idx = j0 + int(above[0]) if len(above) else None

    # volume across contractions
    vol_avgs = [leg["vol_avg"] for leg in legs]
    vol_ratios = _leg_ratios(vol_avgs)
    vol_pairs = [(i, j) for i in range(len(vol_avgs)) for j in range(i + 1, len(vol_avgs))]
    vol_monotonicity = (sum(vol_avgs[j] < vol_avgs[i] for i, j in vol_pairs) / len(vol_pairs)) if vol_pairs else None

    # final tight zone: the last N sessions, ending the bar BEFORE a breakout so the
    # breakout-volume spike never dilutes the pre-breakout dry-up reading
    zone_days = config["vcp_final_zone_days"]
    zone_end = breakout_idx - 1 if breakout_idx is not None else last
    zone_start = max(zone_end - zone_days + 1, 0)
    zone_vol = float(np.mean(vol_f[zone_start:zone_end + 1]))
    zone_base20 = _mean_before(vol_f, zone_start, _BASELINE_SHORT)
    zone_base50 = _mean_before(vol_f, zone_start, _BASELINE_LONG)
    prev_close = np.r_[close_f[0], close_f[:-1]]
    true_range = np.maximum(high_f - low_f, np.maximum(abs(high_f - prev_close), abs(low_f - prev_close)))
    zone_tr = float(np.median(true_range[zone_start:zone_end + 1]))
    prior_tr = (float(np.median(true_range[zone_start - _BASELINE_SHORT:zone_start]))
                if zone_start >= _BASELINE_SHORT else None)
    zone_high = float(high_f[zone_start:zone_end + 1].max())
    zone_range_pct = (zone_high - float(low_f[zone_start:zone_end + 1].min())) / zone_high * 100

    # breakout demand (averages exclude the bar being measured)
    avg20 = _mean_before(vol_f, last, _BASELINE_SHORT)
    avg50 = _mean_before(vol_f, last, _BASELINE_LONG)
    breakout_avg20 = _mean_before(vol_f, breakout_idx, _BASELINE_SHORT) if breakout_idx is not None else None

    final = legs[-1] if legs else None
    final_leg_base50 = _mean_before(vol_f, off + final["idx_high"], _BASELINE_LONG) if final else None

    # Extension from the short/intermediate MAs — how far price has run away from
    # its moving averages (EMA10/21 per Minervini's convention, 50-day SMA).
    # Computed off the full df so the 50-day window has enough history.
    full_close = df["Close"]

    def _pct_ext_ema(span):
        ema = full_close.ewm(span=span, adjust=False).mean().iloc[-1]
        return (current_price - ema) / ema * 100 if pd.notna(ema) and ema else None

    def _pct_ext_sma(span):
        ma = full_close.rolling(span).mean().iloc[-1]
        return (current_price - ma) / ma * 100 if pd.notna(ma) and ma else None

    def _ratio(a, b):
        return a / b if a is not None and b else None

    metrics = {
        # v1 keys (base-specific since v2)
        "contraction_pcts": [_r(d * 100) for d in depths],
        "contraction_count": len(legs),
        "contraction_ratios": [_r(r, 3) for r in ratios],
        "contraction_monotonicity": _r(shrink_fraction, 3),
        "contractions_decreasing": False,            # set by assess_vcp_quality
        "volume_dryup_ratio": _r(vol_avgs[-1] / vol_avgs[0], 3) if len(vol_avgs) >= 2 and vol_avgs[0] else None,
        "volume_leg_avgs": [round(v) for v in vol_avgs],
        "pivot_price_candidate": _r(pivot),
        "current_price": _r(current_price),
        "pct_below_pivot": _r(-pct_from_pivot) if pct_from_pivot is not None else None,
        "pct_ext_ema10": _r(_pct_ext_ema(10)),
        "pct_ext_ema21": _r(_pct_ext_ema(21)),
        "pct_ext_ma50": _r(_pct_ext_sma(50)),
        # v2: base
        "vcp_metrics_version": VCP_METRICS_VERSION,
        "base_start": str(base.index[legs[0]["idx_high"]].date()) if legs else None,
        "base_end": str(base.index[-1].date()),
        "base_duration_sessions": (len(base) - 1 - legs[0]["idx_high"]) if legs else None,
        "ignored_contraction_count": len(structure["ignored_legs"]),
        "contraction_confirmed": [leg["confirmed"] for leg in legs],
        "contraction_duration_sessions": [leg["duration"] for leg in legs],
        "contraction_high_prices": [_r(h) for h in leg_highs],
        "contraction_low_prices": [_r(lo) for lo in leg_lows],
        # v2: tightening
        "contraction_shrink_fraction": _r(shrink_fraction, 3),
        "largest_contraction_expansion_ratio": _r(max(ratios), 3) if ratios else None,
        "first_to_last_depth_ratio": _r(depths[-1] / depths[0], 3) if len(depths) >= 2 else None,
        "higher_low_flags": higher_low_flags,
        "higher_low_fraction": _r(sum(higher_low_flags) / len(higher_low_flags), 3) if higher_low_flags else None,
        "recovery_ratios": [_r(x, 3) for x in recovery_ratios],
        "resistance_dispersion_pct": _r((max(leg_highs) - min(leg_highs)) / max(leg_highs) * 100) if legs else None,
        # v2: final contraction / tight zone
        "final_contraction_pct": _r(depths[-1] * 100) if legs else None,
        "final_contraction_duration": final["duration"] if final else None,
        "final_contraction_confirmed": final["confirmed"] if final else None,
        "final_contraction_volume_ratio": _r(_ratio(final["vol_avg"], final_leg_base50), 3) if final else None,
        "final_zone_days": zone_days,
        "final_zone_pre_breakout": breakout_idx is not None,
        "final_zone_volume_ratio_20d": _r(_ratio(zone_vol, zone_base20), 3),
        "final_zone_volume_ratio_50d": _r(_ratio(zone_vol, zone_base50), 3),
        "final_zone_range_pct": _r(zone_range_pct),
        "final_range_compression_ratio": _r(_ratio(zone_tr, prior_tr), 3),
        # v2: volume across contractions
        "volume_contraction_ratios": [_r(r, 3) for r in vol_ratios],
        "volume_shrink_fraction": _r(sum(r < 1 for r in vol_ratios) / len(vol_ratios), 3) if vol_ratios else None,
        "volume_monotonicity": _r(vol_monotonicity, 3),
        "largest_volume_expansion_ratio": _r(max(vol_ratios), 3) if vol_ratios else None,
        # v2: pivot / breakout
        "pct_from_pivot": _r(pct_from_pivot),
        "sessions_since_breakout": (last - breakout_idx) if breakout_idx is not None else None,
        "current_volume": round(float(vol_f[-1])),
        "avg_volume_20d": round(avg20) if avg20 is not None else None,
        "avg_volume_50d": round(avg50) if avg50 is not None else None,
        "current_volume_vs_20d": _r(_ratio(float(vol_f[-1]), avg20), 3),
        "current_volume_vs_50d": _r(_ratio(float(vol_f[-1]), avg50), 3),
        "breakout_volume_vs_20d": (_r(_ratio(float(vol_f[breakout_idx]), breakout_avg20), 3)
                                   if breakout_idx is not None else None),
    }
    metrics.update(assess_vcp_quality(metrics, config))

    if return_details:
        details = {
            "base": base,
            "pivots": structure["pivots"],
            "contraction_legs": legs,                       # selected base legs (each with "vol_avg")
            "ignored_legs": structure["ignored_legs"],
            "base_start_idx": legs[0]["idx_high"] if legs else None,
            "zone_start_idx": zone_start - off, "zone_end_idx": zone_end - off,
            "zone_avg_volume": zone_vol, "avg_volume_50d": avg50,
            "breakout_idx": (breakout_idx - off) if breakout_idx is not None else None,
        }
        return metrics, details
    return metrics


# ── deterministic assessment ──────────────────────────────────────────────

def _classify_sequence(values, max_expansion, strong_last_first, acceptable_last_first, none_label):
    """strong | acceptable | weak | <none_label> | insufficient for a chronological
    sequence that should shrink (contraction depths, or volume per contraction)."""
    if len(values) < 2 or any(v is None for v in values) or not values[0]:
        return "insufficient"
    ratios = _leg_ratios(values)
    last_first = values[-1] / values[0]
    violations = sum(r >= 1 for r in ratios)
    if last_first >= 1 or (1 - violations / len(ratios)) < 0.5:
        return none_label
    if violations == 0 and last_first <= strong_last_first:
        return "strong"
    if violations <= 1 and max(ratios) <= max_expansion and last_first <= acceptable_last_first:
        return "acceptable"
    return "weak"


def _tightening_quality(contraction_pcts, config=CONFIG):
    return _classify_sequence(contraction_pcts, config["vcp_max_expansion_ratio"],
                              _TIGHTENING_STRONG_LAST_FIRST, _TIGHTENING_ACCEPTABLE_LAST_FIRST, "not_contracting")


def _assess_tightening(contractions, config=CONFIG):
    """Back-compat helper: (leg ratios, shrink fraction, is_contracting) for a depth
    sequence. is_contracting is True only for strong/acceptable tightening."""
    if len(contractions) < 2:
        return [], None, False
    ratios = _leg_ratios(contractions)
    shrink = sum(r < 1.0 for r in ratios) / len(ratios)
    return ratios, shrink, _tightening_quality(contractions, config) in ("strong", "acceptable")


def assess_vcp_quality(m, config=CONFIG):
    """Explainable, rule-based reading of the raw metrics. Each component is
    reported separately; ``vcp_numeric_quality`` combines them. Detector
    heuristics only — never a buy signal and never generated by Claude."""
    max_exp = config["vcp_max_expansion_ratio"]
    prox = config["vcp_pivot_proximity_pct"]
    notes = []
    count = m["contraction_count"]

    tightening = _tightening_quality(m["contraction_pcts"], config)
    volume = _classify_sequence(m["volume_leg_avgs"], max_exp, _VOLUME_STRONG_LAST_FIRST,
                                _VOLUME_ACCEPTABLE_LAST_FIRST, "none")

    if count < 2:
        coherence = "insufficient"
    elif count > _MAX_BASE_CONTRACTIONS:
        coherence = "weak"
        notes.append(f"{count} swings under one band (choppy range)")
    elif m["higher_low_fraction"] == 1:
        coherence = "strong"
    elif m["higher_low_fraction"] >= 0.5:
        coherence = "acceptable"
    else:
        coherence = "weak"

    # Tightness is judged on the final contraction's depth; candle-range compression
    # is reported (and noted when ranges are not shrinking) but not used as a gate,
    # because a few wide bars inside a shallow pullback say little about the base.
    fc, rc = m["final_contraction_pct"], m["final_range_compression_ratio"]
    if fc is None:
        final_tightness = "unknown"
    elif fc <= _TIGHT_FINAL_CONTRACTION_PCT:
        final_tightness = "tight"
    else:
        final_tightness = "loose"

    p, since = m["pct_from_pivot"], m["sessions_since_breakout"]
    if p is None:
        pivot_state = "unknown"
    elif p < -prox:
        pivot_state = "far_below_pivot"
    elif p <= 0:
        pivot_state = "coiling_below_pivot"
    elif p <= prox:
        pivot_state = "fresh_breakout" if since is not None and since < config["vcp_final_zone_days"] else "above_pivot"
    else:
        pivot_state = "extended_above_pivot"

    breakout_volume_confirmed = None
    if pivot_state == "fresh_breakout" and m["breakout_volume_vs_20d"] is not None:
        breakout_volume_confirmed = m["breakout_volume_vs_20d"] >= config["vcp_breakout_volume_ratio"]

    good = ("strong", "acceptable")
    if count < 2 or tightening in ("not_contracting", "insufficient"):
        overall = "absent"
    elif (tightening == "strong" and coherence in good and volume in good and final_tightness == "tight"
          and pivot_state in ("coiling_below_pivot", "fresh_breakout") and breakout_volume_confirmed is not False):
        overall = "strong"
    elif tightening in good and coherence in good and volume != "none":
        overall = "acceptable"
    else:
        overall = "weak"

    if count < 2:
        notes.append(f"{count} contraction(s) in the base: no sequence to judge")
    if tightening in ("weak", "not_contracting", "acceptable"):
        notes.append(f"price tightening {tightening}")
    if m["largest_contraction_expansion_ratio"] and m["largest_contraction_expansion_ratio"] > max_exp:
        notes.append(f"a later contraction expanded {m['largest_contraction_expansion_ratio']}x")
    if coherence in ("weak", "acceptable") and m["higher_low_fraction"] is not None and count <= _MAX_BASE_CONTRACTIONS:
        notes.append(f"higher lows {m['higher_low_fraction']:.0%}")
    if volume in ("none", "weak"):
        notes.append(f"volume dry-up {volume}")
    if m["largest_volume_expansion_ratio"] and m["largest_volume_expansion_ratio"] > max_exp:
        notes.append(f"contraction volume expanded {m['largest_volume_expansion_ratio']}x")
    if final_tightness == "loose":
        notes.append(f"final contraction {fc}% is not tight")
    if rc is not None and rc > 1.0:
        notes.append(f"final-zone candle ranges not compressing ({rc}x prior 20 sessions)")
    if m["final_contraction_confirmed"] is False:
        notes.append("final contraction still forming at the right edge")
    if pivot_state == "far_below_pivot":
        notes.append(f"price {abs(p)}% below pivot")
    elif pivot_state == "extended_above_pivot":
        notes.append(f"price {p}% above pivot")
    if breakout_volume_confirmed is False:
        notes.append(f"breakout volume {m['breakout_volume_vs_20d']}x 20d avg")
    if m["ignored_contraction_count"]:
        notes.append(f"{m['ignored_contraction_count']} earlier swing(s) outside the base ignored")

    return {
        "contractions_decreasing": tightening in good,
        "tightening_quality": tightening,
        "volume_dryup_quality": volume,
        "base_coherence": coherence,
        "final_tightness": final_tightness,
        "pivot_state": pivot_state,
        "breakout_volume_confirmed": breakout_volume_confirmed,
        "vcp_numeric_quality": overall,
        "vcp_quality_notes": notes,
    }


# ── verification chart ────────────────────────────────────────────────────

def render_vcp_annotated_chart(ticker, df, config=CONFIG, out_dir=None):
    """Verification chart drawn from the exact computation handed to Claude, to
    answer "did the detector pick the right base?": shaded base window, base
    pivots (filled) vs ignored pivots (grey) vs forming right-edge pivots
    (hollow), each base contraction with its High/Low depth, the candidate
    pivot, per-contraction and final-zone average volume vs the 50-day average,
    plus a metrics footer. Returns the PNG path, or None.
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
    legs = details["contraction_legs"]

    full_close = df["Close"]
    ema10 = full_close.ewm(span=10, adjust=False).mean().reindex(idx)
    ema21 = full_close.ewm(span=21, adjust=False).mean().reindex(idx)
    ma50 = full_close.rolling(50).mean().reindex(idx)
    ma150 = full_close.rolling(150).mean().reindex(idx)
    ma200 = full_close.rolling(200).mean().reindex(idx)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(10.24, 7.68), dpi=250, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    if details["base_start_idx"] is not None:
        for ax in (ax_price, ax_vol):
            ax.axvspan(idx[details["base_start_idx"]], idx[-1], color="gold", alpha=0.12, zorder=0,
                       label="selected base" if ax is ax_price else None)
    ax_price.vlines(idx, base["Low"], base["High"], color="lightgray", linewidth=0.8, zorder=1)
    ax_price.plot(idx, close, label="Close", linewidth=1.3, color="black", zorder=2)
    ax_price.plot(idx, ema10, label="EMA10", linewidth=0.9, color="tab:gray", linestyle="--")
    ax_price.plot(idx, ema21, label="EMA21", linewidth=0.9, color="tab:brown", linestyle=":")
    ax_price.plot(idx, ma50, label="MA50", linewidth=1, color="tab:orange")
    ax_price.plot(idx, ma150, label="MA150", linewidth=1, color="tab:blue")
    ax_price.plot(idx, ma200, label="MA200", linewidth=1, color="tab:red")

    base_pivot_idx = {leg["idx_high"] for leg in legs} | {leg["idx_low"] for leg in legs}
    for i, kind, val, confirmed in details["pivots"]:
        marker = "v" if kind == "high" else "^"
        if i not in base_pivot_idx:
            ax_price.scatter([idx[i]], [val], marker=marker, color="silver", s=30, zorder=5)
        elif confirmed:
            ax_price.scatter([idx[i]], [val], marker=marker, s=50, zorder=6,
                             color="tab:red" if kind == "high" else "tab:green")
        else:
            ax_price.scatter([idx[i]], [val], marker=marker, s=60, zorder=6, facecolors="none",
                             edgecolors="tab:orange", linewidths=1.5)

    # Each contraction drawn from its MEASURED High bar to its MEASURED (later) Low bar,
    # with small markers on the exact High/Low prices, so the chronology is auditable.
    for n, leg in enumerate(legs, start=1):
        dh, dl = idx[leg["measured_high_idx"]], idx[leg["measured_low_idx"]]
        style = "-" if leg["confirmed"] else "--"
        color = "tab:purple" if leg["confirmed"] else "tab:orange"
        ax_price.plot([dh, dl], [leg["high_price"], leg["low_price"]], color=color, linewidth=1.2,
                      linestyle=style, alpha=0.85, zorder=4)
        ax_price.scatter([dh, dl], [leg["high_price"], leg["low_price"]], marker="o", s=14, zorder=7,
                         color=color, label="measured High -> later Low" if n == 1 else None)
        label = f"T{n} -{leg['depth'] * 100:.1f}%" + ("" if leg["confirmed"] else " forming")
        ax_price.annotate(label, xy=(dl, leg["low_price"]), xytext=(0, -12), textcoords="offset points",
                          ha="center", fontsize=7.5, color=color)

    pivot = metrics["pivot_price_candidate"]
    x0 = idx[details["base_start_idx"]] if details["base_start_idx"] is not None else idx[0]
    ax_price.hlines(pivot, x0, idx[-1], color="tab:red", linestyle=":", linewidth=1.1)
    ax_price.text(x0, pivot, f" candidate pivot {pivot}", va="bottom", ha="left", fontsize=8, color="tab:red")
    ax_price.axhline(metrics["current_price"], color="gray", linestyle=":", linewidth=0.8)
    ax_price.text(idx[-1], metrics["current_price"], f"{metrics['current_price']} ",
                  va="bottom", ha="right", fontsize=8, color="gray")

    ax_price.set_title(f"{ticker} — VCP metrics verification "
                       f"({metrics['vcp_numeric_quality']}, {metrics['pivot_state']})")
    ax_price.legend(loc="upper left", fontsize=7)
    ax_price.grid(alpha=0.3)

    colors = ["tab:green" if c >= o else "tab:red" for o, c in zip(base["Open"], base["Close"])]
    ax_vol.bar(idx, base["Volume"], color=colors, width=1.0, alpha=0.7)
    for n, leg in enumerate(legs):
        v0, v1 = leg["volume_window"]                      # bars after the High through the Low
        ax_vol.hlines(leg["vol_avg"], idx[v0], idx[v1], color="tab:purple", linewidth=2.2, zorder=5,
                      label="avg vol per contraction (excl. high bar)" if n == 0 else None)
    zs, ze = details["zone_start_idx"], details["zone_end_idx"]
    if 0 <= zs <= ze < len(idx):
        ax_vol.hlines(details["zone_avg_volume"], idx[zs], idx[ze], color="tab:blue", linewidth=2.6, zorder=6,
                      label=f"final zone avg ({metrics['final_zone_volume_ratio_50d']}x 50d)")
    if details["avg_volume_50d"]:
        ax_vol.axhline(details["avg_volume_50d"], color="black", linestyle="--", linewidth=0.8, label="50d avg")
    ax_vol.set_ylabel("Volume")
    ax_vol.legend(loc="upper left", fontsize=7)
    ax_vol.grid(alpha=0.3)

    m = metrics
    metrics_text = "\n".join([
        f"base {m['base_start']}..{m['base_end']} ({m['base_duration_sessions']} sessions, "
        f"{m['ignored_contraction_count']} earlier swings ignored)   |   contractions %: {m['contraction_pcts']}   "
        f"confirmed: {m['contraction_confirmed']}",
        f"leg ratios {m['contraction_ratios']} -> tightening {m['tightening_quality']}   |   lows "
        f"{m['contraction_low_prices']} higher-low {m['higher_low_fraction']}   |   resistance dispersion "
        f"{m['resistance_dispersion_pct']}% -> coherence {m['base_coherence']}",
        f"vol per leg {m['volume_leg_avgs']} ratios {m['volume_contraction_ratios']} -> dry-up "
        f"{m['volume_dryup_quality']}   |   final zone vol {m['final_zone_volume_ratio_20d']}x 20d / "
        f"{m['final_zone_volume_ratio_50d']}x 50d, range compression {m['final_range_compression_ratio']}",
        f"pivot {m['pivot_price_candidate']} current {m['current_price']} ({m['pct_from_pivot']:+}%) "
        f"{m['pivot_state']}   |   vol today {m['current_volume_vs_20d']}x 20d, breakout vol "
        f"{m['breakout_volume_vs_20d']}   |   ext EMA10 {m['pct_ext_ema10']}% EMA21 {m['pct_ext_ema21']}% "
        f"MA50 {m['pct_ext_ma50']}%",
    ]) if m["pct_from_pivot"] is not None else ""

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.text(0.5, 0.055, metrics_text, ha="center", va="center", fontsize=6.3, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95))

    out_dir = out_dir or os.path.join(config["chart_dir"], "vcp_debug")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Claude review routing ─────────────────────────────────────────────────
# The numeric detector is the first-stage classifier; Claude is the visual reviewer
# for ambiguous cases. These routing heuristics are ours (not book rules) and never
# change the numeric result.

LLM_REVIEW_MODES = ("off", "selective", "all")
NUMERIC_POSITIVE = ("strong", "acceptable")
_REVIEW_WICK_PCT = 3.0    # a contraction High/Low set by an intraday wick this far beyond the candle body
_REVIEW_GAP_PCT = 5.0     # a session gap (open vs prior close) this large inside the selected base


def llm_review_mode(config=CONFIG):
    """off | selective | all. ``vcp_live_analysis`` (VCP_LIVE_ANALYSIS=0) stays the
    master token-free switch and forces off."""
    if not config.get("vcp_live_analysis", True):
        return "off"
    mode = str(config.get("vcp_llm_review_mode") or "selective").strip().lower()
    if mode not in LLM_REVIEW_MODES:
        logger.warning("VCP_LLM_REVIEW_MODE=%r is not one of %s; using selective.", mode, ", ".join(LLM_REVIEW_MODES))
        return "selective"
    return mode


def vcp_agreement(numeric_quality, is_vcp_pattern):
    """Numeric positive = strong/acceptable; Claude positive = is_vcp_pattern.
    Returns agree_positive | agree_negative | numeric_positive_claude_negative |
    numeric_negative_claude_positive, or None when either side is missing."""
    if not numeric_quality or is_vcp_pattern is None:
        return None
    numeric, claude = numeric_quality in NUMERIC_POSITIVE, bool(is_vcp_pattern)
    if numeric == claude:
        return "agree_positive" if numeric else "agree_negative"
    return "numeric_positive_claude_negative" if numeric else "numeric_negative_claude_positive"


def structure_warnings(details):
    """Visually unusual structure the numeric measurement cannot judge: contraction
    extremes set by long intraday wicks, and large session gaps inside the base."""
    if not details or not details.get("contraction_legs"):
        return []
    base = details["base"]
    o, h, low, c = (base[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    warnings = []
    for n, leg in enumerate(details["contraction_legs"], start=1):
        i = leg["measured_high_idx"]
        body_top = max(o[i], c[i])
        if body_top and (h[i] - body_top) / body_top * 100 >= _REVIEW_WICK_PCT:
            warnings.append(f"T{n} High set by a {(h[i] - body_top) / body_top * 100:.1f}% intraday wick")
        j = leg["measured_low_idx"]
        body_bottom = min(o[j], c[j])
        if body_bottom and (body_bottom - low[j]) / body_bottom * 100 >= _REVIEW_WICK_PCT:
            warnings.append(f"T{n} Low set by a {(body_bottom - low[j]) / body_bottom * 100:.1f}% intraday wick")
    start = details.get("base_start_idx")
    if start is not None:
        gaps = [((o[k] - c[k - 1]) / c[k - 1] * 100, k) for k in range(max(start, 1), len(c)) if c[k - 1]]
        if gaps:
            gap, k = max(gaps, key=lambda g: abs(g[0]))
            if abs(gap) >= _REVIEW_GAP_PCT:
                warnings.append(f"{gap:+.1f}% session gap on {pd.Timestamp(base.index[k]).date()} inside the base")
    return warnings


def llm_review_decision(metrics, details, config=CONFIG):
    """(requested, reasons). Selective mode sends Claude only the cases where visual
    judgment can change the reading:

    - numeric quality acceptable or weak (the grey zone);
    - a strong setup at a fresh breakout (fresh vs. extended is a visual call);
    - a strong setup whose base has an intraday-wick extreme or a large gap (a
      wick or gap can prop up a false positive, or set a questionable pivot).

    A strong, cleanly coiling setup and an absent result stand on the numbers: a
    forming final leg alone does not trigger review, and wicks/gaps in an absent
    (clearly non-contracting / insufficient) structure cannot make it a VCP. For
    acceptable/weak candidates the structure warnings are passed as context.
    ``all`` reviews everything and still records the selective reasons for later
    comparison.
    """
    mode = llm_review_mode(config)
    quality = metrics.get("vcp_numeric_quality")
    reasons = []
    if quality in ("acceptable", "weak"):
        reasons.append(f"numeric quality {quality}")
        if metrics.get("final_contraction_confirmed") is False:
            reasons.append("final contraction still forming")
        if metrics.get("pivot_state") in ("fresh_breakout", "extended_above_pivot"):
            reasons.append(f"pivot state {metrics['pivot_state']}")
    if quality == "strong" and metrics.get("pivot_state") in ("fresh_breakout", "above_pivot", "extended_above_pivot"):
        reasons.append(f"strong setup at {metrics['pivot_state']}")
    if quality in ("strong", "acceptable", "weak"):
        reasons += structure_warnings(details)
    if mode == "off":
        return False, reasons
    if mode == "all":
        return True, reasons or ["full review mode (numeric result is clear)"]
    return bool(reasons), reasons


# ── Claude prompt ─────────────────────────────────────────────────────────

def _contraction_lines(legs):
    """One line per base contraction with its measured dates, so Claude can find
    each leg on the (unannotated) chart."""
    lines = []
    for n, leg in enumerate(legs or [], start=1):
        lines.append(f"  T{n}: {leg['measured_high_date']} High {leg['measured_high_price']:.2f} -> "
                     f"{leg['measured_low_date']} Low {leg['measured_low_price']:.2f} "
                     f"(-{leg['depth'] * 100:.2f}%, {'confirmed' if leg['confirmed'] else 'FORMING'})")
    return "\n".join(lines)


# Parses the structure_warnings() wording above; keep the two in step.
_WICK_REASON = re.compile(r"^T(\d+) (High|Low) set by a [\d.]+% intraday wick$")
_GAP_REASON = re.compile(r"session gap on \S+ inside the base$")


def _routed_structure_concerns(review_reasons):
    """(wick_reasons as (leg_no, side, text), gap_reasons) among the routing reasons."""
    wicks, gaps = [], []
    for reason in review_reasons or []:
        w = _WICK_REASON.match(reason)
        if w:
            wicks.append((int(w.group(1)), w.group(2), reason))
        elif _GAP_REASON.search(reason):
            gaps.append(reason)
    return wicks, gaps


def _structure_review_block(metrics, legs, review_reasons):
    """Mandatory rationale checks for the wick/gap warnings that caused routing.
    Visual review only: the detector's pivot and metrics stay authoritative."""
    wicks, gaps = _routed_structure_concerns(review_reasons)
    if not wicks and not gaps:
        return ""
    pivot = metrics.get("pivot_price_candidate")
    lines = ["Routed structure concerns (MANDATORY: your rationale must address each one explicitly; the",
             "detector's numbers stay as supplied and pivot_price may remain the candidate pivot):"]
    for leg_no, side, reason in wicks:
        leg = legs[leg_no - 1] if legs and 0 < leg_no <= len(legs) else None
        sets_pivot = (side == "High" and leg is not None and pivot is not None
                      and round(leg["measured_high_price"], 2) == round(pivot, 2))
        if sets_pivot:
            lines.append(
                f"- {reason}, and that wick high IS the candidate pivot ({pivot}). Include a sentence starting\n"
                f"  \"Wick/pivot review:\" that compares the wick high(s) near the pivot with the nearby candle-body\n"
                f"  highs, closes and any repeated resistance, and states whether the candidate pivot is visually\n"
                f"  well supported, questionable because of a wick, or better read as a higher level above a\n"
                f"  body-based consolidation. Do not accept the pivot merely because it was supplied.")
        elif side == "High":
            lines.append(
                f"- {reason}. Include a sentence starting \"Wick/pivot review:\" stating whether the candle bodies\n"
                f"  support that leg's High (and so the base's resistance) or whether the wick overstates it.")
        else:
            lines.append(
                f"- {reason}. Include a sentence starting \"Wick/pivot review:\" stating whether the candle bodies\n"
                f"  support that leg's Low or whether the wick exaggerates the contraction depth.")
    for reason in gaps:
        lines.append(
            f"- {reason}. Include a sentence starting \"Gap review:\" stating whether the gap creates a distorted\n"
            f"  contraction leg, makes the base look V-shaped, or does not materially weaken the setup.")
    return "\n".join(lines) + "\n"


def _build_prompt(ticker, metrics, config=CONFIG, legs=None, review_reasons=None):
    m = metrics
    breakout = ""
    if m["sessions_since_breakout"] is not None:
        breakout = (f"; first close above the pivot {m['sessions_since_breakout']} session(s) ago on "
                    f"{m['breakout_volume_vs_20d']}x its prior 20-day average volume")
    forming = ("the final contraction is still FORMING at the right edge (no confirming bars yet; it may deepen)"
               if m["final_contraction_confirmed"] is False else "all listed contractions are confirmed swings")
    leg_lines = _contraction_lines(legs)
    routed = ("This candidate was routed to you for visual review because: " + "; ".join(review_reasons) + ".\n"
              if review_reasons else "")
    routed += _structure_review_block(m, legs, review_reasons)
    return f"""You are analyzing {ticker} for a Minervini-style Volatility Contraction Pattern (VCP)
entry setup. Core concept: inside a Stage 2 uptrend, price builds ONE base under a common
resistance area through successive shallower pullbacks (volatility contracting), lows tend
to rise, supply dries up, and the base ends in a tight area just below a pivot that a valid
breakout clears on expanding volume.

The attached image is a plain daily candlestick chart (last {config['chart_lookback_days']} sessions) with
EMA10/EMA21, MA50/150/200 and a volume panel; the detector's contractions are NOT drawn on it.
A deterministic detector measured the numbers below from the same daily OHLCV (data up to the
latest session only). Treat every number as ground truth: do not re-estimate depths, prices or
volume ratios from the image, and quote the detector's values when you cite numbers. Its
thresholds are heuristics, not book rules. You may disagree with its numeric read; your job is
the visual judgment the numbers cannot make.
{routed}
Selected base: {m['base_start']} to {m['base_end']} ({m['base_duration_sessions']} sessions).
{m['ignored_contraction_count']} earlier swing(s) were excluded because their highs sat outside
the base's resistance band; {forming}.
{leg_lines + chr(10) if leg_lines else ''}- Contraction depths %, High-to-later-Low, chronological: {m['contraction_pcts']} (confirmed: {m['contraction_confirmed']}; durations: {m['contraction_duration_sessions']} sessions)
- Leg-over-leg depth ratios (<1 = shrinking): {m['contraction_ratios']}; largest expansion {m['largest_contraction_expansion_ratio']}x; tightening: {m['tightening_quality']}
- Contraction highs / lows: {m['contraction_high_prices']} / {m['contraction_low_prices']}; higher-low fraction {m['higher_low_fraction']}; resistance dispersion {m['resistance_dispersion_pct']}%; share of each drop recovered: {m['recovery_ratios']}
- Final contraction {m['final_contraction_pct']}%; final {m['final_zone_days']}-session zone range {m['final_zone_range_pct']}%; candle-range compression vs prior 20 sessions {m['final_range_compression_ratio']}x
- Avg volume per contraction (pullback bars only): {m['volume_leg_avgs']}; leg-over-leg ratios {m['volume_contraction_ratios']}; dry-up: {m['volume_dryup_quality']}
- Final-zone avg volume{' (pre-breakout)' if m['final_zone_pre_breakout'] else ''} vs prior 20d / 50d average: {m['final_zone_volume_ratio_20d']}x / {m['final_zone_volume_ratio_50d']}x
- Latest session volume vs prior 20d / 50d average: {m['current_volume_vs_20d']}x / {m['current_volume_vs_50d']}x
- Candidate pivot (highest right-side contraction high of this base): {m['pivot_price_candidate']}; current price {m['current_price']} ({m['pct_from_pivot']}% from pivot, {m['pivot_state']}){breakout}
- Extension above EMA10 / EMA21 / MA50: {m['pct_ext_ema10']}% / {m['pct_ext_ema21']}% / {m['pct_ext_ma50']}%
- Detector's numeric read: {m['vcp_numeric_quality']} ({'; '.join(m['vcp_quality_notes']) or 'no issues flagged'}). A heuristic summary, not a buy signal.

How to judge:
- Trend first: valid only in a Stage 2 uptrend (price above rising MA50/150/200). Otherwise avoid.
- One base: locate the contractions above on the chart and confirm they form one coherent
  consolidation under a common resistance with generally rising lows. If shrinking numbers come
  from unrelated swings of an advancing trend, or the base is loose, choppy or V-shaped, it is not
  a VCP whatever the ratios say.
- Unusual structure: if a contraction High/Low comes from a long intraday wick or a gap bar, judge
  whether the underlying price action (candle bodies) still forms the contraction the number implies.
- Tight near the pivot: the tightest action should sit just below the pivot with candles visibly
  narrowing. A FORMING final contraction is provisional; do not treat it as a completed tight area.
- Volume: judge from the ratios above, not from bar heights; say whether the volume panel visually
  supports them. Supply should recede into the final zone; a buy_now breakout needs clearly expanding
  volume on the breakout.
- Pivot: use the candidate unless the chart shows a cleaner breakout level; say which you used and
  whether the pivot is stale or visually questionable.
- Position: coiling just below the pivot -> mature / wait_for_breakout; a fresh breakout near the
  pivot on expanding volume -> breaking_out / buy_now; price far above a stale pivot, or stretched
  well above the EMA10/21 or MA50 (roughly >10-15% above MA50) -> not a fresh breakout, prefer
  wait_for_better_setup or avoid; far below the pivot -> mid-base drawdown, not an entry.

Consistency rules (your answer is rejected if the structured fields contradict each other):
- is_vcp_pattern=true only if your rationale affirms the chart forms ONE coherent VCP base.
  If your rationale concludes it is not one coherent VCP (e.g. an advancing trend with normal
  pullbacks, or a loose/choppy/V-shaped range), is_vcp_pattern must be false.
- is_vcp_pattern=false -> pattern_stage is not_present (or failed if a real VCP broke down), and
  entry_recommendation is wait_for_better_setup or avoid.
- pattern_stage forming/mature/breaking_out -> is_vcp_pattern=true. Use forming only for a genuine
  VCP that is still forming, not for any developing consolidation.

Output guidance:
- pattern_stage: forming (genuine VCP, final tightening not complete) | mature (tight base formed near
  pivot, ready) | breaking_out (pushing through the pivot on volume) | failed (a VCP that broke down /
  lost the base) | not_present (no VCP).
- entry_recommendation: buy_now (breaking out above the pivot on rising volume now) |
  wait_for_breakout (mature tight base near pivot, not yet through) | wait_for_better_setup (genuine
  uptrend but base immature/loose/extended) | avoid (not a VCP, or not in an uptrend).
- contraction_count_observed: the contractions you accept as part of the base.
- volume_dry_up_confirmed: true when the volume ratios and the chart agree that supply receded.
- suggested_stop_loss (only if valid): just below the final contraction's low (or the base low if
  tighter); state the level in the rationale.
- pivot_price: the breakout trigger you would actually use.

Be skeptical: require a real uptrend, one coherent base, and a tight, quiet area near the pivot."""


# ── Claude call ───────────────────────────────────────────────────────────

_JSON_TYPES = {"boolean": bool, "string": str, "integer": int, "number": (int, float), "null": type(None)}


def _matches(value, spec):
    if "anyOf" in spec:
        return any(_matches(value, s) for s in spec["anyOf"])
    expected = _JSON_TYPES[spec["type"]]
    if isinstance(value, bool) and spec["type"] in ("integer", "number"):
        return False
    return isinstance(value, expected) and ("enum" not in spec or value in spec["enum"])


def validate_verdict(verdict, schema=VCP_SCHEMA):
    """Minimal validation against the flat VCP_SCHEMA. Returns a list of problems
    (empty = valid). Structured outputs should already guarantee this; the check
    keeps a malformed response from being stored as if it were a verdict."""
    if not isinstance(verdict, dict):
        return [f"verdict is {type(verdict).__name__}, not an object"]
    problems = [f"missing field: {k}" for k in schema["required"] if k not in verdict]
    if schema.get("additionalProperties") is False:
        problems += [f"unexpected field: {k}" for k in verdict if k not in schema["properties"]]
    problems += [f"invalid value for {k}: {verdict[k]!r}" for k, spec in schema["properties"].items()
                 if k in verdict and not _matches(verdict[k], spec)]
    return problems


_ACTIONABLE = ("buy_now", "wait_for_breakout")
_VCP_STAGES = ("forming", "mature", "breaking_out")


def semantic_problems(verdict):
    """Deterministic contradictions between structured fields (schema-valid verdicts).
    A verdict with any of these is rejected for that ticker, never silently fixed."""
    positive, stage, entry = verdict["is_vcp_pattern"], verdict["pattern_stage"], verdict["entry_recommendation"]
    problems = []
    if not positive and stage in _VCP_STAGES:
        problems.append(f"is_vcp_pattern=false with pattern_stage={stage}")
    if positive and stage in ("not_present", "failed"):
        problems.append(f"is_vcp_pattern=true with pattern_stage={stage}")
    if not positive and entry in _ACTIONABLE:
        problems.append(f"is_vcp_pattern=false with entry_recommendation={entry}")
    if stage in ("not_present", "failed") and entry in _ACTIONABLE:
        problems.append(f"pattern_stage={stage} with entry_recommendation={entry}")
    return problems


_NEGATED_VCP = re.compile(
    r"\b(?:not|isn't|is not|no longer|never)\s+(?:a|an|one)?\s*(?:valid|clean|true|genuine|coherent|textbook|single)?\s*"
    r"(?:vcp|coherent (?:vcp|base|consolidation))\b|rather than (?:one|a single) (?:coherent )?(?:base|consolidation|vcp)",
    re.IGNORECASE)


def rationale_warnings(verdict):
    """Non-blocking: flags a positive verdict whose rationale reads as a rejection."""
    if verdict.get("is_vcp_pattern") and _NEGATED_VCP.search(verdict.get("rationale") or ""):
        return ["is_vcp_pattern=true but the rationale describes it as not one coherent VCP"]
    return []


def prompt_adherence_warnings(verdict, review_reasons):
    """Non-blocking: flags a verdict whose rationale ignores a wick/gap concern that
    caused routing. Recorded for evaluation; never invalidates the verdict."""
    wicks, gaps = _routed_structure_concerns(review_reasons)
    rationale = verdict.get("rationale") or ""
    warnings = []
    if wicks and not (re.search(r"\bwick", rationale, re.IGNORECASE)
                      and re.search(r"\bpivot|\bresistance|\bhigh\b|\blow\b|\bbod(?:y|ies)\b", rationale,
                                    re.IGNORECASE)):
        warnings.append("routed for a wick warning but the rationale does not review the wick / pivot quality")
    if gaps and not re.search(r"\bgap", rationale, re.IGNORECASE):
        warnings.append("routed for a session gap but the rationale does not review the gap")
    return warnings


def new_claude_stats():
    return {"sent": 0, "ok": 0, "errors": 0, "schema_failures": 0, "semantic_failures": 0, "refusals": 0,
            "truncated": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0}


def analyze_chart(client, ticker, chart_path, metrics, config=CONFIG, legs=None, stats=None, review_reasons=None):
    """Sends the chart image + computed metrics to Claude and returns the
    structured VCP verdict as a dict, or an ``{"error": ...}`` dict for a refusal,
    truncation, missing text, invalid JSON, schema violation or contradictory
    structured fields. ``stats`` (from ``new_claude_stats``) accumulates call
    counts and token usage."""
    stats = stats if stats is not None else new_claude_stats()
    with open(chart_path, "rb") as f:
        image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    stats["sent"] += 1
    response = client.messages.create(
        model=config["anthropic_model"],
        # room for adaptive thinking at high effort plus the JSON verdict
        max_tokens=16000,
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
                {"type": "text", "text": _build_prompt(ticker, metrics, config, legs, review_reasons)},
            ],
        }],
    )
    usage = getattr(response, "usage", None)
    for key in ("input_tokens", "output_tokens"):
        stats[key] += int(getattr(usage, key, 0) or 0)

    if response.stop_reason == "refusal":
        stats["refusals"] += 1
        stats["errors"] += 1
        return {"error": "refusal", "detail": str(getattr(response, "stop_details", None))}
    if response.stop_reason == "max_tokens":
        stats["truncated"] += 1
        stats["errors"] += 1
        return {"error": "truncated", "stop_reason": "max_tokens"}

    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None:
        stats["errors"] += 1
        return {"error": "no_text_block", "stop_reason": response.stop_reason}

    try:
        verdict = json.loads(text_block.text)
    except (TypeError, ValueError) as e:
        stats["schema_failures"] += 1
        stats["errors"] += 1
        return {"error": "invalid_json", "detail": str(e)[:300]}
    problems = validate_verdict(verdict)
    if problems:
        stats["schema_failures"] += 1
        stats["errors"] += 1
        return {"error": "schema_validation", "detail": "; ".join(problems)[:500]}
    contradictions = semantic_problems(verdict)
    if contradictions:
        stats["semantic_failures"] += 1
        stats["errors"] += 1
        return {"error": "semantic_inconsistency",
                "detail": ("; ".join(contradictions) + " | raw: " + json.dumps(verdict))[:1500]}
    stats["ok"] += 1
    return verdict


class _RetryCounter(logging.Handler):
    """Counts the Anthropic SDK's own automatic retries (it logs each one)."""

    def __init__(self, stats):
        super().__init__(logging.INFO)
        self.stats = stats

    def emit(self, record):
        if "Retrying request" in record.getMessage():
            self.stats["retries"] += 1


def run_vcp_analysis(candidates_df, chart_paths, config=CONFIG):
    """candidates_df: ranked DataFrame (top N already sliced by caller).
    chart_paths: {symbol: png_path} from stage_charts.generate_charts.
    Returns a DataFrame of numeric metrics + (optional) Claude verdicts per symbol.

    Every candidate gets numeric metrics and a verification chart. Claude is
    called per ``llm_review_mode``: off (never), selective (``llm_review_decision``)
    or all. Routing, agreement and rationale warnings go in the ``llm_review``
    column (stored with the verdict, never with the numeric metrics). A failure
    for one ticker becomes that ticker's error row; the run continues.
    """
    live_analysis = config.get("vcp_live_analysis", True)
    mode = llm_review_mode(config)
    client = None
    stats = new_claude_stats()
    sdk_logger = logging.getLogger("anthropic._base_client")
    retry_counter, previous_level = _RetryCounter(stats), sdk_logger.level
    if mode != "off":
        sdk_logger.addHandler(retry_counter)
        if sdk_logger.getEffectiveLevel() > logging.INFO:
            sdk_logger.setLevel(logging.INFO)

    rows, requested, not_requested = [], 0, 0
    try:
        for _, row in candidates_df.iterrows():
            symbol = row["Symbol"]
            chart_path = chart_paths.get(symbol)
            if not chart_path or not os.path.exists(chart_path):
                logger.warning("  %s: no chart available, skipping VCP analysis", symbol)
                continue

            from .stage_charts import fetch_chart_data
            df = fetch_chart_data(symbol, config)
            metrics, details = compute_vcp_metrics(df, config, return_details=True)
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

            review, reasons = llm_review_decision(metrics, details, config)
            review_info = {"mode": mode, "requested": review, "reasons": reasons, "agreement": None,
                           "rationale_warnings": [], "prompt_adherence_warnings": []}
            if not live_analysis:
                logger.info("  %s: skipping Claude analysis (token-free mode) — verify chart only", symbol)
                verdict = {"skipped": "token_free_mode"}
            elif not review:
                not_requested += 1
                verdict = {"skipped": "llm_review_off" if mode == "off" else "not_requested"}
                logger.info("  %s: numeric %s stands — Claude not requested (%s mode)", symbol,
                            metrics["vcp_numeric_quality"], mode)
            else:
                requested += 1
                if client is None:
                    import anthropic
                    client = anthropic.Anthropic()
                logger.info("  Analyzing %s with %s (%s)...", symbol, config["anthropic_model"], "; ".join(reasons))
                try:
                    verdict = analyze_chart(client, symbol, chart_path, metrics, config,
                                            legs=details["contraction_legs"], stats=stats, review_reasons=reasons)
                    if "error" in verdict:
                        logger.warning("  %s: VCP analysis returned an error verdict: %s %s", symbol,
                                       verdict["error"], verdict.get("detail", ""))
                    else:
                        review_info["agreement"] = vcp_agreement(metrics["vcp_numeric_quality"],
                                                                 verdict.get("is_vcp_pattern"))
                        review_info["rationale_warnings"] = rationale_warnings(verdict)
                        review_info["prompt_adherence_warnings"] = prompt_adherence_warnings(verdict, reasons)
                        if review_info["prompt_adherence_warnings"]:
                            logger.warning("  %s: prompt adherence: %s", symbol,
                                           "; ".join(review_info["prompt_adherence_warnings"]))
                        logger.info("  %s: verdict=%s stage=%s confidence=%s (numeric %s, %s)",
                                    symbol, verdict.get("entry_recommendation"), verdict.get("pattern_stage"),
                                    verdict.get("confidence"), metrics["vcp_numeric_quality"],
                                    review_info["agreement"])
                except Exception as e:
                    logger.error("  %s: VCP analysis failed", symbol, exc_info=True)
                    stats["errors"] += 1
                    verdict = {"error": f"{type(e).__name__}: {e}"[:500]}

            rows.append({"Symbol": symbol, **metrics, **verdict, "llm_review": review_info})
    finally:
        if mode != "off":
            sdk_logger.removeHandler(retry_counter)
            sdk_logger.setLevel(previous_level)

    if live_analysis:
        logger.info("Stage G Claude review (%s mode): %d requested, %d not requested; %d sent, %d ok, %d errors "
                    "(%d schema/JSON, %d contradictory, %d refusals, %d truncated), %d SDK retries, "
                    "%d input / %d output tokens.", mode, requested, not_requested, stats["sent"], stats["ok"],
                    stats["errors"], stats["schema_failures"], stats["semantic_failures"], stats["refusals"],
                    stats["truncated"], stats["retries"], stats["input_tokens"], stats["output_tokens"])
    logger.info("Stage G: VCP analysis complete for %d candidates.", len(rows))
    return pd.DataFrame(rows)
