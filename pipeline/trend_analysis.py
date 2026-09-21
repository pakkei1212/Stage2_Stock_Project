"""Top-N cross-session trend monitoring (Stage H) — pure and deterministic.

Describes persistence and change in the ranked list; it does not predict
anything and uses no LLM.

The time axis is the ordered list of *recorded trading sessions* (canonical
runs) inside the lookback window. Weekends and exchange holidays are never
sessions, so they can't break a streak. A session the laptop never recorded is
simply absent from the axis: comparisons step over it, and the gap is reported
(``missing_sessions``, only within the recorded span) rather than hidden.

Conventions:
  * rank 1 is best; ``rank_delta`` > 0 means the symbol moved UP
    (``previous_rank - current_rank``).
  * "N-session" deltas compare against the recorded session N positions back.

Categories (first matching rule wins; thresholds from config):
  🚀 rising      rank improved by >= trend_rank_move_min vs the previous session
  ⬇ falling      rank worsened by >= trend_rank_move_min vs the previous session
  🔥 persistent  in Top-N for >= trend_persistent_min_streak consecutive sessions
  🆕 new         in Top-N now, not in Top-N in the previous session
  ·  steady      none of the above
Membership lists are separate: ``new_entries`` (entered Top-N) and ``dropped``
(left Top-N) — a symbol can be both rising and a new entrant.
No category is assigned when there is no previous recorded session.
"""
from dataclasses import dataclass, field
from typing import Optional

RISING = "rising"
FALLING = "falling"
PERSISTENT = "persistent"
NEW = "new"
STEADY = "steady"

CATEGORY_LABELS = {
    PERSISTENT: "🔥 Persistent",
    RISING: "🚀 Rising",
    NEW: "🆕 New",
    FALLING: "⬇ Falling",
    STEADY: "· Steady",
}


@dataclass
class SymbolTrend:
    symbol: str
    rank: int
    composite_score: Optional[float]
    previous_rank: Optional[int] = None
    rank_delta_1: Optional[int] = None
    rank_5_ago: Optional[int] = None
    rank_delta_5: Optional[int] = None
    previous_score: Optional[float] = None
    score_delta_1: Optional[float] = None
    score_5_ago: Optional[float] = None
    score_delta_5: Optional[float] = None
    top_n_streak: int = 1
    top_n_appearances: int = 1
    is_new_entry: bool = False
    category: Optional[str] = None
    vcp_verdict: Optional[str] = None
    previous_vcp_verdict: Optional[str] = None
    vcp_verdict_changed: bool = False
    vcp_confidence: Optional[str] = None
    previous_vcp_confidence: Optional[str] = None
    vcp_confidence_changed: bool = False
    rank_history: list = field(default_factory=list)   # [(session, rank|None)] oldest→newest


@dataclass
class DroppedSymbol:
    symbol: str
    previous_rank: int
    current_rank: Optional[int]   # None = not in today's ranked population at all


@dataclass
class TrendReport:
    current_session: object
    previous_session: object
    sessions: list
    missing_sessions: list
    top_n: int
    entries: list
    new_entries: list
    dropped: list

    @property
    def has_history(self):
        return self.previous_session is not None

    def by_category(self, category):
        return [e for e in self.entries if e.category == category]


def vcp_verdict_label(row):
    """The Stage-G verdict as stored: entry_recommendation when analysed,
    otherwise the non-verdict status (skipped / error), or None."""
    if not row:
        return None
    status = row.get("vcp_status")
    if status == "analyzed":
        return row.get("vcp_entry_recommendation")
    return status


def _index(rows):
    by_session = {}
    for r in rows:
        by_session.setdefault(str(r["trading_date"]), {})[r["symbol"]] = r
    return by_session


def analyze(rows, sessions, *, top_n, rank_move_min=3, persistent_min_streak=3, expected_sessions=None):
    """Build a TrendReport.

    rows      stock_results dicts (symbol, trading_date, final_rank, composite_score, vcp_*)
    sessions  recorded sessions in the lookback window, ascending; last = current
    expected_sessions  exchange sessions in the same window (for gap reporting)
    """
    sessions = sorted(sessions)
    if not sessions:
        raise ValueError("analyze() needs at least the current session")
    keys = [str(s) for s in sessions]
    by_session = _index(rows)
    current_key = keys[-1]
    current = by_session.get(current_key, {})
    previous_key = keys[-2] if len(keys) >= 2 else None
    previous = by_session.get(previous_key, {}) if previous_key else {}
    five_key = keys[-6] if len(keys) >= 6 else None
    five = by_session.get(five_key, {}) if five_key else {}

    def in_top(session_key, symbol):
        r = by_session.get(session_key, {}).get(symbol)
        return r is not None and r["final_rank"] <= top_n

    current_top = sorted((r for r in current.values() if r["final_rank"] <= top_n),
                         key=lambda r: r["final_rank"])

    entries = []
    for r in current_top:
        sym = r["symbol"]
        t = SymbolTrend(symbol=sym, rank=r["final_rank"], composite_score=r.get("composite_score"))

        prev = previous.get(sym)
        if prev is not None:
            t.previous_rank = prev["final_rank"]
            t.rank_delta_1 = prev["final_rank"] - r["final_rank"]
            t.previous_score = prev.get("composite_score")
            t.score_delta_1 = _delta(r.get("composite_score"), prev.get("composite_score"))
        old = five.get(sym)
        if old is not None:
            t.rank_5_ago = old["final_rank"]
            t.rank_delta_5 = old["final_rank"] - r["final_rank"]
            t.score_5_ago = old.get("composite_score")
            t.score_delta_5 = _delta(r.get("composite_score"), old.get("composite_score"))

        streak = 0
        for k in reversed(keys):
            if not in_top(k, sym):
                break
            streak += 1
        t.top_n_streak = streak
        t.top_n_appearances = sum(in_top(k, sym) for k in keys)
        t.rank_history = [(k, by_session.get(k, {}).get(sym, {}).get("final_rank")) for k in keys]

        t.is_new_entry = previous_key is not None and not in_top(previous_key, sym)

        t.vcp_verdict = vcp_verdict_label(r)
        t.vcp_confidence = r.get("vcp_confidence") if r.get("vcp_status") == "analyzed" else None
        if prev is not None:
            t.previous_vcp_verdict = vcp_verdict_label(prev)
            t.previous_vcp_confidence = prev.get("vcp_confidence") if prev.get("vcp_status") == "analyzed" else None
            both_analyzed = r.get("vcp_status") == "analyzed" and prev.get("vcp_status") == "analyzed"
            t.vcp_verdict_changed = both_analyzed and t.vcp_verdict != t.previous_vcp_verdict
            t.vcp_confidence_changed = both_analyzed and t.vcp_confidence != t.previous_vcp_confidence

        t.category = _categorise(t, previous_key is not None, rank_move_min, persistent_min_streak)
        entries.append(t)

    dropped = []
    if previous_key is not None:
        for r in sorted(previous.values(), key=lambda r: r["final_rank"]):
            if r["final_rank"] <= top_n and not in_top(current_key, r["symbol"]):
                now_row = current.get(r["symbol"])
                dropped.append(DroppedSymbol(r["symbol"], r["final_rank"],
                                             now_row["final_rank"] if now_row else None))

    # Gaps are expected sessions *inside* the recorded span; sessions before the
    # first recording in the window are just pre-history, not missed runs.
    missing = []
    if expected_sessions is not None:
        recorded = set(keys)
        missing = [s for s in expected_sessions if str(s) not in recorded and str(s) > keys[0]]

    return TrendReport(
        current_session=sessions[-1],
        previous_session=sessions[-2] if len(sessions) >= 2 else None,
        sessions=sessions,
        missing_sessions=missing,
        top_n=top_n,
        entries=entries,
        new_entries=[e.symbol for e in entries if e.is_new_entry],
        dropped=dropped,
    )


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def _categorise(t, has_previous, rank_move_min, persistent_min_streak):
    if not has_previous:
        return None
    if t.rank_delta_1 is not None and t.rank_delta_1 >= rank_move_min:
        return RISING
    if t.rank_delta_1 is not None and t.rank_delta_1 <= -rank_move_min:
        return FALLING
    if t.top_n_streak >= persistent_min_streak:
        return PERSISTENT
    if t.is_new_entry:
        return NEW
    return STEADY
