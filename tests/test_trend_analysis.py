"""Stage H: deterministic Top-N trend metrics and categories over synthetic
trading-session history."""
from datetime import date

import pytest

from pipeline import trend_analysis as ta
from pipeline import trading_calendar as tc


def _rows(session, ranking, scores=None, vcp=None):
    """ranking: list of symbols in rank order."""
    rows = []
    for i, sym in enumerate(ranking):
        rows.append({
            "trading_date": str(session), "symbol": sym, "final_rank": i + 1,
            "composite_score": (scores or {}).get(sym, 14.0 - i),
            **({"vcp_status": "analyzed", "vcp_entry_recommendation": vcp[sym][0], "vcp_confidence": vcp[sym][1]}
               if vcp and sym in vcp else {"vcp_status": None}),
        })
    return rows


S = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4),
     date(2026, 9, 8), date(2026, 9, 9)]   # 09-07 Labor Day, 09-05/06 weekend


def _analyze(history, sessions=None, top_n=3, **kw):
    sessions = sessions or [s for s, _ in history]
    rows = [r for _, rs in history for r in rs]
    return ta.analyze(rows, sessions, top_n=top_n, **kw)


def test_first_session_has_no_history_and_no_categories():
    report = _analyze([(S[0], _rows(S[0], ["AAA", "BBB", "CCC"]))])
    assert not report.has_history
    assert all(e.category is None for e in report.entries)
    assert report.new_entries == [] and report.dropped == []


def test_new_entrant_and_dropped_symbol():
    report = _analyze([
        (S[0], _rows(S[0], ["AAA", "BBB", "CCC", "DDD"])),
        (S[1], _rows(S[1], ["AAA", "BBB", "DDD", "CCC"])),
    ])
    assert report.new_entries == ["DDD"]
    assert [(d.symbol, d.previous_rank, d.current_rank) for d in report.dropped] == [("CCC", 3, 4)]


def test_dropped_symbol_missing_from_population_has_no_current_rank():
    report = _analyze([
        (S[0], _rows(S[0], ["AAA", "BBB", "CCC"])),
        (S[1], _rows(S[1], ["AAA", "BBB", "EEE"])),
    ])
    assert [(d.symbol, d.current_rank) for d in report.dropped] == [("CCC", None)]


def test_rank_rise_fall_and_same_rank_deltas():
    report = _analyze([
        (S[0], _rows(S[0], ["AAA", "BBB", "CCC", "x1", "x2", "RISE"])),
        (S[1], _rows(S[1], ["RISE", "AAA", "BBB", "CCC"])),
    ], top_n=6)
    e = {t.symbol: t for t in report.entries}
    assert e["RISE"].rank_delta_1 == 5 and e["RISE"].category == ta.RISING
    assert e["AAA"].rank_delta_1 == -1 and e["AAA"].category == ta.STEADY
    assert e["CCC"].rank_delta_1 == -1


def test_falling_requires_threshold_move():
    report = _analyze([
        (S[0], _rows(S[0], ["FALL", "a", "b", "c", "d"])),
        (S[1], _rows(S[1], ["a", "b", "c", "FALL", "d"])),
    ], top_n=5)
    fall = next(t for t in report.entries if t.symbol == "FALL")
    assert fall.rank_delta_1 == -3 and fall.category == ta.FALLING


def test_same_rank_is_zero_delta():
    report = _analyze([(S[0], _rows(S[0], ["AAA", "BBB"])), (S[1], _rows(S[1], ["AAA", "BBB"]))])
    assert report.entries[0].rank_delta_1 == 0


def test_five_session_movement_and_score_deltas():
    history = [(S[0], _rows(S[0], ["x", "y", "z", "w", "AAA"], scores={"AAA": 9.0}))]
    for s in S[1:5]:
        history.append((s, _rows(s, ["x", "y", "AAA"], scores={"AAA": 10.0})))
    history.append((S[5], _rows(S[5], ["AAA", "x", "y"], scores={"AAA": 12.5})))
    report = _analyze(history, top_n=5)
    a = next(t for t in report.entries if t.symbol == "AAA")
    assert a.rank_5_ago == 5 and a.rank_delta_5 == 4
    assert a.previous_score == 10.0 and a.score_delta_1 == 2.5
    assert a.score_5_ago == 9.0 and a.score_delta_5 == 3.5


def test_top_n_streak_and_lookback_appearances():
    history = [
        (S[0], _rows(S[0], ["AAA", "b", "c"])),
        (S[1], _rows(S[1], ["b", "c", "d", "AAA"])),    # AAA out of top 3 -> breaks streak
        (S[2], _rows(S[2], ["AAA", "b", "c"])),
        (S[3], _rows(S[3], ["b", "AAA", "c"])),
        (S[4], _rows(S[4], ["b", "c", "AAA"])),
    ]
    a = next(t for t in _analyze(history).entries if t.symbol == "AAA")
    assert a.top_n_streak == 3
    assert a.top_n_appearances == 4
    assert a.category == ta.PERSISTENT


def test_weekend_and_holiday_do_not_break_streak():
    # 09-04 (Fri) -> 09-08 (Tue, after the Labor Day Monday): consecutive sessions.
    sessions = tc.sessions_between(date(2026, 9, 3), date(2026, 9, 9))
    assert sessions == [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 8), date(2026, 9, 9)]
    history = [(s, _rows(s, ["AAA", "BBB", "CCC"])) for s in sessions]
    a = _analyze(history).entries[0]
    assert a.top_n_streak == 4 and a.category == ta.PERSISTENT


def test_unrecorded_session_is_reported_as_gap_not_hidden():
    window = tc.recent_trading_sessions(4, date(2026, 9, 9))
    recorded = [date(2026, 9, 3), date(2026, 9, 8), date(2026, 9, 9)]   # 09-04 missed
    rows = [r for s in recorded for r in _rows(s, ["AAA", "BBB", "CCC"])]
    report = ta.analyze(rows, recorded, top_n=3, expected_sessions=window)
    assert report.missing_sessions == [date(2026, 9, 4)]
    assert report.previous_session == date(2026, 9, 8)


def test_vcp_verdict_and_confidence_change():
    report = _analyze([
        (S[0], _rows(S[0], ["AAA", "BBB"], vcp={"AAA": ("wait_for_better_setup", "medium"),
                                                "BBB": ("avoid", "low")})),
        (S[1], _rows(S[1], ["AAA", "BBB"], vcp={"AAA": ("wait_for_breakout", "high"),
                                                "BBB": ("avoid", "low")})),
    ])
    a, b = report.entries
    assert a.vcp_verdict_changed and a.previous_vcp_verdict == "wait_for_better_setup"
    assert a.vcp_confidence_changed and a.vcp_confidence == "high"
    assert not b.vcp_verdict_changed and not b.vcp_confidence_changed


def test_verdict_change_ignored_when_previous_was_not_analysed():
    report = _analyze([
        (S[0], _rows(S[0], ["AAA"])),
        (S[1], _rows(S[1], ["AAA"], vcp={"AAA": ("buy_now", "high")})),
    ])
    assert not report.entries[0].vcp_verdict_changed


def test_rising_new_entrant_is_both_rising_and_new():
    report = _analyze([
        (S[0], _rows(S[0], ["a", "b", "c", "d", "e", "NEW"])),
        (S[1], _rows(S[1], ["a", "NEW", "b"])),
    ])
    new = next(t for t in report.entries if t.symbol == "NEW")
    assert new.category == ta.RISING and new.is_new_entry and "NEW" in report.new_entries


def test_analyze_requires_a_session():
    with pytest.raises(ValueError):
        ta.analyze([], [], top_n=10)


def test_sessions_before_first_recording_are_not_reported_as_gaps():
    window = tc.recent_trading_sessions(10, date(2026, 9, 9))
    recorded = [date(2026, 9, 8), date(2026, 9, 9)]            # history only just started
    rows = [r for s in recorded for r in _rows(s, ["AAA"])]
    assert ta.analyze(rows, recorded, top_n=3, expected_sessions=window).missing_sessions == []
