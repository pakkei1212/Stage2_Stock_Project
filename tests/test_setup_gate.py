"""Stage I gate: which stored Stage-G setups earn an entry plan.

The gate reads Stage-G output and never changes it: the VCP verdict, its
metrics and Claude's answer are inputs only.
"""
from pipeline import setup_gate as gate
from stage_i_fakes import non_stage_g_row, stage_g_row


def test_strong_numeric_setup_coiling_below_pivot_is_ready():
    state, reasons = gate.evaluate_setup(stage_g_row(quality="strong",
                                                     pivot_state="coiling_below_pivot"))
    assert state == gate.READY
    assert "numeric strong · coiling_below_pivot" in reasons


def test_acceptable_setup_at_a_fresh_breakout_is_breakout_triggered():
    state, _ = gate.evaluate_setup(stage_g_row(quality="acceptable", pivot_state="fresh_breakout",
                                               price=102.0))
    assert state == gate.BREAKOUT_TRIGGERED


def test_weak_and_absent_numeric_quality_are_not_ready():
    for quality in ("weak", "absent", None):
        state, reasons = gate.evaluate_setup(stage_g_row(quality=quality))
        assert state == gate.NOT_READY
        assert "numeric quality" in reasons[0]


def test_extended_above_pivot_is_missed_not_ready():
    state, reasons = gate.evaluate_setup(stage_g_row(pivot_state="extended_above_pivot", price=112.0))
    assert state == gate.MISSED_EXTENDED
    assert "no chase" in reasons[0]


def test_far_below_pivot_is_not_ready():
    state, reasons = gate.evaluate_setup(stage_g_row(pivot_state="far_below_pivot", price=88.0))
    assert state == gate.NOT_READY
    assert "far_below_pivot" in reasons[0]


def test_above_pivot_without_a_fresh_breakout_is_not_ready():
    state, _ = gate.evaluate_setup(stage_g_row(pivot_state="above_pivot", price=101.0))
    assert state == gate.NOT_READY


def test_claude_reviewed_and_disagreed_is_manual_review_never_ready():
    row = stage_g_row(quality="strong", pivot_state="coiling_below_pivot", claude=False)
    state, reasons = gate.evaluate_setup(row)
    assert state == gate.MANUAL_REVIEW
    assert "is_vcp_pattern=false" in reasons[0]
    assert not gate.is_plannable(state)


def test_claude_not_requested_does_not_block_a_clean_numeric_setup():
    row = stage_g_row(quality="strong", pivot_state="coiling_below_pivot", claude=None)
    state, reasons = gate.evaluate_setup(row)
    assert state == gate.READY
    assert any("not requested" in r for r in reasons)


def test_claude_agreeing_keeps_the_setup_ready():
    state, _ = gate.evaluate_setup(stage_g_row(claude=True))
    assert state == gate.READY


def test_gate_skips_rows_stage_g_never_analysed():
    rows = [stage_g_row("AAA"), non_stage_g_row("ZZZZ")]
    gated = gate.gate_rows(rows)
    assert [row["symbol"] for row, _, _ in gated] == ["AAA"]


def test_plannable_rows_keeps_ready_and_breakout_only():
    rows = [
        stage_g_row("AAA", quality="strong", pivot_state="coiling_below_pivot"),
        stage_g_row("BBB", quality="acceptable", pivot_state="fresh_breakout", price=101.0),
        stage_g_row("CCC", quality="weak"),
        stage_g_row("DDD", claude=False),
        stage_g_row("EEE", pivot_state="extended_above_pivot", price=120.0),
    ]
    assert [(row["symbol"], state) for row, state, _ in gate.plannable_rows(rows)] == [
        ("AAA", gate.READY), ("BBB", gate.BREAKOUT_TRIGGERED)]


def test_gate_does_not_mutate_the_stage_g_row():
    row = stage_g_row()
    before = dict(row)
    gate.evaluate_setup(row)
    assert row == before
