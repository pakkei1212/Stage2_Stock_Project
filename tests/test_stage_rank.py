"""Stage E: verify Composite Score arithmetic, Max Score, missing-fundamentals
handling, and final sort order against hand-computed expected values.
"""
import pandas as pd

from pipeline.stage_rank import FUND_COLS, TECH_COLS, score_and_rank


def _stage_d_row(symbol, tech_true_count, last_close=100.0, rs_3mo=0.0):
    """Builds a Stage D row with exactly `tech_true_count` technical criteria True."""
    row = {"Symbol": symbol, "Last Close": last_close, "RS vs NASDAQ (3mo)": rs_3mo}
    for i, col in enumerate(TECH_COLS):
        row[col] = i < tech_true_count
    return row


def _fund_row(symbol, fund_true_count):
    row = {"Symbol": symbol}
    for i, col in enumerate(FUND_COLS):
        row[col] = i < fund_true_count
    row["Fundamentals Score"] = fund_true_count
    return row


def test_composite_score_is_technical_plus_weighted_fundamentals():
    stage_d = pd.DataFrame([_stage_d_row("AAA", tech_true_count=8)])
    fundamentals = pd.DataFrame([_fund_row("AAA", fund_true_count=4)])
    sector_df = pd.DataFrame([{"Symbol": "AAA", "Sector": "Technology", "Industry": "Software"}])
    market_cap = pd.DataFrame([{"Symbol": "AAA", "Market Cap": 10_000_000_000}])

    config = {"fundamentals_weight": 1.0}
    result = score_and_rank(stage_d, fundamentals, sector_df, market_cap, config)
    row = result.iloc[0]

    assert row["Technical Score"] == 8
    assert row["Fundamentals Score"] == 4
    assert row["Composite Score"] == 12          # 8 + 1.0 * 4
    assert row["Max Score"] == 14                # 8 + 1.0 * 6


def test_fundamentals_weight_scales_the_fundamentals_contribution():
    stage_d = pd.DataFrame([_stage_d_row("AAA", tech_true_count=4)])
    fundamentals = pd.DataFrame([_fund_row("AAA", fund_true_count=6)])
    sector_df = pd.DataFrame([{"Symbol": "AAA", "Sector": "Technology", "Industry": "Software"}])
    market_cap = pd.DataFrame([{"Symbol": "AAA", "Market Cap": 10_000_000_000}])

    config = {"fundamentals_weight": 0.5}
    result = score_and_rank(stage_d, fundamentals, sector_df, market_cap, config)
    row = result.iloc[0]

    assert row["Composite Score"] == 7.0   # 4 + 0.5 * 6
    assert row["Max Score"] == 11.0        # 8 + 0.5 * 6


def test_missing_fundamentals_row_is_treated_as_zero_not_dropped():
    # BBB passed Stage D but Stage D2 fundamentals lookup failed/timed out for it,
    # so it has no row in fundamentals_df at all.
    stage_d = pd.DataFrame([
        _stage_d_row("AAA", tech_true_count=5),
        _stage_d_row("BBB", tech_true_count=5),
    ])
    fundamentals = pd.DataFrame([_fund_row("AAA", fund_true_count=3)])  # no BBB row
    sector_df = pd.DataFrame([
        {"Symbol": "AAA", "Sector": "Technology", "Industry": "Software"},
        {"Symbol": "BBB", "Sector": "Technology", "Industry": "Software"},
    ])
    market_cap = pd.DataFrame([
        {"Symbol": "AAA", "Market Cap": 10_000_000_000},
        {"Symbol": "BBB", "Market Cap": 5_000_000_000},
    ])

    config = {"fundamentals_weight": 1.0}
    result = score_and_rank(stage_d, fundamentals, sector_df, market_cap, config)
    bbb = result[result["Symbol"] == "BBB"].iloc[0]

    assert bbb["Fundamentals Score"] == 0
    assert bbb["Composite Score"] == 5   # equals its Technical Score alone
    assert not result["Composite Score"].isna().any()


def test_sorted_by_composite_score_desc_then_rs_3mo_desc_as_tiebreak():
    stage_d = pd.DataFrame([
        _stage_d_row("LOW", tech_true_count=2, rs_3mo=0.50),
        _stage_d_row("HIGH", tech_true_count=8, rs_3mo=0.10),
        _stage_d_row("TIE_WEAK_RS", tech_true_count=5, rs_3mo=0.01),
        _stage_d_row("TIE_STRONG_RS", tech_true_count=5, rs_3mo=0.20),
    ])
    fundamentals = pd.DataFrame([
        _fund_row("LOW", 0), _fund_row("HIGH", 0),
        _fund_row("TIE_WEAK_RS", 0), _fund_row("TIE_STRONG_RS", 0),
    ])
    symbols = ["LOW", "HIGH", "TIE_WEAK_RS", "TIE_STRONG_RS"]
    sector_df = pd.DataFrame([{"Symbol": s, "Sector": "Technology", "Industry": "Software"} for s in symbols])
    market_cap = pd.DataFrame([{"Symbol": s, "Market Cap": 10_000_000_000} for s in symbols])

    config = {"fundamentals_weight": 1.0}
    result = score_and_rank(stage_d, fundamentals, sector_df, market_cap, config)

    # HIGH (score 8) first; the two score-5 ties ordered by RS 3mo desc; LOW (score 2) last.
    assert result["Symbol"].tolist() == ["HIGH", "TIE_STRONG_RS", "TIE_WEAK_RS", "LOW"]
    assert result["Composite Score"].tolist() == sorted(result["Composite Score"].tolist(), reverse=True)


def test_scores_stay_within_their_documented_bounds():
    stage_d = pd.DataFrame([_stage_d_row(f"T{i}", tech_true_count=i) for i in range(9)])
    fundamentals = pd.DataFrame([_fund_row(f"T{i}", fund_true_count=min(i, 6)) for i in range(9)])
    symbols = [f"T{i}" for i in range(9)]
    sector_df = pd.DataFrame([{"Symbol": s, "Sector": "Technology", "Industry": "Software"} for s in symbols])
    market_cap = pd.DataFrame([{"Symbol": s, "Market Cap": 10_000_000_000} for s in symbols])

    config = {"fundamentals_weight": 1.0}
    result = score_and_rank(stage_d, fundamentals, sector_df, market_cap, config)

    assert result["Technical Score"].between(0, 8).all()
    assert result["Fundamentals Score"].between(0, 6).all()
    assert result["Composite Score"].between(0, 14).all()
    assert (result["Max Score"] == 14).all()
