"""Stage E: composite technical + fundamentals ranking.

Ported from notebooks/nasdaq_stage2_screener.ipynb, unchanged logic.
"""
from .config import CONFIG

TECH_COLS = ["Price Above MAs", "MA150 Above MA200", "200d MA Rising",
             "MAs Stacked Bullish", "Above 52w Low (25%+)", "Near 52w High",
             "RS Positive", "Volume Confirms Uptrend"]

FUND_COLS = ["EPS Growth OK", "Sales Growth OK", "EPS Trend OK", "Sales Trend OK", "ROE OK", "Profitable"]


def score_and_rank(stage_d_results, fundamentals_df, sector_df, market_cap_stats, config=CONFIG):
    df = stage_d_results.copy()
    df["Technical Score"] = df[TECH_COLS].sum(axis=1)
    df = df.merge(fundamentals_df, on="Symbol", how="left")
    df["Fundamentals Score"] = df["Fundamentals Score"].fillna(0)
    df["Composite Score"] = df["Technical Score"] + config["fundamentals_weight"] * df["Fundamentals Score"]
    df["Max Score"] = 8 + config["fundamentals_weight"] * 6

    df = df.merge(sector_df[["Symbol", "Sector", "Industry"]], on="Symbol", how="left")
    df = df.merge(market_cap_stats[["Symbol", "Market Cap"]], on="Symbol", how="left")

    cols = (["Symbol", "Sector", "Industry", "Market Cap", "Composite Score", "Max Score",
             "Technical Score", "Fundamentals Score", "Last Close"]
            + TECH_COLS + FUND_COLS
            + ["200d MA Slope %", "RS vs NASDAQ (3mo)", "RS vs NASDAQ (6mo)",
               "Pct Below 52w High", "Pct Above 52w Low",
               "Quarterly EPS Growth YoY", "Quarterly Sales Growth YoY",
               "Years of Annual Data", "Avg ROE", "Avg Profit Margin",
               "MA50", "MA150", "MA200", "52w High", "52w Low"])
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values(["Composite Score", "RS vs NASDAQ (3mo)"], ascending=False).reset_index(drop=True)
