"""Stage A/B/C: universe, liquidity/cap filter, sector classification and ranking.

Ported from notebooks/nasdaq_stage2_screener.ipynb (Stage A-C cells) with no
behavioral changes — the notebook remains the source of truth for interactive
exploration; this module is the native-Python copy used by the scheduled
weekly pipeline.
"""
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import CONFIG, get_sector_cache_path

GICS_TO_YAHOO = {
    "Information Technology": "Technology",
    "Financials":             "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples":       "Consumer Defensive",
    "Health Care":            "Healthcare",
    "Communication Services": "Communication Services",
    "Industrials":            "Industrials",
    "Materials":              "Basic Materials",
    "Real Estate":            "Real Estate",
    "Energy":                 "Energy",
    "Utilities":              "Utilities",
}

GICS_INDUSTRY_TO_YAHOO = {
    "Semiconductors":                             "Semiconductors",
    "Semiconductor Equipment":                    "Semiconductor Equipment & Materials",
    "Systems Software":                           "Software—Infrastructure",
    "Application Software":                       "Software—Application",
    "Internet Services & Infrastructure":         "Software—Infrastructure",
    "Interactive Media & Services":               "Internet Content & Information",
    "Technology Hardware, Storage & Peripherals": "Computer Hardware",
    "Biotechnology":                              "Biotechnology",
    "Pharmaceuticals":                            "Drug Manufacturers—General",
    "Health Care Equipment":                      "Medical Devices",
    "Managed Health Care":                        "Healthcare Plans",
    "Restaurants":                                "Restaurants",
    "Automobile Manufacturers":                   "Auto Manufacturers",
    "Broadline Retail":                           "Discount Stores",
    "Specialty Retail":                           "Specialty Retail",
    "Electric Utilities":                         "Utilities—Regulated Electric",
    "Integrated Telecommunication Services":      "Telecom Services",
    "Wireless Telecommunication Services":        "Telecom Services",
    "Oil, Gas & Consumable Fuels":                "Oil & Gas E&P",
    "Investment Banking & Brokerage":             "Capital Markets",
    "Consumer Finance":                           "Credit Services",
}


# ─────────────────────── Stage A: universe ────────────────────────────────

def get_nasdaq_universe():
    """Pull full NASDAQ-listed symbol directory from Nasdaq Trader."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-screener/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    lines = [l for l in resp.text.strip().split("\n")
             if not l.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")].copy()
    df = df[["Symbol", "Security Name"]].dropna()
    df = df[~df["Symbol"].str.contains(r"[\^\.\$]", regex=True, na=False)]
    return df.reset_index(drop=True)


def fallback_nasdaq100():
    """Fallback: NASDAQ-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        if "Ticker" in t.columns or "Symbol" in t.columns:
            col = "Ticker" if "Ticker" in t.columns else "Symbol"
            return t[[col]].rename(columns={col: "Symbol"})
    raise RuntimeError("Could not find ticker table on Wikipedia NASDAQ-100 page.")


def load_universe(config=CONFIG):
    try:
        universe_df = get_nasdaq_universe()
        print(f"Full NASDAQ universe: {len(universe_df)} symbols")
    except Exception as e:
        print(f"Primary source failed ({e}). Falling back to NASDAQ-100.")
        universe_df = fallback_nasdaq100()
        print(f"Fallback universe: {len(universe_df)} symbols")

    if config["max_universe_for_testing"]:
        universe_df = universe_df.head(config["max_universe_for_testing"])
        print(f"(Testing mode: capped at {len(universe_df)} symbols)")
    return universe_df


# ─────────────────────── Stage B: liquidity + market cap ──────────────────

def batch_download(tickers, period="1mo", interval="1d", config=CONFIG):
    """Batch-download OHLCV for a list of tickers with retries and sleep between batches."""
    all_data = {}
    batch_size = config["batch_size"]
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        for attempt in range(config["max_retries"]):
            try:
                data = yf.download(
                    batch, period=period, interval=interval,
                    group_by="ticker", auto_adjust=True,
                    progress=False, threads=True,
                )
                break
            except Exception as e:
                print(f"  Batch {batch_num}/{n_batches} attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  Batch {batch_num}/{n_batches} failed after retries.")
            continue

        for t in batch:
            try:
                df_t = data if len(batch) == 1 else (
                    data[t] if t in data.columns.get_level_values(0) else None
                )
                if df_t is not None and not df_t.dropna(how="all").empty:
                    all_data[t] = df_t.dropna(how="all")
            except Exception:
                continue

        print(f"  Batch {batch_num}/{n_batches} done ({len(batch)} tickers).")
        time.sleep(config["batch_sleep_sec"])

    return all_data


def cheap_liquidity_filter(universe_df, config=CONFIG):
    """Stage B part 1: price + dollar-volume filter via 1-month batched data."""
    tickers = universe_df["Symbol"].tolist()
    print(f"Pulling {config['bulk_lookback_days']}d data for {len(tickers)} tickers...")
    raw = batch_download(tickers, period=f"{config['bulk_lookback_days']}d", config=config)

    survivors, rows = [], []
    for t, df in raw.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        last_price = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else np.nan
        avg_dollar_vol = (df["Close"] * df["Volume"]).mean()
        if pd.isna(last_price) or pd.isna(avg_dollar_vol):
            continue
        passes = (config["min_price"] <= last_price <= config["max_price"]
                  and avg_dollar_vol >= config["min_avg_dollar_volume"])
        rows.append({"Symbol": t, "Last Price": last_price,
                     "Avg Dollar Volume (1mo)": avg_dollar_vol, "Passed": passes})
        if passes:
            survivors.append(t)

    print(f"Liquidity filter: {len(survivors)} / {len(tickers)} passed.")
    return survivors, pd.DataFrame(rows)


def _get_market_cap_timeout(ticker, timeout_sec=8):
    """Fetch market cap with hard timeout — fast_info first, .info fallback."""
    for fn in (
        lambda _t=ticker: yf.Ticker(_t).fast_info.get("marketCap", np.nan),
        lambda _t=ticker: yf.Ticker(_t).info.get("marketCap", np.nan),
    ):
        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(fn)
        ex.shutdown(wait=False)
        try:
            cap = future.result(timeout=timeout_sec)
            if cap and not (isinstance(cap, float) and np.isnan(cap)):
                return cap
        except (FuturesTimeoutError, Exception):
            pass
    return np.nan


def market_cap_filter(tickers, config=CONFIG):
    """Stage B part 2: market-cap filter (mid+large cap only)."""
    rows, survivors = [], []
    print(f"Fetching market cap for {len(tickers)} liquidity survivors...")
    for i, t in enumerate(tickers):
        market_cap = _get_market_cap_timeout(t)
        min_cap = config["min_market_cap"] or 0
        max_cap = config["max_market_cap"] or np.inf
        passes = pd.notna(market_cap) and (min_cap <= market_cap <= max_cap)
        rows.append({"Symbol": t, "Market Cap": market_cap, "Passed": passes})
        if passes:
            survivors.append(t)
        time.sleep(0.3)
    cap_df = pd.DataFrame(rows)
    print(f"Market cap filter: {len(survivors)} / {len(tickers)} are mid/large cap.")
    return survivors, cap_df


# ─────────────────────── Stage C: sector classification + ranking ─────────

def _normalise_gics(gics_sector, gics_industry):
    return (GICS_TO_YAHOO.get(gics_sector, gics_sector),
            GICS_INDUSTRY_TO_YAHOO.get(gics_industry, gics_industry))


def _build_wikipedia_sector_map():
    """One bulk fetch each from Wikipedia S&P500 and NASDAQ-100. No per-ticker calls."""
    mapping = {}
    sources = [
        ("S&P 500",
         "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
         "Symbol", "GICS Sector", "GICS Sub-Industry"),
        ("NASDAQ-100",
         "https://en.wikipedia.org/wiki/Nasdaq-100",
         "Ticker", "GICS Sector", "GICS Sub-Industry"),
    ]
    for name, url, sym_col, sec_col, ind_col in sources:
        try:
            tables = pd.read_html(url)
            df = next((t for t in tables
                       if sym_col in t.columns and sec_col in t.columns), None)
            if df is None:
                print(f"  {name}: columns not found — skipping")
                continue
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip().replace(".", "-")
                gs = str(row.get(sec_col, "Unknown")).strip()
                gi = str(row.get(ind_col, "Unknown")).strip()
                if sym and gs not in ("nan", "", "Unknown"):
                    mapping[sym] = _normalise_gics(gs, gi)
            print(f"  {name}: {len(df)} tickers")
        except Exception as e:
            print(f"  {name} failed: {e}")
    return mapping


def _fetch_sector_yf_timeout(ticker, timeout_sec=8):
    """yfinance .info per-ticker fallback with hard timeout."""
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yf_sector")
    future = ex.submit(lambda _t=ticker: yf.Ticker(_t).info)
    ex.shutdown(wait=False)
    try:
        info = future.result(timeout=timeout_sec)
        return (info.get("sector", "Unknown") or "Unknown",
                info.get("industry", "Unknown") or "Unknown")
    except FuturesTimeoutError:
        return "Timeout", "Timeout"
    except Exception:
        return "Unknown", "Unknown"


def _save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame.from_dict(cache, orient="index") \
      .rename_axis("Symbol").to_csv(path)


def get_sector_map(tickers, cache_path=None, timeout_sec=8, checkpoint_every=25):
    """3-tier sector lookup with CSV caching.
      Tier 1: Wikipedia S&P 500  (bulk, free, reliable)
      Tier 2: Wikipedia NASDAQ-100 (bulk, free, reliable)
      Tier 3: yfinance .info with per-call timeout (residual only)
    """
    cache_path = cache_path or get_sector_cache_path()
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path).set_index("Symbol").to_dict("index")
        print(f"Loaded {len(cache)} cached sector entries")
    else:
        cache = {}

    to_classify = [t for t in tickers if t not in cache]
    print(f"Sector cache hits: {len(tickers)-len(to_classify)} | To classify: {len(to_classify)}")

    if to_classify:
        print("Tier 1+2: Wikipedia bulk fetch...")
        wiki_map = _build_wikipedia_sector_map()

        still_needed = []
        for t in to_classify:
            if t in wiki_map:
                sector, industry = wiki_map[t]
                cache[t] = {"Sector": sector, "Industry": industry}
            else:
                still_needed.append(t)

        wiki_hits = len(to_classify) - len(still_needed)
        print(f"  Wikipedia covered {wiki_hits}/{len(to_classify)}; Tier 3 needed for {len(still_needed)}")
        if wiki_hits:
            _save_cache(cache, cache_path)

        if still_needed:
            print(f"Tier 3: yfinance for {len(still_needed)} tickers...")
            for i, t in enumerate(still_needed):
                sector, industry = _fetch_sector_yf_timeout(t, timeout_sec)
                cache[t] = {"Sector": sector, "Industry": industry}
                if (i + 1) % checkpoint_every == 0:
                    _save_cache(cache, cache_path)
                time.sleep(0.3)
            _save_cache(cache, cache_path)

    _save_cache(cache, cache_path)
    results = {t: cache.get(t, {"Sector": "Unknown", "Industry": "Unknown"}) for t in tickers}
    df = pd.DataFrame.from_dict(results, orient="index").reset_index() \
        .rename(columns={"index": "Symbol"})
    classified = len(df[~df["Sector"].isin(["Unknown", "Timeout"])])
    print(f"Stage C: {classified} tickers classified")
    return df


def rank_sector_strength(sector_df, config=CONFIG):
    """Rank sectors by median stock RS vs the NASDAQ Composite."""
    lookback = config["sector_rs_lookback_days"]
    benchmark = config["rs_benchmark"]

    price_data = batch_download(sector_df["Symbol"].tolist(),
                                 period=f"{lookback+10}d", interval="1d", config=config)
    bench_raw = yf.download(benchmark, period=f"{lookback+10}d",
                             interval="1d", auto_adjust=True, progress=False)["Close"]
    bench_close = bench_raw.iloc[:, 0] if isinstance(bench_raw, pd.DataFrame) else bench_raw

    bench_ret_val = bench_close.iloc[-1] / bench_close.iloc[0] - 1
    bench_ret = bench_ret_val.item() if hasattr(bench_ret_val, "item") else float(bench_ret_val)

    returns = {}
    for t, df in price_data.items():
        closes = df["Close"].dropna()
        if len(closes) >= 2:
            returns[t] = closes.iloc[-1] / closes.iloc[0] - 1

    ret_df = pd.DataFrame(list(returns.items()), columns=["Symbol", f"Return_{lookback}d"])
    merged = sector_df.merge(ret_df, on="Symbol", how="inner")

    sector_strength = (
        merged.groupby("Sector")[f"Return_{lookback}d"]
        .median().reset_index()
        .rename(columns={f"Return_{lookback}d": "Median Stock Return"})
    )
    sector_strength["Benchmark Return"] = bench_ret
    sector_strength["Relative Strength"] = (
        sector_strength["Median Stock Return"] - sector_strength["Benchmark Return"]
    )
    return sector_strength.sort_values("Relative Strength", ascending=False).reset_index(drop=True), merged


def run_stage_abc(config=CONFIG):
    """Runs Stage A (universe) -> B (liquidity+cap) -> C (sector strength).

    Returns (stage_c_survivors, sector_df, market_cap_stats).
    """
    universe_df = load_universe(config)

    liquidity_survivors, _ = cheap_liquidity_filter(universe_df, config)
    stage_b_survivors, market_cap_stats = market_cap_filter(liquidity_survivors, config)

    sector_df = get_sector_map(stage_b_survivors)
    sector_df = sector_df[~sector_df["Sector"].isin(["Unknown", "Timeout"])]

    sector_strength, merged_with_returns = rank_sector_strength(sector_df, config)
    top_sectors = sector_strength.head(config["top_n_sectors"])["Sector"].tolist()
    print(f"Top {config['top_n_sectors']} hottest sectors: {top_sectors}")

    stage_c_survivors = merged_with_returns[
        merged_with_returns["Sector"].isin(top_sectors)
    ]["Symbol"].tolist()
    print(f"Stage C: {len(stage_c_survivors)} tickers in top sectors.")

    return stage_c_survivors, sector_df, market_cap_stats
