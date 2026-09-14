"""Stage A/B/C: universe, liquidity/cap filter, sector classification and ranking.

Ported from notebooks/nasdaq_stage2_screener.ipynb (Stage A-C cells) with no
behavioral changes — the notebook remains the source of truth for interactive
exploration; this module is the native-Python copy used by the scheduled
weekly pipeline.
"""
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import ohlcv_cache
from .config import CONFIG, get_sector_cache_path

logger = logging.getLogger(__name__)

# Wikipedia (and some other sites) 403 requests carrying urllib's default
# User-Agent — pd.read_html() doesn't let you set headers, so fetch the HTML
# ourselves with a browser-like UA and hand pandas the text instead of a URL.
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-screener/1.0)"}


def _fetch_text(url, timeout=15):
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _call_with_retries(fn, retries=3, delay=2, default=None, label="yfinance call"):
    """Run a callable with retries and graceful fallback on transient network errors."""
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                logger.warning("%s failed after %d attempts: %s", label, retries, e)
    return default if default is not None else None


# Maps raw scraped sector labels -> Yahoo Finance's sector taxonomy (what
# yfinance's own .info["sector"] returns, for consistency across all 3 tiers
# in get_sector_map). Covers both GICS (S&P 500 Wikipedia table) and ICB
# (NASDAQ-100 Wikipedia table) labels; entries already spelled the Yahoo way
# (e.g. "Technology", "Industrials") fall through .get()'s default unchanged.
GICS_TO_YAHOO = {
    "Information Technology": "Technology",
    "Financials":             "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples":       "Consumer Defensive",
    "Health Care":            "Healthcare",
    "Communication Services": "Communication Services",
    "Telecommunications":     "Communication Services",  # ICB label (NASDAQ-100 table)
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
    text = _fetch_text(url)
    lines = [l for l in text.strip().split("\n")
             if not l.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")].copy()
    df = df[["Symbol", "Security Name"]].dropna()
    df = df[~df["Symbol"].str.contains(r"[\^\.\$]", regex=True, na=False)]
    return df.reset_index(drop=True)


def fallback_nasdaq100():
    """Fallback: NASDAQ-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
    tables = pd.read_html(io.StringIO(_fetch_text(url)))
    for t in tables:
        if "Ticker" in t.columns or "Symbol" in t.columns:
            col = "Ticker" if "Ticker" in t.columns else "Symbol"
            return t[[col]].rename(columns={col: "Symbol"})
    raise RuntimeError("Could not find ticker table on Wikipedia NASDAQ-100 page.")


def load_universe(config=CONFIG):
    try:
        universe_df = get_nasdaq_universe()
        logger.info("Full NASDAQ universe: %d symbols", len(universe_df))
    except Exception as e:
        logger.warning("Primary universe source failed (%s). Falling back to NASDAQ-100.", e)
        universe_df = fallback_nasdaq100()
        logger.info("Fallback universe: %d symbols", len(universe_df))

    if config["max_universe_for_testing"]:
        universe_df = universe_df.head(config["max_universe_for_testing"])
        logger.info("(Testing mode: capped at %d symbols)", len(universe_df))
    return universe_df


# ─────────────────────── Stage B: liquidity + market cap ──────────────────

def _period_to_days(period):
    """Approximate a yfinance period string ('400d', '1mo', '1y') as calendar days.
    Unrecognised/'max'-like values return a large number so the cache treats the
    request as needing full history."""
    p = str(period).strip().lower()
    try:
        if p.endswith("mo"):
            return int(p[:-2]) * 30
        if p.endswith("d"):
            return int(p[:-1])
        if p.endswith("y"):
            return int(p[:-1]) * 365
        if p.endswith("wk"):
            return int(p[:-2]) * 7
    except ValueError:
        pass
    return 3650


def _download_batches(tickers, config, interval="1d", **dl_kwargs):
    """Batched OHLCV download with retries + per-ticker fallback. Parameterised
    by yfinance download kwargs (``period=`` for a full window, or
    ``start=``/``end=`` for a delta). Returns {ticker: DataFrame}."""
    all_data = {}
    tickers = list(tickers)
    batch_size = config["batch_size"]
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        data = None
        for attempt in range(config["max_retries"]):
            try:
                data = yf.download(
                    batch, interval=interval, group_by="ticker",
                    auto_adjust=True, progress=False, threads=True, **dl_kwargs,
                )
                break
            except Exception as e:
                logger.warning("Batch %d/%d attempt %d failed: %s", batch_num, n_batches, attempt + 1, e)
                time.sleep(2 * (attempt + 1))

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            logger.warning("Batch %d/%d failed after retries; trying per-ticker fallback.", batch_num, n_batches)
            for t in batch:
                try:
                    single = _call_with_retries(
                        lambda _t=t: yf.download(
                            _t, interval=interval, auto_adjust=True, progress=False, **dl_kwargs,
                        ),
                        retries=config["max_retries"],
                        delay=2,
                        default=None,
                        label=f"{t} history",
                    )
                    if single is not None and not (isinstance(single, pd.DataFrame) and single.empty):
                        if isinstance(single.columns, pd.MultiIndex):
                            single = single.copy()
                            single.columns = single.columns.get_level_values(0)
                        all_data[t] = single.dropna(how="all")
                except Exception:
                    continue
            time.sleep(config["batch_sleep_sec"])
            continue

        for t in batch:
            try:
                # group_by="ticker" makes the ticker the outer column level, so
                # data[t] yields flat OHLCV columns for both single- and
                # multi-ticker batches. (A 1-ticker batch still comes back
                # MultiIndex as ('AAPL','Open') — selecting data[t] flattens it.)
                if isinstance(data.columns, pd.MultiIndex):
                    df_t = data[t] if t in data.columns.get_level_values(0) else None
                else:
                    df_t = data  # already-flat single-ticker frame
                if df_t is not None and not df_t.dropna(how="all").empty:
                    all_data[t] = df_t.dropna(how="all")
            except Exception:
                continue

        logger.info("Batch %d/%d done (%d tickers).", batch_num, n_batches, len(batch))
        time.sleep(config["batch_sleep_sec"])

    return all_data


def _window(df, start_needed):
    """Return the slice of ``df`` from ``start_needed`` onward."""
    return df[df.index >= start_needed].copy()


def batch_download(tickers, period="1mo", interval="1d", config=CONFIG, use_cache=None):
    """Batch-download OHLCV for a list of tickers.

    When the Parquet cache is enabled (``config['ohlcv_cache_enabled']``), each
    ticker is served one of three ways, minimising network work:
      - **cache hit**: fresh + deep enough → returned straight from disk
      - **delta**: deep enough but stale → fetch only the days since the last
        cached bar, then append (with split-aware merge)
      - **full**: missing (cold) or not enough history depth → fetch the whole
        ``period`` window
    Newly fetched data is merged back into the cache. Returns {ticker: DataFrame}.
    """
    tickers = list(tickers)
    if use_cache is None:
        use_cache = config.get("ohlcv_cache_enabled", False)

    if not use_cache:
        return _download_batches(tickers, config, interval=interval, period=period)

    cache_dir = config.get("ohlcv_cache_dir", os.path.join("data", "ohlcv_cache"))
    max_age = config.get("ohlcv_cache_max_age_days", 1)
    overlap_days = config.get("ohlcv_cache_overlap_days", 5)
    today = pd.Timestamp(pd.Timestamp.now("UTC").date())
    start_needed = today - pd.Timedelta(days=_period_to_days(period))

    result, full, delta = {}, [], {}
    for t in tickers:
        cached = ohlcv_cache.load(t, cache_dir)
        if cached is None:
            full.append(t)
        elif not ohlcv_cache.covers(cached, start_needed):
            full.append(t)                                  # need deeper history
        elif ohlcv_cache.is_fresh(cached, today, max_age):
            result[t] = _window(cached, start_needed)       # serve straight from cache
        else:
            delta[t] = cached                               # deep enough but stale → append

    logger.info(
        "OHLCV cache: %d served from disk, %d delta-fetch, %d full-fetch (of %d).",
        len(result), len(delta), len(full), len(tickers),
    )

    if full:
        fetched = _download_batches(full, config, interval=interval, period=period)
        for t, df in fetched.items():
            merged = ohlcv_cache.merge(ohlcv_cache.load(t, cache_dir), df)
            ohlcv_cache.save(t, merged, cache_dir)
            result[t] = _window(merged, start_needed)

    if delta:
        oldest_last = min(pd.to_datetime(c.index.max()) for c in delta.values())
        start = (oldest_last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetched = _download_batches(list(delta), config, interval=interval, start=start, end=end)
        for t, cached in delta.items():
            new = fetched.get(t)
            if new is not None and not new.empty:
                merged = ohlcv_cache.merge(cached, new)
                ohlcv_cache.save(t, merged, cache_dir)
                result[t] = _window(merged, start_needed)
            else:
                logger.debug("%s: delta fetch empty — serving stale cached window", t)
                result[t] = _window(cached, start_needed)   # stale beats nothing

    return result


def cheap_liquidity_filter(universe_df, config=CONFIG):
    """Stage B part 1: price + dollar-volume filter via 1-month batched data."""
    tickers = universe_df["Symbol"].tolist()
    logger.info("Pulling %dd data for %d tickers...", config["bulk_lookback_days"], len(tickers))
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

    logger.info("Liquidity filter: %d / %d passed.", len(survivors), len(tickers))
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


def _load_market_cap_cache(path, ttl_days):
    """Return {Symbol: Market Cap} for entries newer than ttl_days (market caps
    barely move week to week, so re-fetching every run wastes minutes of the
    per-ticker, throttled loop below)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        df["AsOf"] = pd.to_datetime(df["AsOf"])
        cutoff = pd.Timestamp(pd.Timestamp.now("UTC").date()) - pd.Timedelta(days=ttl_days)
        fresh = df[df["AsOf"] >= cutoff]
        return dict(zip(fresh["Symbol"], fresh["Market Cap"]))
    except Exception:
        logger.warning("Market cap cache read failed — ignoring", exc_info=True)
        return {}


def _update_market_cap_cache(path, newly_fetched):
    """Merge freshly-fetched caps (stamped today) into the CSV, keeping the
    latest AsOf per symbol."""
    if not path or not newly_fetched:
        return
    today = pd.Timestamp(pd.Timestamp.now("UTC").date())
    new_df = pd.DataFrame(
        [{"Symbol": s, "Market Cap": c, "AsOf": today} for s, c in newly_fetched.items()]
    )
    try:
        if os.path.exists(path):
            existing = pd.read_csv(path)
            existing["AsOf"] = pd.to_datetime(existing["AsOf"])
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined = (combined.sort_values("AsOf")
                    .drop_duplicates("Symbol", keep="last")
                    .sort_values("Symbol"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined.to_csv(path, index=False)
    except Exception:
        logger.warning("Market cap cache write failed", exc_info=True)


def market_cap_filter(tickers, config=CONFIG):
    """Stage B part 2: market-cap filter (mid+large cap only)."""
    rows, survivors = [], []
    cache_path = config.get("market_cap_cache_path")
    ttl_days = config.get("market_cap_cache_ttl_days", 7)
    cached_caps = _load_market_cap_cache(cache_path, ttl_days) if cache_path else {}
    newly_fetched, reused = {}, 0

    logger.info("Fetching market cap for %d liquidity survivors (%d fresh in cache)...",
                len(tickers), len(cached_caps))
    for t in tickers:
        if t in cached_caps and pd.notna(cached_caps[t]):
            market_cap = cached_caps[t]
            reused += 1
        else:
            market_cap = _get_market_cap_timeout(t)
            if pd.notna(market_cap):
                newly_fetched[t] = market_cap
            time.sleep(0.3)
        min_cap = config["min_market_cap"] or 0
        max_cap = config["max_market_cap"] or np.inf
        passes = pd.notna(market_cap) and (min_cap <= market_cap <= max_cap)
        rows.append({"Symbol": t, "Market Cap": market_cap, "Passed": passes})
        if passes:
            survivors.append(t)
        else:
            logger.debug("%s: market cap %s outside [%s, %s] — dropped.", t, market_cap, min_cap, max_cap)

    _update_market_cap_cache(cache_path, newly_fetched)
    cap_df = pd.DataFrame(rows)
    logger.info("Market cap filter: %d / %d are mid/large cap (%d reused from cache).",
                len(survivors), len(tickers), reused)
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
         "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
         "Ticker", "ICB Industry[1]", "ICB Subsector[1]"),
    ]
    for name, url, sym_col, sec_col, ind_col in sources:
        try:
            tables = pd.read_html(io.StringIO(_fetch_text(url)))
            df = next((t for t in tables
                       if sym_col in t.columns and sec_col in t.columns), None)
            if df is None:
                logger.warning("  %s: columns not found — skipping", name)
                continue
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip().replace(".", "-")
                gs = str(row.get(sec_col, "Unknown")).strip()
                gi = str(row.get(ind_col, "Unknown")).strip()
                if sym and gs not in ("nan", "", "Unknown"):
                    mapping[sym] = _normalise_gics(gs, gi)
            logger.info("  %s: %d tickers", name, len(df))
        except Exception:
            logger.warning("  %s sector lookup failed", name, exc_info=True)
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
        logger.info("Loaded %d cached sector entries", len(cache))
    else:
        cache = {}

    to_classify = [t for t in tickers if t not in cache]
    logger.info("Sector cache hits: %d | To classify: %d", len(tickers) - len(to_classify), len(to_classify))

    if to_classify:
        logger.info("Tier 1+2: Wikipedia bulk fetch...")
        wiki_map = _build_wikipedia_sector_map()

        still_needed = []
        for t in to_classify:
            if t in wiki_map:
                sector, industry = wiki_map[t]
                cache[t] = {"Sector": sector, "Industry": industry}
            else:
                still_needed.append(t)

        wiki_hits = len(to_classify) - len(still_needed)
        logger.info("  Wikipedia covered %d/%d; Tier 3 needed for %d", wiki_hits, len(to_classify), len(still_needed))
        if wiki_hits:
            _save_cache(cache, cache_path)

        if still_needed:
            logger.info("Tier 3: yfinance for %d tickers...", len(still_needed))
            for i, t in enumerate(still_needed):
                sector, industry = _fetch_sector_yf_timeout(t, timeout_sec)
                if sector in ("Timeout", "Unknown"):
                    logger.debug("%s: Tier 3 sector lookup returned %s", t, sector)
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
    logger.info("Stage C: %d tickers classified", classified)
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
    logger.info("Top %d hottest sectors: %s", config["top_n_sectors"], top_sectors)

    stage_c_survivors = merged_with_returns[
        merged_with_returns["Sector"].isin(top_sectors)
    ]["Symbol"].tolist()
    logger.info("Stage C: %d tickers in top sectors.", len(stage_c_survivors))

    return stage_c_survivors, sector_df, market_cap_stats
