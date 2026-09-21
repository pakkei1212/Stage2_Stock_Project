# NASDAQ Stage 2 Screener — Docker Environment

Dockerised Jupyter environment for the **NASDAQ Stage 2 / Minervini Trend Template screener**  
(`nasdaq_stage2_screener.ipynb`).

## What's included

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12-slim image with all dependencies |
| `docker-compose.yml` | Three services: interactive Jupyter, scheduled notebook runner, session-aware pipeline scheduler |
| `requirements.txt` | Pinned Python dependencies |
| `run_notebook.sh` | Script used by the `runner` service to execute the notebook headlessly |
| `run_weekly.sh` | Manual one-shot helper that runs the native-Python pipeline |
| `scripts/` | Windows laptop operation: `start_scheduler.ps1` (loads `.env`, runs the scheduler) and `register_windows_task.ps1` |
| `.env.example` | Template for `.env`: secrets (empty), schedule, Telegram, history and trend settings |
| `notebooks/` | Interactive exploration notebook — `nasdaq_stage2_screener.ipynb` |
| `pipeline/` | Native-Python weekly pipeline: screen → rank → chart → Claude VCP vision analysis |
| `data/` | Persistent outputs & caches: `sector_cache.csv`, `market_cap_cache.csv`, `ohlcv_cache/` price history, watchlist CSVs, chart PNGs, VCP reports, logs, `history/` SQLite, `runs/<session>/run.json` manifests |
| `tests/` | Pytest suite verifying each pipeline stage's output correctness (see [Testing](#testing)) |

---

## Weekly pipeline (`pipeline/`)

A native-Python (non-notebook) version of the screener, extended with automated chart
generation and an LLM entry-point pass:

```
Stage A-C : universe, liquidity/cap filter, sector strength   (pipeline/data_sources.py)
Stage D-D2: Minervini Trend Template + fundamentals screen    (pipeline/stage_screen.py)
Stage E   : composite technical+fundamentals ranking          (pipeline/stage_rank.py)
Stage F   : daily price/volume chart PNG per top candidate    (pipeline/stage_charts.py)
Stage G   : Claude vision VCP entry-point analysis (top N)    (pipeline/stage_vcp_analysis.py)
Stage H   : persist history, Top-N trends, Telegram report     (pipeline/scheduler.py)
```

Stage H is strictly post-Stage-G: it never changes what Stage A–G screen or how
they score. See [Scheduled operation](#scheduled-operation-stage-h).

### Scoring methodology (Stage E)

Each candidate that survives the Stage D screen gets three numbers: a **Technical
Score**, a **Fundamentals Score**, and a **Composite Score** that combines them.

**Technical Score (0–8)** — one point per Minervini Trend Template criterion that's
true (`pipeline/stage_screen.py::compute_stage2_metrics`):

| Criterion | Condition |
|---|---|
| Price Above MAs | Last close > 150-day MA **and** > 200-day MA |
| MA150 Above MA200 | 150-day MA > 200-day MA |
| 200d MA Rising | 200-day MA today > 200-day MA 20 trading days ago (`ma_slope_window`) |
| MAs Stacked Bullish | 50-day MA > 150-day MA **and** > 200-day MA |
| Above 52w Low (25%+) | Price is at least 25% above its 52-week low (`above_low_pct`) |
| Near 52w High | Price is within 25% of its 52-week high (`near_high_pct`) |
| RS Positive | Stock's 3-month **and** 6-month returns both beat the NASDAQ Composite (`^IXIC`) benchmark |
| Volume Confirms Uptrend | Over the last 60 trading days, average volume on up days > average volume on down days |

**Fundamentals Score (0–6)** — one point per CANSLIM-style criterion, computed only
for the (small) Stage D survivor set via `yfinance` (`pipeline/stage_screen.py::fundamentals_screen`):

| Criterion | Condition |
|---|---|
| EPS Growth OK | Latest quarterly YoY EPS growth ≥ 20% (`min_quarterly_eps_growth`) |
| Sales Growth OK | Latest quarterly YoY revenue growth ≥ 10% (`min_quarterly_sales_growth`) |
| EPS Trend OK | Annual EPS grew year-over-year in **every** fiscal year yfinance exposes |
| Sales Trend OK | Annual revenue grew year-over-year in every fiscal year exposed |
| ROE OK | Average return on equity across those years ≥ 17% (`min_roe`) |
| Profitable | Average net profit margin across those years > 0% (`min_profit_margin`) |

If fundamentals data can't be fetched for a symbol (timeout, missing filings), it
scores 0 rather than being dropped.

**Composite Score** (`pipeline/stage_rank.py::score_and_rank`):

```
Composite Score = Technical Score + fundamentals_weight × Fundamentals Score
Max Score        = 8 + fundamentals_weight × 6
```

`fundamentals_weight` defaults to `1.0` (`pipeline/config.py`), so Composite Score
is simply Technical + Fundamentals out of a max of 14 — raise the weight to make
fundamentals count for more, or set it below 1 to lean more technical. The final
watchlist is sorted by Composite Score descending, with ties broken by 3-month
relative strength vs. NASDAQ.

Stage G computes Volatility Contraction Pattern signals numerically from
price/volume data first, then sends the chart image **plus** those computed
numbers to Claude (`claude-opus-4-8` by default) for a structured verdict
(`is_vcp_pattern`, `pivot_price`, `suggested_stop_loss`, `entry_recommendation`,
etc.). The model confirms and interprets the pattern rather than detecting it
from pixels. Stage G only analyses the already-ranked top candidates. It never
changes Composite Score or ranking, and nothing it outputs is a buy signal.

### VCP detector (Stage G, `stage_vcp_analysis.py`)

**Concept vs. heuristic.** The core concept is Minervini's: inside a Stage 2
uptrend, price builds *one* base under a common resistance area through
successive shallower pullbacks. Lows tend to rise, volume dries up, and the base
ends in a tight, quiet area just below a pivot that a breakout clears on
expanding volume. **Every number the detector uses is our own heuristic**, chosen
on synthetic structure, not a rule quoted from the book. Treat them as settings
to be judged against stored Stage-H history.

| Step | What the detector does |
|---|---|
| Swing structure | Close-based. **Confirmed** pivots are the highest/lowest close within ±`vcp_pivot_window_days` (5), so only bars with 5 bars on each side qualify. **Forming** pivots in the last 5 bars use only data up to today (a truncated window). Swings smaller than `vcp_min_swing_pct` (2%) are noise and pruned. |
| Contraction depth | Structure fixes the leg; measurement is strictly chronological (`_measure_leg`). First the **measured High**: the highest High from the bar before the swing high up to the bar before the swing low. Then the **contraction Low**: the lowest Low only in bars *after* both that High bar and the swing-high bar, up to the bar after the swing low (never past today). Depth = (High − Low) / High. The same bar never supplies both, a Low earlier than the High is never used, and if no later bar trades below the High no contraction is recorded. Each leg keeps `measured_high/low_idx/date/price` for auditing, and the chart draws the leg between those two points. A leg is `confirmed` only if both its pivots are. |
| Candidate base | Start from the most recent leg and add earlier legs while all their highs stay within `vcp_resistance_band_pct` (10%) of the base's highest high. The first leg outside the band (a lower step of an advancing trend, or a much higher old peak) ends the base, and it and all earlier legs are ignored. Shrinking pullbacks from unrelated parts of an uptrend therefore don't form one VCP. |
| Tightening | Leg-over-leg ratios, shrink fraction, largest expansion, last÷first. `strong`: every leg shrinks and last ≤ 0.5× first. `acceptable`: at most one violation, no leg above `vcp_max_expansion_ratio` (1.25×) of the prior, last ≤ 0.75× first. `not_contracting`: last ≥ first or most legs expand. Otherwise `weak`. So 24→14→8→4 is strong, 22→13→14→7 acceptable, 25→10→20→8 weak. |
| Structure | `contraction_low_prices`, `higher_low_fraction` (base coherence: all higher = strong, at least half = acceptable), `recovery_ratios` (share of each drop recovered), `resistance_dispersion_pct`. |
| Pivot | `pivot_price_candidate` = the highest High among the base's right-side contraction highs (the resistance being tested). A minor bounce high doesn't qualify, nor does an old high outside the base. `pct_from_pivot` (signed) → `pivot_state`: `far_below_pivot` (> `vcp_pivot_proximity_pct` 5% below), `coiling_below_pivot`, `fresh_breakout` (first close above within `vcp_final_zone_days`), `above_pivot`, `extended_above_pivot`. Claude may still pick a cleaner level. |
| Volume across contractions | `volume_leg_avgs` (mean volume over the pullback bars: from the bar after the measured High through the measured Low, so a high-volume breakout or gap bar at the peak never counts as selling), plus leg-over-leg ratios, shrink fraction, pairwise monotonicity and largest expansion. The classification uses the same rules as tightening, so 10M→5M→12M→4M is `weak` even though last÷first is 0.4. |
| Final tight zone | The last `vcp_final_zone_days` (5) sessions, ending the bar *before* a breakout so the breakout spike doesn't mask the dry-up. Reports average volume vs the prior 20/50 sessions, range %, and candle-range compression (median true range vs the prior 20 sessions). |
| Breakout demand | Today's volume and the breakout bar's volume vs the prior 20/50-day averages, which exclude that bar. `breakout_volume_confirmed` requires at least `vcp_breakout_volume_ratio` (1.5×). |
| Assessment | `assess_vcp_quality` reports each component (`tightening_quality`, `volume_dryup_quality`, `base_coherence`, `final_tightness` (final contraction ≤ 10%), `pivot_state`) and a combined `vcp_numeric_quality` (`strong`/`acceptable`/`weak`/`absent`) with plain-language `vcp_quality_notes`. Deterministic, no ML, no Claude. |

**Claude's role.** The prompt gives these numbers as ground truth and tells Claude
not to re-estimate depths or volume from pixels. It asks for what numbers can't
settle:
- whether the contractions visually form one coherent base, rather than shrinking numbers from unrelated swings;
- whether the base is constructive or loose and choppy;
- whether the tight area really sits just below the pivot;
- whether the candidate pivot is sensible;
- whether price above the pivot is a fresh breakout or a stale, extended move;
- whether the base fits a Stage 2 trend.

A forming right-edge contraction is flagged as provisional.

**Numeric detector first, Claude as a selective reviewer.** `VCP_LLM_REVIEW_MODE`
decides which candidates Claude sees when `VCP_LIVE_ANALYSIS=1`
(`VCP_LIVE_ANALYSIS=0` still means no Claude calls):

| Mode | Claude reviews |
|---|---|
| `selective` (default) | numeric `acceptable`/`weak` (the grey zone); `strong` at a fresh breakout; `strong` whose base has a contraction extreme set by a ≥3% intraday wick or a ≥5% session gap |
| `all` | every Stage-G candidate (research: collect numeric-vs-Claude data) |
| `off` | nobody (metrics + verification charts only) |

In selective mode, a strong setup coiling below its pivot with clean structure, and
an absent result, stand on the numbers. A forming final leg alone doesn't trigger
review, and wicks or gaps can't turn a clearly non-contracting structure into a VCP.
Routing reasons are passed to Claude and stored with the verdict. Skipped names
show "Claude: not requested", which is never a negative verdict. Routing thresholds
are ours and never change the numeric result. On 2026-09-14 this sent 8 of 20
candidates to Claude instead of 20.

What Claude receives and returns:
- **Input:** the plain Stage F candlestick chart (no detector annotations), the
  numbers, and each contraction's measured High/Low dates so it can find the
  legs on the chart. It is told it may disagree with `vcp_numeric_quality`.
- **Output:** a `VCP_SCHEMA` verdict, requested with `output_config.format` on
  `claude-opus-4-8` (adaptive thinking, `max_tokens` 16000), and validated
  again locally.
- **Failures degrade to that ticker's error row** and never fail Stage G: a
  refusal, a truncated response, invalid JSON, a schema violation, or
  contradictory fields.
- **Contradictory fields are rejected.** The prompt's consistency rules are:
  - `is_vcp_pattern=true` only if the rationale affirms one coherent VCP;
  - `false` means `pattern_stage` is `not_present` or `failed`, and the
    recommendation is not `buy_now`/`wait_for_breakout`;
  - `forming` only for a genuine VCP.

  A verdict that breaks these is stored as `semantic_inconsistency` with the raw
  answer, never silently fixed. A positive verdict whose rationale reads as a
  rejection is kept but flagged (`rationale_warnings`).
- **Diagnostics:** a Stage G log line counts routed/skipped names, calls,
  errors, SDK retries and tokens.

The numeric metrics (`vcp_metrics_json`) and Claude's verdict (`vcp_verdict_json`,
including the `review` record: mode, requested, reasons, agreement, warnings) are
stored separately. Neither overwrites the other.

To check verdict stability without touching history or Telegram, run
`python -m pipeline.vcp_eval CACC HALO --repeats 3`. It makes real Claude calls
(symbols × repeats) and writes `data/reports/vcp_eval_<ts>/results.json`.

**Verification chart** (`charts/<run_ts>/vcp_debug/<SYMBOL>.png`):
- shading marks the selected base;
- base pivots are filled, ignored pivots grey, forming pivots hollow orange;
- each contraction is labelled `T1 -24.4%`, with dashed forming legs;
- lines show the candidate pivot and the per-contraction, final-zone and 50-day volume averages;
- a metrics footer lists the numbers.

Use it to answer "did the detector pick the right base?"

**Metrics history.** `vcp_metrics_version` 2 rows add all of the above to
`vcp_metrics_json`. The 13 original keys remain, but are now computed over the
selected base with High/Low depths, a stricter `contractions_decreasing`, and a
base pivot. Older rows keep their original, smaller key set and are never
migrated.

Run it directly:

```bash
# one-shot throwaway container — nothing needs to be running first
docker compose run --rm --entrypoint python pipeline -m pipeline.run_pipeline
# or locally, from the project root, with dependencies installed:
python -m pipeline.run_pipeline
```

The pipeline code is baked into the image at build time, so rebuild after
editing anything under `pipeline/`. Only the `jupyter` service declares a build
context — `pipeline` and `runner` just reuse the image it produces, so
`--build pipeline` is a no-op and the build must be aimed at `jupyter`:

```bash
docker compose build jupyter
```

`docker exec stage2_pipeline ...` also works, but requires the long-lived
`stage2_pipeline` container to already exist. It's gated behind the `scheduled`
Compose profile, so it won't be created by a plain `docker compose up jupyter`.
If `docker exec` fails with `No such container: stage2_pipeline`, bring it up
first:

```bash
docker compose --profile scheduled up -d pipeline
```

That starts the long-running session-aware scheduler (see
[Scheduled operation](#scheduled-operation-stage-h)) — `docker exec` then lets you
trigger an on-demand run inside it without waiting for the next check.

Or let it run on a schedule (see [Scheduled operation](#scheduled-operation-stage-h)). Set
`ANTHROPIC_API_KEY` in `.env` first — Stage G is skipped with an error per
ticker if it's missing. `VCP_TOP_N` (default 20) controls how many top-ranked
candidates get the paid vision analysis each run.

### Logging

Every stage logs through Python's `logging` module (`pipeline/logging_config.py`)
instead of `print()` — each run writes to **both**:

- **stdout** — visible via `docker compose logs -f pipeline` or in your terminal
- **`data/logs/pipeline_<timestamp>.log`** — a full timestamped audit log per run
  (survives after container log buffers rotate away), including per-ticker drop
  reasons at `DEBUG` and full tracebacks for any exception

Every stage module logs through its own `logging.getLogger(__name__)`, nested
under the `pipeline` logger, so log lines are tagged by the stage that produced
them (`pipeline.data_sources`, `pipeline.stage_screen`, `pipeline.stage_rank`, etc.).

Control verbosity with `LOG_LEVEL` in `.env` (`DEBUG` / `INFO` default / `WARNING` / `ERROR`):

```env
LOG_LEVEL=DEBUG   # see why individual tickers were dropped at each stage
```

Outputs:
- `data/reports/watchlist_YYYYMMDD.csv` — full ranked watchlist (Stage E)
- `data/reports/vcp_analysis_YYYYMMDD.csv` — VCP verdicts for the top N candidates
- `data/charts/<run_timestamp>/<SYMBOL>.png` — the chart each verdict was based on
- `data/logs/pipeline_<timestamp>.log` — full audit log for that run (see [Logging](#logging))
- scheduled runs additionally write `data/history/stage2_history.sqlite3` and
  `data/runs/<trading_date>/run.json` (see [Scheduled operation](#scheduled-operation-stage-h))

### OHLCV price cache

To avoid re-downloading full price history for the whole ~4,300-ticker universe
every run, downloaded OHLCV is cached to Parquet (`pipeline/ohlcv_cache.py`,
one file per ticker under `data/ohlcv_cache/`). On each run `batch_download`
serves each ticker one of three ways:

- **cache hit** — cached data is fresh and deep enough → returned from disk, no network
- **delta** — deep enough but stale → fetch only the days since the last cached bar and append
- **full** — missing or not enough history → download the whole requested window

**Price basis: split- and dividend-adjusted, always.** yfinance is called with
`auto_adjust=True` everywhere in this project, so Open/High/Low/Close come back
back-adjusted for *both* splits and dividends (Volume for splits only), and the
newest bar is the real traded price. A split or dividend therefore retroactively
rescales all past prices; the delta merge re-fetches a few days of overlap and,
if it detects a uniform price shift, rescales the cached history onto the new
basis so the series stays continuous. Market caps are cached the same way
(`data/market_cap_cache.csv`, 1-week TTL) since they barely move week to week.

That basis is the contract the whole Stage-I corporate-action layer is built on
(`corporate_actions.PRICE_BASIS`): because the bars are already adjusted,
**nothing downstream may adjust a price a second time**. See
"[9. Corporate actions](#9-corporate-actions)" under Stage I.

A cached bar is only served as-is if it was written after its session closed
(+ `SESSION_CLOSE_BUFFER_MINUTES`). A bar cached while the US market was open is
intraday data, so the next run delta-fetches that ticker and the overlap replaces
the partial bar (`ohlcv_cache.last_bar_complete`, judged by the cache file's
write time).

The cache is on by default. Disable it (always fetch fresh) with `OHLCV_CACHE=0`
in `.env`, or delete `data/ohlcv_cache/` to force a clean rebuild. Tuning knobs
(`ohlcv_cache_max_age_days`, `ohlcv_cache_overlap_days`, `market_cap_cache_ttl_days`)
live in `pipeline/config.py`.

---

## Scheduled operation (Stage H)

`pipeline/scheduler.py` wraps the unchanged Stage A–G run with trading-session
awareness, idempotent history, Top-N trend monitoring and Telegram reporting.
The scheduled path and `python -m pipeline.run_pipeline` call the same
`run_pipeline.run_screen()`, so there is only one Stage A–G implementation.

```
scheduler wakes (startup, or the daily local trigger)
  → US session currently trading?               yes → stop (bars would be partial)
  → latest *completed* NASDAQ session (exchange calendar, not the local date)
  → already has a successful result?            yes → stop
  → process lock + SQLite claim                 held → stop (no duplicate work)
  → ^IXIC latest daily bar == target session?   retry, then DATA_NOT_READY
  → Stage A–G → persist full ranked list (atomic) → run.json → trends → Telegram
```

### Laptop operation

The current deployment is a **Windows laptop**, usually on between **09:00 and 23:00
Asia/Singapore**. There is no 24/7 server, so the design goal is to check
whenever the laptop is on and never lose or duplicate a session:

- **Daily check at 09:15 Asia/Singapore** (`LOCAL_SCHEDULER_TIME`,
  `LOCAL_SCHEDULER_TIMEZONE`), on **every calendar day, weekends included**.
- **Startup check** (`RUN_ON_STARTUP_CHECK=1`) as soon as the scheduler starts,
  so a late start (e.g. 11:37) still processes that day's session.
- The scheduler checks every day, but **Stage A–G only runs when a completed
  NASDAQ session has not yet been processed successfully**.
- The daily 09:15 check is the default and the recommended laptop setting. Other
  check frequencies are available (see below) for flexibility and testing.

### Scheduler check frequency

> **A more frequent scheduler check does not mean Stage A–G runs more often.**
> Every check goes through the same session/idempotency logic, so the pipeline
> still runs at most once per completed XNAS session. Later checks for that
> session just log "already has a successful result" and exit in milliseconds.

`SCHEDULE_MODE` picks when the long-running scheduler performs its main checks.
All modes use `LOCAL_SCHEDULER_TIMEZONE`, run every calendar day, and keep the
startup check (`RUN_ON_STARTUP_CHECK`).

| Mode | Settings | Main checks |
|---|---|---|
| `daily` (default; also used when `SCHEDULE_MODE` is missing) | `LOCAL_SCHEDULER_TIME=09:15` | once a day at 09:15 |
| `times` | `SCHEDULE_TIMES=09:15,13:00,18:00` | at each listed time |
| `interval` | `SCHEDULE_INTERVAL_MINUTES=60` (15–1440) | every N minutes after the previous check |

```env
# Recommended (normal laptop operation)
SCHEDULE_MODE=daily
LOCAL_SCHEDULER_TIME=09:15
LOCAL_SCHEDULER_TIMEZONE=Asia/Singapore

# Several fixed checks a day
SCHEDULE_MODE=times
SCHEDULE_TIMES=09:15,13:00,18:00

# A check every hour
SCHEDULE_MODE=interval
SCHEDULE_INTERVAL_MINUTES=60
```

Only the variables of the active mode are read. Invalid values stop the
scheduler with a clear error and exit code 2, and are never silently
reinterpreted: an unknown mode, an interval that isn't a whole number from 15 to
1440, a time that isn't `HH:MM` between 00:00 and 23:59, or a time listed twice.
`--dry-run` and `history_cli status` show the parsed schedule and the next check.
In `interval` mode the next check depends on the running loop, so they say so
rather than guess. Changing the frequency only needs an `.env` edit and a
scheduler restart. The Windows task (log on + 09:05 watchdog) stays the same.

**Three different clocks, easy to confuse:**

| What | Setting | Purpose |
|---|---|---|
| Main check frequency | `SCHEDULE_MODE` + time/times/interval | when a normal check (`check_and_run`) happens |
| Readiness retries | `DATA_READY_*`, `DEFERRED_RETRY_*` | while one session's data is late: short retries, then hourly probes until the cutoff |
| Internal wake loop | `SCHEDULER_POLL_SECONDS` (60) | the loop wakes every minute to notice due checks and sleep/resume; this does no work by itself |

**Startup.** The startup check runs immediately in every mode. The next main
check is the next occurrence after it ends. For a start at 11:00, that is 09:15
the next day (`daily`), 13:00 (`times` 09:15/13:00/18:00), or about 12:00
(`interval` 60). An occurrence that comes due while the startup check is still
running is covered by it, so the same moment is never checked twice.

**Sleep/resume.** Missed checks are never replayed. If the laptop sleeps
12:00–17:00 with `times` 09:15/13:00/18:00, it runs **one** catch-up check on
waking (for the missed 13:00) and then the normal 18:00 check. If it wakes at
19:00, the missed 13:00 and 18:00 collapse into one check. In `interval` mode
one catch-up check runs on waking and the interval restarts from there.

**Late data with frequent checks.** There is only ever one readiness episode
per session:
- While a deferred retry is pending, a main check for that session **joins the
  episode**. It does one readiness probe with no new claim, no new
  `DATA_NOT_READY` row and no extra alert, and the next retry is re-based from
  that probe. An hourly schedule therefore does not create a second hourly
  retry stream.
- After the `DEFERRED_RETRY_CUTOFF`, or after a `PIPELINE_FAILED` run, main
  checks leave that session **on hold for the rest of the local day**. They
  don't restart short retries, send more alerts, or rerun Stage A–G. The next
  local day, a scheduler restart, or a manual `--run-once` tries again.
- During US market hours, repeated checks record `DEFERRED_MARKET_OPEN` once per
  US session, not once per check.

### Trading date vs. execution date

Results are stored under the **NASDAQ session date**, not the Singapore date the
run happened on. The exchange calendar (`exchange_calendars`, XNAS, holidays
and early closes included) decides which session that is; there is no
"yesterday" arithmetic.

| Singapore time (check) | New York time | Session processed |
|---|---|---|
| Tue 09:15 | Mon 21:15 (EDT) | **Monday** |
| Sat 09:15 | Fri 21:15 | **Friday** |
| Sun 11:37 (Sat was missed) | Sat 23:37 | **Friday**, once |
| Mon 09:15 | Sun 21:15 | Friday, already done, so nothing runs |
| Tue 09:15 after a US holiday Monday | Mon 21:15 | the **previous Friday** (if not done) |
| Tue 22:00 | Tue 10:00 | nothing, because Tuesday's session is trading |

**Why Saturday matters:** Friday's US session closes at 04:00 or 05:00 Saturday,
Singapore time, so Friday's screen is normally produced on *Saturday* morning.
A Monday–Friday local schedule would never process Friday. A session counts as
completed `SESSION_CLOSE_BUFFER_MINUTES` (60) after its actual close, so
early-close days (13:00 ET) complete earlier. DST shifts are handled through the
calendar's UTC session times.

The **US market is open from 21:30 or 22:30 until 04:00 or 05:00 Singapore
time** (depending on US daylight saving), which overlaps the laptop's evening.
A check during those hours does **not** run Stage A–G, because upstream daily
data would contain a partial bar for the live session. If the latest completed
session is still unprocessed, a `DEFERRED_MARKET_OPEN` row is recorded (never
canonical, not a failure) and the log says
`US market currently open …; screening of <session> deferred to next safe check.`
The next safe startup or 09:15 check resolves the latest completed session
again. By then a newer session has usually closed overnight, so that newer
session is processed, and the deferred one remains a recorded gap under the
one-session rule below.

### Startup catch-up and the one-session limit

- Late start with the session missing: the startup check runs it once, and the
  09:15 check later that day finds it done and does nothing.
- Laptop off for a week: only the **latest** completed session is processed.
  Earlier missed sessions are logged as gaps and **never replayed automatically**.

This limit is built into how the pipeline works, so it is not a setting. Stage
A–G always screens the *latest* bars the data provider exposes, so a historical
session cannot be re-screened correctly: a "backfill" would produce today's
screen labelled with an old date. For the same reason there is no backfill
command.

### Data readiness and deferred retries

Before Stage A–G, one lightweight download checks that `^IXIC`'s latest daily
bar is the target session. A bar *newer* than the target (a session already
trading) also blocks the run.

1. **Short retries.** The check retries every `DATA_READY_RETRY_MINUTES` (10), up
   to `DATA_READY_MAX_ATTEMPTS` (3). With a 09:15 check that means attempts at
   09:15, 09:25 and 09:35. If none succeeds, that attempt is recorded as
   `DATA_NOT_READY` and **no screen is generated**.
2. **Deferred retries** (`DEFERRED_RETRY_ENABLED=1`, long-running scheduler only).
   The session stays pending and is re-probed every `DEFERRED_RETRY_MINUTES` (60),
   at 10:35, 11:35 and so on. Each probe is a single readiness download that
   writes no run row. Stage A–G starts only once a probe finds the data ready,
   runs exactly once, and cancels any further retries for that session. A probe
   that finds the session already completed (for example by a manual run) just
   logs `Deferred target session … already completed; no action required.`
3. **Cutoff.** Retries stop at `DEFERRED_RETRY_CUTOFF` (20:30 local time, before
   the US open). The last retry is moved up to the cutoff itself. The log then
   says `Target session … still not ready at 20:30 SGT. Deferring until the next
   safe scheduler/startup check.`, and one Telegram ops alert is sent (rather
   than one per attempt). The session is **not** marked permanently failed: any
   later safe startup or daily check resolves and retries it.
4. **Sleep.** If the laptop sleeps through one or more retry times, the scheduler
   runs a **single** catch-up probe on waking (`Missed deferred retry due …`),
   not one per missed hour. If it wakes after the cutoff, it drops the pending
   retry instead.

`--run-once` performs only the short retries; deferred retries belong to the
long-running loop. A `PIPELINE_FAILED` run is not retried the same day, so
Stage A–G never repeats automatically.

### Idempotency, locking and `--force`

- **One canonical result per trading session**, enforced by a partial unique
  index in SQLite. A run becomes canonical only in the same transaction that
  writes all of its ranked rows, so a crash never leaves a partial "success".
- **One active attempt per session**: a second unique index rejects a second
  `WAITING_FOR_DATA`/`RUNNING` claim. Claims older than `RUN_STALE_AFTER_HOURS`
  (6) are marked `ABANDONED` (e.g. after a crash or reboot mid-run).
- **OS file locks**: `data/history/pipeline_run.lock` is held around each
  attempt, and `scheduler_instance.lock` is held by the long-running loop (a
  second loop exits). The OS releases both if the process dies.
- `--force` re-runs a session that already succeeded, and the new run replaces
  the old one atomically (recorded in `superseded_by_run_id`). It does not
  bypass the data-readiness or market-open guards.

Run states: `WAITING_FOR_DATA → RUNNING → RESULTS_PERSISTED → NOTIFIED` (or
`NOTIFICATION_FAILED`). Three different outcomes produce no result:
`DATA_NOT_READY` (the provider is late and the session stays pending),
`DEFERRED_MARKET_OPEN` (running now would be unsafe and the session stays
pending), and `PIPELINE_FAILED` (Stage A–G broke). A crashed claim becomes
`ABANDONED`. **Telegram success never decides pipeline success.** An empty
ranked result is treated as `PIPELINE_FAILED` (it is almost always a data
outage), so it is retried instead of being stored as a successful empty day.

### Commands

```bash
# existing manual run: unchanged, no history or Telegram
python -m pipeline.run_pipeline

# scheduled-style one-shot: resolve session → skip if done → readiness → A–G → persist → notify
python -m pipeline.scheduler --run-once
python -m pipeline.scheduler --run-once --force     # deliberately re-run the latest session
python -m pipeline.scheduler --dry-run              # show what a check would do right now
python -m pipeline.scheduler --resend               # re-send the latest stored report (no A–G)
python -m pipeline.scheduler --resend 2026-09-14
python -m pipeline.scheduler --test-telegram        # one connectivity-test message; no run, no history
python -m pipeline.scheduler                        # long-running loop (what the task/container runs)
```

On Windows, run these through the wrapper so `.env` is loaded:
`powershell -ExecutionPolicy Bypass -File scripts\start_scheduler.ps1 --run-once`.

### History (SQLite)

`data/history/stage2_history.sqlite3` (`HISTORY_DB_PATH`) is the canonical
long-term store; the CSV, PNG and JSON artifacts remain the human-readable
outputs.

| Table | One row per | Holds |
|---|---|---|
| `pipeline_runs` | attempt | `trading_date`, status, `is_canonical`, trigger, UTC start/end, local timezone, data-readiness result, `git_commit` (+dirty), `config_hash`, redacted `config_json`, library versions, stage counts, artifact paths, failed stage + error |
| `stock_results` | ranked symbol per run | **the full ranked population** (not just the Top N): rank, composite/technical/fundamentals score, sector, market cap, last close, RS, Stage-G verdict fields, `vcp_metrics_json`, `vcp_verdict_json`, chart paths, and the complete Stage E row as `ranking_json` |
| `notification_delivery` | send attempt | channel, kind (`report`/`chart`/`ops_alert`), status, attempts, redacted error, Telegram message ids |

`config_hash` covers screening parameters only (Telegram, path and scheduling
settings are excluded). That lets you tell whether a rank changed because the
screen changed or because the market changed. Secrets are never stored.

Stage counts recorded per run: universe, liquidity, market cap, sector
classified, top sectors (C), Stage D screened, trend template 8/8, fundamentals
scored, ranked, charts, VCP analysed, VCP pattern, and VCP
buy_now/wait_for_breakout. A sudden drop (e.g. 8/8 passes falling from 80 to 3)
usually means a data problem.

Each run also writes `data/runs/<trading_date>/run.json` (run id, status,
timestamps, commit, config hash, stage counts, artifact paths, Telegram status).

Inspect history without opening SQLite:

```bash
python -m pipeline.history_cli runs                    # recent attempts and their status
python -m pipeline.history_cli top --sessions 10       # Top-N rank matrix across sessions
python -m pipeline.history_cli symbol NVDA             # one symbol's rank/score/VCP history
python -m pipeline.history_cli status                  # latest session vs latest success; would a check run now
```

### Top-N trends

For each current Top-N symbol (`TREND_TOP_N=10`, `TREND_LOOKBACK_SESSIONS=10`),
history provides: current and previous rank, 1- and 5-session rank change,
composite score change over 1 and 5 sessions, consecutive Top-N streak,
appearances in the lookback window, new entries, drop-outs, and VCP verdict or
confidence changes. All comparisons step through **recorded trading sessions**.
Weekends and holidays are never sessions, so they can't break a streak; an
unrecorded session is reported as a gap.

Categories follow fixed rules (first match wins; no LLM involved):

| Label | Rule |
|---|---|
| 🚀 Rising | rank improved by ≥ `TREND_RANK_MOVE_MIN` (3) vs the previous session |
| ⬇ Falling | rank worsened by ≥ 3 |
| 🔥 Persistent | in Top-N for ≥ `TREND_PERSISTENT_MIN_STREAK` (3) consecutive sessions |
| 🆕 New | in Top-N now, not in the previous session |
| 🚪 Left | in Top-N last session, not now (listed with its new rank) |

These describe how the list persists and changes. They are not predictions or
buy/sell signals.

### Telegram

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Send your bot a message, then open
   `https://api.telegram.org/bot<token>/getUpdates` in a browser and read your
   chat id from `message.chat.id`.
3. Put both in your local `.env` (never in code, docs or tests):
   ```env
   TELEGRAM_ENABLED=1
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```

Once results are saved and trends are calculated, two separate text messages are
sent: a decision-making **dashboard** and an optional **debug summary**.

**The dashboard** answers: what is ready, what to watch, what needs review, what
I already hold, and what to do next. Top to bottom:

```
🎯 ACTION BOARD
🟢 Ready: MEDP
🟡 Watch: INCY
🟠 Review: FRHC, HALO
💼 Open positions: 2 (1 need attention)

🟢 READY — MEDP                       <- the persisted Stage-I plan, verbatim
Rank #5 · Healthcare · 13/14
VCP: STRONG · 16.9 → 8.3% (forming)
Price: 620.48 · Pivot: 629.69 · -1.5%
📌 Entry plan   Trigger / Max chase / Breakout volume / Status
🛡 Risk plan    Initial stop / Required vs allowed risk / ATR
📦 Position plan  Suggested shares · portfolio % / binding constraint
Action: WAIT FOR BREAKOUT

🟡 WATCH — …  (Why watch: ✓/⚠ lines from the stored detector components)
🟠 REVIEW — … (Why review: the numeric/Claude disagreement, Claude's rationale)
💼 OPEN POSITIONS   (HOLD compact; WATCH/EXIT/STOP/corporate-action review in detail)
📊 OVERALL LEADERS  (#1 SLDE · 14/14 · Financials · ⚪ No VCP — Composite order)
📈 WATCHLIST CHANGES (🆕 New · ⬆ Up · ⬇ Down · 🚪 Left Top N · 🔥 Persistent)
```

The buckets are read from the **persisted** `setup_plans` of the run — the
formatter never recomputes an entry, a stop or a size, and adds no score:

| Bucket | Rule |
|---|---|
| 🟢 Ready | `READY` / `BREAKOUT_TRIGGERED` with a usable plan: trade plan `OK`, entry status still inside the plan, >0 suggested shares |
| 🟡 Watch | a numeric setup that is not actionable yet (below the entry range, `MISSED_EXTENDED`, or a plannable setup whose plan is `SKIP_RISK_TOO_WIDE` / `NO_PRICE_DATA` / 0 shares) |
| 🟠 Review | `MANUAL_REVIEW`, or a numeric/Claude disagreement |
| ⚪ No VCP | ranked, but no VCP worth acting on |

Within a bucket the order is the VCP priority of `vcp_review_set` (strong →
acceptable → disagreement → other Claude-positive), then Composite rank.
Numeric and Claude results are never merged into one score. Messages are split
below 3,800 characters without breaking a stock entry across messages.

**Buttons** (need `TRADE_BOT_ENABLED=1` and the separate `trade_bot` process):
a READY setup gets `🟢 Open Position`, or `💼 View Position` when an OPEN trade
already exists; each held position gets `💼 View Position`. MANUAL_REVIEW,
NOT_READY, SKIP_RISK_TOO_WIDE and MISSED_EXTENDED never get an Open Position
button. A READY setup whose verification chart is sent carries its button on
the chart; the rest sit under the last report message, so no button appears
twice. See [Open Position](#telegram-trade-journal) below.

**The debug summary** (`TELEGRAM_DEBUG_SUMMARY=1`, default) is a separate
`🛠 DEBUG — DAILY PIPELINE` message after the charts: the Stage A–E counts, the
Stage-G counts (`VCP metrics computed`, `Numeric VCP positive`, `Claude
reviewed`, `Claude VCP positive`, `VCP review union`, `Numeric vs Claude: N agree
· M disagree` — only names Claude actually reviewed), the Stage-I gate states
(READY / BREAKOUT_TRIGGERED / NOT_READY / MANUAL_REVIEW / MISSED_EXTENDED) and
plan statuses (`Trade plan OK`, `SKIP_RISK_TOO_WIDE`) — which answers "the VCP
union is 4, so why is nothing READY?" — the position-monitor states, the
corporate-action counts and run facts (run id, status, runtime, data readiness,
Claude mode, config hash, main-report status). It never contains credentials,
raw environment values, config JSON or local paths. A failed debug send is
recorded as a `debug` notification and never changes the run's outcome;
`TELEGRAM_DEBUG_SUMMARY=0` turns it off.

With `TELEGRAM_SEND_CHARTS=1`, the Stage-G **VCP verification charts** of the VCP
review set are sent (up to `TELEGRAM_VCP_CHART_MAX`, 10, in VCP priority order).
These are the existing `vcp_debug/` PNGs; nothing is re-rendered or downloaded.
Each caption starts with the dashboard bucket; a READY caption carries the
stored price, pivot, entry trigger, initial stop and suggested size, the others
the numeric line, Claude's line and `Agreement: YES / DISAGREE` where Claude
reviewed it.

Sends are retried with backoff, honouring Telegram's `retry_after`. If Telegram
still fails, the run stays successful (`NOTIFICATION_FAILED`) and the report can
be re-sent with `--resend`. With `TELEGRAM_OPS_ALERTS=1`, `PIPELINE_FAILED` and
`DATA_NOT_READY` also send a short alert with no secrets or stack traces. The
bot token is never logged: it is removed from error messages and filtered out
of every log handler.

### Windows Task Scheduler setup

1. Create the venv and install dependencies (once):
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
2. `copy .env.example .env`, then fill in `ANTHROPIC_API_KEY`, the Telegram
   settings, and `VCP_LIVE_ANALYSIS` (1 = spend tokens on Stage G, 0 = token-free).
3. Check it works without running anything:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\start_scheduler.ps1 --dry-run
   ```
4. Register the task. It runs as you at log on, plus a daily 09:05 trigger that
   restarts the scheduler if it isn't running (a second copy is ignored):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\register_windows_task.ps1
   Start-ScheduledTask -TaskName 'Stage2 NASDAQ Scheduler'
   ```
   To remove it: `scripts\register_windows_task.ps1 -Unregister`.

   To set it up by hand in Task Scheduler instead: *Create Task* → *Triggers*:
   "At log on" (your user) and "Daily 09:05" → *Actions*: `powershell.exe` with
   arguments `-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "<repo>\scripts\start_scheduler.ps1"`,
   *Start in* `<repo>` → *Settings*: "Run task as soon as possible after a
   scheduled start is missed", "If the task is already running: Do not start a
   new instance", and uncheck "Stop the task if it runs longer than".

The task only keeps the process running; all calendar, catch-up and
idempotency decisions happen in Python. The 09:05 trigger uses Windows' own
clock and timezone. It's only a watchdog: if the loop is already running, the
new task instance is ignored, and a stray second copy exits 0 on the instance
lock. The 09:15 Singapore check itself (or whatever `SCHEDULE_MODE` configures)
is always timed by Python. Don't add extra Windows tasks for more checks. The loop wakes every minute and
compares the clock with the next due check, so a laptop that slept through
09:15 checks as soon as it wakes.

### Docker alternative

The `pipeline` Compose service runs the same `python -m pipeline.scheduler`
loop (Docker Desktop must be set to start at login):

```bash
docker compose build jupyter                        # after pulling these changes (new dependencies)
docker compose --profile scheduled up -d pipeline
docker compose logs -f pipeline
```

Run the scheduler in **one** place only: the Windows task *or* the container,
not both. File locks are not reliably shared across the Docker Desktop bind
mount (the SQLite claim still prevents duplicate canonical results). A future
always-on server would only need a different `LOCAL_SCHEDULER_TIME` /
`LOCAL_SCHEDULER_TIMEZONE` (e.g. `18:00` `America/New_York`); session
resolution, storage, trends and Telegram stay the same.

### Logs

- `data/logs/scheduler_<start>.log`: the scheduler's decisions, including the
  local/New York/UTC clock, resolved session, skip reasons, readiness attempts,
  storage, trends, Telegram sends and retries, and the next scheduled check.
- `data/logs/pipeline_<run_ts>.log`: the Stage A–G audit log for each run (same
  timestamp as `data/charts/<run_ts>/`).

### Limitations

- No historical backfill (see above); a missed session stays a gap.
- A partial bar cached by a *manual* run during US market hours is detected
  (cache file written before that session's close + buffer) and re-fetched by
  the next run. The check relies on the cache file's modification time, so
  copying `data/ohlcv_cache/` with a tool that resets timestamps can make an
  old partial bar look complete; delete the cache after such a copy.
- Stage F/G still download the top-N candidates again instead of reusing the
  cached Stage-D bars (see CLAUDE.md). This is left unchanged so VCP numbers
  stay identical.
- `run_vcp_only` still reads charts from `data/charts/<SYMBOL>.png` and does not
  write to history.

---

## Trade journal and position monitor (Stage I)

Stage I is the advisory layer between "the screener found a setup" and "I am
holding a position". It closes the loop:

```
VCP setup ready  ->  Layer 2: entry plan  ->  you trade, manually, elsewhere
                 ->  you record the fill (Telegram or CLI)
                 ->  corporate actions keep the journal on the right share basis
                 ->  Layer 3: stop / position / winning-exit monitoring
                 ->  Telegram advisory alerts
```

### Advisory-only boundary

```
NO brokerage API      NO IBKR credentials    NO account access
NO order placement    NO automatic execution
```

The app only **records fills you supply**, **calculates plans**, **monitors
public market data** and **sends alerts**. Nothing in `pipeline/` imports a
broker client or contains an order-placement code path, and
`tests/test_stage_i_integration.py` asserts exactly that (imports, call names
and `requirements.txt`). Telegram wording is deliberately advisory: no message
ever says an order was placed.

Stage I is also strictly downstream of Stage G: it never changes screening,
scoring, ranking, VCP metrics, pivots or Claude routing, and a failure in this
layer never fails a pipeline run. Its config keys (`trade_*`) are excluded from
`run_metadata.config_hash` for the same reason.

> **Every threshold below is one of our provisional heuristics**, chosen from
> structure/volatility reasoning and not fitted to any outcome. They are not
> rules prescribed by Minervini — the Minervini-style *principles* (cut failed
> trades quickly, never widen a stop, protect profits progressively, give real
> leaders room) are what the design follows. Re-evaluate the numbers once the
> journal holds real trade history.

### 1. The VCP-ready gate (`setup_gate.py`)

Runs over the *stored* Stage-G rows of each canonical run and records one state
per analysed symbol:

| State | Meaning |
|---|---|
| `NOT_READY` | numeric quality not strong/acceptable, or the pivot state is unusable |
| `MANUAL_REVIEW` | numerically clean, but Claude reviewed it and returned `is_vcp_pattern=false` |
| `READY` | strong/acceptable and coiling below the pivot |
| `BREAKOUT_TRIGGERED` | strong/acceptable at a fresh breakout |
| `MISSED_EXTENDED` | already extended above the pivot — no chase |

```
numeric quality in {strong, acceptable}
AND pivot_state in {coiling_below_pivot, fresh_breakout}
AND NOT far_below_pivot / extended_above_pivot
AND NOT (Claude actually reviewed it and said "no VCP")
```

"Claude not requested" (selective review skipped it, review off, token-free
mode) is **not** a negative verdict and never blocks a clean numeric setup. A
real disagreement parks the setup in `MANUAL_REVIEW` instead of silently
dropping or promoting it. The underlying VCP verdict is never altered.

Layers 2 and 3 are generated only for `READY` / `BREAKOUT_TRIGGERED`.

### 2. Layer 2 — entry plan (`entry_plan.py`)

Deterministic, per plannable setup:

| Field | Source |
|---|---|
| `pivot_price` | the **numeric** detector's `pivot_price_candidate` — always authoritative |
| `entry_trigger_price` | pivot + `ENTRY_TRIGGER_BUFFER_PCT` (default 0.1%) |
| `maximum_chase_price` | pivot + `MAX_CHASE_PCT` (default 5%) |
| `distance_to_pivot_pct` | current close vs pivot |
| `breakout_volume_ratio` | Stage-G `breakout_volume_vs_20d` |
| `required_breakout_volume_ratio` | the existing `vcp_breakout_volume_ratio` (1.5) |
| `entry_status` | `AWAIT_BREAKOUT` / `BREAKOUT_ACTIONABLE` / `BREAKOUT_VOLUME_UNCONFIRMED` / `EXTENDED_NO_CHASE` / `NO_PIVOT` |
| `claude` | the stored verdict, for context only |

Claude's own `pivot_price` and `suggested_stop_loss` are carried alongside but
**never** become the entry or stop price.

### 3. Layer 3 — initial risk plan (`risk_model.py`)

Structure first, volatility as the floor, market cap only as a modifier of the
*ceiling*:

```
structural_stop           = final contraction low - STRUCTURAL_STOP_BUFFER_PCT
structural_risk_pct       = (entry - structural_stop) / entry
volatility_required_risk  = ATR_STOP_MULTIPLE x ATR20%
required_risk_pct         = max(structural_risk_pct, volatility_required_risk)
allowed_max_risk_pct      = market-cap tier, x illiquidity factor, capped by MAX_RISK_PCT

required > allowed  ->  trade_plan_status = SKIP_RISK_TOO_WIDE
```

A stop that is too wide is a **skip** — never a widened stop, and never one
tightened back into the noise. Market-cap tiers (`trade_market_cap_tiers`,
`(floor, max_risk_pct, trail_atr_multiple)`): >=$200B -> 7% / 2.0x, >=$10B ->
8% / 2.5x, >=$2B -> 10% / 3.0x, unknown -> 8% / 2.5x. Thin liquidity multiplies
the ceiling by `ILLIQUID_CEILING_FACTOR`. Actual volatility outranks market cap
for trailing room: ATR% at or above `HIGH_VOLATILITY_ATR_PCT` adds
`HIGH_VOLATILITY_TRAIL_BONUS` to the multiple, whatever the tier.

### 4. Position sizing

```python
risk_budget        = portfolio_value * risk_per_trade_pct
risk_based_shares  = risk_budget / (entry_price - stop_price)
portion_budget     = portfolio_value * max_position_portion
portion_based      = portion_budget / entry_price
suggested_shares   = floor(min(risk_based_shares, portion_based))
```

`PORTFOLIO_VALUE`, `RISK_PER_TRADE_PCT` and `MAX_POSITION_PORTION_PCT` are
planning parameters you set in `.env`. No brokerage balance is read or
required. A `SKIP_RISK_TOO_WIDE` plan is never sized.

### 5. The ledger (`trade_store.py`)

A **separate SQLite file** — `data/trades/trade_journal.sqlite3` — so real
trades never share tables (or a database) with screening history. The only link
is the recorded `setup_run_id` / `setup_date`.

| Table | Contents |
|---|---|
| `setup_plans` | per run + symbol: gate state, entry plan, risk plan, sizing |
| `trades` | one row per position: status, remaining shares, average cost, realised P&L, initial stop, planned portion, setup linkage, notes |
| `trade_fills` | every partial buy/sell, in order |
| `position_snapshots` | one row per open trade per session (all monitored numbers + state) |
| `trade_events` | OPEN / BUY / SELL / CLOSE / MONITOR_STATE audit trail |
| `pending_actions` | unconfirmed Telegram actions — the only place they live |

Partial buys re-average the cost; partial sells book realised P&L against that
average and leave it unchanged; selling everything closes the trade. A partial
unique index allows at most one OPEN trade per symbol, so "sell CACC" is never
ambiguous — and the symbol can be traded again after that trade is closed.

### 6. Telegram trade journal

A separate long-poll process (it does not touch the scheduler):

```powershell
scripts\start_trade_bot.ps1            # or: python -m pipeline.trade_bot
python -m pipeline.trade_bot --once    # drain queued updates and exit
```

```
/buy CACC 622.50 20      record a buy fill
/sell CACC 680 10        record a partial sell
/close CACC 705          sell everything still held
/positions               open positions + monitor state
/position CACC           one position in detail
/trades                  recorded trades
/help
```

`/buy` automatically attaches the latest recorded setup (pivot, suggested stop,
VCP quality, `setup_run_id`). Every mutation is a two-step — nothing is written
until **Confirm**:

```
🟢 Record new position?

CACC
Buy: 20 @ $622.50
Value: $12,450.00

Setup: STRONG VCP · READY (2026-09-15)
Pivot: $621.52
Suggested initial stop: $594.20 (4.6% risk)
Planned portfolio portion: 12.5%

Journal entry only — no order is placed.
[ ✅ Confirm ]  [ ✖ Cancel ]
```

Confirm is claimed with a conditional UPDATE, so a second tap (or a re-delivered
callback) writes nothing and answers "Already handled." Unconfirmed actions
expire after `TRADE_CONFIRM_TTL_MINUTES`.

**Authorization.** `TELEGRAM_ALLOWED_CHAT_ID` (comma-separated; falls back to
`TELEGRAM_CHAT_ID`) plus the optional `TELEGRAM_ALLOWED_USER_ID` form an
explicit allowlist. Updates from any other chat are ignored entirely — no write,
no reply, just a log line naming the chat id. Tokens are never logged.

<a id="telegram-trade-journal"></a>**Open Position / View Position.** With
`TRADE_BOT_ENABLED=1`, a READY setup carries `🟢 Open Position`. It means
"record the position I actually took in the local journal" — never "place an
order". Tapping it asks `Actual fill price?`, then `Number of shares?` (or reply
`631 20` at once), then shows the usual Confirm/Cancel preview: the persisted
Stage-I plan (entry trigger, max chase, suggested stop, suggested size, planned
portion) beside the actual fill (slippage from trigger, risk to the stop, actual
portion) and any planned-vs-actual warnings — fill below the breakout trigger,
above the max chase price, more shares than suggested, a larger portfolio
portion than planned. Warnings never block an explicit Confirm. Nothing is
written to `trades` / `trade_fills` before Confirm, a duplicate Confirm writes
nothing, and the preview says "Journal entry only — no brokerage order is
placed."

The planned trade is kept apart from the actual one: a new trade stores the
plan's `entry_plan_json` / `risk_plan_json` (+ sizing), and its OPEN event keeps
a `planned` / `actual` / `warnings` record. The `setup_plans` row is never
modified. If an OPEN trade already exists the button is `💼 View Position`
(identical to `/position SYMBOL`); there is never a second OPEN trade for a
symbol, and further buys go through `/buy`. The older `Record Buy` button on
earlier messages starts the same flow.

### 7. Daily position monitor

Part of the normal Stage-H run, **independent of today's screen**: every open
position is monitored even if the symbol dropped out of Stage G entirely. Bars
come through the existing cached `batch_download`. Per position: close, shares,
average cost, unrealised P&L%, highest price since entry, MFE, MAE, ATR20 /
ATR%, EMA10, EMA21, SMA50, recent swing low, distance from the initial stop and
from the current protective level, and days since entry.

The protective level is
`structure + ATR allowance + market-cap/liquidity modifier + profit cushion`,
with two hard rules: it never moves down, and it is never tighter than the
stock's own ATR room unless it was already there. Cushions unlock progressively
tighter references (breakeven -> swing low -> EMA21-ATR -> EMA10-ATR).

| State | Fires when |
|---|---|
| `STOP_TRIGGERED` | close at/below the protective level (or the initial stop) |
| `EXIT_REVIEW` | failed breakout back below the pivot soon after entry, structure break (below EMA21 **and** the prior swing low), or an abnormal high-volume reversal below EMA10 |
| `PARTIAL_PROFIT_REVIEW` | large cushion **and** a climactic extension above EMA10 |
| `TIGHTEN_PROTECTION` | protection just ratcheted up, or a winner closed below EMA10 |
| `WATCH` | below EMA10 but structure intact |
| `HOLD` | nothing to do |

There is deliberately no universal "take profit at +20%" rule: a cushion alone
never produces a sell suggestion.

The daily report gains a section after the screener/VCP report:

```
💼 Open positions

CACC
30 shares @ 625.00
Close: 671.40 · +7.4%
High since entry: 682.70
ATR: 2.2%
EMA10: 659.30
EMA21: 642.70
Initial stop: 594.20
Protective level: 648.10
Status: HOLD

⚠ HALO
10 shares @ 50.00
Close: 56.15 · +12.3%
close below EMA10; still above EMA21 (54.00) and above the recent swing low (52.00)
Status: WATCH

🔴 CACC — EXIT REVIEW
30 shares @ 625.00
Close: 590.10 · -5.6%
Reason:
• structure break: close below EMA21 (612.40) and the recent swing low (598.00)
Suggested action: review position

Advisory only — this app records the fills you report and monitors public
market data; it never places, modifies or cancels an order.
```

### 8. CLI fallback

For when Telegram is unavailable — same service layer, same confirmation rule:

```powershell
python -m pipeline.trade_cli add CACC --price 622.50 --shares 20
python -m pipeline.trade_cli sell CACC --price 680 --shares 10
python -m pipeline.trade_cli close CACC --price 705
python -m pipeline.trade_cli positions
python -m pipeline.trade_cli show CACC
python -m pipeline.trade_cli trades --all
python -m pipeline.trade_cli plan            # recorded gate states + plans
python -m pipeline.trade_cli monitor         # re-run the monitor now
python -m pipeline.trade_cli actions CACC    # corporate actions (also --all)
python -m pipeline.trade_cli record-action CACC --type SPLIT --date 2026-10-01 --ratio 2
python -m pipeline.trade_cli resolve CACC    # clear a corporate-action review
```

`--yes` skips the interactive confirmation for scripted use.

### 9. Corporate actions

`pipeline/corporate_actions.py` keeps the journal honest when the *share basis*
or the *cash* changes underneath a position. It runs inside the daily cycle,
before the position monitor:

```
screening → setup plans → corporate-action detection
          → safe corporate-action adjustment → position monitor → Telegram report
```

Every open position is checked whether or not today's screener saw it, and each
symbol is probed at most once per session (`corporate_action_checks`), so
repeating the cycle costs no extra Yahoo calls.

#### One basis, never adjusted twice

| What | Basis |
| --- | --- |
| Technical OHLCV | split- **and** dividend-adjusted (`auto_adjust=True`) |
| Your recorded fills | raw as-traded price and quantity — **immutable** |
| Stored pivots / stops | the adjusted basis of the run date, i.e. as-traded |
| Price P&L | latest real close vs. your raw average cost |
| Corporate-action adjustments | the journal's *derived* values only |

Because Yahoo has already removed every ex-dividend drop from the series, the
monitor adds nothing back: `dividend_price_allowance` returns `0.0` for the
adjusted basis, and only a (hypothetical) raw-price source would get a
compensation — in exactly that one place. `test_corporate_actions.py` asserts
both halves, so a future change cannot quietly start double-counting.

#### What is adjusted automatically

Only arithmetic that is unambiguous:

| Action | Effect |
| --- | --- |
| `SPLIT` / `REVERSE_SPLIT` | `shares × ratio`, and average cost, initial stop and pivot `÷ ratio` |
| `STOCK_DIVIDEND` | the same share-basis restatement (a 10% dividend is ratio 1.10) |
| `CASH_DIVIDEND` / `SPECIAL_DIVIDEND` | books `dividend_income`; shares, fills and average cost untouched |
| `SYMBOL_CHANGE` | the trade keeps its id, fills, entry date, P&L and setup linkage; only the symbol used for future market data changes |

A 2-for-1 on 20 shares at \$620 with a \$590 stop becomes 40 shares at \$310
with a \$295 stop: economic value identical, realised P&L already earned
untouched, the original fill still reading `BUY 20 @ 620.00`. A protective level
recorded *before* the split is rebased forward when the monitor carries it
(otherwise yesterday's 96.0 would read as an instant stop-out against today's
51.0 close); the stored snapshot itself is never rewritten.

#### What is flagged instead of guessed

`MERGER`, `SPINOFF`, `RIGHTS`, `DELISTING` and anything unclassified are
recorded and set the position to `CORPORATE_ACTION_REVIEW`, which outranks every
technical state — so no failed-breakout, stop-out or structure-break conclusion
is drawn while the action is unresolved. No cost-basis allocation, exchange
ratio or new share quantity is ever invented. A cash acquisition with final
terms reaches `AWAITING_CONFIRMATION` and tells you the `/close` to record; it
is never applied on its own.

Two more cases deliberately refuse to guess:

- **a fill dated on the ex-date** — entitlement cannot be resolved from what we
  store, so the dividend is recorded for review instead of estimated;
- **a special dividend** — the distribution rebases the price series against a
  stop and pivot recorded on the pre-distribution price. The cash is booked, and
  the stop is **not** moved; the position is held in review so you decide.

Missing market data is never read as a sale: after
`MISSING_BARS_REVIEW_SESSIONS` consecutive empty sessions a `DELISTING` action
is recorded and the position goes into review.

#### Idempotency

Each event has a `fingerprint` (symbol, date, type, amount) with a unique index,
so re-detecting it inserts nothing; applying it is a conditional status `UPDATE`
in the same transaction that mutates the trade, so a duplicate apply writes
nothing at all. Running the daily cycle ten times leaves the same share count,
the same average cost and the same dividend income.

#### Audit trail

```powershell
python -m pipeline.trade_cli actions CACC     # one symbol
python -m pipeline.trade_cli actions --all    # everything recorded
python -m pipeline.trade_cli show CACC        # fills + actions + sales together
python -m pipeline.trade_cli record-action ABC --type SYMBOL_CHANGE --date 2026-10-01 --new-symbol XYZ
python -m pipeline.trade_cli resolve CACC --stop 295   # clear a review with YOUR number
```

`record-action` covers what Yahoo does not expose (ticker changes, mergers) and
previews before writing, like every other mutating command. The invariant the
audit trail protects is:

```
original fills + corporate actions + sales = the current position
```

`/position SYMBOL` and the daily report add price P&L, dividend income, total
P&L and the actions since entry — but only for positions that have actually seen
one, so an ordinary position reports exactly as before.

### Stage I limitations

- The thresholds are unvalidated placeholders: there is no trade history to tune
  them against yet, and they were deliberately **not** fitted to the handful of
  VCP examples on hand.
- The monitor runs once per completed session, on daily bars. An intraday stop
  breach is only seen at the next close, and every state is advisory.
- The Open Position / View Position buttons need the bot process running;
  without `TRADE_BOT_ENABLED=1` they are not attached.
- Corporate-action detection covers what yfinance exposes: splits and dividends.
  Ticker changes, mergers, spin-offs, rights and warrants have to be entered with
  `trade_cli record-action`, and only cash terms are ever acted on. Cost-basis
  allocation for a spin-off or a stock-for-stock exchange is not computed at all.
- A corporate action is applied to the *open* trade only. Closed trades keep
  their recorded history unchanged, which is the point, but it means a dividend
  whose ex-date falls inside a position you have since closed is not booked.
- The special-dividend threshold is another of our heuristics; a genuinely large
  ordinary dividend will be flagged for review, which is the safe direction.
- Adjusted history means MFE/MAE since entry are measured on back-adjusted
  highs/lows, so a dividend-paying position understates both slightly.
- One open position per symbol; no scaling plans, no options, no shorts, and
  P&L ignores commissions, fees and currency effects.
- The entry plan is computed from the session's close, so `distance_to_pivot`
  ages until the next run.

---

## Testing

`tests/` verifies each pipeline stage's output is correct and reasonable using
synthetic OHLCV/fundamentals data — no network calls, no live yfinance/Claude
requests, fully deterministic.

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest tests/ -v
```

| File | What it checks |
|---|---|
| `test_data_sources_filters.py` | Stage B liquidity/market-cap filters keep exactly the survivors their thresholds imply; Stage C sector relative-strength ranking is arithmetically correct |
| `test_stage_screen_technical.py` | Stage D: a synthetic strong uptrend scores all 8/8 Technical Score criteria, a sustained downtrend scores near 0, and too-short history returns `None` instead of crashing |
| `test_stage_screen_fundamentals.py` | Stage D2: strong/weak synthetic financials score 6/6 and 0/6 respectively; missing yfinance data degrades to a 0 score instead of raising |
| `test_stage_rank.py` | Stage E: `Composite Score = Technical + fundamentals_weight × Fundamentals`, `Max Score` formula, missing-fundamentals rows treated as 0 (not dropped), sort order (Composite desc, RS 3mo tiebreak), and scores staying within their documented bounds |
| `test_vcp_metrics.py` | Stage G's numeric VCP detection: a textbook decreasing-pullback pattern is flagged as contracting with volume dry-up; a choppy flat series doesn't false-positive; insufficient history returns `None` |
| `test_vcp_structure.py` | VCP detector on deterministic synthetic paths: textbook / imperfect / poor sequences, shrinking swings of an uptrend not treated as one base, falling lows, High/Low depth, forming right-edge contraction and no-lookahead guards, volume dry-up vs misleading last/first ratios, final-zone dryness, 1.8× breakout volume, base-specific pivot (repeated resistance, bounce high, stale, far below) |
| `test_vcp_review_routing.py` | Selective Claude review: strong-coiling / forming-strong / absent skipped, acceptable / weak / strong fresh-breakout reviewed, wick/gap routing for strong only, `all` and `off` modes, token-free compatibility, routing reasons in the prompt, contradictory-field rejection, rationale warnings, review record stored with the verdict, Telegram sends the Stage-G verification chart |
| `test_stage_charts.py` | Stage F chart rendering: empty data returns `None` gracefully; valid data writes a real, non-empty PNG |
| `test_ohlcv_cache.py` | OHLCV cache: fresh data served without fetching, stale cache delta-fetches the tail, cold/too-shallow tickers full-fetch, split-aware merge (cached history rescaled after a split, untouched within tolerance), delta-failure falls back to stale; market caps reused within TTL and refetched after |
| `test_ticker_failures.py` | Batch download skips tickers that fail rather than raising |
| `test_run_pipeline.py` | `run_screen()` keeps the artifacts and early exits `main()` always had, and adds stage counts |
| `test_trading_calendar.py` | NASDAQ sessions: weekends, holidays, early close, DST start/end, Singapore→New York date mapping, in-progress detection |
| `test_scheduler.py` | Startup catch-up, same-day duplicate checks, Saturday/Sunday handling, one-session catch-up limit, data readiness (no real sleeps), run/instance locks, SQLite claims, `--force`, pipeline failure + ops alerts, Telegram failure isolation, resend, the daily loop |
| `test_scheduler_deferred.py` | Late provider ready on a deferred retry (runs once), never ready → stops at the cutoff with one alert and stays pending, sleep through a retry → one catch-up probe, wake during US hours → `DEFERRED_MARKET_OPEN`, next-morning recovery, no duplicate after success |
| `test_history_store.py` | Run/result persistence, full ranked population, JSON round-trip, transaction rollback, duplicate-session protection, force supersede, stale claims, redacted config + hash, manifest |
| `test_trend_analysis.py` | New/dropped, rise/fall/same, 1- and 5-session deltas, score deltas, streaks, weekend/holiday continuity, gaps, VCP verdict changes |
| `test_telegram_notifier.py` | Mocked Bot API: message/photo, retries, `retry_after`, bounded failure, token redaction, HTML escaping, chunking |
| `test_stage_h_integration.py` | Two scheduled sessions end to end: real `run_screen` + ranking over synthetic stages → SQLite → trends → Telegram payload |
| `test_stage_h_ops.py` | Log secret redaction, file lock, per-run log file, history CLI, and the real `python -m pipeline.scheduler --dry-run` entrypoint |
| `test_stage_h_status.py` | `--test-telegram` (sends one message, no history, redacted failure), status summary agrees with the dry-run decision, `history_cli status` never creates the database |
| `test_check_schedule.py` | `SCHEDULE_MODE` parsing: daily default/back-compat, times sorting, interval bounds, clear errors for bad modes/times/intervals/time zones, next-occurrence arithmetic |
| `test_scheduler_modes.py` | Loop behaviour per mode: checks without extra Stage A–G runs or reports, startup interaction, sleep/resume collapse, one readiness episode with hourly checks, same-day hold after failure/cutoff, market-open guard with one audit row, exit 2 on invalid config |
| `test_setup_gate.py` | Stage I gate: READY / BREAKOUT_TRIGGERED / MISSED_EXTENDED / NOT_READY, Claude disagreement -> MANUAL_REVIEW, "Claude not requested" never blocking, rows Stage G never analysed skipped, the Stage-G row left unmutated |
| `test_entry_plan.py` | Layer 2: trigger and chase prices, breakout-volume confirmation against the Stage-G threshold, extended / no-pivot handling, and Claude never setting the entry price |
| `test_risk_model.py` | Layer 3: ATR/EMA/SMA arithmetic, structural stop vs volatility floor, cap tiers + liquidity + hard cap, SKIP_RISK_TOO_WIDE instead of a widened stop, risk-based vs portion-based sizing and the smaller-wins rule |
| `test_trade_store.py` | Ledger: multiple buy fills re-averaging cost, partial-sell P&L, full close, oversell rejection, one open trade per symbol, snapshot upsert, pending-action single claim and expiry |
| `test_trade_journal.py` | Gate -> persisted plans (bars fetched only for plannable setups), the preview/apply split, setup linkage on a recorded trade, degradation when the price download fails |
| `test_telegram_trade.py` | Authorization allowlist (chat and user), Confirm/Cancel, duplicate-callback protection, expiry, partial sell/close, the Record-Buy conversation, read-only command formatting |
| `test_trade_bot.py` | Polling runner against a fake Bot API: reply delivery, offsets, backlog skipping, callback answering / keyboard retirement, handler errors not killing the loop |
| `test_position_monitor.py` | Snapshot numbers (MFE/MAE from entry), protective-level ratchet and ATR room, HOLD/WATCH/TIGHTEN/PARTIAL/EXIT/STOP transitions, positions monitored with no screen involvement, degradation on missing bars |
| `test_trade_cli.py` | CLI fallback: the confirmation prompt, add/sell/close, positions/show/trades/plan output, oversell error |
| `test_stage_i_integration.py` | Stage I inside a scheduled run: plans persisted, a held symbol absent from the screen still monitored, the Telegram positions section (also on `--resend`), Record-Buy buttons, failures not failing the run, and the advisory-only boundary (no broker imports, no order calls) |

A shared autouse fixture blocks socket connections, so any test that reaches
the network fails.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac/Windows) or Docker Engine + Docker Compose v2 (Linux)
- The notebook file: `nasdaq_stage2_screener.ipynb`

---

## Quick start

### 1. Clone / set up the project folder

```
nasdaq-stage2/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run_notebook.sh
├── .env.example
├── .gitignore
├── .dockerignore
├── notebooks/
│   └── nasdaq_stage2_screener.ipynb   ← put your notebook here
└── data/                              ← auto-created on first run
```

### 2. Copy the environment file

```bash
cp .env.example .env
# Edit .env: secrets (ANTHROPIC_API_KEY, TELEGRAM_*), VCP_LIVE_ANALYSIS, schedule
```

`.env` is gitignored. `.env.example` only ever contains empty secrets.

### 3. Build the image

```bash
docker compose build
```

This installs all pinned Python dependencies into the image. Only needed once,  
or again after changing `requirements.txt` or `Dockerfile`.

### 4. Start the Jupyter server

```bash
docker compose up jupyter
```

Then open **http://localhost:8888** in your browser.  
The `notebooks/` folder is mounted — any changes you make persist on your host machine.

---

## Running on a schedule (optional)

For the screening pipeline, see [Scheduled operation](#scheduled-operation-stage-h)
(Windows Task Scheduler or the `pipeline` service). This section covers the
separate **notebook** runner.

The `runner` service executes the notebook headlessly via `nbconvert` on a cron schedule,  
saving a timestamped output notebook to `data/runs/` after each run.

```bash
# Start the interactive notebook AND the scheduled runner together
docker compose --profile scheduled up
```

Configure the schedule in `.env`. `SCHEDULE` is this notebook runner's cron line
only. It does not affect the Stage-H scheduler, which uses `SCHEDULE_MODE`,
`LOCAL_SCHEDULER_TIME`, `SCHEDULE_TIMES` and `SCHEDULE_INTERVAL_MINUTES`. An
old `PIPELINE_SCHEDULE` entry is obsolete and not read by anything.

```env
# Run Mon–Fri at 18:00 UTC (default)
SCHEDULE=0 18 * * 1-5

# Or daily at 09:30 UTC
SCHEDULE=30 9 * * *
```

Check runner logs:

```bash
docker compose --profile scheduled logs -f runner
```

---

## Common commands

```bash
# Build the image
docker compose build

# Start Jupyter (foreground, see logs)
docker compose up jupyter

# Start Jupyter (background)
docker compose up -d jupyter

# Stop everything
docker compose down

# Rebuild image after requirements change
docker compose build --no-cache

# Open a shell inside the running container (for debugging)
docker exec -it stage2_jupyter bash

# Run the notebook manually (one-off, no schedule)
docker exec stage2_jupyter /bin/sh /app/run_notebook.sh

# Tail Jupyter logs
docker compose logs -f jupyter
```

---

## Folder structure (runtime)

```
data/
├── sector_cache.csv        ← cached sector/industry lookups (persists across restarts)
├── market_cap_cache.csv    ← cached market caps (1-week TTL)
├── ohlcv_cache/            ← per-ticker Parquet price history (delta-fetched each run)
│   └── AAPL.parquet
├── stage2_watchlist_YYYYMMDD.csv   ← output watchlists
├── reports/               ← pipeline watchlist + VCP analysis CSVs
├── charts/<run_ts>/       ← Stage F chart PNGs (+ vcp_debug/) per run
├── logs/                  ← scheduler_<ts>.log + per-run pipeline_<run_ts>.log
├── history/
│   ├── stage2_history.sqlite3   ← canonical run/result/notification history
│   └── *.lock                   ← scheduler process locks
├── trades/
│   └── trade_journal.sqlite3    ← Stage I: your recorded fills, plans and position snapshots
└── runs/
    ├── 2026-09-14/run.json      ← per-session run manifest
    └── 20250621_180012_nasdaq_stage2_screener.ipynb  ← notebook runner archives
```

The `data/` directory is mounted as a Docker volume, so everything inside  
survives container restarts and `docker compose down`.

---

## Changing Python dependencies

1. Edit `requirements.txt`
2. Rebuild: `docker compose build --no-cache`
3. Restart: `docker compose up jupyter`

---

## Security note

By default, the Jupyter server runs **without a token** (fine for local use behind Docker).  
To add auth, set `JUPYTER_TOKEN=your_secret_token` in `.env` and open:  
`http://localhost:8888/?token=your_secret_token`

Do **not** expose port 8888 to the public internet without authentication.

Secrets (`ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) live only
in `.env`, which is gitignored; they never reach code, tests, docs, SQLite,
manifests or logs (every log handler carries a redacting filter, and the Bot API
URL embeds the token, so URLs and raw `requests` errors are never logged).

The trade journal (Stage I) holds no credentials of any kind, because it needs
none: there is **no brokerage API, no account access and no order placement**
anywhere in this project. It records the fills you report, calculates advisory
plans and monitors public market data. Mutating Telegram commands are accepted
only from `TELEGRAM_ALLOWED_CHAT_ID`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Port 8888 already in use | Change `JUPYTER_PORT=8889` in `.env` |
| `yfinance` rate-limit errors | Increase `batch_sleep_sec` in notebook CONFIG |
| Sector cache stale | Delete `data/sector_cache.csv` and re-run Stage C |
| Runner service not starting | Make sure you used `--profile scheduled` flag |
| Notebook not found in runner | Check `NOTEBOOK_NAME` in `.env` matches the filename in `notebooks/` |
| Stage G (VCP analysis) errors per ticker | Set `ANTHROPIC_API_KEY` in `.env` |
| Pipeline sector cache stale | Delete `data/sector_cache.csv` (separate from the notebook's own cache) and re-run |
| Scheduler "did nothing" | Normal if the session is already done or the US market is open — check `data/logs/scheduler_*.log` or run `python -m pipeline.scheduler --dry-run` |
| `DATA_NOT_READY` | Upstream hasn't published the session's bar yet. The scheduler keeps retrying hourly until `DEFERRED_RETRY_CUTOFF`, then leaves it for the next startup/daily check. Inspect with `python -m pipeline.history_cli runs` |
| `DEFERRED_MARKET_OPEN` | The check ran while the US market was open, so nothing was screened on purpose. The next safe startup/09:15 check reconsiders the session |
| "Scheduler already running (instance lock held)" | Expected when the 09:05 watchdog fires while the loop is up (exits 0). If you see it from a manual start, a loop is already running — run only one |
| Telegram report missing | `history_cli runs` shows `NOTIFICATION_FAILED`; fix credentials, then `python -m pipeline.scheduler --resend` |
| Suspect stale/corrupt cached prices | Delete `data/ohlcv_cache/` (or a single `<SYMBOL>.parquet`) to force a full re-download, or set `OHLCV_CACHE=0` |
