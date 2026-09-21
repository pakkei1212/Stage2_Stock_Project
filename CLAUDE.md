# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A NASDAQ Stage 2 / Minervini Trend Template stock screener. Two parallel
implementations of the same screening logic coexist:

- `notebooks/nasdaq_stage2_screener.ipynb` — interactive exploration, and the
  stated source of truth for the Stage A–E logic
- `pipeline/` — a native-Python port of that notebook (`data_sources.py`,
  `stage_screen.py`, `stage_rank.py` all carry "ported … unchanged logic"
  docstrings) plus stages the notebook doesn't have: chart rendering (Stage F),
  Claude vision VCP analysis (Stage G), and scheduling/history/trends/Telegram
  (Stage H, strictly post-G)

Changing screening semantics in `pipeline/` diverges it from the notebook.
Either mirror the change or say explicitly that the port now leads. Stage H
must never change Stage A–G screening, scoring, ranking, VCP metrics or the VCP
prompt/schema. Stage I (trade journal, entry/risk plans, corporate actions, position monitor)
sits downstream of both and is advisory only: it never places, modifies or
cancels a broker order, and there is no brokerage connection anywhere in this
project.

## Commands

```bash
# Tests — synthetic data only, no network (an autouse fixture blocks sockets), ~30s
python -m pytest tests/ -q
python -m pytest tests/test_vcp_metrics.py -v          # one file
python -m pytest tests/ -k "delta_fetch" -v            # by name

# Full pipeline, locally (writes into ./data) — no history, no Telegram
python -m pipeline.run_pipeline

# Stage H: session-aware one-shot / dry run / force / resend / long-running loop
python -m pipeline.scheduler --run-once
python -m pipeline.scheduler --dry-run
python -m pipeline.scheduler --run-once --force
python -m pipeline.scheduler --resend [YYYY-MM-DD]
python -m pipeline.scheduler --test-telegram     # one test message; no run, no history
# check frequency: SCHEDULE_MODE=daily|times|interval (see .env.example); bad config -> exit 2
python -m pipeline.scheduler

# Stage I: trade journal (advisory; records fills YOU report — no broker anywhere)
python -m pipeline.trade_cli add CACC --price 622.50 --shares 20   # --yes skips the prompt
python -m pipeline.trade_cli sell CACC --price 680 --shares 10
python -m pipeline.trade_cli close CACC --price 705
python -m pipeline.trade_cli positions | show CACC | trades --all
python -m pipeline.trade_cli plan                # recorded gate states + entry/risk plans
python -m pipeline.trade_cli monitor             # re-run the position monitor now
python -m pipeline.trade_cli actions CACC | actions --all        # corporate-action ledger
python -m pipeline.trade_cli record-action ABC --type SYMBOL_CHANGE --date 2026-10-01 --new-symbol XYZ
python -m pipeline.trade_cli resolve CACC --stop 295             # clear a CORPORATE_ACTION_REVIEW
python -m pipeline.trade_bot --once              # Telegram journal bot (separate process)

# History inspection
python -m pipeline.history_cli runs
python -m pipeline.history_cli top --sessions 10
python -m pipeline.history_cli symbol NVDA
python -m pipeline.history_cli status            # read-only; never creates the DB

# Windows (loads .env, then runs the scheduler with any args)
powershell -ExecutionPolicy Bypass -File scripts\start_scheduler.ps1 --dry-run

# Stage G only — reuses the latest watchlist CSV + already-rendered charts
python -m pipeline.run_vcp_only --top-n 5
python -m pipeline.run_vcp_only --watchlist data/reports/watchlist_20260901.csv

# Claude VCP stability check — real API calls, writes only data/reports/vcp_eval_<ts>/ (no history/Telegram)
python -m pipeline.vcp_eval CACC HALO --repeats 3

# Docker: one-shot pipeline run, nothing needs to be running first
docker compose run --rm --entrypoint python pipeline -m pipeline.run_pipeline

# Docker: rebuild after editing pipeline/ or requirements.txt (see gotcha below)
docker compose build jupyter

# Docker: interactive Jupyter on :8888
docker compose up -d jupyter
```

There is no linter or formatter configured. The local `.venv` is Python 3.11;
the image is 3.12.

## Architecture

`pipeline/run_pipeline.py::run_screen` is the whole Stage A–G control flow —
sequential, narrowing the ticker set through Stage D (D2 and E keep every
Stage D row). `main()` and the scheduler both call it; there is no second copy.

```
A  load_universe            ~4,300 NASDAQ symbols (Nasdaq Trader txt; Wikipedia NASDAQ-100 fallback)
B  cheap_liquidity_filter   price + avg dollar volume, from 30d bars
B  market_cap_filter        $2B floor, per-ticker yfinance call (CSV-cached, 7d TTL)
C  get_sector_map           3 tiers: Wikipedia S&P500 → Wikipedia NDX → yfinance .info
C  rank_sector_strength     keep tickers in the top 5 sectors by median RS vs ^IXIC
D  run_stage2_screen        8 Minervini criteria from 400d bars → Technical Score
D2 fundamentals_screen      6 CANSLIM criteria via yfinance → Fundamentals Score
E  score_and_rank           Composite = Technical + fundamentals_weight × Fundamentals
F  generate_charts          candlestick + volume PNG for the top vcp_top_n
G  run_vcp_analysis         numeric VCP metrics, then optionally Claude vision verdict
H  scheduler.check_and_run  session resolve → readiness → run_screen → SQLite → trends → Telegram
I  run_trade_layer          setup gate → entry/risk plans → corporate actions → open-position monitor → report section
```

Structural conventions that run through everything:

**Config threading.** `pipeline/config.py` exposes one `CONFIG` dict; every
stage function takes `config=CONFIG` as a keyword so callers (and tests) can
pass a modified copy. Set `max_universe_for_testing` to a few hundred for a
fast end-to-end run. Paths in `CONFIG` are relative (`data/…`), so entrypoints
must run from the repo root (or `/app` in the container). Stage H keys are
operational and excluded from `run_metadata.config_hash`.

**Degrade, never raise.** Per-ticker network failures are caught and logged,
and the ticker is dropped or scored 0 — a bad symbol never aborts a run. Drop
reasons go to `logger.debug`, so `LOG_LEVEL=DEBUG` is how you find out why a
ticker vanished. Modules log under the `pipeline` logger; **modules runnable
with `python -m` must use an explicit `logging.getLogger("pipeline.<name>")`**
— under `-m`, `__name__` is `"__main__"` and INFO logs silently vanish.
`setup_logging()` is idempotent and adds a secret-redacting filter to every
handler; the scheduler writes `scheduler_<ts>.log` and tees each Stage A–G run
into `pipeline_<run_ts>.log` via `run_log_file`.

**`data_sources.batch_download` is the single OHLCV choke point.** It wraps a
three-way Parquet cache (`ohlcv_cache.py`): fresh-and-deep-enough → served from
disk; deep but stale → fetch only the delta and append; missing or too shallow
→ full fetch. Because yfinance runs with `auto_adjust=True`, splits retroactively
rescale history, so `ohlcv_cache.merge` compares the overlap window and rescales
cached bars when the median close ratio deviates >2% (`SPLIT_TOL`).

**Stage G is metrics-first.** The chain is
`detect_vcp_structure` → `compute_vcp_metrics` → `assess_vcp_quality`, and the
numbers go to Claude *alongside* the chart PNG (`json_schema` output,
`VCP_SCHEMA` unchanged). Claude confirms and interprets the pattern; it does not
read numbers off pixels. Key rules (details in README "VCP detector"):
- **Structure ≠ depth.** Swings are detected on Close. Confirmed pivots use a
  symmetric ±`vcp_pivot_window_days` window, which only reaches bars with
  enough data on both sides. The last w bars use `_find_forming_pivots`
  (truncated window, no lookahead), so a right-edge contraction is reported
  with `confirmed: false` instead of being dropped. Swings under
  `vcp_min_swing_pct` are pruned. Depth is measured chronologically (`_measure_leg`): the High first, then the Low only
  from later bars (never the pivot bar's own range). Pullback volume excludes the High bar.
- **One base, not every swing.** Legs are added right-to-left while all leg
  highs stay within `vcp_resistance_band_pct`. Earlier legs are ignored, so a
  stair-step uptrend's shrinking pullbacks are not a VCP.
- **Pivot is base-specific.** It is the highest right-side contraction high,
  and the final-zone volume is measured *before* any breakout bar.
- **All thresholds are our heuristics**, not Minervini rules: `vcp_*` CONFIG
  keys plus the documented `_TIGHTENING_*` / `_VOLUME_*` / `_TIGHT_*`
  constants. Don't describe them as book rules, and don't tune them on famous
  winners or future returns.
- **Compatibility.** The 13 v1 metric keys stay, with base-specific meaning
  since `VCP_METRICS_VERSION` 2. Stage H stores every non-verdict key in
  `vcp_metrics_json` (no whitelist), and old rows are never migrated.
- **Isolation.** Stage G output never feeds Composite Score or ranking.
- **Claude verdict handling.** `analyze_chart` validates every response
  (`validate_verdict` against the flat `VCP_SCHEMA`). Refusal, `max_tokens`
  truncation, invalid JSON and schema violations become `{"error": ...}` rows,
  and one ticker's failure never stops Stage G. Contradictory structured fields
  (`semantic_problems`, e.g. `is_vcp_pattern=false` with a VCP stage) are
  rejected as `semantic_inconsistency`, never silently fixed. Claude receives
  the plain Stage-F chart plus measured contraction dates and its routing
  reasons.
- **Claude is a selective reviewer** (`VCP_LLM_REVIEW_MODE`, default
  `selective`; `all` for research; `off`; `VCP_LIVE_ANALYSIS=0` forces off).
  `llm_review_decision` routes acceptable/weak, strong at a fresh breakout, and
  strong with wick/gap `structure_warnings`. Strong-coiling and absent results
  stand. Routing never changes numeric metrics; the `llm_review` record is
  stored under `vcp_verdict_json["review"]`. Skipped = `{"skipped":
  "not_requested"}`, which is not a negative verdict and not counted in
  agreement. A wick/gap routing reason adds a mandatory `Wick/pivot review:` /
  `Gap review:` rationale instruction (`_structure_review_block`, which parses
  the `structure_warnings` wording). `prompt_adherence_warnings` flags a
  rationale that ignores it. This is non-blocking and the schema is unchanged.
- **Telegram is a dashboard plus a separate debug message.** The main report
  (`build_report`) is Action Board → READY → WATCH → REVIEW → Open positions →
  Overall leaders (Composite order) → Watchlist changes. Its buckets
  (`classify_setup`) are read from the **persisted** `setup_plans` of the run;
  READY needs `READY`/`BREAKOUT_TRIGGERED` *and* a usable plan (`plan_is_usable`).
  The formatter prints stored plan values and never recomputes entry/risk/size.
  Within a bucket the order is the `vcp_review_set` VCP priority, then rank.
  Stage counts, Stage-G/I counts, monitor and corporate-action counts live only
  in `build_debug_report` (`TELEGRAM_DEBUG_SUMMARY`, default 1; a failed debug
  send is a `debug` notification, never a run failure). Photos are the stored
  Stage-G `vcp_debug` charts of the VCP set (`select_vcp_chart_rows`,
  `TELEGRAM_VCP_CHART_MAX`), never Stage-F top-rank charts. Never merge numeric
  and Claude results or overwrite metrics with Claude's reading. Use
  `pipeline.vcp_eval` (non-persistent) for repeat runs, never `--force`.

`render_vcp_annotated_chart` re-runs the same computation with
`return_details=True` and draws the selected base (shaded), base vs ignored vs
forming pivots, the T1..Tn legs, the pivot and the volume averages into
`vcp_debug/`. Look at that chart when a metric or the chosen base seems wrong.

## Stage H: scheduled operation

Current deployment is a **Windows laptop** (on roughly 09:00–23:00
Asia/Singapore), via Windows Task Scheduler → `scripts/start_scheduler.ps1` →
`python -m pipeline.scheduler` (long-running). The `pipeline` Compose service
runs the same loop as an alternative. Invariants to preserve:

1. **WHEN ≠ WHICH.** The trigger is local wall-clock, set by `check_schedule.py`
   (`SCHEDULE_MODE`: `daily` default at `LOCAL_SCHEDULER_TIME` 09:15 / `times`
   `SCHEDULE_TIMES` / `interval` `SCHEDULE_INTERVAL_MINUTES` 15–1440, all in
   `LOCAL_SCHEDULER_TIMEZONE` Asia/Singapore). Checks run **every calendar day
   including weekends**, plus a startup check. *Which* session is always
   `trading_calendar.latest_completed_session(now)` from the XNAS exchange
   calendar. Never use local weekday or `today - 1 day`. A future VPS changes
   only the trigger config. Check frequency never changes run frequency: every
   main check is `check_and_run`, so Stage A–G still runs at most once per
   session. Windows Task Scheduler stays a launcher/watchdog; never add tasks
   per check time. The legacy `SCHEDULE` var is the Docker notebook runner's
   cron only, and `PIPELINE_SCHEDULE` is unused.
2. **The NASDAQ session date is authoritative** — Monday's session processed
   Tuesday 09:15 SGT is stored as Monday; Friday is processed Saturday morning.
3. **Automatic catch-up is exactly one session** (the latest completed). This is
   structural, not a tunable: Stage A–G screens the *latest* upstream bars, so a
   historical session cannot be re-screened. Don't add a backfill that would
   label today's data with an old date.
4. **No run while a US session is in progress** (21:30/22:30–04:00/05:00 SGT) —
   daily bars would be partial. The check records a non-canonical
   `DEFERRED_MARKET_OPEN` row and the session stays pending for the next safe
   check. The readiness guard (`readiness.py`) also requires `^IXIC`'s latest
   bar == target session.
4a. **Late data is deferred, not failed.** After the short readiness retries a
   `DATA_NOT_READY` session is re-probed by the loop every
   `DEFERRED_RETRY_MINUTES` (one download, no run row, trigger `deferred`) until
   `DEFERRED_RETRY_CUTOFF` (20:30 local), then left for the next startup/daily
   check. Stage A–G runs once when ready. Missed retries during sleep collapse
   into one probe. Keep `DATA_NOT_READY`, `DEFERRED_MARKET_OPEN` and
   `PIPELINE_FAILED` distinct; only the first gets same-day retries. All retries go
   through `check_and_run`; `plan_deferred_retry` only decides *when*.
4b. **One episode per session, whatever the check frequency** (`run_forever` loop
   state). While a retry is `pending`, a main check for the same session runs with
   `probe_first=True` (one download, no claim/row) and re-bases the pending retry,
   so main checks and deferred retries merge into one stream. After
   `PIPELINE_FAILED` or the deferred cutoff, a `SameDayHold` makes main checks
   skip that session until the next local day (startup/manual checks ignore it).
   Repeated market-open checks record one `DEFERRED_MARKET_OPEN` row per US
   session (`record_market_open`). The next main check is computed from the end
   of the previous check, so an occurrence due during a check counts as covered,
   and missed occurrences during sleep collapse into one.
5. **One canonical result per trading session.** SQLite partial unique indexes
   (`ux_runs_canonical`, `ux_runs_active`) plus OS file locks (`run_lock.py`).
   A run becomes canonical only in the transaction that writes all its
   `stock_results`. `--force` supersedes atomically.
6. **SQLite (`data/history/stage2_history.sqlite3`) is canonical history**;
   CSV/PNG/`data/runs/<date>/run.json` are human-readable artifacts. The **full
   ranked population** is stored, not just the Top N.
7. **Top-N trends derive from that history** (`trend_analysis.py`, pure and
   deterministic, over recorded trading sessions — weekends/holidays can't break
   streaks). No LLM classifies trends.
8. **Telegram is downstream and non-critical.** Pipeline success is
   `is_canonical`, never notification success. Failures → `NOTIFICATION_FAILED`,
   re-send with `--resend` (no Stage A–G rerun). Ops alerts never mask the
   original failure.
9. **Credentials never enter the repo** — not code, tests, docs, SQLite,
   manifests or logs. `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`/`ANTHROPIC_API_KEY`
   come only from the environment (`.env`, gitignored); `.env.example` has empty
   values. The Bot API URL contains the token, so never log URLs or raw
   `requests` exceptions without `redact`.

Tests for this layer inject a fake clock/sleep/fetch/Telegram/run_screen through
`scheduler.Deps` (`tests/stage_h_fakes.py`) — nothing sleeps or hits the network.
`Deps.fetch_price_history`, `Deps.fetch_corporate_actions` and
`Deps.trade_store` do the same for Stage I.

## Stage I: trade journal, corporate actions and position monitor (advisory only)

Strictly post-Stage-G. It reads stored Stage-G output, records the fills the
*user* reports, keeps that record on the right share basis when a corporate
action happens, and monitors held positions on public data. Invariants:

1. **There is no broker.** No brokerage API, no IBKR/any credentials, no account
   access, no order placement, no automatic execution — not even behind a flag.
   The app records user-supplied fills, calculates plans, monitors public market
   data and sends alerts. Never add a client library, an order call, or wording
   ("order placed/filled/submitted") that implies execution.
   `tests/test_stage_i_integration.py` fails the build on broker imports, order
   call names in `pipeline/*.py`, or a broker package in `requirements.txt`.
2. **Never touches Stage A–G.** Screening, scoring, ranking, VCP metrics, pivot
   logic, the VCP prompt/schema and Claude routing are inputs only. The gate
   records its own state and never rewrites a verdict. `trade_*` config keys are
   excluded from `run_metadata.config_hash` (OPERATIONAL_KEY_PREFIXES), and
   `scheduler.run_trade_layer` swallows its own failures: a Stage-I problem never
   changes a run's canonical outcome.
3. **Separate database.** `data/trades/trade_journal.sqlite3` (`trade_db_path`),
   never the screening history file. The link is the plain `setup_run_id` /
   `setup_date` text on a trade. Tables: `setup_plans`, `trades`, `trade_fills`,
   `position_snapshots`, `trade_events`, `pending_actions`, `corporate_actions`,
   `corporate_action_checks`. Schema v2 adds columns to `trades`
   (`dividend_income`, `symbol_history_json`, `needs_review`, `review_reason`)
   and `position_snapshots` (`price_pnl`, `dividend_income`, `total_pnl`,
   `corporate_actions_json`); `trade_store._migrate` adds them in place and
   rewrites no row.
4. **The numeric pivot is authoritative.** `pivot_price_candidate` is the entry
   reference; Claude's `pivot_price` / `suggested_stop_loss` are context only.
   Breakout-volume confirmation reuses `vcp_breakout_volume_ratio` — do not add a
   second opinion about what a valid breakout is.
5. **Gate rule** (`setup_gate.py`): numeric quality in {strong, acceptable} AND
   pivot_state in {coiling_below_pivot, fresh_breakout}; `extended_above_pivot`
   is `MISSED_EXTENDED`; a Claude review that actually ran and said
   `is_vcp_pattern=false` is `MANUAL_REVIEW`, never an automatic READY. "Claude
   not requested" is not a negative verdict and must never block a clean numeric
   setup. Plans exist only for READY / BREAKOUT_TRIGGERED.
6. **Structure first, volatility as the floor, market cap only as a modifier.**
   `risk_model.build_risk_plan`: the final contraction low sets the stop; a stop
   inside `trade_atr_stop_multiple` x ATR% widens to that floor; the market-cap
   tier and liquidity set the *ceiling*. Required > ceiling is
   `SKIP_RISK_TOO_WIDE` — never widen the stop, never tighten it into the noise,
   never size the trade. Never implement "market cap -> fixed stop".
7. **Sizing is the smaller of risk-based and portion-based**, floored
   (`position_size`). `portfolio_value` is a `.env` number; no balance is read.
8. **Every mutation is confirmed.** `/buy`, `/sell`, `/close` (and the CLI) build
   a preview and write nothing until Confirm. `trade_store.claim_pending` is a
   conditional UPDATE, so a duplicate callback writes nothing. Mutating commands
   and callbacks are accepted only from `TELEGRAM_ALLOWED_CHAT_ID` (falling back
   to `TELEGRAM_CHAT_ID`), optionally narrowed by `TELEGRAM_ALLOWED_USER_ID`;
   everything from another chat is ignored, with no reply.
9. **Monitoring is ledger-driven, not screen-driven.** `monitor_open_positions`
   iterates open trades, so a held stock that has left Stage G is still
   monitored. Bars come through `data_sources.batch_download` (the existing
   cache), never a new download path.
10. **Protection ratchets, never widens.** The protective level is
    max(initial stop, previous level, cushion-unlocked references) and is then
    held at least `trail_atr_multiple` x ATR below the close. There is no
    universal "+20% take profit": `PARTIAL_PROFIT_REVIEW` needs a cushion *and* a
    climactic extension.
11. **All Stage-I thresholds are OUR provisional heuristics** (`trade_*`), not
    Minervini rules, and were not fitted to the current VCP examples. Keep them
    configurable and documented, and evaluate them against recorded trade
    outcomes before tuning.
12. **One price basis, adjusted exactly once.** `yf.download(...,
    auto_adjust=True)` means every bar is split- *and* dividend-adjusted
    (Volume: splits only) and the newest bar is the real traded price;
    `ohlcv_cache.merge` keeps the cache on that one basis. Recorded fills are
    raw as-traded prices and are **immutable**; stored pivots/stops share the
    fills' basis. So the bars need no correction from us — only the journal's
    *derived* values do. `corporate_actions.PRICE_BASIS` states the contract and
    `dividend_price_allowance` is the single place a raw-price source would ever
    be compensated (it returns 0.0 for our adjusted bars). Never add a second
    price adjustment anywhere.
13. **Safe automation boundary** (`corporate_actions.AUTO_APPLY_TYPES`). Applied
    automatically because the arithmetic is unambiguous: SPLIT / REVERSE_SPLIT /
    STOCK_DIVIDEND (`shares × ratio`, per-share references `÷ ratio`, economic
    value and already-earned realised P&L unchanged), CASH_DIVIDEND /
    SPECIAL_DIVIDEND (books `dividend_income`; shares, fills and average cost
    untouched) and an unambiguous SYMBOL_CHANGE (same `trade_id`, fills, entry
    date, P&L and setup linkage — never close-and-reopen). MERGER, SPINOFF,
    RIGHTS, DELISTING and anything unclassified are recorded and flagged
    `CORPORATE_ACTION_REVIEW`; a cash acquisition with final terms reaches
    `AWAITING_CONFIRMATION` and waits for the user's own `/close`. Never invent
    a cost-basis allocation, an exchange ratio or a share quantity.
14. **Ambiguity is recorded, not estimated.** A fill dated on the ex-date leaves
    entitlement undeterminable → review, not a guess. A SPECIAL_DIVIDEND books
    the cash but does **not** move the stop/pivot it just rebased the price
    series against → review. Missing market data is never a sale: after
    `trade_missing_bars_review_sessions` empty sessions a DELISTING action is
    recorded. `CORPORATE_ACTION_REVIEW` outranks every technical state in
    `classify_position`, so no mechanical exit conclusion is drawn while an
    action is unresolved; `trade_cli resolve` is the only way out, with the
    user's own numbers.
15. **Corporate actions run before the monitor**, inside `run_trade_layer`:
    plans → detect → safe apply → monitor → report. A held position is checked
    even when the screener never saw it, and each symbol is probed once per
    session (`corporate_action_checks`). Idempotency is structural: a unique
    `fingerprint` index makes re-detection a no-op, and `apply_action_to_trade`
    flips the status conditionally inside the same transaction that mutates the
    trade, so a duplicate apply writes nothing. The monitor rebases a protective
    level carried forward across a share-basis change (`rebase_price`) instead
    of rewriting the stored snapshot.

The bot (`pipeline/trade_bot.py`) is a separate long-poll process from the
scheduler; `telegram_trade.py` holds all its decisions and does no I/O, so the
tests drive it with plain dicts.

## Gotchas

- **`VCP_LIVE_ANALYSIS` defaults differ.** `config.py` defaults it on (`"1"`);
  `docker-compose.yml` defaults it to `0`. So a containerised run computes
  metrics and verification charts but makes **no** Claude call and writes
  `{"skipped": "token_free_mode"}` verdicts (stored as `vcp_status='skipped'`),
  while a bare local run does spend tokens. Set it explicitly whenever the
  answer matters. The Windows wrapper takes it from `.env`.
- **Only the `jupyter` service has a build context.** `pipeline` and `runner`
  reuse the `nasdaq-stage2-screener:latest` image it produces, so
  `docker compose build pipeline` silently does nothing. Pipeline code is
  `COPY`-ed into the image (not volume-mounted — only `./data` is), so edits to
  `pipeline/` need `docker compose build jupyter` before they take effect in a
  container.
- **`pipeline` and `runner` sit behind the `scheduled` Compose profile**, so
  `stage2_pipeline` does not exist after a plain `docker compose up`, and
  `docker exec stage2_pipeline …` fails with "No such container". Use the
  `docker compose run` one-shot above, or `docker compose --profile scheduled up -d pipeline`.
- **Run the scheduler in one place.** OS file locks aren't reliably shared
  between a Windows host process and a container over the Docker Desktop bind
  mount; only the SQLite claim protects that combination.
- **Windows PowerShell 5.1 + native stderr.** Under `$ErrorActionPreference =
  'Stop'`, any stderr from Python becomes a terminating error — the wrapper
  switches to `Continue` before invoking Python and sets `PYTHONIOENCODING=utf-8`
  (the console code page is cp1252; avoid `─`, `→`, `↳` in log/CLI text).
- **`.env` is not loaded by Python.** Compose reads it; on Windows the wrapper
  script loads it. A bare `python -m pipeline.scheduler` sees only the real
  environment.
- **`run_pipeline` isolates charts per run** by overriding `chart_dir` with a
  `<run_ts>` subdirectory. `run_vcp_only` does *not* — it looks for
  `data/charts/<SYMBOL>.png` at the root, so pass charts there or point
  `chart_dir` at the run's subdirectory, otherwise every symbol is skipped for
  "no chart found". Stage H stores each run's chart paths in `stock_results`.
- **Stage F/G bypass the OHLCV cache.** `stage_charts.fetch_chart_data` calls
  `yf.download` directly, and `run_vcp_analysis` calls it *again* per symbol —
  so the top-N candidates are downloaded twice more after the cached Stage D
  pull. Left as-is deliberately: cached bars can differ slightly from a fresh
  `auto_adjust` download after a split rescale, which could shift VCP numbers.
- **Partial-day bars in the OHLCV cache.** A "fresh" (≤1 day old) cached bar is
  served only if the cache file was written after that bar's session close +
  `session_close_buffer_minutes` (`ohlcv_cache.last_bar_complete`); otherwise the
  ticker is delta-fetched and the overlap replaces the intraday bar. The signal is
  the Parquet file's mtime, so anything that rewrites cache files without a fetch
  (or copies them with fresh timestamps) defeats it.
- **The trade journal DB is created by a normal run**, because Stage I records a
  gate state for every Stage-G symbol. `position_monitor` and the report section
  open it with `create=False`, so they are no-ops when nothing was ever recorded.
- **The Open/View Position buttons need `TRADE_BOT_ENABLED=1`** *and* the separate
  `python -m pipeline.trade_bot` process; the scheduler never polls Telegram.
  `op:SYM` asks fill price → shares → the normal Confirm preview (plan vs
  actual); `vp:SYM` is `/position SYM`; `rb:SYM` (old Record Buy) = `op:`. A
  READY chart carries its button; the report keyboard holds the rest.
- **A corporate action mutates the trade row, not the fills.** `trade_fills` is
  the immutable record of what the user reported; after a 2-for-1 the fill still
  reads `BUY 20 @ 620.00` while the trade reads 40 @ 310. Anything that wants
  "shares as of date X" must rebase the fills through `basis_factor`, which is
  what `eligible_shares_at` does for dividend entitlement.
- **Corporate actions are applied to the OPEN trade only.** Closed trades keep
  their recorded history (that is the point), so a dividend whose ex-date falls
  inside a since-closed position is not booked.
- **An empty ranked result is `PIPELINE_FAILED`, not a successful empty day** —
  it's almost always a data outage, and failing keeps the session retryable.
- `ANTHROPIC_API_KEY` is never read explicitly — `anthropic.Anthropic()` picks
  it up from the environment, and it must be present in `.env` for the compose
  pass-through to forward it.

## Tests

Stage I tests use `tests/stage_i_fakes.py` (`stage_g_row` for a stored Stage-G
row, `make_bars` for a chosen close path) and inject every price fetch, so the
gate, plans, ledger, monitor, Telegram handling and CLI are all exercised
without a network, a broker or Claude.

`tests/conftest.py` builds synthetic OHLCV (`make_ohlcv` with a controllable
drift and volume bias; `make_vcp_df` for a textbook 20%→12%→6% contraction) and
an autouse fixture that blocks socket connections and unsets Telegram
credentials. Every yfinance / requests / Anthropic / Telegram call is
monkeypatched or injected, so assertions are exact rather than "some real
ticker happened to pass". Keep new tests on that footing — a test that reaches
the network is a broken test here. Use temporary SQLite files (`tmp_path`).
