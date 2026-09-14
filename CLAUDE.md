# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A NASDAQ Stage 2 / Minervini Trend Template stock screener. Two parallel
implementations of the same screening logic coexist:

- `notebooks/nasdaq_stage2_screener.ipynb` — interactive exploration, and the
  stated source of truth for the Stage A–E logic
- `pipeline/` — a native-Python port of that notebook (`data_sources.py`,
  `stage_screen.py`, `stage_rank.py` all carry "ported … unchanged logic"
  docstrings) plus two stages the notebook doesn't have: chart rendering
  (Stage F) and Claude vision VCP analysis (Stage G)

Changing screening semantics in `pipeline/` diverges it from the notebook.
Either mirror the change or say explicitly that the port now leads.

## Commands

```bash
# Tests — synthetic data only, no network, ~30s
python -m pytest tests/ -q
python -m pytest tests/test_vcp_metrics.py -v          # one file
python -m pytest tests/ -k "delta_fetch" -v            # by name

# Full pipeline, locally (writes into ./data)
python -m pipeline.run_pipeline

# Stage G only — reuses the latest watchlist CSV + already-rendered charts
python -m pipeline.run_vcp_only --top-n 5
python -m pipeline.run_vcp_only --watchlist data/reports/watchlist_20260901.csv

# Docker: one-shot pipeline run, nothing needs to be running first
docker compose run --rm --entrypoint python pipeline -m pipeline.run_pipeline

# Docker: rebuild after editing pipeline/ (see gotcha below)
docker compose build jupyter

# Docker: interactive Jupyter on :8888
docker compose up -d jupyter
```

There is no linter or formatter configured. The local `.venv` is Python 3.11;
the image is 3.12.

## Architecture

`pipeline/run_pipeline.py::main` is the whole control flow — seven stages,
sequential, each one narrowing the ticker set:

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
```

Two structural conventions run through everything:

**Config threading.** `pipeline/config.py` exposes one `CONFIG` dict; every
stage function takes `config=CONFIG` as a keyword so callers (and tests) can
pass a modified copy. Set `max_universe_for_testing` to a few hundred for a
fast end-to-end run. Paths in `CONFIG` are relative (`data/…`), so entrypoints
must run from the repo root (or `/app` in the container).

**Degrade, never raise.** Per-ticker network failures are caught and logged,
and the ticker is dropped or scored 0 — a bad symbol never aborts a run. Drop
reasons go to `logger.debug`, so `LOG_LEVEL=DEBUG` is how you find out why a
ticker vanished. Every module logs via `logging.getLogger(__name__)` under the
`pipeline` logger; `setup_logging()` is idempotent and is called once by the
entrypoint, writing to stdout **and** `data/logs/pipeline_<run_ts>.log`.

**`data_sources.batch_download` is the single OHLCV choke point.** It wraps a
three-way Parquet cache (`ohlcv_cache.py`): fresh-and-deep-enough → served from
disk; deep but stale → fetch only the delta and append; missing or too shallow
→ full fetch. Because yfinance runs with `auto_adjust=True`, splits retroactively
rescale history, so `ohlcv_cache.merge` compares the overlap window and rescales
cached bars when the median close ratio deviates >2% (`SPLIT_TOL`).

**Stage G is metrics-first.** `compute_vcp_metrics` derives contraction depths,
leg-over-leg ratios, per-contraction volume dry-up, candidate pivot and MA
extension numerically from OHLCV. Those numbers are then sent to Claude
*alongside* the chart PNG with a `json_schema` output format (`VCP_SCHEMA`), so
the model confirms and narrates a pattern rather than reading it off pixels.
`render_vcp_annotated_chart` re-runs the same computation with
`return_details=True` and draws the intermediate pivots/legs onto a verification
chart in `vcp_debug/` — that chart exists to check the numbers against the
price action, and is what to look at when a metric seems wrong.

## Gotchas

- **`VCP_LIVE_ANALYSIS` defaults differ.** `config.py` defaults it on (`"1"`);
  `docker-compose.yml` defaults it to `0`. So a containerised run computes
  metrics and verification charts but makes **no** Claude call and writes
  `{"skipped": "token_free_mode"}` verdicts, while a bare local run does spend
  tokens. Set it explicitly whenever the answer matters.
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
- **`run_pipeline` isolates charts per run** by overriding `chart_dir` with a
  `<run_ts>` subdirectory. `run_vcp_only` does *not* — it looks for
  `data/charts/<SYMBOL>.png` at the root, so pass charts there or point
  `chart_dir` at the run's subdirectory, otherwise every symbol is skipped for
  "no chart found".
- **Stage F/G bypass the OHLCV cache.** `stage_charts.fetch_chart_data` calls
  `yf.download` directly, and `run_vcp_analysis` calls it *again* per symbol —
  so the top-N candidates are downloaded twice more after the cached Stage D
  pull.
- **`README.md` documents a `.env.example` that isn't in the repo.** `.env` is
  gitignored; the environment variables that actually exist are the ones listed
  in `x-common-env` in `docker-compose.yml` and read via `os.environ` in
  `config.py`.
- `ANTHROPIC_API_KEY` is never read explicitly — `anthropic.Anthropic()` picks
  it up from the environment, and it must be present in `.env` for the compose
  pass-through to forward it.

## Tests

`tests/conftest.py` builds synthetic OHLCV (`make_ohlcv` with a controllable
drift and volume bias; `make_vcp_df` for a textbook 20%→12%→6% contraction).
Every yfinance / requests / Anthropic call is monkeypatched, so assertions are
exact rather than "some real ticker happened to pass". Keep new tests on that
footing — a test that reaches the network is a broken test here.

One gap to be aware of: `README.md`'s test table claims `test_ohlcv_cache.py`
covers the "split-aware merge", but no test exercises `ohlcv_cache.merge`'s
corporate-action rescaling path — the riskiest piece of the cache is untested.
