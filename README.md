# NASDAQ Stage 2 Screener — Docker Environment

Dockerised Jupyter environment for the **NASDAQ Stage 2 / Minervini Trend Template screener**  
(`nasdaq_stage2_screener.ipynb`).

## What's included

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12-slim image with all dependencies |
| `docker-compose.yml` | Three services: interactive Jupyter, scheduled notebook runner, scheduled weekly pipeline |
| `requirements.txt` | Pinned Python dependencies |
| `run_notebook.sh` | Script used by the `runner` service to execute the notebook headlessly |
| `run_weekly.sh` | Script used by the `pipeline` service to run the native-Python weekly pipeline |
| `.env.example` | Template for local config overrides (port, schedule, token, Anthropic key) |
| `notebooks/` | Interactive exploration notebook — `nasdaq_stage2_screener.ipynb` |
| `pipeline/` | Native-Python weekly pipeline: screen → rank → chart → Claude VCP vision analysis |
| `data/` | Persistent outputs: `sector_cache.csv`, watchlist CSVs, chart PNGs, VCP reports, run archives |
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
```

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

Stage G computes Volatility Contraction Pattern signals (pullback depths, volume
dry-up) numerically from price/volume data first, then sends the chart image
**plus** those computed numbers to Claude (`claude-opus-4-8` by default) for a
structured verdict (`is_vcp_pattern`, `pivot_price`, `suggested_stop_loss`,
`entry_recommendation`, etc.) — the model confirms/narrates the pattern rather
than detecting it from pixels alone.

Run it directly:

```bash
docker exec stage2_pipeline python -m pipeline.run_pipeline
# or locally, from the project root, with dependencies installed:
python -m pipeline.run_pipeline
```

`docker exec` requires the `stage2_pipeline` container to already exist. It's
gated behind the `scheduled` Compose profile, so it won't be created by a plain
`docker compose up jupyter` (or by `docker compose build`). If `docker exec`
fails with `No such container: stage2_pipeline`, bring it up first:

```bash
docker compose --profile scheduled up -d --build pipeline
```

Then retry the `docker exec` command above. This starts the container on its
cron schedule (`PIPELINE_SCHEDULE`, default Fridays 21:00 UTC) — `docker exec`
just lets you trigger an on-demand run inside it without waiting for that
schedule.

Or let it run on a schedule via the `pipeline` service (see below). Set
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
- `data/charts/<SYMBOL>.png` — the chart each verdict was based on
- `data/logs/pipeline_<timestamp>.log` — full audit log for that run (see [Logging](#logging))

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
| `test_stage_charts.py` | Stage F chart rendering: empty data returns `None` gracefully; valid data writes a real, non-empty PNG |
| `test_ticker_failures.py` | Batch download skips tickers that fail rather than raising |

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
# Edit .env if you want to change the port, add a token, or adjust the schedule
```

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

The `runner` service executes the notebook headlessly via `nbconvert` on a cron schedule,  
saving a timestamped output notebook to `data/runs/` after each run.

```bash
# Start the interactive notebook AND the scheduled runner together
docker compose --profile scheduled up
```

Configure the schedule in `.env`:

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
├── stage2_watchlist_YYYYMMDD.csv   ← output watchlists
└── runs/
    └── 20250621_180012_nasdaq_stage2_screener.ipynb  ← timestamped run archives
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
