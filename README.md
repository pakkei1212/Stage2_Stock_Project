# NASDAQ Stage 2 Screener — Docker Environment

Dockerised Jupyter environment for the **NASDAQ Stage 2 / Minervini Trend Template screener**  
(`nasdaq_stage2_screener.ipynb`).

## What's included

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12-slim image with all dependencies |
| `docker-compose.yml` | Two services: interactive Jupyter + optional scheduled runner |
| `requirements.txt` | Pinned Python dependencies |
| `run_notebook.sh` | Script used by the runner service to execute the notebook headlessly |
| `.env.example` | Template for local config overrides (port, schedule, token) |
| `notebooks/` | Drop your `.ipynb` file here |
| `data/` | Persistent outputs: `sector_cache.csv`, watchlist CSVs, run archives |

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
