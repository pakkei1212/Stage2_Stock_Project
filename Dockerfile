# ── Base ────────────────────────────────────────────────────────────────────
# Python 3.12 slim keeps the image lean; matches a common Anaconda env version.
FROM python:3.12-slim

# ── System dependencies ──────────────────────────────────────────────────────
# Only what's actually needed:
#   curl         – health-check / manual debugging
#   ca-certificates – TLS for Yahoo Finance / Nasdaq Trader HTTPS calls
#   git          – optional, handy for notebook version control inside container
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Project files ────────────────────────────────────────────────────────────
# Copy the notebook in at build time so it's available immediately on startup.
# The notebooks/ directory is also mounted as a volume in docker-compose so that
# any changes you make inside the container are persisted on your host machine.
COPY notebooks/ ./notebooks/

# Native-Python weekly pipeline (screen + rank + chart + VCP vision) and the
# entrypoint scripts used by the scheduled runner/pipeline services.
COPY pipeline/ ./pipeline/
COPY run_notebook.sh run_weekly.sh ./

# ── Jupyter config ───────────────────────────────────────────────────────────
# Disable token auth when running locally behind docker-compose
# (the token is still printed in logs if you prefer to use it — just remove --NotebookApp.token='')
# Password and token can be overridden via JUPYTER_TOKEN env var in .env
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

# ── Persistent cache directory ───────────────────────────────────────────────
# sector_cache.csv and any output CSVs will be written here.
# Mount this as a volume so the cache survives container restarts.
RUN mkdir -p /app/data/charts /app/data/reports /app/data/runs

# ── Expose Jupyter port ──────────────────────────────────────────────────────
EXPOSE 8888

# ── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["jupyter", "notebook", \
     "--notebook-dir=/app/notebooks", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
