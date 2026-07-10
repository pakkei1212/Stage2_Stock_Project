"""Central logging setup for the weekly pipeline.

Every stage module logs via `logging.getLogger(__name__)`, which nests under
the "pipeline" logger configured here. Call `setup_logging()` once, at the
top of an entrypoint (pipeline/run_pipeline.py::main), before any stage runs.

Two sinks:
  - stdout, for `docker compose logs -f pipeline` / interactive runs
  - a timestamped file under data/logs/, one per run, for auditability
    (which tickers passed/failed each stage, config used, errors with
    tracebacks) that survives after the container log buffer rotates away.
"""
import logging
import os
import sys
from datetime import datetime

from .config import CONFIG

_configured = False


def setup_logging(config=CONFIG):
    """Idempotent — safe to call more than once; only configures handlers on
    the first call. Returns the "pipeline" logger."""
    logger = logging.getLogger("pipeline")

    global _configured
    if _configured:
        return logger

    os.makedirs(config["log_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["log_dir"], f"pipeline_{timestamp}.log")

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.setLevel(config["log_level"])
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    _configured = True
    logger.info("Logging initialised (level=%s) — audit log: %s", config["log_level"], log_path)
    return logger
