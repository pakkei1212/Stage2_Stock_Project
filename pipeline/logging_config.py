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
from contextlib import contextmanager
from datetime import datetime

from .config import CONFIG

_configured = False

# Values of these env vars are scrubbed from every log line on every handler.
_SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "JUPYTER_TOKEN")

_FORMAT = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SecretRedactingFilter(logging.Filter):
    """Defence in depth: replaces any secret env-var value that reaches a log
    record (e.g. inside an exception message) with ***. Secrets are read at
    filter time so values loaded after logging setup are covered too."""

    def filter(self, record):
        secrets = [v for v in (os.environ.get(k) for k in _SECRET_ENV_VARS) if v and len(v) >= 6]
        if not secrets:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = message
        for secret in secrets:
            redacted = redacted.replace(secret, "***")
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        exc_text = record.exc_text
        if exc_text:
            for secret in secrets:
                exc_text = exc_text.replace(secret, "***")
        if redacted != message or exc_text != record.exc_text:
            record.msg, record.args = redacted, None
            record.exc_text = exc_text
        return True


def setup_logging(config=CONFIG, run_timestamp=None, log_prefix="pipeline"):
    """Idempotent — safe to call more than once; only configures handlers on
    the first call. Returns the "pipeline" logger.

    ``run_timestamp`` (YYYYMMDD_HHMMSS) names the audit log file; pass the same
    value used for the run's chart subdirectory so the two line up. Defaults to
    the current time if not given."""
    logger = logging.getLogger("pipeline")

    global _configured
    if _configured:
        return logger

    os.makedirs(config["log_dir"], exist_ok=True)
    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(config["log_dir"], f"{log_prefix}_{timestamp}.log")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_FORMAT)
    console_handler.addFilter(SecretRedactingFilter())

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(_FORMAT)
    file_handler.addFilter(SecretRedactingFilter())

    logger.setLevel(config["log_level"])
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    _configured = True
    logger.info("Logging initialised (level=%s) — audit log: %s", config["log_level"], log_path)
    return logger


@contextmanager
def run_log_file(config, run_timestamp):
    """Temporarily tee the "pipeline" logger into data/logs/pipeline_<run_ts>.log.

    The long-running scheduler configures logging once (scheduler_<ts>.log);
    this gives each Stage A-G run its own audit file with the same naming
    run_pipeline uses, matching the data/charts/<run_ts>/ directory. Yields the path.
    """
    logger = logging.getLogger("pipeline")
    os.makedirs(config["log_dir"], exist_ok=True)
    path = os.path.join(config["log_dir"], f"pipeline_{run_timestamp}.log")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_FORMAT)
    handler.addFilter(SecretRedactingFilter())
    logger.addHandler(handler)
    try:
        yield path
    finally:
        logger.removeHandler(handler)
        handler.close()
