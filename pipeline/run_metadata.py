"""Run reproducibility metadata + per-session manifest (Stage H).

Captures what is needed to tell "rank changed because the market changed"
from "rank changed because the screen changed": git commit (+ dirty flag), a
redacted config snapshot, a hash over the *screening* keys only, and library
versions that affect upstream data.
"""
import hashlib
import json
import logging
import os
import platform
import re
import subprocess

from .history_store import to_jsonable

logger = logging.getLogger(__name__)

# Keys matching this are never written to SQLite / manifests.
_SECRET_KEY_RE = re.compile(r"(api_?key|token|secret|password|chat_id|credential)", re.IGNORECASE)

# Operational (post-Stage-G / housekeeping) keys: excluded from config_hash so
# that toggling Telegram or moving a directory doesn't look like a screen change.
OPERATIONAL_KEY_PREFIXES = (
    "scheduler_", "session_close_", "run_", "data_ready_", "deferred_", "history_", "trend_",
    "telegram_", "log_", "report_dir", "chart_dir", "ohlcv_cache_dir",
    "market_cap_cache_path",
    # Stage I (trade journal / position monitor) is advisory and post-Stage-G:
    # changing a risk or sizing parameter must not look like a screen change.
    "trade_",
)

SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "JUPYTER_TOKEN")


def redacted_config(config):
    return {k: to_jsonable(v) for k, v in sorted(config.items()) if not _SECRET_KEY_RE.search(k)}


def config_hash(config):
    screening = {k: v for k, v in redacted_config(config).items()
                 if not k.startswith(OPERATIONAL_KEY_PREFIXES)}
    blob = json.dumps(screening, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def git_info(repo_dir="."):
    """(commit, dirty) — falls back to $GIT_COMMIT (e.g. baked into an image)."""
    commit = dirty = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
                                capture_output=True, text=True, timeout=10, check=True).stdout
        dirty = bool(status.strip())
    except Exception:
        logger.debug("git metadata unavailable", exc_info=True)
    if not commit:
        commit = os.environ.get("GIT_COMMIT") or None
    return commit, dirty


def environment_info():
    info = {"python": platform.python_version()}
    for pkg in ("pandas", "numpy", "yfinance", "anthropic", "exchange_calendars"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:
            info[pkg] = None
    return info


def collect(config):
    commit, dirty = git_info()
    snapshot = redacted_config(config)
    return {
        "git_commit": commit,
        "git_dirty": None if dirty is None else int(dirty),
        "config_hash": config_hash(config),
        "config_json": json.dumps(snapshot, sort_keys=True, default=str),
        "environment_json": json.dumps(environment_info(), sort_keys=True),
    }


def manifest_path(config, trading_date):
    return os.path.join(config["run_manifest_dir"], str(trading_date), "run.json")


def write_manifest(config, run, extra=None):
    """Write data/runs/<trading_date>/run.json from a pipeline_runs row. Never raises."""
    try:
        path = manifest_path(config, run["trading_date"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        doc = {
            "run_id": run.get("run_id"),
            "trading_date": run.get("trading_date"),
            "status": run.get("status"),
            "is_canonical": bool(run.get("is_canonical")),
            "trigger": run.get("trigger"),
            "forced": bool(run.get("forced")),
            "started_at_utc": run.get("started_at_utc"),
            "completed_at_utc": run.get("completed_at_utc"),
            "local_execution_timezone": run.get("local_execution_timezone"),
            "local_started_at": run.get("local_started_at"),
            "data_ready_latest_bar": run.get("data_ready_latest_bar"),
            "git_commit": run.get("git_commit"),
            "git_dirty": run.get("git_dirty"),
            "config_hash": run.get("config_hash"),
            "stage_counts": json.loads(run["stage_counts_json"]) if run.get("stage_counts_json") else None,
            "artifacts": json.loads(run["artifacts_json"]) if run.get("artifacts_json") else None,
            "history_db": config.get("history_db_path"),
            **(extra or {}),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=False, default=str)
        os.replace(tmp, path)
        return path
    except Exception:
        logger.warning("Failed to write run manifest", exc_info=True)
        return None
