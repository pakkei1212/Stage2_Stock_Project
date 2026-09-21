"""Cross-process, non-blocking file lock (Stage H).

Uses an OS byte-range lock (``msvcrt.locking`` on Windows, ``fcntl.flock``
elsewhere) rather than "does the lock file exist": the OS releases the lock
when the holding process exits or crashes, so there is no stale-lock cleanup
and no PID liveness guessing.

Scope: processes on the same machine/filesystem. A Windows host process and a
Docker container sharing ./data through a bind mount do not reliably see each
other's locks — run the scheduler in one of those places, not both. The
SQLite active-claim index (history_store) is the second line of defence.
"""
import json
import logging
import os
import socket
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class LockHeld(Exception):
    """Raised when another process already holds the lock."""


class FileLock:
    def __init__(self, path):
        self.path = path
        self._fh = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fh = open(self.path, "a+")
        try:
            _lock(fh)
        except OSError:
            fh.close()
            raise LockHeld(self.path)
        # Informational only — the OS lock, not this text, is authoritative.
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({
                "pid": os.getpid(), "host": socket.gethostname(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            }))
            fh.flush()
        except OSError:
            pass
        self._fh = fh
        return self

    def release(self):
        if self._fh is None:
            return
        try:
            _unlock(self._fh)
        except OSError:
            logger.debug("lock release failed for %s", self.path, exc_info=True)
        finally:
            self._fh.close()
            self._fh = None

    @property
    def held(self):
        return self._fh is not None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


if os.name == "nt":
    import msvcrt

    def _lock(fh):
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh):
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
