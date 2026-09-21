"""Block idle sleep while a Stage A-G run is in progress (Stage H, Windows).

A full run takes an hour or more and starts at 09:15 local, often while nobody
is at the keyboard. If the laptop's idle timer suspends it mid-run, every
per-ticker fetch in the sleep/wake window fails DNS and "degrades" (scored 0
fundamentals, dropped bars), and the run can still become canonical with a
distorted ranking.

``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` resets the
system idle timer for as long as it is held — the same mechanism media players
use. It does NOT override a deliberate sleep (lid close, Start > Sleep, power
button) or a critical-battery hibernate. Elsewhere this is a no-op.
"""
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_setter():
    if os.name != "nt":
        return None
    import ctypes
    return ctypes.windll.kernel32.SetThreadExecutionState


@contextmanager
def keep_awake(setter=None):
    """Hold the idle-sleep block for the duration of the ``with`` body. Never raises.

    The execution state is per thread, so release happens on the same thread
    that acquired it (the caller's ``with``). Yields True if the block is held.
    """
    setter = setter if setter is not None else _windows_setter()
    held = False
    if setter is not None:
        try:
            held = bool(setter(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        except Exception:
            logger.debug("SetThreadExecutionState failed", exc_info=True)
        if held:
            logger.info("Idle sleep blocked for the duration of the run "
                        "(closing the lid or choosing Sleep still suspends the machine).")
        else:
            logger.warning("Could not block idle sleep; a laptop sleeping mid-run degrades the results.")
    try:
        yield held
    finally:
        if held:
            try:
                setter(ES_CONTINUOUS)
            except Exception:
                logger.debug("SetThreadExecutionState release failed", exc_info=True)
