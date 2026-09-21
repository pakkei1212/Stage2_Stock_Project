"""Stage I: the Telegram trade-journal bot (long-polling runner).

A separate, optional process from the Stage-H scheduler — it only reads
Telegram updates and writes the trade journal, so it never touches a pipeline
run, the screening history, or any Stage A-G logic::

    python -m pipeline.trade_bot              # long-poll until stopped
    python -m pipeline.trade_bot --once       # drain what is queued and exit
    python -m pipeline.trade_bot --backlog    # also process updates queued before startup

All decisions live in :mod:`pipeline.telegram_trade` (authorization, parsing,
Confirm/Cancel, duplicate-callback protection); this module only moves bytes.
By default the backlog queued while the bot was down is skipped, so a stale
``/buy`` typed yesterday does not resurface as a fresh confirmation prompt.

There is no brokerage connection here or anywhere else: the bot records fills
the user reports and answers questions about them.
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from . import telegram_trade as tt
from . import trade_store as tstore
from .config import CONFIG
from .logging_config import setup_logging
from .telegram_notifier import TelegramClient, TelegramError

logger = logging.getLogger("pipeline.trade_bot")


@dataclass
class BotDeps:
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    sleep: Callable[[float], None] = time.sleep
    client_factory: Callable = lambda config: TelegramClient.from_env(config)
    store: Optional[object] = None


def _store(config, deps):
    if deps.store is None:
        deps.store = tstore.open_store(config, create=True)
    return deps.store


def _send(client, chat_id, reply, store):
    message_id, _ = client.send_message(reply.text, reply_markup=reply.reply_markup, chat_id=chat_id)
    if reply.pending_token:
        store.set_pending_message_id(reply.pending_token, message_id)
    return message_id


def handle_one(update, config, deps, client):
    """Process a single update and send whatever it produced. Never raises."""
    store = _store(config, deps)
    now = deps.now()
    chat_id = (((update.get("message") or update.get("edited_message") or {}).get("chat") or {}).get("id")
               or (((update.get("callback_query") or {}).get("message") or {}).get("chat") or {}).get("id"))
    try:
        response = tt.handle_update(update, config, store=store, now=now)
    except Exception:
        logger.exception("Trade bot: update handling failed — the bot keeps running")
        return None
    if not response.authorized:
        if response.callback_answer and update.get("callback_query"):
            _safe(client.answer_callback_query, update["callback_query"].get("id"),
                  response.callback_answer)
        return response
    try:
        if update.get("callback_query"):
            _safe(client.answer_callback_query, update["callback_query"].get("id"),
                  response.callback_answer)
        if response.retire_markup and response.retire_markup[1] is not None:
            _safe(client.edit_message_reply_markup, *response.retire_markup)
        for reply in response.replies:
            _send(client, chat_id, reply, store)
    except TelegramError as e:
        logger.warning("Trade bot: could not deliver a reply: %s", e)
    if response.applied:
        logger.info("Trade bot: recorded %s", response.applied.get("trade_id"))
    return response


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:                       # answering a callback is cosmetic
        logger.debug("Telegram side call failed: %s", type(e).__name__)
        return None


def poll_once(config, deps, client, offset, *, timeout=0):
    """One getUpdates round. Returns the next offset."""
    updates, _ = client.get_updates(offset=offset, timeout=timeout)
    for update in updates:
        handle_one(update, config, deps, client)
        offset = update["update_id"] + 1
    return offset


def run_forever(config=CONFIG, deps=None, *, max_iterations=None, process_backlog=False):
    deps = deps or BotDeps()
    store = _store(config, deps)
    client = deps.client_factory(config)
    poll_timeout = int(config.get("trade_bot_poll_seconds", 20))

    offset = None
    if not process_backlog:
        try:
            updates, _ = client.get_updates(offset=-1, timeout=0)
            if updates:
                offset = updates[-1]["update_id"] + 1
                logger.info("Skipping %d update(s) queued before startup (use --backlog to process them).",
                            len(updates))
        except TelegramError as e:
            logger.warning("Could not read the update backlog: %s", e)

    logger.info("Trade journal bot started (advisory only; no brokerage connection). "
                "Allowlisted chats: %d.", len(tt.allowed_chat_ids()))
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            expired = store.expire_pending(deps.now())
            if expired:
                logger.info("Expired %d unconfirmed action(s).", expired)
            offset = poll_once(config, deps, client, offset, timeout=poll_timeout)
        except TelegramError as e:
            logger.warning("Telegram polling error: %s", e)
            deps.sleep(min(60, poll_timeout))
        except Exception:
            logger.exception("Unexpected trade-bot error — continuing")
            deps.sleep(min(60, poll_timeout))
    return 0


def main(argv=None, config=CONFIG):
    parser = argparse.ArgumentParser(prog="python -m pipeline.trade_bot",
                                     description="Telegram trade journal bot (advisory only).")
    parser.add_argument("--once", action="store_true", help="Process queued updates once and exit.")
    parser.add_argument("--backlog", action="store_true",
                        help="Also process updates queued before startup (skipped by default).")
    args = parser.parse_args(argv)

    setup_logging(config, log_prefix="trade_bot")
    if not tt.allowed_chat_ids():
        logger.error("No allowlisted chat: set TELEGRAM_ALLOWED_CHAT_ID (or TELEGRAM_CHAT_ID) "
                     "before starting the trade bot.")
        return 2
    try:
        return run_forever(config, max_iterations=1 if args.once else None,
                           process_backlog=args.backlog or args.once)
    except TelegramError as e:
        logger.error("Trade bot could not start: %s", e)
        return 1
    except KeyboardInterrupt:                    # pragma: no cover - interactive
        logger.info("Trade bot stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
