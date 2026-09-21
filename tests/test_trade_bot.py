"""Stage I bot runner: polling, delivery and offsets, with a fake Bot API client.

The runner only moves bytes — the decisions are tested in test_telegram_trade.py.
"""
import pytest

from pipeline import trade_bot
from pipeline import trade_store as ts
from stage_i_fakes import trade_config

CHAT = 4242


@pytest.fixture(autouse=True)
def allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_ID", str(CHAT))


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


class FakeBotClient:
    """Serves queued getUpdates batches; records everything sent."""

    def __init__(self, batches=()):
        self.batches = [list(b) for b in batches]
        self.sent = []
        self.answered = []
        self.retired = []
        self.offsets = []
        self._next_id = 500

    def get_updates(self, offset=None, timeout=0):
        self.offsets.append(offset)
        return (self.batches.pop(0) if self.batches else []), 1

    def send_message(self, text, reply_markup=None, chat_id=None):
        self.sent.append((chat_id, text, reply_markup))
        self._next_id += 1
        return self._next_id, 1

    def answer_callback_query(self, callback_query_id, text=None):
        self.answered.append((callback_query_id, text))
        return None, 1

    def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
        self.retired.append((chat_id, message_id))
        return None, 1


def message_update(update_id, text, chat=CHAT):
    return {"update_id": update_id,
            "message": {"chat": {"id": chat}, "from": {"id": 1}, "text": text}}


def deps_for(config, client):
    return trade_bot.BotDeps(client_factory=lambda cfg: client, sleep=lambda s: None,
                             store=ts.open_store(config, create=True))


def test_poll_once_delivers_replies_and_advances_the_offset(config):
    client = FakeBotClient([[message_update(10, "/buy CACC 622.50 20")]])
    deps = deps_for(config, client)

    offset = trade_bot.poll_once(config, deps, client, None)
    assert offset == 11
    chat_id, text, markup = client.sent[0]
    assert chat_id == CHAT and "Record position?" in text
    assert markup["inline_keyboard"][0][0]["callback_data"].endswith(":confirm")
    assert deps.store.open_positions() == []            # still unconfirmed


def test_confirming_through_the_bot_records_the_trade(config):
    client = FakeBotClient([[message_update(1, "/buy CACC 622.50 20")]])
    deps = deps_for(config, client)
    trade_bot.poll_once(config, deps, client, None)
    token = deps.store.pending_for_chat(CHAT)[0]["token"]

    client.batches.append([{"update_id": 2, "callback_query": {
        "id": "cb", "data": f"ta:{token}:confirm", "from": {"id": 1},
        "message": {"message_id": 9, "chat": {"id": CHAT}}}}])
    trade_bot.poll_once(config, deps, client, 2)

    assert deps.store.open_trade("CACC")["shares"] == 20.0
    assert client.answered[-1] == ("cb", "Recorded.")
    assert client.retired == [(CHAT, 9)]


def test_updates_from_other_chats_are_neither_answered_nor_recorded(config):
    client = FakeBotClient([[message_update(3, "/buy CACC 1 1", chat=1111)]])
    deps = deps_for(config, client)
    trade_bot.poll_once(config, deps, client, None)
    assert client.sent == []
    assert deps.store.open_positions() == []


def test_the_pending_message_id_is_stored_for_later_retirement(config):
    client = FakeBotClient([[message_update(4, "/buy CACC 10 1")]])
    deps = deps_for(config, client)
    trade_bot.poll_once(config, deps, client, None)
    pending = deps.store.pending_for_chat(CHAT)[0]
    assert pending["message_id"] == str(client._next_id)


def test_startup_skips_the_backlog_by_default(config):
    backlog = [message_update(20, "/buy CACC 10 1")]
    client = FakeBotClient([backlog])
    deps = deps_for(config, client)
    trade_bot.run_forever(config, deps, max_iterations=1)

    assert client.offsets[0] == -1                      # backlog probe
    assert client.offsets[1] == 21                      # resumed past it
    assert deps.store.pending_for_chat(CHAT) == []
    assert client.sent == []


def test_backlog_is_processed_when_asked(config):
    client = FakeBotClient([[message_update(20, "/positions")]])
    deps = deps_for(config, client)
    trade_bot.run_forever(config, deps, max_iterations=1, process_backlog=True)
    assert any("No open positions" in text for _, text, _ in client.sent)


def test_a_handler_error_does_not_stop_the_loop(config, monkeypatch):
    monkeypatch.setattr(trade_bot.tt, "handle_update",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    client = FakeBotClient([[message_update(5, "/positions")]])
    deps = deps_for(config, client)
    assert trade_bot.poll_once(config, deps, client, None) == 6
    assert client.sent == []


def test_expired_confirmations_are_swept_each_iteration(config):
    client = FakeBotClient([])
    deps = deps_for(config, client)
    deps.store.create_pending("buy", str(CHAT), {"symbol": "CACC"},
                              expires_at=ts.utc_now().replace(year=2020))
    trade_bot.run_forever(config, deps, max_iterations=1, process_backlog=True)
    assert deps.store.pending_for_chat(CHAT) == []


def test_main_refuses_to_start_without_an_allowlist(monkeypatch, config):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert trade_bot.main([], config) == 2
