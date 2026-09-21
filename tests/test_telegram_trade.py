"""Stage I Telegram interface: authorization, the Confirm/Cancel flow, duplicate
callback protection and the Record-Buy shortcut.

Pure logic — ``telegram_trade.handle_update`` never touches the network, and the
autouse fixture in conftest blocks sockets anyway.
"""
from datetime import timedelta

import pytest

from pipeline import telegram_trade as tt
from pipeline import trade_journal as tj
from pipeline import trade_store as ts
from stage_i_fakes import RUN_ID, SESSION, flat_bars, stage_g_row, trade_config

CHAT = 4242
OTHER_CHAT = 9999
USER = 777
NOW = ts.utc_now()


@pytest.fixture
def config(tmp_path):
    return trade_config(tmp_path)


@pytest.fixture
def store(config):
    return ts.open_store(config, create=True)


@pytest.fixture(autouse=True)
def allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_ID", str(CHAT))
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_ID", raising=False)


def message(text, chat=CHAT, user=USER):
    return {"message": {"chat": {"id": chat}, "from": {"id": user}, "text": text}}


def callback(data, chat=CHAT, user=USER, message_id=55):
    return {"callback_query": {"id": "cb1", "data": data, "from": {"id": user},
                               "message": {"message_id": message_id, "chat": {"id": chat}}}}


def seed_setup(config, store, symbol="CACC", pivot=621.52, price=610.0):
    # Final contraction low close to the pivot, so the persisted plan is usable
    # (trade plan OK, >0 shares) rather than SKIP_RISK_TOO_WIDE.
    lows = {"contraction_low_prices": [560.0, 585.0, 596.0]}
    tj.generate_setup_plans(RUN_ID, SESSION, [stage_g_row(symbol, pivot=pivot, price=price,
                                                          market_cap=5e9, metrics=lows)],
                            config, store=store, fetch=lambda symbols, cfg=None:
                            {s: flat_bars(60, price) for s in symbols})


def buy_and_confirm(config, store, text="/buy CACC 622.50 20"):
    pending = tt.handle_update(message(text), config, store=store, now=NOW)
    token = pending.replies[0].pending_token
    return pending, tt.handle_update(callback(f"ta:{token}:confirm"), config, store=store, now=NOW)


# ── authorization ─────────────────────────────────────────────────────────

def test_mutation_from_another_chat_is_rejected_and_writes_nothing(config, store):
    response = tt.handle_update(message("/buy CACC 622.50 20", chat=OTHER_CHAT), config,
                                store=store, now=NOW)
    assert response.authorized is False
    assert response.replies == []
    assert store.open_positions() == [] and store.pending_for_chat(OTHER_CHAT) == []


def test_read_command_from_another_chat_is_also_ignored(config, store):
    assert tt.handle_update(message("/positions", chat=OTHER_CHAT), config, store=store,
                            now=NOW).authorized is False


def test_callback_from_another_chat_cannot_confirm_an_action(config, store):
    seed_setup(config, store)
    pending = tt.handle_update(message("/buy CACC 622.50 20"), config, store=store, now=NOW)
    token = pending.replies[0].pending_token
    response = tt.handle_update(callback(f"ta:{token}:confirm", chat=OTHER_CHAT), config,
                                store=store, now=NOW)
    assert response.authorized is False
    assert store.open_positions() == []
    assert store.get_pending(token)["status"] == ts.PENDING


def test_chat_allowlist_falls_back_to_the_report_chat(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_ID", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1234")
    assert tt.allowed_chat_ids() == {"1234"}
    assert tt.authorize("1234")[0] is True
    assert tt.authorize("4321")[0] is False


def test_no_allowlist_means_nothing_is_authorized(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    allowed, reason = tt.authorize(CHAT)
    assert allowed is False and "no allowlist" in reason


def test_optional_user_allowlist_narrows_an_allowed_chat(monkeypatch, config, store):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_ID", "777")
    assert tt.authorize(CHAT, 777)[0] is True
    assert tt.authorize(CHAT, 778)[0] is False
    assert tt.handle_update(message("/buy CACC 1 1", user=778), config, store=store,
                            now=NOW).authorized is False


def test_multiple_allowlisted_chats_are_supported(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_ID", "111, 222")
    assert tt.allowed_chat_ids() == {"111", "222"}


# ── confirmation flow ─────────────────────────────────────────────────────

def test_buy_command_only_creates_a_pending_action(config, store):
    seed_setup(config, store)
    response = tt.handle_update(message("/buy CACC 622.50 20"), config, store=store, now=NOW)
    reply = response.replies[0]

    assert "Record position?" in reply.text
    assert "CACC" in reply.text and "622.50" in reply.text
    assert "Pivot: $621.52" in reply.text
    assert "Journal entry only — no brokerage order is placed." in reply.text
    assert reply.reply_markup["inline_keyboard"][0][0]["text"].endswith("Confirm")
    assert store.open_positions() == []                          # nothing written yet
    assert store.get_pending(reply.pending_token)["status"] == ts.PENDING


def test_confirm_records_the_trade(config, store):
    seed_setup(config, store)
    _, confirmed = buy_and_confirm(config, store)
    trade = store.open_trade("CACC")

    assert confirmed.callback_answer == "Recorded."
    assert "Position opened" in confirmed.replies[0].text
    assert trade["shares"] == 20.0 and trade["average_cost"] == 622.50
    assert trade["setup_run_id"] == RUN_ID
    assert confirmed.retire_markup == (CHAT, 55)


def test_a_duplicate_confirm_tap_writes_nothing_more(config, store):
    seed_setup(config, store)
    pending, _ = buy_and_confirm(config, store)
    token = pending.replies[0].pending_token

    again = tt.handle_update(callback(f"ta:{token}:confirm"), config, store=store, now=NOW)
    assert again.callback_answer == "Already handled."
    assert again.replies == []
    trade = store.open_trade("CACC")
    assert trade["shares"] == 20.0
    assert len(store.fills(trade["trade_id"])) == 1


def test_cancel_records_nothing(config, store):
    seed_setup(config, store)
    pending = tt.handle_update(message("/buy CACC 622.50 20"), config, store=store, now=NOW)
    token = pending.replies[0].pending_token
    cancelled = tt.handle_update(callback(f"ta:{token}:cancel"), config, store=store, now=NOW)

    assert "Cancelled" in cancelled.replies[0].text
    assert store.open_positions() == []
    assert store.get_pending(token)["status"] == ts.CANCELLED


def test_confirming_after_a_cancel_is_impossible(config, store):
    seed_setup(config, store)
    pending = tt.handle_update(message("/buy CACC 622.50 20"), config, store=store, now=NOW)
    token = pending.replies[0].pending_token
    tt.handle_update(callback(f"ta:{token}:cancel"), config, store=store, now=NOW)
    late = tt.handle_update(callback(f"ta:{token}:confirm"), config, store=store, now=NOW)
    assert late.callback_answer == "Already handled."
    assert store.open_positions() == []


def test_an_expired_confirmation_is_not_applied(config, store):
    seed_setup(config, store)
    pending = tt.handle_update(message("/buy CACC 622.50 20"), config, store=store, now=NOW)
    token = pending.replies[0].pending_token
    later = NOW + timedelta(minutes=int(config["trade_confirm_ttl_minutes"]) + 5)
    response = tt.handle_update(callback(f"ta:{token}:confirm"), config, store=store, now=later)
    assert response.callback_answer == "Expired."
    assert store.open_positions() == []


def test_sell_and_close_also_require_confirmation(config, store):
    store.record_buy("CACC", 600.0, 20)
    pending = tt.handle_update(message("/sell CACC 680 10"), config, store=store, now=NOW)
    assert "partial sell" in pending.replies[0].text.lower()
    assert store.open_trade("CACC")["shares"] == 20.0

    token = pending.replies[0].pending_token
    tt.handle_update(callback(f"ta:{token}:confirm"), config, store=store, now=NOW)
    assert store.open_trade("CACC")["shares"] == 10.0

    closing = tt.handle_update(message("/close CACC 705"), config, store=store, now=NOW)
    assert "Close position?" in closing.replies[0].text
    tt.handle_update(callback(f"ta:{closing.replies[0].pending_token}:confirm"), config,
                     store=store, now=NOW)
    assert store.open_trade("CACC") is None
    assert store.trades_for_symbol("CACC")[0]["status"] == ts.CLOSED


def test_selling_more_than_held_is_refused_before_any_pending_action(config, store):
    store.record_buy("CACC", 600.0, 5)
    response = tt.handle_update(message("/sell CACC 680 10"), config, store=store, now=NOW)
    assert "only 5 held" in response.replies[0].text
    assert store.pending_for_chat(CHAT) == []


def test_malformed_commands_return_usage_and_create_no_pending_action(config, store):
    for text in ("/buy CACC", "/buy CACC abc 10", "/buy 12 34 56"):
        response = tt.handle_update(message(text), config, store=store, now=NOW)
        assert "Usage" in response.replies[0].text
    assert store.pending_for_chat(CHAT) == []


# ── Open Position / View Position buttons ────────────────────────────────
# "Open Position" records MY fill in the local journal: fill price, then shares,
# then the usual Confirm. Nothing reaches trades/fills before Confirm.

def open_position(config, store, symbol="CACC"):
    return tt.handle_update(callback(f"op:{symbol}"), config, store=store, now=NOW)


def test_open_position_asks_for_the_fill_price_first(config, store):
    seed_setup(config, store)
    response = open_position(config, store)
    text = response.replies[0].text
    assert "Open Position — CACC" in text and "Actual fill price?" in text
    assert "Journal entry only — no brokerage order is placed." in text
    assert "Plan: trigger $622.14" in text                       # persisted plan, shown as-is
    assert store.pending_for_chat(CHAT, kind=tt.POSITION_INPUT)
    assert store.open_positions() == [] and store.recent_trades() == []


def test_open_position_then_asks_for_shares(config, store):
    seed_setup(config, store)
    open_position(config, store)
    asked = tt.handle_update(message("631"), config, store=store, now=NOW)
    text = asked.replies[0].text
    assert "Number of shares?" in text and "$631.00" in text
    plan = store.latest_setup_plan("CACC")
    assert f"Plan suggests {plan['suggested_shares']:g} shares." in text
    assert store.open_positions() == []


def test_open_position_rejects_non_numeric_input_without_advancing(config, store):
    seed_setup(config, store)
    open_position(config, store)
    reply = tt.handle_update(message("about 630"), config, store=store, now=NOW).replies[0].text
    assert "Actual fill price for CACC?" in reply
    assert store.pending_for_chat(CHAT, kind=tt.POSITION_INPUT)[0]["payload"]["step"] == "price"


def _to_confirmation(config, store, price="631", shares="20"):
    open_position(config, store)
    tt.handle_update(message(price), config, store=store, now=NOW)
    return tt.handle_update(message(shares), config, store=store, now=NOW).replies[0]


def test_open_position_confirmation_shows_the_persisted_plan(config, store):
    seed_setup(config, store)
    plan = store.latest_setup_plan("CACC")
    reply = _to_confirmation(config, store)
    text = reply.text
    assert "🟢 <b>Record position?</b>" in text and "Buy: 20 @ $631.00" in text
    assert "Position value: $12,620.00" in text
    assert "Setup: STRONG VCP · READY" in text and "Pivot: $621.52" in text
    assert f"Entry trigger: ${plan['entry_trigger_price']:,.2f}" in text
    assert f"Suggested stop: ${plan['suggested_initial_stop']:,.2f}" in text
    assert f"Suggested size: {plan['suggested_shares']:g} shares" in text
    assert f"Planned portion: {plan['position_portion_pct']:.1f}%" in text
    slippage = (631 - plan["entry_trigger_price"]) / plan["entry_trigger_price"] * 100
    assert f"Slippage from trigger: {slippage:+.2f}%" in text
    assert "Journal entry only — no brokerage order is placed." in text
    assert reply.reply_markup["inline_keyboard"][0][0]["text"].endswith("Confirm")
    assert store.open_positions() == [] and store.recent_trades() == []      # nothing before Confirm


def test_planned_vs_actual_warnings(config, store):
    seed_setup(config, store)
    plan = store.latest_setup_plan("CACC")
    below = _to_confirmation(config, store, price="600", shares="1").text
    assert "Fill is below planned breakout trigger" in below

    over_shares = int(plan["suggested_shares"]) + 5
    above = _to_confirmation(config, store, price="700", shares=str(over_shares)).text
    assert "Fill is above maximum chase price" in above
    assert "Actual shares exceed suggested position size" in above
    assert "Actual portfolio portion exceeds plan" in above
    assert store.open_positions() == []


def test_warnings_do_not_block_an_explicit_confirm_and_plan_is_kept_apart(config, store):
    seed_setup(config, store)
    plan_before = store.latest_setup_plan("CACC")
    reply = _to_confirmation(config, store, price="700", shares="3")
    assert "Fill is above maximum chase price" in reply.text
    done = tt.handle_update(callback(f"ta:{reply.pending_token}:confirm"), config, store=store, now=NOW)
    assert done.callback_answer == "Recorded."

    trade = store.open_trade("CACC")
    assert trade["average_cost"] == 700.0 and trade["shares"] == 3.0          # the ACTUAL trade
    assert trade["initial_stop"] == plan_before["suggested_initial_stop"]
    assert ts.loads(trade["entry_plan_json"])["entry_trigger_price"] == plan_before["entry_trigger_price"]
    assert "sizing" in ts.loads(trade["risk_plan_json"])
    opened = [e for e in store.events(trade["trade_id"]) if e["event_type"] == "OPEN"][0]
    record = ts.loads(opened["detail"])
    assert record["planned"]["entry_trigger_price"] == plan_before["entry_trigger_price"]
    assert record["actual"]["price"] == 700.0
    assert any("maximum chase" in w for w in record["warnings"])
    assert store.latest_setup_plan("CACC") == plan_before                    # plan row untouched


def test_open_position_confirm_records_once_and_duplicates_write_nothing(config, store):
    seed_setup(config, store)
    reply = _to_confirmation(config, store)
    first = tt.handle_update(callback(f"ta:{reply.pending_token}:confirm"), config, store=store, now=NOW)
    second = tt.handle_update(callback(f"ta:{reply.pending_token}:confirm"), config, store=store, now=NOW)
    assert first.callback_answer == "Recorded." and second.callback_answer == "Already handled."
    trade = store.open_trade("CACC")
    assert trade["shares"] == 20.0 and len(store.fills(trade["trade_id"])) == 1
    assert trade["setup_run_id"] == RUN_ID


def test_open_position_cancel_writes_nothing(config, store):
    seed_setup(config, store)
    reply = _to_confirmation(config, store)
    cancelled = tt.handle_update(callback(f"ta:{reply.pending_token}:cancel"), config, store=store, now=NOW)
    assert "nothing was recorded" in cancelled.replies[0].text
    assert store.open_positions() == [] and store.recent_trades() == []


def test_slash_cancel_abandons_the_conversation(config, store):
    seed_setup(config, store)
    open_position(config, store)
    tt.handle_update(message("/cancel"), config, store=store, now=NOW)
    assert tt.handle_update(message("631"), config, store=store, now=NOW).replies == []
    assert store.recent_trades() == []


def test_an_expired_open_position_request_does_not_advance(config, store):
    seed_setup(config, store)
    open_position(config, store)
    later = NOW + timedelta(minutes=int(config["trade_confirm_ttl_minutes"]) + 5)
    reply = tt.handle_update(message("631"), config, store=store, now=later).replies[0].text
    assert "expired" in reply
    assert store.pending_for_chat(CHAT, kind=tt.POSITION_INPUT) == []


def test_price_and_shares_in_one_reply_also_work(config, store):
    seed_setup(config, store)
    open_position(config, store)
    typed = tt.handle_update(message("631 20"), config, store=store, now=NOW)
    assert "Record position?" in typed.replies[0].text and store.open_positions() == []


def test_open_position_on_a_held_symbol_shows_it_instead_of_a_second_trade(config, store):
    seed_setup(config, store)
    store.record_buy("CACC", 625.0, 30)
    response = open_position(config, store)
    text = response.replies[0].text
    assert "💼 <b>CACC</b>" in text and "/buy CACC PRICE SHARES" in text
    assert store.pending_for_chat(CHAT, kind=tt.POSITION_INPUT) == []
    assert len(store.open_positions()) == 1


def test_view_position_button_equals_the_position_command(config, store):
    store.record_buy("CACC", 625.0, 30)
    via_button = tt.handle_update(callback("vp:CACC"), config, store=store, now=NOW).replies[0].text
    via_command = tt.handle_update(message("/position CACC"), config, store=store, now=NOW).replies[0].text
    assert via_button == via_command and "CACC" in via_button


def test_legacy_record_buy_button_starts_the_open_position_flow(config, store):
    seed_setup(config, store)
    response = tt.handle_update(callback("rb:CACC"), config, store=store, now=NOW)
    assert "Actual fill price?" in response.replies[0].text
    assert store.open_positions() == []          # an alert never implies a purchase


def test_open_position_from_another_chat_is_ignored(config, store):
    seed_setup(config, store)
    response = tt.handle_update(callback("op:CACC", chat=OTHER_CHAT), config, store=store, now=NOW)
    assert response.authorized is False and response.replies == []
    assert store.pending_for_chat(OTHER_CHAT) == []


def test_free_text_without_a_pending_open_position_is_ignored(config, store):
    assert tt.handle_update(message("622.50 20"), config, store=store, now=NOW).replies == []


def test_position_keyboards_carry_the_symbol():
    assert tt.open_position_keyboard("cacc")["inline_keyboard"][0][0]["callback_data"] == "op:CACC"
    assert tt.view_position_keyboard("cacc")["inline_keyboard"][0][0]["callback_data"] == "vp:CACC"


# ── read-only commands ────────────────────────────────────────────────────

def test_positions_command_formats_open_positions(config, store):
    trade_id, _ = store.record_buy("CACC", 625.0, 30)
    store.save_snapshot({"trade_id": trade_id, "symbol": "CACC", "trading_date": SESSION,
                         "close": 671.4, "pnl_pct": 7.4, "monitor_state": "HOLD",
                         "current_protective_level": 648.1})
    text = tt.handle_update(message("/positions"), config, store=store, now=NOW).replies[0].text
    assert "CACC" in text and "671.40" in text and "+7.4%" in text and "HOLD" in text
    assert "no orders are placed" in text


def test_positions_command_with_nothing_held(config, store):
    assert "No open positions" in tt.handle_update(message("/positions"), config, store=store,
                                                   now=NOW).replies[0].text


def test_position_detail_command(config, store):
    store.record_buy("CACC", 625.0, 30)
    text = tt.handle_update(message("/position CACC"), config, store=store, now=NOW).replies[0].text
    assert "CACC" in text and "Fills" in text


def test_trades_command_lists_recorded_trades(config, store):
    store.record_buy("CACC", 625.0, 30)
    text = tt.handle_update(message("/trades"), config, store=store, now=NOW).replies[0].text
    assert "CACC" in text and "OPEN" in text


def test_help_lists_the_commands(config, store):
    text = tt.handle_update(message("/help"), config, store=store, now=NOW).replies[0].text
    assert "/buy" in text and "/sell" in text and "/close" in text
    assert "no broker is connected" in text


def test_unknown_command_is_answered_with_help(config, store):
    assert "Unknown command" in tt.handle_update(message("/moon"), config, store=store,
                                                 now=NOW).replies[0].text


def test_command_parsing_strips_bot_suffix():
    assert tt.parse_command("/buy@my_bot CACC 10 1") == ("buy", ["CACC", "10", "1"])
    assert tt.parse_command("hello") == (None, [])
