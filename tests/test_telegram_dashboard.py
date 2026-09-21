"""The Telegram daily dashboard: Action Board, READY plans, buttons, compact
leaders/changes, and the separate debug message.

Presentation only. Every READY / entry / risk / sizing value comes from a
persisted ``setup_plans`` row and is printed as stored — the formatter never
recomputes a plan. No network, no broker, no Claude.
"""
import json

import pytest

from pipeline import entry_plan as ep
from pipeline import risk_model as rm
from pipeline import scheduler as sch
from pipeline import telegram_notifier as tg
from pipeline import trend_analysis as ta
from stage_h_fakes import FAKE_TOKEN, FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt
from stage_i_fakes import stage_g_row

SESSION = "2026-09-16"


def _row(symbol, rank, *, quality="strong", pivot_state="coiling_below_pivot", claude=None, sector="Healthcare",
         metrics=None, **claude_fields):
    row = stage_g_row(symbol, quality=quality, pivot_state=pivot_state, claude=claude, rank=rank,
                      metrics=metrics, claude_fields=claude_fields or None)
    row.update({"sector": sector, "composite_score": 13.0, "max_score": 14.0})
    return row


def _plan(symbol, state="READY", *, entry_status="AWAIT_BREAKOUT", plan_status="OK", shares=20.0, **kw):
    """A setup_plans row as trade_store returns it — deliberately odd numbers, so
    any recomputation would show up as a mismatch."""
    plan = {
        "run_id": "run-1", "trading_date": SESSION, "symbol": symbol, "setup_state": state,
        "vcp_numeric_quality": "strong", "pivot_state": "coiling_below_pivot",
        "entry_status": entry_status, "trade_plan_status": plan_status,
        "pivot_price": 629.69, "current_price": 620.48, "entry_trigger_price": 630.32,
        "maximum_chase_price": 648.58, "suggested_initial_stop": 594.2, "required_risk_pct": 5.73,
        "allowed_max_risk_pct": 7.0, "suggested_shares": shares, "position_portion_pct": 8.04,
        "entry_plan": {"distance_to_pivot_pct": -1.46, "required_breakout_volume_ratio": 1.5},
        "risk_plan": {"atr_pct": 2.31, "atr20": 14.33},
        "sizing": {"binding_constraint": "portion"},
    }
    plan.update(kw)
    return plan


def _trend(entries=(), new=(), dropped=(), has_prev=True):
    return ta.TrendReport(current_session=SESSION, previous_session="2026-09-15" if has_prev else None,
                          sessions=[], missing_sessions=[], top_n=10, entries=list(entries),
                          new_entries=list(new), dropped=list(dropped))


def _medp_halo_frhc():
    """A realistic mixed run: MEDP READY, INCY watch, HALO + FRHC disagreements."""
    rows = [
        _row("SLDE", 1, quality="absent", sector="Financials"),
        _row("HALO", 4, quality="weak", claude=True, pattern_stage="forming", confidence="medium",
             entry_recommendation="wait_for_breakout",
             rationale="One coherent forming base with shrinking pullbacks into resistance."),
        _row("MEDP", 5, metrics={"contraction_pcts": [16.9, 8.3], "final_contraction_confirmed": False}),
        _row("INCY", 13, quality="acceptable", pivot_state="far_below_pivot",
             metrics={"contraction_pcts": [12.7, 7.0, 6.7], "final_contraction_confirmed": False,
                      "tightening_quality": "acceptable", "volume_dryup_quality": "weak",
                      "pct_from_pivot": -4.6}),
        _row("FRHC", 16, quality="acceptable", pivot_state="far_below_pivot", claude=False,
             pattern_stage="not_present", confidence="medium", entry_recommendation="avoid",
             rationale="Choppy advancing structure rather than one coherent VCP."),
    ]
    plans = {"MEDP": _plan("MEDP"),
             "INCY": _plan("INCY", "NOT_READY", entry_status=None, plan_status=None),
             "HALO": _plan("HALO", "NOT_READY", entry_status=None, plan_status=None),
             "FRHC": _plan("FRHC", "NOT_READY", entry_status=None, plan_status=None)}
    return rows, plans


def _text(rows, plans, **kw):
    return "\n".join(tg.build_report(SESSION, {"universe": 4331}, rows, kw.pop("trend", _trend()),
                                     plans=plans, **kw))


# ── Action Board ─────────────────────────────────────────────────────────

def test_action_board_classifies_ready_watch_review_from_persisted_states():
    rows, plans = _medp_halo_frhc()
    positions = [{"symbol": "CACC", "monitor_state": "HOLD"}, {"symbol": "NVDA", "monitor_state": "EXIT_REVIEW"}]
    text = _text(rows, plans, positions=positions)
    assert ("🎯 <b>ACTION BOARD</b>\n🟢 Ready: MEDP\n🟡 Watch: INCY\n🟠 Review: FRHC, HALO\n"
            "💼 Open positions: 2 (1 need attention)") in text
    # Review follows VCP priority (numeric-acceptable FRHC before weak HALO), then rank
    order = [text.index(h) for h in ("ACTION BOARD", "READY — MEDP", "WATCH — INCY", "REVIEW — FRHC", "REVIEW — HALO",
                                     "OPEN POSITIONS", "OVERALL LEADERS", "WATCHLIST CHANGES")]
    assert order == sorted(order)


@pytest.mark.parametrize("plan, bucket", [
    (_plan("X"), tg.READY),
    (_plan("X", "BREAKOUT_TRIGGERED", entry_status="BREAKOUT_ACTIONABLE"), tg.READY),
    (_plan("X", plan_status="SKIP_RISK_TOO_WIDE", shares=0), tg.WATCH),
    (_plan("X", plan_status="NO_PRICE_DATA", shares=0), tg.WATCH),
    (_plan("X", entry_status="EXTENDED_NO_CHASE"), tg.WATCH),
    (_plan("X", shares=0), tg.WATCH),
    (_plan("X", "MISSED_EXTENDED"), tg.WATCH),
    (_plan("X", "MANUAL_REVIEW"), tg.REVIEW),
    (_plan("X", "NOT_READY"), tg.WATCH),          # numeric strong, below the entry range
    (None, tg.WATCH),                             # no journal: never READY without a stored plan
])
def test_bucket_comes_from_the_persisted_plan(plan, bucket):
    assert tg.classify_setup(_row("X", 1), plan).bucket == bucket


def test_ranked_rows_without_a_vcp_are_no_setup():
    assert tg.classify_setup(_row("X", 1, quality="absent"), None).bucket == tg.NO_SETUP
    unanalysed = _row("Y", 2)
    unanalysed["vcp_status"] = None
    assert tg.classify_setup(unanalysed, _plan("Y")).bucket == tg.NO_SETUP


# ── READY: the persisted Stage-I plan, verbatim ──────────────────────────

def test_ready_block_renders_the_persisted_entry_risk_and_sizing_plan():
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans)
    block = text[text.index("🟢 <b>READY — MEDP</b>"):text.index("🟡 <b>WATCH")]
    for line in ("Rank #5 · Healthcare · 13/14",
                 "VCP: STRONG · 16.9 → 8.3% (forming)",
                 "Volume: STRONG",
                 "Price: 620.48 · Pivot: 629.69 · -1.5%",
                 "📌 <b>Entry plan</b>", "Trigger: 630.32 · Max chase: 648.58",
                 "Breakout volume: ≥1.50× 20D avg", "Status: AWAIT BREAKOUT",
                 "🛡 <b>Risk plan</b>", "Initial stop: 594.20", "Required risk: 5.7% · Allowed max: 7.0%",
                 "ATR: 2.3% (14.33)",
                 "📦 <b>Position plan</b>", "Suggested: 20 shares · 8.0% of portfolio",
                 "Binding constraint: portfolio portion", "Trade plan: OK",
                 "Claude: not requested (clear numeric setup)",
                 "Action: <b>WAIT FOR BREAKOUT</b>"):
        assert line in block, line


def test_formatter_never_recomputes_a_plan(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the Telegram formatter must not recompute Stage-I plans")
    for module, name in ((ep, "build_entry_plan"), (rm, "build_risk_plan"), (rm, "position_size"),
                         (rm, "size_for_plan")):
        monkeypatch.setattr(module, name, boom)
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans)
    assert "Suggested: 20 shares" in text                  # the stored number, not a fresh one
    tg.build_chart_caption(rows[2], plans["MEDP"])
    tg.build_debug_report(SESSION, {}, rows, plans=plans)


def test_ready_setup_already_held_says_so():
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans, open_trades=[{"symbol": "MEDP", "shares": 20, "average_cost": 631.0}])
    assert "💼 Already held: 20 @ 631.00 — see View Position" in text
    assert "#5 <b>MEDP</b> · 13/14 · Healthcare · 🟢 Ready · 💼" in text


# ── WATCH / REVIEW are readable ──────────────────────────────────────────

def test_watch_reasons_are_readable():
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans)
    block = text[text.index("🟡 <b>WATCH — INCY</b>"):text.index("🟠 <b>REVIEW")]
    assert ("🟡 <b>WATCH — INCY</b>\nRank #13 · Healthcare · 13/14\n"
            "VCP: ACCEPTABLE · 12.7 → 7.0 → 6.7% (forming)\nVolume: WEAK · Pivot: -4.6%\nWhy watch:\n"
            "✓ price contractions are tightening\n⚠ volume confirmation is weak\n"
            "⚠ final contraction still forming\n⚠ price 4.6% below pivot — not yet in entry range") in block
    assert block.rstrip().endswith("Action: <b>WAIT</b>")


def test_review_disagreements_are_readable():
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans)
    halo = text[text.index("🟠 <b>REVIEW — HALO</b>"):text.index("📊")]
    assert "Numeric: WEAK" in halo and "Claude: VCP · forming · wait_for_breakout · medium" in halo
    assert "Why review:\nNumeric contraction quality is WEAK, but Claude sees a forming VCP." in halo
    assert "Claude: “One coherent forming base" in halo and "Action: <b>MANUAL REVIEW</b>" in halo
    frhc = text[text.index("🟠 <b>REVIEW — FRHC</b>"):text.index("🟠 <b>REVIEW — HALO</b>")]
    assert "Numeric detector rates it ACCEPTABLE, but Claude does not see one coherent VCP." in frhc
    assert "Choppy advancing structure" in frhc and "Action: <b>NOT READY</b>" in frhc


def test_skip_risk_too_wide_is_explained():
    view = tg.classify_setup(_row("X", 1), _plan("X", plan_status="SKIP_RISK_TOO_WIDE", shares=0,
                                                   required_risk_pct=9.34, allowed_max_risk_pct=8.0))
    assert view.action == "NO TRADE — STOP TOO WIDE"
    assert "⚠ stop needs 9.3% risk, above the 8.0% allowed (SKIP_RISK_TOO_WIDE)" in view.reasons


# ── buttons ──────────────────────────────────────────────────────────────

def _callbacks(markup):
    return [b["callback_data"] for r in (markup or {}).get("inline_keyboard", []) for b in r]


def test_ready_without_a_trade_gets_open_position():
    rows, plans = _medp_halo_frhc()
    assert _callbacks(tg.build_report_keyboard(rows, plans, [], [])) == ["op:MEDP"]


def test_ready_with_an_open_trade_gets_view_position():
    rows, plans = _medp_halo_frhc()
    keyboard = tg.build_report_keyboard(rows, plans, [{"symbol": "MEDP", "shares": 20}], [])
    assert _callbacks(keyboard) == ["vp:MEDP"]
    assert keyboard["inline_keyboard"][0][0]["text"] == "💼 View Position · MEDP"


@pytest.mark.parametrize("plan", [
    _plan("X", "MANUAL_REVIEW"), _plan("X", "NOT_READY"), _plan("X", "MISSED_EXTENDED"),
    _plan("X", plan_status="SKIP_RISK_TOO_WIDE", shares=0),
])
def test_non_ready_states_get_no_open_position_button(plan):
    assert tg.build_report_keyboard([_row("X", 1)], {"X": plan}, [], []) is None


def test_open_positions_get_view_buttons_and_chart_symbols_are_not_repeated():
    rows, plans = _medp_halo_frhc()
    positions = [{"symbol": "CACC", "monitor_state": "HOLD"}]
    keyboard = tg.build_report_keyboard(rows, plans, [{"symbol": "CACC"}], positions, exclude={"MEDP"})
    assert _callbacks(keyboard) == ["vp:CACC"]


def test_ready_chart_caption_and_button():
    rows, plans = _medp_halo_frhc()
    caption = tg.build_chart_caption(rows[2], plans["MEDP"])
    assert caption.startswith("🟢 <b>MEDP</b> · READY · Rank 5")
    for line in ("Price: 620.48 · Pivot: 629.69 · -1.5%", "Entry trigger: 630.32", "Initial stop: 594.20",
                 "Suggested size: 20 shares", "Action: WAIT FOR BREAKOUT"):
        assert line in caption
    assert tg.position_keyboard("MEDP", held=False)["inline_keyboard"] == [
        [{"text": "🟢 Open Position", "callback_data": "op:MEDP"}]]
    assert tg.position_keyboard("MEDP", held=True)["inline_keyboard"][0][0]["callback_data"] == "vp:MEDP"


# ── positions, leaders, changes ──────────────────────────────────────────

def test_hold_position_is_compact_and_problems_get_detail():
    hold = {"symbol": "CACC", "monitor_state": "HOLD", "shares": 30, "average_cost": 625.0, "close": 671.4,
            "pnl_pct": 7.4, "current_protective_level": 648.1, "ema10": 660.0, "atr_pct": 2.2,
            "reasons_json": json.dumps(["above EMA21"])}
    assert tg._position_block(hold) == ("<b>CACC</b>\n30 shares @ 625.00\nClose: 671.40 · +7.4%\n"
                                        "Protective level: 648.10\nStatus: HOLD")
    watch = {**hold, "monitor_state": "WATCH", "reasons_json": json.dumps(["closed below EMA21"])}
    block = tg._position_block(watch)
    assert block.startswith("⚠ <b>CACC</b> — WATCH") and "EMA10: 660.00" in block
    assert "Why:\n• closed below EMA21" in block
    stop = {**hold, "monitor_state": "STOP_TRIGGERED", "reasons_json": json.dumps(["close below stop"])}
    assert "Reason:\n• close below stop" in tg._position_block(stop)


def test_corporate_action_review_still_leads_the_positions_section():
    snaps = [{"symbol": "AAA", "monitor_state": "HOLD"},
             {"symbol": "BBB", "monitor_state": "CORPORATE_ACTION_REVIEW"}]
    blocks = tg.build_positions_blocks(snaps)
    assert blocks[0] == "💼 <b>OPEN POSITIONS</b>" and "BBB" in blocks[1] and "AAA" in blocks[2]


def test_watchlist_changes_are_compact():
    entries = [ta.SymbolTrend("SLDE", 1, 14, previous_rank=1, top_n_streak=5, category=ta.PERSISTENT),
               ta.SymbolTrend("DXCM", 2, 14, previous_rank=3, top_n_streak=5, category=ta.PERSISTENT),
               ta.SymbolTrend("AVAH", 9, 12, previous_rank=None, is_new_entry=True, category=ta.NEW),
               ta.SymbolTrend("KNSA", 10, 12, previous_rank=18, is_new_entry=True, category=ta.RISING),
               ta.SymbolTrend("MEDP", 5, 13, previous_rank=9, top_n_streak=2, category=ta.RISING),
               ta.SymbolTrend("HALO", 8, 13, previous_rank=4, top_n_streak=3, category=ta.FALLING)]
    trend = _trend(entries, new=["AVAH", "KNSA"],
                   dropped=[ta.DroppedSymbol("UFPT", 6, 14), ta.DroppedSymbol("IBKR", 10, None)])
    block = tg._changes_block(trend)
    assert block == ("📈 <b>WATCHLIST CHANGES</b> <i>(vs 2026-09-15)</i>\n"
                     "🆕 New: AVAH #9, KNSA #18 → #10\n"
                     "⬆ Up: MEDP #9 → #5\n"
                     "⬇ Down: HALO #4 → #8\n"
                     "🚪 Left Top 10: UFPT #6 → #14, IBKR #10 → unranked\n"
                     "🔥 Persistent: SLDE, DXCM")


def test_quiet_day_says_nothing_changed():
    assert "No changes in the Top 10." in tg._changes_block(_trend())


# ── the separate debug message ───────────────────────────────────────────

STAGE_COUNTS = {"universe": 4331, "liquidity": 1273, "market_cap": 776, "stage_c": 367, "stage_d": 353,
                "trend_template_pass": 91, "ranked": 353, "vcp_analyzed": 5}


def test_main_report_contains_no_stage_summary():
    rows, plans = _medp_halo_frhc()
    text = _text(rows, plans)
    for leaked in ("Stage summary", "Universe", "Liquidity:", "Stage D screened", "VCP review union",
                   "Numeric vs Claude", "Claude reviewed"):
        assert leaked not in text


def test_debug_report_keeps_every_count():
    rows, plans = _medp_halo_frhc()
    plans["ZZZ"] = _plan("ZZZ", plan_status="SKIP_RISK_TOO_WIDE", shares=0)
    plans["YYY"] = _plan("YYY", "MANUAL_REVIEW")
    plans["XXX"] = _plan("XXX", "MISSED_EXTENDED")
    snaps = [{"symbol": "A", "monitor_state": "HOLD"}, {"symbol": "B", "monitor_state": "EXIT_REVIEW"},
             {"symbol": "C", "monitor_state": "STOP_TRIGGERED"}, {"symbol": "D", "monitor_state": "TIGHTEN_PROTECTION"}]
    actions = [{"symbol": "A", "status": "APPLIED"}, {"symbol": "B", "status": "REVIEW_REQUIRED"},
               {"symbol": "C", "status": "AWAITING_CONFIRMATION"}]
    run = {"run_id": "abc123", "status": "RESULTS_PERSISTED", "is_canonical": 1,
           "pipeline_started_at_utc": "2026-09-17T01:15:00+00:00", "completed_at_utc": "2026-09-17T01:27:31+00:00",
           "data_ready_latest_bar": "2026-09-16", "data_ready_attempts": 1, "config_hash": "cafebabe"}
    text = "\n".join(tg.build_debug_report(SESSION, STAGE_COUNTS, rows, run=run, plans=plans, positions=snaps,
                                           open_trade_count=4, corporate_actions=actions,
                                           config={"vcp_live_analysis": True, "vcp_llm_review_mode": "selective"},
                                           buttons_enabled=True, report_status="SENT (1 message(s))"))
    assert text.startswith("🛠 <b>DEBUG — DAILY PIPELINE</b>\nSession: 2026-09-16")
    for line in ("Universe: 4,331", "Liquidity: 1,273", "Market cap: 776", "Top sectors: 367",
                 "Stage D screened: 353", "Trend template 8/8: 91", "Ranked: 353",
                 "VCP metrics computed: 5", "Numeric VCP positive: 3", "Claude reviewed: 2",
                 "Claude VCP positive: 1", "VCP review union: 4", "Numeric vs Claude: 0 agree · 2 disagree",
                 # Stage I: "VCP union = 4, but how many became READY?"
                 "Setup plans: 7", "READY: 2", "BREAKOUT_TRIGGERED: 0", "NOT_READY: 3", "MANUAL_REVIEW: 1",
                 "MISSED_EXTENDED: 1", "Trade plan OK: 1", "SKIP_RISK_TOO_WIDE: 1",
                 "Dashboard: 🟢 1 ready · 🟡 1 watch · 🟠 2 review", "READY setup buttons: 1",
                 "Open positions: 4", "Positions monitored: 4", "HOLD: 1", "WATCH: 0", "TIGHTEN: 1",
                 "EXIT REVIEW: 1", "STOP TRIGGERED: 1",
                 "Detected: 3", "Applied automatically: 1", "Review required: 2",
                 "Run id: <code>abc123</code>", "Status: RESULTS_PERSISTED (canonical)", "Runtime: 12m 31s",
                 "Data readiness: latest bar 2026-09-16 · 1 attempt(s)", "Claude mode: selective",
                 "Main report: SENT (1 message(s))"):
        assert line in text, line


def test_debug_report_never_contains_secrets_or_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-real")
    config = make_config(tmp_path)
    rows, plans = _medp_halo_frhc()
    run = {"run_id": "r1", "status": "NOTIFIED", "config_json": json.dumps({"log_dir": str(tmp_path)})}
    text = "\n".join(tg.build_debug_report(SESSION, STAGE_COUNTS, rows, run=run, plans=plans, config=config))
    for secret in (FAKE_TOKEN, "sk-ant", str(tmp_path), "TELEGRAM_BOT_TOKEN", "config_json"):
        assert secret not in text


def _run(tmp_path, telegram, **cfg):
    config = make_config(tmp_path, telegram_enabled=True, **cfg)
    deps = make_deps(FakeClock(sgt("2026-09-15 09:15")), FakeScreen(["AAA", "BBB"]),
                     latest_bar=sgt("2026-09-14 12:00").date(), telegram=telegram)
    return config, sch.check_and_run("scheduled", config, deps)


def test_debug_message_is_a_separate_delivery(tmp_path):
    telegram = FakeTelegram()
    config, outcome = _run(tmp_path, telegram)
    assert [m[:1] for m in telegram.messages] == ["📈", "🛠"]
    assert "Universe: 4,300" in telegram.messages[1] and "Universe" not in telegram.messages[0]
    kinds = [(n["kind"], n["status"]) for n in sch.hs.HistoryStore(config["history_db_path"])
             .notifications_for_run(outcome.run_id)]
    assert ("report", "SENT") in kinds and ("debug", "SENT") in kinds


def test_debug_disabled_sends_no_debug_message(tmp_path):
    telegram = FakeTelegram()
    _run(tmp_path, telegram, telegram_debug_summary=False)
    assert len(telegram.messages) == 1 and not telegram.messages[0].startswith("🛠")


def test_debug_send_failure_does_not_fail_the_run(tmp_path):
    class DebugFails(FakeTelegram):
        def send_message(self, text, reply_markup=None, chat_id=None):
            if text.startswith("🛠"):
                raise tg.TelegramError("Telegram sendMessage failed: HTTP 502: Bad Gateway", attempts=3)
            return super().send_message(text, reply_markup=reply_markup, chat_id=chat_id)

    config, outcome = _run(tmp_path, DebugFails())
    assert outcome.decision == sch.COMPLETED and outcome.notification_status == "SENT"
    store = sch.hs.HistoryStore(config["history_db_path"])
    assert store.get_run(outcome.run_id)["status"] == sch.hs.NOTIFIED
    kinds = [(n["kind"], n["status"]) for n in store.notifications_for_run(outcome.run_id)]
    assert ("debug", "FAILED") in kinds and ("report", "SENT") in kinds


def test_debug_config_key_is_operational_not_part_of_the_config_hash():
    from pipeline import run_metadata
    assert "telegram_debug_summary".startswith(run_metadata.OPERATIONAL_KEY_PREFIXES)
