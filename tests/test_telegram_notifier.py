"""Stage H: Telegram client (mocked HTTP), retries, redaction, HTML escaping
and deterministic chunking. Never contacts Telegram."""
import logging

import pytest
import requests

from pipeline import telegram_notifier as tg
from pipeline import trend_analysis as ta
from stage_h_fakes import FAKE_TOKEN


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class _Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "data": data, "files": files})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


OK = _Resp(200, {"ok": True, "result": {"message_id": 42}})


def _client(session, sleeps=None, max_attempts=3):
    return tg.TelegramClient(FAKE_TOKEN, "999", session=session, max_attempts=max_attempts,
                             sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_missing_credentials_raise_telegram_error():
    with pytest.raises(tg.TelegramError):
        tg.TelegramClient("", "123", session=_Session())


def test_normal_message_uses_html_parse_mode_and_returns_message_id():
    session = _Session(OK)
    mid, attempts = _client(session).send_message("<b>hi</b>")
    assert (mid, attempts) == (42, 1)
    assert session.calls[0]["data"]["parse_mode"] == "HTML"
    assert session.calls[0]["data"]["chat_id"] == "999"


def test_retry_on_server_error_then_success():
    sleeps = []
    session = _Session(_Resp(502, {"ok": False, "description": "Bad Gateway"}), OK)
    mid, attempts = _client(session, sleeps).send_message("x")
    assert mid == 42 and attempts == 2 and sleeps == [2.0]


def test_rate_limit_honours_retry_after():
    sleeps = []
    session = _Session(_Resp(429, {"ok": False, "description": "Too Many Requests",
                                   "parameters": {"retry_after": 7}}), OK)
    _client(session, sleeps).send_message("x")
    assert sleeps == [7.0]


def test_client_error_is_not_retried():
    session = _Session(_Resp(400, {"ok": False, "description": "Bad Request: can't parse entities"}))
    with pytest.raises(tg.TelegramError) as exc:
        _client(session).send_message("x")
    assert exc.value.attempts == 1 and len(session.calls) == 1


def test_persistent_failure_is_bounded():
    sleeps = []
    session = _Session(*[_Resp(503, {"ok": False})] * 3)
    with pytest.raises(tg.TelegramError) as exc:
        _client(session, sleeps).send_message("x")
    assert exc.value.attempts == 3 and len(session.calls) == 3 and len(sleeps) == 2


def test_token_never_appears_in_errors_or_logs(caplog):
    # requests embeds the full (token-bearing) URL in connection errors.
    url_error = requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: "
                                         f"/bot{FAKE_TOKEN}/sendMessage")
    session = _Session(url_error, url_error, url_error)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(tg.TelegramError) as exc:
        _client(session).send_message("x")
    assert FAKE_TOKEN not in str(exc.value)
    assert FAKE_TOKEN not in caplog.text
    assert "***" in str(exc.value)


def test_send_photo_posts_multipart_with_caption(tmp_path):
    png = tmp_path / "AAA.png"
    png.write_bytes(b"\x89PNG")
    session = _Session(OK)
    _client(session).send_photo(str(png), caption="<b>#1 AAA</b>")
    call = session.calls[0]
    assert call["url"].endswith("/sendPhoto") and "photo" in call["files"]
    assert call["data"]["caption"] == "<b>#1 AAA</b>"


def test_send_text_chunks_reports_partial_failure():
    session = _Session(OK, _Resp(400, {"ok": False, "description": "nope"}))
    result = tg.send_text_chunks(_client(session), ["one", "two"])
    assert not result.ok and result.message_ids == [42] and "nope" in result.error


# ── formatting / chunking ────────────────────────────────────────────────

def test_short_output_is_one_message():
    chunks = tg.chunk_blocks("HEADER", ["a", "b"])
    assert chunks == ["HEADER\n\na\n\nb"]


def test_long_output_splits_on_block_boundaries_with_continuation_header():
    blocks = [f"<b>{i}. SYM{i}</b>\n" + "x" * 300 for i in range(40)]
    chunks = tg.chunk_blocks("HEADER", blocks, limit=3800)
    assert len(chunks) > 1
    assert all(len(c) <= 3800 for c in chunks)
    assert chunks[0].startswith("HEADER") and all(c.startswith("HEADER <i>(cont.)</i>") for c in chunks[1:])
    # every block appears intact in exactly one chunk
    for b in blocks:
        assert sum(b in c for c in chunks) == 1


def test_oversize_single_block_is_hard_split_under_limit():
    chunks = tg.chunk_blocks("H", ["y" * 9000], limit=3800)
    assert len(chunks) >= 3 and all(len(c) <= 3800 for c in chunks)
    assert sum(c.count("y") for c in chunks) == 9000


def _trend(entries, has_prev=True, new=(), dropped=()):
    return ta.TrendReport(current_session="2026-09-14", previous_session="2026-09-11" if has_prev else None,
                          sessions=[], missing_sessions=[], top_n=10, entries=entries,
                          new_entries=list(new), dropped=list(dropped))


def _result(rank, symbol, **kw):
    return {"final_rank": rank, "symbol": symbol, "composite_score": 12.4, "max_score": 14.0,
            "technical_score": 8, "fundamentals_score": 4.4, "sector": "Technology", **kw}


def test_report_escapes_special_characters():
    results = [_result(1, "A&B<C>", sector="R&D <Labs>", vcp_status="analyzed",
                       vcp_entry_recommendation="wait_for_breakout", vcp_pattern_stage="mature",
                       vcp_confidence="high")]
    chunks = tg.build_report("2026-09-14", {"universe": 4312}, results, _trend([]))
    text = "\n".join(chunks)
    assert "A&amp;B&lt;C&gt;" in text and "R&amp;D &lt;Labs&gt;" in text
    assert "A&B<C>" not in text


def test_report_uses_actual_stage_g_terminology_and_counts():
    e = ta.SymbolTrend(symbol="NVDA", rank=1, composite_score=12.4, previous_rank=4, rank_delta_1=3,
                       top_n_streak=6, category=ta.RISING)
    p = ta.SymbolTrend(symbol="AVGO", rank=2, composite_score=12.0, previous_rank=2, rank_delta_1=0,
                       top_n_streak=4, category=ta.PERSISTENT)
    results = [
        _result(1, "NVDA", vcp_status="analyzed", vcp_is_pattern=1, vcp_entry_recommendation="wait_for_breakout",
                vcp_pattern_stage="mature", vcp_confidence="high", vcp_pivot_price=120.5),
        _result(2, "AVGO", vcp_status="skipped"),
    ]
    counts = {"universe": 4312, "stage_d": 83, "vcp_analyzed": 10}
    text = "\n".join(tg.build_report("2026-09-14", counts, results,
                                     _trend([e, p], new=["MU"], dropped=[ta.DroppedSymbol("XYZ", 7, 15)])))
    assert "Universe" not in text and "Stage D screened" not in text        # stage summary -> debug message
    assert "⬆ Up: NVDA #4 → #1" in text and "🔥 Persistent: AVGO" in text
    assert "🆕 New: MU" in text and "🚪 Left Top 10: XYZ #7 → #15" in text
    assert "Claude: VCP · mature · wait_for_breakout · high" in text           # REVIEW block
    assert "#1 <b>NVDA</b> · 12.4/14 · Technology · 🟠 Review" in text
    assert "#2 <b>AVGO</b> · 12.4/14 · Technology · ⚪ No VCP" in text
    for invented in ("STRONG", "PASS", "BUY"):
        assert invented not in text
    debug = "\n".join(tg.build_debug_report("2026-09-14", counts, results))
    assert "Universe: 4,312" in debug and "Stage D screened: 83" in debug


def test_first_session_report_says_no_history():
    text = "\n".join(tg.build_report("2026-09-14", {}, [_result(1, "AAA")], _trend([], has_prev=False)))
    assert "First recorded session" in text


def test_alerts_contain_no_secrets(monkeypatch):
    text = tg.build_pipeline_failed_alert("2026-09-14", "D", "data/logs/pipeline_x.log", "ValueError: boom")
    assert "Stage: D" in text and "data/logs/pipeline_x.log" in text
    ready = tg.build_data_not_ready_alert("2026-09-14", "2026-09-11", 3)
    assert "Latest available bar: 2026-09-11" in ready and "Pipeline not executed." in ready


def test_token_free_run_summary_does_not_claim_vcp_analysis_or_zero_actionable():
    results = [_result(1, "AAA", vcp_status="skipped"), _result(2, "BBB", vcp_status="skipped"), _result(3, "CCC")]
    counts = {"ranked": 3, "vcp_analyzed": 2, "vcp_actionable": 0}
    text = "\n".join(tg.build_debug_report("2026-09-14", counts, results))
    assert "VCP metrics computed: 2 (Claude vision off: token-free mode)" in text
    assert "VCP analysed" not in text and "buy_now / wait_for_breakout" not in text


# ── VCP setups (rank-independent), numeric vs Claude kept separate ───────

import json as _json


@pytest.mark.parametrize("numeric, claude, expected", [
    ("strong", 1, "agree_positive"), ("acceptable", True, "agree_positive"),
    ("weak", 0, "agree_negative"), ("absent", False, "agree_negative"),
    ("strong", 0, "numeric_positive_claude_negative"), ("absent", 1, "numeric_negative_claude_positive"),
    (None, 1, None), ("strong", None, None),
])
def test_vcp_agreement_classification(numeric, claude, expected):
    assert tg.vcp_agreement(numeric, claude) == expected


def _metrics(**kw):
    base = {"vcp_numeric_quality": "strong", "contraction_pcts": [18.16, 9.31, 5.02],
            "final_contraction_confirmed": False, "volume_dryup_quality": "acceptable",
            "pivot_price_candidate": 620.96, "pct_from_pivot": -2.61, "pivot_state": "coiling_below_pivot"}
    base.update(kw)
    return _json.dumps(base)


def _skipped(reason):
    return _json.dumps({"skipped": reason})


def _selective_run():
    """Selective mode: 20 Stage-G rows; Claude reviewed only 4; a rank-18 strong setup."""
    rows = []
    for rank in range(1, 21):
        rows.append(_result(rank, f"S{rank:02d}", vcp_status="skipped", vcp_verdict_json=_skipped("not_requested"),
                            vcp_metrics_json=_metrics(vcp_numeric_quality="absent", contraction_pcts=[11.1, 14.6],
                                                      final_contraction_confirmed=True)))
    rows[17] = _result(18, "CACC", vcp_status="skipped", vcp_verdict_json=_skipped("not_requested"),
                       vcp_metrics_json=_metrics(), vcp_debug_chart_path="charts/vcp_debug/CACC.png")
    rows[3] = _result(4, "HALO", vcp_status="analyzed", vcp_is_pattern=1, vcp_pattern_stage="mature",
                      vcp_entry_recommendation="wait_for_breakout", vcp_confidence="medium",
                      vcp_metrics_json=_metrics(vcp_numeric_quality="acceptable", contraction_pcts=[6.35, 5.64, 4.57],
                                                volume_dryup_quality="strong", pct_from_pivot=-2.89),
                      vcp_debug_chart_path="charts/vcp_debug/HALO.png")
    rows[7] = _result(8, "SRCE", vcp_status="analyzed", vcp_is_pattern=1, vcp_pattern_stage="forming",
                      vcp_entry_recommendation="wait_for_better_setup", vcp_confidence="low",
                      vcp_metrics_json=_metrics(vcp_numeric_quality="weak", contraction_pcts=[12.56, 4.45, 7.89],
                                                final_contraction_confirmed=True, pct_from_pivot=-4.24),
                      vcp_debug_chart_path="charts/vcp_debug/SRCE.png")
    rows[4] = _result(5, "MEDP", vcp_status="analyzed", vcp_is_pattern=0, vcp_pattern_stage="not_present",
                      vcp_entry_recommendation="avoid", vcp_confidence="medium",
                      vcp_metrics_json=_metrics(vcp_numeric_quality="acceptable", contraction_pcts=[16.88, 8.27],
                                                pct_from_pivot=-5.44, pivot_state="far_below_pivot"),
                      vcp_debug_chart_path="charts/vcp_debug/MEDP.png")
    rows[15] = _result(16, "FRHC", vcp_status="analyzed", vcp_is_pattern=0, vcp_pattern_stage="not_present",
                       vcp_entry_recommendation="wait_for_better_setup", vcp_confidence="low",
                       vcp_metrics_json=_metrics(vcp_numeric_quality="weak", final_contraction_confirmed=True))
    return rows


def test_vcp_section_is_rank_independent_and_ordered_by_vcp_relevance():
    rows = _selective_run()
    order = [r["symbol"] for r in tg.vcp_review_set(rows)]
    # strong (rank 18) first, then acceptable by rank (HALO 4, MEDP 5), then the disagreement (SRCE)
    assert order == ["CACC", "HALO", "MEDP", "SRCE"]
    text = "\n".join(tg.build_report("2026-09-14", {"ranked": 353, "vcp_analyzed": 20}, rows,
                                     _trend([], has_prev=False)))
    # no persisted Stage-I plans -> nothing is READY; numeric setups wait, disagreements are reviews
    assert ("🎯 <b>ACTION BOARD</b>\n🟢 Ready: none\n🟡 Watch: CACC, HALO\n🟠 Review: MEDP, SRCE\n"
            "💼 Open positions: 0") in text
    assert ("🟡 <b>WATCH — CACC</b>\nRank #18 · Technology · 12.4/14\n"
            "VCP: STRONG · 18.2 → 9.3 → 5.0% (forming)\nVolume: ACCEPTABLE · Pivot: -2.6%\nWhy watch:\n"
            "✓ volume dries up through the base\n⚠ final contraction still forming\n"
            "⚠ no Stage-I plan recorded for this session\n"
            "Claude: not requested (clear numeric setup)\nAction: <b>WAIT</b>") in text
    assert text.index("WATCH — CACC") < text.index("WATCH — HALO") < text.index("REVIEW — MEDP") \
        < text.index("REVIEW — SRCE")
    assert "FRHC —" not in text and "S01 —" not in text                        # weak + Claude -, absent


def test_summary_counts_only_include_names_claude_actually_reviewed():
    text = "\n".join(tg.build_debug_report("2026-09-14", {"vcp_analyzed": 20}, _selective_run()))
    assert "VCP metrics computed: 20" in text
    assert "Numeric VCP positive: 3" in text                   # CACC strong, HALO + MEDP acceptable
    assert "Claude reviewed: 4" in text and "Claude VCP positive: 2" in text
    assert "VCP review union: 4" in text
    assert "Numeric vs Claude: 2 agree · 2 disagree" in text   # HALO, FRHC agree; SRCE, MEDP disagree
    assert "VCP pattern:" not in text and "VCP analysed" not in text


def test_not_requested_is_not_a_negative_claude_verdict():
    rows = _selective_run()
    cacc = next(r for r in rows if r["symbol"] == "CACC")
    assert tg.claude_state(cacc) == "not_requested" and tg._row_agreement(cacc) is None
    assert tg.vcp_priority(cacc) == 0
    absent = rows[0]
    assert tg._row_agreement(absent) is None and tg.vcp_priority(absent) is None


def test_overall_leaders_stay_composite_ranked_and_compact():
    rows = _selective_run()
    text = "\n".join(tg.build_report("2026-09-14", {}, rows, _trend([], has_prev=False), max_candidates=10))
    top = text[text.index("📊 <b>OVERALL LEADERS</b>"):text.index("📈 <b>WATCHLIST CHANGES</b>")]
    assert all(f"#{r} <b>" in top for r in range(1, 11)) and "#18 <b>CACC" not in top
    assert "#4 <b>HALO</b> · 12.4/14 · Technology · 🟡 Watch" in top
    assert "#5 <b>MEDP</b> · 12.4/14 · Technology · 🟠 Review" in top
    assert "#1 <b>S01</b> · 12.4/14 · Technology · ⚪ No VCP" in top
    assert len(top.strip().split("\n")) == 11                  # header + one line per leader
    assert "Numeric:" not in top and "%" not in top            # VCP metrics are not repeated


def test_vcp_charts_follow_vcp_priority_and_limit_not_rank():
    rows = _selective_run()
    assert [r["symbol"] for r in tg.select_vcp_chart_rows(rows, 10)] == ["CACC", "HALO", "MEDP", "SRCE"]
    assert [r["symbol"] for r in tg.select_vcp_chart_rows(rows, 2)] == ["CACC", "HALO"]
    rows[17]["vcp_debug_chart_path"] = None                    # no verification chart stored -> not selected
    assert [r["symbol"] for r in tg.select_vcp_chart_rows(rows, 2)] == ["HALO", "MEDP"]


def test_vcp_chart_captions():
    rows = {r["symbol"]: r for r in _selective_run()}
    cacc = tg.build_chart_caption(rows["CACC"])
    assert cacc.startswith("🟡 <b>CACC</b> · WATCH · Rank 18\nNumeric: STRONG · 18.2 → 9.3 → 5.0% (forming)")
    assert "Pivot: -2.6%" in cacc and "Claude: not requested" in cacc and "Agreement" not in cacc
    assert "Claude: VCP · mature · wait_for_breakout · medium\nAgreement: YES" in tg.build_chart_caption(rows["HALO"])
    srce = tg.build_chart_caption(rows["SRCE"])
    assert srce.startswith("🟠 <b>SRCE</b> · REVIEW · Rank 8") and "Agreement: DISAGREE" in srce
    assert srce.endswith("Action: MANUAL REVIEW")
    assert all(len(tg.build_chart_caption(r)) < tg.MAX_CAPTION_CHARS for r in rows.values())


def test_review_off_and_token_free_labels():
    off = _result(1, "OFF", vcp_status="skipped", vcp_verdict_json=_skipped("llm_review_off"),
                  vcp_metrics_json=_metrics())
    free = _result(2, "FREE", vcp_status="skipped", vcp_verdict_json=_skipped("token_free_mode"),
                   vcp_metrics_json=_metrics())
    err = _result(3, "ERR", vcp_status="error", vcp_metrics_json=_metrics(vcp_numeric_quality="weak"))
    text = "\n".join(tg.build_report("2026-09-14", {"vcp_analyzed": 3}, [off, free, err],
                                     _trend([], has_prev=False)))
    assert "Claude: not run (review mode off)" in text and "Claude: not run (token-free mode)" in text
    debug = "\n".join(tg.build_debug_report("2026-09-14", {"vcp_analyzed": 3}, [off, free, err]))
    assert "Claude errors: 1" in debug and "Numeric vs Claude" not in debug


def test_unreadable_or_v1_metrics_do_not_break_the_report():
    results = [_result(1, "OLD", vcp_status="analyzed", vcp_is_pattern=1,
                       vcp_metrics_json='{"contraction_pcts": [20, 10]}'),
               _result(2, "BAD", vcp_status="analyzed", vcp_is_pattern=0, vcp_metrics_json="{not json")]
    text = "\n".join(tg.build_report("2026-09-14", {}, results, _trend([], has_prev=False)))
    assert "🟠 Review: OLD" in text and "🟠 <b>REVIEW — OLD</b>" in text and "BAD —" not in text
    debug = "\n".join(tg.build_debug_report("2026-09-14", {}, results))
    assert "Numeric vs Claude" not in debug and "Claude VCP positive: 1" in debug


