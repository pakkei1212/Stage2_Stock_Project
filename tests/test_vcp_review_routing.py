"""Stage G: numeric detector first, Claude as selective visual reviewer.

Routing modes (off / selective / all), the selective rule, structural triggers
(wick / gap), semantic consistency of Claude verdicts, persistence of the review
record next to the verdict, and Telegram reuse of the Stage-G verification chart.
Claude and Telegram are fakes; nothing touches the network.
"""
import json
import os

import pandas as pd
import pytest

from conftest import make_path_df
from pipeline import history_store as hs
from pipeline import scheduler as sch
from pipeline import stage_vcp_analysis as g
from pipeline.config import CONFIG
from stage_h_fakes import FakeClock, FakeScreen, FakeTelegram, make_config, make_deps, sgt

M = 1_000_000
STRONG_COIL = [(12, 76.0, 10 * M), (12, 98.0, 6 * M), (9, 84.3, 7 * M), (9, 97.5, 5 * M),
               (7, 89.7, 4 * M), (6, 97.5, 3.5 * M), (5, 93.6, 2 * M), (6, 96.8, 1.5 * M)]
STRONG_FORMING = [(12, 80.0, 10 * M), (12, 98.0, 6 * M), (9, 87.2, 6 * M), (9, 97.5, 4 * M),
                  (7, 91.6, 3 * M), (8, 97.0, 2.5 * M), (3, 94.1, 1.5 * M)]
STRONG_BREAKOUT = [(12, 76.0, 10 * M), (12, 98.0, 5 * M), (9, 84.3, 7 * M), (9, 97.5, 5 * M),
                   (7, 89.7, 4 * M), (6, 97.0, 3 * M), (5, 93.6, 2 * M), (25, 96.5, 2 * M), (1, 99.8, 3.6 * M)]
ACCEPTABLE = [(12, 78.0, 10 * M), (12, 98.0, 6 * M), (9, 85.3, 7 * M), (9, 97.0, 5 * M),
              (9, 83.4, 6 * M), (9, 96.0, 4 * M), (6, 89.3, 3 * M), (6, 95.0, 2 * M)]
WEAK = [(12, 75.0, 10 * M), (12, 98.0, 6 * M), (7, 88.2, 7 * M), (7, 97.0, 5 * M),
        (10, 77.6, 8 * M), (10, 96.0, 5 * M), (6, 88.3, 4 * M), (6, 95.0, 3 * M)]
ABSENT = [(10, 85.0, 8 * M), (10, 118.0, 6 * M), (8, 106.2, 6 * M), (10, 138.0, 5 * M),
          (7, 129.7, 5 * M), (10, 160.0, 4 * M), (6, 155.2, 3 * M), (5, 158.0, 3 * M)]
FIXTURES = {"STRONG": STRONG_COIL, "FORMING": STRONG_FORMING, "BREAKOUT": STRONG_BREAKOUT,
            "ACCEPT": ACCEPTABLE, "WEAK": WEAK, "ABSENT": ABSENT}

VERDICT = {"is_vcp_pattern": True, "pattern_stage": "mature", "contraction_count_observed": 3,
           "volume_dry_up_confirmed": True, "pivot_price": 98.3, "suggested_stop_loss": 92.0,
           "confidence": "medium", "entry_recommendation": "wait_for_breakout", "rationale": "One coherent base."}


def _computed(segments):
    return g.compute_vcp_metrics(make_path_df(segments), CONFIG, return_details=True)


def _cfg(mode, live=True):
    return {**CONFIG, "vcp_llm_review_mode": mode, "vcp_live_analysis": live}


# ── modes ────────────────────────────────────────────────────────────────

def test_review_mode_resolution_and_backward_compatibility():
    assert g.llm_review_mode(_cfg("selective")) == "selective"
    assert g.llm_review_mode(_cfg("ALL ")) == "all"
    assert g.llm_review_mode(_cfg("off")) == "off"
    assert g.llm_review_mode(_cfg("all", live=False)) == "off"          # VCP_LIVE_ANALYSIS=0 still wins
    assert g.llm_review_mode({**CONFIG, "vcp_live_analysis": True, "vcp_llm_review_mode": None}) == "selective"
    assert g.llm_review_mode(_cfg("banana")) == "selective"


# ── the selective rule ───────────────────────────────────────────────────

@pytest.mark.parametrize("name, quality, requested", [
    ("STRONG", "strong", False),        # clear coiling strong setup stands on the numbers
    ("FORMING", "strong", False),       # a forming final leg alone does not trigger review
    ("ABSENT", "absent", False),        # clearly not contracting
    ("ACCEPT", "acceptable", True),
    ("WEAK", "weak", True),
    ("BREAKOUT", "strong", True),       # fresh breakout vs extended is a visual call
])
def test_selective_rule(name, quality, requested):
    metrics, details = _computed(FIXTURES[name])
    assert metrics["vcp_numeric_quality"] == quality
    review, reasons = g.llm_review_decision(metrics, details, _cfg("selective"))
    assert review is requested and bool(reasons) is requested


def test_selective_reasons_are_explicit():
    m, d = _computed(ACCEPTABLE)
    assert g.llm_review_decision(m, d, _cfg("selective"))[1] == ["numeric quality acceptable"]
    m, d = _computed(STRONG_BREAKOUT)
    assert g.llm_review_decision(m, d, _cfg("selective"))[1] == ["strong setup at fresh_breakout"]


def test_all_mode_reviews_clear_cases_and_keeps_selective_reasons():
    m, d = _computed(STRONG_COIL)
    assert g.llm_review_decision(m, d, _cfg("all")) == (True, ["full review mode (numeric result is clear)"])
    m, d = _computed(WEAK)
    assert g.llm_review_decision(m, d, _cfg("all")) == (True, ["numeric quality weak"])


def test_off_mode_never_requests():
    m, d = _computed(WEAK)
    assert g.llm_review_decision(m, d, _cfg("off"))[0] is False


def test_wick_or_gap_structure_routes_even_a_clear_strong_setup_to_claude():
    df = make_path_df(STRONG_COIL)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    assert g.structure_warnings(d) == [] and not g.llm_review_decision(m, d, _cfg("selective"))[0]

    wick = df.copy()
    off = len(wick) - len(d["base"])
    t2 = off + d["contraction_legs"][1]["measured_high_idx"]
    wick.iloc[t2, wick.columns.get_loc("High")] = wick["Close"].iloc[t2] * 1.06     # 6% upper wick on T2's High bar
    m2, d2 = g.compute_vcp_metrics(wick, CONFIG, return_details=True)
    warnings = g.structure_warnings(d2)
    assert any("High set by a" in w and "intraday wick" in w for w in warnings)
    assert g.llm_review_decision(m2, d2, _cfg("selective"))[0] is True

    gap = df.copy()
    k = off + d["contraction_legs"][2]["measured_low_idx"] + 2
    gap.iloc[k, gap.columns.get_loc("Open")] = gap["Close"].iloc[k - 1] * 0.93      # -7% gap inside the base
    gap.iloc[k, gap.columns.get_loc("Low")] = min(gap["Low"].iloc[k], gap["Open"].iloc[k])
    m3, d3 = g.compute_vcp_metrics(gap, CONFIG, return_details=True)
    assert any("session gap" in w for w in g.structure_warnings(d3))


def test_wicks_in_a_clearly_absent_structure_do_not_route_to_claude():
    df = make_path_df(ABSENT)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    off = len(df) - len(d["base"])
    hi = off + d["contraction_legs"][0]["measured_high_idx"]
    df.iloc[hi, df.columns.get_loc("High")] = df["Close"].iloc[hi] * 1.08
    m2, d2 = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    assert m2["vcp_numeric_quality"] == "absent" and g.structure_warnings(d2)
    assert g.llm_review_decision(m2, d2, _cfg("selective")) == (False, [])


def test_structure_warnings_are_context_for_grey_zone_candidates():
    df = make_path_df(WEAK)
    m, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    off = len(df) - len(d["base"])
    hi = off + d["contraction_legs"][0]["measured_high_idx"]
    df.iloc[hi, df.columns.get_loc("High")] = df["Close"].iloc[hi] * 1.05
    m2, d2 = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    review, reasons = g.llm_review_decision(m2, d2, _cfg("selective"))
    assert review and reasons[0].startswith("numeric quality") and any("intraday wick" in r for r in reasons)


# ── run_vcp_analysis routing ─────────────────────────────────────────────

class _Client:
    def __init__(self):
        self.tickers = []
        self.messages = self

    def create(self, **kwargs):
        text = kwargs["messages"][0]["content"][1]["text"]
        self.tickers.append(text.split("You are analyzing ")[1].split(" ")[0])

        class _B:
            type = "text"
            text = json.dumps(VERDICT)

        class _R:
            stop_reason = "end_turn"
            usage = None
            content = [_B()]
        return _R()


def _run(tmp_path, monkeypatch, mode, live=True):
    client = _Client()
    created = []

    def factory():
        created.append(1)
        return client
    monkeypatch.setattr("anthropic.Anthropic", factory)
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda s, c: make_path_df(FIXTURES[s]))
    charts = {}
    for s in FIXTURES:
        charts[s] = str(tmp_path / f"{s}.png")
        open(charts[s], "wb").write(b"png")
    cfg = {**_cfg(mode, live), "chart_dir": str(tmp_path / "charts")}
    result = g.run_vcp_analysis(pd.DataFrame({"Symbol": list(FIXTURES)}), charts, cfg)
    return result.set_index("Symbol"), client, created


def test_selective_mode_calls_claude_only_for_ambiguous_candidates(tmp_path, monkeypatch):
    rows, client, _ = _run(tmp_path, monkeypatch, "selective")
    assert sorted(client.tickers) == ["ACCEPT", "BREAKOUT", "WEAK"]
    for sym in ("STRONG", "FORMING", "ABSENT"):
        assert rows.loc[sym, "skipped"] == "not_requested"
        assert rows.loc[sym, "llm_review"]["requested"] is False
        assert pd.isna(rows.loc[sym, "is_vcp_pattern"])                  # no verdict invented
    assert rows.loc["WEAK", "llm_review"] == {"mode": "selective", "requested": True, "reasons": ["numeric quality weak"],
                                              "agreement": "numeric_negative_claude_positive",
                                              "rationale_warnings": [], "prompt_adherence_warnings": []}
    assert rows.loc["ACCEPT", "llm_review"]["agreement"] == "agree_positive"
    assert (tmp_path / "charts" / "vcp_debug" / "STRONG.png").exists()     # verification chart for every candidate


def test_all_mode_reviews_every_candidate(tmp_path, monkeypatch):
    rows, client, _ = _run(tmp_path, monkeypatch, "all")
    assert sorted(client.tickers) == sorted(FIXTURES)
    assert "skipped" not in rows.columns and all(r["requested"] for r in rows["llm_review"])


def test_off_mode_and_token_free_make_no_claude_client(tmp_path, monkeypatch):
    rows, client, created = _run(tmp_path, monkeypatch, "off")
    assert created == [] and client.tickers == [] and set(rows["skipped"]) == {"llm_review_off"}
    rows, client, created = _run(tmp_path, monkeypatch, "all", live=False)
    assert created == [] and set(rows["skipped"]) == {"token_free_mode"}


def test_prompt_tells_claude_why_it_was_routed(tmp_path, monkeypatch):
    seen = {}

    class Capture(_Client):
        def create(self, **kwargs):
            seen["prompt"] = kwargs["messages"][0]["content"][1]["text"]
            return super().create(**kwargs)
    monkeypatch.setattr("anthropic.Anthropic", lambda: Capture())
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda s, c: make_path_df(WEAK))
    chart = tmp_path / "W.png"
    chart.write_bytes(b"png")
    g.run_vcp_analysis(pd.DataFrame({"Symbol": ["W"]}), {"W": str(chart)},
                       {**_cfg("selective"), "chart_dir": str(tmp_path)})
    p = seen["prompt"]
    assert "routed to you for visual review because: numeric quality weak" in p
    assert "is_vcp_pattern=false -> pattern_stage is not_present" in p and "Use forming only for a genuine" in p


# ── semantic consistency ─────────────────────────────────────────────────

@pytest.mark.parametrize("changes, problem", [
    (dict(is_vcp_pattern=False, pattern_stage="forming", entry_recommendation="wait_for_better_setup"),
     "is_vcp_pattern=false with pattern_stage=forming"),
    (dict(is_vcp_pattern=False, pattern_stage="not_present", entry_recommendation="wait_for_breakout"),
     "is_vcp_pattern=false with entry_recommendation=wait_for_breakout"),
    (dict(is_vcp_pattern=True, pattern_stage="not_present"), "is_vcp_pattern=true with pattern_stage=not_present"),
    (dict(pattern_stage="failed", entry_recommendation="buy_now"), "pattern_stage=failed with entry_recommendation=buy_now"),
])
def test_contradictory_structured_fields_are_rejected(tmp_path, changes, problem):
    verdict = {**VERDICT, **changes}
    assert problem in g.semantic_problems(verdict)
    chart = tmp_path / "c.png"
    chart.write_bytes(b"png")

    class C(_Client):
        def create(self, **kwargs):
            class _B:
                type = "text"
                text = json.dumps(verdict)

            class _R:
                stop_reason = "end_turn"
                usage = None
                content = [_B()]
            return _R()
    stats = g.new_claude_stats()
    m, d = _computed(WEAK)
    out = g.analyze_chart(C(), "X", str(chart), m, CONFIG, stats=stats)
    assert out["error"] == "semantic_inconsistency" and problem in out["detail"] and stats["semantic_failures"] == 1


@pytest.mark.parametrize("verdict", [
    dict(VERDICT),
    {**VERDICT, "is_vcp_pattern": False, "pattern_stage": "not_present", "entry_recommendation": "avoid"},
    {**VERDICT, "is_vcp_pattern": False, "pattern_stage": "failed", "entry_recommendation": "wait_for_better_setup"},
    {**VERDICT, "pattern_stage": "forming", "entry_recommendation": "wait_for_better_setup"},
])
def test_consistent_verdicts_pass(verdict):
    assert g.semantic_problems(verdict) == []


def test_rationale_that_rejects_a_positive_verdict_is_warned_not_silently_fixed():
    srce_like = {**VERDICT, "rationale": "However the selected base is not a clean VCP; it reads as an advancing trend."}
    assert g.rationale_warnings(srce_like) and g.semantic_problems(srce_like) == []
    assert g.rationale_warnings({**VERDICT, "rationale": "Three tightening contractions form one coherent VCP base."}) == []
    assert g.rationale_warnings({**VERDICT, "is_vcp_pattern": False, "pattern_stage": "not_present",
                                 "entry_recommendation": "avoid", "rationale": "not a VCP"}) == []


# ── wick / gap prompt adherence ──────────────────────────────────────────

def _wick_df(leg, factor=1.04):
    """STRONG_COIL with an upper wick on one leg's High bar (leg 2 -> the wick high becomes the pivot)."""
    df = make_path_df(STRONG_COIL)
    _, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    i = len(df) - len(d["base"]) + d["contraction_legs"][leg]["measured_high_idx"]
    df.iloc[i, df.columns.get_loc("High")] = max(df["Open"].iloc[i], df["Close"].iloc[i]) * factor
    return df


def _strong_with_wick(leg):
    return g.compute_vcp_metrics(_wick_df(leg), CONFIG, return_details=True)


def _strong_with_gap():
    df = make_path_df(STRONG_COIL)
    _, d = g.compute_vcp_metrics(df, CONFIG, return_details=True)
    k = len(df) - len(d["base"]) + d["contraction_legs"][2]["measured_low_idx"] + 2
    df.iloc[k, df.columns.get_loc("Open")] = df["Close"].iloc[k - 1] * 0.93
    df.iloc[k, df.columns.get_loc("Low")] = min(df["Low"].iloc[k], df["Open"].iloc[k])
    return g.compute_vcp_metrics(df, CONFIG, return_details=True)


def test_wick_defined_pivot_gets_an_explicit_pivot_quality_instruction():
    m, d = _strong_with_wick(2)
    review, reasons = g.llm_review_decision(m, d, _cfg("selective"))
    assert review and reasons == ["T3 High set by a 4.0% intraday wick"]
    assert m["pivot_price_candidate"] == round(d["contraction_legs"][2]["measured_high_price"], 2)
    p = g._build_prompt("X", m, CONFIG, d["contraction_legs"], reasons)
    assert f"that wick high IS the candidate pivot ({m['pivot_price_candidate']})" in p
    assert '"Wick/pivot review:"' in p and "Do not accept the pivot merely because it was supplied" in p
    for wording in ("visually\n  well supported", "questionable because of a wick", "body-based consolidation"):
        assert wording in p
    assert "Gap review" not in p


def test_wick_that_does_not_set_the_pivot_still_requires_a_review_but_not_the_pivot_wording():
    m, d = _strong_with_wick(0)
    _, reasons = g.llm_review_decision(m, d, _cfg("selective"))
    p = g._build_prompt("X", m, CONFIG, d["contraction_legs"], reasons)
    assert '"Wick/pivot review:"' in p and "IS the candidate pivot" not in p


def test_gap_routed_candidate_gets_an_explicit_gap_quality_instruction():
    m, d = _strong_with_gap()
    _, reasons = g.llm_review_decision(m, d, _cfg("selective"))
    assert any("session gap" in r for r in reasons)
    p = g._build_prompt("X", m, CONFIG, d["contraction_legs"], reasons)
    assert '"Gap review:"' in p and "distorted\n  contraction leg" in p and "V-shaped" in p
    assert "does not materially weaken the setup" in p


def test_no_structure_block_without_wick_or_gap_reasons():
    m, d = _computed(WEAK)
    p = g._build_prompt("X", m, CONFIG, d["contraction_legs"], ["numeric quality weak"])
    assert "Routed structure concerns" not in p and "Wick/pivot review" not in p


WICK = ["T2 High set by a 4.6% intraday wick"]
GAP = ["numeric quality acceptable", "+22.5% session gap on 2026-07-23 inside the base"]


@pytest.mark.parametrize("reasons, rationale, expected", [
    (WICK, "Wick/pivot review: the 621.52 pivot is a wick high; bodies top out near 608, so it is questionable.", 0),
    (WICK, "The tight matched right-side highs at 621.52 define resistance. Wait for the breakout.", 1),
    (GAP, "Gap review: the gap creates a distorted, V-shaped first leg.", 0),
    (GAP, "Tightening looks genuine and volume dried up.", 1),
    (WICK + GAP[1:], "Wick/pivot review: the pivot wick is questionable. Nothing else to add.", 1),
    (["numeric quality weak", "final contraction still forming"], "Loose base.", 0),
    ([], "Anything.", 0),
])
def test_prompt_adherence_warning_only_when_the_routed_concern_is_ignored(reasons, rationale, expected):
    assert len(g.prompt_adherence_warnings({**VERDICT, "rationale": rationale}, reasons)) == expected


def test_adherence_warning_is_recorded_without_invalidating_the_verdict(tmp_path, monkeypatch):
    class Capture(_Client):
        prompt = None

        def create(self, **kwargs):
            Capture.prompt = kwargs["messages"][0]["content"][1]["text"]
            return super().create(**kwargs)                  # VERDICT's rationale never mentions the wick
    monkeypatch.setattr("anthropic.Anthropic", lambda: Capture())
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda s, c: _wick_df(2))
    chart = tmp_path / "C.png"
    chart.write_bytes(b"png")
    rows = g.run_vcp_analysis(pd.DataFrame({"Symbol": ["C"]}), {"C": str(chart)},
                              {**_cfg("selective"), "chart_dir": str(tmp_path)}).set_index("Symbol")
    assert "IS the candidate pivot" in Capture.prompt
    assert "error" not in rows.columns and rows.loc["C", "is_vcp_pattern"] == True          # verdict kept
    review = rows.loc["C", "llm_review"]
    assert review["agreement"] == "agree_positive" and review["rationale_warnings"] == []
    assert review["prompt_adherence_warnings"] == [
        "routed for a wick warning but the rationale does not review the wick / pivot quality"]


# ── persistence + Telegram charts ────────────────────────────────────────

def test_review_record_is_stored_with_the_verdict_not_the_numeric_metrics(tmp_path):
    store = hs.HistoryStore(str(tmp_path / "h.sqlite3"))
    run_id = store.claim_session("2026-09-14", "manual")
    store.mark_running(run_id, latest_bar="2026-09-14", attempts=1, run_timestamp="t", metadata={})
    ranked = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Composite Score": [14, 13, 12]})
    vcp = pd.DataFrame([
        {"Symbol": "AAA", "vcp_numeric_quality": "strong", "skipped": "not_requested",
         "llm_review": {"mode": "selective", "requested": False, "reasons": [], "agreement": None}},
        {"Symbol": "BBB", "vcp_numeric_quality": "weak", **VERDICT,
         "llm_review": {"mode": "selective", "requested": True, "reasons": ["numeric quality weak"],
                        "agreement": "numeric_negative_claude_positive"}},
        {"Symbol": "CCC", "vcp_numeric_quality": "acceptable", "error": "semantic_inconsistency",
         "detail": "is_vcp_pattern=false with pattern_stage=forming | raw: {...}",
         "llm_review": {"mode": "selective", "requested": True, "reasons": ["numeric quality acceptable"]}},
    ])
    store.persist_results(run_id, "2026-09-14", ranked, vcp)
    rows = {r["symbol"]: r for r in store.results_for_run(run_id)}
    for r in rows.values():
        assert "llm_review" not in json.loads(r["vcp_metrics_json"]) and "review" not in json.loads(r["vcp_metrics_json"])
    assert rows["AAA"]["vcp_status"] == "skipped" and rows["AAA"]["vcp_is_pattern"] is None
    assert json.loads(rows["AAA"]["vcp_verdict_json"])["review"]["requested"] is False
    bbb = json.loads(rows["BBB"]["vcp_verdict_json"])
    assert bbb["review"]["agreement"] == "numeric_negative_claude_positive" and rows["BBB"]["vcp_is_pattern"] == 1
    ccc = json.loads(rows["CCC"]["vcp_verdict_json"])
    assert rows["CCC"]["vcp_status"] == "error" and "pattern_stage=forming" in ccc["error_detail"]
    assert json.loads(rows["BBB"]["vcp_metrics_json"])["vcp_numeric_quality"] == "weak"


def test_telegram_sends_the_existing_vcp_verification_chart_not_the_stage_f_chart(tmp_path):
    cfg = make_config(tmp_path, telegram_enabled=True, telegram_send_charts=True, telegram_vcp_chart_max=10)
    tele = FakeTelegram()
    outcome = sch.check_and_run("scheduled", cfg, make_deps(FakeClock(sgt("2026-09-15 09:15")),
                                                          FakeScreen(["AAA", "BBB", "CCC"]), latest_bar=pd.Timestamp("2026-09-14").date(),
                                                          telegram=tele))
    assert outcome.decision == sch.COMPLETED
    assert tele.photos and all(os.sep + "vcp_debug" + os.sep in path for path, _ in tele.photos)
    assert all(os.path.exists(path) for path, _ in tele.photos)
    assert "🎯 <b>ACTION BOARD</b>" in tele.messages[0] and "🎯 VCP setups" not in tele.messages[0]
