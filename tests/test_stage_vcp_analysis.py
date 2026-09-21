"""Stage G: verify the Claude vision VCP call and the run_vcp_analysis loop
around it (chart lookup, metric computation, error handling), independent of
the numeric contraction math covered in test_vcp_metrics.py.
"""
import json
import os

import pandas as pd
import pytest

from conftest import make_vcp_df
from pipeline.config import CONFIG
from pipeline.stage_vcp_analysis import analyze_chart, compute_vcp_metrics, run_vcp_analysis

SAMPLE_METRICS = compute_vcp_metrics(make_vcp_df(), CONFIG)

VALID_VERDICT = {
    "is_vcp_pattern": True,
    "pattern_stage": "mature",
    "contraction_count_observed": 3,
    "volume_dry_up_confirmed": True,
    "pivot_price": 123.45,
    "suggested_stop_loss": 110.0,
    "confidence": "high",
    "entry_recommendation": "wait_for_breakout",
    "rationale": "Three tightening contractions with volume drying up.",
}


class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, stop_reason="end_turn", content=None, stop_details=None):
        self.stop_reason = stop_reason
        self.content = content if content is not None else []
        self.stop_details = stop_details


class _DummyMessages:
    def __init__(self, response_fn):
        self._response_fn = response_fn

    def create(self, **kwargs):
        return self._response_fn(kwargs)


class _DummyAnthropicClient:
    def __init__(self, response_fn):
        self.messages = _DummyMessages(response_fn)


def _fake_chart_path(tmp_path, symbol):
    path = tmp_path / f"{symbol}.png"
    path.write_bytes(b"not a real png, just bytes to base64-encode")
    return str(path)


def test_analyze_chart_parses_structured_json_response(tmp_path):
    chart_path = _fake_chart_path(tmp_path, "TEST")
    client = _DummyAnthropicClient(lambda kwargs: _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))]))

    verdict = analyze_chart(client, "TEST", chart_path, metrics=SAMPLE_METRICS, config=CONFIG)

    assert verdict == VALID_VERDICT


def test_analyze_chart_returns_error_on_refusal(tmp_path):
    chart_path = _fake_chart_path(tmp_path, "TEST")
    client = _DummyAnthropicClient(lambda kwargs: _Response(stop_reason="refusal", stop_details="policy"))

    verdict = analyze_chart(client, "TEST", chart_path, metrics=SAMPLE_METRICS, config=CONFIG)

    assert verdict == {"error": "refusal", "detail": "policy"}


def test_analyze_chart_returns_error_when_no_text_block(tmp_path):
    chart_path = _fake_chart_path(tmp_path, "TEST")
    client = _DummyAnthropicClient(lambda kwargs: _Response(stop_reason="end_turn", content=[]))

    verdict = analyze_chart(client, "TEST", chart_path, metrics=SAMPLE_METRICS, config=CONFIG)

    assert verdict == {"error": "no_text_block", "stop_reason": "end_turn"}


def test_run_vcp_analysis_skips_symbols_without_a_chart_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: _DummyAnthropicClient(lambda kwargs: _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))])),
    )
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda symbol, config: make_vcp_df())

    candidates_df = pd.DataFrame({"Symbol": ["HASCHART", "NOCHART"]})
    chart_paths = {"HASCHART": _fake_chart_path(tmp_path, "HASCHART")}

    result = run_vcp_analysis(candidates_df, chart_paths, config={**CONFIG, "chart_dir": str(tmp_path)})

    assert list(result["Symbol"]) == ["HASCHART"]


def test_run_vcp_analysis_skips_symbols_with_insufficient_history(tmp_path, monkeypatch):
    n = CONFIG["vcp_pivot_window_days"] * 4 - 1  # one short of compute_vcp_metrics' minimum
    dates = pd.bdate_range("2023-01-02", periods=n)
    too_short = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000.0},
        index=dates,
    )
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: _DummyAnthropicClient(lambda kwargs: _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))])),
    )
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda symbol, config: too_short)

    candidates_df = pd.DataFrame({"Symbol": ["THIN"]})
    chart_paths = {"THIN": _fake_chart_path(tmp_path, "THIN")}

    result = run_vcp_analysis(candidates_df, chart_paths, config=CONFIG)

    assert result.empty


def test_run_vcp_analysis_merges_verdict_with_metrics_into_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: _DummyAnthropicClient(lambda kwargs: _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))])),
    )
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda symbol, config: make_vcp_df())

    candidates_df = pd.DataFrame({"Symbol": ["GOOD"]})
    chart_paths = {"GOOD": _fake_chart_path(tmp_path, "GOOD")}

    result = run_vcp_analysis(candidates_df, chart_paths, config={**CONFIG, "chart_dir": str(tmp_path)})

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Symbol"] == "GOOD"
    # Numeric metrics (from compute_vcp_metrics) and the Claude verdict should
    # both be present on the same row.
    assert row["contraction_count"] >= 2
    assert row["entry_recommendation"] == "wait_for_breakout"
    assert row["pattern_stage"] == "mature"


def test_run_vcp_analysis_captures_exceptions_as_an_error_row_instead_of_crashing(tmp_path, monkeypatch):
    def _raise(_kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr("anthropic.Anthropic", lambda: _DummyAnthropicClient(_raise))
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda symbol, config: make_vcp_df())

    candidates_df = pd.DataFrame({"Symbol": ["FLAKY"]})
    chart_paths = {"FLAKY": _fake_chart_path(tmp_path, "FLAKY")}

    result = run_vcp_analysis(candidates_df, chart_paths, config={**CONFIG, "chart_dir": str(tmp_path)})

    assert len(result) == 1
    assert "API unavailable" in result.iloc[0]["error"]


# ── response validation, diagnostics, prompt grounding ───────────────────

class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


def _client_returning(text, stop_reason="end_turn", usage=None):
    def respond(kwargs):
        r = _Response(stop_reason=stop_reason, content=[_TextBlock(text)] if text is not None else [])
        r.usage = usage
        return r
    return _DummyAnthropicClient(respond)


def test_truncated_response_is_an_error_verdict_not_a_crash(tmp_path):
    from pipeline.stage_vcp_analysis import new_claude_stats
    stats = new_claude_stats()
    verdict = analyze_chart(_client_returning('{"is_vcp_pattern": tr', stop_reason="max_tokens"), "T",
                            _fake_chart_path(tmp_path, "T"), SAMPLE_METRICS, CONFIG, stats=stats)
    assert verdict == {"error": "truncated", "stop_reason": "max_tokens"}
    assert stats["truncated"] == 1 and stats["errors"] == 1 and stats["ok"] == 0


def test_malformed_json_is_an_error_verdict(tmp_path):
    verdict = analyze_chart(_client_returning("not json {"), "T", _fake_chart_path(tmp_path, "T"),
                            SAMPLE_METRICS, CONFIG)
    assert verdict["error"] == "invalid_json"


@pytest.mark.parametrize("mutate, problem", [
    (lambda v: v.pop("confidence"), "missing field: confidence"),
    (lambda v: v.update(pattern_stage="imminent"), "invalid value for pattern_stage"),
    (lambda v: v.update(is_vcp_pattern="yes"), "invalid value for is_vcp_pattern"),
    (lambda v: v.update(contraction_count_observed=True), "invalid value for contraction_count_observed"),
    (lambda v: v.update(score=9), "unexpected field: score"),
])
def test_schema_violations_are_error_verdicts(tmp_path, mutate, problem):
    from pipeline.stage_vcp_analysis import new_claude_stats
    bad = dict(VALID_VERDICT)
    mutate(bad)
    stats = new_claude_stats()
    verdict = analyze_chart(_client_returning(json.dumps(bad)), "T", _fake_chart_path(tmp_path, "T"),
                            SAMPLE_METRICS, CONFIG, stats=stats)
    assert verdict["error"] == "schema_validation" and problem in verdict["detail"]
    assert stats["schema_failures"] == 1


def test_valid_response_accumulates_usage(tmp_path):
    from pipeline.stage_vcp_analysis import new_claude_stats, validate_verdict
    assert validate_verdict(VALID_VERDICT) == []
    assert validate_verdict(dict(VALID_VERDICT, pivot_price=None, suggested_stop_loss=None)) == []
    stats = new_claude_stats()
    for _ in range(2):
        analyze_chart(_client_returning(json.dumps(VALID_VERDICT), usage=_Usage(3000, 700)), "T",
                      _fake_chart_path(tmp_path, "T"), SAMPLE_METRICS, CONFIG, stats=stats)
    assert stats["sent"] == 2 and stats["ok"] == 2 and stats["input_tokens"] == 6000 and stats["output_tokens"] == 1400


def test_prompt_locates_each_contraction_by_date_and_flags_forming(tmp_path):
    from conftest import make_path_df
    from pipeline.stage_vcp_analysis import _build_prompt
    M = 1_000_000
    df = make_path_df([(12, 80.0, 10 * M), (12, 98.0, 6 * M), (9, 87.2, 6 * M), (9, 97.5, 4 * M),
                       (7, 91.6, 3 * M), (8, 97.0, 2.5 * M), (3, 94.1, 1.5 * M)])
    metrics, details = compute_vcp_metrics(df, CONFIG, return_details=True)
    captured = {}

    def respond(kwargs):
        captured.update(kwargs)
        return _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))])
    analyze_chart(_DummyAnthropicClient(respond), "EDGE", _fake_chart_path(tmp_path, "EDGE"), metrics, CONFIG,
                  legs=details["contraction_legs"])
    prompt = captured["messages"][0]["content"][1]["text"]
    last = details["contraction_legs"][-1]
    assert f"T4: {last['measured_high_date']} High" in prompt and "FORMING)" in prompt
    assert prompt.count(" -> ") >= 4 and "still FORMING at the right edge" in prompt
    assert str(metrics["contraction_pcts"]) in prompt and str(metrics["volume_leg_avgs"]) in prompt
    assert metrics["pivot_state"] in prompt and metrics["vcp_numeric_quality"] in prompt
    assert "You may disagree with its numeric read" in prompt and "detector's contractions are NOT drawn" in prompt
    assert captured["output_config"]["format"]["schema"] is not None and captured["max_tokens"] >= 16000
    assert _build_prompt("X", metrics) .count("T1:") == 0          # legs are optional


def test_one_ticker_failure_does_not_fail_the_run(tmp_path, monkeypatch):
    def respond(kwargs):
        text = kwargs["messages"][0]["content"][1]["text"]
        if "analyzing BAD " in text:
            raise RuntimeError("503 overloaded")
        if "analyzing ODD " in text:
            return _Response(content=[_TextBlock("{broken")])
        return _Response(content=[_TextBlock(json.dumps(VALID_VERDICT))])
    monkeypatch.setattr("anthropic.Anthropic", lambda: _DummyAnthropicClient(respond))
    monkeypatch.setattr("pipeline.stage_charts.fetch_chart_data", lambda symbol, config: make_vcp_df())
    symbols = ["GOOD", "BAD", "ODD", "FINE"]
    result = run_vcp_analysis(pd.DataFrame({"Symbol": symbols}), {s: _fake_chart_path(tmp_path, s) for s in symbols},
                              config={**CONFIG, "chart_dir": str(tmp_path), "vcp_live_analysis": True})
    rows = result.set_index("Symbol")
    assert list(result["Symbol"]) == symbols
    assert "503 overloaded" in rows.loc["BAD", "error"] and rows.loc["ODD", "error"] == "invalid_json"
    assert rows.loc["GOOD", "entry_recommendation"] == "wait_for_breakout" and pd.isna(rows.loc["FINE", "error"])
    assert rows.loc["GOOD", "vcp_numeric_quality"] == rows.loc["BAD", "vcp_numeric_quality"]   # metrics kept on error rows
