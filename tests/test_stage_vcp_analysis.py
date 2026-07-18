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
    client = _DummyAnthropicClient(lambda kwargs: _Response(stop_reason="max_tokens", content=[]))

    verdict = analyze_chart(client, "TEST", chart_path, metrics=SAMPLE_METRICS, config=CONFIG)

    assert verdict == {"error": "no_text_block", "stop_reason": "max_tokens"}


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
