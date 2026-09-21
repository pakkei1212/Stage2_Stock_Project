"""pipeline.vcp_eval: repeated, non-persistent Claude VCP evaluation (Claude mocked)."""
import json
import os

from conftest import make_vcp_df
from pipeline import vcp_eval
from pipeline.config import CONFIG


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    stop_reason = "end_turn"
    usage = None

    def __init__(self, text):
        self.content = [_Block(text)]


class _Client:
    def __init__(self, verdicts):
        self.verdicts, self.calls = list(verdicts), 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return _Resp(json.dumps(v))


def _verdict(**kw):
    base = {"is_vcp_pattern": True, "pattern_stage": "mature", "contraction_count_observed": 3,
            "volume_dry_up_confirmed": True, "pivot_price": 117.8, "suggested_stop_loss": 108.0,
            "confidence": "high", "entry_recommendation": "wait_for_breakout", "rationale": "ok"}
    base.update(kw)
    return base


def test_repeated_evaluation_reports_stability_and_never_touches_history(tmp_path):
    cfg = {**CONFIG, "report_dir": str(tmp_path), "history_db_path": str(tmp_path / "h.sqlite3")}
    client = _Client([_verdict(), _verdict(pivot_price=118.5), _verdict(confidence="medium"),
                      _verdict(is_vcp_pattern=False, pattern_stage="not_present", entry_recommendation="avoid"),
                      RuntimeError("overloaded"), _verdict(is_vcp_pattern=False, pattern_stage="not_present",
                                                          entry_recommendation="avoid")])
    rendered = []

    def render(symbol, df, config):
        path = os.path.join(config["chart_dir"], f"{symbol}.png")
        os.makedirs(config["chart_dir"], exist_ok=True)
        open(path, "wb").write(b"png")
        rendered.append(path)
        return path

    results, stats, out_dir = vcp_eval.evaluate(["AAA", "BBB"], 3, cfg, client=client,
                                                fetch=lambda s, c: make_vcp_df(), render=render,
                                                out_dir=str(tmp_path / "eval"))
    assert client.calls == 6 and stats["ok"] == 5 and stats["errors"] == 1
    a, b = (r["stability"] for r in results)
    assert a["is_vcp_pattern"]["stable"] and not a["confidence"]["stable"] and a["pivot_price"]["spread_pct"] > 0
    assert b["successful_runs"] == 2 and b["failed_runs"] == 1 and b["all_stable"]
    assert results[0]["numeric"]["vcp_numeric_quality"] is not None
    assert set(results[0]["selective_routing"]) == {"requested", "reasons"}
    assert results[0]["runs"][0]["review"]["agreement"] in ("agree_positive", "numeric_negative_claude_positive")
    saved = json.load(open(tmp_path / "eval" / "results.json", encoding="utf-8"))
    assert saved["repeats"] == 3 and len(saved["results"]) == 2
    assert not os.path.exists(cfg["history_db_path"])                 # no SQLite history written
    assert all(p.startswith(str(tmp_path / "eval")) for p in rendered)


def test_repeats_are_capped():
    cfg = dict(CONFIG)
    client = _Client([_verdict()] * vcp_eval.MAX_REPEATS)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        chart = os.path.join(d, "x.png")
        open(chart, "wb").write(b"png")
        vcp_eval.evaluate(["AAA"], 99, cfg, client=client, fetch=lambda s, c: make_vcp_df(),
                          render=lambda s, df, c: chart, out_dir=d)
    assert client.calls == vcp_eval.MAX_REPEATS
