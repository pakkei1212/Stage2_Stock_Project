"""run_pipeline: the Stage A-G body extracted into run_screen() must behave as
main() always did (same artifacts, same early exits), while exposing the
ranked/VCP frames and stage counts Stage H needs."""
import os

import pandas as pd

from pipeline import data_sources as ds
from pipeline import run_pipeline
from pipeline.stage_rank import FUND_COLS, TECH_COLS


def _config(tmp_path):
    from pipeline.config import CONFIG
    return {**CONFIG, "report_dir": str(tmp_path / "reports"), "chart_dir": str(tmp_path / "charts"),
            "log_dir": str(tmp_path / "logs"), "vcp_top_n": 2}


def _stub(monkeypatch, stage_c_symbols=("AAA", "BBB", "CCC"), stage_d_empty=False):
    calls = {}

    def abc(config, stats=None):
        stats.update({"universe": 10, "liquidity": 6, "market_cap": 5, "sector_classified": 4,
                      "stage_c": len(stage_c_symbols)})
        sector = pd.DataFrame([{"Symbol": s, "Sector": "Tech", "Industry": "Semis"} for s in stage_c_symbols])
        caps = pd.DataFrame([{"Symbol": s, "Market Cap": 1e10} for s in stage_c_symbols])
        return list(stage_c_symbols), sector, caps

    def d(tickers, config):
        if stage_d_empty:
            return pd.DataFrame()
        rows = []
        for i, t in enumerate(tickers):
            row = {"Symbol": t, "Last Close": 50.0, "RS vs NASDAQ (3mo)": 0.1 * i}
            row.update({c: (i == 0 or j < 7) for j, c in enumerate(TECH_COLS)})
            rows.append(row)
        return pd.DataFrame(rows)

    def d2(tickers, config):
        return pd.DataFrame([{"Symbol": t, **{c: True for c in FUND_COLS}, "Fundamentals Score": 6} for t in tickers])

    def charts(tickers, config):
        calls["chart_dir"] = config["chart_dir"]
        return {t: os.path.join(config["chart_dir"], f"{t}.png") for t in tickers}

    def vcp(candidates, chart_paths, config):
        calls["vcp_symbols"] = list(candidates["Symbol"])
        return pd.DataFrame([{"Symbol": s, "is_vcp_pattern": s == "AAA", "entry_recommendation":
                              "buy_now" if s == "AAA" else "avoid", "pattern_stage": "mature",
                              "confidence": "high"} for s in candidates["Symbol"]])

    monkeypatch.setattr(run_pipeline, "run_stage_abc", abc)
    monkeypatch.setattr(run_pipeline, "run_stage2_screen", d)
    monkeypatch.setattr(run_pipeline, "fundamentals_screen", d2)
    monkeypatch.setattr(run_pipeline, "generate_charts", charts)
    monkeypatch.setattr(run_pipeline, "run_vcp_analysis", vcp)
    return calls


def test_run_screen_returns_frames_artifacts_and_stage_counts(tmp_path, monkeypatch):
    calls = _stub(monkeypatch)
    stats = {}
    result = run_pipeline.run_screen(_config(tmp_path), "20260915_091500", stats)

    assert list(result.final_watchlist["Symbol"])[:1] == ["AAA"]
    assert calls["vcp_symbols"] == list(result.final_watchlist["Symbol"].head(2))
    assert calls["chart_dir"] == os.path.join(str(tmp_path / "charts"), "20260915_091500")
    assert result.vcp_debug_dir == os.path.join(calls["chart_dir"], "vcp_debug")
    assert os.path.exists(result.watchlist_path) and os.path.exists(result.vcp_report_path)
    assert stats["universe"] == 10 and stats["stage_d"] == 3 and stats["ranked"] == 3
    assert stats["trend_template_pass"] == 1 and stats["vcp_analyzed"] == 2
    assert stats["vcp_pattern"] == 1 and stats["vcp_actionable"] == 1
    assert stats["current_stage"] is None


def test_run_screen_early_exit_after_stage_d_matches_main(tmp_path, monkeypatch):
    _stub(monkeypatch, stage_d_empty=True)
    result = run_pipeline.run_screen(_config(tmp_path), "20260915_091500", {})
    assert result.stopped_after == "D" and result.final_watchlist.empty
    assert not os.path.exists(tmp_path / "reports")     # no watchlist written, as before


def test_main_still_runs_the_same_pipeline(tmp_path, monkeypatch):
    calls = _stub(monkeypatch)
    monkeypatch.setattr(run_pipeline, "setup_logging", lambda config, run_timestamp=None: None)
    assert run_pipeline.main(_config(tmp_path)) is None
    assert os.listdir(tmp_path / "reports")
    assert "vcp_symbols" in calls


def test_run_stage_abc_stats_are_optional_and_do_not_change_results(monkeypatch):
    universe = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Security Name": ["a", "b", "c"]})
    monkeypatch.setattr(ds, "load_universe", lambda config: universe)
    monkeypatch.setattr(ds, "cheap_liquidity_filter", lambda df, config: (["AAA", "BBB"], pd.DataFrame()))
    monkeypatch.setattr(ds, "market_cap_filter", lambda t, config: (["AAA", "BBB"],
                        pd.DataFrame({"Symbol": ["AAA", "BBB"], "Market Cap": [1e10, 1e10]})))
    monkeypatch.setattr(ds, "get_sector_map", lambda t: pd.DataFrame(
        {"Symbol": ["AAA", "BBB"], "Sector": ["Tech", "Unknown"], "Industry": ["x", "y"]}))
    strength = pd.DataFrame({"Sector": ["Tech"], "Relative Strength": [0.1]})
    monkeypatch.setattr(ds, "rank_sector_strength", lambda df, config: (
        strength, pd.DataFrame({"Symbol": ["AAA"], "Sector": ["Tech"]})))
    cfg = {"top_n_sectors": 5}

    plain = ds.run_stage_abc(cfg)
    stats = {}
    with_stats = ds.run_stage_abc(cfg, stats=stats)
    assert plain[0] == with_stats[0] == ["AAA"]
    assert stats == {"current_stage": "C", "universe": 3, "liquidity": 2, "market_cap": 2,
                     "sector_classified": 1, "stage_c": 1}
