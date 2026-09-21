"""Stage H end-to-end, no network: scheduled check -> NASDAQ session -> data
ready -> the real run_screen (Stage A-D2/F/G stubbed with synthetic data,
real Stage E ranking) -> SQLite -> trend comparison -> Telegram payload via the
real TelegramClient over a fake HTTP session."""
import logging
from datetime import date

import pandas as pd

from pipeline import history_store as hs
from pipeline import run_pipeline
from pipeline import scheduler as sch
from pipeline import telegram_notifier as tg
from pipeline.logging_config import SecretRedactingFilter
from pipeline.stage_rank import FUND_COLS, TECH_COLS
from stage_h_fakes import FAKE_TOKEN, FakeClock, make_config, make_deps, sgt

FRI = date(2026, 9, 18)
MON = date(2026, 9, 21)


class _Resp:
    status_code = 200

    def __init__(self, mid):
        self._mid = mid

    def json(self):
        return {"ok": True, "result": {"message_id": self._mid}}


class _RecordingSession:
    def __init__(self):
        self.posts = []

    def post(self, url, data=None, files=None, timeout=None):
        self.posts.append((url.rsplit("/", 1)[-1], dict(data or {}), bool(files)))
        return _Resp(len(self.posts))


def _stub_stages(monkeypatch, universe):
    """universe: {symbol: (tech_true, fund_true, rs_3mo)} — drives real score_and_rank."""
    def stage_abc(config, stats=None):
        stats.update({"current_stage": "C", "universe": 4312, "liquidity": 1800, "market_cap": 900,
                      "sector_classified": 850, "stage_c": len(universe)})
        sector = pd.DataFrame([{"Symbol": s, "Sector": "Technology", "Industry": "Semis"} for s in universe])
        caps = pd.DataFrame([{"Symbol": s, "Market Cap": 5e10} for s in universe])
        return list(universe), sector, caps

    def stage_d(tickers, config):
        rows = []
        for s in tickers:
            tech, _, rs = universe[s]
            row = {"Symbol": s, "Last Close": 100.0, "RS vs NASDAQ (3mo)": rs}
            row.update({c: i < tech for i, c in enumerate(TECH_COLS)})
            rows.append(row)
        return pd.DataFrame(rows)

    def stage_d2(tickers, config):
        rows = []
        for s in tickers:
            fund = universe[s][1]
            row = {"Symbol": s, **{c: i < fund for i, c in enumerate(FUND_COLS)}, "Fundamentals Score": fund}
            rows.append(row)
        return pd.DataFrame(rows)

    def charts(tickers, config):
        import os
        os.makedirs(config["chart_dir"], exist_ok=True)
        paths = {}
        for t in tickers:
            p = os.path.join(config["chart_dir"], f"{t}.png")
            open(p, "wb").write(b"\x89PNG")
            paths[t] = p
        return paths

    def vcp(candidates, chart_paths, config):
        import os
        debug_dir = os.path.join(config["chart_dir"], "vcp_debug")          # the per-run chart dir
        os.makedirs(debug_dir, exist_ok=True)
        for s in candidates["Symbol"]:
            open(os.path.join(debug_dir, f"{s}.png"), "wb").write(b"\x89PNG debug")
        return pd.DataFrame([{"Symbol": s, "contraction_count": 3, "contraction_pcts": [18.0, 9.0, 4.0],
                              "is_vcp_pattern": True, "pattern_stage": "mature", "confidence": "high",
                              "entry_recommendation": "wait_for_breakout", "pivot_price": 101.0,
                              "suggested_stop_loss": 95.0, "rationale": "ok"} for s in candidates["Symbol"]])

    monkeypatch.setattr(run_pipeline, "run_stage_abc", stage_abc)
    monkeypatch.setattr(run_pipeline, "run_stage2_screen", stage_d)
    monkeypatch.setattr(run_pipeline, "fundamentals_screen", stage_d2)
    monkeypatch.setattr(run_pipeline, "generate_charts", charts)
    monkeypatch.setattr(run_pipeline, "run_vcp_analysis", vcp)


def test_end_to_end_two_sessions_trend_and_telegram(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "4242")
    cfg = make_config(tmp_path, telegram_enabled=True, telegram_send_charts=True, telegram_vcp_chart_max=2,
                      trend_top_n=3, vcp_top_n=3)
    session = _RecordingSession()

    def deps_for(clock, bar):
        d = make_deps(clock, latest_bar=bar)
        d.run_screen = run_pipeline.run_screen          # the real Stage A-G orchestration
        d.telegram_factory = lambda config: tg.TelegramClient.from_env(config, session=session,
                                                                        sleep=lambda s: None)
        return d

    # Session 1: Friday, processed Saturday 09:15 SGT.
    _stub_stages(monkeypatch, {"AAA": (8, 6, 0.3), "BBB": (8, 5, 0.2), "CCC": (8, 4, 0.1),
                               "DDD": (7, 3, 0.1), "EEE": (6, 2, 0.1), "FFF": (5, 1, 0.1)})
    clock = FakeClock(sgt("2026-09-19 09:15"))
    first = sch.check_and_run("scheduled", cfg, deps_for(clock, FRI))
    assert first.decision == sch.COMPLETED and first.trading_date == FRI

    # Session 2: Monday, processed Tuesday 09:15 SGT (weekend between). FFF jumps
    # from #6 to #1; CCC falls out of the Top 3.
    _stub_stages(monkeypatch, {"FFF": (8, 6, 0.9), "AAA": (8, 6, 0.3), "BBB": (8, 5, 0.2),
                               "CCC": (7, 4, 0.1), "DDD": (7, 3, 0.1), "EEE": (6, 2, 0.1)})
    clock.current = sgt("2026-09-22 09:15")
    session.posts.clear()
    caplog.set_level(logging.DEBUG, logger="pipeline")
    for h in logging.getLogger("pipeline").handlers:
        h.addFilter(SecretRedactingFilter())
    second = sch.check_and_run("scheduled", cfg, deps_for(clock, MON))
    assert second.decision == sch.COMPLETED and second.trading_date == MON
    assert second.notification_status == "SENT"

    store = hs.HistoryStore(cfg["history_db_path"])
    assert store.canonical_sessions() == [FRI, MON]
    mon = store.results_for_run(second.run_id)
    assert [r["symbol"] for r in mon] == ["FFF", "AAA", "BBB", "CCC", "DDD", "EEE"]   # full population
    assert mon[0]["vcp_status"] == "analyzed" and mon[5]["vcp_status"] is None
    run = store.get_run(second.run_id)
    assert run["status"] == hs.NOTIFIED and run["count_universe"] == 4312 and run["count_ranked"] == 6
    assert run["count_trend_template_pass"] == 3

    trend = sch.build_trend_report(store, MON, cfg)
    assert trend.previous_session == FRI and trend.missing_sessions == []
    fff = trend.entries[0]
    assert (fff.symbol, fff.previous_rank, fff.rank, fff.category) == ("FFF", 6, 1, "rising")
    assert trend.new_entries == ["FFF"]
    assert [(d.symbol, d.previous_rank, d.current_rank) for d in trend.dropped] == [("CCC", 3, 4)]
    aaa = next(e for e in trend.entries if e.symbol == "AAA")
    assert aaa.top_n_streak == 2 and aaa.rank_delta_1 == -1

    methods = [p[0] for p in session.posts]
    # main report, then the Stage-G verification charts, then the separate debug summary
    assert methods == ["sendMessage", "sendPhoto", "sendPhoto", "sendMessage"]
    text = session.posts[0][1]["text"]
    assert "Trading session: <b>2026-09-21</b>" in text
    assert "🆕 New: FFF #6 → #1" in text and "🚪 Left Top 3: CCC #3 → #4" in text
    # Claude says VCP but the stub stored no numeric rating -> a REVIEW, never READY
    assert "🟠 <b>REVIEW — FFF</b>" in text and "Claude: VCP · mature · wait_for_breakout · high" in text
    assert text.index("ACTION BOARD") < text.index("REVIEW — FFF") < text.index("OVERALL LEADERS") \
        < text.index("WATCHLIST CHANGES")
    assert "Stage D screened" not in text and "Universe:" not in text          # no stage summary here
    debug = session.posts[-1][1]["text"]
    assert debug.startswith("🛠 <b>DEBUG — DAILY PIPELINE</b>") and "Universe: 4,312" in debug
    captions = [p[1]["caption"] for p in session.posts if p[0] == "sendPhoto"]
    assert len(captions) == 2 and all("Claude: VCP · mature" in c for c in captions)      # VCP debug charts
    assert session.posts[0][1]["chat_id"] == "4242"
    assert FAKE_TOKEN not in caplog.text

    manifest = (tmp_path / "runs" / "2026-09-21" / "run.json").read_text(encoding="utf-8")
    assert '"telegram_status": "SENT"' in manifest and FAKE_TOKEN not in manifest
    assert FAKE_TOKEN not in (run["config_json"] or "")
