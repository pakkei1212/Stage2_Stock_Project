"""Stage F: verify chart rendering handles empty data gracefully and produces
a real PNG file for valid data, without touching the network or the real
data/charts directory.
"""
import os

from conftest import make_ohlcv
from pipeline.stage_charts import render_chart


def test_render_chart_returns_none_for_empty_data():
    import pandas as pd

    assert render_chart("EMPTY", pd.DataFrame(), config={"chart_lookback_days": 130, "chart_dir": "unused"}) is None
    assert render_chart("NONE", None, config={"chart_lookback_days": 130, "chart_dir": "unused"}) is None


def test_render_chart_writes_a_png_file_for_valid_data(tmp_path):
    df = make_ohlcv(n=200, seed=1)
    config = {"chart_lookback_days": 130, "chart_dir": str(tmp_path)}

    out_path = render_chart("TEST", df, config=config)

    assert out_path == os.path.join(str(tmp_path), "TEST.png")
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
