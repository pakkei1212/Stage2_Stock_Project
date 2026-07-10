import pandas as pd

from pipeline.data_sources import batch_download


def test_batch_download_skips_failed_tickers(monkeypatch):
    class DummyYF:
        def download(self, batch, **kwargs):
            if isinstance(batch, list):
                raise RuntimeError('TLS connect error: invalid library')
            raise AssertionError('unexpected input')

    monkeypatch.setattr('pipeline.data_sources.yf.download', DummyYF().download)

    result = batch_download(['BCAR'], config={'batch_size': 1, 'max_retries': 1, 'batch_sleep_sec': 0})

    assert result == {}
