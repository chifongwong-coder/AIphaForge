"""v2.8.6 Commit A — load_yahoo behavior against stubbed yfinance.

yfinance >= 1.x returns MultiIndex (Price, Ticker) columns even for a
single symbol; the pre-fix single-symbol branch crashed on
``'tuple' object has no attribute 'lower'``. All tests inject a stub
``yfinance`` module via ``sys.modules`` — no network, no real
yfinance dependency (the in-function import consults sys.modules
first, so the stub wins even if the real package is installed).
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from aiphaforge.data import load_yahoo

_PRICE_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
_EXPECTED_COLS = ["open", "high", "low", "close", "volume"]


def _plain_frame(periods: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=periods)
    base = pd.Series(
        range(100, 100 + periods), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": [1e6] * periods,
        },
        index=idx,
    )


def _multiindex_frame(tickers: list[str]) -> pd.DataFrame:
    flat = _plain_frame()
    cols = pd.MultiIndex.from_product(
        [_PRICE_FIELDS, tickers], names=["Price", "Ticker"])
    df = pd.DataFrame(index=flat.index, columns=cols, dtype=float)
    for ticker in tickers:
        for field in _PRICE_FIELDS:
            df[(field, ticker)] = flat[field]
    return df


def _stub_yfinance(monkeypatch, frame: pd.DataFrame) -> None:
    stub = types.ModuleType("yfinance")

    def download(symbols, start=None, end=None, auto_adjust=True,
                 progress=False):
        return frame

    stub.download = download
    monkeypatch.setitem(sys.modules, "yfinance", stub)


def test_single_symbol_multiindex_columns_flattened(monkeypatch):
    _stub_yfinance(monkeypatch, _multiindex_frame(["SPY"]))
    df = load_yahoo("SPY", start="2024-01-01")
    assert not isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == _EXPECTED_COLS


def test_single_symbol_plain_columns_regression(monkeypatch):
    _stub_yfinance(monkeypatch, _plain_frame())
    df = load_yahoo("SPY")
    assert list(df.columns) == _EXPECTED_COLS


def test_multi_symbol_returns_dict_per_ticker(monkeypatch):
    _stub_yfinance(monkeypatch, _multiindex_frame(["AAA", "BBB"]))
    result = load_yahoo(["AAA", "BBB"])
    assert set(result) == {"AAA", "BBB"}
    for df in result.values():
        assert list(df.columns) == _EXPECTED_COLS


def test_empty_download_raises_value_error(monkeypatch):
    _stub_yfinance(monkeypatch, pd.DataFrame())
    with pytest.raises(ValueError, match="No data returned"):
        load_yahoo("SPY")


def test_yfinance_missing_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"aiphaforge\[data\]"):
        load_yahoo("SPY")
