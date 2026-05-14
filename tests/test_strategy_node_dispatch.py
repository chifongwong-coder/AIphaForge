"""v2.5 Commit A — _resolve_child_signals helper unit tests."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    _resolve_child_signals,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    # Required: _FALLBACK_WARNED is module-level mutable state that
    # would leak across tests and make "first-time WARNING" assertions
    # order-dependent. Clear before AND after each test.
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _ohlcv(periods: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _ModernChild(BaseStrategy):
    """Test child that overrides generate_signals directly."""

    name = "modern_child"

    def generate_signals(self, data):
        if isinstance(data, dict):
            return {sym: pd.Series(1.0, index=df.index) for sym, df in data.items()}
        return pd.Series(1.0, index=data.index)


class _LegacyChild(BaseStrategy):
    """Test child that only implements _compute."""

    name = "legacy_child"

    def _compute(self, df):
        return pd.Series(-1.0, index=df.index)


class _BrokenModernChild(BaseStrategy):
    """generate_signals always raises; _compute returns flat zeros."""

    name = "broken_modern_child"

    def generate_signals(self, data):
        raise KeyError("close")

    def _compute(self, df):
        return pd.Series(0.0, index=df.index)


class _Composite:
    """Sentinel composite class for _resolve_child_signals' first arg."""


class TestAutoMode:
    def test_resolve_auto_uses_generate_signals_for_modern_child(self):
        df = _ohlcv()
        out = _resolve_child_signals(
            _Composite, _ModernChild(), df, mode="auto",
        )
        assert (out == 1.0).all()

    def test_resolve_auto_falls_back_to_compute_on_exception_single_asset(self):
        df = _ohlcv()
        out = _resolve_child_signals(
            _Composite, _BrokenModernChild(), df, mode="auto",
        )
        # Fell back to _compute which returns zeros.
        assert (out == 0.0).all()

    def test_resolve_auto_raises_on_multi_asset_fallback_attempt(self):
        data = {"A": _ohlcv(seed=1), "B": _ohlcv(seed=2)}
        with pytest.raises(ValueError, match="multi-asset"):
            _resolve_child_signals(
                _Composite, _BrokenModernChild(), data, mode="auto",
            )


class TestGenerateSignalsMode:
    def test_resolve_generate_signals_propagates_child_exceptions(self):
        df = _ohlcv()
        with pytest.raises(KeyError, match="close"):
            _resolve_child_signals(
                _Composite, _BrokenModernChild(), df, mode="generate_signals",
            )


class TestLegacyComputeMode:
    def test_resolve_legacy_compute_rejects_multi_asset_data(self):
        data = {"A": _ohlcv(seed=1)}
        with pytest.raises(TypeError, match="legacy_compute"):
            _resolve_child_signals(
                _Composite, _LegacyChild(), data, mode="legacy_compute",
            )


class TestFallbackWarningPolicy:
    def test_resolve_auto_first_fallback_logs_warning(self, caplog):
        df = _ohlcv()
        with caplog.at_level(logging.WARNING, logger="aiphaforge.strategies"):
            _resolve_child_signals(
                _Composite, _BrokenModernChild(), df, mode="auto",
            )
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "_BrokenModernChild" in r.message
        ]
        assert len(warnings) == 1, (
            f"expected exactly 1 WARNING, got {len(warnings)} "
            f"(records: {[r.message for r in caplog.records]})"
        )

    def test_resolve_auto_repeat_fallback_drops_to_debug(self, caplog):
        df = _ohlcv()
        child = _BrokenModernChild()
        # First call: WARNING.
        with caplog.at_level(logging.DEBUG, logger="aiphaforge.strategies"):
            _resolve_child_signals(_Composite, child, df, mode="auto")
            first_warnings = [
                r for r in caplog.records if r.levelno == logging.WARNING
            ]
            assert len(first_warnings) == 1
            caplog.clear()
            # Second call with same (composite_cls, child_cls): DEBUG only.
            _resolve_child_signals(_Composite, child, df, mode="auto")
            second_warnings = [
                r for r in caplog.records if r.levelno == logging.WARNING
            ]
            second_debugs = [
                r for r in caplog.records if r.levelno == logging.DEBUG
            ]
            assert second_warnings == [], "second fallback must not warn"
            assert len(second_debugs) >= 1, "second fallback should log debug"
