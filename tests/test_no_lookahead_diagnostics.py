"""v2.3 Commit I — no-lookahead diagnostics tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.diagnostics import (
    assert_factor_no_lookahead,
    assert_signal_no_lookahead,
)


def _ohlcv(periods: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {"close": closes}, index=pd.bdate_range("2024-01-01", periods=periods),
    )


class TestAssertFactorNoLookahead:
    def test_passes_for_rolling_mean(self):
        # Rolling mean is a textbook no-lookahead factor.
        def factor(data):
            return data["close"].rolling(10).mean()

        assert_factor_no_lookahead(factor, _ohlcv())

    def test_fails_for_full_sample_zscore(self):
        # Full-sample z-score reads mean / std over the ENTIRE series
        # — at t, the value depends on data[t+1:]. This is the
        # canonical lookahead bug that NaN-tail misses but
        # prefix-slice catches (master plan §6.2 rationale #1).
        def lookahead_factor(data):
            close = data["close"]
            return (close - close.mean()) / close.std()

        with pytest.raises(AssertionError, match="lookahead"):
            assert_factor_no_lookahead(lookahead_factor, _ohlcv())

    def test_fails_for_shift_minus_one(self):
        # Classic future-leak: shift(-1) returns the NEXT bar's
        # value at the current index. Detected via NaN-state
        # mismatch — the truncated edge becomes NaN while the
        # full-data version still carries the future bar's value.
        def lookahead_factor(data):
            return data["close"].shift(-1)

        with pytest.raises(AssertionError, match="lookahead|NaN-state"):
            assert_factor_no_lookahead(lookahead_factor, _ohlcv())


class TestAssertSignalNoLookahead:
    def test_passes_for_threshold_strategy_callable(self):
        # Plain callable form (no generate_signals method).
        def signal(data):
            close = data["close"]
            sma = close.rolling(10).mean()
            sig = pd.Series(np.nan, index=close.index, dtype=float)
            sig[close > sma] = 1.0
            sig[close < sma] = -1.0
            return sig

        assert_signal_no_lookahead(signal, _ohlcv())

    def test_fails_for_strategy_using_future_label(self):
        # generate_signals form on a strategy that peeks at the next
        # bar's price.
        class CheaterStrategy:
            def generate_signals(self, data):
                next_close = data["close"].shift(-1)
                sig = pd.Series(np.nan, index=data.index, dtype=float)
                sig[next_close > data["close"]] = 1.0
                sig[next_close < data["close"]] = -1.0
                return sig

        with pytest.raises(AssertionError, match="lookahead|NaN-state"):
            assert_signal_no_lookahead(CheaterStrategy(), _ohlcv())
