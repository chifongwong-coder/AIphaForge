"""v2.3 Commit C — backward compat for strategies._transitions_only.

The private name is preserved as a thin alias of the new public
``signals.transitions_only`` so any external caller that imported
``from aiphaforge.strategies import _transitions_only`` keeps
working. Snapshot tests pin built-in strategy outputs at the
v2.2.2 baseline so the alias indirection cannot silently change
strategy behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.signals import transitions_only
from aiphaforge.strategies import (
    MACrossover,
    RSIMeanReversion,
    _transitions_only,
)


def _deterministic_ohlcv(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * n,
        },
        index=idx,
    )


class TestPrivateAliasMatchesPublicHelper:
    def test_old_private_transitions_only_matches_new_public_function(self):
        # Cross-fixture sample to make sure the alias really delegates
        # rather than carrying a stale copy of the old implementation.
        for seed in [0, 1, 42, 2024, 9999]:
            rng = np.random.default_rng(seed)
            idx = pd.date_range("2024-01-01", periods=20)
            raw = pd.Series(
                rng.choice([1.0, 0.0, -1.0, np.nan], size=20, p=[0.3, 0.2, 0.3, 0.2]),
                index=idx,
            )
            old_out = _transitions_only(raw)
            new_out = transitions_only(raw)
            pd.testing.assert_series_equal(old_out, new_out)


class TestBuiltInStrategySignalSnapshots:
    """Pin v2.2.2 signal outputs so the alias indirection in Commit C
    cannot silently shift built-in strategy behavior."""

    def test_ma_crossover_signals_unchanged_snapshot(self):
        df = _deterministic_ohlcv()
        sig = MACrossover(short=5, long=20).generate_signals(df)
        non_nan = sig.dropna()
        # Snapshot captured against v2.2.2 baseline at:
        # np.random.default_rng(42), 60 bdates from 2024-01-01,
        # MACrossover(5, 20). Two transitions.
        assert list(non_nan.index.strftime("%Y-%m-%d")) == [
            "2024-01-26", "2024-03-20",
        ]
        assert list(non_nan.values) == [1.0, -1.0]

    def test_rsi_signals_unchanged_snapshot(self):
        df = _deterministic_ohlcv()
        sig = RSIMeanReversion(
            period=14, oversold=30, overbought=70,
        ).generate_signals(df)
        non_nan = sig.dropna()
        # One transition under v2.2.2 baseline.
        assert list(non_nan.index.strftime("%Y-%m-%d")) == ["2024-03-11"]
        assert list(non_nan.values) == [-1.0]
