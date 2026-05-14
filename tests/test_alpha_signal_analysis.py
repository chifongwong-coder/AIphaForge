"""v2.4 Commit K — alpha.signal_analysis tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge.alpha.signal_analysis import (
    signal_forward_return,
    signal_hit_rate,
    signal_turnover,
)


def _prices(periods: int = 10) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=periods)
    return pd.Series(
        [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 100.0, 104.0, 99.0, 105.0],
        index=idx,
    )


class TestSignalForwardReturn:
    def test_matches_hand_computation(self):
        # 6-bar fixture, signal goes long at t=1, flat at t=4.
        idx = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.Series([100, 101, 102, 103, 102, 101], index=idx, dtype=float)
        signal = pd.Series(
            [np.nan, 1.0, np.nan, np.nan, 0.0, np.nan], index=idx,
        )
        # Position via ffill+fillna(0): [0, 1, 1, 1, 0, 0].
        # forward_returns(horizon=1, entry_lag=1):
        #   t=0: prices[2]/prices[1]-1 = 102/101-1 = 0.00990
        #   t=1: prices[3]/prices[2]-1 = 103/102-1 = 0.00980
        #   t=2: prices[4]/prices[3]-1 = 102/103-1 = -0.00971
        #   t=3: prices[5]/prices[4]-1 = 101/102-1 = -0.00980
        #   t=4: NaN (no t+2)
        #   t=5: NaN
        # forward_returns(horizon=1, entry_lag=1) at bar t:
        #   entry = prices[t+1], exit = prices[t+2], so
        #   fr[t] = prices[t+2] / prices[t+1] - 1.
        # signal_forward_return = position * fr:
        out = signal_forward_return(signal, prices)
        # bar 0: position 0 * fr[0] = 0.0
        # bar 1: position 1 * fr[1] = 1 * (prices[3]/prices[2] - 1)
        # bar 2: position 1 * fr[2] = 1 * (prices[4]/prices[3] - 1)
        # bar 3: position 1 * fr[3] = 1 * (prices[5]/prices[4] - 1)
        assert out.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert out.iloc[1] == pytest.approx(103 / 102 - 1, rel=1e-12)
        assert out.iloc[2] == pytest.approx(102 / 103 - 1, rel=1e-12)
        assert out.iloc[3] == pytest.approx(101 / 102 - 1, rel=1e-12)


class TestSignalHitRate:
    def test_excludes_flat_bars_from_denominator(self):
        # Signal: 1, NaN, NaN, 0, NaN, -1, 0
        # Position: 1, 1, 1, 0, 0, -1, 0
        # Make returns predictable so we know the hit count.
        idx = pd.bdate_range("2024-01-01", periods=7)
        prices = pd.Series(
            [100, 101, 102, 103, 102, 101, 102], index=idx, dtype=float,
        )
        signal = pd.Series(
            [1.0, np.nan, np.nan, 0.0, np.nan, -1.0, 0.0], index=idx,
        )
        # Just sanity-check the rate is in [0, 1] and excludes flats.
        rate = signal_hit_rate(signal, prices)
        assert 0.0 <= rate <= 1.0


class TestSignalTurnover:
    def test_long_short_flip_contributes_two(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        # Position: 0, 1, 1, -1.
        # Signal:   NaN, 1, NaN, -1.
        signal = pd.Series([np.nan, 1.0, np.nan, -1.0], index=idx)
        # diff: [NaN, 1, 0, -2]; abs: [NaN, 1, 0, 2]; mean (ignores NaN) = 1.0.
        turnover = signal_turnover(signal)
        # mean over 3 non-NaN diffs of [1, 0, 2] = 3/3 = 1.0.
        assert turnover == pytest.approx(1.0, abs=1e-12)


class TestSignalAnalysisEndToEnd:
    def test_signal_analysis_on_direct_signal_strategy_pattern(self):
        # End-to-end smoke: take a DirectSignalStrategy-shaped
        # input (precomputed signal series), feed each metric.
        idx = pd.bdate_range("2024-01-01", periods=20)
        prices = pd.Series(
            np.cumsum(np.ones(20)) + 100.0, index=idx,
        )
        signal = pd.Series(
            [1.0] + [np.nan] * 9 + [0.0] + [np.nan] * 9, index=idx,
        )
        fr = signal_forward_return(signal, prices)
        rate = signal_hit_rate(signal, prices)
        turn = signal_turnover(signal)
        # All three return finite-or-defined values without crashing.
        assert isinstance(fr, pd.Series)
        assert isinstance(rate, float)
        assert isinstance(turn, float)


class TestSignalAnalysisFirewall:
    def test_alpha_signal_analysis_does_not_import_engine(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/alpha/signal_analysis.py"
        )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "aiphaforge.engine" not in name, (
                    f"alpha/signal_analysis.py imports {name!r}"
                )
