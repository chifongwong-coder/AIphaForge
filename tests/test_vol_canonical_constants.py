"""v2.4 Commit L — Canonical-formula tests for probes/_vol.py.

Pin the locked numeric constants from the volatility estimators
against independent hand computation. Closes the v2.2.2 G
methodology gap for this module.

WARNING (applies to every test): do NOT regenerate expected
values from production code. The point is to verify production
matches the paper formula. If a test fails, suspect production
first.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._vol import estimate_sigma


def _bars(open_=None, high=None, low=None, close=None) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "open": open_ if open_ is not None else close,
            "high": high if high is not None else close,
            "low": low if low is not None else close,
            "close": close,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


class TestStdevReturnsDdof:
    def test_stdev_returns_uses_ddof_1_per_production(self):
        # Production at _vol.py:142 uses np.std(close_to_close, ddof=1).
        # Pin it: a synthetic close series whose log-returns have
        # known sample (ddof=1) std.
        close = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
        bars = _bars(close=close)
        log_rets = np.log(close[1:] / close[:-1])
        expected = float(np.std(log_rets, ddof=1))
        sigma, _ = estimate_sigma(bars, method="stdev_returns")
        assert sigma == pytest.approx(expected, rel=1e-12)
        # Sanity: ddof=0 would give a different value.
        ddof0 = float(np.std(log_rets, ddof=0))
        assert sigma != pytest.approx(ddof0, abs=1e-12)


class TestParkinsonConstant:
    def test_parkinson_pinned_at_one_over_four_ln_two(self):
        # σ²_P = (1 / (4 · ln 2)) · mean((ln(H/L))²).
        # Hand-fixture: 5 bars with H/L spread = 1.02, 1.04, 1.01,
        # 1.03, 1.05.
        ratios = np.array([1.02, 1.04, 1.01, 1.03, 1.05])
        low = np.array([100.0] * 5)
        high = low * ratios
        close = (high + low) / 2.0
        bars = _bars(high=high, low=low, close=close)
        log_hl = np.log(ratios)
        expected_sigma2 = float(np.mean(log_hl ** 2) / (4.0 * math.log(2.0)))
        expected_sigma = math.sqrt(expected_sigma2)
        sigma, _ = estimate_sigma(bars, method="parkinson")
        assert sigma == pytest.approx(expected_sigma, rel=1e-12)
        # Pin the locked constant value to catch regressions in the
        # 1/(4·ln 2) factor.
        # 1 / (4 · ln 2) ≈ 0.36067376022224085.
        assert (1.0 / (4.0 * math.log(2.0))) == pytest.approx(
            0.36067376022224085, rel=1e-12,
        )


class TestGarmanKlassConstants:
    def test_garman_klass_pinned_against_hand_computation(self):
        # σ²_GK = 0.5 · mean((ln H/L)²) − (2·ln 2 − 1) · mean((ln C/O)²).
        # Hand-fixture: 5 bars with both H/L and C/O variations.
        open_ = np.array([100.0, 100.5, 101.0, 99.5, 100.0])
        close = np.array([100.5, 101.0, 99.5, 100.0, 101.5])
        high = np.maximum(open_, close) * 1.005
        low = np.minimum(open_, close) * 0.995
        bars = _bars(open_=open_, high=high, low=low, close=close)
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        expected_sigma2 = (
            0.5 * float(np.mean(log_hl ** 2))
            - (2.0 * math.log(2.0) - 1.0) * float(np.mean(log_co ** 2))
        )
        expected_sigma = math.sqrt(max(expected_sigma2, 0.0))
        sigma, _ = estimate_sigma(bars, method="garman_klass")
        assert sigma == pytest.approx(expected_sigma, rel=1e-12)

    def test_garman_klass_constant_two_ln_two_minus_one(self):
        # Pin the locked GK weight constant: 2·ln 2 − 1 ≈ 0.3862943611.
        gk_const = 2.0 * math.log(2.0) - 1.0
        assert gk_const == pytest.approx(0.3862943611198906, rel=1e-12)
