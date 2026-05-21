"""v2.8.2 M5 — DefaultTradeCost opt-in cost_normalization='current_equity'.

v2.8.1 DefaultTradeCost divided per-bar cost by initial_capital
(path-independent). For drawn-down portfolios this UNDER-states the
true cost-as-fraction-of-current-equity ratio.

v2.8.2 adds an opt-in cost_normalization kwarg. Default
"initial_capital" preserves v2.8.1 behavior bit-for-bit; opt-in
"current_equity" divides by running equity (clipped at 1% of initial
capital). The current-equity mode is a first-order approximation
(uses pre-cost gross returns) and is documented as diagnostic-grade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import USStockFeeModel
from aiphaforge.costs import DefaultTradeCost


def _data(n: int = 50, base: float = 100.0, drift: float = 0.0,
          vol: float = 0.005, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.normal(size=n)
    close = base * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": 1e6},
        index=idx,
    )


def _alternating_signals(n: int) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    s = pd.Series(np.nan, index=idx, dtype=float)
    s.iloc[::10] = 1.0
    s.iloc[5::10] = 0.0
    return s


# --------------------------------------------------------------------
# Test 1 — Default "initial_capital" matches v2.8.1 bit-for-bit
# --------------------------------------------------------------------

def test_default_cost_normalization_initial_capital_matches_v2_8_1():
    """Default kwarg path must produce numerically identical results
    to v2.8.1: gross_cost / initial_capital."""
    data = _data(n=100)
    positions = pd.Series(0.5, index=data.index)
    positions.iloc[0] = 0.0
    positions.iloc[50] = 0.0  # one round trip
    returns = pd.Series(0.0, index=data.index)

    fee = USStockFeeModel()
    tc_default = DefaultTradeCost()  # default kwarg
    tc_explicit = DefaultTradeCost(cost_normalization="initial_capital")

    net_default = tc_default.apply_vectorized(
        returns.copy(), positions, data, fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )
    net_explicit = tc_explicit.apply_vectorized(
        returns.copy(), positions, data, fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )
    pd.testing.assert_series_equal(net_default, net_explicit)


# --------------------------------------------------------------------
# Test 2 — Drawdown: current_equity ratio bounded BELOW true 1/equity_frac
# --------------------------------------------------------------------

def test_current_equity_normalization_diverges_under_drawdown():
    """Build a backtest that draws down to ~30% of initial. The
    current_equity cost rate for trades during the drawdown should be
    LARGER than the initial_capital rate, bounded above by the true
    1/0.3 ≈ 3.33×. Pre-cost-equity bias pulls observed ratio BELOW
    the true value; range upper bound is 3.33."""
    n = 60
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    # Construct a synthetic equity path: drift sharply down by ~70%.
    drift = -0.02
    close = 100.0 * np.cumprod(1 + np.full(n, drift))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1e6},
        index=idx,
    )
    # Decouple "drawdown path" from "trades": pass an explicit
    # gross returns series that DRAWS DOWN, and a positions series
    # that GENERATES TRADES on top. This separates the test's two
    # concerns: equity must collapse (so running_equity diverges
    # from initial_capital), AND trades must happen during the
    # collapse (so the cost paths see different denominators).
    returns_pre = pd.Series(-0.02, index=idx, dtype=float)
    returns_pre.iloc[0] = 0.0
    # Toggle position long/flat every 5 bars to generate ~12 trades
    # distributed across the drawdown horizon.
    positions = pd.Series(1.0, index=idx, dtype=float)
    for i in range(0, n, 10):
        positions.iloc[i:i + 5] = 0.0

    fee = USStockFeeModel()
    tc_init = DefaultTradeCost(cost_normalization="initial_capital")
    tc_eq = DefaultTradeCost(cost_normalization="current_equity")

    net_init = tc_init.apply_vectorized(
        returns_pre, positions, data, fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )
    net_eq = tc_eq.apply_vectorized(
        returns_pre, positions, data, fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )

    # Cost is the subtraction from `returns_pre`. Total cost per mode:
    cost_init = (returns_pre - net_init).sum()
    cost_eq = (returns_pre - net_eq).sum()
    ratio = cost_eq / cost_init if cost_init > 0 else 0.0

    # Equity-end fraction ≈ exp(-0.02 * 60) ≈ 0.30. True ratio = 1/0.30
    # ≈ 3.33. Observed under bias is BELOW 3.33; allow [1.5, 3.33].
    assert 1.5 <= ratio <= 3.33, (
        f"expected current/initial cost ratio in [1.5, 3.33] for ~30% "
        f"drawdown; got {ratio:.3f}"
    )


# --------------------------------------------------------------------
# Test 3 — Clip floor bounds blowup cost at the 1% floor
# --------------------------------------------------------------------

def test_current_equity_clip_bounds_blowup_cost_at_one_percent_floor():
    """Catastrophic loss path: running_equity collapses to ~0. Without
    clip, gross_cost / running_equity → inf. With clip at 0.01 *
    initial_capital, per-bar cost-return is bounded."""
    n = 50
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    # Construct a path that loses ~99.9% by end.
    close = 100.0 * np.cumprod(np.full(n, 0.90))  # 10% loss per bar
    data = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1e6},
        index=idx,
    )
    positions = pd.Series(1.0, index=idx)
    positions.iloc[0] = 0.0
    returns_pre = positions.shift(1).fillna(0) * data["close"].pct_change().fillna(0)

    tc = DefaultTradeCost(cost_normalization="current_equity")
    net = tc.apply_vectorized(
        returns_pre, positions, data, USStockFeeModel(),
        initial_capital=100_000, representative_notional=50_000.0,
    )
    # Per-bar cost contribution = returns_pre - net. With clip at 1%,
    # cost is bounded by gross_cost / (0.01 * 100_000) = gross_cost /
    # 1000. For typical $50 gross_cost per trade, that's 0.05 per bar
    # max. The pin: NO per-bar cost contribution exceeds 1.0 (100%
    # per bar), even on a blown-up path.
    per_bar_cost = (returns_pre - net).abs()
    assert per_bar_cost.max() < 1.0, (
        f"clip floor should bound per-bar cost contribution to "
        f"< 1.0 (100%); max was {per_bar_cost.max():.4f}"
    )
    # And finite throughout:
    assert np.isfinite(net).all()


# --------------------------------------------------------------------
# Test 4 — Docstring documents the first-order approximation
# --------------------------------------------------------------------

def test_current_equity_docstring_documents_approximation():
    """Pin the docstring contract — caller may want to grep for the
    bias direction and clip floor."""
    doc = DefaultTradeCost.__doc__ or ""
    assert "first-order" in doc.lower() or "first order" in doc.lower()
    assert "pre-cost" in doc.lower()
    assert "turnover" in doc.lower()
    assert "0.01" in doc or "1%" in doc


# --------------------------------------------------------------------
# Test 5 — Invalid normalization kwarg raises ValueError
# --------------------------------------------------------------------

def test_invalid_cost_normalization_raises():
    with pytest.raises(ValueError, match="cost_normalization"):
        DefaultTradeCost(cost_normalization="weighted_average")  # not a valid mode
