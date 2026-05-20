"""v2.8.1 H1 — DefaultTradeCost representative-notional fix.

v2.8.0 bug: ``DefaultTradeCost.apply_vectorized`` called
``fee_model.estimate_commission_rate()`` with no args, getting
``(price=100, size=100)`` defaults. For US stock fee models with
``min_commission=$1.0``, that produced a 1% cost rate applied to every
trade — ~100× the real per-trade rate.

v2.8.1 fix: query the rate with a representative ``(price, size)``
pair derived from ``BacktestEngine.position_sizer + initial_capital``,
or from a user-passed ``representative_notional`` kwarg. Degenerate
inputs (no notional derivable, all-NaN close, etc.) fall through to
zero cost with a one-time warning — NOT to the bare-default 1% bug.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    ChinaAShareFeeModel,
    CryptoFuturesFeeModel,
    CryptoSpotFeeModel,
    SimpleFeeModel,
    USStockFeeModel,
    ZeroFeeModel,
)
from aiphaforge.config import BacktestConfig
from aiphaforge.costs import DefaultTradeCost
from aiphaforge.position_sizing import FixedSizer, FractionSizer


# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------

def _make_price_data(n: int = 252, base: float = 150.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = 0.0005 + 0.01 * rng.normal(size=n)
    close = base * np.cumprod(1 + rets)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
         "volume": 1e6},
        index=idx,
    )


def _signals_alternating(n: int) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    s = pd.Series(0.0, index=idx)
    # One round-trip per ~20 bars; gives a handful of trades.
    s.iloc[::20] = 1.0
    s.iloc[10::20] = 0.0
    return s


# --------------------------------------------------------------------
# Test 1 — H1 headline: US-stock vectorized rate is no longer inflated
# --------------------------------------------------------------------

def test_default_trade_cost_vectorized_us_stock_not_inflated():
    """US stock vectorized cost rate must reflect actual per-trade size,
    not the 1% bare-default that v2.8.0 produced."""
    data = _make_price_data(n=252, base=150.0)
    signals = _signals_alternating(len(data))

    fee_model = USStockFeeModel(commission_per_share=0.005, min_commission=1.0)

    eng = BacktestEngine(
        fee_model=fee_model,
        initial_capital=100_000,
        mode="vectorized",
        position_size=0.95,
    )
    eng.set_signals(signals)
    result = eng.run(data)

    # The bug produced equity ~0.4x after a year of weekly trading
    # (1% per trade × ~25 trades). Fixed version should produce
    # equity within a few percent of the no-cost run.
    final_equity = result.equity_curve.iloc[-1]
    assert final_equity > 0.5 * 100_000, (
        f"v2.8.0 bug regression: US-stock vectorized equity collapsed "
        f"to {final_equity:.0f} (expected > $50k after cost fix)."
    )


# --------------------------------------------------------------------
# Test 2 — vectorized ≈ event-driven for linear fee models (tight)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "fee_model_factory",
    [
        lambda: SimpleFeeModel(commission_rate=0.001, slippage_pct=0.0005),
        lambda: CryptoSpotFeeModel(),
        lambda: CryptoFuturesFeeModel(),
        lambda: ZeroFeeModel(),
    ],
    ids=["Simple", "CryptoSpot", "CryptoFutures", "Zero"],
)
def test_default_trade_cost_vectorized_matches_event_driven_linear(fee_model_factory):
    """For linear-in-notional fee models, vectorized and event-driven
    final equity should match within 5% (tight tolerance)."""
    data = _make_price_data(n=252, base=150.0)
    signals = _signals_alternating(len(data))

    eng_vec = BacktestEngine(
        fee_model=fee_model_factory(),
        initial_capital=100_000,
        mode="vectorized",
        position_size=0.95,
    )
    eng_vec.set_signals(signals)
    res_vec = eng_vec.run(data)

    eng_evt = BacktestEngine(
        fee_model=fee_model_factory(),
        initial_capital=100_000,
        mode="event_driven",
        position_size=0.95,
    )
    eng_evt.set_signals(signals)
    res_evt = eng_evt.run(data)

    ratio = res_vec.equity_curve.iloc[-1] / res_evt.equity_curve.iloc[-1]
    assert 0.95 <= ratio <= 1.05, (
        f"vectorized/event-driven equity ratio {ratio:.3f} outside "
        f"[0.95, 1.05] for {fee_model_factory().__class__.__name__}"
    )


# --------------------------------------------------------------------
# Test 3 — vectorized ≈ event-driven for min-commission models (loose)
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "fee_model_factory",
    [
        lambda: USStockFeeModel(commission_per_share=0.005, min_commission=1.0),
        lambda: ChinaAShareFeeModel(),
    ],
    ids=["USStock", "ChinaAShare"],
)
def test_default_trade_cost_vectorized_matches_event_driven_min_commission(
    fee_model_factory,
):
    """For fee models with a min-commission floor, per-trade rate
    varies with size; vectorized uses a single representative rate, so
    tolerance is wider [0.7, 1.4]."""
    data = _make_price_data(n=252, base=150.0)
    signals = _signals_alternating(len(data))

    eng_vec = BacktestEngine(
        fee_model=fee_model_factory(),
        initial_capital=100_000,
        mode="vectorized",
        position_size=0.95,
    )
    eng_vec.set_signals(signals)
    res_vec = eng_vec.run(data)

    eng_evt = BacktestEngine(
        fee_model=fee_model_factory(),
        initial_capital=100_000,
        mode="event_driven",
        position_size=0.95,
    )
    eng_evt.set_signals(signals)
    res_evt = eng_evt.run(data)

    ratio = res_vec.equity_curve.iloc[-1] / res_evt.equity_curve.iloc[-1]
    assert 0.7 <= ratio <= 1.4, (
        f"vectorized/event-driven equity ratio {ratio:.3f} outside "
        f"[0.7, 1.4] for {fee_model_factory().__class__.__name__}"
    )


# --------------------------------------------------------------------
# Test 4 — degenerate input falls through safely (zero cost + warn)
# --------------------------------------------------------------------

def test_default_trade_cost_vectorized_nan_data_falls_back_safely():
    """When data['close'] is all-NaN, the cost path must NOT fall back
    to no-args estimate_commission_rate() (that's the v2.8.0 bug).
    Expected: zero cost + UserWarning."""
    n = 50
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    data = pd.DataFrame(
        {"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
         "volume": 1.0},
        index=idx,
    )
    positions = pd.Series(0.5, index=idx)
    positions.iloc[0] = 0.0  # one trade
    returns = pd.Series(0.0, index=idx)

    tc = DefaultTradeCost()
    fee_model = USStockFeeModel(commission_per_share=0.005, min_commission=1.0)

    # Reset the module-level "once" flag so the warning fires for this test.
    import aiphaforge.costs as costs_module
    costs_module._DEGENERATE_WARNED = False

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        net = tc.apply_vectorized(
            returns, positions, data, fee_model, initial_capital=100_000,
            representative_notional=10_000.0,  # provided, but close is all-NaN
        )

    # Returns unchanged.
    assert (net.fillna(0) == returns.fillna(0)).all()
    # Warning fired.
    user_warnings = [w for w in captured if issubclass(w.category, UserWarning)]
    assert any("representative" in str(w.message).lower() for w in user_warnings)


# --------------------------------------------------------------------
# Test 5 — BacktestConfig pickles cleanly with the new fields
# --------------------------------------------------------------------

def test_backtest_config_pickle_with_legacy_no_representative_notional():
    """Pickle round-trip of a freshly-constructed BacktestConfig keeps
    representative_notional / representative_size at their None default
    (defensive future-proofing per R9)."""
    cfg = BacktestConfig(initial_capital=50_000)
    blob = pickle.dumps(cfg)
    revived = pickle.loads(blob)
    assert revived.representative_notional is None
    assert revived.representative_size is None
    assert revived.initial_capital == 50_000


# --------------------------------------------------------------------
# Test 6 — Q3 user-wins precedence (BacktestEngine kwarg path)
# --------------------------------------------------------------------

def test_user_passed_representative_notional_wins_over_engine_default():
    """User-passed BacktestEngine(representative_notional=X) must
    survive _build_config() untouched — engine only computes when None."""
    eng = BacktestEngine(
        initial_capital=100_000,
        mode="vectorized",
        position_size=0.5,
        representative_notional=99_999.0,
    )
    cfg = eng._build_config()
    assert cfg.representative_notional == 99_999.0, (
        f"engine overwrote user-passed representative_notional: "
        f"{cfg.representative_notional}"
    )


# --------------------------------------------------------------------
# Test 7 — B1 FixedSizer path computes non-zero cost
# --------------------------------------------------------------------

def test_fixed_sizer_vectorized_uses_size_times_median_close():
    """FixedSizer in vectorized mode must NOT collapse to zero cost.
    Cost should match event-driven (with min-commission floor tolerance)."""
    data = _make_price_data(n=252, base=200.0)
    signals = _signals_alternating(len(data))

    # FixedSizer(size=50 shares); at $200/share notional ≈ $10k per trade.
    fee_model = USStockFeeModel(commission_per_share=0.005, min_commission=1.0)

    eng_vec = BacktestEngine(
        fee_model=fee_model,
        initial_capital=100_000,
        mode="vectorized",
        position_sizing="fixed_size",
        position_size=50.0,
    )
    eng_vec.set_signals(signals)
    res_vec = eng_vec.run(data)

    eng_evt = BacktestEngine(
        fee_model=fee_model,
        initial_capital=100_000,
        mode="event_driven",
        position_sizing="fixed_size",
        position_size=50.0,
    )
    eng_evt.set_signals(signals)
    res_evt = eng_evt.run(data)

    # Non-zero cost: vectorized run must not be ≥ event-driven by more
    # than the min-commission floor dispersion (i.e. NOT zero-cost).
    cost_vec = 100_000 - res_vec.equity_curve.iloc[-1] + (
        res_evt.equity_curve.iloc[-1] - 100_000
    )
    # The B1 path produces a real cost — the run should not show
    # NO trading-cost drag at all. Compare to a true zero-fee run:
    eng_zero = BacktestEngine(
        fee_model=ZeroFeeModel(),
        initial_capital=100_000,
        mode="vectorized",
        position_sizing="fixed_size",
        position_size=50.0,
    )
    eng_zero.set_signals(signals)
    res_zero = eng_zero.run(data)
    # FixedSizer + USStockFeeModel must show MORE cost than ZeroFeeModel.
    assert res_vec.equity_curve.iloc[-1] < res_zero.equity_curve.iloc[-1], (
        "FixedSizer vectorized produced zero cost — B1 path not engaged."
    )

    # And within the loose [0.7, 1.4] band of event-driven.
    ratio = res_vec.equity_curve.iloc[-1] / res_evt.equity_curve.iloc[-1]
    assert 0.7 <= ratio <= 1.4, (
        f"FixedSizer vectorized vs event-driven ratio {ratio:.3f} "
        f"outside [0.7, 1.4]"
    )


# --------------------------------------------------------------------
# Test 8 — engine helper resolves sizer types correctly
# --------------------------------------------------------------------

def test_engine_resolve_representative_trade_per_sizer():
    """`_resolve_representative_trade` returns the right tuple per
    sizer type (Q-α formula for FractionSizer, B1 size for FixedSizer,
    user-pass precedence)."""
    # FractionSizer — min(0.95, max_position_size=1.0) * 100_000 = 95_000
    eng = BacktestEngine(
        initial_capital=100_000, position_size=0.95, max_position_size=1.0,
    )
    notional, size = eng._resolve_representative_trade()
    assert notional == pytest.approx(95_000.0)
    assert size is None

    # FractionSizer with max_position_size clamp.
    eng2 = BacktestEngine(
        initial_capital=100_000, position_size=0.95, max_position_size=0.3,
    )
    notional2, _ = eng2._resolve_representative_trade()
    assert notional2 == pytest.approx(30_000.0)  # min(0.95, 0.3) * 100k

    # FixedSizer — returns size, no notional.
    eng3 = BacktestEngine(
        initial_capital=100_000, position_sizing="fixed_size", position_size=42.0,
    )
    notional3, size3 = eng3._resolve_representative_trade()
    assert notional3 is None
    assert size3 == 42.0

    # User-passed wins.
    eng4 = BacktestEngine(
        initial_capital=100_000, position_size=0.95,
        representative_notional=12_345.0,
    )
    notional4, _ = eng4._resolve_representative_trade()
    assert notional4 == 12_345.0
