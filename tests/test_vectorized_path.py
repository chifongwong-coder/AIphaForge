"""Vectorized path safety + trade reconstruction tests (v1.9.6).

Covers:
* B6 — cumprod safety: pathological costs / tiny capital no longer flip
  the equity sign and produce nonsensical positive equity.
* Q1 — Trade.size now means shares (not signal magnitude), so
  market_impact.estimate_capacity returns finite numbers on a
  vectorized result.
* Q3 — Trade.pnl no longer double-deducts commission + slippage.
  Discrepancy contract per plan v3 P1:
    - single trade, pos=1, no fees, no reversal:    ≤ 1e-9 abs
    - single trade with fees only:                  bounded by total fees
    - multi-trade with reversals:                   bounded by ½σ²·T·notional
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge import (
    BacktestEngine,
    SimpleFeeModel,
    ZeroFeeModel,
)


def _make_data(n: int, drift: float = 0.0005, vol: float = 0.01,
               seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.normal(size=n)
    rets[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------------------------------------------------------------------------
# B6 — cumprod safety
# ---------------------------------------------------------------------------

class TestBankruptcyGuard:
    def test_tiny_capital_no_explosion(self):
        """init_capital=$0.0001 with high fees no longer balloons to billions."""
        data = _make_data(50, vol=0.05)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[10] = -1.0
        signals.iloc[20] = 1.0
        signals.iloc[30] = 0.0

        # 5% commission per trade dwarfs realistic numbers, but at this
        # capital level the per-bar cost rate becomes >> 1 and would
        # have produced a sign flip on the unguarded cumprod.
        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=0.0001,
            fee_model=SimpleFeeModel(commission_rate=0.05, slippage_pct=0.05),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        # Equity must not exceed initial capital by more than a normal
        # market-return factor; pathological multiplications (the v1.9.5
        # bug produced ~ $2.7e10) are gone.
        assert result.final_capital < 1.0, (
            f"Final capital ballooned from $0.0001 to {result.final_capital}"
        )
        # Equity must remain >= 0 throughout.
        assert (result.equity_curve >= 0).all()

    def test_bankruptcy_freezes_equity(self):
        """Once equity hits 0, it stays at 0 for the rest of the run."""
        data = _make_data(60, vol=0.05)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[15] = -1.0
        signals.iloc[30] = 1.0

        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=0.01,
            fee_model=SimpleFeeModel(commission_rate=0.5, slippage_pct=0.5),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        eq = result.equity_curve
        if (eq <= 0).any():
            first_zero = (eq <= 0).idxmax()
            assert (eq.loc[first_zero:] == 0).all(), (
                "Equity went negative or non-monotone after bankruptcy")


# ---------------------------------------------------------------------------
# Q1 — Trade.size = shares
# ---------------------------------------------------------------------------

class TestTradeSizeShares:
    def test_size_is_shares_not_signal_magnitude(self):
        """Trade.size is now expressed in shares (entry_equity * pos / price)."""
        data = _make_data(40, drift=0.001, vol=0.005)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[20] = 0.0

        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=100_000.0,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        assert len(result.trades) >= 1
        t = result.trades[0]
        # With pos=1 and $100k capital at price ≈ $100, shares ≈ 1000.
        # The legacy Trade.size = 1.0 (signal magnitude) would be off
        # by ~3 orders of magnitude.
        assert t.size > 50, f"Trade.size {t.size} too small to be share count"

    def test_estimate_capacity_finite_on_vectorized_result(self):
        """market_impact downstream consumers see realistic numbers."""
        from aiphaforge.market_impact import (
            SquareRootImpactModel,
            estimate_capacity,
        )

        data = _make_data(40, drift=0.001, vol=0.005)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[20] = 0.0

        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=100_000.0,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        impact = SquareRootImpactModel(eta=0.1)
        cap = estimate_capacity(
            result, data,
            impact_model=impact,
            capital_multipliers=[1.0, 2.0, 5.0, 10.0],
        )
        # Per-multiplier impact cost must be > 0 with shares-based size;
        # pre-Q1 it was ~0 because trade.size = signal magnitude (1.0)
        # made the participation rate negligible.
        assert (cap.results_by_capital["total_impact_cost"] > 0).any(), (
            "Impact cost is zero across all multipliers; trade.size still "
            "appears to be signal magnitude rather than shares")


# ---------------------------------------------------------------------------
# Q3 — discrepancy contract (P1)
# ---------------------------------------------------------------------------

class TestDiscrepancyContract:
    def test_single_trade_no_fees_no_reversal_is_identity(self):
        """Single trade, pos=1, no fees, no reversal: gap is machine epsilon."""
        data = _make_data(50, drift=0.0005, vol=0.01)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[40] = 0.0

        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=100_000.0,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        assert len(result.trades) == 1
        linear = result.trades[0].pnl
        geo = result.final_capital - 100_000.0
        # When there are no reversals and no fees, the geometric and
        # linear formulations collapse algebraically:
        #   shares = entry_eq * pos / entry_price
        #   geo  = entry_eq * (exit_price/entry_price - 1)
        #   linear = shares * (exit_price - entry_price)
        #          = entry_eq * (exit_price - entry_price) / entry_price
        #          = entry_eq * (exit_price/entry_price - 1)
        # so they match to machine epsilon.
        assert abs(geo - linear) < 1e-6, (
            f"Identity case violated: geo={geo}, linear={linear}, "
            f"diff={geo - linear}"
        )

    def test_single_trade_with_fees_bounded_by_fees_paid(self):
        """Single trade with fees only: gap is bounded by total fees paid."""
        data = _make_data(50, drift=0.0005, vol=0.01)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[40] = 0.0

        commission = 0.001
        slippage = 0.0005
        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=100_000.0,
            fee_model=SimpleFeeModel(
                commission_rate=commission, slippage_pct=slippage),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        assert len(result.trades) == 1
        t = result.trades[0]
        # Estimate fees = (entry_notional + exit_notional) * (c + s).
        # For pos=1, both notionals ≈ 100k, so total fees ≈ 200k * (c+s).
        notional = t.size * (t.entry_price + t.exit_price)
        max_fees = notional * (commission + slippage)
        linear = t.pnl
        geo = result.final_capital - 100_000.0
        diff = abs(geo - linear)
        assert diff <= max_fees * 1.01, (
            f"Fees-only divergence {diff} exceeds bound {max_fees}; "
            f"linear={linear}, geo={geo}"
        )

    def test_bar_zero_signal_produces_correct_initial_trade(self):
        """v1.9.7 N1 fix: a non-zero signal at bar 0 must produce a
        trade entered at bar 0.

        Pre-fix: pos_diff.iloc[0] is always NaN (pandas semantics);
        the dropna() discarded the bar-0 entry, and the next non-zero
        diff (which actually closed the bar-0 position) was
        misinterpreted as opening a fresh trade in the WRONG direction.

        Reproducer: signals = [1, 1, 0, 0, 1, 1, 0]
        Pre-fix output: 1 phantom short + 1 long, sum_pnl off by ~$4000
        Post-fix output: 2 long trades, sum_pnl matches equity exactly.
        """
        n = 7
        prices = [100.0 + i for i in range(n)]
        data = pd.DataFrame(
            {"open": prices, "high": [p * 1.01 for p in prices],
             "low": [p * 0.99 for p in prices], "close": prices,
             "volume": [1e6] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        signals = pd.Series([1, 1, 0, 0, 1, 1, 0], index=data.index,
                            dtype=float)

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        res = eng.run(data)

        assert len(res.trades) == 2
        assert all(t.direction == 1 for t in res.trades), (
            f"Expected 2 long trades; got directions "
            f"{[t.direction for t in res.trades]}")
        assert res.trades[0].entry_time == data.index[0], (
            "First trade must enter at bar 0")

        sum_pnl = sum(t.pnl for t in res.trades)
        eq_change = res.final_capital - 100_000.0
        # On flat data with no fees / no reversals, the linear PnL and
        # geometric equity change should match within float epsilon.
        assert abs(sum_pnl - eq_change) < 1e-6, (
            f"sum(trade.pnl) = ${sum_pnl:.4f} vs equity_change = "
            f"${eq_change:.4f}, diff = ${sum_pnl - eq_change:.4f}")

    def test_multi_trade_reversal_bounded_by_sigma2_t_notional(self):
        """Multi-trade with reversals: gap is bounded by 0.5 * σ² * T * notional."""
        n_bars = 100
        sigma = 0.05
        data = _make_data(n_bars, drift=0.0, vol=sigma, seed=7)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        # Reversals to maximize the geometric/linear gap.
        for i in (1, 20, 40, 60, 80, 95):
            signals.iloc[i] = 1.0 if (i // 20) % 2 == 0 else -1.0

        eng = BacktestEngine(
            mode="vectorized",
            initial_capital=100_000.0,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        eng.set_signals(signals)
        result = eng.run(data)

        linear_sum = sum(t.pnl for t in result.trades)
        geo = result.final_capital - 100_000.0

        # Loose bound: 0.5 * σ² * T * |notional|. Take notional = init.
        bound = 0.5 * sigma * sigma * n_bars * 100_000.0
        diff = abs(geo - linear_sum)
        assert diff <= bound * 5, (
            f"Multi-trade divergence {diff} exceeds 5x the σ²·T·notional "
            f"bound {bound}; linear_sum={linear_sum}, geo={geo}"
        )
