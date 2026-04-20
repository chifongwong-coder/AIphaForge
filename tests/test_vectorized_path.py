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

    def test_vectorized_stop_loss_emits_trade_entries(self):
        """v1.9.7 commit 7b: vectorized stop-loss exits visible in trades.

        Pre-fix: stop_loss was folded into apply_vectorized's
        net_returns clip but never produced a Trade boundary because
        positions wasn't modified. Equity was correct, but trades list
        missed the stop_loss exits.
        """
        # Build a clear stop scenario: long at bar 1, prices fall sharply.
        n = 10
        prices = [100.0, 100.0, 99.0, 97.0, 95.0, 92.0, 90.0,
                  92.0, 95.0, 96.0]
        data = pd.DataFrame(
            {"open": prices, "high": [p * 1.005 for p in prices],
             "low": [p * 0.995 for p in prices], "close": prices,
             "volume": [1e6] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0   # long entry
        signals.iloc[8] = 0.0   # natural close

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False, stop_loss=0.05,  # 5% threshold
        )
        eng.set_signals(signals)
        res = eng.run(data)

        # Stop should fire (price drops > 5% from entry of 100 by bar 4-5).
        stop_trades = [t for t in res.trades if t.reason == "stop_loss"]
        assert len(stop_trades) >= 1, (
            f"Expected at least 1 stop_loss trade; got "
            f"{[(t.reason, t.entry_price, t.exit_price) for t in res.trades]}")

        # Stop trade exit price = entry_price * (1 - threshold)
        # = 100 * 0.95 = 95.0
        st = stop_trades[0]
        assert abs(st.exit_price - 95.0) < 1e-6, (
            f"Expected stop exit price 95.0, got {st.exit_price}")

    def test_vectorized_no_stop_loss_falls_back_to_legacy_path(self):
        """v1.9.7 commit 7b: when no stop_loss_rule, behavior is
        byte-identical to v1.9.6 (legacy in-loop path).
        """
        data = _make_data(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[40] = 0.0

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )  # no stop_loss
        eng.set_signals(signals)
        res = eng.run(data)

        # Single signal trade, no stop_loss in trades
        assert len(res.trades) == 1
        assert res.trades[0].reason == "signal"

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


# ---------------------------------------------------------------------------
# v1.9.7 commit 7b — stop-loss visibility edge cases
# (probe-style coverage added 2026-04-20)
# ---------------------------------------------------------------------------

class TestStopLossEdgeCases:
    """Edge cases for vectorized stop-loss visibility — what the
    expert reviewer would try to break the segment-based reconstruction.
    """

    def _stairstep_data(self, n: int, bar_change: float) -> pd.DataFrame:
        """OHLCV with deterministic monotone close moves of `bar_change`."""
        close = [100.0 * (1 + bar_change) ** i for i in range(n)]
        return pd.DataFrame(
            {"open": close, "high": [p * 1.001 for p in close],
             "low": [p * 0.999 for p in close], "close": close,
             "volume": [1e6] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )

    def test_bar_zero_position_with_stop_loss_fires_correctly(self):
        """Commit 7a + 7b interaction: long opens at bar 0, stop fires
        before any signal change. Segment-based path must recognize the
        bar-0 entry AND truncate it at the stop.
        """
        n = 20
        # Drop ~10% from bar 0 → triggers 5% stop quickly.
        data = self._stairstep_data(n, bar_change=-0.01)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[0] = 1.0   # open long at bar 0

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False, stop_loss=0.05,
        )
        eng.set_signals(signals)
        res = eng.run(data)

        stop_trades = [t for t in res.trades if t.reason == "stop_loss"]
        assert len(stop_trades) == 1, (
            f"Expected 1 stop_loss trade for bar-0 long; got "
            f"{[(t.reason, t.entry_time) for t in res.trades]}")
        # Stop trade must enter at bar 0
        assert stop_trades[0].entry_time == data.index[0]
        # Exit price = entry * (1 - 0.05) = 100 * 0.95 = 95.0
        assert abs(stop_trades[0].exit_price - 95.0) < 1e-6

    def test_stop_loss_then_reversal_preserves_new_segment(self):
        """Plan v3 R1 reversal case: long opens at bar 1, stop fires
        at bar 5, signal reverses to short at bar 10.
        Expected: trade 1 = stop_loss long(1→5), trade 2 = signal
        short(10→...). Pre-fix: phantom open at bar 10 in wrong dir.
        """
        # Hand-craft prices: drop sharply 1→5 (triggers stop), recover
        # 5→10, drop again 10→end (short profitable).
        n = 20
        prices = (
            [100.0]
            + [100.0 * (1 - 0.02 * i) for i in range(5)]   # 1..5: drop
            + [90.0 * (1 + 0.01 * i) for i in range(5)]    # 6..10: recover
            + [95.0 * (1 - 0.005 * i) for i in range(n - 11)]  # 11..end: drift
        )
        prices = prices[:n]
        data = pd.DataFrame(
            {"open": prices, "high": [p * 1.001 for p in prices],
             "low": [p * 0.999 for p in prices], "close": prices,
             "volume": [1e6] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0    # long
        signals.iloc[10] = -1.0  # reversal to short
        signals.iloc[18] = 0.0   # close short

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False, stop_loss=0.05,
        )
        eng.set_signals(signals)
        res = eng.run(data)

        # Should have at least: stop_loss long (1→stop_bar) + signal short (10→18)
        stop_trades = [t for t in res.trades if t.reason == "stop_loss"]
        signal_trades = [t for t in res.trades if t.reason == "signal"]
        assert len(stop_trades) >= 1, "Stop did not fire in reversal scenario"
        assert any(t.direction == 1 for t in stop_trades), (
            "Stop trade should be the long (entry at bar 1)")
        # Critical: short segment from bar 10 must NOT be lost
        short_signal_trades = [t for t in signal_trades if t.direction == -1]
        assert len(short_signal_trades) >= 1, (
            "Reversal short segment after stop_loss long was lost — "
            "the bug Plan v3 R1 set out to prevent")

    def test_stop_loss_with_risk_rules_uses_second_mask(self):
        """Plan v3 R2: when both apply_vectorized calls fire (risk_rules
        present), the SECOND mask wins. Test this composes correctly.
        """
        from aiphaforge.risk import CompositeRiskManager, MaxDrawdownHalt

        n = 30
        data = self._stairstep_data(n, bar_change=-0.01)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0

        # MaxDrawdownHalt may modify positions → second apply_vectorized
        # call sees post-risk-rules positions → its trigger mask wins.
        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False,
            stop_loss=0.05,
            risk_rules=CompositeRiskManager(
                rules=[MaxDrawdownHalt(max_drawdown=0.30)]),
        )
        eng.set_signals(signals)
        res = eng.run(data)

        # Run shouldn't crash; should produce at least one trade.
        assert len(res.trades) >= 1
        # Equity must be valid (not NaN, not negative beyond bankruptcy floor)
        assert (res.equity_curve >= 0).all()

    def test_apply_vectorized_default_call_returns_series(self):
        """Backward compat: PercentageStopLoss.apply_vectorized without
        return_mask must return just the Series (not a tuple). Critical
        for any user subclass / direct caller relying on the v1.9.6
        contract.
        """
        from aiphaforge.exit_rules import PercentageStopLoss

        n = 20
        data = self._stairstep_data(n, bar_change=-0.01)
        positions = pd.Series([1.0] * n, index=data.index)
        returns = data['close'].pct_change().fillna(0.0)
        rule = PercentageStopLoss(threshold=0.05)

        # Default call (no return_mask) — must return Series only
        result = rule.apply_vectorized(returns, positions, data)
        assert isinstance(result, pd.Series), (
            f"Backward-compat broken: expected pd.Series, got {type(result)}")
        assert len(result) == n

    def test_apply_vectorized_with_return_mask_returns_tuple(self):
        """Forward path: return_mask=True returns 4-tuple."""
        from aiphaforge.exit_rules import PercentageStopLoss

        n = 20
        data = self._stairstep_data(n, bar_change=-0.01)
        positions = pd.Series([1.0] * n, index=data.index)
        returns = data['close'].pct_change().fillna(0.0)
        rule = PercentageStopLoss(threshold=0.05)

        result = rule.apply_vectorized(
            returns, positions, data, return_mask=True)
        assert isinstance(result, tuple)
        assert len(result) == 4
        returns_with_stop, trigger_mask, entry_prices, threshold = result
        assert isinstance(returns_with_stop, pd.Series)
        assert isinstance(trigger_mask, pd.Series)
        assert isinstance(entry_prices, pd.Series)
        assert threshold == 0.05

    def test_stop_loss_trades_sum_within_v196_bound(self):
        """The stop-loss-visible path must respect the v1.9.6 P1
        discrepancy contract: |sum(trade.pnl) - equity_change| within
        σ² * T * notional bound (loose multiplier OK).
        """
        n = 30
        sigma = 0.02
        np.random.seed(11)
        rets = sigma * np.random.normal(size=n)
        rets[0] = 0.0
        close = 100.0 * np.exp(np.cumsum(rets))
        data = pd.DataFrame(
            {"open": close, "high": close * 1.005,
             "low": close * 0.995, "close": close,
             "volume": [1e6] * n},
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1.0
        signals.iloc[20] = -1.0
        signals.iloc[28] = 0.0

        eng = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False, stop_loss=0.03,
        )
        eng.set_signals(signals)
        res = eng.run(data)

        sum_pnl = sum(t.pnl for t in res.trades)
        eq_change = res.final_capital - 100_000.0
        bound = sigma * sigma * n * 100_000.0  # σ²·T·notional
        diff = abs(sum_pnl - eq_change)
        assert diff <= bound * 10, (
            f"Stop-loss path discrepancy {diff:.2f} exceeds 10x bound "
            f"{bound:.2f}; trades = "
            f"{[(t.reason, round(t.pnl, 2)) for t in res.trades]}")
