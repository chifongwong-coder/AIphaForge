"""v2.4 Commit H — AlphaScreener tests.

Includes the strict R11 firewall test asserting the evaluator
does not import any execution-layer module.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge.alpha.evaluator import AlphaScreener
from aiphaforge.alpha.report import AlphaScreenConfig, FactorReport
from aiphaforge.factors import FactorSet, FactorSpec


def _prices(periods: int = 60, n_symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(n_symbols)]
    out = pd.DataFrame(
        {c: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
         for c in cols},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )
    return out


class TestAlphaScreenerSingleDataFrame:
    def test_evaluate_single_factor(self):
        prices = _prices(periods=40)
        # Constructed factor: forward 5-day return shifted back —
        # cheating, but gives non-zero IC at horizon=5.
        factor = prices.pct_change(5).shift(-5)
        screener = AlphaScreener()
        report = screener.evaluate(factor, prices)
        assert isinstance(report, FactorReport)
        assert report.factor_name == "factor"
        assert "ic_mean" in report.metrics
        assert "coverage_mean" in report.metrics
        assert 1 in report.ic
        assert 5 in report.ic

    def test_handles_empty_factor_gracefully(self):
        prices = _prices(periods=20)
        empty = pd.DataFrame(
            np.nan, index=prices.index, columns=prices.columns,
            dtype=float,
        )
        report = AlphaScreener().evaluate(empty, prices)
        # All-NaN metrics, but the call doesn't raise.
        assert pd.isna(report.metrics["ic_mean"])
        assert report.metrics["coverage_mean"] == 0.0

    def test_multiple_horizons(self):
        prices = _prices(periods=50)
        factor = prices.pct_change(3).shift(-3)
        report = AlphaScreener().evaluate(
            factor, prices,
            config=AlphaScreenConfig(horizons=(1, 3, 7)),
        )
        assert set(report.ic.keys()) == {1, 3, 7}
        assert set(report.rank_ic.keys()) == {1, 3, 7}
        assert set(report.quantile_returns.keys()) == {1, 3, 7}

    def test_spec_direction_flips_ic_sign(self):
        prices = _prices(periods=40, seed=99)
        # Build a factor with KNOWN negative correlation to forward returns.
        factor = -prices.pct_change(5).shift(-5)  # inverted forward return
        # Without spec: IC mean is negative.
        rep_unsigned = AlphaScreener().evaluate(factor, prices)
        ic_unsigned = rep_unsigned.metrics["ic_mean"]
        rank_ic_unsigned = rep_unsigned.metrics["rank_ic_mean"]
        assert ic_unsigned < 0
        # With spec.direction=-1: sign flips, IC reported positive.
        spec = FactorSpec(name="negfac", direction=-1)
        rep_signed = AlphaScreener().evaluate(factor, prices, spec=spec)
        ic_signed = rep_signed.metrics["ic_mean"]
        rank_ic_signed = rep_signed.metrics["rank_ic_mean"]
        assert ic_signed == pytest.approx(-ic_unsigned, rel=1e-12)
        # Per quant-review v2 final-pass nit: the spec.direction
        # sign-flip applies to BOTH IC and RankIC. The production
        # code at evaluator.py:86-87 multiplies both metrics by
        # `sign`; pin the RankIC half explicitly to catch a
        # regression that drops the flip on one of the two.
        assert rank_ic_signed == pytest.approx(
            -rank_ic_unsigned, rel=1e-12,
        )


class TestAlphaScreenerFactorSetDispatch:
    def test_evaluate_factor_set_returns_mapping(self):
        prices = _prices(periods=40)
        factor_a = prices.pct_change(2).shift(-2)
        factor_b = -prices.pct_change(3).shift(-3)
        fs = FactorSet(
            values={"momentum_2": factor_a, "rev_3": factor_b},
            specs={
                "momentum_2": FactorSpec(name="momentum_2", direction=1),
                "rev_3": FactorSpec(name="rev_3", direction=-1),
            },
        )
        out = AlphaScreener().evaluate(fs, prices)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"momentum_2", "rev_3"}
        for name, rep in out.items():
            assert rep.factor_name == name


class TestAlphaScreenerQuantileReturns:
    def test_quantile_returns_match_alphalens_convention(self):
        # Construct a fixture where the per-quantile per-period
        # mean forward return is exactly hand-computable.
        idx = pd.bdate_range("2024-01-01", periods=3)
        cols = ["A", "B", "C", "D", "E"]
        # Same factor every bar so the binning is the same.
        # Factor scores: A=10, B=20, C=30, D=40, E=50.
        # With q=5 (5 quantiles, 5 names, no ties): each name
        # lands in its own quantile bucket 0-4.
        factor = pd.DataFrame(
            {c: [v] * 3 for c, v in zip(cols, [10, 20, 30, 40, 50])},
            index=idx,
        )
        # Prices that produce known forward returns.
        prices = pd.DataFrame(
            {c: [100.0, 100.0, 100.0] for c in cols}, index=idx,
        )
        # Set bar 1 prices so forward 1-day return = bucket_id*0.01.
        prices.iloc[1, 0] = 100.0  # A: returns 0% (bucket 0)
        prices.iloc[1, 1] = 100.0  # B: returns 0% from prior bar (bucket 1)
        # Actually simpler: make bar 2 specific values to control
        # forward_return at bar 0 (entry_lag=1 default, horizon=1).
        # fr[0, c] = prices[2, c] / prices[1, c] - 1.
        prices.iloc[1, :] = 100.0  # entry prices all the same
        # Per-bucket exit prices to give 1%, 2%, 3%, 4%, 5%.
        prices.iloc[2, 0] = 101.0  # A bucket 0: +1%
        prices.iloc[2, 1] = 102.0  # B bucket 1: +2%
        prices.iloc[2, 2] = 103.0
        prices.iloc[2, 3] = 104.0
        prices.iloc[2, 4] = 105.0

        report = AlphaScreener().evaluate(
            factor, prices,
            config=AlphaScreenConfig(horizons=(1,), quantiles=5),
        )
        # quantile_returns[1] is a DataFrame keyed by quantile_id.
        # At t=0, the per-quantile means equal the per-bucket fwd return:
        # quantile 0 → A → 0.01, quantile 1 → B → 0.02, etc.
        q1 = report.quantile_returns[1]
        assert q1.iloc[0, 0] == pytest.approx(0.01, rel=1e-6)
        assert q1.iloc[0, 1] == pytest.approx(0.02, rel=1e-6)
        assert q1.iloc[0, 2] == pytest.approx(0.03, rel=1e-6)
        assert q1.iloc[0, 3] == pytest.approx(0.04, rel=1e-6)
        assert q1.iloc[0, 4] == pytest.approx(0.05, rel=1e-6)


class TestAlphaScreenerColumnAlignment:
    def test_column_misalignment_error_distinguishes_set_vs_order(self):
        prices = _prices(periods=10)
        # Same set, wrong order → reindex hint in error message.
        scrambled = prices[list(reversed(prices.columns))]
        with pytest.raises(ValueError, match="reindex"):
            AlphaScreener().evaluate(scrambled, prices)
        # Different sets → "do not match" message names columns.
        prices_other = prices.copy()
        prices_other.columns = list(prices.columns[:-1]) + ["BOGUS"]
        with pytest.raises(ValueError, match="do not match"):
            AlphaScreener().evaluate(prices, prices_other)


class TestAlphaScreenerR11Firewall:
    def test_alpha_screener_no_engine_or_execution_imports(self):
        # R11 (master plan v1.0 §7, extended): the entire execution
        # layer is forbidden inside the alpha namespace.
        forbidden = {
            "aiphaforge.engine",
            "aiphaforge.fees",
            "aiphaforge.broker",
            "aiphaforge.market_impact",
            "aiphaforge.position_sizing",
            "aiphaforge.risk",
            "aiphaforge.portfolio",
            "aiphaforge.optimizer",
            "aiphaforge.portfolio_optimizer",
            "aiphaforge.capital_allocator",
            "aiphaforge.margin",
            "aiphaforge.costs",
            "aiphaforge.latency",
        }
        for module_path in [
            "src/aiphaforge/alpha/evaluator.py",
            "src/aiphaforge/alpha/report.py",
            "src/aiphaforge/alpha/labels.py",
            "src/aiphaforge/alpha/metrics.py",
            "src/aiphaforge/alpha/rank_stats.py",
            "src/aiphaforge/alpha/__init__.py",
        ]:
            path = Path(__file__).resolve().parent.parent / module_path
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    for forb in forbidden:
                        assert forb not in name, (
                            f"{module_path} imports {name!r} — breaks "
                            f"R11 alpha firewall (forbidden: {forb})"
                        )
