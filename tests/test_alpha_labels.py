"""v2.4 Commit F — alpha.labels.forward_returns tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge.alpha.labels import forward_returns


def _prices(periods: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    b = 200.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, periods)))
    return pd.DataFrame(
        {"A": a, "B": b}, index=pd.bdate_range("2024-01-01", periods=periods),
    )


class TestForwardReturnsBasics:
    def test_default_is_simple(self):
        prices = _prices(periods=10, seed=42)
        out = forward_returns(prices, horizon=2)
        # Hand-check at t=0: close[3] / close[1] - 1.
        expected = prices.iloc[3, 0] / prices.iloc[1, 0] - 1.0
        assert out.iloc[0, 0] == pytest.approx(expected, rel=1e-12)

    def test_simple_matches_hand_computation_5_bar_fixture(self):
        # Pin a deterministic fixture, hand-walk every output.
        idx = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 99.0, 102.0, 98.0]}, index=idx,
        )
        out = forward_returns(prices, horizon=2, entry_lag=1)
        # entry_lag=1, horizon=2: at t, entry=close[t+1], exit=close[t+3]
        # t=0: close[3]/close[1] - 1 = 102/101 - 1 = 0.00990099...
        # t=1: close[4]/close[2] - 1 = 98/99 - 1 = -0.01010101...
        # t=2: NaN (close[5] doesn't exist)
        assert out.iloc[0, 0] == pytest.approx(102/101 - 1, rel=1e-12)
        assert out.iloc[1, 0] == pytest.approx(98/99 - 1, rel=1e-12)
        assert pd.isna(out.iloc[2, 0])
        assert pd.isna(out.iloc[3, 0])

    def test_uses_entry_lag(self):
        prices = _prices(periods=20)
        # entry_lag=2 vs entry_lag=1, horizon=1: result at t with
        # lag=2 should equal result at t+1 with lag=1.
        h1 = forward_returns(prices, horizon=1, entry_lag=1)
        h2 = forward_returns(prices, horizon=1, entry_lag=2)
        # h2.iloc[0] uses entry=close[2], exit=close[3] which is
        # the same as h1.iloc[1] (entry=close[2], exit=close[3]).
        pd.testing.assert_series_equal(
            h2.iloc[0], h1.iloc[1], check_names=False,
        )

    def test_columns_match_prices_columns(self):
        prices = _prices(periods=10)
        out = forward_returns(prices, horizon=2)
        assert list(out.columns) == list(prices.columns)


class TestForwardReturnsLogMode:
    def test_log_mode_compounds_additively_pointwise(self):
        # Strict pointwise identity (per quant-review v1):
        # fr_log[t, h=2] + fr_log[t+2, h=2] == fr_log[t, h=4]
        # for non-overlapping log windows. Telescoping log →
        # this is exact within float epsilon.
        prices = _prices(periods=30, seed=99)
        fr2 = forward_returns(prices, horizon=2, return_type="log")
        fr4 = forward_returns(prices, horizon=4, return_type="log")
        # Test bars where both halves are defined.
        for t in range(0, len(prices) - 6):
            for col in prices.columns:
                lhs = fr2.iloc[t][col] + fr2.iloc[t + 2][col]
                rhs = fr4.iloc[t][col]
                if pd.isna(lhs) or pd.isna(rhs):
                    continue
                assert lhs == pytest.approx(rhs, abs=1e-12), (
                    f"log-additivity broke at t={t}, col={col}: "
                    f"{lhs} vs {rhs}"
                )


class TestForwardReturnsRobustness:
    def test_handles_missing_prices_via_nan(self):
        # NaN in prices propagates to NaN in forward returns —
        # NOT silently zero-filled (R3).
        idx = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.DataFrame(
            {"A": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109]},
            index=idx, dtype=float,
        )
        out = forward_returns(prices, horizon=2, entry_lag=1)
        # At t=0: entry=close[1]=101, exit=close[3]=103 → finite.
        assert not pd.isna(out.iloc[0, 0])
        # At t=1: entry=close[2]=NaN → NaN propagates.
        assert pd.isna(out.iloc[1, 0])

    def test_invalid_return_type_raises(self):
        with pytest.raises(ValueError, match="return_type"):
            forward_returns(_prices(), horizon=1, return_type="bogus")  # type: ignore


class TestAlphaLabelsFirewall:
    def test_alpha_labels_does_not_import_engine(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/alpha/labels.py"
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
                    f"alpha/labels.py imports {name!r}"
                )
