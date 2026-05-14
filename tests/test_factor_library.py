"""v2.4 Commit D — factor library tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from aiphaforge.diagnostics import assert_factor_no_lookahead
from aiphaforge.factor_library import (
    MASpreadFactor,
    MomentumFactor,
    RSIFactor,
    VolumeZScoreFactor,
    VWAPDistanceFactor,
)


def _ohlcv(periods: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": rng.uniform(1e5, 2e6, periods),
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


# ---------------------------------------------------------------------------
# No-lookahead via v2.3 diagnostics
# ---------------------------------------------------------------------------


class TestFactorLibraryNoLookahead:
    def test_rsi_no_lookahead(self):
        assert_factor_no_lookahead(RSIFactor(period=14).compute, _ohlcv())

    def test_momentum_no_lookahead(self):
        assert_factor_no_lookahead(MomentumFactor(window=20).compute, _ohlcv())

    def test_ma_spread_no_lookahead(self):
        assert_factor_no_lookahead(
            MASpreadFactor(short=5, long=20).compute, _ohlcv(),
        )

    def test_vwap_distance_no_lookahead(self):
        assert_factor_no_lookahead(
            VWAPDistanceFactor(window=20).compute, _ohlcv(),
        )

    def test_volume_zscore_no_lookahead(self):
        assert_factor_no_lookahead(
            VolumeZScoreFactor(window=20).compute, _ohlcv(),
        )


# ---------------------------------------------------------------------------
# Sanity bounds + name format pinning
# ---------------------------------------------------------------------------


class TestFactorSanity:
    def test_rsi_within_zero_to_hundred(self):
        out = RSIFactor(period=14).compute(_ohlcv())
        non_nan = out.dropna()
        assert (non_nan >= 0).all().all()
        assert (non_nan <= 100).all().all()

    def test_momentum_sign_matches_underlying_return_sign(self):
        # Strictly monotone-up close series → momentum > 0 every bar
        # after warmup.
        idx = pd.bdate_range("2024-01-01", periods=30)
        closes = pd.Series(np.linspace(100, 200, 30), index=idx)
        df = pd.DataFrame(
            {
                "open": closes, "high": closes * 1.01,
                "low": closes * 0.99, "close": closes,
                "volume": [1e6] * 30,
            },
            index=idx,
        )
        out = MomentumFactor(window=10).compute(df)
        non_nan = out.dropna().iloc[:, 0]
        assert (non_nan > 0).all()

    def test_vwap_distance_uses_rolling_not_running_vwap(self):
        # On a price + volume fixture where running VWAP and
        # rolling VWAP must diverge: build an early high-volume
        # block at price 100, then a sustained price drop to 90
        # with low volume. Running VWAP stays anchored near 100;
        # rolling VWAP follows the recent block toward 90.
        n = 40
        idx = pd.bdate_range("2024-01-01", periods=n)
        closes = np.concatenate([[100] * 10, np.linspace(99, 90, 30)])
        volumes = np.concatenate([[1e8] * 10, [1e3] * 30])  # huge early
        df = pd.DataFrame(
            {
                "open": closes, "high": closes * 1.001,
                "low": closes * 0.999, "close": closes,
                "volume": volumes,
            },
            index=idx,
        )
        # Rolling VWAP at end (window=20): only sees the low-volume
        # tail, so vwap ≈ recent average price ~94, distance small.
        rolling = VWAPDistanceFactor(window=20).compute(df).iloc[-1, 0]
        # If we were using running cumulative VWAP, the huge early
        # block would dominate sum_pv/sum_v at the end → distance
        # large NEGATIVE (close ~90 vs running vwap near 100).
        assert abs(rolling) < 0.1, (
            f"rolling-VWAP distance at end should be small (~recent "
            f"average), got {rolling:.4f} — looks like running-cumulative "
            f"VWAP semantic accidentally got used"
        )

    def test_factor_name_format_pinned(self):
        # User-facing API contract — these column names appear in
        # FactorSet keys and downstream Alphalens-style reports.
        assert RSIFactor(period=14).name == "rsi_14"
        assert MomentumFactor(window=20).name == "momentum_20"
        assert MASpreadFactor(short=5, long=20).name == "ma_spread_5_20_sma"
        assert VWAPDistanceFactor(window=20).name == "vwap_distance_20"
        assert VolumeZScoreFactor(window=20).name == "volume_zscore_20"

    def test_vwap_distance_zero_volume_window_returns_nan(self):
        # All-zero-volume window → 0/0 → NaN naturally via numpy/pandas.
        idx = pd.bdate_range("2024-01-01", periods=30)
        closes = pd.Series([100.0] * 30, index=idx)
        # First 10 bars zero volume, then real volume.
        volumes = np.concatenate([[0.0] * 10, [1e6] * 20])
        df = pd.DataFrame(
            {
                "open": closes, "high": closes * 1.01,
                "low": closes * 0.99, "close": closes,
                "volume": volumes,
            },
            index=idx,
        )
        out = VWAPDistanceFactor(window=5).compute(df)
        # Bar 9 (last bar of zero-volume block, after warmup): window=5
        # covers bars 5-9, all zero volume → NaN.
        assert pd.isna(out.iloc[9, 0])


# ---------------------------------------------------------------------------
# Architectural firewall (R7)
# ---------------------------------------------------------------------------


class TestFactorLibraryFirewall:
    def test_factor_library_does_not_import_engine(self):
        # AST guard per master plan v1.0 §1.2: factor_library.py
        # must NOT statically import aiphaforge.engine. Research-
        # only library; no execution-layer dependencies.
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/factor_library.py"
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
                    f"factor_library.py statically imports {name!r}, "
                    f"breaking the factor-library → engine firewall"
                )
