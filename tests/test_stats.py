"""v2.9.0 — aiphaforge.stats shared-primitives hoist.

Covers the new neutral module, the back-compat re-exports from the
former homes, the dissolved calendars -> probes reverse dependency,
and basic functional correctness of the hoisted primitives. Deep
formula correctness stays pinned by test_rank_canonical_formulas.py
and test_significance_canonical_formulas.py (which now exercise the
re-exported objects).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge import stats

_SRC = Path(__file__).resolve().parent.parent / "src" / "aiphaforge"


class TestStatsSurface:
    def test_all_is_the_public_trio(self):
        assert set(stats.__all__) == {
            "IntegrityCheckResult",
            "midranks",
            "tie_corrected_spearman",
        }

    def test_all_symbols_resolve(self):
        for name in stats.__all__:
            assert hasattr(stats, name)

    def test_private_bootstrap_primitives_importable(self):
        # Private (not in __all__) but importable for internal reuse.
        from aiphaforge.stats import (  # noqa: F401
            _block_bootstrap_indices,
            _reconstruct_ohlcv,
            _stationary_block_bootstrap,
        )


class TestBackCompatReExports:
    def test_rank_primitives_same_object_everywhere(self):
        from aiphaforge import probes
        from aiphaforge.alpha import midranks as alpha_midranks
        from aiphaforge.alpha import (
            tie_corrected_spearman as alpha_tie,
        )
        from aiphaforge.probes import _rank

        assert stats.midranks is _rank.midranks
        assert stats.midranks is probes.midranks
        assert stats.midranks is alpha_midranks
        assert stats.tie_corrected_spearman is _rank.tie_corrected_spearman
        assert stats.tie_corrected_spearman is probes.tie_corrected_spearman
        assert stats.tie_corrected_spearman is alpha_tie

    def test_integrity_result_same_object_everywhere(self):
        from aiphaforge.calendars.core import (
            IntegrityCheckResult as cal_result,
        )
        from aiphaforge.probes.transforms import (
            IntegrityCheckResult as tr_result,
        )

        assert stats.IntegrityCheckResult is tr_result
        assert stats.IntegrityCheckResult is cal_result

    def test_bootstrap_primitives_same_object_everywhere(self):
        from aiphaforge import significance
        from aiphaforge.probes import transforms

        assert (
            stats._block_bootstrap_indices
            is significance._block_bootstrap_indices
            is transforms._block_bootstrap_indices
        )
        assert (
            stats._reconstruct_ohlcv
            is significance._reconstruct_ohlcv
            is transforms._reconstruct_ohlcv
        )


class TestDeletedShim:
    def test_alpha_rank_stats_shim_is_gone(self):
        # v2.9.0 deleted the alpha.rank_stats shim — the one documented
        # breaking change. Pin it so a future accidental re-creation of
        # the shim is caught. (aiphaforge.alpha.midranks /
        # .tie_corrected_spearman remain available; only the submodule
        # path is removed.)
        with pytest.raises(ModuleNotFoundError):
            import aiphaforge.alpha.rank_stats  # noqa: F401


class TestPublicAllAdditions:
    def test_midranks_now_in_probes_all(self):
        # v2.9.0 fixes the latent gap: midranks was missing from
        # probes.__all__ (only tie_corrected_spearman was listed).
        from aiphaforge import probes

        assert "midranks" in probes.__all__
        assert "tie_corrected_spearman" in probes.__all__


class TestReverseDependencyDissolved:
    def test_calendars_core_does_not_import_probes(self):
        # The whole point of the hoist: calendars no longer reaches
        # into the probes namespace for IntegrityCheckResult.
        tree = ast.parse((_SRC / "calendars" / "core.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    "aiphaforge.probes"
                ), f"calendars/core.py still imports {node.module!r}"

    def test_stats_is_a_leaf_no_aiphaforge_imports(self):
        # stats must import nothing from aiphaforge so it is safe to
        # import from anywhere without a cycle.
        tree = ast.parse((_SRC / "stats.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("aiphaforge")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("aiphaforge")


class TestFunctional:
    def test_midranks_average_ties(self):
        # 100 and 100 tie for ranks 2,3 -> both get 2.5.
        ranks = stats.midranks([10.0, 100.0, 100.0, 5.0])
        np.testing.assert_allclose(ranks, [2.0, 3.5, 3.5, 1.0])

    def test_tie_corrected_spearman_perfect_negative(self):
        x = stats.midranks([1.0, 2.0, 3.0, 4.0, 5.0])
        y = stats.midranks([5.0, 4.0, 3.0, 2.0, 1.0])
        assert stats.tie_corrected_spearman(x, y) == -1.0

    def test_reconstruct_ohlcv_preserves_index_and_columns(self):
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        base = np.linspace(100.0, 120.0, n)
        data = pd.DataFrame(
            {
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base,
                "volume": np.full(n, 1000.0),
            },
            index=idx,
        )
        rng = np.random.default_rng(7)
        indices = stats._block_bootstrap_indices(n, block_size=5, rng=rng)
        out = stats._reconstruct_ohlcv(data, indices)
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]
        assert out.index.equals(data.index)
        # Prices stay positive (epsilon guard) and OHLC ordering is
        # preserved on every reconstructed bar.
        assert (out["close"] > 0).all()
        assert (out["high"] >= out[["open", "close", "low"]].max(axis=1)).all()
        assert (out["low"] <= out[["open", "close", "high"]].min(axis=1)).all()

    def test_block_bootstrap_indices_in_range(self):
        rng = np.random.default_rng(3)
        idx = stats._block_bootstrap_indices(50, block_size=8, rng=rng)
        assert idx.shape == (50,)
        assert idx.dtype.kind == "i"
        assert idx.min() >= 0 and idx.max() < 50

    def test_block_bootstrap_indices_have_block_structure(self):
        # The whole point of the block bootstrap (vs uniform iid
        # sampling) is that resampled indices come in contiguous runs:
        # at least one adjacent pair must be sequential (idx[i+1] ==
        # idx[i] + 1, modulo circular wrap). A regression that swapped
        # the moved body for uniform sampling would fail this.
        rng = np.random.default_rng(11)
        n = 100
        idx = stats._block_bootstrap_indices(n, block_size=20, rng=rng)
        steps = (idx[1:] - idx[:-1]) % n
        assert (steps == 1).any(), "no contiguous block run found"
