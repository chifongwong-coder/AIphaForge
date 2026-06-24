"""v2.9.1 — significance module split into a subpackage (V2.9-S2).

Pins the behaviour-preserving contract of the split: the full pre-split
import surface (public + private), object-identity of the S1-hoisted
re-exports, ``__module__`` held at the package path for pickle/repr
parity, the two cross-family edges, the package shape, and the
optional-dependency (scipy/arch) leaf-import guarantee. The numerical
correctness of the moved code stays pinned by
test_significance_canonical_formulas.py / the tests_internal unit suites,
which import via the preserved paths.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import aiphaforge.significance as S
from aiphaforge import stats

_PKG = Path(S.__file__).resolve().parent

_PUBLIC = [
    "BootstrapResult", "CorrectionResult", "DSRResult", "MonteCarloResult",
    "PSRResult", "PermutationResult", "bootstrap_ci", "bootstrap_metrics",
    "build_returns_matrix", "deflated_sharpe_ratio", "generate_paths",
    "monte_carlo_test", "multiple_comparison_correction", "permutation_test",
    "probabilistic_sharpe_ratio",
]
# Private helpers tests import by name from aiphaforge.significance.
_PRIVATE = [
    "_HIGHER_IS_BETTER", "_LOWER_IS_BETTER", "_make_metric_fn",
    "_resolve_returns_and_td", "_permute_signals", "_normal_paths",
    "_run_backtest_and_extract", "_compute_benchmark_metric", "_bonferroni",
    "_benjamini_hochberg", "_arch_mcs", "_build_returns_matrix_from_cache",
]
_TRIO = ["_stationary_block_bootstrap", "_block_bootstrap_indices",
         "_reconstruct_ohlcv"]
_SUBMODULES = ["_shared", "psr_dsr", "bootstrap", "paths", "monte_carlo",
               "correction"]


class TestReExportCompleteness:
    @pytest.mark.parametrize("name", _PUBLIC + _PRIVATE + _TRIO)
    def test_name_importable_from_package(self, name):
        assert hasattr(S, name), f"{name} not re-exported from significance"

    @pytest.mark.parametrize("name", _TRIO)
    def test_trio_is_identical_to_stats_object(self, name):
        # The S1-hoisted primitives must be the SAME object, not a copy.
        assert getattr(S, name) is getattr(stats, name)

    def test_all_is_the_canonical_15(self):
        assert sorted(S.__all__) == sorted(_PUBLIC)
        assert len(S.__all__) == 15


class TestModulePreservation:
    @pytest.mark.parametrize("name", _PUBLIC)
    def test_module_held_at_package_path(self, name):
        # All 15 public symbols (6 classes + 9 functions) keep
        # __module__ == "aiphaforge.significance" so pickle/repr/help are
        # byte-identical to the pre-split flat module. Fails if the
        # reassertion is dropped or applied to classes only.
        assert getattr(S, name).__module__ == "aiphaforge.significance"

    def test_result_pickle_round_trip_stores_package_path(self):
        import pickle

        import numpy as np
        b = S.BootstrapResult("sharpe_ratio", 1.0, 1.0, 0.1, 0.8, 1.2, 0.95,
                              100, np.zeros(5))
        loaded = pickle.loads(pickle.dumps(b))
        assert loaded.metric_name == "sharpe_ratio"
        assert type(loaded).__module__ == "aiphaforge.significance"


class TestCrossFamilyEdges:
    def test_correction_imports_resolve(self):
        # The two cross-family edges correction depends on must wire up.
        from aiphaforge.significance.correction import (
            _build_returns_matrix_from_cache,  # noqa: F401  (corr -> monte_carlo)
            bootstrap_ci,  # noqa: F401  (corr -> bootstrap)
        )


class TestPackageShape:
    def test_is_a_package(self):
        assert hasattr(S, "__path__")

    def test_submodules_import(self):
        import importlib
        for sub in _SUBMODULES:
            importlib.import_module(f"aiphaforge.significance.{sub}")

    def test_no_submodule_declares_all(self):
        # Load-bearing for the API-lock auto-discovery: only the package
        # __init__ is the locked surface. A submodule with __all__ would
        # become a newly-locked module.
        import importlib
        for sub in _SUBMODULES:
            mod = importlib.import_module(f"aiphaforge.significance.{sub}")
            assert not hasattr(mod, "__all__"), (
                f"significance.{sub} must not declare __all__")


class TestImportDirection:
    def _imports_of(self, submodule):
        tree = ast.parse((_PKG / f"{submodule}.py").read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1:
                mods.add(node.module or "")
        return mods

    def test_shared_imports_nothing_intra_package(self):
        # _shared is the leaf; it must not import a sibling submodule.
        intra = {m for m in self._imports_of("_shared")
                 if m in _SUBMODULES}
        assert intra == set(), f"_shared imports siblings: {intra}"

    def test_correction_has_no_reverse_edge(self):
        # correction may import bootstrap/monte_carlo (forward) but must
        # not pull the monte-carlo *driver* (no cycle risk).
        src = (_PKG / "correction.py").read_text()
        assert "generate_paths" not in src
        assert "monte_carlo_test" not in src


def test_package_imports_without_scipy_or_arch():
    # The package and every submodule must import with scipy AND arch
    # unavailable (all uses are function-local lazy imports); a fresh
    # interpreter proves no lazy import leaked to module level.
    code = (
        "import sys\n"
        "sys.modules['scipy'] = None\n"
        "sys.modules['arch'] = None\n"
        "import importlib\n"
        "importlib.import_module('aiphaforge.significance')\n"
        "for s in ['_shared','psr_dsr','bootstrap','paths','monte_carlo','correction']:\n"
        "    importlib.import_module('aiphaforge.significance.' + s)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_mcs_arch_import_error_survives_move():
    # Pins that the moved _arch_mcs import-guard message survives the file
    # move (the MCS algorithm body needs `arch`, absent in CI). Calls the
    # re-exported helper directly — the arch import lives inside it.
    import numpy as np
    rng = np.random.default_rng(0)
    returns_matrix = rng.standard_normal((30, 3))
    with pytest.raises(ImportError, match="arch"):
        S._arch_mcs(returns_matrix, alpha=0.10, n_bootstrap=50,
                    block_size=None, random_state=42)
