"""
Statistical Significance Testing
=================================

Bootstrap confidence intervals, permutation tests, Monte Carlo path
simulation, and multiple comparison correction for backtest results.

Answers: "Is this strategy's performance real or luck?"

Four tools:

1. **Bootstrap CI**: Confidence intervals for any performance metric via
   stationary block bootstrap (Politis-Romano).
2. **Permutation Test**: p-value for strategy alpha by shuffling signal timing.
3. **Monte Carlo Path Simulation** (v1.6): Generate synthetic market paths,
   run the full strategy/agent on each. Test robustness across different
   possible histories.
4. **Multiple Comparison Correction** (v1.6): Correct optimizer results for
   data snooping when testing many parameter combos.

Example::

    from aiphaforge.significance import bootstrap_ci, permutation_test

    ci = bootstrap_ci(result, metric="sharpe_ratio", confidence=0.95)
    print(f"Sharpe: {ci.observed:.2f} [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]")

    perm = permutation_test(data, strategy=my_strategy, metric="sharpe_ratio")
    print(f"p-value: {perm.p_value:.4f}")

    from aiphaforge.significance import monte_carlo_test, generate_paths

    mc = monte_carlo_test(data, strategy=my_strategy, metric="sharpe_ratio")
    print(f"MC 5th/95th: [{mc.pct_5:.2f}, {mc.pct_95:.2f}]")
"""

# v2.9.1: significance was split from a single 1649-LOC module into this
# subpackage (master-plan V2.9-S2). This __init__ re-exports the entire
# pre-split surface so every existing ``from aiphaforge.significance import X``
# keeps working unchanged — the public API AND the private helpers that tests
# import by name.

# Re-exports below cover both the public surface (tracked via __all__) and the
# private helpers tests import by name (F401-suppressed, kept out of __all__).
# The S1-hoisted block-bootstrap primitives are re-exported from aiphaforge.stats
# for the historical
# ``from aiphaforge.significance import _block_bootstrap_indices`` path.
from ..stats import (  # noqa: F401
    _block_bootstrap_indices,
    _reconstruct_ohlcv,
    _stationary_block_bootstrap,
)
from ._shared import (  # noqa: F401
    _HIGHER_IS_BETTER,
    _LOWER_IS_BETTER,
    _make_metric_fn,
)
from .bootstrap import (
    BootstrapResult,
    PermutationResult,
    _permute_signals,  # noqa: F401
    bootstrap_ci,
    bootstrap_metrics,
    permutation_test,
)
from .correction import (  # noqa: F401
    CorrectionResult,
    _arch_mcs,
    _benjamini_hochberg,
    _bonferroni,
    _compute_benchmark_metric,
    multiple_comparison_correction,
)
from .monte_carlo import (
    MonteCarloResult,
    _build_returns_matrix_from_cache,  # noqa: F401
    build_returns_matrix,
    monte_carlo_test,
)
from .paths import _normal_paths, _run_backtest_and_extract, generate_paths  # noqa: F401
from .psr_dsr import (
    DSRResult,
    PSRResult,
    _resolve_returns_and_td,  # noqa: F401
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "BootstrapResult",
    "CorrectionResult",
    "DSRResult",
    "MonteCarloResult",
    "PSRResult",
    "PermutationResult",
    "bootstrap_ci",
    "bootstrap_metrics",
    "build_returns_matrix",
    "deflated_sharpe_ratio",
    "generate_paths",
    "monte_carlo_test",
    "multiple_comparison_correction",
    "permutation_test",
    "probabilistic_sharpe_ratio",
]

# Hold __module__ at the package path on every public symbol so repr, pickle,
# and help() are byte-identical to the pre-v2.9.1 flat module (pickle resolves
# classes/functions by __module__.__qualname__; the submodule split must stay
# invisible to downstream pickles written under <=2.9.0). Intentional — do NOT
# "simplify" this away; it is the v2.9.1 back-compat contract.
for _obj in (
    BootstrapResult, PermutationResult, MonteCarloResult, CorrectionResult,
    PSRResult, DSRResult, probabilistic_sharpe_ratio, deflated_sharpe_ratio,
    bootstrap_metrics, bootstrap_ci, permutation_test, generate_paths,
    monte_carlo_test, build_returns_matrix, multiple_comparison_correction,
):
    _obj.__module__ = "aiphaforge.significance"
del _obj
