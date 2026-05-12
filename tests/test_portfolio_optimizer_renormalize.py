"""v2.2.2 Commit A regression tests.

Verifies the post-SLSQP clamp+renormalize step keeps the long-only
budget at exactly 1.0 (within float epsilon). Prior behavior was
clamp-only, leaving the budget at 1 + n*ftol where ftol ~ 1e-6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.portfolio_optimizer import (
    MeanVarianceOptimizer,
    MinimumVarianceOptimizer,
)


def _ill_conditioned_returns(n_assets: int = 5, n_obs: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = [f"A{i}" for i in range(n_assets)]
    factor = rng.normal(0.0, 0.01, n_obs)
    rets = np.column_stack(
        [factor + rng.normal(0.0, 1e-5, n_obs) for _ in range(n_assets)]
    )
    return pd.DataFrame(rets, columns=cols)


class TestMeanVarianceRenormalize:
    def test_weights_sum_to_exactly_one_after_clamp(self):
        data = _ill_conditioned_returns()
        opt = MeanVarianceOptimizer(allow_short=False)
        w = opt.compute_weights(data)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-12, (
            f"long-only weight sum must equal 1.0 exactly within float "
            f"epsilon; got {total!r} (diff {total - 1.0:.2e})"
        )


class TestMinimumVarianceRenormalize:
    def test_weights_sum_to_exactly_one_after_clamp(self):
        data = _ill_conditioned_returns()
        opt = MinimumVarianceOptimizer(allow_short=False)
        w = opt.compute_weights(data)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-12, (
            f"long-only MinVar weight sum must equal 1.0 exactly within "
            f"float epsilon; got {total!r} (diff {total - 1.0:.2e})"
        )


class TestOptimizerAllZeroClampPathology:
    def test_renorm_does_not_divide_by_zero(self, monkeypatch):
        # If SLSQP somehow returned all-negative weights, the long-only
        # clamp would zero everything and the renormalize step would
        # face total == 0. The guard must fall through with all-zero
        # weights (visible failure) rather than producing NaN.
        # scipy.optimize.minimize is imported inside compute_weights;
        # patch it on the scipy module directly so the local import
        # picks up the fake.
        import scipy.optimize as sp_opt
        from scipy.optimize import OptimizeResult

        original_minimize = sp_opt.minimize

        def fake_minimize(*args, **kwargs):
            real = original_minimize(*args, **kwargs)
            return OptimizeResult(
                x=-np.ones_like(real.x) * 1e-9,
                success=True,
                message="patched all-negatives",
                fun=real.fun,
                jac=real.jac if hasattr(real, "jac") else None,
                nit=real.nit if hasattr(real, "nit") else 0,
            )

        monkeypatch.setattr(sp_opt, "minimize", fake_minimize)
        data = _ill_conditioned_returns()
        opt = MeanVarianceOptimizer(allow_short=False)
        w = opt.compute_weights(data)
        # All weights should be 0 (no NaN, no ZeroDivisionError).
        for sym, val in w.items():
            assert val == 0.0, f"{sym}: expected 0.0 with all-neg input, got {val!r}"


@pytest.mark.parametrize("opt_cls", [MeanVarianceOptimizer, MinimumVarianceOptimizer])
class TestRenormGatedByAllowShort:
    def test_allow_short_path_skips_clamp_and_renorm(self, opt_cls):
        # The clamp+renorm only fires when allow_short=False. With
        # allow_short=True, SLSQP returns whatever it returns (may
        # legitimately have negative weights), no post-processing.
        data = _ill_conditioned_returns()
        opt = opt_cls(allow_short=True)
        w = opt.compute_weights(data)
        total = sum(w.values())
        # Budget is still ~1.0 but only within SLSQP's ftol since the
        # renormalize path is bypassed.
        assert abs(total - 1.0) < 1e-4
