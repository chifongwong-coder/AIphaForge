"""
Portfolio Optimization Algorithms
=================================

Pluggable portfolio optimizers that compute target weights from
historical return data. Can be used standalone or via
OptimizedRebalanceHook for periodic rebalancing.

Built-in optimizers:
- EqualWeightOptimizer: 1/N allocation (no dependencies)
- InverseVolatilityOptimizer: Inverse-volatility weighting (numpy only)
- MeanVarianceOptimizer: Markowitz mean-variance (scipy required)
- RiskParityOptimizer: Equal risk contribution (scipy required)
- MinimumVarianceOptimizer: Global minimum variance (scipy required)

Naming clarity:
    EqualWeightOptimizer vs EqualWeightAllocator (capital_allocator.py):
    - Optimizer: takes return data, outputs target weights (sum to 1).
    - Allocator: takes signals + cash, outputs per-signal budget (cash).
    Different layers, different purposes.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


class BasePortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers.

    Subclasses must implement ``compute_weights`` which takes a
    DataFrame of historical returns and returns a dict mapping
    asset names to target weights.
    """

    @abstractmethod
    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Optional[Dict[str, float]]:
        """Compute target weights from historical returns.

        Parameters:
            returns: DataFrame of asset returns (T x N). Columns are
                asset names, rows are time periods.

        Returns:
            Dict mapping symbol to weight (sum ~ 1.0), or None if
            the optimizer fails and on_failure="none".
        """
        ...


class EqualWeightOptimizer(BasePortfolioOptimizer):
    """Equal-weight (1/N) allocation.

    The simplest optimizer: each asset receives 1/N weight.
    No external dependencies required.
    """

    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Dict[str, float]:
        cols = list(returns.columns)
        if not cols:
            return {}
        w = 1.0 / len(cols)
        return {sym: w for sym in cols}


class InverseVolatilityOptimizer(BasePortfolioOptimizer):
    """Inverse-volatility weighting.

    Assets with lower volatility receive higher weights, proportional
    to 1/sigma. Falls back to equal weight when all volatilities are
    zero.

    Parameters:
        lookback: Number of trailing periods to use for volatility
            estimation. If None, uses all available data.
    """

    def __init__(self, lookback: Optional[int] = None) -> None:
        self.lookback = lookback

    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Dict[str, float]:
        cols = list(returns.columns)
        if not cols:
            return {}

        data = returns if self.lookback is None else returns.iloc[-self.lookback:]
        vols = data.std().values

        if np.all(vols == 0) or np.all(np.isnan(vols)):
            # Zero-vol fallback: equal weight
            w = 1.0 / len(cols)
            return {sym: w for sym in cols}

        # Replace zero/nan vols with a large value so they get near-zero weight
        safe_vols = np.where((vols == 0) | np.isnan(vols), np.inf, vols)
        inv_vols = 1.0 / safe_vols
        weights = inv_vols / inv_vols.sum()
        return dict(zip(cols, weights))


def _postprocess_weights(
    raw_weights: np.ndarray, columns: object,
    bounds: list, allow_short: bool = False,
) -> Dict[str, float]:
    """Clip to bounds and normalize to sum=1 after scipy solve."""
    w = raw_weights.copy()
    # Clip to bounds
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            w[i] = max(w[i], lo)
        if hi is not None:
            w[i] = min(w[i], hi)
    # Normalize to sum=1
    total = w.sum()
    if total > 0:
        w = w / total
    elif not allow_short:
        n = len(w)
        w = np.ones(n) / n if n > 0 else w
    return dict(zip(columns, w))


def _check_min_rows(
    data: pd.DataFrame, on_failure: str,
) -> Optional[Dict[str, float]]:
    """Return failure result if data has too few rows for covariance."""
    n = len(data.columns)
    t = len(data)
    if t < max(n + 1, 2):
        msg = (f"Insufficient data: {t} rows for {n} assets. "
               f"Need at least {max(n + 1, 2)} rows for covariance.")
        return _handle_failure(on_failure, msg, data.columns)
    return None  # OK to proceed


def _handle_failure(
    on_failure: str, message: str, columns: object,
) -> Optional[Dict[str, float]]:
    """Shared failure handling for scipy-based optimizers."""
    if on_failure == "raise":
        raise RuntimeError(f"Optimizer failed: {message}")
    if on_failure == "none":
        return None
    # "equal_weight" (default)
    warnings.warn(f"Optimizer failed: {message}. Equal weight fallback.")
    col_list = list(columns)
    n = len(col_list)
    if n == 0:
        return {}
    return {sym: 1.0 / n for sym in col_list}


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """Markowitz mean-variance optimizer: max w'mu - (lambda/2) w'Sigma w.

    Requires scipy (import guarded per-method, not module-level).

    Parameters:
        risk_aversion: Lambda parameter (default 1.0). Frequency-independent
            because mu and sigma are annualized internally.
        lookback: Window for estimating mu and Sigma.
        trading_days: For annualization (default 252).
        allow_short: Allow negative weights (default False).
        max_weight: Per-asset weight cap (default 1.0).
        regularize_eps: Diagonal regularization epsilon (default 1e-6).
            Applied when N >= T to fix singular covariance.
        on_failure: Failure handling mode:
            "equal_weight" (default), "raise", or "none".
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        lookback: int = 252,
        trading_days: int = 252,
        allow_short: bool = False,
        max_weight: float = 1.0,
        regularize_eps: float = 1e-6,
        on_failure: str = "equal_weight",
    ) -> None:
        self.risk_aversion = risk_aversion
        self.lookback = lookback
        self.trading_days = trading_days
        self.allow_short = allow_short
        self.max_weight = max_weight
        self.regularize_eps = regularize_eps
        self.on_failure = on_failure

    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Optional[Dict[str, float]]:
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "MeanVarianceOptimizer requires scipy. "
                "Install with: pip install scipy"
            )

        data = returns.iloc[-self.lookback:]
        n = len(data.columns)
        t = len(data)

        if n == 0:
            return {}

        # Check minimum rows for covariance
        fail = _check_min_rows(data, self.on_failure)
        if fail is not None:
            return fail

        # Annualize
        mu = data.mean().values * self.trading_days
        sigma = data.cov().values * self.trading_days

        # Regularize if needed
        if n >= t:
            warnings.warn(
                f"N({n}) >= T({t}): covariance may be ill-conditioned. "
                f"Adding diagonal regularization eps={self.regularize_eps}."
            )
            sigma += self.regularize_eps * np.eye(n)

        def objective(w: np.ndarray) -> float:
            ret = w @ mu
            risk = w @ sigma @ w
            return -(ret - 0.5 * self.risk_aversion * risk)

        def jac(w: np.ndarray) -> np.ndarray:
            return -(mu - self.risk_aversion * sigma @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if self.allow_short:
            bounds = [(-self.max_weight, self.max_weight)] * n
        else:
            bounds = [(0, self.max_weight)] * n

        x0 = np.ones(n) / n
        result = minimize(
            objective, x0, jac=jac, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        if not result.success:
            return _handle_failure(self.on_failure, str(result.message), data.columns)

        return _postprocess_weights(
            result.x, data.columns, bounds, self.allow_short)


class RiskParityOptimizer(BasePortfolioOptimizer):
    """Equal risk contribution (risk parity) optimizer.

    Uses normalized risk contribution: (RC_i / port_var - 1/n)^2.
    Numerically stable regardless of portfolio variance scale.

    Requires scipy (import guarded per-method).

    Parameters:
        lookback: Window for covariance estimation.
        trading_days: For annualization (default 252).
        regularize_eps: Diagonal regularization epsilon (default 1e-6).
        on_failure: "equal_weight" (default), "raise", or "none".
    """

    def __init__(
        self,
        lookback: int = 252,
        trading_days: int = 252,
        regularize_eps: float = 1e-6,
        on_failure: str = "equal_weight",
    ) -> None:
        self.lookback = lookback
        self.trading_days = trading_days
        self.regularize_eps = regularize_eps
        self.on_failure = on_failure

    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Optional[Dict[str, float]]:
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "RiskParityOptimizer requires scipy. "
                "Install with: pip install scipy"
            )

        data = returns.iloc[-self.lookback:]
        n = len(data.columns)

        if n == 0:
            return {}

        fail = _check_min_rows(data, self.on_failure)
        if fail is not None:
            return fail

        sigma = data.cov().values * self.trading_days
        t = len(data)

        if n >= t:
            warnings.warn(
                f"N({n}) >= T({t}): covariance may be ill-conditioned. "
                f"Adding diagonal regularization eps={self.regularize_eps}."
            )
            sigma += self.regularize_eps * np.eye(n)

        target = 1.0 / n

        def risk_budget_objective(w: np.ndarray) -> float:
            port_var = w @ sigma @ w
            if port_var <= 0:
                return 0.0
            marginal = sigma @ w
            risk_contrib = w * marginal / port_var
            return float(np.sum((risk_contrib - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1.0)] * n
        x0 = np.ones(n) / n

        result = minimize(
            risk_budget_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        if not result.success:
            return _handle_failure(self.on_failure, str(result.message), data.columns)

        return _postprocess_weights(result.x, data.columns, bounds)


class MinimumVarianceOptimizer(BasePortfolioOptimizer):
    """Global minimum variance portfolio: min w'Sigma w.

    Requires scipy (import guarded per-method).

    Parameters:
        lookback: Window for covariance estimation.
        trading_days: For annualization (default 252).
        allow_short: Allow negative weights (default False).
        regularize_eps: Diagonal regularization epsilon (default 1e-6).
        on_failure: "equal_weight" (default), "raise", or "none".
    """

    def __init__(
        self,
        lookback: int = 252,
        trading_days: int = 252,
        allow_short: bool = False,
        regularize_eps: float = 1e-6,
        on_failure: str = "equal_weight",
    ) -> None:
        self.lookback = lookback
        self.trading_days = trading_days
        self.allow_short = allow_short
        self.regularize_eps = regularize_eps
        self.on_failure = on_failure

    def compute_weights(
        self, returns: pd.DataFrame, **kwargs: object,
    ) -> Optional[Dict[str, float]]:
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "MinimumVarianceOptimizer requires scipy. "
                "Install with: pip install scipy"
            )

        data = returns.iloc[-self.lookback:]
        n = len(data.columns)

        if n == 0:
            return {}

        fail = _check_min_rows(data, self.on_failure)
        if fail is not None:
            return fail

        sigma = data.cov().values * self.trading_days
        t = len(data)

        if n >= t:
            warnings.warn(
                f"N({n}) >= T({t}): covariance may be ill-conditioned. "
                f"Adding diagonal regularization eps={self.regularize_eps}."
            )
            sigma += self.regularize_eps * np.eye(n)

        def objective(w: np.ndarray) -> float:
            return float(w @ sigma @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return 2 * sigma @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = (
            [(None, None)] * n if self.allow_short
            else [(0, 1)] * n
        )
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, jac=jac, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        if not result.success:
            return _handle_failure(self.on_failure, str(result.message), data.columns)

        return _postprocess_weights(
            result.x, data.columns, bounds, self.allow_short)
