"""
Market Impact Models & Strategy Capacity Estimation
====================================================

Provides order-size-dependent cost models that estimate the price
movement caused by your own trading, plus tools to estimate maximum
strategy capacity before returns degrade below a threshold.

Models implemented:
    - LinearImpactModel: Simple proportional impact.
    - SquareRootImpactModel: Almgren-Chriss temporary + permanent impact.
    - PowerLawImpactModel: Generalized power-law impact.

Utilities:
    - Parkinson and close-to-close volatility estimators.
    - Average daily volume (ADV) computation.
    - Corwin-Schultz bid-ask spread estimator.
    - Calibration helpers and capacity estimation.
"""

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class BaseImpactModel(ABC):
    """Abstract base class for market impact models."""

    @abstractmethod
    def estimate_impact(
        self,
        order_size: float,
        price: float,
        adv: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        """Estimate price impact as a fraction of price.

        Parameters:
            order_size: Shares being traded (absolute).
            price: Current market price.
            adv: Average daily volume (rolling N-day average).
            volatility: Daily volatility estimate.

        Returns:
            Impact as positive fraction (e.g., 0.001 = 10 bps).
        """
        ...

    def _check_participation(self, size: float, adv: float) -> None:
        """Warn when the order exceeds 50% of ADV."""
        if adv > 0 and size / adv > 0.5:
            warnings.warn(
                f"Participation rate {size / adv:.1%} exceeds 50% of ADV. "
                f"Market impact estimates may be unreliable."
            )


# ---------------------------------------------------------------------------
# Built-in impact models
# ---------------------------------------------------------------------------

class LinearImpactModel(BaseImpactModel):
    """Linear impact model: impact = eta * (size / ADV).

    Parameters:
        eta: Proportionality coefficient (default 0.1, typical range
            0.05-0.20 for equities).
    """

    def __init__(self, eta: float = 0.1) -> None:
        self.eta = eta

    def estimate_impact(
        self,
        order_size: float,
        price: float,
        adv: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        if adv <= 0 or order_size <= 0:
            return 0.0
        self._check_participation(order_size, adv)
        return self.eta * (order_size / adv)


class SquareRootImpactModel(BaseImpactModel):
    """Almgren-Chriss square-root impact model.

    temporary = sigma * eta * sqrt(size / ADV)
    permanent = sigma * gamma * (size / ADV)
    total = temporary + permanent

    Parameters:
        eta: Temporary impact coefficient (default 0.5).
        gamma: Permanent impact coefficient (default 0.1).
            Set to 0 to disable permanent impact.

    References:
        Almgren & Chriss (2000), "Optimal execution of portfolio
        transactions".
        Almgren et al. (2005), "Direct estimation of equity market
        impact".
    """

    def __init__(self, eta: float = 0.5, gamma: float = 0.1) -> None:
        self.eta = eta
        self.gamma = gamma

    def estimate_impact(
        self,
        order_size: float,
        price: float,
        adv: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        if adv <= 0 or order_size <= 0:
            return 0.0
        self._check_participation(order_size, adv)
        participation = order_size / adv
        temporary = volatility * self.eta * math.sqrt(participation)
        permanent = volatility * self.gamma * participation
        return temporary + permanent


class PowerLawImpactModel(BaseImpactModel):
    """Power-law impact model: impact = sigma * eta * (size / ADV) ^ alpha.

    Parameters:
        eta: Coefficient (default 0.5).
        alpha: Exponent (default 0.5 = equivalent to square-root).
    """

    def __init__(self, eta: float = 0.5, alpha: float = 0.5) -> None:
        self.eta = eta
        self.alpha = alpha

    def estimate_impact(
        self,
        order_size: float,
        price: float,
        adv: float,
        volatility: float,
        **kwargs: Any,
    ) -> float:
        if adv <= 0 or order_size <= 0:
            return 0.0
        self._check_participation(order_size, adv)
        participation = order_size / adv
        return volatility * self.eta * (participation ** self.alpha)


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------

def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20,
) -> float:
    """Parkinson (1980) high-low volatility estimator (scalar).

    Uses the most recent *lookback* bars.  More efficient than
    close-to-close (5x lower variance) and captures intraday movement.

    sigma = sqrt(1 / (4 * N * ln2) * sum(ln(H/L)^2))
    """
    h = high.iloc[-lookback:]
    lo = low.iloc[-lookback:]
    log_hl = np.log(h.values / lo.values)
    n = len(log_hl)
    if n == 0:
        return 0.0
    return float(np.sqrt(np.sum(log_hl ** 2) / (4 * n * np.log(2))))


def parkinson_volatility_series(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Rolling Parkinson volatility (full Series for pre-computation).

    Returns a Series aligned with the input index.
    """
    log_hl_sq = np.log(high / low) ** 2
    rolling_mean = log_hl_sq.rolling(window=lookback, min_periods=1).mean()
    return np.sqrt(rolling_mean / (4 * np.log(2)))


def close_volatility(
    close: pd.Series,
    lookback: int = 20,
) -> float:
    """Standard close-to-close return volatility (scalar).

    Uses the most recent *lookback* log returns.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    tail = log_returns.iloc[-lookback:]
    if len(tail) == 0:
        return 0.0
    return float(tail.std())


# ---------------------------------------------------------------------------
# ADV estimators
# ---------------------------------------------------------------------------

def compute_adv(
    volume: pd.Series,
    lookback: int = 20,
) -> float:
    """Average daily volume from the most recent *lookback* bars."""
    tail = volume.iloc[-lookback:]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def compute_adv_series(
    volume: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Rolling average daily volume (full Series for pre-computation)."""
    return volume.rolling(window=lookback, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Spread estimator
# ---------------------------------------------------------------------------

def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> float:
    """Corwin-Schultz (2012) bid-ask spread estimator.

    Estimates effective spread from two-day high-low ranges.
    No order book needed -- uses only OHLCV data.

    Returns spread as fraction of price (e.g. 0.002 = 20 bps).
    Returns 0.0 when fewer than 2 bars are available.
    """
    if len(high) < 2 or len(low) < 2:
        return 0.0

    # Two-day combined high/low
    high_2 = high.rolling(2).max()
    low_2 = low.rolling(2).min()

    # Beta = sum of squared single-day log(H/L)
    log_hl_sq = np.log(high / low) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)

    # Gamma = log(two-day-high / two-day-low) ^ 2
    gamma = np.log(high_2 / low_2) ** 2

    # Alpha
    sqrt_2 = math.sqrt(2)
    denom = 3 - 2 * sqrt_2  # ≈ 0.172 (Corwin & Schultz 2012)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

    # Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Take the last valid value; clamp to >= 0
    valid = spread.dropna()
    if len(valid) == 0:
        return 0.0
    return float(max(valid.iloc[-1], 0.0))


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

_MARKET_PARAMS: Dict[str, Dict[str, float]] = {
    "us_large_cap": {"eta": 0.5, "gamma": 0.1, "alpha": 0.5},
    "us_small_cap": {"eta": 0.8, "gamma": 0.2, "alpha": 0.6},
    "china_a": {"eta": 0.6, "gamma": 0.15, "alpha": 0.55},
    "crypto_spot": {"eta": 1.0, "gamma": 0.3, "alpha": 0.5},
    "crypto_futures": {"eta": 0.7, "gamma": 0.2, "alpha": 0.5},
}


def suggested_impact_params(
    market: str = "us_large_cap",
) -> Dict[str, float]:
    """Suggest impact model parameters by market type.

    Based on Almgren et al. (2005) and Frazzini et al. (2018).

    Supported markets: "us_large_cap", "us_small_cap", "china_a",
    "crypto_spot", "crypto_futures".

    Returns dict with eta, gamma, alpha keys.
    """
    if market not in _MARKET_PARAMS:
        raise ValueError(
            f"Unknown market '{market}'. Supported: "
            f"{', '.join(sorted(_MARKET_PARAMS))}"
        )
    return dict(_MARKET_PARAMS[market])


# ---------------------------------------------------------------------------
# Capacity estimation
# ---------------------------------------------------------------------------

@dataclass
class CapacityResult:
    """Result of strategy capacity estimation.

    Attributes:
        estimated_capacity: Max capital before Sharpe drops below
            *min_sharpe*.  None if never drops below threshold.
        results_by_capital: DataFrame with columns [capital_multiplier,
            capital, total_impact_cost, adjusted_return,
            adjusted_sharpe].
        impact_model: Name of the impact model used.
        min_sharpe: Sharpe threshold used.
    """
    estimated_capacity: Optional[float]
    results_by_capital: pd.DataFrame
    impact_model: str
    min_sharpe: float


def estimate_capacity(
    result: Any,
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    impact_model: Optional[BaseImpactModel] = None,
    capital_multipliers: Optional[List[float]] = None,
    min_sharpe: float = 1.0,
    trading_days: int = 252,
    volatility_method: str = "parkinson",
    adv_lookback: int = 20,
) -> CapacityResult:
    """Estimate strategy capacity from existing backtest trades.

    Scales trade sizes by *capital_multipliers*, computes cumulative
    impact cost, and adjusts returns/Sharpe.  Uses bisection between
    adjacent multipliers for a precise capacity boundary.

    Parameters:
        result: A BacktestResult from a previous backtest run.
        data: OHLCV data (single-asset DataFrame or dict of DataFrames).
        impact_model: Impact model to use (default SquareRootImpactModel).
        capital_multipliers: List of scale factors to try (e.g.
            [1, 2, 5, 10, 20]).  Default [1, 2, 5, 10, 20, 50].
        min_sharpe: Minimum acceptable Sharpe ratio.
        trading_days: Number of trading days per year.
        volatility_method: "parkinson" or "close".
        adv_lookback: Lookback window for ADV computation.

    Returns:
        CapacityResult with estimated capacity and detailed results.
    """
    if impact_model is None:
        impact_model = SquareRootImpactModel()

    if capital_multipliers is None:
        capital_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    # Normalise data to dict form. v1.9.7 made result.symbols reliable
    # for single-asset runs (was the v1.9.6 capacity-fallback workaround).
    # The result.trades[0] branch stays as defense-in-depth for any
    # custom core that produces a result with empty symbols.
    if isinstance(data, pd.DataFrame):
        if result.symbols:
            sym = result.symbols[0]
        elif result.trades:
            sym = result.trades[0].symbol
        else:
            sym = "default"
        data_dict: Dict[str, pd.DataFrame] = {sym: data}
    else:
        data_dict = data

    # Pre-compute volatility and ADV per symbol
    vol_map: Dict[str, float] = {}
    adv_map: Dict[str, float] = {}
    for sym, df in data_dict.items():
        if volatility_method == "parkinson":
            vol_map[sym] = parkinson_volatility(
                df['high'], df['low'], adv_lookback)
        else:
            vol_map[sym] = close_volatility(df['close'], adv_lookback)
        adv_map[sym] = compute_adv(df['volume'], adv_lookback)

    # Baseline metrics from result
    base_return = result.total_return
    base_initial = result.initial_capital

    # Daily returns for Sharpe adjustment
    daily_rets = result.daily_returns

    n_days = len(result.equity_curve) if len(result.equity_curve) > 0 else 1

    rows: List[Dict[str, Any]] = []
    for mult in sorted(capital_multipliers):
        # Scaled impact: trades are *mult* times larger
        total_impact_cost = 0.0
        for trade in result.trades:
            sym = trade.symbol
            scaled_size = trade.size * mult
            price = trade.entry_price
            adv = adv_map.get(sym, 0.0)
            vol = vol_map.get(sym, 0.0)
            impact_frac = impact_model.estimate_impact(
                scaled_size, price, adv, vol)
            total_impact_cost += impact_frac * scaled_size * price

        capital = base_initial * mult
        impact_drag = total_impact_cost / capital if capital > 0 else 0.0
        adjusted_return = base_return - impact_drag

        # Adjust Sharpe: approximate by subtracting daily impact drag
        daily_drag = impact_drag / n_days if n_days > 0 else 0.0
        if daily_rets is not None and len(daily_rets) > 0:
            std_r = float(daily_rets.std())
            adj_mean = float(daily_rets.mean()) - daily_drag
            adj_sharpe = (
                adj_mean / std_r * math.sqrt(trading_days)
                if std_r > 0 else 0.0
            )
        else:
            adj_sharpe = 0.0

        rows.append({
            'capital_multiplier': mult,
            'capital': capital,
            'total_impact_cost': total_impact_cost,
            'adjusted_return': adjusted_return,
            'adjusted_sharpe': adj_sharpe,
        })

    results_df = pd.DataFrame(rows)

    # Find capacity via bisection between adjacent multipliers
    estimated_capacity = _bisect_capacity(
        results_df, result, data_dict, impact_model,
        vol_map, adv_map, base_initial, base_return,
        daily_rets, n_days, trading_days, min_sharpe,
    )

    model_name = type(impact_model).__name__
    return CapacityResult(
        estimated_capacity=estimated_capacity,
        results_by_capital=results_df,
        impact_model=model_name,
        min_sharpe=min_sharpe,
    )


def _bisect_capacity(
    results_df: pd.DataFrame,
    result: Any,
    data_dict: Dict[str, pd.DataFrame],
    impact_model: BaseImpactModel,
    vol_map: Dict[str, float],
    adv_map: Dict[str, float],
    base_initial: float,
    base_return: float,
    daily_rets: Optional[pd.Series],
    n_days: int,
    trading_days: int,
    min_sharpe: float,
) -> Optional[float]:
    """Bisect between adjacent multipliers to find precise capacity."""
    sharpes = results_df['adjusted_sharpe'].tolist()
    mults = results_df['capital_multiplier'].tolist()

    # If all multipliers still above threshold, capacity is unbounded
    if all(s >= min_sharpe for s in sharpes):
        return None

    # Find the first crossing
    lo_idx: Optional[int] = None
    for i in range(len(sharpes)):
        if sharpes[i] < min_sharpe:
            lo_idx = i
            break

    if lo_idx is None:
        return None

    if lo_idx == 0:
        # Even the smallest multiplier is below threshold
        return base_initial * mults[0]

    lo_mult = mults[lo_idx - 1]
    hi_mult = mults[lo_idx]

    # Bisection (10 iterations for ~1000x precision)
    for _ in range(10):
        mid = (lo_mult + hi_mult) / 2.0
        sharpe = _sharpe_at_multiplier(
            mid, result, impact_model, vol_map, adv_map,
            base_initial, base_return, daily_rets, n_days, trading_days,
        )
        if sharpe >= min_sharpe:
            lo_mult = mid
        else:
            hi_mult = mid

    return base_initial * (lo_mult + hi_mult) / 2.0


def _sharpe_at_multiplier(
    mult: float,
    result: Any,
    impact_model: BaseImpactModel,
    vol_map: Dict[str, float],
    adv_map: Dict[str, float],
    base_initial: float,
    base_return: float,
    daily_rets: Optional[pd.Series],
    n_days: int,
    trading_days: int,
) -> float:
    """Compute adjusted Sharpe at a given capital multiplier."""
    total_impact_cost = 0.0
    for trade in result.trades:
        sym = trade.symbol
        scaled_size = trade.size * mult
        price = trade.entry_price
        adv = adv_map.get(sym, 0.0)
        vol = vol_map.get(sym, 0.0)
        impact_frac = impact_model.estimate_impact(
            scaled_size, price, adv, vol)
        total_impact_cost += impact_frac * scaled_size * price

    capital = base_initial * mult
    impact_drag = total_impact_cost / capital if capital > 0 else 0.0
    daily_drag = impact_drag / n_days if n_days > 0 else 0.0

    if daily_rets is not None and len(daily_rets) > 0:
        std_r = float(daily_rets.std())
        adj_mean = float(daily_rets.mean()) - daily_drag
        return adj_mean / std_r * math.sqrt(trading_days) if std_r > 0 else 0.0
    return 0.0
