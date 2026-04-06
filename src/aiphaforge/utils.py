"""
Utility Functions

Common constants and helper functions for quantitative finance calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

# Standard number of trading days per year for stock markets
TRADING_DAYS_STOCK: int = 252


def validate_ohlcv(
    data: pd.DataFrame,
    required: Optional[List[str]] = None,
) -> None:
    """Validate that a DataFrame contains the required OHLCV columns.

    Args:
        data: DataFrame to validate.
        required: List of required column names. Defaults to
            ["open", "high", "low", "close", "volume"].

    Raises:
        ValueError: If any required columns are missing.
        TypeError: If *data* is not a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(data).__name__}")

    if required is None:
        required = ["open", "high", "low", "close", "volume"]

    columns_lower = [c.lower() for c in data.columns]
    missing = [col for col in required if col.lower() not in columns_lower]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a DatetimeIndex.

    If the index is not already a DatetimeIndex, attempt to convert it.
    Returns a copy with the converted index.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with a DatetimeIndex.

    Raises:
        ValueError: If the index cannot be converted to DatetimeIndex.
    """
    if isinstance(data.index, pd.DatetimeIndex):
        return data

    try:
        data = data.copy()
        data.index = pd.to_datetime(data.index)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert index to DatetimeIndex: {exc}"
        ) from exc
    return data


def calculate_returns(equity_series: pd.Series) -> pd.Series:
    """Compute percentage returns from an equity or price series.

    Args:
        equity_series: Series of equity values or prices.

    Returns:
        Series of percentage returns (first value is NaN, then dropped).
    """
    returns = equity_series.pct_change().dropna()
    return returns


def sharpe_ratio(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_STOCK,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns.
        trading_days: Number of trading days per year for annualization.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Annualized Sharpe ratio, or 0.0 if standard deviation is zero.
    """
    if len(returns) == 0:
        return 0.0

    # Convert annual risk-free rate to per-period rate
    rf_per_period = risk_free_rate / trading_days
    excess = returns - rf_per_period
    std = excess.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float((excess.mean() / std) * np.sqrt(trading_days))


def sortino_ratio(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_STOCK,
    risk_free_rate: float = 0.0,
    downside_method: str = "full",
) -> float:
    """Calculate the annualized Sortino ratio.

    Args:
        returns: Series of periodic returns.
        trading_days: Number of trading days per year for annualization.
        risk_free_rate: Annualized risk-free rate.
        downside_method: Method for computing downside deviation.
            ``"full"`` (default): standard formula using all observations,
            ``sqrt(mean(min(excess, 0)^2))``.
            ``"negative_only"``: use only negative excess returns,
            ``sqrt(mean(neg^2))``.

    Returns:
        Annualized Sortino ratio, or 0.0 if downside deviation is zero.
    """
    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / trading_days
    excess = returns - rf_per_period

    if downside_method == "negative_only":
        downside = excess[excess < 0]
        if len(downside) == 0:
            return 0.0
        downside_std = np.sqrt((downside ** 2).mean())
    else:
        # "full": compute min(excess, 0) for ALL observations
        clipped = np.minimum(excess, 0.0)
        downside_std = np.sqrt((clipped ** 2).mean())

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    return float((excess.mean() / downside_std) * np.sqrt(trading_days))


def max_drawdown(equity: pd.Series) -> float:
    """Calculate the maximum drawdown from an equity curve.

    Args:
        equity: Series of portfolio equity values.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.15 means 15% drawdown).
        Returns 0.0 if the equity series is empty or has no drawdown.
    """
    if len(equity) == 0:
        return 0.0

    cumulative_max = equity.cummax()
    drawdowns = (cumulative_max - equity) / cumulative_max
    mdd = drawdowns.max()

    if np.isnan(mdd):
        return 0.0

    return float(mdd)


def annualize_return(
    total_return: float,
    n_days: int,
    trading_days: int = TRADING_DAYS_STOCK,
) -> float:
    """Annualize a total return given the number of trading days.

    Args:
        total_return: Total return as a fraction (e.g. 0.5 for 50%).
        n_days: Number of trading days in the period.
        trading_days: Number of trading days per year.

    Returns:
        Annualized return as a fraction.
    """
    if n_days <= 0:
        return 0.0

    years = n_days / trading_days
    if years == 0:
        return 0.0

    # Handle negative total returns that would make (1 + total_return) <= 0
    if total_return <= -1.0:
        return -1.0

    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def annualize(
    value: float,
    trading_days: int = TRADING_DAYS_STOCK,
    is_volatility: bool = False,
) -> float:
    """General annualization helper.

    Args:
        value: Per-period value to annualize.
        trading_days: Number of trading days per year.
        is_volatility: If True, annualize using sqrt(trading_days)
            (appropriate for volatility/std). If False, multiply by
            trading_days (appropriate for returns).

    Returns:
        Annualized value.
    """
    if is_volatility:
        return float(value * np.sqrt(trading_days))
    return float(value * trading_days)
