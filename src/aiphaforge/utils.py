"""
Utility Functions

Common constants and helper functions for quantitative finance calculations.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Standard number of trading days per year for stock markets
TRADING_DAYS_STOCK: int = 252


def _resolve_trading_days(
    trading_days: Union[int, Dict[str, int]],
    symbol: str,
    *,
    default: int = TRADING_DAYS_STOCK,
    warn_missing: bool = False,
    _warned: Optional[Set[str]] = None,
) -> int:
    """Resolve a per-symbol annualisation factor from a scalar or dict.

    Parameters:
        trading_days: Either a single integer (used for every symbol) or
            a {symbol: int} mapping.
        symbol: Symbol whose factor we want.
        default: Value to return when ``trading_days`` is a dict and
            ``symbol`` is not a key. Defaults to TRADING_DAYS_STOCK (252).
        warn_missing: If True, emit a one-time UserWarning per missing
            symbol. Combined with ``_warned`` to dedupe across calls in
            the same run.
        _warned: Optional set of symbols we have already warned about,
            mutated in place. Pass an external set to avoid duplicate
            warnings inside one backtest run.

    Returns:
        Annualisation factor (int).
    """
    if isinstance(trading_days, int):
        return trading_days
    if symbol in trading_days:
        return int(trading_days[symbol])
    if warn_missing and (_warned is None or symbol not in _warned):
        warnings.warn(
            f"trading_days dict missing entry for {symbol!r}; "
            f"falling back to {default}.",
            UserWarning,
            stacklevel=2,
        )
        if _warned is not None:
            _warned.add(symbol)
    return int(default)


def _normalize_trading_days(
    trading_days: Union[int, Dict[str, int]],
    active_symbols: Iterable[str],
    *,
    portfolio_override: Optional[int] = None,
    default: int = TRADING_DAYS_STOCK,
) -> Tuple[int, Dict[str, int]]:
    """Normalise scalar-or-dict trading_days to (portfolio_int, per_symbol_dict).

    Resolves the portfolio-level annualisation factor from the dict if
    ``portfolio_override`` is None, emitting a UserWarning when the
    active subset is mixed.

    Parameters:
        trading_days: Scalar int or {symbol: int} dict from the user.
        active_symbols: Symbols actually used in this backtest run; only
            their values participate in auto-inference.
        portfolio_override: If not None, used directly for the portfolio
            level (no warning). Typically the user's
            ``portfolio_trading_days`` argument.
        default: Fallback for symbols not in the dict.

    Returns:
        Tuple ``(portfolio_trading_days, per_symbol_map)``.
        ``per_symbol_map`` always maps every active symbol to an int.
    """
    active = list(active_symbols)

    if isinstance(trading_days, int):
        per_symbol = {sym: int(trading_days) for sym in active}
        portfolio = portfolio_override if portfolio_override is not None else int(trading_days)
        return portfolio, per_symbol

    # Dict case
    warned: Set[str] = set()
    per_symbol = {
        sym: _resolve_trading_days(
            trading_days, sym, default=default,
            warn_missing=True, _warned=warned,
        )
        for sym in active
    }

    if portfolio_override is not None:
        return int(portfolio_override), per_symbol

    # Auto-infer from values of symbols actually used in this run
    used_values = {per_symbol[sym] for sym in active}
    if not used_values:
        return int(default), per_symbol
    if len(used_values) == 1:
        # All equal — silent, unambiguous
        return int(next(iter(used_values))), per_symbol

    # Mixed values with no explicit portfolio choice: refuse rather than
    # silently pick one. A mixed-asset (e.g. stocks+crypto) portfolio has
    # no objectively correct single annualisation, and earlier attempts
    # to auto-infer (max, min) all misled users who trusted the default.
    sorted_vals = sorted(used_values)
    raise ValueError(
        f"Ambiguous trading_days: active symbols have mixed values "
        f"{sorted_vals}. Portfolio-level metrics need a single "
        f"annualisation factor; pass portfolio_trading_days= explicitly "
        f"(e.g. 252 for a stock-dominated book, 365 for crypto)."
    )


def validate_ohlcv(
    data: pd.DataFrame,
    required: Optional[List[str]] = None,
    validation_level: str = "warn",
) -> None:
    """Validate that a DataFrame contains the required OHLCV columns.

    Args:
        data: DataFrame to validate.
        required: List of required column names. Defaults to
            ["open", "high", "low", "close", "volume"].
        validation_level: Validation strictness level.
            ``"strict"``: raise ``ValueError`` on any data quality issue.
            ``"warn"`` (default): emit ``warnings.warn`` on issues.
            ``"none"``: only check column existence (legacy behaviour).

    Raises:
        ValueError: If required columns are missing, or (in strict mode) if
            any data quality check fails.
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

    # If validation_level is 'none', skip data quality checks
    if validation_level == "none":
        return

    def _report(message: str) -> None:
        if validation_level == "strict":
            raise ValueError(message)
        else:
            warnings.warn(message)

    # Check for unsorted (out-of-order) timestamps
    if hasattr(data.index, 'is_monotonic_increasing') and not data.index.is_monotonic_increasing:
        _report("OHLCV: index timestamps are not monotonically increasing (unsorted)")

    # Only run quality checks if the relevant columns exist
    has_open = "open" in columns_lower
    has_high = "high" in columns_lower
    has_low = "low" in columns_lower

    # high < low check
    if has_high and has_low:
        invalid = data["high"] < data["low"]
        n_invalid = int(invalid.sum())
        if n_invalid > 0:
            _report(f"OHLCV: {n_invalid} rows where high < low")

    # open outside [low, high] range
    if has_open and has_high and has_low:
        outside = (data["open"] < data["low"]) | (data["open"] > data["high"])
        n_outside = int(outside.sum())
        if n_outside > 0:
            _report(f"OHLCV: {n_outside} rows where open is outside [low, high]")

    # NaN in price columns
    price_cols = [c for c in ["open", "high", "low", "close"] if c in columns_lower]
    if price_cols:
        n_nan = int(data[price_cols].isna().any(axis=1).sum())
        if n_nan > 0:
            _report(f"OHLCV: {n_nan} rows with NaN in price columns")

    # Non-finite prices (inf / -inf). NaN is reported separately above.
    if price_cols:
        non_finite_mask = ~np.isfinite(data[price_cols].to_numpy(dtype=float))
        # Exclude NaN entries (already reported); just count inf/-inf rows.
        nan_mask = data[price_cols].isna().to_numpy()
        infinite_only = non_finite_mask & ~nan_mask
        n_inf = int(infinite_only.any(axis=1).sum())
        if n_inf > 0:
            _report(f"OHLCV: {n_inf} rows with non-finite (inf) prices")

    # Non-positive prices (price <= 0 is invalid for OHLC). Volume can be 0
    # and is intentionally not checked here.
    if price_cols:
        n_nonpos = int((data[price_cols] <= 0).any(axis=1).sum())
        if n_nonpos > 0:
            _report(f"OHLCV: {n_nonpos} rows with non-positive prices")

    # Duplicate timestamps
    if isinstance(data.index, pd.DatetimeIndex):
        n_dup = int(data.index.duplicated().sum())
        if n_dup > 0:
            _report(f"OHLCV: {n_dup} duplicate timestamps in index")


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


def build_unified_timeline(
    data_dict: Dict[str, pd.DataFrame],
) -> Tuple[pd.DatetimeIndex, Dict[str, Set]]:
    """Build a unified timeline from multiple per-asset DataFrames.

    Returns:
        Tuple of (timeline, bar_availability) where:
        - timeline: pd.DatetimeIndex — sorted union of all timestamps.
        - bar_availability: Dict[str, set] — per-symbol set of timestamps
          for O(1) membership checks.
    """
    if len(data_dict) == 1:
        sym, df = next(iter(data_dict.items()))
        return df.index, {sym: set(df.index)}

    all_indices = []
    bar_availability: Dict[str, set] = {}
    for sym, df in data_dict.items():
        if len(df) == 0:
            warnings.warn(f"Asset '{sym}' has an empty DataFrame")
        bar_availability[sym] = set(df.index)
        all_indices.append(df.index)

    if not all_indices:
        return pd.DatetimeIndex([]), {}

    timeline = all_indices[0]
    for idx in all_indices[1:]:
        timeline = timeline.union(idx)
    timeline = timeline.sort_values()

    return timeline, bar_availability


def build_secondary_lookup(
    primary_timeline: pd.DatetimeIndex,
    secondary_df: pd.DataFrame,
    align: str = "close",
) -> pd.Series:
    """For each primary timestamp, find the last completed secondary bar.

    Parameters:
        primary_timeline: Timestamps from the primary (highest frequency) data.
        secondary_df: OHLCV DataFrame for a single secondary timeframe/asset.
        align: Alignment mode.
            ``"close"``: secondary timestamps are bar CLOSE times.  Bar is
            complete AT this timestamp.  ``searchsorted(side='right') - 1``.
            ``"open"``: secondary timestamps are bar OPEN times.  Bar
            completes at the NEXT bar's open.  ``searchsorted(side='left') - 1``.

    Returns:
        pd.Series indexed by *primary_timeline* whose values are secondary
        timestamps (or ``NaT`` when no completed bar exists yet).
    """
    sec_idx = secondary_df.index
    side = 'right' if align == 'close' else 'left'
    positions = sec_idx.searchsorted(primary_timeline, side=side) - 1

    result = pd.Series(pd.NaT, index=primary_timeline)
    valid = positions >= 0
    result[valid] = sec_idx[positions[valid]]
    return result


def compute_buy_and_hold(data: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Compute the buy-and-hold equity curve from OHLCV data.

    Assumes the entire capital is invested at the first close price
    and held throughout.

    Args:
        data: OHLCV DataFrame with a ``close`` column.
        initial_capital: Starting capital.

    Returns:
        pd.Series: Buy-and-hold equity curve indexed by ``data.index``.
    """
    close = data["close"]
    if close.iloc[0] <= 0:
        return pd.Series(initial_capital, index=close.index)
    return close / close.iloc[0] * initial_capital


def extract_trades_vectorized(
    data: pd.DataFrame,
    positions: pd.Series,
    signals: pd.Series,
    equity: pd.Series,
    fee_model: Any,
    initial_capital: float,
    symbol: str = "default",
    stop_loss_info: Optional[tuple] = None,
) -> List[Any]:
    """Extract trade records from vectorized backtest results.

    This is a standalone utility function moved out of BacktestEngine so
    it can be reused by different execution cores.

    Parameters:
        data: OHLCV DataFrame.
        positions: Position series (1, -1, 0).
        signals: Raw signal series.
        equity: Equity curve series.
        fee_model: Fee model instance (used to estimate trade costs).
        initial_capital: Starting capital.
        symbol: Instrument symbol.
        stop_loss_info: Optional ``(trigger_mask, entry_prices, threshold)``
            tuple from ``PercentageStopLoss.apply_vectorized(..., return_mask=True)``.
            When provided, the function uses a segment-based reconstruction
            that emits ``Trade(reason='stop_loss')`` entries with the
            correct stop exit price; reversal segments are preserved.
            When None (default), falls through to the legacy in-loop
            implementation that v1.9.6 shipped (preserves numerical
            contract for the no-stop-loss case).

    Returns:
        List of Trade objects.
    """
    # Import here to avoid circular dependency
    from .results import Trade

    # v1.9.7 commit 7b: segment-based path when stop_loss is wired.
    if stop_loss_info is not None:
        trigger_mask, entry_prices, threshold = stop_loss_info
        if trigger_mask is not None and trigger_mask.any():
            return _extract_trades_with_stop_loss(
                data, positions, equity, initial_capital, symbol,
                trigger_mask, entry_prices, threshold,
            )
        # else: stop_loss was wired but never triggered → fall through
        # to legacy path (identical behavior).

    trades: List[Any] = []

    # Find position change points.
    # NOTE: pos_diff.iloc[0] is always NaN (pandas .diff() semantics).
    # Without bar-0 priming below, a non-zero positions.iloc[0] would
    # be invisible to this loop, and the next non-zero diff (which
    # actually CLOSES the bar-0 position) would be misinterpreted as
    # opening a fresh position in the wrong direction. v1.9.7 fix:
    # prime entry_time from positions.iloc[0] before walking entries.
    pos_diff = positions.diff()
    entries = pos_diff[pos_diff != 0].dropna()

    entry_time = None
    entry_price = None
    entry_direction = None
    entry_size = None
    trade_id = 0

    # v1.9.7 bar-0 priming: if the strategy emits a non-zero position
    # at bar 0, treat it as an open at bar 0 with size = |positions[0]|.
    if len(positions) > 0 and positions.iloc[0] != 0:
        entry_time = positions.index[0]
        entry_price = data.loc[entry_time, 'close']
        entry_direction = 1 if positions.iloc[0] > 0 else -1
        entry_size = abs(positions.iloc[0])

    for idx, change in entries.items():
        price = data.loc[idx, 'close']

        if entry_time is None:
            # Open position
            if change != 0:
                entry_time = idx
                entry_price = price
                entry_direction = 1 if change > 0 else -1
                entry_size = abs(change)
        else:
            # Check if closing
            new_pos = positions.loc[idx]
            if (
                new_pos == 0
                or (entry_direction > 0 and change < 0)
                or (entry_direction < 0 and change > 0)
            ):
                # Estimate real shares from equity at entry time
                entry_equity = (
                    equity.loc[entry_time]
                    if entry_time in equity.index
                    else initial_capital
                )
                # Skip zero-share trades (occurs when bankruptcy already
                # zeroed the entry-time equity).
                if entry_price <= 0 or entry_equity <= 0:
                    entry_time = None
                    continue
                estimated_shares = entry_equity * entry_size / entry_price
                if estimated_shares <= 0:
                    entry_time = None
                    continue

                # Linear PnL: shares * direction * price-change. Fees
                # are NOT deducted here — apply_vectorized has already
                # absorbed commission and slippage into the equity
                # curve, so deducting again would double-count. The
                # resulting Trade.pnl is a path-independent linear
                # approximation; see Trade.__doc__ for the discrepancy
                # contract vs. the geometric equity curve.
                trade_id += 1
                pnl = entry_direction * (price - entry_price) * estimated_shares

                trades.append(Trade(
                    trade_id=f"VT{trade_id:04d}",
                    symbol=symbol,
                    direction=entry_direction,
                    entry_time=entry_time,
                    exit_time=idx,
                    entry_price=entry_price,
                    exit_price=price,
                    size=estimated_shares,
                    pnl=pnl,
                    pnl_pct=(price / entry_price - 1) * entry_direction,
                    reason="signal",
                ))

                # If reversing position
                if new_pos != 0:
                    entry_time = idx
                    entry_price = price
                    entry_direction = 1 if new_pos > 0 else -1
                    entry_size = abs(new_pos)
                else:
                    entry_time = None

    return trades


def _extract_trades_with_stop_loss(
    data: pd.DataFrame,
    positions: pd.Series,
    equity: pd.Series,
    initial_capital: float,
    symbol: str,
    trigger_mask: pd.Series,
    entry_prices: pd.Series,
    threshold: float,
) -> List[Any]:
    """Segment-based trade reconstruction that emits stop_loss exits.

    Pre-computes (entry, close, direction, size, close_kind) tuples by
    walking pos_diff once, then truncates each segment by the first
    in-segment stop trigger. Reversal segments preserve the next
    segment's open bar (the stop only truncates THIS segment's close).
    Stop_loss exit price uses the formula from
    ``PercentageStopLoss.apply_vectorized``: ``entry_price * (1 -
    threshold * direction)``.

    See plan v3 R1 for why this is segment-based rather than retrofit
    into the in-loop state machine of ``extract_trades_vectorized``.
    """
    import numpy as np

    from .results import Trade

    trades: List[Any] = []

    # ---- Pre-compute segments by walking pos_diff -----------------
    pos_diff = positions.diff()
    entries = pos_diff[pos_diff != 0].dropna()

    # bar-0 priming (mirrors the v1.9.7 commit 7a fix in the legacy path)
    segments: List[tuple] = []  # (entry_bar, direction, size, close_bar, close_kind)
    current = None  # (entry_bar, direction, size)
    if len(positions) > 0 and positions.iloc[0] != 0:
        current = (positions.index[0],
                   1 if positions.iloc[0] > 0 else -1,
                   abs(positions.iloc[0]))

    for idx, change in entries.items():
        if current is None:
            if change != 0:
                current = (idx, 1 if change > 0 else -1,
                           abs(positions.loc[idx]))
        else:
            new_pos = positions.loc[idx]
            if new_pos == 0:
                segments.append((*current, idx, "flat"))
                current = None
            elif np.sign(new_pos) != current[1]:
                segments.append((*current, idx, "reversal"))
                current = (idx, 1 if new_pos > 0 else -1,
                           abs(new_pos))
            # else: same direction, size change; mirror legacy
            # close-and-reopen behavior
            elif new_pos != current[2]:
                segments.append((*current, idx, "flat"))
                current = (idx, 1 if new_pos > 0 else -1,
                           abs(new_pos))

    # If a position is still open at end-of-data, include it as a
    # candidate segment ONLY if a stop fires within (so we can emit
    # the stop_loss trade). Without this, an open-at-end position with
    # an in-segment stop is silently dropped — bug surfaced by the
    # bar-0 + stop-loss test.
    # Open positions that don't hit a stop remain unrepresented, which
    # matches the legacy in-loop path (no trade emitted for never-
    # closed positions).
    if current is not None:
        end_bar = positions.index[-1]
        seg_stops = trigger_mask[(trigger_mask.index > current[0])
                                  & (trigger_mask.index <= end_bar)
                                  & trigger_mask]
        if not seg_stops.empty:
            segments.append((*current, end_bar, "open"))

    # ---- Truncate segments by stop triggers -----------------------
    truncated: List[tuple] = []
    for entry_bar, direction, size_signal, close_bar, close_kind in segments:
        # Stops fire strictly between (entry_bar, close_bar) for natural
        # closes — at entry there's no PnL yet, at close the natural
        # close already exits. For "open" segments (no natural close),
        # include the end_bar in the search since nothing else can take
        # precedence at that bar.
        if close_kind == "open":
            seg_mask = trigger_mask[(trigger_mask.index > entry_bar)
                                    & (trigger_mask.index <= close_bar)
                                    & trigger_mask]
        else:
            seg_mask = trigger_mask[(trigger_mask.index > entry_bar)
                                    & (trigger_mask.index < close_bar)
                                    & trigger_mask]
        if not seg_mask.empty:
            stop_bar = seg_mask.index[0]
            truncated.append(
                (entry_bar, direction, size_signal, stop_bar, "stop_loss"))
        elif close_kind == "open":
            # Open at end with no stop — drop (matches legacy behavior).
            continue
        else:
            truncated.append(
                (entry_bar, direction, size_signal, close_bar, close_kind))

    # ---- Emit Trade objects ---------------------------------------
    trade_id = 0
    for entry_bar, direction, size_signal, close_bar, close_kind in truncated:
        entry_price = data.loc[entry_bar, 'close']
        entry_equity = (
            equity.loc[entry_bar] if entry_bar in equity.index
            else initial_capital
        )
        if entry_price <= 0 or entry_equity <= 0:
            continue
        shares = entry_equity * size_signal / entry_price
        if shares <= 0:
            continue

        if close_kind == "stop_loss":
            # Match apply_vectorized:112 exactly
            exit_price = entry_price * (1 - threshold * direction)
            reason = "stop_loss"
        else:  # "flat" or "reversal"
            exit_price = data.loc[close_bar, 'close']
            reason = "signal"

        pnl = direction * (exit_price - entry_price) * shares
        trade_id += 1
        trades.append(Trade(
            trade_id=f"VT{trade_id:04d}",
            symbol=symbol,
            direction=direction,
            entry_time=entry_bar,
            exit_time=close_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            size=shares,
            pnl=pnl,
            pnl_pct=(exit_price / entry_price - 1) * direction,
            reason=reason,
        ))
    return trades


def calculate_trade_metrics(trades: List[Any]) -> Dict[str, float]:
    """Compute trade-level statistics from a list of Trade objects.

    Each trade must have ``pnl`` (float), ``entry_time``, and ``exit_time``
    attributes.  The returned dict uses the same keys as
    ``BacktestEngine._calculate_metrics`` so it can be merged directly.

    Args:
        trades: List of Trade objects (or any objects with the attributes
            described above).

    Returns:
        Dict with keys: num_trades, win_rate, num_winners, num_losers,
        avg_win, avg_loss, profit_factor, avg_holding_days.
    """
    metrics: Dict[str, float] = {}
    metrics['num_trades'] = len(trades)

    if len(trades) > 0:
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        metrics['win_rate'] = len(winners) / len(trades)
        metrics['num_winners'] = len(winners)
        metrics['num_losers'] = len(losers)

        # Average win/loss
        metrics['avg_win'] = float(np.mean([t.pnl for t in winners])) if winners else 0.0
        metrics['avg_loss'] = float(np.mean([t.pnl for t in losers])) if losers else 0.0

        # Profit factor
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))
        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses
        else:
            metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0

        # Average holding time
        holding_times = [
            (t.exit_time - t.entry_time).total_seconds() / 86400
            for t in trades
        ]
        metrics['avg_holding_days'] = float(np.mean(holding_times)) if holding_times else 0.0

    else:
        metrics['win_rate'] = 0.0
        metrics['num_winners'] = 0.0
        metrics['num_losers'] = 0.0
        metrics['avg_win'] = 0.0
        metrics['avg_loss'] = 0.0
        metrics['profit_factor'] = 0.0
        metrics['avg_holding_days'] = 0.0

    return metrics
