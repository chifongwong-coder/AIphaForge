"""
Data Loaders
=============

Convenience utilities for loading OHLCV data from common sources.
Not imported by the main package — use directly::

    from aiphaforge.data import load_csv, load_yahoo

Optional dependencies:
- yfinance (for load_yahoo): ``pip install aiphaforge[data]``
"""

from typing import Dict, List, Optional, Union

import pandas as pd

from .utils import validate_ohlcv

# Standard OHLCV column name variants for auto-detection
_OHLCV_ALIASES = {
    "open": ["open", "Open", "OPEN", "o"],
    "high": ["high", "High", "HIGH", "h"],
    "low": ["low", "Low", "LOW", "l"],
    "close": ["close", "Close", "CLOSE", "c", "adj close", "Adj Close"],
    "volume": ["volume", "Volume", "VOLUME", "v", "vol", "Vol"],
}

_DATE_HINTS = ["date", "time", "datetime", "timestamp", "dt", "trade_date"]


def load_csv(
    path: str,
    date_col: Optional[str] = None,
    ohlcv_cols: Optional[Dict[str, str]] = None,
    validation: str = "warn",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Parameters:
        path: Path to CSV file.
        date_col: Name of the date/time column. If None, auto-detect
            by matching column names against common patterns.
        ohlcv_cols: Explicit column mapping, e.g.,
            ``{"Open": "open", "Close": "close", ...}``. If None,
            auto-detect by case-insensitive matching.
        validation: Validation level for the output ('strict', 'warn', 'none').

    Returns:
        pd.DataFrame with lowercase OHLCV columns and DatetimeIndex.

    Raises:
        ValueError: If date column or OHLCV columns cannot be detected.
    """
    df = pd.read_csv(path)

    # --- Detect date column ---
    if date_col is None:
        candidates = [
            c for c in df.columns
            if c.lower().strip() in _DATE_HINTS
        ]
        if len(candidates) == 1:
            date_col = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"Multiple date column candidates: {candidates}. "
                f"Use date_col='...' to specify.")
        else:
            raise ValueError(
                f"Cannot auto-detect date column from: {list(df.columns)}. "
                f"Use date_col='...' to specify.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # --- Map OHLCV columns ---
    if ohlcv_cols is not None:
        df = df.rename(columns=ohlcv_cols)
    else:
        rename_map = {}
        for target, aliases in _OHLCV_ALIASES.items():
            for col in df.columns:
                if col.strip() in aliases:
                    rename_map[col] = target
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

    # Ensure lowercase
    df.columns = [c.lower() for c in df.columns]

    validate_ohlcv(df, validation_level=validation)
    return df


def load_yahoo(
    symbols: Union[str, List[str]],
    start: Optional[str] = None,
    end: Optional[str] = None,
    validation: str = "warn",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load OHLCV data from Yahoo Finance via yfinance.

    Parameters:
        symbols: Single ticker string or list of tickers.
        start: Start date string (e.g., '2020-01-01').
        end: End date string.
        validation: Validation level for the output.

    Returns:
        Single symbol: pd.DataFrame with OHLCV columns.
        Multiple symbols: Dict[str, pd.DataFrame] ready for multi-asset.

    Raises:
        ImportError: If yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance required for load_yahoo. "
            "Install with: pip install aiphaforge[data]")

    single = isinstance(symbols, str)
    if single:
        symbols_list = [symbols]
    else:
        symbols_list = list(symbols)

    raw = yf.download(
        symbols_list,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for {symbols_list}")

    if single:
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        validate_ohlcv(df, validation_level=validation)
        return df

    # Multi-symbol: yfinance returns MultiIndex columns (price, symbol)
    result = {}
    for sym in symbols_list:
        try:
            sym_df = raw.xs(sym, level="Ticker", axis=1).copy()
        except KeyError:
            # Fallback for different yfinance versions
            sym_df = raw[sym].copy() if sym in raw.columns.get_level_values(0) else None
        if sym_df is None or sym_df.empty:
            continue
        sym_df.columns = [c.lower() for c in sym_df.columns]
        validate_ohlcv(sym_df, validation_level=validation)
        result[sym] = sym_df

    if not result:
        raise ValueError(f"No valid data for any symbol in {symbols_list}")

    return result
