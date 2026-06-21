"""v2.4 Commit G — Per-timestamp factor evaluation metrics.

Cross-sectional Pearson IC, tie-corrected Spearman RankIC, and
non-NaN coverage. All three operate per-timestamp on wide
``DataFrame[index=datetime, columns=symbol]`` inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.stats import midranks, tie_corrected_spearman


def _validate_columns(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
) -> None:
    """R9: factor / forward-return columns must align EXACTLY.

    Order-sensitive equality (positional pairing of the columns is
    the implicit symbol assignment). Set-equal-but-different-order
    error message tells the user how to reindex.
    """
    if not factor.columns.equals(forward_ret.columns):
        f_set = set(factor.columns)
        r_set = set(forward_ret.columns)
        if f_set == r_set:
            raise ValueError(
                f"factor and forward-return columns share the same set "
                f"({sorted(f_set)}) but the order differs. Reindex via "
                f"`.reindex(columns=prices.columns)` to align positions."
            )
        only_factor = sorted(f_set - r_set)
        only_returns = sorted(r_set - f_set)
        raise ValueError(
            f"factor and forward-return columns do not match. Only in "
            f"factor: {only_factor[:5]}; only in forward_ret: "
            f"{only_returns[:5]}"
        )


def ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
) -> pd.Series:
    """Per-timestamp cross-sectional Pearson IC.

    At each row t, compute ``corr(factor[t, :], forward_ret[t, :])``
    across symbols (joint dropna). Returns NaN when fewer than 2
    non-NaN paired observations exist or when either side has zero
    variance at that timestamp.
    """
    _validate_columns(factor, forward_ret)
    out = pd.Series(
        np.nan, index=factor.index, dtype=float, name="ic",
    )
    for t in factor.index:
        f_row = factor.loc[t]
        r_row = forward_ret.loc[t]
        valid = (~f_row.isna()) & (~r_row.isna())
        if int(valid.sum()) < 2:
            continue
        f_v = f_row[valid].to_numpy(dtype=float)
        r_v = r_row[valid].to_numpy(dtype=float)
        if f_v.std() == 0 or r_v.std() == 0:
            continue
        out.loc[t] = float(np.corrcoef(f_v, r_v)[0, 1])
    return out


def rank_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
) -> pd.Series:
    """Per-timestamp cross-sectional tie-corrected Spearman RankIC.

    Pipeline (pinned per quant-review of v2.4 plan):
      1. Joint dropna on factor[t, :] and forward_ret[t, :].
      2. ``midranks`` of both 1-D arrays
         (:func:`aiphaforge.probes._rank.midranks`).
      3. ``tie_corrected_spearman(rank_factor, rank_return)``.

    Returns NaN when fewer than 2 non-NaN paired observations.
    """
    _validate_columns(factor, forward_ret)
    out = pd.Series(
        np.nan, index=factor.index, dtype=float, name="rank_ic",
    )
    for t in factor.index:
        f_row = factor.loc[t]
        r_row = forward_ret.loc[t]
        valid = (~f_row.isna()) & (~r_row.isna())
        if int(valid.sum()) < 2:
            continue
        f_v = f_row[valid].to_numpy(dtype=float)
        r_v = r_row[valid].to_numpy(dtype=float)
        rk_f = midranks(f_v)
        rk_r = midranks(r_v)
        out.loc[t] = float(tie_corrected_spearman(rk_f, rk_r))
    return out


def coverage(factor: pd.DataFrame) -> pd.Series:
    """Per-timestamp fraction of non-NaN factor cells across symbols."""
    if not isinstance(factor, pd.DataFrame):
        raise TypeError(
            f"factor must be pd.DataFrame, got {type(factor).__name__}"
        )
    return (~factor.isna()).sum(axis=1) / factor.shape[1]


__all__ = ["ic", "rank_ic", "coverage"]
