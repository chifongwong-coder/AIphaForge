"""v2.9.1.1: edge-triggered event-driven size-mode signals.

A repeated, unchanged non-flat signal must NOT re-rebalance a held position
every bar (the phantom-turnover bug). It is treated as "hold", identical to
emitting the signal once then NaN. ``resize_on_repeat_signal=True`` restores the
legacy level-triggered behavior.

See docs/AIphaForge_v2.9.1.1_plan.md and
docs/framework_bug_2026-06-25_daily_return_mark_mismatch.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine


def _trending_ohlcv(n: int = 40, seed: int = 1) -> pd.DataFrame:
    """Trending series with open != close on every bar (the conditions
    under which the phantom-rebalance churn appeared)."""
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.002, 0.012, n))
    open_ = close * (1 + rng.normal(0.0, 0.006, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _run(df: pd.DataFrame, signals: pd.Series, **kwargs):
    eng = BacktestEngine(mode="event_driven", allow_short=True,
                         include_benchmark=False, **kwargs)
    eng.set_signals(signals)
    return eng.run(df)


@pytest.mark.parametrize("c", [1.0, 0.5, -1.0])
def test_const_signal_equals_once_then_nan_equity(c):
    """The acceptance invariant: a constant signal produces a bit-identical
    equity_curve and num_trades to emitting that signal once then NaN."""
    df = _trending_ohlcv()
    const = pd.Series(c, index=df.index)
    once = pd.Series([c] + [float("nan")] * (len(df) - 1), index=df.index)

    r_const = _run(df, const)
    r_once = _run(df, once)

    assert r_const.num_trades == r_once.num_trades
    np.testing.assert_array_equal(
        r_const.equity_curve.values, r_once.equity_curve.values)


def test_const_signal_no_phantom_turnover():
    """A constant long signal enters once and holds — no per-bar rebalance
    fills (pre-fix this churned a fill on nearly every bar)."""
    df = _trending_ohlcv()
    r = _run(df, pd.Series(1.0, index=df.index))
    fills = r.orders[r.orders["filled_size"] > 0]
    assert len(fills) == 1  # only the single entry
    assert (fills["side"] == "buy").all()


def test_signal_change_still_acts():
    """Edge-triggering only suppresses UNCHANGED signals; a changed signal
    (1.0 -> 0.5 -> 0.0) still resizes at each transition."""
    df = _trending_ohlcv(n=30)
    sig = pd.Series(1.0, index=df.index)
    sig.iloc[10:20] = 0.5
    sig.iloc[20:] = 0.0
    r = _run(df, sig)

    fills = r.orders[r.orders["filled_size"] > 0]
    # entry (buy) + trim at 1.0->0.5 (sell) + exit at 0.5->0.0 (sell)
    assert (fills["side"] == "buy").any()
    assert (fills["side"] == "sell").sum() >= 2


def test_resize_on_repeat_signal_restores_legacy():
    """resize_on_repeat_signal=True reproduces the legacy level-triggered
    behavior: a constant signal re-rebalances and churns extra fills."""
    df = _trending_ohlcv()
    const = pd.Series(1.0, index=df.index)

    edge = _run(df, const)
    legacy = _run(df, const, resize_on_repeat_signal=True)

    n_edge = int((edge.orders["filled_size"] > 0).sum())
    n_legacy = int((legacy.orders["filled_size"] > 0).sum())
    assert n_edge == 1
    assert n_legacy > n_edge  # legacy churns
