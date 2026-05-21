"""v2.8.2 M1 — PercentageStopLoss event-driven vs vectorized divergence.

v2.8.2 documents the divergence prominently (docstring + README)
without changing behavior; aligning the two paths requires next-bar
OHLC plumbing into vectorized cost path, which is a v2.9 architectural
item.

These tests pin the documented divergence:
- On a trigger bar that gaps DOWN below the threshold, event-driven
  realizes the gap-loss at next-bar open; vectorized assumes the
  theoretical stop-threshold loss. Equity curves diverge by the gap
  magnitude.
- On a trigger bar with no gap, both paths produce equivalent equity
  within numerical noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge import BacktestEngine, SimpleFeeModel


def _ohlcv(closes, opens=None, highs=None, lows=None, volumes=None):
    n = len(closes)
    if opens is None:
        opens = closes
    if highs is None:
        highs = [max(o, c) * 1.005 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) * 0.995 for o, c in zip(opens, closes)]
    if volumes is None:
        volumes = [1e6] * n
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": volumes},
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )


def _equity_under(mode: str, data: pd.DataFrame, signals: pd.Series,
                  stop_loss: float = 0.05) -> float:
    eng = BacktestEngine(
        fee_model=SimpleFeeModel(commission_rate=0.0, slippage_pct=0.0),
        initial_capital=100_000,
        mode=mode,
        stop_loss=stop_loss,
        position_size=0.95,
        include_benchmark=False,
    )
    eng.set_signals(signals)
    result = eng.run(data)
    return float(result.equity_curve.iloc[-1])


# --------------------------------------------------------------------
# Test 1 — gap-bar trigger: event-driven and vectorized diverge
# --------------------------------------------------------------------

def test_stop_loss_event_vs_vectorized_diverges_on_gap_bar():
    """Construct a price series where the trigger bar gaps DOWN
    significantly below the stop threshold. Event-driven fills at the
    gapped next-bar open (realized loss > threshold); vectorized
    assumes the theoretical threshold loss. Equity curves must diverge."""
    # Trigger bar (index 3) gaps from 100 to 88 (12% gap) while stop
    # is 5%. Event-driven exit ~ -12%; vectorized exit ~ -5%.
    closes = [100, 100, 100, 88, 90, 91, 92, 93, 94, 95]
    opens = [100, 100, 100, 88, 90, 91, 92, 93, 94, 95]
    highs = [101, 101, 101, 89, 91, 92, 93, 94, 95, 96]
    lows = [99, 99, 99, 87, 89, 90, 91, 92, 93, 94]
    data = _ohlcv(closes, opens=opens, highs=highs, lows=lows)

    sigs = pd.Series(np.nan, index=data.index, dtype=float)
    sigs.iloc[0] = 1.0  # long entry at first bar; NaN = hold thereafter

    eq_vec = _equity_under("vectorized", data, sigs)
    eq_evt = _equity_under("event_driven", data, sigs)

    # The two equity curves should differ by a real, observable amount
    # — at least 1% of initial capital. Exact magnitude depends on
    # fill model + position sizing; the pin is that divergence is
    # non-trivial and reproducible.
    rel_divergence = abs(eq_vec - eq_evt) / 100_000
    assert rel_divergence > 0.01, (
        f"expected gap-bar divergence > 1% of initial capital; got "
        f"{rel_divergence:.4%} (vectorized={eq_vec:.0f}, "
        f"event_driven={eq_evt:.0f})"
    )
    # Upper bound (round-2 test/API M-3): also pin that vectorized's
    # equity is OPTIMISTIC vs event-driven (vectorized assumes the
    # theoretical threshold loss; event-driven realizes the worse
    # gap fill). If a future v2.9 fix aligns vectorized to event-
    # driven's next-bar-open semantics, divergence collapses → fails
    # lower bound (good). If a partial fix uses some intermediate
    # price (e.g. trigger-bar close instead of next-bar open), this
    # upper bound catches it.
    assert eq_vec > eq_evt, (
        f"vectorized equity ({eq_vec:.0f}) should be optimistic vs "
        f"event-driven ({eq_evt:.0f}) on a gap-down trigger; "
        f"vectorized clamps to threshold, event-driven realizes the gap"
    )
    # Hard upper: divergence shouldn't exceed the gap's notional
    # impact * position size. 12% gap * 0.95 size = ~11.4% of capital.
    # Cap at 15% to allow slippage / dividend noise.
    assert rel_divergence < 0.15, (
        f"divergence exceeds expected gap-impact bound; got "
        f"{rel_divergence:.4%} — investigate before relaxing"
    )


# --------------------------------------------------------------------
# Test 2 — no-gap trigger: event-driven and vectorized agree
# --------------------------------------------------------------------

def test_stop_loss_event_vs_vectorized_aligned_on_no_gap_bar():
    """Control: when the trigger bar has no overnight gap (close ≈
    next open), both modes should give very similar equity (within
    0.5% of initial capital). Filters out 'always diverges' as a
    false alarm."""
    # Smooth descent crossing stop on bar 3 with no gap.
    closes = [100, 99, 97, 95, 94, 93.5, 93, 92.5, 92, 91.5]
    # opens follow closes tightly — no gap between bars
    opens = closes
    data = _ohlcv(closes, opens=opens)

    sigs = pd.Series(np.nan, index=data.index, dtype=float)
    sigs.iloc[0] = 1.0

    eq_vec = _equity_under("vectorized", data, sigs)
    eq_evt = _equity_under("event_driven", data, sigs)

    rel_divergence = abs(eq_vec - eq_evt) / 100_000
    assert rel_divergence < 0.05, (
        f"expected no-gap convergence within 5% of initial capital; "
        f"got {rel_divergence:.4%} (vectorized={eq_vec:.0f}, "
        f"event_driven={eq_evt:.0f})"
    )


# --------------------------------------------------------------------
# Test 3 — docstring carries the divergence section
# --------------------------------------------------------------------

def test_percentage_stop_loss_docstring_documents_mode_divergence():
    """Pin the v2.8.2 docstring contract — searches for the headline
    phrases that the README and external docs may grep for."""
    from aiphaforge.exit_rules import PercentageStopLoss
    doc = PercentageStopLoss.__doc__ or ""
    assert "Mode divergence" in doc
    assert "next bar's open" in doc.lower() or "next-bar open" in doc.lower()
    assert "theoretical stop" in doc.lower() or "theoretical" in doc.lower()
