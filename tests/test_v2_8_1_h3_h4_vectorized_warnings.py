"""v2.8.1 H3 + H4 — _VECTORIZED_UNSUPPORTED_FIELDS expansion + warnings.

H3: v2.8.0 enumerated only 7 silently-dropped fields. The post-v2.8 audit
surfaced 14 more (fill_model, max_position_size, capital_allocator, the
multi-asset *_ dicts, lot_size, etc.). Vectorized mode silently dropped
all of them, producing misleading "I set X but nothing changed" results.

H4: capital_allocator in vectorized multi-asset is the highest-leverage
silent divergence — vectorized uses static equal-weight; event-driven
uses the user's allocator. The H4 message names the divergence explicitly
("static equal-weight ... is ignored").

FR-G2 (v2.1.3): max_position_size is partially honored in vectorized
(used for cost-estimation sizing via the position_sizer per v2.8.1 H1).
The message says "cost-estimation sizing", not "ignored".
"""
from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine, SimpleFeeModel, ZeroFeeModel


def _data(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )


def _signals(n: int = 20) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    s = pd.Series(np.nan, index=idx, dtype=float)
    s.iloc[5] = 1.0
    return s


def _capture_warnings(**engine_kwargs) -> list[str]:
    """Construct + run a vectorized engine; return warning message strings."""
    eng = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        include_benchmark=False, **engine_kwargs,
    )
    eng.set_signals(_signals())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.run(_data())
    return [str(w.message) for w in caught]


# --------------------------------------------------------------------
# Test 1 — parametrized: each newly-added field fires its warning
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwarg,value,expected_substring",
    [
        ("fill_model", "next_bar_close", "fill_model"),
        ("session_end_time", datetime.time(16, 0), "session_end_time"),
        ("immediate_fill_price", "open", "immediate_fill_price"),
        ("fee_allocation", "pro_rata", "fee_allocation"),
        ("asset_fee_models", {"AAPL": SimpleFeeModel()}, "asset_fee_models"),
        ("asset_fill_models", {"AAPL": "next_bar_close"}, "asset_fill_models"),
        ("asset_max_position_pcts", {"AAPL": 0.5}, "asset_max_position_pcts"),
        ("asset_lot_sizes", {"AAPL": 100}, "asset_lot_sizes"),
        ("lot_size", 100, "lot_size"),
        ("max_position_pct", 0.5, "max_position_pct"),
        ("portfolio_exit_rules", [object()], "portfolio_exit_rules"),
        ("asset_margin_configs", {"AAPL": object()}, "asset_margin_configs"),
    ],
)
def test_vectorized_warning_fires_for_each_newly_added_field(
    kwarg, value, expected_substring,
):
    """Each of the 12 generic-warning newly-added fields must fire a
    warning naming itself when set in vectorized mode."""
    msgs = _capture_warnings(**{kwarg: value})
    assert any(expected_substring in m for m in msgs), (
        f"No warning naming {expected_substring} in {msgs}"
    )


# --------------------------------------------------------------------
# Test 2 — capital_allocator emits the divergence-specific text
# --------------------------------------------------------------------

class _DummyAllocator:
    """Minimal stand-in for a capital allocator."""


def test_capital_allocator_warning_names_static_equal_weight_and_event_driven():
    msgs = _capture_warnings(capital_allocator=_DummyAllocator())
    text = " ".join(msgs)
    assert "static equal-weight" in text
    assert "event_driven" in text
    assert "_DummyAllocator" in text


# --------------------------------------------------------------------
# Test 3 — max_position_size emits the partial-honoring text (FR-G2)
# --------------------------------------------------------------------

def test_max_position_size_warning_describes_partial_honoring():
    msgs = _capture_warnings(max_position_size=0.5)
    # The partial-honor message:
    matching = [m for m in msgs if "max_position_size" in m
                or "cost-estimation" in m]
    assert any("cost-estimation sizing" in m for m in matching), (
        f"max_position_size warning lacks partial-honor wording: {msgs!r}"
    )
    assert any("event_driven" in m for m in matching)
    # Must NOT use the misleading "ignored" wording:
    for m in matching:
        assert "is ignored" not in m, (
            f"max_position_size message uses 'is ignored' incorrectly: {m!r}"
        )


# --------------------------------------------------------------------
# Test 4 — fields the engine honors emit NO warning
# --------------------------------------------------------------------

def test_vectorized_no_warning_for_supported_fields():
    """Setting stop_loss, allow_short, representative_notional, or
    signal_transform must NOT trigger _warn_vectorized_unsupported
    warnings — these are honored by the vectorized core."""
    msgs = _capture_warnings(
        stop_loss=0.05,
        allow_short=False,
        representative_notional=12_345.0,
    )
    for bad_field in ("stop_loss", "allow_short",
                      "representative_notional", "signal_transform"):
        for m in msgs:
            assert not (bad_field in m and "is ignored" in m), (
                f"unexpected unsupported-field warning for {bad_field}: {m!r}"
            )
