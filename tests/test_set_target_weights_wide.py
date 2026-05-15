"""v2.7 Commit C — set_target_weights_wide(weights, snap=...) entry point.

Covers the 13 tests required by v2.7 plan §"Commit C":

- basic 3-symbol equivalence with target_weight_wide_to_schedule baseline
- snap='next' + snap='exact' drop behavior
- universe_alignment='intersection'
- on_collision='raise'
- non-DataFrame rejection
- duplicate-index validation
- NaN weight → explicit close
- strict promotes-collision-to-error
- frozen dataclass invariant
- strict + on_collision='warn' raises
- strict + snap='next' raises
- engine-layer enriched-warning text mentions 'raise'
"""
from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine, _TargetWeightsWideConfig


def _bdays(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _ohlcv(idx: pd.DatetimeIndex, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 1, len(idx)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": 1_000_000.0,
        },
        index=idx,
    )


def _multi_data(symbols=("AAPL", "MSFT", "TSLA"), n: int = 30):
    idx = _bdays(n)
    return {
        sym: _ohlcv(idx, base=100.0 + 10.0 * i, seed=42 + i)
        for i, sym in enumerate(symbols)
    }


# ---------------------------------------------------------------------------
# Conversion plumbing
# ---------------------------------------------------------------------------


def test_set_target_weights_wide_basic_3_symbols():
    # Equivalence with the manual two-step pattern.
    from aiphaforge.signals import target_weight_wide_to_schedule
    data = _multi_data()
    idx = next(iter(data.values())).index
    weights = pd.DataFrame(
        {sym: [1.0 / len(data)] * len(idx) for sym in data},
        index=idx,
    )

    baseline_engine = BacktestEngine(initial_capital=100_000.0)
    schedule = target_weight_wide_to_schedule(weights, data, snap="exact")
    baseline_engine.set_target_weights(schedule)
    baseline = baseline_engine.run(data)

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(weights)
    result = engine.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)


def test_set_target_weights_wide_snap_next():
    # Add a date NOT in data's index → snap='next' rolls forward.
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    # Insert a non-business-day timestamp in the wide DF index.
    extra_idx = idx.tolist() + [pd.Timestamp("2024-01-13")]  # Saturday
    extra_idx = pd.DatetimeIndex(sorted(extra_idx))
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(extra_idx), "MSFT": [0.5] * len(extra_idx)},
        index=extra_idx,
    )
    engine = BacktestEngine(initial_capital=100_000.0)
    # Should NOT raise; snap='next' absorbs the misaligned date.
    engine.set_target_weights_wide(weights, snap="next")
    result = engine.run(data)
    assert result is not None


def test_set_target_weights_wide_snap_exact_drops_non_aligned():
    # snap='exact' default emits a UserWarning when dates are dropped.
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    extra_idx = idx.tolist() + [pd.Timestamp("2024-01-13")]
    extra_idx = pd.DatetimeIndex(sorted(extra_idx))
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(extra_idx), "MSFT": [0.5] * len(extra_idx)},
        index=extra_idx,
    )
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(weights)  # snap='exact' default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        engine.run(data)
    assert any(
        "dropped" in str(w.message) for w in caught
    ), f"expected 'dropped' warning; got: {[str(w.message) for w in caught]}"


def test_set_target_weights_wide_universe_intersection():
    # Two symbols with staggered listing dates → intersection only fires
    # rebalance on dates present in BOTH.
    idx_a = pd.bdate_range("2024-01-01", periods=20)
    idx_b = pd.bdate_range("2024-01-08", periods=20)  # later listing
    data = {"AAPL": _ohlcv(idx_a), "MSFT": _ohlcv(idx_b, base=200)}
    full_idx = idx_a.union(idx_b)
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(full_idx), "MSFT": [0.5] * len(full_idx)},
        index=full_idx,
    )
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(weights, universe_alignment="intersection")
    # Should not crash; intersection narrows the rebalance set.
    result = engine.run(data)
    assert result is not None


def test_set_target_weights_wide_on_collision_raise():
    # snap='next' on a Saturday + a Sunday in the same week both snap to
    # the same Monday → collision. on_collision='raise' surfaces.
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    extra_idx = (
        idx.tolist()
        + [pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-14")]
    )
    extra_idx = pd.DatetimeIndex(sorted(extra_idx))
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(extra_idx), "MSFT": [0.5] * len(extra_idx)},
        index=extra_idx,
    )
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(
        weights, snap="next", on_collision="raise",
    )
    with pytest.raises(ValueError, match="collision"):
        engine.run(data)


def test_set_target_weights_wide_rejects_non_dataframe():
    engine = BacktestEngine()
    with pytest.raises(TypeError, match="set_target_weights_wide expects pd.DataFrame"):
        engine.set_target_weights_wide(pd.Series([0.5]))
    with pytest.raises(TypeError, match="set_target_weights_wide expects pd.DataFrame"):
        engine.set_target_weights_wide({"AAPL": pd.Series([0.5])})
    with pytest.raises(TypeError, match="set_target_weights_wide expects pd.DataFrame"):
        engine.set_target_weights_wide(None)


def test_set_target_weights_wide_validates_duplicate_index_raises():
    idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
    )
    weights = pd.DataFrame({"AAPL": [0.5, 0.5, 0.5, 0.5]}, index=idx)
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.set_target_weights_wide(weights)


def test_set_target_weights_wide_NaN_weight_means_explicit_close():
    # Per v2-decision #14: NaN weight in wide DF is coerced to 0 by the
    # underlying schedule adapter — explicit close on that bar.
    from aiphaforge.signals import target_weight_wide_to_schedule
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(idx), "MSFT": [0.5] * len(idx)},
        index=idx,
    )
    weights.iloc[5, 0] = np.nan  # AAPL weight goes NaN at bar 5
    schedule = target_weight_wide_to_schedule(weights, data)
    aapl_weight_at_bar5 = schedule[idx[5]].get("AAPL", 0.0)
    assert aapl_weight_at_bar5 == 0.0, (
        "NaN weight in wide DF must coerce to 0 (explicit close) "
        "via the schedule adapter contract"
    )


# ---------------------------------------------------------------------------
# strict semantics (v3-decision #20)
# ---------------------------------------------------------------------------


def test_set_target_weights_wide_strict_default_resolution():
    # strict=True with no explicit on_collision / snap defaults to
    # ('raise', 'exact'). Verify via the stored frozen config.
    #
    # Note: collisions under strict=True are physically impossible
    # because strict requires snap='exact', and exact mode never snaps,
    # so the "raise" default never fires in practice. This test pins
    # the default-resolution logic itself, not a runtime trigger.
    idx = _bdays()
    weights = pd.DataFrame({"AAPL": [0.5] * len(idx)}, index=idx)
    engine = BacktestEngine()
    engine.set_target_weights_wide(weights, strict=True)
    cfg = engine._target_weights_wide_config
    assert cfg.snap == "exact", (
        "strict=True must default snap to 'exact' when not explicitly passed"
    )
    assert cfg.on_collision == "raise", (
        "strict=True must default on_collision to 'raise' when not explicitly passed"
    )
    assert cfg.strict is True


def test_set_target_weights_wide_strict_with_on_collision_warn_raises():
    idx = _bdays()
    weights = pd.DataFrame({"AAPL": [0.5] * len(idx)}, index=idx)
    engine = BacktestEngine()
    with pytest.raises(
        ValueError,
        match="strict=True.*incompatible.*on_collision='warn'",
    ):
        engine.set_target_weights_wide(weights, strict=True, on_collision="warn")


def test_set_target_weights_wide_strict_with_snap_next_raises():
    idx = _bdays()
    weights = pd.DataFrame({"AAPL": [0.5] * len(idx)}, index=idx)
    engine = BacktestEngine()
    with pytest.raises(
        ValueError,
        match="strict=True.*incompatible.*snap='next'",
    ):
        engine.set_target_weights_wide(weights, strict=True, snap="next")


# ---------------------------------------------------------------------------
# Frozen dataclass + engine-layer enriched warning
# ---------------------------------------------------------------------------


def test_target_weights_wide_config_is_frozen():
    cfg = _TargetWeightsWideConfig(
        rebalance_dates=None, snap="exact", universe_alignment="union",
        on_collision="warn", strict=False,
    )
    # Frozen: assignment raises FrozenInstanceError.
    with pytest.raises(FrozenInstanceError):
        cfg.snap = "next"
    # Expected 5 fields.
    field_names = {f.name for f in fields(cfg)}
    assert field_names == {
        "rebalance_dates", "snap", "universe_alignment",
        "on_collision", "strict",
    }


def test_set_target_weights_wide_strict_warning_text_mentions_raise():
    # Engine-layer enriched warning fires when on_collision='warn' AND
    # the helper emitted a collision warning. Per plan v3.2, detection
    # uses substring match on "collision" in the captured message.
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    extra_idx = (
        idx.tolist()
        + [pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-14")]
    )
    extra_idx = pd.DatetimeIndex(sorted(extra_idx))
    weights = pd.DataFrame(
        {"AAPL": [0.5] * len(extra_idx), "MSFT": [0.5] * len(extra_idx)},
        index=extra_idx,
    )
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(
        weights, snap="next", on_collision="warn",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        engine.run(data)
    # Two warnings expected: helper's collision warning AND engine's
    # enriched "pass on_collision='raise'" warning.
    engine_msgs = [
        str(w.message) for w in caught
        if "on_collision='raise'" in str(w.message)
        or "strict=True" in str(w.message)
    ]
    assert engine_msgs, (
        f"expected engine-layer enriched warning suggesting "
        f"on_collision='raise' / strict=True; got messages: "
        f"{[str(w.message) for w in caught]}"
    )
