"""v2.0 M2 — tests for transforms, pipeline, and OHLC integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes.transforms import (
    BlockBootstrap,
    DateShift,
    OHLCJitter,
    PriceRebase,
    PriceScale,
    SymbolMasker,
    TransformPipeline,
    WindowShuffle,
    validate_ohlcv_integrity,
)


def _ohlcv(n: int = 60, seed: int = 0, start: float = 100.0) -> pd.DataFrame:
    """Build a synthetic OHLCV frame with valid invariants."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    closes = start * np.cumprod(1.0 + rets)
    spreads = np.abs(rng.normal(0.0, 0.005, size=n)) * closes
    opens = closes * (1.0 + rng.normal(0.0, 0.003, size=n))
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    vol = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------- SymbolMasker ----------

class TestSymbolMasker:
    def test_bijective_mapping(self):
        sm = SymbolMasker(symbols=["AAPL", "MSFT", "GOOG"], seed=42)
        masked = {sm.mask_symbol(s) for s in ["AAPL", "MSFT", "GOOG"]}
        assert len(masked) == 3
        for s in ["AAPL", "MSFT", "GOOG"]:
            assert sm.unmask_symbol(sm.mask_symbol(s)) == s

    def test_deterministic_per_seed(self):
        sm1 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=42)
        sm2 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=42)
        assert sm1.mask_symbol("AAPL") == sm2.mask_symbol("AAPL")

    def test_different_seed_different_mapping(self):
        # With only 2 symbols and a 20-element pool the chance of two
        # different seeds producing the same mapping is small but
        # nonzero. Try several seeds for a robust assertion.
        sm0 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=0)
        differs = False
        for s in range(1, 20):
            other = SymbolMasker(symbols=["AAPL", "MSFT"], seed=s)
            if other.mask_symbol("AAPL") != sm0.mask_symbol("AAPL"):
                differs = True
                break
        assert differs

    def test_duplicate_symbols_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            SymbolMasker(symbols=["AAPL", "AAPL"], seed=0)

    def test_alias_pool_too_small_raises(self):
        with pytest.raises(ValueError, match="alias pool"):
            SymbolMasker(
                symbols=[f"S{i}" for i in range(50)],
                seed=0,
            )

    def test_unknown_mask_unmask_raises(self):
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        with pytest.raises(KeyError):
            sm.mask_symbol("MSFT")
        with pytest.raises(KeyError):
            sm.unmask_symbol("ZZZZZ")

    def test_apply_is_no_op_on_data(self):
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        data = _ohlcv()
        out = sm.apply(data)
        # Frame content should be byte-identical (no copy required).
        pd.testing.assert_frame_equal(out, data)


# ---------- DateShift ----------

class TestDateShift:
    def test_shift_index_forward(self):
        ds = DateShift(offset=pd.DateOffset(years=3))
        data = _ohlcv()
        out = ds.apply(data)
        assert (out.index == data.index + pd.DateOffset(years=3)).all()

    def test_shift_inverse(self):
        ds = DateShift(offset=pd.DateOffset(years=-5))
        ts = pd.Timestamp("2024-03-12")
        assert ds.unshift_date(ds.shift_date(ts)) == ts

    def test_data_values_unchanged(self):
        ds = DateShift(offset=pd.DateOffset(years=-3))
        data = _ohlcv()
        out = ds.apply(data)
        np.testing.assert_array_equal(out["close"].to_numpy(), data["close"].to_numpy())


# ---------- PriceScale ----------

class TestPriceScale:
    def test_factor_applied_to_ohlc(self):
        ps = PriceScale(factor=2.5)
        data = _ohlcv()
        out = ps.apply(data)
        for col in ("open", "high", "low", "close"):
            np.testing.assert_allclose(
                out[col].to_numpy(), data[col].to_numpy() * 2.5
            )

    def test_volume_unchanged(self):
        ps = PriceScale(factor=0.5)
        data = _ohlcv()
        out = ps.apply(data)
        np.testing.assert_array_equal(
            out["volume"].to_numpy(), data["volume"].to_numpy()
        )

    def test_returns_preserved_exactly(self):
        # PriceScale must preserve returns — that's the documented
        # property (and the documented limitation).
        ps = PriceScale(factor=3.7)
        data = _ohlcv()
        out = ps.apply(data)
        r_in = data["close"].pct_change().dropna()
        r_out = out["close"].pct_change().dropna()
        np.testing.assert_allclose(r_in.to_numpy(), r_out.to_numpy())

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            PriceScale(factor=0.0)
        with pytest.raises(ValueError):
            PriceScale(factor=-1.0)

    def test_invertible(self):
        ps = PriceScale(factor=2.5)
        assert ps.unscale_price(ps.scale_price(100.0)) == pytest.approx(100.0)


# ---------- PriceRebase ----------

class TestPriceRebase:
    def test_first_close_equals_base(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=247.5)
        out = pr.apply(data)
        assert out["close"].iloc[0] == pytest.approx(100.0)

    def test_returns_preserved(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=247.5)
        out = pr.apply(data)
        r_in = data["close"].pct_change().dropna()
        r_out = out["close"].pct_change().dropna()
        np.testing.assert_allclose(r_in.to_numpy(), r_out.to_numpy())

    def test_last_factor_recorded(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=200.0)
        pr.apply(data)
        # Initial close ~200 → factor ~0.5.
        assert pr.last_factor is not None
        assert pr.last_factor == pytest.approx(100.0 / data["close"].iloc[0])

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            PriceRebase(base=0.0)


# ---------- OHLCJitter ----------

class TestOHLCJitter:
    def test_deterministic_per_seed(self):
        j = OHLCJitter(bps=20.0)
        data = _ohlcv()
        a = j.apply(data, seed=42)
        b = j.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_output(self):
        j = OHLCJitter(bps=20.0)
        data = _ohlcv()
        a = j.apply(data, seed=42)
        b = j.apply(data, seed=43)
        # At 20bps with 60 bars, identical output is essentially zero
        # probability.
        assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_preserves_ohlc_invariants(self):
        j = OHLCJitter(bps=200.0)  # extreme noise to stress invariants
        data = _ohlcv()
        out = j.apply(data, seed=0)
        result = validate_ohlcv_integrity(out)
        assert result.passed, result.errors

    def test_zero_bps_is_identity(self):
        j = OHLCJitter(bps=0.0)
        data = _ohlcv()
        out = j.apply(data, seed=0)
        np.testing.assert_allclose(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_negative_bps_raises(self):
        with pytest.raises(ValueError):
            OHLCJitter(bps=-1.0)


# ---------- BlockBootstrap ----------

class TestBlockBootstrap:
    def test_deterministic_per_seed(self):
        bb = BlockBootstrap(block_size=10)
        data = _ohlcv(n=100)
        a = bb.apply(data, seed=42)
        b = bb.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_anchored_at_source_first_close(self):
        # Anchoring property: cumulative product is anchored at
        # data.iloc[0]['close']. The output first close equals the
        # anchor times the first sampled bar's return — so realizations
        # cluster near the source first close, not exactly at it.
        bb = BlockBootstrap(block_size=20)
        data = _ohlcv(n=200, start=100.0)
        first_closes = [bb.apply(data, seed=s)["close"].iloc[0] for s in range(20)]
        # Average across realizations should be close to the anchor
        # (since average return is ~1 for a near-zero-drift series).
        assert np.mean(first_closes) == pytest.approx(100.0, rel=0.05)

    def test_index_preserved(self):
        bb = BlockBootstrap(block_size=20)
        data = _ohlcv(n=100)
        out = bb.apply(data, seed=0)
        assert (out.index == data.index).all()

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BlockBootstrap(block_size=0)

    def test_ohlc_integrity_holds(self):
        bb = BlockBootstrap(block_size=15)
        data = _ohlcv(n=100)
        for s in range(5):
            out = bb.apply(data, seed=s)
            result = validate_ohlcv_integrity(out)
            assert result.passed, (s, result.errors)


# ---------- WindowShuffle ----------

class TestWindowShuffle:
    def test_deterministic_per_seed(self):
        ws = WindowShuffle(window=10, swaps=3)
        data = _ohlcv(n=100)
        a = ws.apply(data, seed=42)
        b = ws.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_index_preserved(self):
        ws = WindowShuffle(window=10, swaps=3)
        data = _ohlcv(n=100)
        out = ws.apply(data, seed=0)
        assert (out.index == data.index).all()

    def test_zero_swaps_is_identity(self):
        ws = WindowShuffle(window=10, swaps=0)
        data = _ohlcv(n=100)
        out = ws.apply(data, seed=0)
        np.testing.assert_array_equal(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_short_series_returns_copy(self):
        ws = WindowShuffle(window=100, swaps=5)
        data = _ohlcv(n=50)
        out = ws.apply(data, seed=0)
        np.testing.assert_array_equal(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            WindowShuffle(window=0)
        with pytest.raises(ValueError):
            WindowShuffle(window=10, swaps=-1)


# ---------- OHLC integrity validator ----------

class TestValidator:
    def test_clean_frame_passes(self):
        result = validate_ohlcv_integrity(_ohlcv())
        assert result.passed
        assert result.errors == []

    def test_missing_columns_fail(self):
        data = _ohlcv().drop(columns=["volume"])
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("missing" in e for e in result.errors)

    def test_nan_detected(self):
        data = _ohlcv()
        data.loc[data.index[5], "close"] = np.nan
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("NaN" in e for e in result.errors)

    def test_negative_volume_detected(self):
        data = _ohlcv()
        data.loc[data.index[0], "volume"] = -1.0
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("negative" in e for e in result.errors)

    def test_ohlc_violation_detected(self):
        data = _ohlcv()
        # Force high < close.
        data.loc[data.index[0], "high"] = data.loc[data.index[0], "close"] - 1.0
        result = validate_ohlcv_integrity(data)
        assert not result.passed

    def test_non_monotonic_index_fails(self):
        data = _ohlcv()
        data = data.iloc[[0, 2, 1] + list(range(3, len(data)))]
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("monotonic" in e for e in result.errors)


# ---------- Pipeline ----------

class TestTransformPipeline:
    def test_stage_order_canonical(self):
        # Ordering: metadata -> level -> series, regardless of user order.
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        ps = PriceScale(factor=2.0)
        bb = BlockBootstrap(block_size=10)
        # User order: series first, metadata last — pipeline should
        # reorder.
        pipeline = TransformPipeline(
            transforms=[bb, ps, sm], mode="market_level"
        )
        ordered = pipeline._stage_sorted()
        assert [t.category for t in ordered] == ["metadata", "level", "series"]

    def test_view_only_rejects_non_invertible_without_contract(self):
        # All built-ins that support view_only are invertible. Use a
        # synthetic stub: view-only-capable + non-invertible.
        class _ViewOnlyNonInvertible:
            name = "VONI"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = False
            stochastic = False

            def apply(self, data, *, seed=None):
                return data.copy()

        with pytest.raises(ValueError, match="non-invertible"):
            TransformPipeline(
                transforms=[_ViewOnlyNonInvertible()], mode="view_only"
            )

    def test_view_only_allows_non_invertible_with_signal_only(self):
        class _ViewOnlyNonInvertible:
            name = "VONI"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = False
            stochastic = False

            def apply(self, data, *, seed=None):
                return data.copy()

        # With agent_contract=signal_only, the non-invertible transform
        # is admitted in view_only mode.
        pipeline = TransformPipeline(
            transforms=[_ViewOnlyNonInvertible()],
            mode="view_only",
            agent_contract="signal_only",
        )
        assert pipeline.agent_contract == "signal_only"

    def test_market_level_rejects_view_only_only_transform(self):
        # All built-ins support market_level, so this is checked by
        # passing a fake transform with supports_market_level=False.
        class _ViewOnlyOnly:
            name = "ViewOnlyStub"
            category = "metadata"
            supports_view_only = True
            supports_market_level = False
            order_invertible = True
            stochastic = False

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(ValueError, match="does not support market_level"):
            TransformPipeline(
                transforms=[_ViewOnlyOnly()], mode="market_level"
            )

    def test_apply_runs_validator(self):
        # OHLCJitter at extreme bps can produce invalid bars without
        # the preserve_ohlc clamp; with clamp on (default) the
        # pipeline must produce valid output.
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[OHLCJitter(bps=200.0)], mode="market_level"
        )
        out = pipeline.apply(data, seed=0)
        result = validate_ohlcv_integrity(out)
        assert result.passed

    def test_apply_deterministic_per_outer_seed(self):
        # Stochastic sub-seeds derived from outer seed via SeedSequence.
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[BlockBootstrap(block_size=10)], mode="market_level"
        )
        a = pipeline.apply(data, seed=42)
        b = pipeline.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_apply_different_outer_seed_different_output(self):
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[BlockBootstrap(block_size=10)], mode="market_level"
        )
        a = pipeline.apply(data, seed=42)
        b = pipeline.apply(data, seed=43)
        assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_stochastic_pipeline_flag(self):
        bb = BlockBootstrap(block_size=10)
        ps = PriceScale(factor=2.0)
        assert TransformPipeline([bb], mode="market_level").stochastic
        assert not TransformPipeline([ps], mode="market_level").stochastic

    def test_invalid_transform_raises(self):
        class _Bad:
            name = "Bad"
            # missing other attrs

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(ValueError, match="missing required attribute"):
            TransformPipeline([_Bad()], mode="market_level")
