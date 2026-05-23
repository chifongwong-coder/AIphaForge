"""v2.2 M2 tests — vol-scaled tolerance + per-asset auto-picker."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._vol import (
    apply_vol_scale,
    estimate_sigma,
    estimate_sigma_for_continuation,
    estimate_sigma_for_pointintime,
    resolve_method,
)
from aiphaforge.probes.models import ToleranceProfile, VolScalingSpec


def _bdates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _bars_constant(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = _bdates(n)
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price + 1.0] * n,
            "low": [price - 1.0] * n,
            "close": [price] * n,
            "volume": [1e6] * n,
        },
        index=idx,
    )


def _bars_random(n: int, sigma_target: float = 0.01,
                 seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma_target, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    highs = closes * (1.0 + abs(sigma_target) * 0.5)
    lows = closes * (1.0 - abs(sigma_target) * 0.5)
    opens = closes
    return pd.DataFrame(
        {
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1e6] * n,
        },
        index=_bdates(n),
    )


# ---------- VolScalingSpec validation ----------


class TestVolScalingSpec:
    def test_default_construction(self):
        spec = VolScalingSpec()
        assert spec.window == 20
        assert spec.method == "auto"

    def test_rejects_non_monotone(self):
        with pytest.raises(ValueError, match="monotone"):
            VolScalingSpec(near_factor=2.0, rough_factor=1.0)

    def test_rejects_window_lt_2(self):
        with pytest.raises(ValueError, match="window"):
            VolScalingSpec(window=1)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            VolScalingSpec(method="custom_estimator")

    def test_is_frozen(self):
        spec = VolScalingSpec()
        with pytest.raises(Exception):
            spec.window = 30  # noqa


# ---------- resolve_method (per-asset auto-picker) ----------


class TestResolveMethod:
    def test_auto_us_equity_to_garman_klass(self):
        assert resolve_method("auto", "us_equity_price") == "garman_klass"

    def test_auto_us_equity_strict_to_garman_klass(self):
        assert (
            resolve_method("auto", "us_equity_price_strict")
            == "garman_klass"
        )

    def test_auto_crypto_to_parkinson(self):
        assert resolve_method("auto", "crypto_price") == "parkinson"

    def test_auto_futures_to_parkinson(self):
        assert resolve_method("auto", "futures_price") == "parkinson"

    def test_auto_penny_to_stdev(self):
        assert (
            resolve_method("auto", "penny_stock_price") == "stdev_returns"
        )

    def test_auto_unknown_preset_to_stdev(self):
        assert resolve_method("auto", "fake_preset") == "stdev_returns"

    def test_auto_no_preset_to_stdev(self):
        assert resolve_method("auto", None) == "stdev_returns"

    def test_explicit_method_pass_through(self):
        assert resolve_method("parkinson", "us_equity_price") == "parkinson"


# ---------- estimate_sigma — basic sanity ----------


class TestEstimateSigma:
    def test_constant_series_sigma_zero_via_stdev(self):
        bars = _bars_constant(20)
        sigma, prov = estimate_sigma(bars, "stdev_returns")
        assert sigma == 0.0
        assert prov["method_used"] == "stdev_returns"

    def test_parkinson_on_random_series_finite_positive(self):
        bars = _bars_random(50, sigma_target=0.02)
        sigma, _ = estimate_sigma(bars, "parkinson")
        assert sigma > 0
        assert math.isfinite(sigma)

    def test_garman_klass_on_random_series_finite(self):
        bars = _bars_random(50, sigma_target=0.02)
        sigma, _ = estimate_sigma(bars, "garman_klass")
        assert sigma >= 0
        assert math.isfinite(sigma)

    def test_h_eq_l_fallback_with_actual_h_eq_l_bars(self):
        # Build a window where H == L for >50% of bars.
        idx = _bdates(20)
        df = pd.DataFrame(
            {
                "open": [100.0] * 20,
                "high": [100.0] * 11 + [101.0] * 9,  # 11/20 = 55% H==L
                "low": [100.0] * 11 + [99.0] * 9,
                "close": [100.0] * 20,
                "volume": [1e6] * 20,
            },
            index=idx,
        )
        sigma, prov = estimate_sigma(df, "parkinson")
        assert prov["method_used"] == "stdev_returns"
        assert "fallback_reason" in prov
        assert prov["h_eq_l_fraction"] == pytest.approx(0.55, abs=1e-9)
        assert "underestimate_formula" in prov
        # Closed-form Parkinson degeneracy: sigma_hat = sqrt(1-f) sigma_true.
        assert "sqrt(1 - h_eq_l_fraction)" in prov["underestimate_formula"]
        assert "Parkinson (1980)" in prov["underestimate_formula"]
        # At f=0.55, under-estimate = 1 - sqrt(0.45) ≈ 0.3292.
        assert prov["estimated_underestimate_pct"] == pytest.approx(
            1.0 - math.sqrt(0.45), abs=1e-12,
        )

    def test_underestimate_formula_matches_closed_form_at_canonical_f(self):
        """Pin the closed-form 1 - sqrt(1-f) at f in {0.50, 0.70, 0.90}.

        Confirms the formula matches the Parkinson degeneracy derivation
        (NOT the v2.2-era linear `f * 0.30` heuristic that under-predicted
        the magnitude by ~2x at f=0.5 and ~2.5x at f=0.9).
        """
        for h_eq_l_fraction, expected in [
            (0.50, 1.0 - math.sqrt(0.50)),  # ≈ 0.2929
            (0.70, 1.0 - math.sqrt(0.30)),  # ≈ 0.4523
            (0.90, 1.0 - math.sqrt(0.10)),  # ≈ 0.6838
        ]:
            n = 20
            n_eq = int(round(h_eq_l_fraction * n))
            idx = _bdates(n)
            df = pd.DataFrame(
                {
                    "open": [100.0] * n,
                    "high": [100.0] * n_eq + [101.0] * (n - n_eq),
                    "low": [100.0] * n_eq + [99.0] * (n - n_eq),
                    "close": [100.0] * n,
                    "volume": [1e6] * n,
                },
                index=idx,
            )
            _, prov = estimate_sigma(df, "parkinson")
            assert prov["h_eq_l_fraction"] == pytest.approx(
                h_eq_l_fraction, abs=1e-12,
            )
            assert prov["estimated_underestimate_pct"] == pytest.approx(
                expected, abs=1e-12,
            )

    def test_underestimate_concave_strictly_above_linear_for_f_ge_0_5(
        self,
    ):
        """Regression guard: the closed-form is strictly concave and
        exceeds the legacy linear `f * 0.30` at every f in the
        fallback range [0.5, 1.0). Catches an accidental revert to
        the pre-fix linear heuristic.
        """
        for f in [0.50, 0.55, 0.70, 0.85, 0.95]:
            n = 20
            n_eq = int(round(f * n))
            idx = _bdates(n)
            df = pd.DataFrame(
                {
                    "open": [100.0] * n,
                    "high": [100.0] * n_eq + [101.0] * (n - n_eq),
                    "low": [100.0] * n_eq + [99.0] * (n - n_eq),
                    "close": [100.0] * n,
                    "volume": [1e6] * n,
                },
                index=idx,
            )
            _, prov = estimate_sigma(df, "parkinson")
            closed_form = prov["estimated_underestimate_pct"]
            legacy_linear = prov["h_eq_l_fraction"] * 0.30
            assert closed_form > legacy_linear, (
                f"f={f}: closed-form {closed_form:.4f} "
                f"must exceed legacy linear {legacy_linear:.4f}"
            )

    def test_parkinson_event_day_warning_fires(self):
        # 19 normal bars + 1 event-day bar with |return| > 5σ
        rng = np.random.default_rng(0)
        rets = list(rng.normal(0.0, 0.005, 19))
        rets.append(0.05)  # ≈10× the typical σ
        closes = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.005,
                "low": closes * 0.995,
                "close": closes,
                "volume": [1e6] * 20,
            },
            index=_bdates(20),
        )
        sigma, prov = estimate_sigma(df, "parkinson")
        assert "parkinson_event_day_warning" in prov

    def test_parkinson_event_day_warning_silent_on_benign(self):
        bars = _bars_random(50, sigma_target=0.005, seed=1)
        sigma, prov = estimate_sigma(bars, "parkinson")
        assert "parkinson_event_day_warning" not in prov

    def test_overnight_gap_warning_fires(self):
        # Construct a series with a large overnight gap on bar 5.
        n = 20
        closes = np.full(n, 100.0)
        opens = closes.copy()
        opens[5] = 105.0  # 5% gap up
        highs = np.maximum(opens, closes) + 0.5
        lows = np.minimum(opens, closes) - 0.5
        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [1e6] * n,
            },
            index=_bdates(n),
        )
        sigma, prov = estimate_sigma(df, "parkinson")
        # Should trigger overnight-gap warning per F6 follow-up
        assert "parkinson_overnight_gap_warning" in prov


# ---------- causality contracts ----------


class TestCausalityPointInTime:
    def test_only_uses_bars_strictly_before_answer_timestamp(self):
        n = 30
        df = _bars_random(n, sigma_target=0.01, seed=2)
        ts = df.index[15]
        sigma_a, _ = estimate_sigma_for_pointintime(
            df, ts, window=10, method="stdev_returns",
            preset_name=None,
        )
        # Mutate bar AT the answer timestamp itself.
        df_mod = df.copy()
        df_mod.iloc[15, df_mod.columns.get_loc("close")] = 999.0
        sigma_b, _ = estimate_sigma_for_pointintime(
            df_mod, ts, window=10, method="stdev_returns",
            preset_name=None,
        )
        # Pointintime: bar AT ts must NOT influence σ.
        assert sigma_a == pytest.approx(sigma_b, rel=1e-12)

    def test_modifying_bars_after_does_not_change_sigma(self):
        n = 30
        df = _bars_random(n, sigma_target=0.01, seed=3)
        ts = df.index[15]
        sigma_a, _ = estimate_sigma_for_pointintime(
            df, ts, window=10, method="stdev_returns",
            preset_name=None,
        )
        df_mod = df.copy()
        df_mod.iloc[20, df_mod.columns.get_loc("close")] = 999.0
        sigma_b, _ = estimate_sigma_for_pointintime(
            df_mod, ts, window=10, method="stdev_returns",
            preset_name=None,
        )
        assert sigma_a == pytest.approx(sigma_b, rel=1e-12)


class TestCausalityContinuation:
    def test_uses_last_context_bar(self):
        n = 30
        df = _bars_random(n, sigma_target=0.01, seed=4)
        last_ctx = df.index[15]
        sigma_a, _ = estimate_sigma_for_continuation(
            df, last_ctx, window=10, method="stdev_returns",
            preset_name=None,
        )
        # Modifying bars AFTER last_context must not change σ.
        df_mod = df.copy()
        df_mod.iloc[20, df_mod.columns.get_loc("close")] = 999.0
        sigma_b, _ = estimate_sigma_for_continuation(
            df_mod, last_ctx, window=10, method="stdev_returns",
            preset_name=None,
        )
        assert sigma_a == pytest.approx(sigma_b, rel=1e-12)

    def test_modifying_last_context_bar_changes_sigma(self):
        n = 30
        df = _bars_random(n, sigma_target=0.01, seed=5)
        last_ctx = df.index[15]
        sigma_a, _ = estimate_sigma_for_continuation(
            df, last_ctx, window=10, method="stdev_returns",
            preset_name=None,
        )
        df_mod = df.copy()
        df_mod.iloc[15, df_mod.columns.get_loc("close")] = 200.0
        sigma_b, _ = estimate_sigma_for_continuation(
            df_mod, last_ctx, window=10, method="stdev_returns",
            preset_name=None,
        )
        # Continuation: last context bar IS in the σ window.
        assert sigma_a != pytest.approx(sigma_b, rel=1e-3)


# ---------- apply_vol_scale ----------


class TestApplyVolScale:
    def test_returns_unchanged_when_vol_scale_none(self):
        prof = ToleranceProfile.us_equity_price()
        out = apply_vol_scale(prof, sigma=0.01)
        assert out is prof  # identity short-circuit

    def test_scales_bands_by_sigma_factors(self):
        prof = ToleranceProfile(
            absolute_floor=1e-6,
            exact_threshold=0.01, near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.01, near_range_width=0.02,
            rough_range_width=0.05,
            vol_scale=VolScalingSpec(
                near_factor=1.0, rough_factor=2.0, miss_floor_factor=5.0,
            ),
        )
        scaled = apply_vol_scale(prof, sigma=0.02)
        assert scaled.near_threshold == pytest.approx(0.02)  # 1×0.02
        assert scaled.rough_threshold == pytest.approx(0.04)  # 2×0.02
        # vol_scale cleared after application (prevents double-scaling)
        assert scaled.vol_scale is None

    def test_floor_caps_when_sigma_low(self):
        prof = ToleranceProfile(
            absolute_floor=1e-3,
            exact_threshold=0.01, near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.01, near_range_width=0.02,
            rough_range_width=0.05,
            vol_scale=VolScalingSpec(
                near_factor=1.0, rough_factor=2.0, miss_floor_factor=5.0,
            ),
        )
        scaled = apply_vol_scale(prof, sigma=0.0)
        # σ=0 → near = max(0, 1e-3) = 1e-3
        assert scaled.near_threshold == 1e-3
        assert scaled.rough_threshold == 1e-3

    def test_preserves_max_range_width_and_sign_fields(self):
        prof = ToleranceProfile(
            absolute_floor=1e-6,
            exact_threshold=0.01, near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.01, near_range_width=0.02,
            rough_range_width=0.05,
            max_range_width=0.20, sign_sensitive=True,
            sign_epsilon=0.001,
            vol_scale=VolScalingSpec(),
        )
        scaled = apply_vol_scale(prof, sigma=0.01)
        assert scaled.max_range_width == 0.20
        assert scaled.sign_sensitive is True
        assert scaled.sign_epsilon == 0.001


# ---------- preset name wiring ----------


class TestPresetName:
    def test_us_equity_price_carries_preset_name(self):
        p = ToleranceProfile.us_equity_price()
        assert p.preset_name == "us_equity_price"

    def test_crypto_price_carries_preset_name(self):
        p = ToleranceProfile.crypto_price()
        assert p.preset_name == "crypto_price"

    def test_penny_stock_carries_preset_name(self):
        p = ToleranceProfile.penny_stock_price()
        assert p.preset_name == "penny_stock_price"

    def test_resolve_method_uses_preset_for_us_equity(self):
        p = ToleranceProfile.us_equity_price()
        # auto + us_equity_price → garman_klass
        assert resolve_method("auto", p.preset_name) == "garman_klass"
