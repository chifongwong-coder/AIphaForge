"""v2.2 M5 tests — held-out synthetic anchor."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes.anchors import build_synthetic_anchor


def _make_real_data(
    n: int = 100,
    sigma: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------- Method dispatch ----------


class TestBuildSyntheticAnchor:
    def test_random_walk_volmatched_runs(self):
        real = _make_real_data(n=100)
        anchor = build_synthetic_anchor(
            real, seed=42, method="random_walk_volmatched",
        )
        assert anchor.shape[0] == 100
        assert set(anchor.columns) == set(real.columns)

    def test_block_bootstrap_runs(self):
        real = _make_real_data(n=100)
        anchor = build_synthetic_anchor(
            real, seed=42, method="block_bootstrap",
        )
        assert anchor.shape[0] == 100

    def test_garch_resample_runs(self):
        real = _make_real_data(n=100)
        # n=100 < GJR_MIN_BARS=500 → should hard-fall-back to GARCH
        anchor = build_synthetic_anchor(
            real, seed=42, method="garch_resample",
        )
        assert anchor.shape[0] == 100
        # vol_model_chosen should be "garch" (not "gjr_garch") because
        # n<500 forced the gate to symmetric GARCH.
        chosen = anchor.attrs["vol_model_chosen"]
        assert chosen in {
            "garch", "random_walk_volmatched_fallback"
        }

    def test_rejects_unknown_method(self):
        real = _make_real_data(n=100)
        with pytest.raises(ValueError, match="unknown method"):
            build_synthetic_anchor(real, seed=1, method="bogus")

    def test_rejects_unknown_asset_class(self):
        real = _make_real_data(n=100)
        with pytest.raises(ValueError, match="asset_class"):
            build_synthetic_anchor(
                real, seed=1, asset_class="weird",
            )

    def test_rejects_too_short_real_data(self):
        real = _make_real_data(n=2)
        with pytest.raises(ValueError, match="3 bars"):
            build_synthetic_anchor(real, seed=1)


# ---------- Determinism ----------


class TestDeterminism:
    def test_same_seed_same_output(self):
        real = _make_real_data(n=50)
        a1 = build_synthetic_anchor(
            real, seed=99, method="random_walk_volmatched",
        )
        a2 = build_synthetic_anchor(
            real, seed=99, method="random_walk_volmatched",
        )
        np.testing.assert_array_equal(
            a1["close"].to_numpy(), a2["close"].to_numpy()
        )

    def test_different_seed_different_output(self):
        real = _make_real_data(n=50)
        a1 = build_synthetic_anchor(
            real, seed=1, method="random_walk_volmatched",
        )
        a2 = build_synthetic_anchor(
            real, seed=2, method="random_walk_volmatched",
        )
        assert not np.allclose(
            a1["close"].to_numpy(), a2["close"].to_numpy()
        )


# ---------- Label safety ----------


class TestLabel:
    def test_label_does_not_leak_real_ticker(self):
        real = _make_real_data(n=50)
        anchor = build_synthetic_anchor(
            real, seed=42, method="random_walk_volmatched",
        )
        label = anchor.attrs["synthetic_label"]
        # Should never contain "AAPL", "TSLA", etc.
        for forbidden in ("AAPL", "TSLA", "BRK", "GOOG", "MSFT"):
            assert forbidden not in label
        assert label.startswith("SYN_")

    def test_label_deterministic_from_seed(self):
        real = _make_real_data(n=50)
        a1 = build_synthetic_anchor(
            real, seed=7, method="random_walk_volmatched",
        )
        a2 = build_synthetic_anchor(
            real, seed=7, method="random_walk_volmatched",
        )
        assert a1.attrs["synthetic_label"] == a2.attrs["synthetic_label"]


# ---------- σ-match (loose check) ----------


class TestSigmaMatch:
    def test_random_walk_volmatched_sigma_within_band(self):
        real = _make_real_data(n=200, sigma=0.01)
        anchor = build_synthetic_anchor(
            real, seed=11, method="random_walk_volmatched",
        )
        real_rets = np.diff(np.log(real["close"].to_numpy()))
        anch_rets = np.diff(np.log(anchor["close"].to_numpy()))
        ratio = float(np.std(anch_rets, ddof=1)) / float(
            np.std(real_rets, ddof=1)
        )
        # 200 bars sample SE on σ ≈ 5%; 0.85–1.15 is comfortable.
        assert 0.85 <= ratio <= 1.15


# ---------- Equity-vs-crypto auto-detector ----------


class TestEquityCryptoAutoDetector:
    def test_random_normal_is_crypto_classified(self):
        # No leverage effect → crypto classification
        real = _make_real_data(n=600, sigma=0.01, seed=0)
        anchor = build_synthetic_anchor(
            real, seed=0, asset_class="auto",
            method="random_walk_volmatched",
        )
        prov = anchor.attrs["leverage_corr_provenance"]
        assert prov["is_equity"] is False  # iid normal has no leverage

    def test_records_block_size_and_n_resamples(self):
        real = _make_real_data(n=600, sigma=0.01, seed=1)
        anchor = build_synthetic_anchor(
            real, seed=1, asset_class="auto",
            method="random_walk_volmatched",
        )
        prov = anchor.attrs["leverage_corr_provenance"]
        assert prov["block_size_used"] == 20
        assert prov["n_resamples_used"] == 500

    def test_records_bootstrap_se(self):
        real = _make_real_data(n=600, sigma=0.01, seed=2)
        anchor = build_synthetic_anchor(
            real, seed=2, asset_class="auto",
            method="random_walk_volmatched",
        )
        prov = anchor.attrs["leverage_corr_provenance"]
        assert "bootstrap_se" in prov
        assert math.isfinite(prov["bootstrap_se"])

    def test_explicit_equity_skips_detector(self):
        real = _make_real_data(n=600, sigma=0.01, seed=3)
        anchor = build_synthetic_anchor(
            real, seed=3, asset_class="equity",
            method="random_walk_volmatched",
        )
        # Explicit asset_class → no leverage_corr_provenance
        assert "leverage_corr_provenance" not in anchor.attrs


# ---------- GJR gate ----------


class TestGjrGate:
    def test_n_lt_500_does_not_attempt_gjr(self):
        real = _make_real_data(n=200, seed=4)
        anchor = build_synthetic_anchor(
            real, seed=4, asset_class="equity",
            method="garch_resample",
        )
        chosen = anchor.attrs["vol_model_chosen"]
        # Below threshold → must NOT be gjr_garch
        assert chosen != "gjr_garch"

    def test_records_gjr_gate_note_when_below_threshold(self):
        real = _make_real_data(n=200, seed=5)
        anchor = build_synthetic_anchor(
            real, seed=5, asset_class="equity",
            method="garch_resample",
        )
        prov = anchor.attrs["vol_model_provenance"]
        assert "gjr_gate_note" in prov
        assert "500" in prov["gjr_gate_note"]


# ---------- Output shape ----------


class TestOutputShape:
    def test_anchor_index_matches_real_index(self):
        real = _make_real_data(n=80, seed=6)
        anchor = build_synthetic_anchor(
            real, seed=6, method="random_walk_volmatched",
        )
        pd.testing.assert_index_equal(anchor.index, real.index)

    def test_anchor_columns_match_real_columns(self):
        real = _make_real_data(n=80, seed=7)
        anchor = build_synthetic_anchor(
            real, seed=7, method="random_walk_volmatched",
        )
        assert set(anchor.columns) == set(real.columns)
