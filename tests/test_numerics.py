"""Numerics tests (v1.9.6, Q2 + Q4 + days_per_year).

* BorrowingCostModel — bar-frequency-aware via bar_seconds; calendar
  exposed via days_per_year (default 365). Hourly bars no longer
  charge a full day's rate.
* FundingRateModel — ignores bar_seconds (rate is already per-bar).
* Impact models — early-return 0 on negative or zero order_size.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from aiphaforge.margin import (
    BorrowingCostModel,
    FundingRateModel,
    MarginConfig,
)
from aiphaforge.market_impact import (
    LinearImpactModel,
    PowerLawImpactModel,
    SquareRootImpactModel,
)


@dataclass
class _StubPosition:
    """Minimal Position duck-type for the borrowing cost path."""
    is_short: bool
    is_flat: bool
    market_value: float
    size: float
    avg_entry_price: float

    @property
    def notional_value(self) -> float:
        return abs(self.size * self.avg_entry_price)


def _long_position(notional: float = 100_000.0) -> _StubPosition:
    return _StubPosition(
        is_short=False, is_flat=False,
        market_value=notional, size=1000.0,
        avg_entry_price=notional / 1000.0,
    )


def _short_position(notional: float = 100_000.0) -> _StubPosition:
    return _StubPosition(
        is_short=True, is_flat=False,
        market_value=notional, size=-1000.0,
        avg_entry_price=notional / 1000.0,
    )


def _margin_config(rate: float = 0.05, imr: float = 0.5) -> MarginConfig:
    return MarginConfig(
        initial_margin_ratio=imr,
        maintenance_margin_ratio=imr * 0.6,
        borrowing_rate=rate,
    )


# ---------------------------------------------------------------------------
# Q2 — borrowing cost time-aware
# ---------------------------------------------------------------------------

class TestBorrowingCostBarSeconds:
    def test_hourly_matches_daily_div_24(self):
        """1 hour cost = 1 day cost / 24 (within float tolerance)."""
        model = BorrowingCostModel()
        mc = _margin_config(rate=0.10)
        ts = pd.Timestamp("2024-01-02 09:30")
        pos = _short_position()  # short -> borrow on market_value = 100k

        cost_hourly = model.calculate_cost(
            pos, price=100.0, timestamp=ts, margin_config=mc,
            bar_seconds=3600.0)
        cost_daily = model.calculate_cost(
            pos, price=100.0, timestamp=ts, margin_config=mc,
            bar_seconds=86400.0)

        assert cost_daily > 0
        assert abs(cost_hourly - cost_daily / 24) < 1e-9

    def test_friday_to_monday_accrues_three_days(self):
        """Mon-after-Fri bar accrues 3 calendar days (handled by engine)."""
        model = BorrowingCostModel()
        mc = _margin_config(rate=0.10)
        pos = _short_position()
        # Engine passes seconds; model just multiplies linearly.
        three_days_secs = 3 * 86400.0
        one_day_secs = 86400.0

        cost_3d = model.calculate_cost(
            pos, price=100.0, timestamp=pd.Timestamp("2024-01-08"),
            margin_config=mc, bar_seconds=three_days_secs)
        cost_1d = model.calculate_cost(
            pos, price=100.0, timestamp=pd.Timestamp("2024-01-05"),
            margin_config=mc, bar_seconds=one_day_secs)

        assert abs(cost_3d - cost_1d * 3) < 1e-9

    def test_days_per_year_252_vs_365(self):
        """A-share calendar (252) charges (365/252)x more per second."""
        model_365 = BorrowingCostModel(days_per_year=365)
        model_252 = BorrowingCostModel(days_per_year=252)
        mc = _margin_config(rate=0.10)
        pos = _short_position()
        ts = pd.Timestamp("2024-01-02")

        cost_365 = model_365.calculate_cost(
            pos, price=100.0, timestamp=ts, margin_config=mc,
            bar_seconds=86400.0)
        cost_252 = model_252.calculate_cost(
            pos, price=100.0, timestamp=ts, margin_config=mc,
            bar_seconds=86400.0)

        assert cost_365 > 0
        assert abs(cost_252 / cost_365 - 365 / 252) < 1e-6

    def test_legacy_no_bar_seconds_warns_once(self):
        """Direct calls without bar_seconds emit a one-time DeprecationWarning."""
        # Reset the per-class dedup so the warning is observable.
        BorrowingCostModel._legacy_warned = False
        model = BorrowingCostModel()
        mc = _margin_config(rate=0.10)
        pos = _short_position()
        ts = pd.Timestamp("2024-01-02")

        with pytest.warns(DeprecationWarning, match="bar_seconds"):
            model.calculate_cost(pos, 100.0, ts, mc)
        # Second call: no warning (dedup'd)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.calculate_cost(pos, 100.0, ts, mc)


class TestFundingRateIgnoresBarSeconds:
    def test_funding_rate_unchanged_by_bar_seconds(self):
        model = FundingRateModel(funding_rate_per_bar=0.0001)
        pos = _long_position(notional=50_000.0)
        ts = pd.Timestamp("2024-01-02")

        cost_default = model.calculate_cost(
            pos, 50.0, ts, _margin_config(), bar_seconds=None)
        cost_one_hour = model.calculate_cost(
            pos, 50.0, ts, _margin_config(), bar_seconds=3600.0)
        cost_one_day = model.calculate_cost(
            pos, 50.0, ts, _margin_config(), bar_seconds=86400.0)

        assert cost_default == cost_one_hour == cost_one_day == 50_000 * 0.0001


# ---------------------------------------------------------------------------
# Q4 — impact models guard against negative/zero size
# ---------------------------------------------------------------------------

class TestImpactGuards:
    @pytest.mark.parametrize("model", [
        LinearImpactModel(),
        SquareRootImpactModel(),
        PowerLawImpactModel(),
    ])
    def test_zero_size_returns_zero(self, model):
        assert model.estimate_impact(0.0, 100.0, 1_000_000.0, 0.02) == 0.0

    @pytest.mark.parametrize("model", [
        LinearImpactModel(),
        SquareRootImpactModel(),
        PowerLawImpactModel(),
    ])
    def test_negative_size_returns_zero_no_crash(self, model):
        # Pre-Q4 SquareRootImpactModel raised ValueError on math.sqrt(<0).
        assert model.estimate_impact(-100.0, 100.0, 1_000_000.0, 0.02) == 0.0

    @pytest.mark.parametrize("model", [
        LinearImpactModel(),
        SquareRootImpactModel(),
        PowerLawImpactModel(),
    ])
    def test_zero_adv_returns_zero(self, model):
        assert model.estimate_impact(100.0, 100.0, 0.0, 0.02) == 0.0


# ---------------------------------------------------------------------------
# Engine-side bar_seconds wiring
# ---------------------------------------------------------------------------

class TestEngineBarSecondsWiring:
    """End-to-end: the engine derives bar_seconds and passes it through.

    Exercised via a recording PeriodicCostModel that captures every
    (timestamp, bar_seconds) tuple passed by the engine.
    """

    def _run_with_recording_model(self, freq: str, n_bars: int = 10):
        import numpy as np

        from aiphaforge import BacktestEngine
        from aiphaforge.margin import MarginConfig, PeriodicCostModel

        recorded: list[tuple] = []

        class _Recorder(PeriodicCostModel):
            def calculate_cost(self, position, price, timestamp,
                               margin_config, *, bar_seconds=None):
                recorded.append((timestamp, bar_seconds))
                return 0.0  # don't actually charge anything

        idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq=freq)
        close = 100.0 + np.arange(n_bars) * 0.1
        data = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": [1e6] * n_bars},
            index=idx,
        )
        signals = pd.Series(0.0, index=idx, dtype=float)
        signals.iloc[0] = 1.0  # open and hold

        eng = BacktestEngine(
            mode="event_driven",
            margin_config=MarginConfig(
                initial_margin_ratio=0.5,
                maintenance_margin_ratio=0.3,
                borrowing_rate=0.10,
            ),
            periodic_cost_model=_Recorder(),
            allow_short=True,
            include_benchmark=False,
        )
        eng.set_signals(signals)
        eng.run(data)
        return recorded

    def test_hourly_bar_seconds_is_3600(self):
        recorded = self._run_with_recording_model(freq="1h", n_bars=5)
        # First active bar's delta is 0 (init = first timestamp).
        # Subsequent bars should record bar_seconds ≈ 3600.
        non_zero = [bs for ts, bs in recorded if bs is not None and bs > 0]
        assert len(non_zero) >= 1
        assert all(abs(bs - 3600.0) < 1e-6 for bs in non_zero)

    def test_daily_bar_seconds_is_86400(self):
        recorded = self._run_with_recording_model(freq="D", n_bars=5)
        non_zero = [bs for ts, bs in recorded if bs is not None and bs > 0]
        assert len(non_zero) >= 1
        assert all(abs(bs - 86400.0) < 1e-6 for bs in non_zero)


class TestFlatStretchSemantics:
    """When a position is flat then reopens, the next bar's bar_seconds
    spans the entire flat stretch.

    This is *intentional*: cost accrues only when there's a position to
    borrow against, but when re-opening the engine charges for the whole
    elapsed wall-clock time since the last non-flat bar.
    """

    def test_reopen_after_flat_spans_full_gap(self):
        import numpy as np

        from aiphaforge import BacktestEngine
        from aiphaforge.margin import MarginConfig, PeriodicCostModel

        recorded: list[tuple] = []

        class _Recorder(PeriodicCostModel):
            def calculate_cost(self, position, price, timestamp,
                               margin_config, *, bar_seconds=None):
                recorded.append((timestamp, bar_seconds))
                return 0.0

        n = 10
        idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1h")
        close = 100.0 + np.arange(n) * 0.1
        data = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": [1e6] * n},
            index=idx,
        )
        signals = pd.Series(np.nan, index=idx, dtype=float)
        signals.iloc[0] = 1.0
        signals.iloc[2] = 0.0   # close after 2 bars
        signals.iloc[7] = -1.0  # reopen short 5 bars later

        eng = BacktestEngine(
            mode="event_driven",
            margin_config=MarginConfig(
                initial_margin_ratio=0.5,
                maintenance_margin_ratio=0.3,
                borrowing_rate=0.10,
            ),
            periodic_cost_model=_Recorder(),
            allow_short=True,
            include_benchmark=False,
        )
        eng.set_signals(signals)
        eng.run(data)

        # Position is non-flat at bars 0,1 (initial long), flat 2..6,
        # then short fills on bar 7 or 8 depending on order timing.
        # The first cost call AFTER the flat stretch should record a
        # bar_seconds spanning the gap (> 3600), not just one bar.
        post_flat = [(ts, bs) for ts, bs in recorded
                     if ts >= idx[7] and bs is not None]
        assert post_flat, "No periodic cost recorded after flat stretch"
        first_post_flat_seconds = post_flat[0][1]
        assert first_post_flat_seconds > 3600.0, (
            f"Reopen bar_seconds={first_post_flat_seconds}; "
            f"expected > 3600 (flat span)")
