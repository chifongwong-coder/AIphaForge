"""v2.8.6 Commits E/F — spread model integration.

Commit E: event-driven fills carry the half-spread; realized spread
is recorded per order/trade and reported in summary().
Commit F: vectorized path folds a global FixedSpread into the linear
cost approximation; dynamic spreads warn per run and are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    FixedSpread,
    VolatilitySpread,
    ZeroFeeModel,
)

from .conftest import make_ohlcv

_BPS = 10.0
_HALF_FRAC = _BPS / 1e4 / 2  # taker pays half of the full spread


def _round_trip_run(spread_model=None, periods: int = 20):
    data = make_ohlcv(periods)
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0:5] = 1.0
    engine = BacktestEngine(
        mode="event_driven",
        fee_model=ZeroFeeModel(),
        spread_model=spread_model,
    )
    engine.set_signals(signals)
    return engine.run(data)


def _filled(orders: pd.DataFrame, side: str) -> pd.DataFrame:
    return orders[(orders["side"] == side) & (orders["filled_size"] > 0)]


def test_buy_fill_price_includes_half_spread():
    base = _round_trip_run(spread_model=None)
    spread = _round_trip_run(spread_model=FixedSpread(_BPS))
    # Market orders only in this fixture; with zero fees/slippage the
    # pre-spread price is the raw fill price of the base run.
    base_buy = _filled(base.orders, "buy")["filled_price"].iloc[0]
    spread_buy = _filled(spread.orders, "buy")["filled_price"].iloc[0]
    assert spread_buy == pytest.approx(base_buy * (1 + _HALF_FRAC))
    base_sell = _filled(base.orders, "sell")["filled_price"].iloc[0]
    spread_sell = _filled(spread.orders, "sell")["filled_price"].iloc[0]
    assert spread_sell == pytest.approx(base_sell * (1 - _HALF_FRAC))


def test_trade_spread_cost_recorded_and_gross_pnl_adds_back():
    result = _round_trip_run(spread_model=FixedSpread(_BPS))
    assert result.trades
    assert all(t.spread_cost > 0 for t in result.trades)
    assert result.total_spread == pytest.approx(
        sum(t.spread_cost for t in result.trades))
    for t in result.trades:
        assert t.gross_pnl == pytest.approx(
            t.pnl + t.commission + t.slippage_cost + t.spread_cost)


_SNAPSHOT_CLOSES = [
        100.00123016092391, 100.30042606816971, 100.02584117375997,
        99.13897423972712, 98.68924146367229, 97.71542936767258,
        97.77421662380264, 99.09342193054518, 98.60687603593607,
        97.9969393295441, 98.47814716323813, 98.83023077296852,
        98.93446684870653, 98.01818272571579, 97.98951481407086,
        98.67321317465027, 97.35570837289497, 96.9112111275269,
        95.08611766537084, 93.86781838986253, 92.15484452887414,
        91.93845112304705, 90.7805339502349, 91.02712348714363,
        91.16992138171926, 90.99965577559516, 88.73799279178336,
        88.2612527639042, 88.21845560124407, 88.31847169166174,
        86.97736570609516, 86.56281953447889, 85.7199165397864,
        85.02937836285341, 85.93625588718035, 85.2450852992935,
        85.21736665170462, 85.97436287255682, 85.4740773728871,
        85.37865446661769, 85.47301937589084, 85.5275529735575,
        84.48618439954303, 84.55053687093265, 85.70727055502503,
        84.39146007234194, 85.11983093417341, 85.22148553141896,
        84.67656455423861, 86.38750447033703, 87.04851773145793,
        86.01078963575355, 86.07490551792861, 86.57272458789855,
        86.40944492834551, 87.00156341736553, 86.94371155178061,
        87.52578110930105, 88.79395888690561, 88.19603387702557,
        88.37537616971093, 87.96687339898828, 88.07889871246569,
        87.03941341205615, 86.53665036377559, 86.36703538404976,
        87.14676983249362, 88.15053049286979, 86.99152055077259,
        86.30298838096925, 86.86309508802161, 85.14954483924349,
        84.75606973987001, 84.67365326203048, 85.74473146882303,
        86.33790131078177, 86.05585381124324, 85.73925648775067,
        85.52500894209429, 86.83798398186646,
]


def test_default_no_spread_snapshot_v2_8_5():
    # Snapshot captured on main @ 26a572a (v2.8.5) BEFORE this branch.
    # Do NOT regenerate expected values from production code: the point
    # is that default-config fills are byte-identical to v2.8.5. If
    # this fails, suspect the v2.8.6 broker changes, not the snapshot.
    # Close prices are inlined literals (originally default_rng(7)) so
    # the pin does not depend on NumPy's generator stream stability.
    periods = 80
    closes = np.asarray(_SNAPSHOT_CLOSES)
    data = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[5:20] = 1.0
    signals.iloc[40:60] = 1.0

    # resize_on_repeat_signal=True pins the legacy level-triggered fills
    # this snapshot was captured under (v2.8.5). The v2.9.1.1 default is
    # edge-triggered (repeated unchanged signals no longer re-rebalance);
    # that new default is covered by tests/test_signal_edge_trigger.py.
    # This snapshot remains the v2.8.5 broker-behavior regression guard.
    engine = BacktestEngine(mode="event_driven",
                            resize_on_repeat_signal=True)
    engine.set_signals(signals)
    result = engine.run(data)

    fills = result.orders[result.orders["filled_size"] > 0]
    assert result.final_capital == pytest.approx(
        95772.02820908098, rel=1e-12)
    assert result.num_trades == 22
    assert len(fills) == 37
    expected_first_six = [
        97.87199084042645,
        98.99432850861463,
        98.50826915990014,
        98.09493626887365,
        98.57662531040137,
        98.73140054219554,
    ]
    actual = list(fills["filled_price"].iloc[:6])
    assert actual == pytest.approx(expected_first_six, rel=1e-12)
    assert all(t.spread_cost == 0.0 for t in result.trades)


def test_summary_contains_total_spread_line():
    result = _round_trip_run(spread_model=FixedSpread(_BPS))
    assert "Total Spread" in result.summary()
    base = _round_trip_run(spread_model=None)
    assert "Total Spread" not in base.summary()


def test_per_asset_spread_override():
    data = {"AAA": make_ohlcv(20), "BBB": make_ohlcv(20, start_price=50.0)}
    signals = pd.Series(0.0, index=data["AAA"].index)
    signals.iloc[0:5] = 1.0
    engine = BacktestEngine(
        mode="event_driven",
        fee_model=ZeroFeeModel(),
        spread_model=None,
        asset_spread_models={"BBB": FixedSpread(_BPS)},
    )
    engine.set_signals({"AAA": signals, "BBB": signals})
    result = engine.run(data)
    by_symbol = {t.symbol: t for t in result.trades}
    assert by_symbol["AAA"].spread_cost == 0.0
    assert by_symbol["BBB"].spread_cost > 0.0


def test_volatility_spread_active_without_impact_model():
    # No impact model configured: the vol pipeline must still activate
    # for requires_volatility spread models (with min_bps=0, warmup
    # bars are spread-free and post-warmup spreads are vol-scaled).
    data = make_ohlcv(60)
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[1:3] = 1.0    # round trip inside the warmup window
    signals.iloc[30:35] = 1.0  # round trip after warmup (lookback 20)
    engine = BacktestEngine(
        mode="event_driven",
        fee_model=ZeroFeeModel(),
        spread_model=VolatilitySpread(k=1.0, min_bps=0.0),
    )
    engine.set_signals(signals)
    result = engine.run(data)
    warmup_end = data.index[20]
    warmup_trades = [t for t in result.trades if t.exit_time < warmup_end]
    later_trades = [t for t in result.trades if t.entry_time >= warmup_end]
    assert warmup_trades and later_trades
    # Warmup: vol channel is None -> min_bps floor (0 here).
    assert all(t.spread_cost == 0.0 for t in warmup_trades)
    # Post-warmup: vol-scaled, strictly above the zero floor.
    assert all(t.spread_cost > 0.0 for t in later_trades)


def test_limit_fill_clamped_spread_cost_realized():
    from aiphaforge.broker import Broker
    from aiphaforge.portfolio import Portfolio

    bar = pd.Series({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1e6,
    })
    ts = pd.Timestamp("2024-01-02 10:00")

    # (a) Zero slippage: limit buy touches; spread would push the fill
    # to 101 but the clamp pins it at the limit -> realized spread 0.
    broker = Broker(
        fee_model=ZeroFeeModel(),
        spread_model=FixedSpread(200),  # 2% full, 1% half
        check_buying_power=False,
    )
    broker.set_portfolio(Portfolio(initial_capital=1e6))
    order = broker.create_limit_order("default", "buy", 10, price=100.0)
    broker.submit_order(order, ts)
    broker.process_bar(bar, ts)
    assert order.is_filled
    assert order.filled_price == pytest.approx(100.0)
    assert order.metadata["spread_cost"] == pytest.approx(0.0)

    # (b) Positive slippage already overshoots the limit: the clamp
    # gives the slippage back; a side-signed formula records 0 spread
    # (an absolute-value formula would wrongly record the giveback).
    broker2 = Broker(
        spread_model=FixedSpread(200),  # default fee model: slippage
        check_buying_power=False,
    )
    broker2.set_portfolio(Portfolio(initial_capital=1e6))
    order2 = broker2.create_limit_order("default", "buy", 10, price=100.0)
    broker2.submit_order(order2, ts)
    broker2.process_bar(bar, ts)
    assert order2.is_filled
    assert order2.filled_price == pytest.approx(100.0)  # clamped
    assert order2.metadata["spread_cost"] == pytest.approx(0.0)


def test_partial_fill_spread_cost_uses_filled_size():
    from aiphaforge.broker import Broker, FillModel
    from aiphaforge.portfolio import Portfolio

    broker = Broker(
        fee_model=ZeroFeeModel(),
        fill_model=FillModel.CURRENT_CLOSE,
        spread_model=FixedSpread(_BPS),
        partial_fills=True,
        volume_limit_pct=0.1,
        check_buying_power=False,
    )
    broker.set_portfolio(Portfolio(initial_capital=1e6))
    bar = pd.Series({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 200.0,  # max 20 per bar
    })
    order = broker.create_market_order("default", "buy", 40)
    broker.submit_order(order, pd.Timestamp("2024-01-02 10:00"))
    broker.process_bar(bar, pd.Timestamp("2024-01-02 10:00"))
    half = 100.0 * _HALF_FRAC
    assert order.filled_size == pytest.approx(20.0)
    # spread_cost accrues on the FILLED size, cumulatively per fill.
    assert order.metadata["spread_cost"] == pytest.approx(half * 20.0)
    broker.process_bar(bar, pd.Timestamp("2024-01-02 11:00"))
    assert order.filled_size == pytest.approx(40.0)
    assert order.metadata["spread_cost"] == pytest.approx(half * 40.0)


# ---------------------------------------------------------------------------
# Commit F — vectorized cost path
# ---------------------------------------------------------------------------

def _vectorized_run(spread_model=None, asset_spread_models=None):
    import warnings as warnings_mod

    data = make_ohlcv(60)
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0:30] = 1.0
    engine = BacktestEngine(
        mode="vectorized",
        fee_model=ZeroFeeModel(),
        spread_model=spread_model,
        asset_spread_models=asset_spread_models,
        representative_notional=95_000,
    )
    engine.set_signals(signals)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = engine.run(data)
    return result, caught


def test_vectorized_fixed_spread_reduces_equity_by_bps():
    # Exact fold identity at the DefaultTradeCost level: vectorized
    # positions are WEIGHT units, so the per-bar spread charge in
    # return units is |diff(positions)| * half_spread_rate — one
    # half-spread per side of the traded equity fraction.
    from aiphaforge.costs import DefaultTradeCost

    idx = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.Series([0.0, 0.01, -0.01, 0.0, 0.02, 0.0], index=idx)
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=idx)
    data = pd.DataFrame({"close": [100.0] * 6}, index=idx)
    capital = 100_000.0
    rate = _HALF_FRAC

    base = DefaultTradeCost().apply_vectorized(
        returns, positions, data, ZeroFeeModel(), capital,
        representative_notional=capital)
    folded = DefaultTradeCost(half_spread_rate=rate).apply_vectorized(
        returns, positions, data, ZeroFeeModel(), capital,
        representative_notional=capital)
    expected_cost = positions.diff().abs().fillna(0) * rate
    pd.testing.assert_series_equal(base - folded, expected_cost)


def test_vectorized_spread_magnitude_matches_event_driven():
    # The folded charge must be the same order as the event-driven
    # spread cost (a dimensional bug here once made it ~1000x small:
    # weight-unit positions were multiplied by close and divided by
    # initial capital as if they were share counts).
    data = make_ohlcv(60)
    # Entry mid-window: a position held from bar 0 has no diff() entry
    # side in the vectorized path, which would halve the comparison.
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[10:30] = 1.0

    ed_engine = BacktestEngine(
        mode="event_driven", fee_model=ZeroFeeModel(),
        spread_model=FixedSpread(_BPS))
    ed_engine.set_signals(signals)
    ed_result = ed_engine.run(data)
    ed_cost_return = ed_result.total_spread / ed_result.initial_capital

    def _vec(spread_model):
        import warnings as warnings_mod

        engine = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            spread_model=spread_model,
            representative_notional=95_000)
        engine.set_signals(signals)
        with warnings_mod.catch_warnings():
            warnings_mod.simplefilter("ignore")
            return engine.run(data)

    vec_base = _vec(None)
    vec_spread = _vec(FixedSpread(_BPS))
    vec_delta = (
        vec_base.equity_curve.iloc[-1]
        - vec_spread.equity_curve.iloc[-1]) / vec_base.equity_curve.iloc[0]

    assert ed_cost_return > 0 and vec_delta > 0
    # Event-driven sizes at ~0.95 equity (FractionSizer default) while
    # the vectorized weight path trades the full unit weight; allow a
    # generous band — the bug class this guards against is orders of
    # magnitude, not percent.
    assert vec_delta == pytest.approx(ed_cost_return, rel=0.5)


def test_vectorized_dynamic_spread_warns_and_ignored():
    base, _ = _vectorized_run(spread_model=None)
    result, caught = _vectorized_run(
        spread_model=VolatilitySpread(k=0.1, min_bps=5))
    messages = [str(w.message) for w in caught]
    assert any("FixedSpread only" in m for m in messages)
    pd.testing.assert_series_equal(result.equity_curve, base.equity_curve)


def test_vectorized_asset_spread_overrides_warn_and_ignored():
    base, _ = _vectorized_run(spread_model=None)
    result, caught = _vectorized_run(
        asset_spread_models={"X": FixedSpread(_BPS)})
    messages = [str(w.message) for w in caught]
    assert any("asset_spread_models" in m for m in messages)
    pd.testing.assert_series_equal(result.equity_curve, base.equity_curve)
