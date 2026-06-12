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


def test_default_no_spread_snapshot_v2_8_5():
    # Snapshot captured on main @ 26a572a (v2.8.5) BEFORE this branch.
    # Do NOT regenerate expected values from production code: the point
    # is that default-config fills are byte-identical to v2.8.5. If
    # this fails, suspect the v2.8.6 broker changes, not the snapshot.
    rng = np.random.default_rng(7)
    periods = 80
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
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

    engine = BacktestEngine(mode="event_driven")  # all defaults
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
