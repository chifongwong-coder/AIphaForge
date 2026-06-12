"""v2.8.6 Commits B/C — T+1 settlement constraint.

Commit B: config/engine plumbing and validation.
Commit C: broker-side enforcement (bought-today bucket, fill-time
sellable check, IOC/FOK/exit-rule/margin interactions).

T+1 is the SSE/SZSE cash-equity rule: shares bought today cannot be
sold the same calendar day. Intraday fixtures build their indexes
inline with ``pd.date_range`` (the shared ``tests/_helpers/ohlcv.py``
constructors are business-daily only).
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge import BacktestEngine

from .conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Commit B — config + engine plumbing
# ---------------------------------------------------------------------------


def test_invalid_settlement_value_raises():
    with pytest.raises(ValueError, match=r"t\+0.*t\+1"):
        BacktestEngine(settlement="t+2")


def test_t1_with_vectorized_mode_raises():
    data = make_ohlcv(30)
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(settlement="t+1")  # default mode: vectorized
    engine.set_signals(signals)
    with pytest.raises(ValueError, match="event_driven"):
        engine.run(data)


def test_asset_settlements_t1_vectorized_raises():
    data = make_ohlcv(30)
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(
        settlement="t+0", asset_settlements={"X": "t+1"})
    engine.set_signals(signals)
    with pytest.raises(ValueError, match="event_driven"):
        engine.run(data)


# ---------------------------------------------------------------------------
# Commit C — broker enforcement
# ---------------------------------------------------------------------------

_D0 = pd.Timestamp("2024-01-01 10:00")  # "yesterday" for pre-held setups
_D1A = pd.Timestamp("2024-01-02 10:00")
_D1B = pd.Timestamp("2024-01-02 11:00")
_D2A = pd.Timestamp("2024-01-03 10:00")


def _bar(price: float = 100.0) -> pd.Series:
    return pd.Series({
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "volume": 1e6,
    })


def _make_broker(settlement: str = "t+1"):
    from aiphaforge.broker import Broker, FillModel
    from aiphaforge.portfolio import Portfolio

    portfolio = Portfolio(initial_capital=1_000_000)
    broker = Broker(
        settlement=settlement,
        fill_model=FillModel.CURRENT_CLOSE,
        check_buying_power=False,
    )
    broker.set_portfolio(portfolio)
    return broker, portfolio


def _fill_buy(broker, portfolio, qty: float, ts: pd.Timestamp):
    order = broker.create_market_order("default", "buy", qty)
    broker.submit_order(order, ts)
    broker.process_bar(_bar(), ts)
    assert order.is_filled
    portfolio.update_from_order(order, ts)
    return order


def _intraday_ohlcv(closes_by_day):
    """Hourly OHLCV; ``closes_by_day`` is a list of per-day close lists."""
    idx, closes = [], []
    for d, day_closes in enumerate(closes_by_day):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for h, c in enumerate(day_closes):
            idx.append(day + pd.Timedelta(hours=10 + h))
            closes.append(float(c))
    series = pd.Series(closes, index=pd.DatetimeIndex(idx))
    return pd.DataFrame({
        "open": series, "high": series * 1.01, "low": series * 0.99,
        "close": series, "volume": [1e6] * len(series),
    })


def test_t1_same_day_sell_rejected():
    import warnings as warnings_mod

    data = _intraday_ohlcv([[100, 100, 100, 100], [100, 100, 100, 100]])
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0] = 1.0  # buy fills bar 1 (day 1)
    engine = BacktestEngine(mode="event_driven", settlement="t+1")
    engine.set_signals(signals)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = engine.run(data)
    t1_warnings = [w for w in caught
                   if "T+1 settlement" in str(w.message)]
    assert len(t1_warnings) == 1
    orders = result.orders
    rejected = orders[(orders["side"] == "sell")
                      & (orders["status"] == "rejected")]
    assert not rejected.empty
    assert any("T+1 settlement" in m.get("reject_reason", "")
               for m in rejected["metadata"])


def test_t1_next_day_sell_fills():
    data = _intraday_ohlcv([[100, 100, 100, 100], [100, 100, 100, 100]])
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0] = 1.0
    engine = BacktestEngine(mode="event_driven", settlement="t+1")
    engine.set_signals(signals)
    result = engine.run(data)
    orders = result.orders
    filled_sells = orders[(orders["side"] == "sell")
                          & (orders["filled_size"] > 0)]
    assert not filled_sells.empty
    assert all(t.date() == _D2A.date()
               for t in filled_sells["filled_time"])


def test_t1_partial_fill_settled_then_remainder_expired():
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D0)   # pre-held (yesterday)
    _fill_buy(broker, portfolio, 50, _D1A)   # bought today
    sell = broker.create_market_order("default", "sell", 150)
    broker.submit_order(sell, _D1B)
    broker.process_bar(_bar(), _D1B)
    assert sell.filled_size == pytest.approx(100.0)
    assert sell.status.value == "partially_expired"
    assert sell.metadata["expiry_reason"] == "t+1_settlement"


def test_t0_same_day_sell_fills_regression():
    data = _intraday_ohlcv([[100, 100, 100, 100], [100, 100, 100, 100]])
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0] = 1.0
    engine = BacktestEngine(mode="event_driven")  # default t+0
    engine.set_signals(signals)
    result = engine.run(data)
    orders = result.orders
    filled_sells = orders[(orders["side"] == "sell")
                          & (orders["filled_size"] > 0)]
    assert not filled_sells.empty
    assert filled_sells["filled_time"].iloc[0].date() == _D1A.date()


def test_t1_multiple_buys_same_day_accumulate():
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D0)  # pre-held
    _fill_buy(broker, portfolio, 30, _D1A)
    _fill_buy(broker, portfolio, 20, _D1A)
    sell = broker.create_market_order("default", "sell", 150)
    broker.submit_order(sell, _D1B)
    broker.process_bar(_bar(), _D1B)
    # sellable = 150 - (30 + 20) = 100
    assert sell.filled_size == pytest.approx(100.0)
    assert sell.status.value == "partially_expired"


def test_t1_does_not_block_buys():
    broker, portfolio = _make_broker()
    for qty in (10, 20, 30):
        order = _fill_buy(broker, portfolio, qty, _D1A)
        assert order.is_filled


def test_default_settlement_t0_reaches_broker(monkeypatch):
    import aiphaforge.core_event_driven as ced

    captured = {}
    original = ced.Broker

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ced, "Broker", spy)
    data = make_ohlcv(20)
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(mode="event_driven")
    engine.set_signals(signals)
    engine.run(data)
    assert captured["settlement"] == "t+0"


def test_t1_ioc_sell_same_day_rejected():
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D1A)  # all bought today
    sell = broker.create_market_order(
        "default", "sell", 100, time_in_force="IOC")
    broker.submit_order(sell, _D1B)
    # Direct-_execute_fill path (same-bar immediate processing).
    broker.process_immediate_orders(_bar(), _D1B)
    assert sell.status.value == "rejected"
    assert "T+1 settlement" in sell.metadata["reject_reason"]
    assert sell.filled_size == 0


def test_t1_fok_sell_exceeding_sellable_kills():
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D0)  # pre-held
    _fill_buy(broker, portfolio, 50, _D1A)  # bought today
    sell = broker.create_market_order(
        "default", "sell", 120, time_in_force="FOK")
    broker.submit_order(sell, _D1B)
    broker.process_bar(_bar(), _D1B)
    # sellable = 100 < 120: fill-or-kill kills outright, no clamp.
    assert sell.status.value == "rejected"
    assert sell.filled_size == 0
    assert "fill-or-kill" in sell.metadata["reject_reason"]


def test_t1_stop_loss_same_day_rejected_then_fills_next_day():
    data = _intraday_ohlcv([[100, 100, 88, 88], [88, 88, 88, 88]])
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(
        mode="event_driven", settlement="t+1", stop_loss=0.05)
    engine.set_signals(signals)
    result = engine.run(data)
    orders = result.orders
    sells = orders[orders["side"] == "sell"]
    rejected = sells[sells["status"] == "rejected"]
    filled = sells[sells["filled_size"] > 0]
    assert not rejected.empty           # stop blocked on day 1
    assert not filled.empty             # exit completes on day 2
    assert all(t.date() == _D2A.date() for t in filled["filled_time"])


def test_t1_long_to_short_flip_clamped_to_close():
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D0)  # pre-held long 100
    # Freeze-day buy FIRST (fills at _D1A), then the flip order, so
    # the flip's fill at _D1B actually exercises the bought-today
    # interaction (submitting the flip before the buy would let it
    # fill during the buy's process_bar pass, ahead of the buy).
    _fill_buy(broker, portfolio, 10, _D1A)
    # Single atomic flip order: close 110 + open 70 short.
    flip = broker.create_market_order("default", "sell", 180)
    broker.submit_order(flip, _D1B)
    broker.process_bar(_bar(), _D1B)
    # sellable = settled-at-day-start (100); the 10 bought today are
    # frozen; only the long-reducing settled portion fills.
    assert flip.filled_size == pytest.approx(100.0)
    assert flip.status.value == "partially_expired"
    portfolio.update_from_order(flip, _D1B)
    position = portfolio.get_position("default")
    assert position is None or position.size >= 0  # no short today


def test_t1_same_bar_buy_then_sell_rejected():
    # Day-trade from flat, both orders filling in ONE process_bar
    # batch: the portfolio is only updated after process_bar returns,
    # so the sell-time position read is stale (None) — the broker's
    # day ledger must still catch the freeze.
    broker, portfolio = _make_broker()
    buy = broker.create_market_order("default", "buy", 10)
    sell = broker.create_market_order("default", "sell", 10)
    broker.submit_order(buy, _D1A)
    broker.submit_order(sell, _D1A)
    broker.process_bar(_bar(), _D1A)
    assert buy.is_filled
    assert sell.status.value == "rejected"
    assert "T+1 settlement" in sell.metadata["reject_reason"]


def test_t1_same_bar_ioc_day_trade_rejected():
    # Same-bar hook/agent flow via process_immediate_orders (the
    # direct-_execute_fill path).
    broker, portfolio = _make_broker()
    buy = broker.create_market_order(
        "default", "buy", 10, time_in_force="IOC")
    sell = broker.create_market_order(
        "default", "sell", 10, time_in_force="IOC")
    broker.submit_order(buy, _D1A)
    broker.submit_order(sell, _D1A)
    broker.process_immediate_orders(_bar(), _D1A)
    assert buy.is_filled
    assert sell.status.value == "rejected"
    assert "T+1 settlement" in sell.metadata["reject_reason"]


def test_t1_same_bar_sells_consume_settled_once():
    # Two same-bar sells against 100 settled shares: the second must
    # see the first's consumption via the day ledger (the portfolio
    # position is stale within the batch) and clamp to the remainder.
    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D0)
    sell_a = broker.create_market_order("default", "sell", 60)
    sell_b = broker.create_market_order("default", "sell", 60)
    broker.submit_order(sell_a, _D1A)
    broker.submit_order(sell_b, _D1A)
    broker.process_bar(_bar(), _D1A)
    assert sell_a.filled_size == pytest.approx(60.0)
    assert sell_b.filled_size == pytest.approx(40.0)
    assert sell_b.status.value == "partially_expired"


def test_t1_settled_resting_partial_sell_no_spurious_warning():
    import warnings as warnings_mod

    # A fully-settled sell that volume-clips across bars must not trip
    # the T+1 trigger (the comparison uses the remaining size, not the
    # original order size) — and must not consume the once-per-broker
    # warning that a later genuine T+1 event needs.
    from aiphaforge.broker import Broker, FillModel
    from aiphaforge.portfolio import Portfolio

    portfolio = Portfolio(initial_capital=1_000_000)
    broker = Broker(
        settlement="t+1",
        fill_model=FillModel.CURRENT_CLOSE,
        partial_fills=True,
        volume_limit_pct=0.1,
        check_buying_power=False,
    )
    broker.set_portfolio(portfolio)
    buy = broker.create_market_order("default", "buy", 100)
    broker.submit_order(buy, _D0)
    big_bar = _bar()
    broker.process_bar(big_bar, _D0)
    portfolio.update_from_order(buy, _D0)

    # Bar volumes chosen so the second clip equals the remaining size
    # (the pre-existing multi-bar partial-fill path crashes when the
    # clip exceeds remaining — out of scope here; this test pins only
    # the T+1 trigger comparison).
    clip_bar_1 = _bar()
    clip_bar_1["volume"] = 600.0  # max 60: fills 60 of 100
    clip_bar_2 = _bar()
    clip_bar_2["volume"] = 400.0  # max 40 == remaining
    sell = broker.create_market_order("default", "sell", 100)
    broker.submit_order(sell, _D1A)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        broker.process_bar(clip_bar_1, _D1A)
        broker.process_bar(clip_bar_2, _D1B)
    t1_warnings = [w for w in caught
                   if "T+1 settlement" in str(w.message)]
    assert t1_warnings == []
    assert sell.filled_size == pytest.approx(100.0)


def test_broker_invalid_settlement_raises():
    from aiphaforge.broker import Broker

    with pytest.raises(ValueError, match="settlement"):
        Broker(settlement="T+1")


def test_t1_margin_liquidation_retries_after_rejection():
    from aiphaforge.margin import MarginCallExitRule

    broker, portfolio = _make_broker()
    _fill_buy(broker, portfolio, 100, _D1A)  # bought today

    class _MarginCallPortfolio:
        """Duck-typed view: same positions, margin call always on."""
        is_margin_call = True
        positions = portfolio.positions

        @staticmethod
        def get_position(symbol):
            return portfolio.get_position(symbol)

    broker._portfolio = _MarginCallPortfolio()
    rule = MarginCallExitRule()
    brokers = {"default": broker}
    prices = {"default": 100.0}

    rule.check_portfolio(
        _MarginCallPortfolio(), brokers, ["default"], prices, _D1A)
    assert len(broker.get_pending_orders()) == 1
    # Dedup holds while the order is still active.
    rule.check_portfolio(
        _MarginCallPortfolio(), brokers, ["default"], prices, _D1A)
    assert len(broker.get_pending_orders()) == 1

    broker.process_bar(_bar(), _D1B)  # T+1 rejects the liquidation
    assert not broker.get_pending_orders()

    # Terminal-and-unfilled order releases the dedup: re-submitted.
    rule.check_portfolio(
        _MarginCallPortfolio(), brokers, ["default"], prices, _D1B)
    assert len(broker.get_pending_orders()) == 1


def test_t1_standalone_broker_without_portfolio_raises():
    from aiphaforge.broker import Broker, FillModel

    broker = Broker(
        settlement="t+1",
        fill_model=FillModel.CURRENT_CLOSE,
        check_buying_power=False,
    )
    sell = broker.create_market_order("default", "sell", 100)
    broker.submit_order(sell, _D1A)
    with pytest.raises(RuntimeError, match="portfolio"):
        broker.process_bar(_bar(), _D1A)
