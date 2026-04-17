"""Annualization tests — v1.9.5 per-symbol ``trading_days`` feature."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.strategies import MACrossover
from aiphaforge.utils import (
    TRADING_DAYS_STOCK,
    _normalize_trading_days,
    _resolve_trading_days,
)
from tests.conftest import make_ohlcv

# ---------------------------------------------------------------------------
# utils helpers
# ---------------------------------------------------------------------------


class TestResolveTradingDays:
    def test_scalar(self):
        assert _resolve_trading_days(252, "AAPL") == 252
        assert _resolve_trading_days(365, "BTC-USD") == 365

    def test_dict_hit(self):
        td = {"AAPL": 252, "BTC-USD": 365}
        assert _resolve_trading_days(td, "AAPL") == 252
        assert _resolve_trading_days(td, "BTC-USD") == 365

    def test_dict_miss_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = _resolve_trading_days({"AAPL": 252}, "UNKNOWN")
        assert val == TRADING_DAYS_STOCK
        assert len(caught) == 0

    def test_dict_miss_warn_once(self):
        td = {"AAPL": 252}
        warned = set()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _resolve_trading_days(td, "X", warn_missing=True, _warned=warned)
            _resolve_trading_days(td, "X", warn_missing=True, _warned=warned)
            _resolve_trading_days(td, "Y", warn_missing=True, _warned=warned)
        msgs = [w for w in caught if "missing entry" in str(w.message)]
        assert len(msgs) == 2


class TestNormalizeTradingDays:
    def test_scalar(self):
        port, per = _normalize_trading_days(365, ["AAPL", "BTC-USD"])
        assert port == 365
        assert per == {"AAPL": 365, "BTC-USD": 365}

    def test_scalar_with_override(self):
        port, per = _normalize_trading_days(252, ["AAPL"], portfolio_override=300)
        assert port == 300
        assert per == {"AAPL": 252}

    def test_dict_uniform_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            port, per = _normalize_trading_days(
                {"AAPL": 252, "MSFT": 252}, ["AAPL", "MSFT"])
        assert port == 252
        assert not [w for w in caught if "Mixed" in str(w.message)]

    def test_dict_mixed_emits_warning_and_uses_max(self):
        td = {"AAPL": 252, "BTC-USD": 365}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            port, per = _normalize_trading_days(td, ["AAPL", "BTC-USD"])
        assert port == 365
        assert per == {"AAPL": 252, "BTC-USD": 365}
        assert len([w for w in caught if "Mixed per-symbol" in str(w.message)]) == 1

    def test_dict_override_silent(self):
        td = {"AAPL": 252, "BTC-USD": 365}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            port, per = _normalize_trading_days(
                td, ["AAPL", "BTC-USD"], portfolio_override=252)
        assert port == 252
        assert not [w for w in caught if "Mixed per-symbol" in str(w.message)]

    def test_dict_subset_only_considers_active(self):
        td = {"AAPL": 252, "BTC-USD": 365}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            port, per = _normalize_trading_days(td, ["AAPL"])
        assert port == 252
        assert not [w for w in caught if "Mixed per-symbol" in str(w.message)]


# ---------------------------------------------------------------------------
# Engine — single-asset
# ---------------------------------------------------------------------------


def _run_single(data, *, trading_days):
    engine = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days=trading_days,
    )
    engine.set_strategy(MACrossover(short=10, long=30))
    return engine.run(data, symbol="X")


def test_scalar_252_regression_lock():
    data = make_ohlcv(252)
    engine_default = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
    engine_default.set_strategy(MACrossover(short=10, long=30))
    res_default = engine_default.run(data, symbol="X")
    res_252 = _run_single(data, trading_days=252)
    assert res_252.sharpe_ratio == res_default.sharpe_ratio
    assert res_252.trading_days == 252


def test_scalar_365_scales_sharpe_by_sqrt_ratio():
    data = make_ohlcv(500)
    res_252 = _run_single(data, trading_days=252)
    res_365 = _run_single(data, trading_days=365)
    if res_252.sharpe_ratio != 0:
        ratio = res_365.sharpe_ratio / res_252.sharpe_ratio
        assert ratio == pytest.approx(np.sqrt(365 / 252), rel=1e-9)
    assert res_252.trading_days == 252
    assert res_365.trading_days == 365


def test_result_trading_days_field_populated():
    data = make_ohlcv(100)
    res = _run_single(data, trading_days=500)
    assert res.trading_days == 500
    assert res.per_asset_trading_days == {"X": 500}


def test_result_to_dict_includes_trading_days():
    data = make_ohlcv(100)
    res = _run_single(data, trading_days=365)
    d = res.to_dict()
    assert d["trading_days"] == 365


def test_invalid_trading_days_type_raises():
    with pytest.raises(TypeError, match="must be int or Dict"):
        BacktestEngine(trading_days="252")  # type: ignore[arg-type]


def test_invalid_trading_days_value_raises():
    with pytest.raises(ValueError, match=">= 1"):
        BacktestEngine(trading_days=0)
    with pytest.raises(ValueError, match="BTC"):
        BacktestEngine(trading_days={"BTC": -1})
    with pytest.raises(ValueError, match="portfolio_trading_days"):
        BacktestEngine(portfolio_trading_days=0)


# ---------------------------------------------------------------------------
# Engine — multi-asset + per_asset_metrics populated
# ---------------------------------------------------------------------------


def _multi_data():
    return {"AAPL": make_ohlcv(252), "BTC-USD": make_ohlcv(252)}


def test_multi_asset_per_asset_metrics_now_populated():
    """Fix of pre-existing gap: result.per_asset_metrics was always None."""
    engine = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 365},
        portfolio_trading_days=252,
    )
    engine.set_strategy(MACrossover(short=10, long=30))
    res = engine.run(_multi_data())

    assert res.per_asset_metrics is not None
    assert set(res.per_asset_metrics.keys()) == {"AAPL", "BTC-USD"}
    for sym, m in res.per_asset_metrics.items():
        assert {"sharpe_ratio", "total_pnl", "max_drawdown",
                "volatility", "trading_days"} <= set(m.keys())


def test_multi_asset_per_asset_metrics_use_per_symbol_trading_days():
    data = _multi_data()
    eng_a = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 252},
        portfolio_trading_days=252,
    )
    eng_a.set_strategy(MACrossover(short=10, long=30))
    res_a = eng_a.run(data)

    eng_b = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 365},
        portfolio_trading_days=252,
    )
    eng_b.set_strategy(MACrossover(short=10, long=30))
    res_b = eng_b.run(data)

    assert res_a.per_asset_metrics["AAPL"]["sharpe_ratio"] == pytest.approx(
        res_b.per_asset_metrics["AAPL"]["sharpe_ratio"], rel=1e-9)
    btc_a = res_a.per_asset_metrics["BTC-USD"]["sharpe_ratio"]
    btc_b = res_b.per_asset_metrics["BTC-USD"]["sharpe_ratio"]
    if btc_a != 0:
        assert btc_b / btc_a == pytest.approx(np.sqrt(365 / 252), rel=1e-9)
    assert res_b.per_asset_metrics["AAPL"]["trading_days"] == 252
    assert res_b.per_asset_metrics["BTC-USD"]["trading_days"] == 365


def test_multi_asset_mixed_dict_without_override_warns():
    engine = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 365},
    )
    engine.set_strategy(MACrossover(short=10, long=30))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = engine.run(_multi_data())
    assert len([w for w in caught if "Mixed per-symbol" in str(w.message)]) == 1
    assert res.trading_days == 365


def test_multi_asset_mixed_dict_with_override_silent():
    engine = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 365},
        portfolio_trading_days=252,
    )
    engine.set_strategy(MACrossover(short=10, long=30))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = engine.run(_multi_data())
    assert not [w for w in caught if "Mixed per-symbol" in str(w.message)]
    assert res.trading_days == 252


def test_multi_asset_benchmark_uses_first_symbol_trading_days():
    """Bug #3 regression: in a mixed AAPL(252) + BTC(365) portfolio
    without a portfolio_trading_days override, the portfolio sharpe
    uses 365 (auto max) but the benchmark is a buy-and-hold of the
    first symbol (AAPL) — so its sharpe must stay annualised by 252.

    Before the fix, benchmark sharpe was wrongly scaled by √(365/252)
    because _calculate_metrics used self._portfolio_trading_days for
    every call, including the benchmark.
    """
    data = _multi_data()

    # Forced portfolio=252 baseline
    eng_baseline = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 252},
        portfolio_trading_days=252,
    )
    eng_baseline.set_strategy(MACrossover(short=10, long=30))
    res_baseline = eng_baseline.run(data)

    # Mixed dict, auto portfolio=max=365
    eng_mixed = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        trading_days={"AAPL": 252, "BTC-USD": 365},
    )
    eng_mixed.set_strategy(MACrossover(short=10, long=30))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore the Mixed-dict warning
        res_mixed = eng_mixed.run(data)

    # Benchmark is buy-and-hold of first symbol (AAPL) → td=252 in both
    bench_baseline = res_baseline.benchmark_metrics["sharpe_ratio"]
    bench_mixed = res_mixed.benchmark_metrics["sharpe_ratio"]
    assert bench_mixed == pytest.approx(bench_baseline, rel=1e-9), (
        f"Benchmark sharpe differs: baseline={bench_baseline:.6f}, "
        f"mixed={bench_mixed:.6f} (Bug #3 — benchmark got wrong td)"
    )

    # Portfolio sharpe should still differ (portfolio td differs)
    assert res_baseline.sharpe_ratio != res_mixed.sharpe_ratio
