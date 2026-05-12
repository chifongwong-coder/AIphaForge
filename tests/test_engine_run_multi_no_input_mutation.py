"""v2.2.2 Commit D — engine._run_multi must not mutate caller's data_dict.

Prior behavior at engine.py:562 was:
    data_dict[sym] = ensure_datetime_index(df).sort_index().copy()
which writes back into the user's dict. Users keeping a reference
saw their frames silently replaced with sorted copies.

The fix builds a local `normalized` dict and rebinds `data_dict =
normalized` so the rest of the method body is unchanged but the
caller's view is preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge import BacktestEngine
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.strategies import MACrossover


def _unsorted_ohlcv(n: int = 30) -> pd.DataFrame:
    """Build a frame whose index is intentionally NOT monotonic so the
    `.sort_index().copy()` step has visible work to do — if the engine
    mutates the input in place, we'd see is_monotonic_increasing flip
    from False to True post-call."""
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    idx = list(pd.bdate_range("2024-01-01", periods=n))
    # Swap two adjacent dates to break monotonicity.
    idx[5], idx[6] = idx[6], idx[5]
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.DatetimeIndex(idx),
    )


def _build_engine() -> BacktestEngine:
    eng = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
    eng.set_strategy(MACrossover(short=3, long=10))
    return eng


class TestRunMultiNoInputMutation:
    def test_run_multi_does_not_sort_caller_frames_in_place(self):
        # Capture the input state BEFORE the engine run.
        data_dict = {"A": _unsorted_ohlcv(), "B": _unsorted_ohlcv()}
        # Both frames start with an unsorted index (we built them that way).
        assert not data_dict["A"].index.is_monotonic_increasing
        assert not data_dict["B"].index.is_monotonic_increasing
        original_a_id = id(data_dict["A"])
        original_b_id = id(data_dict["B"])
        engine = _build_engine()
        engine.run(data_dict)
        # The caller's dict values must be the SAME objects (no
        # reassignment) AND still NOT monotonic (no in-place sort).
        assert id(data_dict["A"]) == original_a_id
        assert id(data_dict["B"]) == original_b_id
        assert not data_dict["A"].index.is_monotonic_increasing, (
            "_run_multi sorted A's index in place; caller's frame "
            "was silently mutated"
        )
        assert not data_dict["B"].index.is_monotonic_increasing

    def test_run_multi_does_not_add_or_remove_keys(self):
        data_dict = {"A": _unsorted_ohlcv(), "B": _unsorted_ohlcv()}
        keys_before = set(data_dict.keys())
        engine = _build_engine()
        engine.run(data_dict)
        keys_after = set(data_dict.keys())
        assert keys_before == keys_after
