"""Lock package version against pyproject.toml.

This test is intentionally added in v1.9.6 commit 1 and will fail until
commit 8 bumps both `__version__` and `pyproject.toml` to '1.9.6'.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import aiphaforge

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "2.0.1"


def _read_pyproject_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def test_version_matches_pyproject():
    assert aiphaforge.__version__ == _read_pyproject_version(), (
        f"aiphaforge.__version__ ({aiphaforge.__version__}) does not "
        f"match pyproject.toml version ({_read_pyproject_version()})"
    )


def test_version_is_current_release():
    assert aiphaforge.__version__ == EXPECTED_VERSION, (
        f"Expected v{EXPECTED_VERSION}; found {aiphaforge.__version__}"
    )


def test_multi_asset_result_symbols_preserves_order():
    """v1.9.7 commit 3 probe: multi-asset path should produce
    result.symbols matching what was passed (sorted, since the engine
    sorts symbols internally).
    """
    import numpy as np
    import pandas as pd

    from aiphaforge import BacktestEngine
    from aiphaforge.fees import ZeroFeeModel

    n = 20
    base = pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
         "close": [100.0] * n, "volume": [1e6] * n},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    sig = pd.Series(np.nan, index=base.index, dtype=float)
    sig.iloc[3] = 1.0

    eng = BacktestEngine(mode="event_driven", fee_model=ZeroFeeModel(),
                         include_benchmark=False)
    eng.set_signals({"BBB": sig, "AAA": sig, "CCC": sig})
    res = eng.run({"BBB": base, "AAA": base, "CCC": base})

    # Engine sorts internally. Sorted unique → ['AAA', 'BBB', 'CCC'].
    assert res.symbols == ["AAA", "BBB", "CCC"]


def test_single_asset_result_has_symbols_populated():
    """v1.9.7: BacktestResult.symbols populated for single-asset runs.

    Pre-fix it was empty for single-asset (only multi-asset set it),
    which silently broke estimate_capacity and any other consumer.
    """
    import numpy as np
    import pandas as pd

    from aiphaforge import BacktestEngine
    from aiphaforge.fees import ZeroFeeModel

    n = 30
    data = pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
         "close": [100.0] * n, "volume": [1e6] * n},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0
    eng = BacktestEngine(mode="event_driven", fee_model=ZeroFeeModel(),
                         include_benchmark=False)
    eng.set_signals(signals)
    res = eng.run(data, symbol="AAPL")
    assert res.symbols == ["AAPL"]
