"""v2.6 Commit B — batch-vs-incremental equivalence test harness."""
from __future__ import annotations

from typing import Callable, Union

import pandas as pd

# Either a v2.4 BaseFactor instance OR a callable that takes the
# data frame and returns the expected Series. The callable form
# is used by Commits C and D where there is no v2.4 batch
# counterpart (rolling mean / std are wrapped via pandas directly).
BatchReference = Union[object, Callable[[pd.DataFrame], pd.Series]]


def assert_batch_incremental_equivalent(
    batch_ref: BatchReference,
    incremental_factor,
    data: pd.DataFrame,
    *,
    rtol: float,
    atol: float = 0.0,
) -> None:
    """Assert batch reference and incremental.run_all match.

    rtol/atol are required (no default) so each call site picks
    the per-factor tolerance from the v2.6 design doc §3.

    The dispatch on ``hasattr(batch_ref, "compute")`` means a
    BaseFactor with .compute() goes the BaseFactor path; anything
    else is called as a function. For very defensive callers
    constructing a callable wrapper that shadows .compute, prefer
    passing a plain function.
    """
    if hasattr(batch_ref, "compute"):
        batch_out = batch_ref.compute(data)
    else:
        batch_out = batch_ref(data)
    incremental_out = incremental_factor.run_all(data)
    pd.testing.assert_series_equal(
        incremental_out.astype(float),
        batch_out.astype(float),
        rtol=rtol,
        atol=atol,
        check_names=False,
    )
