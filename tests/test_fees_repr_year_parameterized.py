"""v2.8.4 M11b — USStockFeeModel.__repr__ year-label parameterization tests."""
from __future__ import annotations

import importlib

from aiphaforge.fees import USStockFeeModel


def test_us_stock_repr_includes_schedule_labels() -> None:
    """Default ``repr(USStockFeeModel())`` keeps the FY2026 + 2026 schedule labels.

    M11b is a refactor; the default output is unchanged from v2.8.3 so
    downstream loggers / consumers do not need to re-pin.
    """
    text = repr(USStockFeeModel())
    assert "FY2026" in text, text
    assert "2026" in text, text


def test_us_stock_repr_uses_module_constants() -> None:
    """Monkey-patching the module constants must flow through to the repr.

    Pins the v2.8.4 M11b refactor contract: the year labels are no longer
    hardcoded inside the formatter; they are resolved from
    ``_REPR_SEC_SCHEDULE_LABEL`` and ``_REPR_FINRA_SCHEDULE_LABEL`` at call
    time.  When SEC publishes a new schedule (FY2027+) the maintainer
    updates two module constants and the repr follows.
    """
    fees_module = importlib.import_module("aiphaforge.fees")
    original_sec = fees_module._REPR_SEC_SCHEDULE_LABEL
    original_finra = fees_module._REPR_FINRA_SCHEDULE_LABEL
    try:
        fees_module._REPR_SEC_SCHEDULE_LABEL = "FY2099"
        fees_module._REPR_FINRA_SCHEDULE_LABEL = "2099"
        text = repr(USStockFeeModel())
        assert "FY2099" in text, text
        assert "[2099]" in text, text
        # The pre-patch labels must NOT leak into the new repr.
        assert "FY2026" not in text, text
    finally:
        fees_module._REPR_SEC_SCHEDULE_LABEL = original_sec
        fees_module._REPR_FINRA_SCHEDULE_LABEL = original_finra
