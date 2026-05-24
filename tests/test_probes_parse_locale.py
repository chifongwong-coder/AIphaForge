"""v2.8.4 M15 — locale-aware ``parse_numeric_answer`` tests.

Pins the ``decimal_separator: Literal["us", "eu"] = "us"`` kwarg:

- The default ``"us"`` mode preserves v2.8.3 outputs verbatim
  (including the known silent-wrong outputs on European-style
  inputs — the test 7 sub-pinnings are the backward-compat gate
  per the v2.8.4 §1 / §4 M15 today's-behavior block).
- The ``"eu"`` opt-in swaps the comma/period roles: comma=decimal,
  period=thousands.  Bracketed ranges accept BOTH ``;`` (canonical,
  silent) and ``,`` (parses, emits a ``UserWarning`` per R5).
- The v2.8.3 M7 broad warn-shim is REMOVED — the same input under
  US default is now silent (its role replaced by the EU opt-in).
- The percent-path ``_try_float`` call is threaded with the locale
  kwarg so ``"1,5%"`` under EU correctly returns ``0.015``.
"""
from __future__ import annotations

import warnings

import pytest

from aiphaforge.probes.scoring import parse_numeric_answer


# ---------- M15 test 1: EU scalar with decimal-comma ----------

def test_eu_scalar_comma_decimal() -> None:
    assert parse_numeric_answer("1,5", decimal_separator="eu") == 1.5


# ---------- M15 test 2: EU scalar with thousands-period + decimal-comma --

def test_eu_scalar_with_thousands_period() -> None:
    assert parse_numeric_answer(
        "1.234,56", decimal_separator="eu",
    ) == pytest.approx(1234.56)


# ---------- M15 test 3: EU worded range -----------------------------------

def test_eu_worded_range() -> None:
    assert parse_numeric_answer(
        "between 1,5 and 2,5", decimal_separator="eu",
    ) == (1.5, 2.5)


# ---------- M15 test 4: EU bracketed range with canonical ';' separator ---

def test_eu_bracket_range_canonical_semicolon() -> None:
    """Canonical EU bracketed range form: decimal-comma scalars
    separated by semicolons.  Parses cleanly with NO warning.

    NOTE: plan §4 M15 step (f) bullet 1 listed `[1;5; 2;5]` as the
    canonical form; that input is internally inconsistent (`;` cannot
    be both the decimal-separator and the value-separator).  Bullet 2
    of the same step said `[1,5; 2,5]` is "clean" — implementation
    follows bullet 2, documented in plan §8 deviations.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning -> error
        assert parse_numeric_answer(
            "[1,5; 2,5]", decimal_separator="eu",
        ) == (1.5, 2.5)


# ---------- M15 test 5: EU bracket-range with ',' separator (warns) -------

def test_eu_bracket_range_with_comma_warns_but_parses() -> None:
    """``[1,5, 2,5]`` under EU is ambiguous (comma is both decimal
    AND separator).  R5 user-lock: accept both, warn on comma form.
    """
    with pytest.warns(UserWarning, match="ambiguous|comma"):
        assert parse_numeric_answer(
            "[1,5, 2,5]", decimal_separator="eu",
        ) == (1.5, 2.5)


# ---------- M15 test 6: EU bracketed scalar (single comma) ----------------

def test_eu_bracket_scalar_with_comma() -> None:
    """``[1,5]`` under EU is the bracketed-scalar form -> ``1.5``.

    Disambiguation: a single comma inside brackets is treated as the
    decimal point of a single scalar; two or more commas is the
    range-with-warning case (test 5).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert parse_numeric_answer(
            "[1,5]", decimal_separator="eu",
        ) == 1.5


# ---------- M15 test 7: US default backward-compat regression guards ------

def test_us_default_preserves_v2_8_3_silent_wrong_outputs() -> None:
    """Pin the verbatim US-default outputs observed at v2.8.4 write
    time.  These are silent-wrong outputs on European-style inputs;
    M15 keeps them stable so v2.8.3 consumers see zero observable
    change in the default code path.  Users with EU-style data opt
    into ``decimal_separator="eu"`` to get correct parsing.
    """
    # Sub-pinnings (a)-(e): pre-v2.8.4 silent-wrong outputs preserved
    assert parse_numeric_answer("1,5") is None
    assert parse_numeric_answer("1,5 to 2,5") == (15.0, 25.0)
    assert parse_numeric_answer("[1,5, 2,5]") == (15.0, 25.0)
    assert parse_numeric_answer("[1,5]") == (1.0, 5.0)
    assert parse_numeric_answer("1,5%") is None

    # Sub-pinnings (f) and (h): two v3 MED-B edge cases under US default
    assert parse_numeric_answer("[1, 5]") == (1.0, 5.0)
    assert parse_numeric_answer("[1,234, 5,678]") == (1234.0, 5678.0)

    # Sub-pinning (i): v3 MED-B SILENT-SHAPE RISK under EU mode.
    # ``[1,234, 5,678]`` under EU treats the comma as a decimal
    # point AND the comma as the value separator (two commas), so
    # the parser sees a range whose values come out 1000x smaller
    # than the US default.  PIN this output explicitly so the
    # silent-shape risk is documented in the test suite (not just
    # in release notes).  See §9 release-notes MED-E caveat.
    with pytest.warns(UserWarning, match="ambiguous|comma"):
        assert parse_numeric_answer(
            "[1,234, 5,678]", decimal_separator="eu",
        ) == pytest.approx((1.234, 5.678))


# ---------- M15 test 8: v2.8.3 M7 shim removal under US default -----------

def test_v2_8_3_m7_warn_shim_removed_under_us_default() -> None:
    """Pins the M7 shim removal: under default US mode, the parser
    is silent for inputs like ``"1,5"`` that used to emit the v2.8.3
    M7 broad-comma warning.  Callers who actually have EU data should
    opt into ``decimal_separator="eu"`` to get correct parsing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning becomes error
        # ``"1,5"`` under US default returns None silently
        assert parse_numeric_answer("1,5") is None
        # ``"12,50"`` also silent under US default
        assert parse_numeric_answer("12,50") is None


# ---------- M15 test 9: percent path threads decimal_separator ------------

def test_eu_percent_decimal_comma() -> None:
    """``"1,5%"`` under EU + ``percent="decimal"`` -> ``0.015``.

    Requires the ``scoring.py`` percent-path ``_try_float`` call to
    receive the ``decimal_separator`` kwarg.  Without that thread the
    parser would silently treat ``"1,5"`` as ``"15"`` and return
    ``0.15`` instead of ``0.015`` — a 10x error.
    """
    result = parse_numeric_answer(
        "1,5%", decimal_separator="eu", percent="decimal",
    )
    assert result == pytest.approx(0.015)
