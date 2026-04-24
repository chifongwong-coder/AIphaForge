"""v2.0.1 M1 — tests for parse_numeric_answer / parse_choice_answer / parse_binary_answer."""
from __future__ import annotations

import pytest

from aiphaforge.probes import (
    parse_binary_answer,
    parse_choice_answer,
    parse_numeric_answer,
)

# ---------- parse_numeric_answer: scalar shapes ----------

class TestParseNumericScalar:
    @pytest.mark.parametrize("text,expected", [
        ("172.06", 172.06),
        ("172", 172.0),
        ("1234.56", 1234.56),
        ("-2.5", -2.5),
        ("\u22122.5", -2.5),  # Unicode minus
        ("1.2e-3", 0.0012),
        ("1.2E+3", 1200.0),
        ("0", 0.0),
    ])
    def test_plain(self, text, expected):
        assert parse_numeric_answer(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text,expected", [
        ("$172.06", 172.06),
        ("172.06 USD", 172.06),
        ("USD 172.06", 172.06),
        ("\u20AC50", 50.0),  # euro
        ("\u00A3100", 100.0),  # pound
    ])
    def test_currency_stripped(self, text, expected):
        assert parse_numeric_answer(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text,expected", [
        ("1,234.56", 1234.56),
        ("1,000,000", 1_000_000.0),
        ("$1,234,567.89", 1_234_567.89),
    ])
    def test_thousands_separators(self, text, expected):
        assert parse_numeric_answer(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text,expected", [
        ("about 172", 172.0),
        ("approximately 170", 170.0),
        ("approx. 170", 170.0),
        ("around 170", 170.0),
        ("roughly 170", 170.0),
        ("~172", 172.0),
        ("\u2248 172", 172.0),  # ≈ approx symbol
    ])
    def test_approximation_prefix_dropped(self, text, expected):
        assert parse_numeric_answer(text) == pytest.approx(expected)


# ---------- parse_numeric_answer: range shapes ----------

class TestParseNumericRange:
    @pytest.mark.parametrize("text,lo,hi", [
        ("[172, 175]", 172.0, 175.0),
        ("(172, 175)", 172.0, 175.0),
        ("172 to 175", 172.0, 175.0),
        ("between 172 and 175", 172.0, 175.0),
        ("from 172 to 175", 172.0, 175.0),
        ("172-175", 172.0, 175.0),  # hyphen WITHOUT spaces = range
        ("172\u2013175", 172.0, 175.0),  # en-dash
        ("172\u2014175", 172.0, 175.0),  # em-dash
        ("172 \u2013 175", 172.0, 175.0),  # en-dash with spaces
    ])
    def test_range_forms(self, text, lo, hi):
        result = parse_numeric_answer(text)
        assert result == (lo, hi), f"input {text!r} → {result}, expected {(lo, hi)}"

    def test_hyphen_with_spaces_is_now_a_range(self):
        # r5 §2.1: hyphen-separated two-number answers are ranges, not
        # arithmetic, regardless of whether the hyphen has spaces.
        # The parser is an answer parser, not an expression evaluator.
        # Migration note in §8 item 11 calls this out.
        assert parse_numeric_answer("172 - 175") == (172.0, 175.0)
        # The unspaced form continues to behave as a range.
        assert parse_numeric_answer("172-175") == (172.0, 175.0)


# ---------- parse_numeric_answer: don't-know / unparseable ----------

class TestParseNumericFailure:
    @pytest.mark.parametrize("text", [
        None, "", "  ", "\t",
        "I don't know", "I dont know", "idk",
        "N/A", "n/a", "None", "unknown", "no answer",
    ])
    def test_dont_know_returns_none(self, text):
        assert parse_numeric_answer(text) is None

    @pytest.mark.parametrize("text", ["up", "higher", "yes", "down"])
    def test_categorical_text_returns_none(self, text):
        assert parse_numeric_answer(text) is None

    def test_pure_garbage_returns_none(self):
        assert parse_numeric_answer("...") is None
        assert parse_numeric_answer("hello world") is None


# ---------- parse_numeric_answer: multi-number permissive mode ----------

class TestParseNumericPermissive:
    def test_multi_number_default_returns_none(self):
        text = "the open was 172.06 and the close was 175.30"
        assert parse_numeric_answer(text) is None

    def test_multi_number_permissive_returns_first(self):
        text = "the open was 172.06 and the close was 175.30"
        assert parse_numeric_answer(text, permissive=True) == pytest.approx(172.06)


# ---------- parse_numeric_answer: strict mode ----------

class TestParseNumericStrict:
    def test_strict_raises_on_none(self):
        with pytest.raises(ValueError, match="input-presence"):
            parse_numeric_answer(None, strict=True)

    def test_strict_raises_on_dont_know(self):
        with pytest.raises(ValueError, match="don't-know-detection"):
            parse_numeric_answer("I don't know", strict=True)

    def test_strict_raises_on_unparseable(self):
        with pytest.raises(ValueError, match="number-extraction"):
            parse_numeric_answer("hello", strict=True)

    def test_strict_raises_on_multi_number(self):
        with pytest.raises(ValueError, match="multi-token disambiguation"):
            parse_numeric_answer(
                "open was 172 and close was 175", strict=True,
            )

    def test_strict_message_includes_input_and_suggestion(self):
        with pytest.raises(ValueError) as exc_info:
            parse_numeric_answer(
                "open 172 close 175", strict=True,
            )
        msg = str(exc_info.value)
        # Verbatim input (or its repr) is in the message.
        assert "172" in msg or "open" in msg
        # The suggestion mentions permissive=True.
        assert "permissive" in msg

    def test_strict_message_truncates_long_input(self):
        long = "x" * 1000
        with pytest.raises(ValueError) as exc_info:
            parse_numeric_answer(long, strict=True)
        msg = str(exc_info.value)
        # The truncation marker '...' shows up.
        assert "..." in msg


# ---------- parse_choice_answer ----------

class TestParseChoiceAnswer:
    UPDOWN = ["up", "down", "unchanged"]

    @pytest.mark.parametrize("text,expected", [
        ("up", "up"), ("higher", "up"), ("Higher", "up"), ("rose", "up"),
        ("down", "down"), ("DOWN", "down"), ("fell", "down"),
        ("flat", "unchanged"), ("Same", "unchanged"),
    ])
    def test_direction_normalized(self, text, expected):
        assert parse_choice_answer(text, self.UPDOWN) == expected

    def test_preserves_template_casing_for_uppercase_allowed(self):
        # If the template ships ["UP", "DOWN", "UNCHANGED"], the
        # parser must return the template's casing — not the canonical
        # lowercase form.
        result = parse_choice_answer("higher", ["UP", "DOWN", "UNCHANGED"])
        assert result == "UP"

    def test_unmatched_returns_none(self):
        assert parse_choice_answer("sideways", self.UPDOWN) is None

    def test_dont_know_returns_none(self):
        assert parse_choice_answer("N/A", self.UPDOWN) is None

    def test_strict_raises_on_unmatched(self):
        with pytest.raises(ValueError, match="choice-match"):
            parse_choice_answer("sideways", self.UPDOWN, strict=True)

    def test_strict_raises_on_none(self):
        with pytest.raises(ValueError, match="input-presence"):
            parse_choice_answer(None, self.UPDOWN, strict=True)

    def test_arbitrary_choices_case_insensitive(self):
        choices = ["BUY", "SELL", "HOLD"]
        assert parse_choice_answer("buy", choices) == "BUY"
        assert parse_choice_answer("Sell", choices) == "SELL"


# ---------- parse_binary_answer ----------

class TestParseBinaryAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("yes", True), ("Yes", True), ("YES", True), ("y", True),
        ("true", True), ("True", True), ("1", True),
        ("no", False), ("No", False), ("NO", False), ("n", False),
        ("false", False), ("False", False), ("0", False),
    ])
    def test_canonical_yes_no(self, text, expected):
        assert parse_binary_answer(text) is expected

    @pytest.mark.parametrize("text", ["up", "down", "higher", "lower"])
    def test_directional_aliases_do_not_overload_binary(self, text):
        # v2.0 fix: directional aliases live in normalize_direction, not
        # normalize_binary, because the meaning depends on the question
        # ("Was the close BELOW $60?" — "down" is hedge, not truth).
        assert parse_binary_answer(text) is None

    def test_unparseable_returns_none(self):
        assert parse_binary_answer("maybe") is None

    def test_dont_know_returns_none(self):
        assert parse_binary_answer("I don't know") is None

    def test_strict_raises_on_unparseable(self):
        with pytest.raises(ValueError, match="binary-match"):
            parse_binary_answer("maybe", strict=True)

    def test_strict_raises_on_none(self):
        with pytest.raises(ValueError, match="input-presence"):
            parse_binary_answer(None, strict=True)


# ---------- v2.0.1 r5 §2.1: percent handling ----------

class TestParseNumericPercent:
    def test_default_reject_returns_none(self):
        # Default percent="reject" — caller has not declared whether
        # the downstream template wants decimal or number form, so the
        # parser refuses to guess.
        assert parse_numeric_answer("2.5%") is None

    def test_decimal_divides_by_100(self):
        assert parse_numeric_answer("2.5%", percent="decimal") == 0.025

    def test_number_keeps_raw(self):
        assert parse_numeric_answer("2.5%", percent="number") == 2.5

    def test_strict_reject_raises(self):
        with pytest.raises(ValueError, match="percent-policy"):
            parse_numeric_answer("2.5%", strict=True)

    def test_negative_percent(self):
        assert parse_numeric_answer("-1.5%", percent="decimal") == -0.015
        assert parse_numeric_answer("\u22121.5%", percent="number") == -1.5

    def test_percent_with_whitespace(self):
        assert parse_numeric_answer("  2.5 %  ", percent="number") == 2.5

    def test_percent_only_when_trailing_sign(self):
        # "2.5" without % is not affected by the percent kwarg.
        assert parse_numeric_answer("2.5") == 2.5
        assert parse_numeric_answer("2.5", percent="decimal") == 2.5

    # ---- v2.0.2 #1: percent policy must survive prefixes ----

    def test_hedged_percent_default_rejects(self):
        # Realistic LLM output: hedge + percent. The percent="reject"
        # default safety must hold even with the "about" prefix.
        assert parse_numeric_answer("about 2.5%") is None
        assert parse_numeric_answer("approximately 2.5%") is None
        assert parse_numeric_answer("~2.5%") is None
        assert parse_numeric_answer("\u22482.5%") is None  # ≈

    def test_hedged_percent_with_decimal_policy(self):
        assert parse_numeric_answer(
            "about 2.5%", percent="decimal",
        ) == 0.025

    def test_hedged_percent_with_number_policy(self):
        assert parse_numeric_answer(
            "about 2.5%", percent="number",
        ) == 2.5

    def test_currency_prefixed_percent_rejects(self):
        # Nonsensical but possible: currency + percent. Must still
        # honor the policy rather than silently dropping the % sign.
        assert parse_numeric_answer("$2.5%") is None
        assert parse_numeric_answer("USD 2.5%") is None
        assert parse_numeric_answer("$2.5%", percent="decimal") == 0.025

    def test_negative_hedged_percent(self):
        # Combines hedge prefix + Unicode minus + percent.
        assert parse_numeric_answer(
            "about \u22121.5%", percent="decimal",
        ) == -0.015

    def test_strict_hedged_percent_raises(self):
        with pytest.raises(ValueError, match="percent-policy"):
            parse_numeric_answer("about 2.5%", strict=True)
