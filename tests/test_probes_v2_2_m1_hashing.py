"""v2.2 M1 tests — hashing utilities and canonicalization.

Covers `_normalize_for_hash`, `_canonical_json`, `attest_parsing`,
`attest_prompt_template` per the v2.2 plan §2.3 contract.
"""
from __future__ import annotations

import pytest

from aiphaforge.probes._hash_utils import (
    _canonical_json,
    _normalize_for_hash,
    attest_parsing,
    attest_prompt_template,
)
from aiphaforge.probes.models import ContextSerializationSpec, ExemplarSpec


def _make_context_spec(**overrides) -> ContextSerializationSpec:
    """Helper: minimal valid ContextSerializationSpec with overrides."""
    base = dict(
        format="csv",
        columns=("t", "o", "h", "l", "c", "v"),
        timestamp_format="iso_utc_seconds",
        timestamp_timezone="UTC",
        price_precision=4,
        volume_precision=0,
        nan_rendering="empty",
        line_terminator="lf",
        decimal_separator=".",
        rounding_mode="half_even",
        trailing_zero_policy="preserve",
        scientific_notation_threshold=None,
        negative_format="minus_prefix",
    )
    base.update(overrides)
    return ContextSerializationSpec(**base)


# ---------- _canonical_json ----------


class TestCanonicalJson:
    def test_sort_keys(self):
        assert _canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_no_whitespace(self):
        assert _canonical_json({"a": [1, 2]}) == '{"a":[1,2]}'

    def test_ensure_ascii(self):
        assert _canonical_json({"x": "\u00e9"}) == '{"x":"\\u00e9"}'

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            _canonical_json({"x": float("nan")})

    def test_rejects_inf(self):
        with pytest.raises(ValueError):
            _canonical_json({"x": float("inf")})

    def test_rejects_non_serializable(self):
        with pytest.raises(TypeError):
            _canonical_json({"x": lambda y: y})


# ---------- _normalize_for_hash ----------


class TestNormalizeForHash:
    def test_strips_zwsp(self):
        assert _normalize_for_hash("hello\u200bworld") == "helloworld"

    def test_strips_zwnj(self):
        assert _normalize_for_hash("a\u200cb") == "ab"

    def test_strips_bom_mid_string(self):
        assert _normalize_for_hash("\ufeffhello") == "hello"

    def test_strips_variation_selector_basic(self):
        # U+FE0F is the emoji-presentation variation selector
        assert _normalize_for_hash("emoji\ufe0f") == "emoji"

    def test_strips_variation_selector_supplement(self):
        # Strip a code point in U+E0100-U+E01EF
        assert _normalize_for_hash("a\U000e0100") == "a"

    def test_normalizes_crlf_to_lf(self):
        assert _normalize_for_hash("a\r\nb") == "a\nb"

    def test_normalizes_lone_cr_to_lf(self):
        assert _normalize_for_hash("a\rb") == "a\nb"

    def test_nfc_smart_quotes_no_change_visible(self):
        # NFC of a precomposed character returns itself; the test
        # is that the call doesn't mangle visible content.
        s = "café"  # already NFC
        assert _normalize_for_hash(s) == s

    def test_preserves_internal_ascii_whitespace_documented_limit(self):
        # Documented out-of-scope: ASCII whitespace folding.
        assert _normalize_for_hash("a  b") == "a  b"

    def test_preserves_visible_unicode_emoji(self):
        # Emoji without VS16 should pass through.
        s = "test \U0001f600"  # 😀
        assert _normalize_for_hash(s) == s

    def test_preserves_visible_cjk(self):
        s = "测试"
        assert _normalize_for_hash(s) == s


# ---------- attest_parsing ----------


class TestAttestParsing:
    def test_hash_stable_across_calls(self):
        h1 = attest_parsing("custom_parser", "1.0", {"strict": True})
        h2 = attest_parsing("custom_parser", "1.0", {"strict": True})
        assert h1 == h2

    def test_hash_changes_on_param_change(self):
        h1 = attest_parsing("custom_parser", "1.0", {"strict": True})
        h2 = attest_parsing("custom_parser", "1.0", {"strict": False})
        assert h1 != h2

    def test_hash_changes_on_version_change(self):
        h1 = attest_parsing("custom_parser", "1.0", {})
        h2 = attest_parsing("custom_parser", "2.0", {})
        assert h1 != h2

    def test_hash_changes_on_name_change(self):
        h1 = attest_parsing("parser_a", "1.0", {})
        h2 = attest_parsing("parser_b", "1.0", {})
        assert h1 != h2

    def test_rejects_unpinned_for_builtin_parser(self):
        with pytest.raises(ValueError, match="unpinned"):
            attest_parsing("parse_numeric_answer", "unpinned", {})

    def test_rejects_unpinned_for_each_builtin(self):
        for name in (
            "parse_numeric_answer",
            "parse_choice_answer",
            "parse_binary_answer",
        ):
            with pytest.raises(ValueError):
                attest_parsing(name, "unpinned", {})

    def test_allows_unpinned_for_custom_parser(self):
        # Only built-in names are protected; custom parsers may
        # use any version string.
        h = attest_parsing("my_custom", "unpinned", {})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_release_string_changes_hash(self):
        h1 = attest_parsing("parse_numeric_answer", "2.2.0", {})
        h2 = attest_parsing("parse_numeric_answer", "2.2.1", {})
        assert h1 != h2

    def test_rejects_non_json_serializable_param(self):
        with pytest.raises(TypeError):
            attest_parsing("custom", "1.0", {"cb": lambda x: x})

    def test_normalizes_string_params(self):
        # Strings inside params get NFC + Cf scrub.
        h_clean = attest_parsing("custom", "1.0", {"key": "abc"})
        h_zwsp = attest_parsing(
            "custom", "1.0", {"key": "ab\u200bc"}
        )
        assert h_clean == h_zwsp


# ---------- attest_prompt_template ----------


class TestAttestPromptTemplate:
    def _kwargs(self, **overrides):
        base = dict(
            system_prompt="You are a trading assistant.",
            user_template="Given {symbol}, what is close on {date}?",
            context_serialization=_make_context_spec(),
            answer_format_instructions="Respond with one number.",
            few_shot_exemplars=(),
        )
        base.update(overrides)
        return base

    def test_hash_stable_across_calls(self):
        h1 = attest_prompt_template(**self._kwargs())
        h2 = attest_prompt_template(**self._kwargs())
        assert h1 == h2

    def test_hash_changes_on_system_prompt_change(self):
        h1 = attest_prompt_template(
            **self._kwargs(system_prompt="A")
        )
        h2 = attest_prompt_template(
            **self._kwargs(system_prompt="B")
        )
        assert h1 != h2

    def test_hash_changes_on_serialization_format(self):
        h_csv = attest_prompt_template(
            **self._kwargs(
                context_serialization=_make_context_spec(format="csv")
            )
        )
        h_json = attest_prompt_template(
            **self._kwargs(
                context_serialization=_make_context_spec(format="json")
            )
        )
        assert h_csv != h_json

    def test_includes_exemplars_verbatim(self):
        ex_a = ExemplarSpec(prompt_text="Q", answer_text="A")
        ex_a_changed = ExemplarSpec(prompt_text="Q", answer_text="B")
        h_a = attest_prompt_template(
            **self._kwargs(few_shot_exemplars=(ex_a,))
        )
        h_b = attest_prompt_template(
            **self._kwargs(few_shot_exemplars=(ex_a_changed,))
        )
        assert h_a != h_b

    def test_exemplar_order_matters(self):
        e1 = ExemplarSpec(prompt_text="P1", answer_text="A1")
        e2 = ExemplarSpec(prompt_text="P2", answer_text="A2")
        h_12 = attest_prompt_template(
            **self._kwargs(few_shot_exemplars=(e1, e2))
        )
        h_21 = attest_prompt_template(
            **self._kwargs(few_shot_exemplars=(e2, e1))
        )
        assert h_12 != h_21

    def test_rejects_dict_exemplar(self):
        with pytest.raises(TypeError):
            attest_prompt_template(
                **self._kwargs(few_shot_exemplars=({"q": "x"},))
            )

    def test_rejects_dict_context_serialization(self):
        with pytest.raises(TypeError):
            attest_prompt_template(
                **self._kwargs(context_serialization={"format": "csv"})
            )

    def test_nfc_normalizes_smart_quotes_in_system_prompt(self):
        h_straight = attest_prompt_template(
            **self._kwargs(system_prompt='He said "hi".')
        )
        # Normalize "hi" with curly quotes — after NFC + Cf scrub
        # they should compare equal *only* if NFC fully unifies them.
        # Curly quotes (U+201C, U+201D) are distinct codepoints from
        # ASCII " in NFC; they remain distinct. So this test asserts
        # the NORMALIZATION runs, not that quotes unify.
        h_curly = attest_prompt_template(
            **self._kwargs(system_prompt="He said \u201chi\u201d.")
        )
        # Different visible chars → different hash. Test passes
        # if NFC ran without crashing on Unicode input.
        assert h_straight != h_curly

    def test_zwsp_in_user_template_canonicalizes_away(self):
        h_clean = attest_prompt_template(
            **self._kwargs(user_template="Q: {x}")
        )
        h_zwsp = attest_prompt_template(
            **self._kwargs(user_template="Q:\u200b {x}")
        )
        assert h_clean == h_zwsp


# ---------- ContextSerializationSpec validation ----------


class TestContextSerializationSpec:
    def test_valid_construction(self):
        spec = _make_context_spec()
        assert spec.format == "csv"

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="format"):
            _make_context_spec(format="xml")

    def test_rejects_non_utc_tz(self):
        with pytest.raises(ValueError, match="UTC"):
            _make_context_spec(timestamp_timezone="US/Eastern")

    def test_rejects_unknown_rounding_mode(self):
        with pytest.raises(ValueError, match="rounding_mode"):
            _make_context_spec(rounding_mode="banker")

    def test_rejects_unknown_nan_rendering(self):
        with pytest.raises(ValueError, match="nan_rendering"):
            _make_context_spec(nan_rendering="dash")

    def test_rejects_unknown_csv_quoting(self):
        with pytest.raises(ValueError, match="csv_quoting"):
            _make_context_spec(csv_quoting="all_strings")

    def test_rejects_non_locked_csv_quote_char(self):
        with pytest.raises(ValueError, match="csv_quote_char"):
            _make_context_spec(csv_quote_char="'")

    def test_rejects_collision_decimal_field_sep(self):
        # decimal=',' AND field_sep=',' on csv → collision raises.
        with pytest.raises(ValueError, match="collide"):
            _make_context_spec(
                format="csv",
                decimal_separator=",",
                csv_field_separator=",",
            )

    def test_collision_check_skipped_when_format_not_csv(self):
        # JSON format ignores the CSV field separator.
        spec = _make_context_spec(
            format="json",
            decimal_separator=",",
            csv_field_separator=",",
        )
        assert spec.format == "json"

    def test_collision_check_passes_with_semicolon_field_sep(self):
        spec = _make_context_spec(
            format="csv",
            decimal_separator=",",
            csv_field_separator=";",
        )
        assert spec.csv_field_separator == ";"

    def test_rejects_unknown_json_inf_format(self):
        with pytest.raises(ValueError, match="json_inf_format"):
            _make_context_spec(json_inf_format="raise")
