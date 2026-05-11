"""v2.2 M1: hashing utilities for AttestedAnswers.

This module is the single source of truth for the canonicalization
rules used by ``parsing_schema_hash`` and ``prompt_template_hash``.
Two reports are score-comparable iff they share both hashes — and
they will share both hashes iff their inputs canonicalize to the
same bytes here.

See ``docs/plans/v2.2-plan-r6.md`` § 2.3 for the full contract.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import ContextSerializationSpec


# Built-in parser names whose behavior depends on module-level
# constants (``_DONT_KNOW_PATTERNS``, ``_APPROX_PREFIXES``) in
# ``probes/scoring.py``. Calling ``attest_parsing`` with one of
# these names AND ``version="unpinned"`` raises, forcing the user
# to pin the AIphaForge release string so silent drift between
# releases is impossible.
_BUILTIN_PARSER_NAMES = frozenset({
    "parse_numeric_answer",
    "parse_choice_answer",
    "parse_binary_answer",
})


def _canonical_json(obj: Any) -> str:
    """Pinned JSON canonicalization — stdlib only, locked kwargs.

    Third-party canonicalizers (canonicaljson, RFC 8785 implementations)
    produce different output for the same input; swapping libraries
    silently drifts hashes across AIphaForge versions. r6 pins to
    stdlib with explicit kwargs.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_for_hash(s: str) -> str:
    """Pre-hash normalization — locked v2.2 (r6).

    Steps:
      1. Line-ending normalization: CRLF -> LF, lone CR -> LF.
      2. unicodedata.normalize('NFC', s).
      3. Strip Unicode category 'Cf' (Format) — ZWSP, ZWNJ, BOM,
         bidi controls, tag chars.
      4. Strip variation selectors U+FE00-U+FE0F and
         U+E0100-U+E01EF (category 'Mn', not 'Cf', so they survive
         step 3).

    NOT in scope (documented limitations):
      - ASCII whitespace folding beyond CRLF normalization.
      - Confusables / homoglyphs (Cyrillic vs Latin).
      - Locale-dependent case folding.
    """
    # 1. Line-ending normalization.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 2. NFC.
    s = unicodedata.normalize("NFC", s)
    # 3 + 4. Filter Cf and variation selectors.
    out_chars = []
    for ch in s:
        if unicodedata.category(ch) == "Cf":
            continue
        cp = ord(ch)
        if 0xFE00 <= cp <= 0xFE0F:
            continue
        if 0xE0100 <= cp <= 0xE01EF:
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _normalize_dict_strings(obj: Any) -> Any:
    """Recursively apply _normalize_for_hash to all string values
    inside a JSON-serializable structure (dicts, lists, tuples, scalars).
    """
    if isinstance(obj, str):
        return _normalize_for_hash(obj)
    if isinstance(obj, dict):
        return {k: _normalize_dict_strings(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_dict_strings(v) for v in obj]
    return obj


def attest_parsing(name: str, version: str, params: dict) -> str:
    """Stable SHA-256 hex digest of (name, version, JSON-canonical params).

    String params undergo _normalize_for_hash (NFC + Cf scrub) before
    hashing.

    ``version`` MUST be pinned to the AIphaForge release string when
    ``name`` matches a built-in parser whose behavior depends on
    module-level constants such as ``_DONT_KNOW_PATTERNS`` or
    ``_APPROX_PREFIXES``. Passing ``version="unpinned"`` for a
    built-in parser raises ``ValueError``.

    Rejects non-JSON-serializable params (callables, numpy arrays)
    with TypeError.
    """
    if name in _BUILTIN_PARSER_NAMES and version == "unpinned":
        raise ValueError(
            f"version='unpinned' not allowed for built-in parser "
            f"{name!r}. Pin to the AIphaForge release string "
            f"(e.g., aiphaforge.__version__) so changes to "
            f"_DONT_KNOW_PATTERNS / _APPROX_PREFIXES across "
            f"releases are reflected in the hash."
        )
    payload = {
        "name": name,
        "version": version,
        "params": _normalize_dict_strings(params),
    }
    # _canonical_json raises TypeError on non-serializable values
    # (callables, numpy arrays, etc.) — desired behavior.
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def attest_prompt_template(
    *,
    system_prompt: str,
    user_template: str,
    context_serialization: "ContextSerializationSpec",
    answer_format_instructions: str,
    few_shot_exemplars: tuple = (),
) -> str:
    """SHA-256 over the JSON-canonical concatenation.

    String inputs (``system_prompt``, ``user_template``,
    ``answer_format_instructions``, ``ExemplarSpec`` text fields) pass
    through ``_normalize_for_hash``.

    Few-shot exemplars hashed in submission order (NOT sorted) — order
    is part of prompt semantics. Each exemplar must be an
    ``ExemplarSpec`` instance; passing a raw dict raises TypeError.
    """
    # Avoid circular import; resolve at call time.
    from .models import ContextSerializationSpec, ExemplarSpec

    if not isinstance(context_serialization, ContextSerializationSpec):
        raise TypeError(
            "context_serialization must be a ContextSerializationSpec "
            "instance; raw dicts are rejected to prevent silent "
            "schema drift"
        )
    for i, ex in enumerate(few_shot_exemplars):
        if not isinstance(ex, ExemplarSpec):
            raise TypeError(
                f"few_shot_exemplars[{i}] must be an ExemplarSpec "
                f"instance; got {type(ex).__name__}"
            )

    payload = {
        "system_prompt": _normalize_for_hash(system_prompt),
        "user_template": _normalize_for_hash(user_template),
        "context_serialization": asdict(context_serialization),
        "answer_format_instructions": _normalize_for_hash(
            answer_format_instructions
        ),
        "few_shot_exemplars": [
            {
                "prompt_text": _normalize_for_hash(ex.prompt_text),
                "answer_text": _normalize_for_hash(ex.answer_text),
                "source_symbol": ex.source_symbol,
                "source_timestamp_iso": ex.source_timestamp_iso,
                "metadata": _normalize_for_hash(ex.metadata),
            }
            for ex in few_shot_exemplars
        ],
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


__all__ = [
    "attest_parsing",
    "attest_prompt_template",
]
