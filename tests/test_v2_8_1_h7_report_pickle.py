"""v2.8.1 H7 — KnowledgeCheckReport pickle compatibility.

v2.8.0 declared KnowledgeCheckReport as @dataclass(frozen=True) with
MappingProxyType-wrapped fields. The wrap broke pickle round-trip
(pickle cannot serialize mappingproxy). v2.8 also hard-removed the
legacy bucket_delta_tango_ci field, leaving old pickles unloadable.

v2.8.1: switch to @dataclass(init=False, frozen=True) with a manual
__init__ that:
  - rejects positional construction (kwargs-only)
  - rejects unknown kwargs (TypeError on typos)
  - translates the legacy bucket_delta_tango_ci kwarg with a
    DeprecationWarning
  - replays __post_init__ in __setstate__ so MappingProxyType
    re-wraps after unpickle.
"""
from __future__ import annotations

import pickle
import warnings
from types import MappingProxyType

import pytest

from aiphaforge.probes.orchestrator import KnowledgeCheckReport
from aiphaforge.probes.scoring import QAProbeReport


def _empty_qa_probe_report(name: str = "real") -> QAProbeReport:
    """Smallest valid QAProbeReport. Field set verified by reading the
    QAProbeReport definition; this test only cares about the shape
    being acceptable to KnowledgeCheckReport, not about its values."""
    from dataclasses import fields, MISSING
    kwargs = {}
    for f in fields(QAProbeReport):
        if f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            # Sensible defaults per type hint:
            if f.type is int or "int" in str(f.type):
                kwargs[f.name] = 0
            elif f.type is float or "float" in str(f.type):
                kwargs[f.name] = 0.0
            elif f.type is str or "str" in str(f.type):
                kwargs[f.name] = ""
            elif "dict" in str(f.type):
                kwargs[f.name] = {}
            elif "list" in str(f.type) or "Sequence" in str(f.type):
                kwargs[f.name] = []
            elif "tuple" in str(f.type):
                kwargs[f.name] = ()
            else:
                kwargs[f.name] = None
    return QAProbeReport(**kwargs)


def _baseline_kwargs() -> dict:
    return dict(
        probe_kind="knowledge",
        real_score=_empty_qa_probe_report("real"),
        anchor_score=None,
        bucket_delta=None,
        paired_sign_test_p=None,
        paired_sign_test_n_positive=None,
        paired_sign_test_n_negative=None,
        paired_sign_test_n_ties=None,
        paired_sign_test_variant=None,
        real_parse_success_rate=1.0,
        real_refusal_rate=0.0,
        anchor_parse_success_rate=None,
        anchor_refusal_rate=None,
        real_effective_rate=1.0,
        anchor_effective_rate=None,
        anchor_validity="NO_ANCHOR",
        persistence_baseline_score=None,
        persistence_caveat=None,
        real_minus_persistence_bucket_delta=None,
        real_vs_persistence_sign_test_p=None,
        parsing_schema_hash="0" * 64,
        parsing_schema_description="test",
        prompt_template_hash="0" * 64,
        prompt_template_description="test",
    )


# --------------------------------------------------------------------
# Test 1 — full pickle round-trip preserves all fields
# --------------------------------------------------------------------

def test_knowledge_check_report_pickle_round_trip_preserves_all_fields():
    kwargs = _baseline_kwargs()
    kwargs["parser_used_distribution_real"] = {"json": 50, "regex": 50}
    kwargs["parser_used_distribution_anchor"] = {"json": 30, "regex": 70}
    kwargs["bucket_delta_ci"] = {"all": (-0.05, 0.05)}
    kwargs["notes"] = ("note-a", "note-b")
    original = KnowledgeCheckReport(**kwargs)

    blob = pickle.dumps(original)
    revived = pickle.loads(blob)

    # MappingProxyType fields preserved as MappingProxyType
    assert isinstance(revived.parser_used_distribution_real, MappingProxyType)
    assert isinstance(revived.parser_used_distribution_anchor, MappingProxyType)
    # Content equal
    assert dict(revived.parser_used_distribution_real) == {"json": 50, "regex": 50}
    assert dict(revived.parser_used_distribution_anchor) == {"json": 30, "regex": 70}

    # Other fields equal
    for f_name in ("probe_kind", "anchor_validity", "real_parse_success_rate",
                   "parsing_schema_hash", "notes"):
        assert getattr(original, f_name) == getattr(revived, f_name)
    assert revived.bucket_delta_ci == {"all": (-0.05, 0.05)}


# --------------------------------------------------------------------
# Test 2 — MappingProxyType is preserved (not lost as plain dict)
# --------------------------------------------------------------------

def test_knowledge_check_report_pickle_preserves_mapping_proxy_immutability():
    kwargs = _baseline_kwargs()
    kwargs["parser_used_distribution_real"] = {"a": 1}
    blob = pickle.dumps(KnowledgeCheckReport(**kwargs))
    revived = pickle.loads(blob)
    # Immutability test: TypeError on mutate attempt.
    with pytest.raises(TypeError):
        revived.parser_used_distribution_real["b"] = 2  # type: ignore[index]


# --------------------------------------------------------------------
# Test 3 — legacy bucket_delta_tango_ci kwarg translates with warning
# --------------------------------------------------------------------

def test_knowledge_check_report_accepts_legacy_bucket_delta_tango_ci_kwarg():
    kwargs = _baseline_kwargs()
    kwargs["bucket_delta_tango_ci"] = {"all": (-0.1, 0.1)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = KnowledgeCheckReport(**kwargs)
    dep_warnings = [w for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and "bucket_delta_tango_ci" in str(w.message)]
    assert len(dep_warnings) == 1
    assert report.bucket_delta_ci == {"all": (-0.1, 0.1)}


# --------------------------------------------------------------------
# Test 4 — unknown kwarg raises TypeError (M4)
# --------------------------------------------------------------------

def test_knowledge_check_report_rejects_unknown_kwarg():
    kwargs = _baseline_kwargs()
    kwargs["bogus_field_does_not_exist"] = 42
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        KnowledgeCheckReport(**kwargs)


# --------------------------------------------------------------------
# Test 5 — positional construction now raises (kwargs-only)
# --------------------------------------------------------------------

def test_knowledge_check_report_rejects_positional_construction():
    with pytest.raises(TypeError):
        # Previously valid under @dataclass(frozen=True); now rejected.
        KnowledgeCheckReport("knowledge")  # type: ignore[misc]


# --------------------------------------------------------------------
# Test 6 — __setstate__ replays __post_init__ (M5)
# --------------------------------------------------------------------

def test_knowledge_check_report_setstate_replays_post_init():
    """If a pickle dropped MappingProxyType to plain dict during dump,
    __setstate__ + __post_init__ re-wraps it on load. Tested by
    constructing the state dict directly and calling __setstate__."""
    report = KnowledgeCheckReport(**_baseline_kwargs())
    state = report.__getstate__()
    # Simulate an old pickle / external state-dict where the proxy was
    # already unwrapped:
    state["parser_used_distribution_real"] = {"json": 1, "regex": 2}
    # Create a fresh blank instance via __new__ and replay:
    blank = KnowledgeCheckReport.__new__(KnowledgeCheckReport)
    blank.__setstate__(state)
    assert isinstance(blank.parser_used_distribution_real, MappingProxyType)
    assert dict(blank.parser_used_distribution_real) == {"json": 1, "regex": 2}


# --------------------------------------------------------------------
# Test 7 — post_init validation still fires via setstate (M5)
# --------------------------------------------------------------------

def test_knowledge_check_report_setstate_replays_post_init_validation():
    """is_pillar_summary=True is invalid; __setstate__ must catch it
    via __post_init__ replay."""
    state = KnowledgeCheckReport(**_baseline_kwargs()).__getstate__()
    state["is_pillar_summary"] = True
    blank = KnowledgeCheckReport.__new__(KnowledgeCheckReport)
    with pytest.raises(ValueError, match="is_pillar_summary"):
        blank.__setstate__(state)
