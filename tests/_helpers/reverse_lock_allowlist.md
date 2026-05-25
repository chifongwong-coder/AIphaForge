# Reverse-lock skip-rule audit findings (v2.8.2 Commit F)

This file documents the pre-implementation audit for v2.8.2 Commit F's
attempt to tighten `test_module_no_accidentally_public_symbols` in
`tests/test_v2_8_public_api_lock.py`.

## Background

The v2.8.1-introduced reverse-lock test catches "accidentally public"
symbols (no `_` prefix, not in `__all__`). To avoid false positives
on cross-module re-imports, it skips:

1. Re-exported modules (`inspect.ismodule`)
2. Symbols whose `__module__` differs from the current module
3. Primitive constants (`bool / int / float / str`) without
   `__module__`

The round-1 user-scenario reviewer (v2.8.1) flagged 3 escape hatches:
- 1.a: bytes/Decimal/datetime instances — foreign `__module__`,
       not class/function → skipped under rule 2
- 1.b: `numpy.nan` re-exports — primitive float, no `__module__`
       → skipped under rule 3
- 1.c: instances of foreign classes — same as 1.a

v2.8.2 round-2 architect N-V2-1 asked for an audit + allowlist file.

## Audit results

The proposed tightening was: in rule 2, require `inspect.isclass(val)
or inspect.isfunction(val)` before skipping. Anything else with foreign
`__module__` would surface.

Running the tightened rule against the current codebase surfaced
**205 flagged symbols**, virtually all of them typing primitives:

| Category | Examples | Count |
|---|---|---|
| Typing generics | `Dict`, `List`, `Optional`, `Tuple`, `Mapping`, `Literal`, `Iterable`, `Union`, `Sequence`, `Any`, `Callable` | ~180 |
| `__future__.annotations` | imported via `from __future__ import annotations` | ~20 |
| Other (collections.abc, typing_extensions) | misc | ~5 |

None of these is a real accidentally-public bug. They are legitimate
type-system imports that pre-date the v2.8 lock. Tightening the rule
without first allowlisting these would generate ~200 CI failures.

## v2.8.2 decision

**Tightening is DEFERRED to v2.8.5 (test hygiene release).** v2.8.5
will:

1. Hoist a shared typing-generic allowlist (`Dict`, `Optional`, etc.)
   into the test helper.
2. Tighten rule 2 with the allowlist applied AFTER `isclass/isfunction`.
3. Run the audit incrementally per-module as `__all__` exports get
   reviewed.

v2.8.2 Commit F ships:
- This audit file as a permanent record.
- 3 high-value v2.8.1-deferred tests (H1 dispatch pin, H6 stacklevel,
  ABC additive break).
- A documentation test pinning the current limitation (foreign-class
  instance passes the reverse lock; v2.8.5 will fix).

## v2.8.5 followup

When v2.8.5 ships the tightener, this file is updated with:
- Final allowlist constants (sourced from typing + __future__).
- Removal date for the documentation-test pin.

## v2.8.5 ship (post-merge fill at release)

Tightener landed in commit J of v2.8.5. The audit-script results at
release time and the in-test allowlist now match exactly.

### Allowlist constants (source-of-truth)

`tests/_helpers/reverse_lock_audit.py::_TYPING_ORIGIN_MODULES`
unioned with `{_FUTURE_MODULE}`:

| Module | Origin of |
|---|---|
| `typing` | `Dict`, `List`, `Optional`, `Tuple`, ... |
| `typing_extensions` | back-ported PEP 612/646 surfaces |
| `collections.abc` | `Iterable`, `Mapping`, `Sequence`, ... |
| `_collections_abc` | CPython internal mirror of the above |
| `__future__` | `annotations` |

The same set is duplicated as `_ALLOWED_FOREIGN_ORIGINS` in
`tests/test_v2_8_public_api_lock.py`. The lockstep is asserted by
`test_reverse_lock_allowlist_matches_audit_script`.

### Audit snapshot at release time

```
total_flagged: 205
by_category:
  typing_generics: 176
  future_imports:  29
  other:           0
```

A regression guard test
(`test_reverse_lock_other_category_remains_zero`) keeps the "other"
bucket at zero — any non-typing / non-future foreign-origin symbol
surfacing means a real accidentally-public surface that needs a `_`
prefix or an `__all__` entry.

### v2.8.2 documentation-test pin removed

`test_reverse_lock_currently_passes_foreign_class_instances` (in
`tests/test_v2_8_2_deferred_test_hardening.py`) was the documentation
pin that recorded the v2.8.1 rule-2 blind spot. v2.8.5 Commit J
removes the pin since the tightener now actively catches the
foreign-class-instance shape via the
`inspect.isclass / inspect.isfunction` predicate.
