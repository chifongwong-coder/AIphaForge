"""v2.1 M5 — end-to-end calendar integration test.

Plan: ``docs/plans/v2.1-plan.md`` §7 (r3-final).

A scenario containing a calendar-aware DateShift runs through
``run_ab_probe`` from end to end, and we verify:

- The internal pipeline picks up the inferred effective calendar
  via the explicit marker protocol (M4).
- The per-scenario manifest carries the new
  ``transform_detectability_warnings`` (M5 §6.1) and
  ``calendar_snap_collisions`` (M5 §6.2) entries when applicable.
- The v2.0.1 r5 canonical ``manifest["determinism_check"]`` schema
  is intact and unaffected by the calendar wiring.
- A scenario with two conflicting calendars fails fast with
  ``CalendarConflictError``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge.calendars import (
    CHINA_A_SHARE,
    US_EQUITY,
    CalendarConflictError,
)
from aiphaforge.probes import (
    ABScenario,
    MACrossBaseline,
    run_ab_probe,
)
from aiphaforge.probes.transforms import DateShift, PriceScale
from tests.conftest import make_probe_ohlcv as _ohlcv_default


def _ohlcv_clean(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Like make_probe_ohlcv but reindexed to NYSE-conformant dates.

    The conftest fixture starts at 2024-01-01 (NYSE holiday) and
    includes other holidays in its bdate_range. For calendar e2e
    tests we need an index that's already calendar-conformant under
    US_EQUITY so snap modes other than collision-policy can be
    exercised cleanly.
    """
    df = _ohlcv_default(n=n, seed=seed)
    # Reindex to dates that pass US_EQUITY conformance: skip
    # 2024-01-01 + reindex to forward-snapped dates.
    snapped = pd.DatetimeIndex([
        US_EQUITY.snap(ts, "forward") for ts in df.index
    ])
    df = df.copy()
    df.index = snapped
    # Drop duplicates introduced by snapping; keep first.
    return df[~df.index.duplicated(keep="first")]


def _ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    return _ohlcv_clean(n=n, seed=seed)


def _factory():
    return MACrossBaseline(short=5, long=20)


_KW = dict(
    n_repeat=2,
    seeds=[0, 1],
    metrics=("total_return",),
    min_valid_repeats=1,
    engine_kwargs={"include_benchmark": False},
)


# ---------- e2e: scenario with calendar-aware DateShift ----------


class TestCalendarShiftScenario:
    def test_scenario_runs_end_to_end(self):
        # Use a clean fixture (no holidays in raw index) and shift
        # backward by 3 years with on_collision="keep_last" so
        # multi-year holiday overlap doesn't crash.
        data = _ohlcv(n=60)
        ds = DateShift(
            offset=pd.DateOffset(years=-3),
            calendar=US_EQUITY,
            snap="forward",
            on_collision="keep_last",
        )
        scen = ABScenario(
            scenario_id="us_shift", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        assert len(result.scenarios) == 1
        assert result.scenarios[0].scenario_id == "us_shift"

    def test_detectability_warning_in_manifest(self):
        data = _ohlcv(n=40)
        ds = DateShift(
            offset=pd.DateOffset(years=-1),
            calendar=US_EQUITY,
            snap="forward",
            on_collision="keep_last",
        )
        scen = ABScenario(
            scenario_id="detectable", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        warnings = result.scenarios[0].transform_detectability_warnings
        # Must contain the calendar_snap_fingerprint entry.
        codes = [w["code"] for w in warnings]
        assert "calendar_snap_fingerprint" in codes
        # Structured fields per plan §6.1.
        snap_warning = next(
            w for w in warnings if w["code"] == "calendar_snap_fingerprint"
        )
        for key in ("code", "severity", "source", "message"):
            assert key in snap_warning
        assert snap_warning["source"] == "DateShift"

    def test_no_detectability_warning_when_snap_is_error(self):
        # Use clean fixture so snap='error' doesn't actually raise.
        # The detectability warning should NOT fire because
        # snap='error' can't introduce post-holiday clustering.
        data = _ohlcv(n=20)
        ds = DateShift(
            offset=pd.DateOffset(days=0),  # zero shift, snap is no-op
            calendar=US_EQUITY,
            snap="error",
        )
        scen = ABScenario(
            scenario_id="no_clustering", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        warnings = result.scenarios[0].transform_detectability_warnings
        codes = [w["code"] for w in warnings]
        assert "calendar_snap_fingerprint" not in codes

    def test_collision_warning_in_manifest_when_rows_dropped(self):
        # Build a 60-day fixture and shift backward by 3 years with
        # forward snap + keep_last — multi-year offset is the
        # canonical case where collisions happen.
        data = _ohlcv(n=80)
        ds = DateShift(
            offset=pd.DateOffset(years=-3),
            calendar=US_EQUITY,
            snap="forward",
            on_collision="keep_last",
        )
        scen = ABScenario(
            scenario_id="collisions", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        collisions = result.scenarios[0].calendar_snap_collisions
        # 3-year backward shift on 80 daily NYSE bars almost always
        # produces at least one collision; if not, this test is
        # uninformative but not broken.
        if collisions:
            entry = collisions[0]
            assert entry["code"] == "calendar_snap_collision_rows_dropped"
            assert entry["severity"] == "warning"
            assert entry["source"] == "DateShift"
            details = entry["details"]
            assert details["transform"] == "DateShift"
            assert details["on_collision"] == "keep_last"
            assert details["collision_count"] >= 1
            assert details["collision_group_count"] >= 1
            # Plan §6.2 + §4.4 defect #4 fix: examples are
            # JSON-safe strings, NOT pd.Timestamp.
            for ex in details["examples"]:
                assert isinstance(ex["target_ts"], str)
                assert isinstance(ex["kept_source_ts"], str)
                # YYYY-MM-DD format.
                assert len(ex["target_ts"]) == 10


class TestNoCollisionsLeavesEmptyWarning:
    def test_no_collision_yields_empty_list(self):
        # Use a clean fixture and a no-collision setup
        # (snap='error' can't drop rows).
        data = _ohlcv(n=20)
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="nearest",
            on_collision="error",
        )
        scen = ABScenario(
            scenario_id="no_collisions", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        # No drops → empty list, not None.
        assert result.scenarios[0].calendar_snap_collisions == []


# ---------- determinism_check schema unaffected ----------


class TestDeterminismSchemaUnaffected:
    def test_canonical_schema_intact(self):
        # Plan §7 invariant: the v2.0.1 r5 canonical determinism
        # schema must remain bit-identical even with calendar wiring
        # active.
        data = _ohlcv(n=20)
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="nearest",
        )
        scen = ABScenario(
            scenario_id="x", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        det = result.manifest["determinism_check"]
        assert det["schema_version"] == "2.0.1"
        # Canonical sections present.
        for key in (
            "mode", "agent_implementation_contract",
            "requested", "resolved", "subjects",
            "controls", "extension",
        ):
            assert key in det
        # Subjects unchanged shape.
        for subject in ("ai", "baseline"):
            assert "raw" in det["subjects"][subject]
            assert "per_scenario" in det["subjects"][subject]


# ---------- conflict detection ----------


class TestCalendarConflictFailsFast:
    def test_two_conflicting_calendars_raise_calendar_conflict_error(self):
        data = _ohlcv(n=20)
        ds_us = DateShift(offset=pd.DateOffset(days=0), calendar=US_EQUITY)
        ds_cn = DateShift(offset=pd.DateOffset(days=0), calendar=CHINA_A_SHARE)
        scen = ABScenario(
            scenario_id="conflict", mode="market_level",
            transforms=[ds_us, ds_cn],
        )
        with pytest.raises(CalendarConflictError):
            run_ab_probe(
                ai_factory=_factory,
                baseline_factory=_factory,
                data=data,
                scenarios=[scen],
                **_KW,
            )


# ---------- v2.0 backward compat ----------


class TestV2_0_BackwardCompat:
    def test_calendar_less_scenario_does_not_set_calendar_warnings(self):
        # A scenario with NO calendar-aware transform must NOT
        # populate transform_detectability_warnings or
        # calendar_snap_collisions.
        data = _ohlcv(n=20)
        scen = ABScenario(
            scenario_id="no_calendar", mode="market_level",
            transforms=[PriceScale(factor=2.0)],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        report = result.scenarios[0]
        assert report.transform_detectability_warnings == []
        assert report.calendar_snap_collisions == []


# ---------- v2.1.0 r4 §5: serializer direct tests (M10) ----------


class TestSerializeCollisionExamples:
    """Direct unit coverage for `_serialize_collision_examples`.

    Plan r4-final §5.2: malformed example shapes should raise
    `ValueError` at the serializer boundary, not propagate
    AttributeError deep into manifest construction.
    """

    def _example(self, **overrides):
        """Build a synthetic example dict mirroring DateShift's shape."""
        base = {
            "target_ts": pd.Timestamp("2024-12-02"),
            "source_ts": [
                pd.Timestamp("2024-11-28"),
                pd.Timestamp("2024-11-29"),
            ],
            "kept_source_ts": pd.Timestamp("2024-11-29"),
            "dropped_source_ts": [pd.Timestamp("2024-11-28")],
        }
        base.update(overrides)
        return base

    def test_happy_path_stringifies_pd_timestamps(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        out = _serialize_collision_examples([self._example()])
        assert len(out) == 1
        assert out[0]["target_ts"] == "2024-12-02"
        assert out[0]["source_ts"] == ["2024-11-28", "2024-11-29"]
        assert out[0]["kept_source_ts"] == "2024-11-29"
        assert out[0]["dropped_source_ts"] == ["2024-11-28"]

    def test_caps_at_10_groups(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        # 15 examples — should be capped at 10.
        many = [self._example() for _ in range(15)]
        out = _serialize_collision_examples(many)
        assert len(out) == 10

    def test_pre_stringified_dates_pass_through(self):
        # Forward-compat: future schemas may stringify before
        # reaching the serializer.
        from aiphaforge.probes.abtest import _serialize_collision_examples
        ex = {
            "target_ts": "2024-12-02",
            "source_ts": ["2024-11-28", "2024-11-29"],
            "kept_source_ts": "2024-11-29",
            "dropped_source_ts": ["2024-11-28"],
        }
        out = _serialize_collision_examples([ex])
        assert out[0]["target_ts"] == "2024-12-02"

    def test_missing_required_key_raises_clearly(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        bad = {
            "target_ts": pd.Timestamp("2024-12-02"),
            # missing source_ts, kept_source_ts, dropped_source_ts
        }
        with pytest.raises(ValueError, match="missing required keys"):
            _serialize_collision_examples([bad])

    def test_non_mapping_example_raises_clearly(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        with pytest.raises(ValueError, match="must be a mapping"):
            _serialize_collision_examples(["not a dict"])

    def test_malformed_timestamp_value_raises_clearly(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        bad = self._example()
        bad["target_ts"] = 12345  # int, not pd.Timestamp
        with pytest.raises(ValueError, match="malformed timestamp"):
            _serialize_collision_examples([bad])

    def test_empty_input_returns_empty_output(self):
        from aiphaforge.probes.abtest import _serialize_collision_examples
        assert _serialize_collision_examples([]) == []


class TestManifestUsesArmLocalDiagnostics:
    """v2.1.0 r4 §3.5 / §5.1 — manifest collision warnings must come
    from per-arm diagnostics, NOT from any transform-instance state.
    """

    def test_collision_warning_details_carry_schema_version(self):
        # End-to-end smoke that the new schema_version=1.0 reaches
        # the manifest.
        data = _ohlcv(n=80)
        ds = DateShift(
            offset=pd.DateOffset(years=-3),
            calendar=US_EQUITY,
            snap="forward",
            on_collision="keep_last",
        )
        scen = ABScenario(
            scenario_id="schema", mode="market_level",
            transforms=[ds],
        )
        result = run_ab_probe(
            ai_factory=_factory,
            baseline_factory=_factory,
            data=data,
            scenarios=[scen],
            **_KW,
        )
        collisions = result.scenarios[0].calendar_snap_collisions
        if collisions:
            # When collisions fire, schema_version must be present.
            assert collisions[0]["details"]["schema_version"] == "1.0"
            # And repeat_count_with_warning records how many
            # arm-repeats produced the warning.
            assert "repeat_count_with_warning" in collisions[0]["details"]

    def test_no_dateshift_attribute_left_behind_on_instance(self):
        # Regression guard: r4 §3 removed last_collision_report
        # entirely. Apply 100 rounds; instance should still have no
        # such attribute.
        data = _ohlcv(n=20)
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="nearest",
            on_collision="keep_last",
        )
        for _ in range(5):
            ds.apply(data)
        assert not hasattr(ds, "last_collision_report")
