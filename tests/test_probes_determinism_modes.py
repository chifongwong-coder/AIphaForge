"""v2.0.1 r5 M4 — tests for the determinism check rework.

Covers plan §5 (resolver, profile semantics, one-arm result dataclass,
canonical schema + legacy mirrors, replayability, status-based
warnings, view-only-hook unsupported handling, NaN/inf serialization,
one-engine-pair-per-arm cost guard).
"""
from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from aiphaforge.probes import ABScenario, MACrossBaseline, run_ab_probe
from aiphaforge.probes.abtest import (
    _arm_result_to_json,
    _check_agent_determinism,
    _legacy_pass,
    _preflight_unsupported,
    _stable_view_fingerprint,
    resolve_determinism_config,
)
from aiphaforge.probes.models import (
    AgentContract,
    AgentImplementationContract,
    DeterminismCheckResult,
    UnsupportedScenarioError,
)
from aiphaforge.probes.transforms import OHLCJitter, PriceScale
from tests.conftest import make_probe_ohlcv as _ohlcv


def _det_factory():
    return MACrossBaseline(short=5, long=20)


def _scenario():
    return ABScenario(
        scenario_id="ps", mode="market_level",
        transforms=[PriceScale(factor=2.0)],
    )


_KW = dict(
    n_repeat=2, seeds=[0, 1],
    metrics=("total_return", "trade_count"),
    min_valid_repeats=1,
    engine_kwargs={"include_benchmark": False},
)


# ---------- Public-name protection ----------

class TestNameCollisionProtection:
    def test_existing_models_agent_contract_not_shadowed(self):
        # Plan §5.1: the existing `AgentContract` (order-shape literal)
        # must remain untouched. The new implementation-shape literal
        # is a separate name.
        from aiphaforge.probes import models as probes_models
        assert probes_models.AgentContract.__args__ == (
            "signal_only", "market_orders_only", "price_orders_allowed",
        )
        assert probes_models.AgentImplementationContract.__args__ == (
            "strategy", "hook", "hook_view_only_capable", "callable_factory",
        )
        # And they must be distinct objects.
        assert AgentContract is not AgentImplementationContract

    def test_agent_implementation_contract_kwarg_uses_new_name(self):
        # Kwarg name is `agent_implementation_contract`, not
        # `agent_contract` (which already exists for the order-shape
        # gate).
        sig = signature(run_ab_probe)
        assert "agent_implementation_contract" in sig.parameters
        # Old kwarg still exists for the order-shape concept.
        assert "agent_contract" in sig.parameters


# ---------- Config resolver ----------

class TestResolveDeterminismConfig:
    def test_auto_raw_only_is_v2_compat(self):
        r = resolve_determinism_config(
            mode="raw_only", profile="auto",
            determinism_metrics=None, determinism_rel_tol=None,
        )
        assert r.profile == "v2_compat"
        assert r.determinism_metrics == ("total_return",)
        assert r.determinism_rel_tol == 1e-12

    def test_auto_per_scenario_is_llm_balanced(self):
        r = resolve_determinism_config(
            mode="per_scenario", profile="auto",
            determinism_metrics=None, determinism_rel_tol=None,
        )
        assert r.profile == "llm_balanced"
        assert r.determinism_metrics == (
            "total_return", "num_trades", "win_rate",
        )
        assert r.determinism_rel_tol == 1e-3

    def test_auto_off_resolves_to_off(self):
        r = resolve_determinism_config(
            mode="off", profile="auto",
            determinism_metrics=None, determinism_rel_tol=None,
        )
        assert r.profile == "off"
        assert r.determinism_metrics == ()
        assert r.determinism_rel_tol is None

    def test_explicit_determinism_metrics_override_profile_metrics(self):
        r = resolve_determinism_config(
            mode="per_scenario", profile="auto",
            determinism_metrics=("sharpe_ratio",),
            determinism_rel_tol=None,
        )
        # Profile resolves to llm_balanced but the explicit metrics win.
        assert r.profile == "llm_balanced"
        assert r.determinism_metrics == ("sharpe_ratio",)
        # Tolerance still comes from the profile.
        assert r.determinism_rel_tol == 1e-3

    def test_explicit_determinism_rel_tol_override_profile_tol(self):
        r = resolve_determinism_config(
            mode="raw_only", profile="auto",
            determinism_metrics=None, determinism_rel_tol=5e-2,
        )
        assert r.profile == "v2_compat"
        # Profile metrics, but explicit tol.
        assert r.determinism_metrics == ("total_return",)
        assert r.determinism_rel_tol == 5e-2

    def test_explicit_profile_overrides_auto(self):
        r = resolve_determinism_config(
            mode="raw_only", profile="llm_balanced",
            determinism_metrics=None, determinism_rel_tol=None,
        )
        assert r.profile == "llm_balanced"
        assert r.determinism_rel_tol == 1e-3

    def test_requested_fields_preserved(self):
        r = resolve_determinism_config(
            mode="per_scenario", profile="v2_compat",
            determinism_metrics=("total_return",), determinism_rel_tol=1e-8,
        )
        assert r.requested_profile == "v2_compat"
        assert r.requested_metrics == ("total_return",)
        assert r.requested_rel_tol == 1e-8


# ---------- Mode-specific shapes ----------

class TestModes:
    def test_off_has_stable_shape_and_status(self):
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="off", **_KW,
        )
        m = result.manifest
        assert m["agent_determinism_check_mode"] == "off"
        assert m["ai_determinism_check_passed"] is None
        assert m["baseline_determinism_check_passed"] is None
        assert m["ai_determinism_check_status"] == "off"
        assert m["baseline_determinism_check_status"] == "off"
        assert m["ai_determinism_failed_metrics"] == []
        assert m["baseline_determinism_failed_metrics"] == []
        assert m["ai_determinism_check_per_scenario"] == {}
        assert m["baseline_determinism_check_per_scenario"] == {}
        # Canonical block: subjects.ai.raw is None.
        assert m["determinism_check"]["subjects"]["ai"]["raw"] is None
        assert m["determinism_check"]["subjects"]["baseline"]["raw"] is None

    def test_raw_only_default_resolves_to_v2_compat(self):
        # The default kwarg shape must preserve v2.0 raw behavior.
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            **_KW,  # no agent_determinism_check= → default "raw_only"
        )
        m = result.manifest
        assert m["agent_determinism_check_mode"] == "raw_only"
        assert m["determinism_profile"] == "v2_compat"
        assert m["determinism_metrics"] == ["total_return"]
        assert m["determinism_rel_tol"] == 1e-12
        assert m["ai_determinism_check_passed"] is True
        assert m["baseline_determinism_check_passed"] is True
        # Per-scenario dicts are empty in raw_only mode.
        assert m["ai_determinism_check_per_scenario"] == {}

    def test_per_scenario_default_resolves_to_llm_balanced(self):
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="per_scenario", **_KW,
        )
        m = result.manifest
        assert m["determinism_profile"] == "llm_balanced"
        assert m["determinism_metrics"] == [
            "total_return", "num_trades", "win_rate",
        ]
        assert m["determinism_rel_tol"] == 1e-3


# ---------- Per-scenario manifest shape ----------

class TestPerScenarioManifest:
    def _result(self):
        return run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="per_scenario", **_KW,
        )

    def test_per_scenario_manifest_shape_has_raw_and_transformed(self):
        m = self._result().manifest
        entry = m["ai_determinism_check_per_scenario"]["ps"]
        # Plan §5.12 mapping table — both arms present.
        for k in (
            "raw_passed", "raw_status", "raw_failed_metrics",
            "raw_metric_values_run_1", "raw_metric_values_run_2",
            "raw_error_type", "raw_error_message",
            "transformed_passed", "transformed_status",
            "transformed_failed_metrics",
            "transformed_metric_values_run_1",
            "transformed_metric_values_run_2",
            "transformed_error_type", "transformed_error_message",
        ):
            assert k in entry, f"missing key {k!r} in per-scenario entry"
        # And specifically NOT a naked bool — round-1 regression.
        assert not isinstance(entry, bool)

    def test_per_scenario_mapping_table_is_followed(self):
        # Plan §5.12: every flat key derives from the corresponding
        # ArmResult key. Cross-check against the canonical block.
        m = self._result().manifest
        canonical_ai = m["determinism_check"]["subjects"]["ai"]["per_scenario"]["ps"]
        flat = m["ai_determinism_check_per_scenario"]["ps"]
        assert flat["raw_passed"] == canonical_ai["raw"]["passed"]
        assert flat["raw_status"] == canonical_ai["raw"]["status"]
        assert flat["raw_failed_metrics"] == canonical_ai["raw"]["failed_metrics"]
        assert flat["transformed_passed"] == canonical_ai["transformed"]["passed"]
        assert flat["transformed_status"] == canonical_ai["transformed"]["status"]

    def test_per_scenario_raw_fields_duplicated_intentionally_without_rerun(self):
        # Plan §5.13: raw fields ARE duplicated under each scenario_id
        # for self-contained export. Multi-scenario probe should show
        # identical raw_metric_values for every scenario id.
        scenarios = [
            ABScenario(scenario_id=f"s{i}", mode="market_level",
                       transforms=[PriceScale(factor=float(2 + i))])
            for i in range(3)
        ]
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), scenarios,
            agent_determinism_check="per_scenario", **_KW,
        )
        per_scen = result.manifest["ai_determinism_check_per_scenario"]
        raw_run1 = [per_scen[s.scenario_id]["raw_metric_values_run_1"] for s in scenarios]
        # All scenarios share the same raw run values byte-for-byte.
        assert raw_run1[0] == raw_run1[1] == raw_run1[2]


# ---------- Cost guard ----------

class TestMultiMetricCost:
    def test_multi_metric_runs_one_pair_not_per_metric(self, monkeypatch):
        # Plan §5.7 / §5.16 #7. Three metrics → 2 engine runs per arm,
        # not 6. Counted via monkeypatch on _run_one_arm so the count
        # is the only invariant under test.
        from aiphaforge.probes import abtest as ab_mod

        original = ab_mod._run_one_arm
        call_counter = {"n": 0}

        def counting(*a, **kw):
            call_counter["n"] += 1
            return original(*a, **kw)

        monkeypatch.setattr(ab_mod, "_run_one_arm", counting)

        resolved = resolve_determinism_config(
            mode="raw_only", profile="llm_balanced",
            determinism_metrics=("total_return", "num_trades", "win_rate"),
            determinism_rel_tol=1e-3,
        )
        _check_agent_determinism(
            _det_factory, _ohlcv(n=30),
            resolved_config=resolved,
            seed=0, engine_kwargs={"include_benchmark": False},
        )
        # Exactly one engine pair per arm.
        assert call_counter["n"] == 2


# ---------- Unsupported scenario handling ----------

class TestUnsupportedScenario:
    def test_view_only_hook_unsupported_is_not_determinism_failure(self):
        # Plan §5.5 / §5.15: view_only with a plain hook is
        # unsupported. The transformed-arm result must be
        # status="unsupported", passed=None, NOT a failure.
        scen = ABScenario(
            scenario_id="vo_hook", mode="view_only",
            transforms=[PriceScale(factor=2.0)],
        )
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [scen],
            agent_determinism_check="per_scenario",
            agent_implementation_contract="hook",
            **_KW,
        )
        m = result.manifest
        entry = m["ai_determinism_check_per_scenario"]["vo_hook"]
        assert entry["transformed_status"] == "unsupported"
        assert entry["transformed_passed"] is None
        # Warning text must say "unsupported" and explicitly disclaim
        # determinism-failure semantics.
        warns = result.scenarios[0].warnings
        unsupported = [w for w in warns if "unsupported" in w]
        assert any("not a determinism failure" in w for w in unsupported)

    def test_unsupported_arm_constructs_distinct_instances_per_subject(
        self, monkeypatch,
    ):
        # v2.0.2 #3 hardening: prior code did
        # ``transformed_baseline = transformed_ai`` in both unsupported
        # branches, so the two subjects shared one frozen
        # DeterminismCheckResult. The dataclass's ``metadata`` dict and
        # ``failed_metrics`` list are mutable; aliasing meant a future
        # consumer mutating one would silently mutate the other. The
        # fix builds two independent instances per subject. We verify
        # by counting DeterminismCheckResult constructions during one
        # unsupported scenario.
        from aiphaforge.probes.models import DeterminismCheckResult

        constructed: list[DeterminismCheckResult] = []
        original_init = DeterminismCheckResult.__init__

        def counting_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.status == "unsupported":
                constructed.append(self)

        monkeypatch.setattr(
            DeterminismCheckResult, "__init__", counting_init,
        )
        scen = ABScenario(
            scenario_id="vo_hook", mode="view_only",
            transforms=[PriceScale(factor=2.0)],
        )
        run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [scen],
            agent_determinism_check="per_scenario",
            agent_implementation_contract="hook",
            **_KW,
        )
        # Exactly two unsupported instances (AI + baseline) and they
        # must be distinct objects, not aliases.
        unsupported = [r for r in constructed if r.status == "unsupported"]
        assert len(unsupported) == 2, (
            f"expected 2 unsupported instances, got {len(unsupported)}"
        )
        assert unsupported[0] is not unsupported[1]
        # Their mutable fields must be independent objects too.
        assert unsupported[0].metadata is not unsupported[1].metadata
        assert (
            unsupported[0].failed_metrics
            is not unsupported[1].failed_metrics
        )

    def test_preflight_unsupported_helper(self):
        # Pure helper: returns reason or None.
        assert _preflight_unsupported(
            contract="hook", scenario_mode="view_only",
        ) is not None
        assert _preflight_unsupported(
            contract="hook_view_only_capable", scenario_mode="view_only",
        ) is None
        assert _preflight_unsupported(
            contract="hook", scenario_mode="market_level",
        ) is None

    def test_unsupported_scenario_error_subclass(self):
        # Plan §5.5: must subclass RuntimeError so callers can catch
        # it without catching unrelated framework errors.
        assert issubclass(UnsupportedScenarioError, RuntimeError)


# ---------- Replayability ----------

class TestReplayability:
    def test_seeded_stochastic_builtin_transform_is_replayable_and_supported(self):
        # Plan §5.8: a seeded stochastic transform with a stable
        # fingerprint must be supported, NOT marked unsupported.
        # The OHLCJitter built-in derives its RNG from `seed=`.
        scen = ABScenario(
            scenario_id="jitter", mode="market_level",
            transforms=[OHLCJitter(bps=5.0)],
        )
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [scen],
            agent_determinism_check="per_scenario", **_KW,
        )
        entry = result.manifest["ai_determinism_check_per_scenario"]["jitter"]
        # Either passed or failed, but never unsupported (the gate is
        # replayability, not stochasticity).
        assert entry["transformed_status"] in ("passed", "failed")

    def test_user_transform_without_replayable_view_is_unsupported(self):
        # Synthesize a user transform that ignores `seed=` and uses
        # global randomness instead. The fingerprint test must catch
        # this and mark the arm unsupported. We jitter only `volume`
        # to keep the OHLC integrity invariants intact (the framework
        # would otherwise reject the view before our determinism check
        # ever runs).
        class _NonReplayable:
            name = "NonReplayable"
            category = "level"
            supports_view_only = False
            supports_market_level = True
            order_invertible = False
            stochastic = True

            def apply(self, data, *, seed=None):
                rng = np.random.default_rng()  # deliberately ignores seed
                out = data.copy()
                out["volume"] = out["volume"] * (
                    1.0 + rng.normal(0.0, 1e-3, size=len(data))
                )
                return out

        scen = ABScenario(
            scenario_id="nonrep", mode="market_level",
            transforms=[_NonReplayable()],
        )
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [scen],
            agent_determinism_check="per_scenario", **_KW,
        )
        entry = result.manifest["ai_determinism_check_per_scenario"]["nonrep"]
        assert entry["transformed_status"] == "unsupported"
        assert entry["transformed_passed"] is None


# ---------- Warnings ----------

class TestWarnings:
    def test_warning_predicate_uses_status_not_passed_bool(self, monkeypatch):
        # Plan §5.14: warnings derive from status, not from
        # `passed is False`. Patch _check_agent_determinism to return
        # a deterministic "failed" result and check the warning fires.
        from aiphaforge.probes import abtest as ab_mod

        def fake_check(*a, **kw):
            return DeterminismCheckResult(
                passed=False, status="failed",
                metric_values_run_1={"total_return": 0.05},
                metric_values_run_2={"total_return": 0.10},
                failed_metrics=["total_return"],
                determinism_metrics=("total_return",),
                determinism_rel_tol=1e-3,
            )

        monkeypatch.setattr(ab_mod, "_check_agent_determinism", fake_check)
        with pytest.warns(UserWarning) as record:
            result = run_ab_probe(
                _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
                agent_determinism_check="per_scenario", **_KW,
            )
        # Raw warning emitted at the top level.
        msgs = [str(w.message) for w in record]
        assert any(
            "subject=ai" in m and "scenario=__raw__" in m and "arm=raw" in m
            and "failed_metrics" in m
            for m in msgs
        )
        # And the per-scenario manifest reflects the "failed" status.
        assert result.manifest["ai_determinism_check_per_scenario"]["ps"][
            "transformed_status"
        ] == "failed"

    def test_warning_contains_subject_scenario_arm_and_failed_metrics(self, monkeypatch):
        # Plan §5.14 contract: every warning carries subject=, scenario=,
        # arm=, and either failed_metrics= or reason=.
        from aiphaforge.probes import abtest as ab_mod

        def fake_check(*a, **kw):
            return DeterminismCheckResult(
                passed=False, status="failed",
                metric_values_run_1={}, metric_values_run_2={},
                failed_metrics=["num_trades"],
                determinism_metrics=("num_trades",),
                determinism_rel_tol=1e-3,
            )

        monkeypatch.setattr(ab_mod, "_check_agent_determinism", fake_check)
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="per_scenario", **_KW,
        )
        # Transformed warning lives in the scenario report.
        warns = result.scenarios[0].warnings
        transformed_warns = [w for w in warns if "arm=transformed" in w]
        assert transformed_warns
        for w in transformed_warns:
            assert "subject=" in w
            assert "scenario=ps" in w
            assert "arm=transformed" in w
            assert "failed_metrics=" in w

    def test_raw_warning_emitted_once_not_per_scenario(self, monkeypatch):
        # Plan §5.14: raw warning emits ONCE per subject regardless of
        # how many scenarios we have. Multi-scenario probe with a
        # raw-failing factory should still only see one raw warning
        # per subject (=2 total, ai + baseline).
        from aiphaforge.probes import abtest as ab_mod

        def fake_check(factory, data, *, transforms=None, **kw):
            return DeterminismCheckResult(
                passed=False, status="failed",
                metric_values_run_1={}, metric_values_run_2={},
                failed_metrics=["total_return"],
                determinism_metrics=("total_return",),
                determinism_rel_tol=1e-3,
            )

        monkeypatch.setattr(ab_mod, "_check_agent_determinism", fake_check)
        scenarios = [
            ABScenario(scenario_id=f"s{i}", mode="market_level",
                       transforms=[PriceScale(factor=float(2 + i))])
            for i in range(3)
        ]
        with pytest.warns(UserWarning) as record:
            run_ab_probe(
                _det_factory, _det_factory, _ohlcv(n=40), scenarios,
                agent_determinism_check="per_scenario", **_KW,
            )
        raw_warns = [
            str(w.message) for w in record
            if "scenario=__raw__" in str(w.message)
        ]
        # Exactly two raw warnings: ai + baseline.
        assert len(raw_warns) == 2


# ---------- Top-level mirror sources ----------

class TestTopLevelMirrors:
    def test_top_level_failed_metrics_are_preserved(self):
        # When raw passes, top-level failed_metrics is [].
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="raw_only", **_KW,
        )
        m = result.manifest
        assert m["ai_determinism_failed_metrics"] == []
        assert m["baseline_determinism_failed_metrics"] == []

    def test_top_level_failed_metrics_derive_from_raw_arm_failed_metrics(
        self, monkeypatch,
    ):
        # Plan §5.11 source rule: top-level mirror is the raw arm's
        # `failed_metrics` list.
        from aiphaforge.probes import abtest as ab_mod

        def fake_check(*a, **kw):
            return DeterminismCheckResult(
                passed=False, status="failed",
                metric_values_run_1={}, metric_values_run_2={},
                failed_metrics=["num_trades", "win_rate"],
                determinism_metrics=(
                    "total_return", "num_trades", "win_rate",
                ),
                determinism_rel_tol=1e-3,
            )

        monkeypatch.setattr(ab_mod, "_check_agent_determinism", fake_check)
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="raw_only", **_KW,
        )
        m = result.manifest
        assert m["ai_determinism_failed_metrics"] == [
            "num_trades", "win_rate",
        ]
        assert m["baseline_determinism_failed_metrics"] == [
            "num_trades", "win_rate",
        ]
        # Mirror also reaches via _legacy_pass.
        assert m["ai_determinism_check_passed"] is False
        assert m["ai_determinism_check_status"] == "failed"

    def test_legacy_pass_fields_bool_except_off(self):
        # raw_only: bool. per_scenario: bool. off: None.
        for mode, expected_passed_type in (
            ("raw_only", bool),
            ("per_scenario", bool),
        ):
            r = run_ab_probe(
                _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
                agent_determinism_check=mode, **_KW,
            )
            assert isinstance(
                r.manifest["ai_determinism_check_passed"],
                expected_passed_type,
            )
        r_off = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="off", **_KW,
        )
        assert r_off.manifest["ai_determinism_check_passed"] is None


# ---------- Canonical schema ----------

class TestCanonicalSchema:
    def test_canonical_determinism_schema_present_and_versioned(self):
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="per_scenario", **_KW,
        )
        canonical = result.manifest["determinism_check"]
        assert canonical["schema_version"] == "2.0.1"
        assert canonical["mode"] == "per_scenario"
        assert "requested" in canonical
        assert "resolved" in canonical
        assert "subjects" in canonical
        # Reserved extension slots present (and empty in v2.0.1).
        assert canonical["controls"] == {}
        assert canonical["extension"] == {}

    def test_legacy_mirrors_derived_from_canonical_schema(self):
        result = run_ab_probe(
            _det_factory, _det_factory, _ohlcv(n=40), [_scenario()],
            agent_determinism_check="per_scenario", **_KW,
        )
        m = result.manifest
        # Legacy mirror fields must agree with canonical.
        canonical = m["determinism_check"]
        assert m["agent_determinism_check_mode"] == canonical["mode"]
        assert m["determinism_profile"] == canonical["resolved"]["profile"]
        assert m["determinism_metrics"] == canonical["resolved"]["determinism_metrics"]
        assert m["determinism_rel_tol"] == canonical["resolved"]["determinism_rel_tol"]
        # Per-scenario flat mirror has same scenario ids as canonical.
        assert set(m["ai_determinism_check_per_scenario"].keys()) == set(
            canonical["subjects"]["ai"]["per_scenario"].keys()
        )


# ---------- Serialization (NaN/inf, missing metrics) ----------

class TestArmResultSerialization:
    def test_metric_values_are_nan_safe_in_manifest_json(self):
        # Plan §5.10: nan / inf normalize to string sentinels.
        result = DeterminismCheckResult(
            passed=False, status="failed",
            metric_values_run_1={"a": float("nan"), "b": float("inf")},
            metric_values_run_2={"a": float("-inf"), "b": 1.0},
            failed_metrics=["a", "b"],
            determinism_metrics=("a", "b"),
            determinism_rel_tol=1e-3,
        )
        arm = _arm_result_to_json(result, metrics=("a", "b"), rel_tol=1e-3)
        assert arm["metric_values_run_1"] == {"a": "nan", "b": "inf"}
        assert arm["metric_values_run_2"] == {"a": "-inf", "b": 1.0}

    def test_missing_metric_values_serialize_as_none_not_omitted(self):
        # Plan §5.10: every requested metric appears as a key; missing
        # → None (never omitted).
        result = DeterminismCheckResult(
            passed=True, status="passed",
            metric_values_run_1={"a": 1.0},  # 'b' missing
            metric_values_run_2={"a": 1.0},  # 'b' missing
            failed_metrics=[],
            determinism_metrics=("a", "b"),
            determinism_rel_tol=1e-3,
        )
        arm = _arm_result_to_json(result, metrics=("a", "b"), rel_tol=1e-3)
        # Both keys present; missing → None.
        assert arm["metric_values_run_1"] == {"a": 1.0, "b": None}
        assert arm["metric_values_run_2"] == {"a": 1.0, "b": None}

    def test_legacy_pass_helper(self):
        # Pure trace through the §5.11 truth table.
        passed = DeterminismCheckResult(
            passed=True, status="passed",
            metric_values_run_1={}, metric_values_run_2={},
            failed_metrics=[], determinism_metrics=(),
            determinism_rel_tol=1e-3,
        )
        failed = DeterminismCheckResult(
            passed=False, status="failed",
            metric_values_run_1={}, metric_values_run_2={},
            failed_metrics=[], determinism_metrics=(),
            determinism_rel_tol=1e-3,
        )
        unsupported = DeterminismCheckResult(
            passed=None, status="unsupported",
            metric_values_run_1={}, metric_values_run_2={},
            failed_metrics=[], determinism_metrics=(),
            determinism_rel_tol=1e-3,
        )
        assert _legacy_pass(passed, mode="raw_only") is True
        assert _legacy_pass(failed, mode="raw_only") is False
        assert _legacy_pass(unsupported, mode="raw_only") is False
        assert _legacy_pass(None, mode="raw_only") is False
        # off mode always None.
        assert _legacy_pass(passed, mode="off") is None
        assert _legacy_pass(None, mode="off") is None


# ---------- Replayability fingerprint helper ----------

class TestReplayFingerprint:
    def test_same_data_same_fingerprint(self):
        df = _ohlcv(n=20, seed=0)
        assert _stable_view_fingerprint(df) == _stable_view_fingerprint(
            df.copy()
        )

    def test_different_data_different_fingerprint(self):
        a = _ohlcv(n=20, seed=0)
        b = _ohlcv(n=20, seed=1)
        assert _stable_view_fingerprint(a) != _stable_view_fingerprint(b)

    # ---- v2.0.2 #5: string-column hardening ----

    def test_fingerprint_handles_string_columns(self):
        # Pre-fix this raised ValueError; the caller swallowed it as
        # `replayable=True`, masking genuine non-determinism in user
        # transforms that emit metadata columns. Now it returns a
        # stable hash and reflects string-column content changes.
        import pandas as pd
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "close": [101.0, 102.0],
            "tag": ["a", "b"],  # non-numeric
        }, index=pd.bdate_range("2024-01-01", periods=2))
        fp1 = _stable_view_fingerprint(df)
        fp2 = _stable_view_fingerprint(df.copy())
        assert fp1 == fp2

    def test_fingerprint_string_column_content_changes_hash(self):
        import pandas as pd
        idx = pd.bdate_range("2024-01-01", periods=2)
        a = pd.DataFrame(
            {"close": [100.0, 101.0], "tag": ["x", "y"]}, index=idx,
        )
        b = pd.DataFrame(
            {"close": [100.0, 101.0], "tag": ["x", "Z"]}, index=idx,
        )
        assert _stable_view_fingerprint(a) != _stable_view_fingerprint(b)

    def test_fingerprint_handles_pandas_string_extension_dtype(self):
        # v2.1.2 regression: pandas 2.2+ may auto-infer
        # ``StringDtype`` (an ExtensionDtype) for object columns
        # when ``pd.options.future.infer_string`` is set. The
        # numpy ``np.issubdtype(..., np.number)`` check raises
        # ``TypeError`` on extension dtypes; the pandas-aware
        # ``pd.api.types.is_numeric_dtype`` handles both.
        # Force the StringDtype regardless of pandas defaults so
        # this test reproduces the CI failure on every machine.
        import pandas as pd
        idx = pd.bdate_range("2024-01-01", periods=2)
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "tag": pd.array(["a", "b"], dtype="string"),
            },
            index=idx,
        )
        assert isinstance(df["tag"].dtype, pd.StringDtype), (
            "test setup: tag column should be StringDtype"
        )
        # Pre-fix: this raised TypeError. Now it returns a stable
        # hash and reflects content changes.
        fp1 = _stable_view_fingerprint(df)
        fp2 = _stable_view_fingerprint(df.copy())
        assert fp1 == fp2

        df_changed = df.copy()
        df_changed["tag"] = pd.array(["a", "Z"], dtype="string")
        assert _stable_view_fingerprint(df_changed) != fp1

    def test_fingerprint_handles_pandas_nullable_int_extension_dtype(self):
        # v2.1.2: the same ``np.issubdtype(..., np.number)`` call
        # also crashed on pandas nullable integer/float extension
        # dtypes (``Int64``, ``Float64``). The pandas-aware
        # ``pd.api.types.is_numeric_dtype`` correctly classifies
        # them as numeric. Lock that path with a content-changes
        # assertion so the numeric branch genuinely runs.
        import pandas as pd
        idx = pd.bdate_range("2024-01-01", periods=3)
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "qty": pd.array([1, 2, 3], dtype="Int64"),
            },
            index=idx,
        )
        assert isinstance(df["qty"].dtype, pd.Int64Dtype), (
            "test setup: qty column should be Int64 extension dtype"
        )
        fp1 = _stable_view_fingerprint(df)
        fp2 = _stable_view_fingerprint(df.copy())
        assert fp1 == fp2

        df_changed = df.copy()
        df_changed["qty"] = pd.array([1, 2, 99], dtype="Int64")
        assert _stable_view_fingerprint(df_changed) != fp1
