"""Tests for decompile.physics_review (B1c-2).

OFFLINE, deterministic, no API calls. Inputs:
- tests/fixtures/live_muse_spark_b1b3_response.json: the REAL Muse Spark structured response
  from the single B1b-3 call (copied verbatim from the API evidence; see its _provenance).
- tests/fixtures/handwritten_interpretation_bounce_single.json: HANDWRITTEN test data.
- SYNTHETIC trajectories (the existing fixture, or generated here from closed-form motion).
Interpretation variants below are hand-made edits for testing, never model output.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import kinematic_fit as kf
from decompile import physics_review as pr
from decompile import scene_interpretation as si
from decompile import trajectories as tr
from decompile.meta_api import strict_json_loads

FIXTURES = Path(__file__).parent / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"
LIVE = FIXTURES / "live_muse_spark_b1b3_response.json"
HANDWRITTEN = FIXTURES / "handwritten_interpretation_bounce_single.json"
LIVE_CONTENT_SHA256 = "c9f16275215e186837832bc1cb6debe8e53b8a34528c989bcc3770a08844d607"


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def traj() -> tr.TrajectorySet:
    return tr.load_trajectories(BOUNCE)


@pytest.fixture(scope="module")
def kin(traj) -> kf.KinematicsReport:
    return kf.analyze_kinematics(traj)


@pytest.fixture(scope="module")
def live_record() -> dict:
    return json.loads(LIVE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_data(live_record) -> dict:
    return strict_json_loads(live_record["content"])


@pytest.fixture(scope="module")
def live(traj, live_data) -> si.SceneInterpretation:
    return si.parse_interpretation(live_data, traj)


@pytest.fixture(scope="module")
def handwritten_data() -> dict:
    return json.loads(HANDWRITTEN.read_text(encoding="utf-8"))["interpretation"]


@pytest.fixture(scope="module")
def handwritten(traj, handwritten_data) -> si.SceneInterpretation:
    return si.parse_interpretation(handwritten_data, traj)


@pytest.fixture(scope="module")
def live_review(live, kin) -> pr.PhysicsReview:
    return pr.review_interpretation(live, kin)


def one(review: pr.PhysicsReview, code: str, subject: str) -> pr.Finding:
    found = [f for f in review.findings if f.code == code and f.subject_id == subject]
    assert len(found) == 1, (code, subject, [(f.code, f.subject_id) for f in review.findings])
    return found[0]


def codes(review: pr.PhysicsReview, severity: str | None = None) -> list[tuple[str, str | None]]:
    return [(f.code, f.subject_id) for f in review.findings if severity is None or f.severity == severity]


def make_set(objects: dict[str, tuple], n: int = 31) -> tr.TrajectorySet:
    times = [Fraction(i, 30) for i in range(n)]
    t = np.array([float(x) for x in times])
    objs = []
    for oid, (fx, fy) in objects.items():
        xs, ys = fx(t), fy(t)
        objs.append({"object_id": oid, "observations": [
            {"frame_index": i, "t": {"num": tt.numerator, "den": tt.denominator}, "x": float(x), "y": float(y)}
            for i, (tt, x, y) in enumerate(zip(times, xs, ys))]})
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": objs})


def minimal(object_ids, motion_class="constant_velocity_like", last=30) -> dict:
    """HANDWRITTEN minimal interpretation (test data)."""
    return {
        "schema_version": si.SCHEMA_VERSION,
        "observed_facts": [
            {"fact_id": f"F{i + 1}", "object_ids": [oid], "statement": f"{oid} is observed in frames 0 to {last}.",
             "evidence": [{"object_id": oid, "frame_index": 0}, {"object_id": oid, "frame_index": last}]}
            for i, oid in enumerate(object_ids)],
        "objects": [{"object_id": oid, "motion_class": motion_class, "motion_description": "Moves steadily.",
                     "supporting_fact_ids": [f"F{i + 1}"]} for i, oid in enumerate(object_ids)],
        "physical_model_hypotheses": [], "event_hypotheses": [], "interaction_hypotheses": [],
        "ambiguities": [{"ambiguity_id": "A1", "about_ids": ["F1"], "description": "Short clip.",
                         "alternatives": ["Any smooth motion."], "what_would_resolve": "More frames."}],
        "data_quality_notes": [],
    }


def with_fact(interp: si.SceneInterpretation, statement: str, frames: tuple[int, int],
              fact_id: str = "F9") -> si.SceneInterpretation:
    fact = si.ObservedFact(fact_id, ("obj_01",), statement,
                           tuple(si.ObsRef("obj_01", f) for f in range(frames[0], frames[1] + 1)))
    return replace(interp, observed_facts=interp.observed_facts + (fact,))


# ------------------------------------------------------------------ the live fixture itself


def test_live_fixture_is_the_real_labelled_response(live_record, live_data, handwritten_data):
    assert "LIVE MUSE SPARK OUTPUT (B1b-3)" in live_record["_provenance"]
    assert "NOT the handwritten interpretation" in live_record["_provenance"]
    assert hashlib.sha256(live_record["content"].encode("utf-8")).hexdigest() == LIVE_CONTENT_SHA256
    assert live_record["response_content_sha256"] == LIVE_CONTENT_SHA256
    assert live_record["model"] == "muse-spark-1.3"
    assert live_data != handwritten_data
    assert set(live_record) == {"_provenance", "model", "response_id", "prompt_version", "input_sha256",
                                "schema_sha256", "response_content_sha256", "content"}
    text = LIVE.read_text(encoding="utf-8").lower()
    for secret in ("bearer", "authorization", "api_key", "model_api_key", "sk-"):
        assert secret not in text


def test_live_fixture_still_validates(live_data, traj):
    report = si.validate_interpretation(live_data, traj)
    assert report.ok and report.warnings == ()


# ------------------------------------------------------------------ B1b-3 regression (9 findings)


def test_b1b3_uniform_acceleration_supported_not_gravity(live_review):
    f = one(live_review, "HYPOTHESIS_CHECK", "H1")
    assert f.verdict == kf.SUPPORTED and f.severity == pr.INFO
    assert "kinematically consistent" in f.message
    assert "does not establish that the cause is gravity" in f.message
    assert f.evidence["magnitude_px_s2"] == pytest.approx(2000, abs=0.05)
    assert 89.9 < f.evidence["direction_deg_image"] < 90.1


def test_b1b3_impulse_supported(live_review):
    f = one(live_review, "HYPOTHESIS_CHECK", "H2")
    assert f.verdict == kf.SUPPORTED and f.severity == pr.INFO
    assert "contact candidate" in f.message and "does not identify the surface" in f.message
    assert f.evidence["gap_frames"] == (18, 19)
    assert one(live_review, "MOTION_CLASS_CHECK", "obj_01").verdict == kf.SUPPORTED


def test_b1b3_linear_drag_not_supported(live_review):
    f = one(live_review, "HYPOTHESIS_CHECK", "H3")
    assert f.verdict == kf.NOT_SUPPORTED and f.severity == pr.WARNING
    assert f.evidence["support_label"] == "weak"
    assert f.evidence["k_upper_times_span"] < kf.DRAG_MAX_UNDETECTED_KT
    assert "does not change this" in one(live_review, "SUPPORT_LABEL_NOTE", "H3").message


def test_b1b3_smooth_reversal_contradicted(live_review):
    f = one(live_review, "SMOOTH_EXPLANATION_CONTRADICTED", "obj_01")
    assert f.verdict == kf.CONTRADICTED
    assert f.evidence["reversal"] == "down_to_up"
    assert f.evidence["ratio_to_uniform_accel"] > 30
    assert "does not by itself identify a surface" in f.message
    alt = one(live_review, "SMOOTH_ALTERNATIVE_CONTRADICTED", "H2")
    assert alt.evidence["alternative"] == "uniform_gravity"  # exact enum token in H2's alternatives


def test_b1b3_e1_window_consistent(live_review):
    f = one(live_review, "EVENT_CONTACT_WINDOW", "E1")
    assert f.verdict == pr.CONSISTENT and f.severity == pr.INFO
    lo, hi = f.evidence["contact_interval_s"]
    assert 0.6 < lo <= hi < 2 / 3
    assert f.evidence["window_s"] == ("3/5", "2/3")


def test_b1b3_e2_brackets_apex_not_collision(live_review):
    f = one(live_review, "EVENT_BRACKETS_TURNING_POINT", "E2")
    assert f.evidence["turning_point"] == "y_minimum"
    assert f.evidence["t_turning_s"] == pytest.approx(1.04795, abs=1e-4)
    assert "not a physical event or collision" in f.message
    assert "highest point in the image" in f.message
    assert not live_review.by_code("EVENT_CONTACT_WINDOW") or all(
        x.subject_id != "E2" for x in live_review.by_code("EVENT_CONTACT_WINDOW"))
    assert "bounce" not in f.message.lower().replace("not a bounce", "")


def test_b1b3_f3_upward_wording_flagged(live_review):
    f = one(live_review, "DIRECTION_WORDING_CONFLICT", "F3")
    assert f.severity == pr.WARNING and f.verdict == pr.FLAGGED
    assert f.evidence["phrase"] == "upward"
    assert f.evidence["frames"] == (0, 18)
    assert "+y is down" in f.message


def test_b1b3_f5_low_wording_flagged(live_review):
    f = one(live_review, "EXTREMUM_WORDING_CONFLICT", "F5")
    assert f.severity == pr.WARNING
    assert f.evidence["phrase"] == "the low at frame 31"
    assert f.evidence["image_extremum"] == "image_high"
    assert "HIGHEST point in the image" in f.message


def test_b1b3_floor_identity_not_confirmed(live_review):
    f = one(live_review, "SURFACE_IDENTITY_NOT_EVALUATED", "H2")
    assert f.verdict == kf.NOT_EVALUATED and f.evidence["plane_hint"] == "floor"
    compat = one(live_review, "SURFACE_DIRECTION_COMPATIBLE", "H2")
    assert "not surface identity" in compat.message
    for finding in live_review.findings:
        assert "floor is confirmed" not in finding.message.lower()


def test_b1b3_only_f3_and_f5_wording_flagged(live_review):
    flagged = {f.subject_id for f in live_review.findings if f.verdict == pr.FLAGGED}
    assert flagged == {"F3", "F5"}  # F4's "from falling to rising" and E1/E2 rationales are correct


def test_b1b3_ambiguities(live_review):
    a1 = one(live_review, "AMBIGUITY_ON_DECIDED_CLAIM", "A1")
    assert a1.severity == pr.WARNING and "H3" in a1.evidence["decided"]
    a2 = one(live_review, "AMBIGUITY_ON_DECIDED_CLAIM", "A2")
    assert a2.severity == pr.INFO and set(a2.evidence["decided"]) == {"H2", "E1"}
    assert "without a velocity discontinuity is contradicted" in a2.message
    assert "free reversal without surface contact" in a2.message  # quoted, not interpreted


def test_b1b3_summary(live_review):
    assert live_review.counts == {
        "error": 0, "warning": len(codes(live_review, pr.WARNING)), "info": len(codes(live_review, pr.INFO))}
    assert sorted(codes(live_review, pr.WARNING)) == sorted([
        ("HYPOTHESIS_CHECK", "H3"), ("AMBIGUITY_ON_DECIDED_CLAIM", "A1"),
        ("DIRECTION_WORDING_CONFLICT", "F3"), ("EXTREMUM_WORDING_CONFLICT", "F5")])
    assert live_review.assessment == "concerns_found"
    assert "does not establish that the interpretation is physically true" in live_review.assessment_note
    assert live_review.method_version == pr.METHOD_VERSION
    assert live_review.kinematics_method_version == kf.METHOD_VERSION


# ------------------------------------------------------------------ handwritten fixture


def test_handwritten_interpretation_is_clean(handwritten, kin):
    r = pr.review_interpretation(handwritten, kin)
    assert r.counts["error"] == 0 and r.counts["warning"] == 0, codes(r)
    assert r.assessment == "no_contradictions_found"
    assert "does not establish" in r.assessment_note
    assert one(r, "EVENT_CONTACT_WINDOW", "E1").verdict == pr.CONSISTENT
    assert not r.by_code("DIRECTION_WORDING_CONFLICT")  # incl. F2 "rising to 38.89 px": a number, not motion
    a3 = one(r, "AMBIGUITY_ON_DECIDED_CLAIM", "A3")
    assert "velocity discontinuity" not in a3.message  # A3 is about H1 only, not the impulse


# ------------------------------------------------------------------ hypothesis mappings and labels


@pytest.mark.parametrize("model", ["other", "object_collision", "quadratic_drag", "magnetism"])
def test_unmapped_or_unevaluated_models(live, kin, model):
    h = replace(live.physical_model_hypotheses[2], model=model, alternatives_considered=())
    interp = replace(live, physical_model_hypotheses=(h,))
    r = pr.review_interpretation(interp, kin)
    f = one(r, "HYPOTHESIS_NOT_EVALUATED", "H3")
    assert f.verdict == kf.NOT_EVALUATED and f.severity == pr.INFO
    assert f.evidence["known_model"] == (model in si.PHYSICAL_MODELS)
    assert not [x for x in r.findings if x.code == "HYPOTHESIS_CHECK"]


def test_hypothesis_about_unknown_object(live, kin):
    h = replace(live.physical_model_hypotheses[0], applies_to=("obj_77",))
    r = pr.review_interpretation(replace(live, physical_model_hypotheses=(h,)), kin)
    assert one(r, "OBJECT_NOT_IN_KINEMATICS", "H1").object_id == "obj_77"


@pytest.mark.parametrize("label", ["weak", "moderate", "strong"])
def test_label_never_changes_drag_verdict(live, kin, label):
    h = replace(live.physical_model_hypotheses[2], support=label)
    r = pr.review_interpretation(replace(live, physical_model_hypotheses=(h,)), kin)
    f = one(r, "HYPOTHESIS_CHECK", "H3")
    assert f.verdict == kf.NOT_SUPPORTED and f.severity == pr.WARNING
    note = one(r, "SUPPORT_LABEL_NOTE", "H3").message
    assert ("overstates" in note) if label == "strong" else ("does not change this" in note)


@pytest.mark.parametrize("label", ["weak", "strong"])
def test_contradicted_model_is_an_error_whatever_the_label(live, kin, label):
    h = replace(live.physical_model_hypotheses[0], model="constant_velocity", support=label)
    r = pr.review_interpretation(replace(live, physical_model_hypotheses=(h,)), kin)
    f = one(r, "HYPOTHESIS_CHECK", "H1")
    assert f.verdict == kf.CONTRADICTED and f.severity == pr.ERROR
    assert r.assessment == "contradictions_found"


def test_weak_label_on_supported_claim_is_noted(live, kin):
    h = replace(live.physical_model_hypotheses[1], support="weak")
    r = pr.review_interpretation(replace(live, physical_model_hypotheses=(h,)), kin)
    assert "understates" in one(r, "SUPPORT_LABEL_NOTE", "H2").message
    assert one(r, "HYPOTHESIS_CHECK", "H2").severity == pr.INFO


@pytest.mark.parametrize("hint, compatible", [("floor", True), ("ceiling", False), ("left_wall", False)])
def test_plane_hint_direction_compatibility_without_identity(live, kin, hint, compatible):
    h = replace(live.physical_model_hypotheses[1], plane_hint=hint)
    r = pr.review_interpretation(replace(live, physical_model_hypotheses=(h,)), kin)
    one(r, "SURFACE_IDENTITY_NOT_EVALUATED", "H2")
    if compatible:
        assert one(r, "SURFACE_DIRECTION_COMPATIBLE", "H2").severity == pr.INFO
    else:
        assert one(r, "SURFACE_DIRECTION_INCOMPATIBLE", "H2").severity == pr.WARNING
    assert one(r, "HYPOTHESIS_CHECK", "H2").verdict == kf.SUPPORTED  # the impulse itself is unaffected


# ------------------------------------------------------------------ events


def reparsed(data: dict, traj, edit) -> si.SceneInterpretation:
    d = copy.deepcopy(data)
    edit(d)
    return si.parse_interpretation(d, traj)


def test_event_window_excluding_contact_is_an_error(live_data, traj, kin):
    def edit(d):
        d["event_hypotheses"][0].update(after_frame_index=24, before_frame_index=26)
    r = pr.review_interpretation(reparsed(live_data, traj, edit), kin)
    f = one(r, "EVENT_CONTACT_WINDOW", "E1")
    assert f.verdict == kf.CONTRADICTED and f.severity == pr.ERROR
    assert one(r, "IMPULSE_WITHOUT_EVENT", "obj_01").severity == pr.WARNING
    assert r.assessment == "contradictions_found"


def test_apex_labelled_as_bounce(live_data, traj, kin):
    def edit(d):
        d["event_hypotheses"] = [dict(d["event_hypotheses"][1], type="possible_bounce")]
        d["ambiguities"] = [a for a in d["ambiguities"] if "E1" not in a["about_ids"]]
    r = pr.review_interpretation(reparsed(live_data, traj, edit), kin)
    assert one(r, "EVENT_CONTACT_WINDOW", "E2").verdict == kf.CONTRADICTED
    apex = one(r, "EVENT_BRACKETS_TURNING_POINT", "E2")
    assert "An apex is not a bounce or collision." in apex.message
    one(r, "IMPULSE_WITHOUT_EVENT", "obj_01")


def test_missing_bounce_event_is_reported(live, kin):
    interp = replace(live, event_hypotheses=live.event_hypotheses[1:], event_windows=live.event_windows[1:],
                     ambiguities=())
    r = pr.review_interpretation(interp, kin)
    f = one(r, "IMPULSE_WITHOUT_EVENT", "obj_01")
    assert f.severity == pr.WARNING and f.evidence["gap_frames"] == (18, 19)


def test_partially_overlapping_window_is_inconclusive(live, kin):
    o = kin.objects[0]
    wide = replace(o, contact=replace(o.contact, profile_interval_s=(0.59, 0.62)))
    r = pr.review_interpretation(live, replace(kin, objects=(wide,)))
    f = one(r, "EVENT_CONTACT_WINDOW", "E1")
    assert f.verdict == kf.INCONCLUSIVE and f.severity == pr.INFO
    assert one(r, "IMPULSE_WITHOUT_EVENT", "obj_01")


def test_other_event_without_counterpart_is_not_evaluated(live_data, traj, kin):
    def edit(d):
        d["event_hypotheses"][1].update(type="possible_occlusion", after_frame_index=5, before_frame_index=7)
    r = pr.review_interpretation(reparsed(live_data, traj, edit), kin)
    assert one(r, "EVENT_NOT_EVALUATED", "E2").verdict == kf.NOT_EVALUATED


# ------------------------------------------------------------------ ambiguities


def test_ambiguity_references(live, kin):
    amb = (
        si.Ambiguity("A1", ("H3",), "Drag or not.", ("with drag",), "longer clip"),       # valid, rejected claim
        si.Ambiguity("A2", ("H9", "F1"), "Unknown ref.", ("x",), "y"),                     # missing reference
        si.Ambiguity("A3", ("F1", "F2"), "About facts only.", ("x",), "y"),                # unrelated to decisions
    )
    r = pr.review_interpretation(replace(live, ambiguities=amb), kin)
    assert one(r, "AMBIGUITY_ON_DECIDED_CLAIM", "A1").severity == pr.WARNING
    assert one(r, "AMBIGUITY_REFERENCE_UNKNOWN", "A2").evidence["reference"] == "H9"
    assert not [f for f in r.findings if f.subject_id == "A3"]
    assert not [f for f in r.findings if f.subject_id == "A2" and f.code == "AMBIGUITY_ON_DECIDED_CLAIM"]


# ------------------------------------------------------------------ direction wording


@pytest.mark.parametrize("statement, frames, flagged", [
    ("The object moves downward (y increases) from frame 0 to frame 18.", (0, 18), False),
    ("The object moves upward from frame 0 to frame 18.", (0, 18), True),
    ("The object is falling from frame 0 to frame 18.", (0, 18), False),
    ("The object moves upward from frame 19 to frame 31.", (19, 31), False),
    ("The object falls from frame 19 to frame 31.", (19, 31), True),
    ("The object is rising between frame 19 and frame 31.", (19, 31), False),
    ("y increases from frame 0 to frame 18.", (0, 18), False),
    ("y decreases from frame 19 to frame 31.", (19, 31), False),
    ("y rises from 70.0 to 430.0 between frames 0 and 18.", (0, 18), False),
    ("The step in y keeps rising to 38.89 px by frame 18.", (0, 18), False),
    ("y first falls and then rises.", (19, 31), False),
    ("It reverses with a low and a high turning behaviour.", (0, 36), False),
])
def test_direction_wording(live, kin, statement, frames, flagged):
    r = pr.review_interpretation(with_fact(live, statement, frames), kin)
    hits = [f for f in r.findings if f.subject_id == "F9" and f.code == "DIRECTION_WORDING_CONFLICT"]
    assert bool(hits) == flagged, [f.message for f in hits]
    for f in hits:
        assert f.severity == pr.WARNING and f.evidence["phrase"] in statement


@pytest.mark.parametrize("statement, flagged", [
    ("The apex at frame 31 is the top of the rebound.", False),
    ("The highest point at frame 31 follows the rise.", False),
    ("Frame 31 is the low point of the arc.", True),
    ("The apex at frame 19 ends the fall.", True),
    ("Frame 19 is the lowest point in the image.", False),
    ("The low at frame 10 is not an extremum.", False),  # frame 10 is no extremum: not judged
])
def test_extremum_wording(live, kin, statement, flagged):
    r = pr.review_interpretation(with_fact(live, statement, (10, 32)), kin)
    hits = [f for f in r.findings if f.subject_id == "F9" and f.code == "EXTREMUM_WORDING_CONFLICT"]
    assert bool(hits) == flagged, [f.message for f in hits]
    for f in hits:
        assert "+y is down" in f.message and f.severity == pr.WARNING


def test_wording_never_produces_errors(live, kin):
    r = pr.review_interpretation(with_fact(live, "It moves upward and the apex at frame 19.", (0, 18)), kin)
    assert all(f.severity == pr.WARNING for f in r.findings if f.verdict == pr.FLAGGED)
    assert r.counts["error"] == 0


# ------------------------------------------------------------------ inconclusive kinematics


def test_inconclusive_kinematics_never_become_errors(live, traj):
    kin_bad = kf.analyze_kinematics(traj, position_sigma_px=1e-5)  # sigma far below the 0.003 px rounding
    assert kin_bad.objects[0].adequacy == "inadequate"
    r = pr.review_interpretation(live, kin_bad)
    assert one(r, "KINEMATICS_INADEQUATE", "obj_01").verdict == kf.INCONCLUSIVE
    assert r.counts["error"] == 0
    for f in r.findings:
        if f.code in ("HYPOTHESIS_CHECK", "EVENT_CONTACT_WINDOW", "MOTION_CLASS_CHECK"):
            assert f.verdict == kf.INCONCLUSIVE and f.severity == pr.INFO, f
    assert not r.by_code("IMPULSE_WITHOUT_EVENT") and not r.by_code("SMOOTH_EXPLANATION_CONTRADICTED")


# ------------------------------------------------------------------ objects and inputs


def test_object_id_mismatch(live, traj):
    data = traj.to_json_dict()
    data["objects"][0]["object_id"] = "obj_02"
    kin_other = kf.analyze_kinematics(tr.TrajectorySet.from_json_dict(data))
    r = pr.review_interpretation(live, kin_other)
    assert one(r, "OBJECT_NOT_INTERPRETED", "obj_02").severity == pr.WARNING
    assert {f.subject_id for f in r.by_code("OBJECT_NOT_IN_KINEMATICS")} >= {"obj_01", "H1", "H2", "H3", "E1", "E2"}
    assert not r.by_code("HYPOTHESIS_CHECK")


def test_different_trajectory_is_an_input_mismatch(live, traj):
    data = traj.to_json_dict()
    for ob in data["objects"][0]["observations"]:  # same ids and frames, shifted clock
        t = Fraction(ob["t"]["num"], ob["t"]["den"]) + 1
        ob["t"] = {"num": t.numerator, "den": t.denominator}
    r = pr.review_interpretation(live, kf.analyze_kinematics(tr.TrajectorySet.from_json_dict(data)))
    assert one(r, "INPUT_MISMATCH", "E1").severity == pr.ERROR


def test_multiple_objects():
    s = make_set({"obj_01": (lambda t: 100 + 40 * t, lambda t: 300 - 25 * t),
                  "obj_02": (lambda t: 500 + 0 * t, lambda t: 50 + 450 * t * t)})
    d = minimal(["obj_01", "obj_02"])
    d["objects"][1]["motion_class"] = "ballistic_like"
    d["physical_model_hypotheses"] = [
        {"hypothesis_id": "H1", "status": "candidate", "model": "constant_velocity", "plane_hint": "not_applicable",
         "applies_to": ["obj_01"], "support": "moderate", "supporting_fact_ids": ["F1"], "contradicting_fact_ids": [],
         "rationale": "Steady motion.", "alternatives_considered": ["uniform_gravity"]},
        {"hypothesis_id": "H2", "status": "candidate", "model": "uniform_gravity", "plane_hint": "not_applicable",
         "applies_to": ["obj_02"], "support": "moderate", "supporting_fact_ids": ["F2"], "contradicting_fact_ids": [],
         "rationale": "Curving path.", "alternatives_considered": ["constant_velocity"]},
    ]
    interp = si.parse_interpretation(d, s)
    r = pr.review_interpretation(interp, kf.analyze_kinematics(s))
    assert one(r, "HYPOTHESIS_CHECK", "H1").object_id == "obj_01"
    assert one(r, "HYPOTHESIS_CHECK", "H1").verdict == kf.SUPPORTED
    assert one(r, "HYPOTHESIS_CHECK", "H2").verdict == kf.SUPPORTED
    assert r.counts["error"] == 0 and r.counts["warning"] == 0, codes(r)
    assert not r.by_code("SMOOTH_ALTERNATIVE_CONTRADICTED")  # no impulse anywhere


def test_clean_interpretation_has_no_errors_or_warnings():
    s = make_set({"obj_01": (lambda t: 100 + 40 * t, lambda t: 300 - 25 * t)})
    r = pr.review_interpretation(si.parse_interpretation(minimal(["obj_01"]), s), kf.analyze_kinematics(s))
    assert r.counts == {"error": 0, "warning": 0, "info": len(r.findings)}
    assert r.assessment == "no_contradictions_found"
    assert "does not establish" in r.assessment_note
    assert codes(r) == [("MOTION_CLASS_CHECK", "obj_01")]


def test_motion_class_mismatch():
    s = make_set({"obj_01": (lambda t: 100 + 40 * t, lambda t: 50 + 450 * t * t)})
    r = pr.review_interpretation(si.parse_interpretation(minimal(["obj_01"]), s), kf.analyze_kinematics(s))
    f = one(r, "MOTION_CLASS_CHECK", "obj_01")
    assert f.verdict == kf.CONTRADICTED and f.severity == pr.ERROR


def test_deterministic_output(live, kin):
    assert pr.review_interpretation(live, kin) == pr.review_interpretation(live, kin)


def test_inputs_are_not_mutated(live, kin):
    interp_copy, kin_copy = copy.deepcopy(live), copy.deepcopy(kin)
    evidence_copy = copy.deepcopy([v.evidence for o in kin.objects for v in o.verdicts.values()])
    pr.review_interpretation(live, kin)
    assert live == interp_copy
    assert kin == kin_copy
    assert [v.evidence for o in kin.objects for v in o.verdicts.values()] == evidence_copy


def test_invalid_inputs(live, kin):
    with pytest.raises(pr.PhysicsReviewError):
        pr.review_interpretation({"observed_facts": []}, kin)
    with pytest.raises(pr.PhysicsReviewError):
        pr.review_interpretation(live, {"objects": []})


def test_module_reads_only_and_never_refits():
    tree = ast.parse(Path(pr.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"__future__", "re", "dataclasses", "fractions", "typing", "decompile",
                        "decompile.scene_interpretation"}, imported
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "analyze_kinematics" not in calls and "chat" not in calls
