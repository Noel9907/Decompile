"""Tests for decompile.scene_interpretation (B1b-1).

OFFLINE: pure functions, no API calls. The interpretation used here is HANDWRITTEN test
data (tests/fixtures/handwritten_interpretation_bounce_single.json), not model output.
"""

from __future__ import annotations

import copy
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import pytest

from decompile import scene_interpretation as si
from decompile import trajectories as tr
from decompile.meta_api import check_strict_schema, validate_instance

FIXTURES = Path(__file__).parent / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"
HANDWRITTEN = FIXTURES / "handwritten_interpretation_bounce_single.json"

# Keywords already exercised by a live B1a structured call (tools/meta_smoke.py).
LIVE_TESTED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items",
                        "enum", "minItems", "maxItems"}


@pytest.fixture(scope="module")
def traj() -> tr.TrajectorySet:
    return tr.load_trajectories(BOUNCE)


@pytest.fixture
def golden() -> dict:
    return copy.deepcopy(json.loads(HANDWRITTEN.read_text(encoding="utf-8"))["interpretation"])


def two_object_set() -> tr.TrajectorySet:
    def ob(k, x, y):
        t = Fraction(k, 30)
        return {"frame_index": k, "t": {"num": t.numerator, "den": t.denominator}, "x": x, "y": y}

    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [
            {"object_id": "obj_01", "observations": [ob(k, 100 + 10 * k, 50) for k in range(4)]},
            {"object_id": "obj_02", "observations": [ob(k, 200, 50) for k in range(4)]},
        ],
    })


def minimal_interpretation(object_ids=("obj_01", "obj_02")) -> dict:
    return {
        "schema_version": si.SCHEMA_VERSION,
        "observed_facts": [
            {"fact_id": f"F{i + 1}", "object_ids": [oid], "statement": f"{oid} is observed in frames 0 to 3.",
             "evidence": [{"object_id": oid, "frame_index": 0}, {"object_id": oid, "frame_index": 3}]}
            for i, oid in enumerate(object_ids)
        ],
        "objects": [
            {"object_id": oid, "motion_class": "insufficient_data", "motion_description": "Few observations.",
             "supporting_fact_ids": [f"F{i + 1}"]}
            for i, oid in enumerate(object_ids)
        ],
        "physical_model_hypotheses": [],
        "event_hypotheses": [],
        "interaction_hypotheses": [],
        "ambiguities": [
            {"ambiguity_id": "A1", "about_ids": ["F1"], "description": "Too few frames.",
             "alternatives": ["Any smooth motion."], "what_would_resolve": "More frames."}
        ],
        "data_quality_notes": [],
    }


def report(data, trajectories) -> si.ValidationReport:
    return si.validate_interpretation(data, trajectories)


def assert_error(data, trajectories, fragment: str) -> None:
    rep = report(data, trajectories)
    assert any(fragment in e for e in rep.errors), rep.errors
    assert rep.event_windows == ()  # never hand out windows from an invalid interpretation


def ground_truth_impact_time() -> float:
    # Ground truth of the SYNTHETIC fixture (see tests/test_trajectories.py); never in the input.
    return math.sqrt(2 * (450.0 - 70.0) / 2000.0)


# ------------------------------------------------------------------ fixtures and provenance


def test_handwritten_fixture_is_labelled_and_separate_from_input():
    wrapper = json.loads(HANDWRITTEN.read_text(encoding="utf-8"))
    assert set(wrapper) == {"_provenance", "interpretation"}
    assert "HANDWRITTEN TEST DATA" in wrapper["_provenance"]
    assert "NOT Muse Spark output" in wrapper["_provenance"]
    assert "NOT a Meta API response" in wrapper["_provenance"]
    trajectory_text = BOUNCE.read_text(encoding="utf-8")
    assert "handwritten" not in trajectory_text.lower()
    assert "fact_id" not in trajectory_text and "hypothesis" not in trajectory_text


# ------------------------------------------------------------------ schema


def walk(node: Any, path: str = "$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def test_schema_is_strict_and_uses_only_live_tested_keywords(traj):
    schema = si.build_interpretation_schema(traj)
    check_strict_schema(schema)
    for path, node in walk(schema):
        if "type" in node and isinstance(node["type"], str):
            assert set(node) <= LIVE_TESTED_KEYWORDS, (path, set(node) - LIVE_TESTED_KEYWORDS)
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, path
            assert node["required"] == list(node["properties"]), path


def test_schema_object_id_enum_comes_from_input(traj):
    for ts in (traj, two_object_set()):
        schema = si.build_interpretation_schema(ts)
        enums = [n["enum"] for p, n in walk(schema) if p.endswith(".object_id") and "enum" in n]
        assert enums and all(e == list(ts.object_ids) for e in enums)


def test_schema_has_no_numeric_fields_except_frame_indices(traj):
    schema = si.build_interpretation_schema(traj)
    numeric = sorted(p for p, n in walk(schema) if n.get("type") in ("number", "integer"))
    assert all(p.rsplit(".", 1)[-1] in {"frame_index", "after_frame_index", "before_frame_index"}
               for p in numeric), numeric
    assert not any(n.get("type") == "number" for _, n in walk(schema))
    props = {k for _, n in walk(schema) if "properties" in n for k in n["properties"]}
    for forbidden in ("t", "time", "timestamp", "gravity", "restitution", "drag", "velocity", "mass"):
        assert forbidden not in props


def test_no_object_identity_guess_anywhere(traj, golden):
    # Trajectories carry no visual information, so no category may be guessed.
    schema = si.build_interpretation_schema(traj)
    obj_props = schema["properties"]["objects"]["items"]["properties"]
    assert list(obj_props) == ["object_id", "motion_class", "motion_description", "supporting_fact_ids"]
    all_props = {k for _, n in walk(schema) if "properties" in n for k in n["properties"]}
    assert not {p for p in all_props if "category" in p or "confidence" in p or "label" in p}
    assert set(golden["objects"][0]) == set(obj_props)
    assert "category" not in HANDWRITTEN.read_text(encoding="utf-8")
    assert not hasattr(si, "CONFIDENCE_LEVELS")
    fields = set(si.ObjectInterpretation.__dataclass_fields__)
    assert fields == {"object_id", "motion_class", "motion_description", "supporting_fact_ids"}


@pytest.mark.parametrize("field, value", [("category_guess", "ball"), ("category_confidence", "low")])
def test_category_fields_are_rejected_by_schema(traj, golden, field, value):
    golden["objects"][0][field] = value
    rep = report(golden, traj)
    assert any("additional" in e.lower() or field in e for e in rep.errors), rep.errors


def test_object_ids_are_neutral(traj):
    assert traj.object_ids == ("obj_01",)
    assert all(not re.search(r"ball|block|box|cube|sphere", oid) for oid in two_object_set().object_ids)


def test_schema_marks_every_hypothesis_as_candidate(traj):
    schema = si.build_interpretation_schema(traj)
    statuses = [n for p, n in walk(schema) if p.endswith(".status")]
    assert len(statuses) == 3 and all(n["enum"] == ["candidate"] for n in statuses)
    supports = [n for p, n in walk(schema) if p.endswith(".support")]
    assert supports and all(n["enum"] == ["weak", "moderate", "strong"] for n in supports)


def test_handwritten_conforms_to_schema(traj, golden):
    assert validate_instance(golden, si.build_interpretation_schema(traj)) == []


# ------------------------------------------------------------------ the handwritten interpretation


def test_handwritten_facts_are_established_by_their_cited_observations(traj, golden):
    # Each claim is checked using ONLY the observations the fact cites.
    facts = {f["fact_id"]: f for f in golden["observed_facts"]}
    obj = traj.get("obj_01")

    def cited(fid):
        frames = [r["frame_index"] for r in facts[fid]["evidence"]]
        assert len(frames) <= si.MAX_EVIDENCE_REFS
        return {f: obj.observation_at(f) for f in frames}

    def ys(obs, frames):
        return [obs[f].y for f in frames]

    # F1: y increases on every step 0..19 -> needs every frame 0..19.
    o = cited("F1")
    assert sorted(o) == list(range(20))
    y = ys(o, range(20))
    assert all(b > a for a, b in zip(y, y[1:]))

    # F2: the step in y grows on every step during frames 0..18; first 1.11, last 38.89.
    o = cited("F2")
    assert sorted(o) == list(range(19))
    steps = [round(b - a, 2) for a, b in zip(ys(o, range(19)), ys(o, range(1, 19)))]
    assert all(b > a for a, b in zip(steps, steps[1:]))
    assert steps[0] == 1.11 and steps[-1] == 38.89

    # F3: y decreases on every step 19..31 and increases on every step 31..36.
    o = cited("F3")
    assert sorted(o) == list(range(19, 37))
    down, up = ys(o, range(19, 32)), ys(o, range(31, 37))
    assert all(b < a for a, b in zip(down, down[1:])) and all(b > a for a, b in zip(up, up[1:]))

    # F4: frame 19 is a local maximum of y (vs frames 18 and 20), with the quoted values.
    o = cited("F4")
    assert sorted(o) == [18, 19, 20]
    assert ys(o, [18, 19, 20]) == [430.0, 435.71, 409.18]
    assert o[19].y > o[18].y and o[19].y > o[20].y
    assert round(traj.coordinate_system.image_height - o[19].y) == 44

    # F5: x at the cited frames 0, 5, ..., 35 rises by the same 10.00 px between them.
    o = cited("F5")
    frames = list(range(0, 36, 5))
    assert sorted(o) == frames
    xs = [o[f].x for f in frames]
    assert xs == [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]
    assert {round(b - a, 2) for a, b in zip(xs, xs[1:])} == {10.0}

    # F6: over frames 19..36 the smallest y is 264.01 at frame 31, and it exceeds y at frame 0.
    o = cited("F6")
    assert sorted(o) == [0] + list(range(19, 37))
    after = {f: o[f].y for f in range(19, 37)}
    assert min(after.values()) == 264.01 and min(after, key=after.get) == 31
    assert after[31] > o[0].y == 70.0

    # The bounce evidence is kept: the event and the contact hypothesis cite F3 and F4.
    assert {"F3", "F4"} <= set(golden["event_hypotheses"][0]["supporting_fact_ids"])
    assert {"F3", "F4"} <= set(golden["physical_model_hypotheses"][1]["supporting_fact_ids"])


def test_handwritten_is_valid_with_no_warnings(traj, golden):
    rep = report(golden, traj)
    assert rep.ok, rep.errors
    assert rep.warnings == ()


def test_bounce_bracket_is_frames_18_to_20_with_19_inside(traj, golden):
    # The event window is a conservative bracket, not a timestamp. For the synthetic
    # bounce it runs from frame 18 to frame 20 and frame 19 is an interior observation:
    # positions alone show the reversal around frame 19 but not on which side of it.
    (window,) = report(golden, traj).event_windows
    assert window == si.EventWindow("E1", 18, 20, Fraction(3, 5), Fraction(2, 3), (19,))
    assert isinstance(window.t_after, Fraction) and isinstance(window.t_before, Fraction)
    assert window.t_after == traj.frame_time(18) and window.t_before == traj.frame_time(20)
    # The true impact (ground truth, never in the input) lies inside the bracket, but at
    # neither bound and not at the interior frame: the bracket does not date the event.
    t_hit = ground_truth_impact_time()
    assert window.t_after < t_hit < window.t_before
    assert t_hit != float(traj.frame_time(19))


def test_bracket_bounds_are_exact_fraction_arithmetic(traj, golden):
    (window,) = report(golden, traj).event_windows
    assert window.t_before - window.t_after == Fraction(1, 15)  # exactly two frame periods
    assert (window.t_after + window.t_before) / 2 == traj.frame_time(19)


def test_event_window_keeps_rational_precision():
    # Times that collide as floats must still give distinct, exact window bounds.
    a = Fraction(1, 3)
    b = a + Fraction(1, 9 * 10**17)
    ts = tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [{"object_id": "obj_01", "observations": [
            {"frame_index": f, "t": {"num": t.numerator, "den": t.denominator}, "x": 1.0, "y": 1.0}
            for f, t in ((0, Fraction(0)), (1, a), (2, b))
        ]}],
    })
    data = minimal_interpretation(("obj_01",))
    data["observed_facts"][0]["evidence"] = [{"object_id": "obj_01", "frame_index": 1}]
    data["event_hypotheses"] = [{
        "event_id": "E1", "status": "candidate", "type": "possible_stop", "object_ids": ["obj_01"],
        "after_frame_index": 1, "before_frame_index": 2, "support": "weak",
        "supporting_fact_ids": ["F1"], "rationale": "Position does not change.",
    }]
    (window,) = report(data, ts).event_windows
    assert float(window.t_after) == float(window.t_before)
    assert window.t_after == a and window.t_before == b and window.t_after < window.t_before


def test_parse_returns_data_structures(traj, golden):
    interp = si.parse_interpretation(golden, traj)
    assert [f.fact_id for f in interp.observed_facts] == ["F1", "F2", "F3", "F4", "F5", "F6"]
    assert interp.observed_facts[3].evidence == tuple(si.ObsRef("obj_01", f) for f in (18, 19, 20))
    assert interp.objects[0].motion_class == "bouncing_like"
    models = {(h.model, h.plane_hint) for h in interp.physical_model_hypotheses}
    assert models == {("uniform_gravity", "not_applicable"), ("plane_contact", "floor")}
    assert all(h.status == "candidate" for h in interp.physical_model_hypotheses)
    assert interp.event_hypotheses[0].type == "possible_bounce"
    assert interp.interaction_hypotheses == ()
    assert interp.event_windows[0].interior_frame_indices == (19,)


def test_parse_raises_with_errors_and_warnings(traj, golden):
    golden["physical_model_hypotheses"][0]["rationale"] = "This proves gravity is 9.81 m/s^2."
    with pytest.raises(si.SceneInterpretationError) as info:
        si.parse_interpretation(golden, traj)
    assert info.value.errors and any("certainty language" in w for w in info.value.warnings)


@pytest.mark.parametrize("bad", [None, [], "text", 3, {}])
def test_garbage_gives_schema_errors_not_exceptions(traj, bad):
    rep = report(bad, traj)
    assert not rep.ok and all(e.startswith("schema:") for e in rep.errors)


# ------------------------------------------------------------------ schema-level rejections


Mut = Callable[[dict], None]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra_field=1),
        lambda d: d["observed_facts"][0].update(confidence="high"),
        lambda d: d["observed_facts"][0]["evidence"][0].update(object_id="ghost_01"),
        lambda d: d["physical_model_hypotheses"][0].update(status="proven"),
        lambda d: d["physical_model_hypotheses"][0].update(support="certain"),
        lambda d: d["physical_model_hypotheses"][0].update(model="magic"),
        lambda d: d["physical_model_hypotheses"][0].update(gravity=2000.0),
        lambda d: d["event_hypotheses"][0].update(t=0.61),
        lambda d: d["event_hypotheses"][0].update(after_frame_index=18.0),
        lambda d: d["objects"][0].pop("motion_class"),
        lambda d: d.update(objects=[]),
        lambda d: d["observed_facts"][0].update(evidence=[]),
        lambda d: d.update(schema_version="decompile.scene_interpretation/2"),
    ],
)
def test_schema_violations(traj, golden, mutation: Mut):
    mutation(golden)
    rep = report(golden, traj)
    assert rep.errors and all(e.startswith("schema:") for e in rep.errors), rep.errors


# ------------------------------------------------------------------ semantic rejections (bounce fixture)


def _set(path: list, value) -> Mut:
    def fn(d):
        node = d
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return fn


H1, E1, F1 = ["physical_model_hypotheses", 0], ["event_hypotheses", 0], ["observed_facts", 0]


@pytest.mark.parametrize(
    "mutation, fragment",
    [
        (_set(F1 + ["fact_id"], "fact1"), "does not match"),
        (_set(H1 + ["hypothesis_id"], "F1"), "does not match"),
        (_set(["physical_model_hypotheses", 1, "hypothesis_id"], "H1"), "'H1' is declared more than once"),
        (_set(H1 + ["supporting_fact_ids"], ["F1", "F99"]), "'F99' is not a declared observed fact"),
        (_set(H1 + ["supporting_fact_ids"], ["H2"]), "'H2' is not a declared observed fact"),
        (_set(H1 + ["supporting_fact_ids"], ["F1", "F1"]), "duplicate fact reference 'F1'"),
        (_set(H1 + ["contradicting_fact_ids"], ["F1"]), "both supporting and contradicting"),
        (_set(["objects", 0, "supporting_fact_ids"], ["F7"]), "'F7' is not a declared observed fact"),
        (_set(F1 + ["evidence", 0, "frame_index"], 99), "has no observation at frame 99"),
        (_set(F1 + ["evidence", 0, "frame_index"], -1), "has no observation at frame -1"),
        (_set(F1 + ["evidence", 1], {"object_id": "obj_01", "frame_index": 0}), "duplicate reference"),
        (_set(F1 + ["evidence"], [{"object_id": "obj_01", "frame_index": k} for k in range(21)]),
         "21 references (max 20)"),
        (_set(F1 + ["object_ids"], ["obj_01", "obj_01"]), "duplicate 'obj_01'"),
        (lambda d: d["objects"].append(copy.deepcopy(d["objects"][0])), "interpreted more than once"),
        (_set(E1 + ["after_frame_index"], 20), "must be < before_frame_index"),
        (_set(E1 + ["after_frame_index"], 25), "must be < before_frame_index"),
        (_set(E1 + ["before_frame_index"], 40), "has no observation at frame 40"),
        (_set(E1 + ["after_frame_index"], 16), "too wide"),
        (_set(E1 + ["type"], "possible_collision"), "possible_collision needs at least two distinct objects"),
        (_set(E1 + ["object_ids"], ["obj_01", "obj_01"]), "duplicate 'obj_01'"),
        (_set(["physical_model_hypotheses", 1, "plane_hint"], "not_applicable"), "plane_contact needs a plane_hint"),
        (_set(H1 + ["plane_hint"], "floor"), "plane_hint must be 'not_applicable'"),
        (_set(H1 + ["model"], "object_collision"), "at least two distinct objects"),
        (_set(H1 + ["applies_to"], ["obj_01", "obj_01"]), "duplicate 'obj_01'"),
        (lambda d: d["interaction_hypotheses"].append({
            "interaction_id": "I1", "status": "candidate", "object_ids": ["obj_01", "obj_01"],
            "type": "collision", "support": "weak", "supporting_fact_ids": ["F3"], "rationale": "Reversal."}),
         "an interaction needs at least two distinct objects"),
        (_set(["ambiguities", 0, "about_ids"], ["E9"]), "'E9' is not a declared fact"),
        (_set(["ambiguities", 0, "about_ids"], ["A2"]), "'A2' is not a declared fact"),
        (_set(["ambiguities", 0, "about_ids"], ["E1", "E1"]), "duplicate 'E1'"),
        (_set(F1 + ["statement"], "   "), "must not be empty"),
        (_set(["ambiguities", 0, "alternatives"], [""]), "must not be empty"),
        (_set(H1 + ["rationale"], "x" * 601), "601 characters (max 600)"),
        (_set(["data_quality_notes"], ["note"] * 51), "51 items (max 50)"),
    ],
)
def test_semantic_errors(traj, golden, mutation: Mut, fragment: str):
    mutation(golden)
    assert_error(golden, traj, fragment)


@pytest.mark.parametrize(
    "after, before, interior",
    [
        (18, 19, ()),  # a single gap is a valid (narrower) bracket
        (19, 20, ()),  # also valid: the validator cannot know which gap holds the contact
        (18, 20, (19,)),  # the handwritten bracket
        (17, 20, (18, 19)),  # two interior observations: the limit
        (18, 21, (19, 20)),
    ],
)
def test_brackets_with_up_to_two_interior_frames_are_accepted(traj, golden, after, before, interior):
    golden["event_hypotheses"][0].update(after_frame_index=after, before_frame_index=before)
    (window,) = report(golden, traj).event_windows
    assert window.interior_frame_indices == interior
    assert (window.t_after, window.t_before) == (Fraction(after, 30), Fraction(before, 30))


@pytest.mark.parametrize("after, before", [(16, 20), (17, 21), (0, 36)])
def test_brackets_with_three_or_more_interior_frames_are_rejected(traj, golden, after, before):
    golden["event_hypotheses"][0].update(after_frame_index=after, before_frame_index=before)
    assert_error(golden, traj, f"max {si.MAX_EVENT_INTERIOR_FRAMES})")


def test_max_interior_frames_is_two():
    assert si.MAX_EVENT_INTERIOR_FRAMES == 2


# ------------------------------------------------------------------ multi-object rules


def test_minimal_two_object_interpretation_is_valid():
    rep = report(minimal_interpretation(), two_object_set())
    assert rep.ok and rep.warnings == (), (rep.errors, rep.warnings)


def test_missing_object_coverage():
    data = minimal_interpretation()
    data["objects"].pop()
    assert_error(data, two_object_set(), "no interpretation for input objects ['obj_02']")


def test_fact_object_ids_must_match_cited_evidence():
    data = minimal_interpretation()
    data["observed_facts"][0]["object_ids"] = ["obj_01", "obj_02"]
    assert_error(data, two_object_set(), "do not match the objects cited in evidence")


def test_event_frames_must_exist_for_every_involved_object():
    ts = two_object_set()
    data = minimal_interpretation()
    data["event_hypotheses"] = [{
        "event_id": "E1", "status": "candidate", "type": "possible_collision",
        "object_ids": ["obj_01", "obj_02"], "after_frame_index": 2, "before_frame_index": 3,
        "support": "weak", "supporting_fact_ids": ["F1", "F2"], "rationale": "They meet."}]
    rep = report(data, ts)
    assert rep.ok, rep.errors
    assert rep.event_windows[0].t_before == Fraction(1, 10)
    data["event_hypotheses"][0]["before_frame_index"] = 4
    assert_error(data, ts, "'obj_01' has no observation at frame 4")


def test_valid_interaction_and_irrelevant_fact_warning():
    data = minimal_interpretation()
    data["interaction_hypotheses"] = [{
        "interaction_id": "I1", "status": "candidate", "object_ids": ["obj_01", "obj_02"],
        "type": "independent", "support": "weak", "supporting_fact_ids": ["F1"], "rationale": "No contact seen."}]
    data["objects"][1]["supporting_fact_ids"] = ["F1"]  # F1 is about obj_01 only
    rep = report(data, two_object_set())
    assert rep.ok, rep.errors
    assert rep.warnings == ("objects[1].supporting_fact_ids: fact 'F1' is about none of ['obj_02']",)


def test_ambiguity_may_refer_to_objects():
    data = minimal_interpretation()
    data["ambiguities"][0]["about_ids"] = ["obj_02"]
    assert report(data, two_object_set()).ok


# ------------------------------------------------------------------ warnings


@pytest.mark.parametrize(
    "mutation, fragment",
    [
        (_set(H1 + ["rationale"], "This confirms a constant downward pull."), "certainty language 'confirms'"),
        (_set(E1 + ["rationale"], "Definitely a bounce."), "certainty language 'Definitely'"),
        (_set(["objects", 0, "motion_description"], "Proven to bounce."), "certainty language 'Proven'"),
        (_set(["ambiguities"], []), "ambiguities: none recorded"),
        (_set(H1 + ["supporting_fact_ids"], []), "no supporting facts"),
        (_set(E1 + ["supporting_fact_ids"], []), "no supporting facts"),
        (_set(H1 + ["alternatives_considered"], []), "no alternatives considered"),
        (lambda d: d["physical_model_hypotheses"][0].update(contradicting_fact_ids=["F6"]),
         "support 'strong' despite contradicting facts"),
    ],
)
def test_warnings_do_not_reject(traj, golden, mutation: Mut, fragment: str):
    mutation(golden)
    rep = report(golden, traj)
    assert rep.ok, rep.errors
    assert any(fragment in w for w in rep.warnings), rep.warnings


# ------------------------------------------------------------------ parameter leaks


LEAKS = [
    "gravity is about 9.8 m/s^2",
    "Gravity = 2000",
    "g = 2000",
    "g≈1960",
    "restitution of 0.7",
    "a restitution coefficient around 0.70",
    "the drag coefficient is 0.1",
    "it falls at 1200 px/s",
    "speed of 860 pixels per second",
    "accelerates at 2000 px/s²",
    "acceleration roughly 2000",
    "downward acceleration in px/s^2",
    "keeps about 70% of its speed",
    "rebound height is 68 % of the drop",
    "vy ≈ -863",
    "an energy loss of 0.51",
    "gravity is 9.81",
    "gravity is 9.8",
    "g ≈ 9.81",
    "g=9.8",
    "g ~ 9,81",
    "Earth's 9.81",
    "consistent with 9.81 for gravity",
    "free fall at 9.8",
    "gravitational pull near 9.80665",
    "9.81 m/s^2",
    "standard gravity, 9.81",
    "speed ratio of 0.7",
    "1.2e3 px/s",
]


@pytest.mark.parametrize("text", LEAKS)
def test_detect_parameter_leaks(text):
    assert si.detect_parameter_leaks(text), text


CLEAN = [
    "y rises from 435.71 to 264.01 between frames 19 and 31.",
    "x increases by 2.00 px per frame, from 100.0 at frame 0 to 172.0 at frame 36.",
    "The step in y grows by a similar amount each frame.",
    "Consistent with a constant downward acceleration.",
    "The largest observed y is 435.71 at frame 19, near the bottom of the 480 px image.",
    "Positions are given to 0.01 px and show no visible noise.",
    "Contact between frames 18 and 19, e.g. near frame 19.",
    "Speed appears to drop after the reversal.",
    "Comparing numerical fits with and without drag.",
    # Observed coordinates that happen to look like standard gravity are not estimates.
    "The observed y coordinate is 9.81 px at frame 4.",
    "x is 9.8 at frame 2 and 19.81 at frame 7.",
    "y changes from 98.1 to 9.85 between frames 3 and 4.",
    "Frame 9 has y = 9.81 and frame 10 has y = 12.4.",
    "Between frames 98 and 99 the object moves 9.8 px.",
    "At frame 12, x is 109.81 and y is 9.8; e.g. the top of the image.",
]


@pytest.mark.parametrize("text", CLEAN)
def test_observations_are_not_flagged_as_leaks(text):
    assert si.detect_parameter_leaks(text) == []


@pytest.mark.parametrize(
    "path",
    [
        ["observed_facts", 0, "statement"],
        ["objects", 0, "motion_description"],
        ["physical_model_hypotheses", 0, "rationale"],
        ["physical_model_hypotheses", 0, "alternatives_considered", 0],
        ["event_hypotheses", 0, "rationale"],
        ["ambiguities", 0, "description"],
        ["ambiguities", 0, "alternatives", 0],
        ["ambiguities", 0, "what_would_resolve"],
        ["data_quality_notes", 0],
    ],
)
def test_leaks_rejected_in_every_text_field(traj, golden, path):
    _set(path, "restitution of 0.7")(golden)
    assert_error(golden, traj, "numerical physical-parameter estimate")


def test_leak_in_interaction_rationale_rejected():
    data = minimal_interpretation()
    data["interaction_hypotheses"] = [{
        "interaction_id": "I1", "status": "candidate", "object_ids": ["obj_01", "obj_02"],
        "type": "collision", "support": "weak", "supporting_fact_ids": ["F1"],
        "rationale": "Impact speed of 120 px/s."}]
    assert_error(data, two_object_set(), "interaction_hypotheses[0].rationale: numerical physical-parameter")
