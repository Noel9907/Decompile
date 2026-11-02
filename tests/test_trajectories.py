"""Tests for decompile.trajectories (B1b-1). OFFLINE: pure functions, no API calls."""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import trajectories as tr

FIXTURES = Path(__file__).parent / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"

# Ground truth of the SYNTHETIC fixture. It lives only here, never in the fixture file.
GT_Y0, GT_A, GT_FLOOR, GT_X0, GT_VX, GT_SCALE, GT_FPS = 70.0, 2000.0, 450.0, 100.0, 60.0, 0.7, 30


def gt_position(t: float) -> tuple[float, float]:
    t_hit = math.sqrt(2 * (GT_FLOOR - GT_Y0) / GT_A)
    if t <= t_hit:
        y = GT_Y0 + 0.5 * GT_A * t * t
    else:
        d = t - t_hit
        y = GT_FLOOR - GT_SCALE * GT_A * t_hit * d + 0.5 * GT_A * d * d
    return GT_X0 + GT_VX * t, y


def obs(frame, num, den, x=10.0, y=20.0):
    return {"frame_index": frame, "t": {"num": num, "den": den}, "x": x, "y": y}


def obs_at(frame, fps=30, **kw):
    t = Fraction(frame, fps)
    return obs(frame, t.numerator, t.denominator, **kw)


def valid_data(n_obs=3, object_ids=("obj_01",)):
    return {
        "schema": tr.SCHEMA,
        "synthetic": True,
        "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {
            "units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
            "image_width": 640, "image_height": 480,
        },
        "objects": [
            {"object_id": oid, "observations": [obs_at(k) for k in range(n_obs)]}
            for oid in object_ids
        ],
    }


def reasons_of(data) -> list[str]:
    with pytest.raises(tr.TrajectoryError) as info:
        tr.TrajectorySet.from_json_dict(data)
    return info.value.reasons


def assert_rejected(data, fragment: str) -> None:
    reasons = reasons_of(data)
    assert any(fragment in r for r in reasons), reasons


# ------------------------------------------------------------------ the fixture


def test_fixture_loads_and_is_labelled_synthetic():
    ts = tr.load_trajectories(BOUNCE)
    assert BOUNCE.name.startswith("synthetic_")
    assert ts.synthetic is True and ts.source == "SYNTHETIC_ANALYTIC_FIXTURE"
    assert ts.object_ids == ("obj_01",)
    assert len(ts.objects[0].observations) == 37
    assert ts.warnings == ()
    assert ts.coordinate_system == tr.CoordinateSystem(640, 480)


def test_fixture_timestamps_are_exact_fractions():
    ts = tr.load_trajectories(BOUNCE)
    for k, ob in enumerate(ts.objects[0].observations):
        assert ob.frame_index == k
        assert isinstance(ob.t, Fraction) and ob.t == Fraction(k, GT_FPS)


def test_fixture_matches_independent_analytic_motion():
    ts = tr.load_trajectories(BOUNCE)
    for ob in ts.objects[0].observations:
        x, y = gt_position(float(ob.t))
        assert abs(ob.x - x) <= 0.005 and abs(ob.y - y) <= 0.005, ob


def test_fixture_contains_no_ground_truth_or_expected_answers():
    text = BOUNCE.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"schema", "synthetic", "source", "coordinate_system", "objects"}
    for o in data["objects"]:
        assert set(o) == {"object_id", "observations"}
        for ob in o["observations"]:
            assert set(ob) == {"frame_index", "t", "x", "y"}
    lowered = text.lower()
    for word in ("gravity", "restitution", "drag", "bounce", "floor", "truth", "expected",
                 "hypothes", "interpret", "2000", "0.7,", "ball", "category"):
        assert word not in lowered, word


def test_round_trip_is_exact():
    raw = json.loads(BOUNCE.read_text(encoding="utf-8"))
    ts = tr.TrajectorySet.from_json_dict(raw)
    assert ts.to_json_dict() == raw
    again = tr.loads_trajectories(json.dumps(ts.to_json_dict()))
    assert again == ts


def test_frame_time_and_lookup():
    ts = tr.load_trajectories(BOUNCE)
    assert ts.frame_time(18) == Fraction(3, 5)
    assert ts.get("obj_01").observation_at(19).y == 435.71
    assert ts.get("obj_01").observation_at(99) is None
    with pytest.raises(KeyError):
        ts.frame_time(99)
    with pytest.raises(KeyError):
        ts.get("nope")


# ------------------------------------------------------------------ timestamps


@pytest.mark.parametrize(
    "t, expected",
    [
        (Fraction(0), {"num": 0, "den": 1}),
        (Fraction(1, 30), {"num": 1, "den": 30}),
        (Fraction(2, 4), {"num": 1, "den": 2}),
        (Fraction(1001, 30000), {"num": 1001, "den": 30000}),
    ],
)
def test_timestamp_serialisation(t, expected):
    assert tr.timestamp_to_json(t) == expected
    assert tr.timestamp_from_json(expected) == t


@pytest.mark.parametrize("bad", [0.5, 1, "1/30", None])
def test_timestamp_to_json_requires_fraction(bad):
    with pytest.raises(TypeError):
        tr.timestamp_to_json(bad)


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ({"num": True, "den": 30}, "bool not accepted"),
        ({"num": 1, "den": True}, "bool not accepted"),
        ({"num": 1.0, "den": 30}, "must be an integer"),
        ({"num": 1, "den": 30.0}, "must be an integer"),
        ({"num": "1", "den": 30}, "must be an integer"),
        ({"num": 1, "den": 0}, "must be positive"),
        ({"num": 1, "den": -30}, "must be positive"),
        ({"num": -1, "den": 30}, "must be >= 0"),
        ({"num": 2, "den": 60}, "not in lowest terms"),
        ({"num": 0, "den": 30}, "not in lowest terms"),
        ({"num": 2**63, "den": 1}, "2**63-1"),
        ({"num": 1, "den": 30, "float": 0.033}, "unknown keys"),
        ({"num": 1}, "missing keys"),
        ([1, 30], "must be an object"),
        (0.0333, "must be an object"),
        (None, "must be an object"),
    ],
)
def test_timestamp_rejections(bad, fragment):
    with pytest.raises(tr.TrajectoryError) as info:
        tr.timestamp_from_json(bad)
    assert any(fragment in r for r in info.value.reasons), info.value.reasons


def test_ordering_is_exact_even_when_floats_collide():
    a = Fraction(1, 3)
    b = a + Fraction(1, 9 * 10**17)  # (3e17+1)/9e17, lowest terms, within int64
    assert float(a) == float(b) and a < b
    data = valid_data()
    data["objects"][0]["observations"] = [
        obs(0, 0, 1),
        obs(1, a.numerator, a.denominator),
        obs(2, b.numerator, b.denominator),
    ]
    ts = tr.TrajectorySet.from_json_dict(data)
    assert ts.objects[0].observations[2].t == b  # kept exactly, not rounded to float

    data["objects"][0]["observations"][1:] = [
        obs(1, b.numerator, b.denominator),
        obs(2, a.numerator, a.denominator),
    ]
    assert_rejected(data, "t: not strictly increasing")


def test_equal_times_rejected():
    data = valid_data()
    data["objects"][0]["observations"][2] = obs(2, 1, 30)
    assert_rejected(data, "t: not strictly increasing")


def test_timestamps_need_not_be_uniform():
    data = valid_data()
    data["objects"][0]["observations"] = [obs(0, 0, 1), obs(1, 1001, 30000), obs(5, 1, 6)]
    tr.TrajectorySet.from_json_dict(data)


# ------------------------------------------------------------------ structure and labels


def mutate(fn):
    data = valid_data()
    fn(data)
    return data


@pytest.mark.parametrize(
    "fn, fragment",
    [
        (lambda d: d.update(ground_truth={"g": 2000}), "unknown keys ['ground_truth']"),
        (lambda d: d.update(notes="ball bounces"), "unknown keys ['notes']"),
        (lambda d: d.pop("source"), "missing keys ['source']"),
        (lambda d: d.update(schema="decompile.trajectories/2"), "$.schema"),
        (lambda d: d.update(synthetic=False), "contradicts source"),
        (lambda d: d.update(synthetic=1), "must be a boolean"),
        (lambda d: d.update(source="CLASSICAL_TRACKER"), "contradicts source"),
        (lambda d: d.update(source="SAM"), "must be one of"),
        (lambda d: d["coordinate_system"].update(y_axis="up"), "y_axis"),
        (lambda d: d["coordinate_system"].update(units="m"), "units"),
        (lambda d: d["coordinate_system"].update(image_width=0), "image_width"),
        (lambda d: d["coordinate_system"].update(image_height=True), "image_height"),
        (lambda d: d["coordinate_system"].update(scale_m_per_px=0.01), "unknown keys"),
        (lambda d: d.update(coordinate_system=None), "must be an object"),
        (lambda d: d["objects"][0].update(label="ball"), "unknown keys ['label']"),
        (lambda d: d["objects"][0]["observations"][0].update(vx=3.0), "unknown keys ['vx']"),
        (lambda d: d["objects"][0]["observations"][0].pop("t"), "missing keys ['t']"),
        (lambda d: d["objects"][0]["observations"][0].pop("x"), "missing keys ['x']"),
        (lambda d: d.update(objects=[]), "1..8 objects"),
        (lambda d: d.update(objects={}), "must be an array"),
        (lambda d: d["objects"][0].update(observations=d["objects"][0]["observations"][:2]), "3..300"),
        (lambda d: d["objects"][0].update(observations=None), "must be an array"),
    ],
)
def test_structure_rejections(fn, fragment):
    assert_rejected(mutate(fn), fragment)


def test_real_source_with_synthetic_false_is_accepted():
    data = valid_data()
    data.update(synthetic=False, source="CLASSICAL_TRACKER")
    assert tr.TrajectorySet.from_json_dict(data).synthetic is False


@pytest.mark.parametrize("bad", ["Ball_01", "1ball", "ball-01", "ball 01", "", "b" * 33, "obj_01\n", 7, None])
def test_object_id_rejected(bad):
    data = valid_data()
    data["objects"][0]["object_id"] = bad
    assert_rejected(data, "object_id: must match")


def test_object_id_limits_accepted():
    data = valid_data(object_ids=("b", "b" * 32, "obj_2_x"))
    assert tr.TrajectorySet.from_json_dict(data).object_ids == ("b", "b" * 32, "obj_2_x")


def test_duplicate_object_id_rejected():
    assert_rejected(valid_data(object_ids=("obj_01", "obj_01")), "duplicate 'obj_01'")


def test_object_count_limit():
    tr.TrajectorySet.from_json_dict(valid_data(object_ids=tuple(f"o{i}" for i in range(8))))
    assert_rejected(valid_data(object_ids=tuple(f"o{i}" for i in range(9))), "1..8 objects")


def test_observation_count_limits():
    tr.TrajectorySet.from_json_dict(valid_data(n_obs=300))
    assert_rejected(valid_data(n_obs=301), "3..300")
    assert_rejected(valid_data(n_obs=201, object_ids=("a", "b", "c")), "603 observations in total (max 600)")


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None, 2**63])
def test_frame_index_rejected(bad):
    data = valid_data()
    data["objects"][0]["observations"][1]["frame_index"] = bad
    assert_rejected(data, "frame_index: must be a non-negative integer")


def test_frame_index_must_increase():
    data = valid_data()
    data["objects"][0]["observations"][2]["frame_index"] = 1
    assert_rejected(data, "frame_index: not strictly increasing")


@pytest.mark.parametrize("bad", [True, "10", None, [1], float("inf")])
def test_coordinate_value_rejected(bad):
    data = valid_data()
    data["objects"][0]["observations"][1]["x"] = bad
    assert_rejected(data, "x: must be a finite number")


def test_integer_coordinates_accepted_as_float():
    data = valid_data()
    data["objects"][0]["observations"][1]["x"] = 12
    ob = tr.TrajectorySet.from_json_dict(data).objects[0].observations[1]
    assert ob.x == 12.0 and isinstance(ob.x, float)


def test_outside_image_is_a_warning_not_an_error():
    data = valid_data()
    data["objects"][0]["observations"][1]["y"] = 481.0
    ts = tr.TrajectorySet.from_json_dict(data)
    assert len(ts.warnings) == 1 and "outside the image" in ts.warnings[0]


def test_shared_clock_mismatch_rejected():
    data = valid_data(object_ids=("a", "b"))
    data["objects"][1]["observations"][1] = obs(1, 1, 29)
    assert_rejected(data, "disagrees with")


def test_time_must_follow_frame_order_across_objects():
    data = valid_data(object_ids=("a", "b"))
    data["objects"][0]["observations"] = [obs(0, 0, 1), obs(1, 1, 30), obs(3, 1, 10)]
    data["objects"][1]["observations"] = [obs(0, 0, 1), obs(2, 1, 5), obs(4, 2, 5)]  # frame 2 after frame 3
    assert_rejected(data, "time does not increase with frame index across objects")


def test_all_reasons_are_reported_together():
    data = valid_data()
    data["synthetic"] = "yes"
    data["objects"][0]["object_id"] = "Bad"
    data["objects"][0]["observations"][0]["t"] = {"num": 1, "den": 0}
    assert len(reasons_of(data)) >= 3


# ------------------------------------------------------------------ strict JSON loading


@pytest.mark.parametrize(
    "text, fragment",
    [
        ('{"schema": "x", "schema": "y"}', "duplicate JSON object key"),
        ("NaN", "non-finite"),
        ('{"x": Infinity}', "non-finite"),
        ('{"x": 1e999}', "not finite"),
        ("[1, 2]", "top level must be a JSON object"),
        ("not json", "not strict JSON"),
    ],
)
def test_strict_json(text, fragment):
    with pytest.raises(tr.TrajectoryError) as info:
        tr.loads_trajectories(text)
    assert fragment in str(info.value)


def test_nan_inside_observation_rejected_by_strict_json():
    text = json.dumps(valid_data()).replace('"x": 10.0', '"x": NaN', 1)
    with pytest.raises(tr.TrajectoryError, match="non-finite"):
        tr.loads_trajectories(text)


def test_non_utf8_file_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_bytes(b'{"schema": "\xff"}')
    with pytest.raises(tr.TrajectoryError, match="not UTF-8"):
        tr.load_trajectories(p)


def test_error_names_the_file(tmp_path):
    p = tmp_path / "t.json"
    data = copy.deepcopy(valid_data())
    data["extra"] = 1
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(tr.TrajectoryError) as info:
        tr.load_trajectories(p)
    assert info.value.source == str(p) and str(p) in str(info.value)
