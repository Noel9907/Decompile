"""Tests for decompile.synth (P2, A4-lite independent generator).

OFFLINE, deterministic. Closed-form motion appears ONLY in these tests, as an independent
check of the numerical generator; the generator itself integrates numerically.
"""

from __future__ import annotations

import ast
import math
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import synth
from decompile import trajectories as tr


@pytest.fixture(scope="module")
def demo():
    return synth.generate(synth.PREREGISTERED_DEMO)


# ------------------------------------------------------------------ the pre-registered demo


def test_preregistered_parameters_are_unchanged():
    s = synth.PREREGISTERED_DEMO
    assert (s.fps, s.n_frames, s.x0, s.y0, s.vx0, s.vy0, s.ax, s.ay) == (30, 50, 100.0, 45.0, 60.0, 0.0, 0.0, 1000.0)
    assert s.planes == (synth.Plane("y", 450.0, "less", 0.3),)
    assert s.drag_k_per_s == 0.0 and s.round_decimals == 2
    assert synth.PREREGISTERED_TRAIN_FRACTION == Fraction(7, 10)


def test_demo_contacts_from_numerical_integration(demo):
    c = demo.ground_truth.contacts
    assert len(c) == 3
    # independent closed-form check (tests only): t1 = sqrt(2*405/1000), then flights 2*e^k*v1/a
    t1 = math.sqrt(2 * 405 / 1000)
    t2 = t1 + 2 * 0.3 * 900 / 1000
    t3 = t2 + 2 * 0.09 * 900 / 1000
    assert (t1, round(t2, 9), round(t3, 9)) == (0.9, 1.44, 1.602)
    for got, want in zip(c, (t1, t2, t3)):
        assert got.t_s == pytest.approx(want, abs=1e-6)
        assert got.y_px == 450.0 and got.axis == "y"
        assert got.v_normal_after_px_s == pytest.approx(-0.3 * got.v_normal_before_px_s, rel=1e-12)
    assert c[0].between_frames == (27, 27)  # lands on frame 27 (t = 0.9 s)
    assert c[1].between_frames == (43, 44) and c[2].between_frames == (48, 49)
    assert demo.ground_truth.rest_time_s is None


def test_demo_split_contains_first_contact_only(demo):
    n_train = math.floor(synth.PREREGISTERED_TRAIN_FRACTION * synth.PREREGISTERED_DEMO.n_frames)
    t_cut = Fraction(n_train, 30)  # first withheld frame time
    assert n_train == 35
    c = demo.ground_truth.contacts
    assert c[0].t_s < t_cut <= c[1].t_s < c[2].t_s


def test_timestamps_exact_and_positions_rounded(demo):
    obs = demo.trajectories.objects[0].observations
    assert len(obs) == 50
    for i, o in enumerate(obs):
        assert o.frame_index == i and o.t == Fraction(i, 30) and isinstance(o.t, Fraction)
        for v in (o.x, o.y):
            assert round(v, 2) == v
    gt = demo.ground_truth
    assert max(abs(o.x - x) for o, x in zip(obs, gt.true_x_px)) <= 0.005 + 1e-12
    assert max(abs(o.y - y) for o, y in zip(obs, gt.true_y_px)) <= 0.005 + 1e-12
    assert max(o.y for o in obs) <= 450.0


def test_trajectory_is_labelled_synthetic_and_carries_no_ground_truth(demo):
    t = demo.trajectories
    assert t.synthetic is True and t.source == "SYNTHETIC_NUMERICAL_GENERATOR" == synth.SOURCE_LABEL
    assert t.source != "SYNTHETIC_ANALYTIC_FIXTURE"
    data = t.to_json_dict()
    assert set(data) == {"schema", "synthetic", "source", "coordinate_system", "objects"}
    assert all(set(o) == {"frame_index", "t", "x", "y"} for o in data["objects"][0]["observations"])
    text = str(data).lower()
    # ground-truth vocabulary (the source label itself legitimately names the generator)
    for word in ("restitution", "contact", "truth", "plane", "generator_method", "substeps", "rk4", "true_y"):
        assert word not in text
    assert tr.TrajectorySet.from_json_dict(data) == t  # nothing beyond the trajectory schema
    assert "RK4" in demo.ground_truth.generator_method  # the method is recorded separately


def test_ground_truth_json_is_separate_and_finite(demo):
    data = synth.ground_truth_to_json(demo.ground_truth)
    assert data["kind"] == "synthetic_ground_truth"
    assert len(data["contacts"]) == 3 and data["spec"]["planes"][0]["restitution"] == 0.3
    assert len(data["true_y_px"]) == 50


def test_generation_is_deterministic():
    assert synth.generate(synth.PREREGISTERED_DEMO) == synth.generate(synth.PREREGISTERED_DEMO)


# ------------------------------------------------------------------ numerical method


def drag_exact(t, q0, v0, g, k):
    vt = g / k
    return q0 + vt * t + (v0 - vt) * (1 - math.exp(-k * t)) / k


def test_rk4_converges_at_fourth_order_with_drag():
    spec = synth.SceneSpec("obj_01", 30, 31, 100.0, 50.0, 300.0, -200.0, 0.0, 1500.0, drag_k_per_s=3.0)
    errors = []
    for substeps in (1, 2, 4):
        gt = synth.generate(spec, substeps=substeps).ground_truth
        errors.append(max(abs(y - drag_exact(f / 30, 50.0, -200.0, 1500.0, 3.0)) for f, y in enumerate(gt.true_y_px)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > 12 and errors[1] / errors[2] > 12  # ~16 for a 4th-order method
    fine = synth.generate(spec, substeps=200).ground_truth
    assert max(abs(y - drag_exact(f / 30, 50.0, -200.0, 1500.0, 3.0)) for f, y in enumerate(fine.true_y_px)) < 1e-8


def test_contact_time_converges_with_step_size():
    spec = synth.SceneSpec("obj_01", 30, 40, 100.0, 45.0, 60.0, 0.0, 0.0, 1000.0,
                           planes=(synth.Plane("y", 450.0, "less", 0.5),), drag_k_per_s=1.5)
    t = [synth.generate(spec, substeps=s).ground_truth.contacts[0].t_s for s in (2, 20, 200)]
    assert abs(t[1] - t[2]) < abs(t[0] - t[2]) and abs(t[1] - t[2]) < 1e-9


def test_zero_restitution_rests_on_the_plane():
    spec = synth.SceneSpec("obj_01", 30, 40, 100.0, 45.0, 60.0, 0.0, 0.0, 1000.0,
                           planes=(synth.Plane("y", 450.0, "less", 0.0),))
    gt = synth.generate(spec, substeps=200).ground_truth
    assert len(gt.contacts) == 1 and gt.contacts[0].resting and gt.rest_time_s == pytest.approx(0.9, abs=1e-9)
    assert all(y == 450.0 for f, y in enumerate(gt.true_y_px) if f > 27)
    assert gt.true_x_px[-1] == pytest.approx(100 + 60 * 39 / 30, abs=1e-9)  # slides on


@pytest.mark.parametrize("kwargs", [dict(fps=0), dict(n_frames=2), dict(x0=float("nan")), dict(drag_k_per_s=-1.0)])
def test_invalid_specs(kwargs):
    base = dict(object_id="obj_01", fps=30, n_frames=10, x0=0.0, y0=0.0, vx0=0.0, vy0=0.0, ax=0.0, ay=0.0) | kwargs
    with pytest.raises(synth.SynthError):
        synth.SceneSpec(**base)


def test_invalid_planes_and_start_positions():
    with pytest.raises(synth.SynthError):
        synth.Plane("y", 450.0, "less", 1.5)
    with pytest.raises(synth.SynthError):
        synth.Plane("z", 450.0, "less", 0.5)
    spec = synth.SceneSpec("obj_01", 30, 10, 0.0, 460.0, 0.0, 0.0, 0.0, 10.0, planes=(synth.Plane("y", 450.0, "less", 0.5),))
    with pytest.raises(synth.SynthError):
        synth.generate(spec)


# ------------------------------------------------------------------ independence


def test_generator_is_independent_of_fitter_and_simulator():
    path = Path(synth.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported, from_decompile = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            if node.module == "decompile":
                from_decompile |= {a.name for a in node.names}
    assert imported == {"__future__", "json", "math", "dataclasses", "fractions", "decompile"}, imported
    assert from_decompile == {"trajectories"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("simulate", "analyze_kinematics", "program_from_kinematics", "scene_program", "kinematic_fit"):
        assert forbidden not in names, forbidden
