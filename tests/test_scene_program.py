"""Tests for decompile.scene_program (P1, A9-lite).

OFFLINE, deterministic. Trajectories are SYNTHETIC (the existing fixture or closed forms
written here). The fine-step reference integrator below is independent of the simulator:
it accumulates small fixed steps and finds crossings by bisection, instead of solving
contact times in closed form. The fixture's ground truth is read from
tests/test_trajectories.py by tests only.

Reproducing the fitted observations is a simulator CONSISTENCY test, not held-out
predictive validation; nothing here demonstrates predictive decompilation.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import kinematic_fit as kf
from decompile import scene_program as sp
from decompile import trajectories as tr

TESTS = Path(__file__).parent
BOUNCE = TESTS / "fixtures" / "synthetic_bounce_single.trajectories.json"


def program(x0=100.0, y0=100.0, vx0=50.0, vy0=0.0, ax=0.0, ay=1500.0, planes=(), t0=Fraction(0)):
    return sp.SceneProgram(sp.Body("obj_01", t0, x0, y0, vx0, vy0), sp.UniformAcceleration(ax, ay), planes)


def floor(y=400.0, e=0.8):
    return sp.AxisPlane("y", y, "less", e)


def frames(n, fps=30, start=0):
    return [Fraction(start + i, fps) for i in range(n)]


def reference(prog: sp.SceneProgram, times, dt=Fraction(1, 30000)):
    """Independent fine-step integrator: exact constant-acceleration steps, crossings by bisection."""
    q = [prog.body.x0, prog.body.y0]
    v = [prog.body.vx0, prog.body.vy0]
    a = [prog.acceleration.ax, prog.acceleration.ay]
    h = float(dt)
    targets = sorted(set(times))
    out, t, k = {}, Fraction(0), 0
    while k < len(targets):
        while k < len(targets) and targets[k] - prog.body.t0 == t:
            out[targets[k]] = (q[0], q[1])
            k += 1
        if k == len(targets):
            break
        remaining = h
        while remaining > 0:
            step = remaining
            nq = [q[i] + v[i] * step + 0.5 * a[i] * step * step for i in range(2)]
            hit = None
            for p in prog.planes:
                i = sp.AXES.index(p.axis)
                if p.sign * (nq[i] - p.position_px) > 0:
                    hit = (p, i)
            if hit is None:
                v = [v[i] + a[i] * step for i in range(2)]
                q = nq
                remaining = 0.0
                continue
            p, i = hit
            lo, hi = 0.0, step
            for _ in range(80):  # bisection on the crossing inside this step
                mid = 0.5 * (lo + hi)
                if p.sign * (q[i] + v[i] * mid + 0.5 * a[i] * mid * mid - p.position_px) > 0:
                    hi = mid
                else:
                    lo = mid
            s = 0.5 * (lo + hi)
            q = [q[j] + v[j] * s + 0.5 * a[j] * s * s for j in range(2)]
            v = [v[j] + a[j] * s for j in range(2)]
            q[i] = p.position_px
            v[i] = -p.restitution * v[i]
            remaining -= s
        t += dt
    return [out[x] for x in times]


@pytest.fixture(scope="module")
def traj():
    return tr.load_trajectories(BOUNCE)


@pytest.fixture(scope="module")
def kin_obj(traj):
    return kf.analyze_kinematics(traj).objects[0]


@pytest.fixture(scope="module")
def kin_obj_supplied(traj):
    return kf.analyze_kinematics(traj, position_sigma_px=0.01 / math.sqrt(12)).objects[0]


@pytest.fixture(scope="module")
def fixture_program(kin_obj):
    return sp.program_from_kinematics(kin_obj)


def ground_truth():
    spec = importlib.util.spec_from_file_location("_gt", TESTS / "test_trajectories.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ free flight


def test_free_flight_at_arbitrary_fraction_times():
    prog = program(x0=10, y0=20, vx0=33, vy0=-140, ax=-7, ay=900, t0=Fraction(7, 3))
    times = [Fraction(7, 3), Fraction(7, 3) + Fraction(1, 7), Fraction(5, 2), Fraction(1234567, 100000), Fraction(3)]
    r = sp.simulate(prog, times)
    assert r.times == tuple(times) and all(isinstance(t, Fraction) for t in r.times)
    for t, x, y, vx, vy in zip(times, r.x_px, r.y_px, r.vx_px_s, r.vy_px_s):
        d = float(t - Fraction(7, 3))
        assert x == pytest.approx(10 + 33 * d - 3.5 * d * d, abs=1e-9)
        assert y == pytest.approx(20 - 140 * d + 450 * d * d, abs=1e-9)
        assert vx == pytest.approx(33 - 7 * d, abs=1e-9) and vy == pytest.approx(-140 + 900 * d, abs=1e-9)
    assert r.events == () and r.termination == "end"


def test_query_order_and_duplicates_preserved():
    prog = program(planes=(floor(),))
    times = [Fraction(3, 2), Fraction(1, 10), Fraction(3, 2), Fraction(0)]
    r = sp.simulate(prog, times)
    assert r.times == tuple(times)
    assert r.y_px[0] == r.y_px[2]
    assert r.y_px[3] == 100.0


def test_time_validation():
    prog = program(t0=Fraction(1))
    with pytest.raises(ValueError):
        sp.simulate(prog, [Fraction(1, 2)])
    with pytest.raises(TypeError):
        sp.simulate(prog, [1.5])
    with pytest.raises(TypeError):
        sp.simulate(prog, [True])
    assert sp.simulate(prog, []).times == ()
    assert sp.simulate(prog, [2]).times == (Fraction(2),)


# ------------------------------------------------------------------ contacts


def test_three_bounces_match_independent_fine_step_reference():
    prog = program(planes=(floor(400.0, 0.8),))
    times = frames(76)  # 2.5 s at 30 fps
    r = sp.simulate(prog, times)
    assert len(r.events) == 3
    ref = reference(prog, times)
    err = max(max(abs(x - rx), abs(y - ry)) for x, y, (rx, ry) in zip(r.x_px, r.y_px, ref))
    assert err < 1e-6, err
    # closed-form contact times of this scene
    t1 = math.sqrt(2 * 300 / 1500)
    v1 = 1500 * t1
    assert r.events[0].t_s == pytest.approx(t1, abs=1e-12)
    assert r.events[1].t_s == pytest.approx(t1 + 2 * 0.8 * v1 / 1500, abs=1e-12)
    assert r.events[2].t_s == pytest.approx(t1 + 2 * 0.8 * v1 / 1500 + 2 * 0.64 * v1 / 1500, abs=1e-12)
    for ev in r.events:
        assert ev.y_px == 400.0 and ev.v_normal_after_px_s == pytest.approx(-0.8 * ev.v_normal_before_px_s)
    assert all(y <= 400.0 for y in r.y_px)


def test_ceiling_like_contact():
    ceiling = sp.AxisPlane("y", 50.0, "greater", 0.6)
    prog = program(y0=300.0, vy0=-900.0, ay=1000.0, planes=(ceiling,))
    times = frames(46)
    r = sp.simulate(prog, times)
    (ev,) = r.events
    assert ev.axis == "y" and ev.y_px == 50.0
    assert ev.v_normal_before_px_s < 0 and ev.v_normal_after_px_s == pytest.approx(-0.6 * ev.v_normal_before_px_s)
    assert min(r.y_px) >= 50.0
    ref = reference(prog, times)
    assert max(abs(y - ry) for y, (_, ry) in zip(r.y_px, ref)) < 1e-6


def test_wall_like_contact_keeps_tangential_motion():
    wall = sp.AxisPlane("x", 500.0, "less", 0.5)
    prog = program(x0=100.0, vx0=300.0, y0=100.0, vy0=-200.0, ay=300.0, planes=(wall,))
    times = frames(61)
    r = sp.simulate(prog, times)
    (ev,) = r.events
    assert ev.axis == "x" and ev.x_px == 500.0 and ev.t_s == pytest.approx(4 / 3, abs=1e-12)
    for t, y, vy in zip(times, r.y_px, r.vy_px_s):  # y is untouched by the x contact
        d = float(t)
        assert y == pytest.approx(100 - 200 * d + 150 * d * d, abs=1e-9)
        assert vy == pytest.approx(-200 + 300 * d, abs=1e-9)
    assert max(r.x_px) <= 500.0
    assert r.vx_px_s[-1] == pytest.approx(-150.0)


def test_zero_restitution_comes_to_rest_and_slides():
    prog = program(planes=(floor(400.0, 0.0),), vx0=40.0)
    times = frames(61)
    r = sp.simulate(prog, times)
    (ev,) = r.events
    assert ev.resting and ev.v_normal_after_px_s == 0.0
    assert r.termination == "resting" and r.rest_time_s == pytest.approx(math.sqrt(0.4))
    after = [i for i, t in enumerate(times) if float(t) > ev.t_s]
    assert after and all(r.y_px[i] == 400.0 and r.vy_px_s[i] == 0.0 for i in after)
    assert all(r.x_px[i] == pytest.approx(100 + 40 * float(times[i]), abs=1e-9) for i in after)


def test_elastic_restitution_conserves_bounce_height():
    prog = program(planes=(floor(400.0, 1.0),))
    times = [Fraction(i, 100) for i in range(500)]
    r = sp.simulate(prog, times)
    assert r.termination == "end" and len(r.events) >= 3
    for ev in r.events:
        assert abs(ev.v_normal_after_px_s) == pytest.approx(abs(ev.v_normal_before_px_s), rel=1e-12)
    gaps = np.diff([ev.t_s for ev in r.events])
    assert np.allclose(gaps, gaps[0], rtol=1e-9)  # periodic
    assert min(r.y_px) >= 100.0 - 1e-9  # never above the release height


def test_zeno_sequence_terminates_at_rest_without_zero_time_contacts():
    prog = program(planes=(floor(400.0, 0.5),))
    r = sp.simulate(prog, [Fraction(10)])
    assert r.termination == "resting"
    assert 10 < len(r.events) < 60
    times = [ev.t_s for ev in r.events]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert r.events[-1].resting and r.y_px[0] == 400.0
    t1 = math.sqrt(0.4)
    assert r.rest_time_s == pytest.approx(t1 * (1 + 2 * 0.5 / (1 - 0.5)), rel=1e-6)  # geometric series limit


def test_max_events_guard():
    prog = program(planes=(floor(400.0, 1.0),))
    with pytest.raises(sp.SimulationLimitError):
        sp.simulate(prog, [Fraction(20)], max_events=3)
    assert len(sp.simulate(prog, [Fraction(1)], max_events=3).events) == 1


def test_simultaneous_contacts_are_rejected():
    # wall at x = 400 reached at t = 1 s; floor at y = 400 reached at t = 1 s (a corner)
    prog = program(x0=100.0, vx0=300.0, y0=100.0, vy0=0.0, ay=600.0,
                   planes=(sp.AxisPlane("x", 400.0, "less", 0.5), floor(400.0, 0.5)))
    with pytest.raises(sp.SimultaneousContactError):
        sp.simulate(prog, [Fraction(2)])
    near = program(x0=100.0, vx0=300.0, y0=100.0, vy0=0.0, ay=600.0,
                   planes=(sp.AxisPlane("x", 400.3, "less", 0.5), floor(400.0, 0.5)))
    assert len(sp.simulate(near, [Fraction(3, 2)]).events) == 2  # 1 ms apart: handled in order


def test_body_that_never_returns():
    prog = program(y0=100.0, vy0=900.0, ay=-500.0, planes=(floor(400.0, 0.9),))
    r = sp.simulate(prog, [Fraction(5)])
    assert len(r.events) == 1 and r.termination == "end"


# ------------------------------------------------------------------ program validation and JSON


@pytest.mark.parametrize("kwargs", [
    dict(restitution=1.2), dict(restitution=-0.1), dict(restitution=float("nan")), dict(position_px=float("inf")),
    dict(axis="z"), dict(keep_side="above"),
])
def test_invalid_planes(kwargs):
    args = dict(axis="y", position_px=400.0, keep_side="less", restitution=0.5) | kwargs
    with pytest.raises(sp.ProgramError):
        sp.AxisPlane(**args)


def test_invalid_programs():
    with pytest.raises(sp.ProgramError):
        program(y0=450.0, planes=(floor(400.0),))  # starts beyond the boundary
    with pytest.raises(sp.ProgramError):
        program(y0=400.0, planes=(floor(400.0),))  # starts exactly on it
    with pytest.raises(sp.ProgramError):
        sp.Body("obj_01", 0.5, 0, 0, 0, 0)  # float t0
    with pytest.raises(sp.ProgramError):
        sp.Body("obj_01", Fraction(0), float("nan"), 0, 0, 0)
    with pytest.raises(sp.ProgramError):
        sp.SceneProgram(sp.Body("o", Fraction(0), 0, 0, 0, 0), sp.UniformAcceleration(0, 1), (),
                        {"bad": float("inf")})


def test_json_round_trip_and_hash(fixture_program):
    data = sp.program_to_json(fixture_program)
    assert data["body"]["t0"] == {"num": 0, "den": 1}
    assert data["units"] == {"length": "px", "time": "s"}
    back = sp.program_from_json(data)
    assert back == fixture_program
    assert sp.program_to_json(back) == data
    h = sp.program_sha256(fixture_program)
    assert h == sp.program_sha256(back) == sp.program_sha256(sp.program_from_kinematics(
        kf.analyze_kinematics(tr.load_trajectories(BOUNCE)).objects[0]))
    changed = replace(fixture_program, planes=(replace(fixture_program.planes[0], restitution=0.5),))
    assert sp.program_sha256(changed) != h


@pytest.mark.parametrize("edit", [
    lambda d: d.update(program_version="decompile.scene_program/0"),
    lambda d: d.update(extra=1),
    lambda d: d["body"].update(t0={"num": 2, "den": 4}),
    lambda d: d["body"].update(x0="1"),
    lambda d: d["planes"][0].update(restitution=1.5),
    lambda d: d["planes"][0].pop("basis"),
    lambda d: d.update(units={"length": "m", "time": "s"}),
])
def test_json_rejects_invalid(fixture_program, edit):
    data = sp.program_to_json(fixture_program)
    edit(data)
    with pytest.raises(sp.ProgramError):
        sp.program_from_json(data)


def test_simulation_is_deterministic(fixture_program, traj):
    times = list(o.t for o in traj.objects[0].observations) + [Fraction(2)]
    assert sp.simulate(fixture_program, times) == sp.simulate(fixture_program, times)


# ------------------------------------------------------------------ the existing fixture


@pytest.mark.parametrize("which, adequacy", [("kin_obj", "not_assessable"), ("kin_obj_supplied", "adequate")])
def test_fixture_program_recovers_boundary_and_restitution(request, which, adequacy):
    prog = sp.program_from_kinematics(request.getfixturevalue(which))
    (plane,) = prog.planes
    assert plane.axis == "y" and plane.keep_side == "less"
    assert plane.position_px == pytest.approx(450.0, abs=0.01)
    assert plane.restitution == pytest.approx(0.7, abs=1e-3)
    assert "not a visually observed surface" in plane.basis
    assert prog.body.t0 == Fraction(0)
    assert prog.acceleration.ay == pytest.approx(2000.0, abs=0.05)
    pv = prog.provenance
    assert pv["fit_adequacy"] == adequacy
    if adequacy == "not_assessable":
        assert "not independently assessed" in pv["fit_adequacy_note"]
    assert pv["tangential_axis"]["axis"] == "x"
    assert pv["fitted_span"]["t_last"] == {"num": 6, "den": 5}
    assert "consistency check, not held-out validation" in pv["note"]


def test_fixture_program_reproduces_the_fit_and_observations(fixture_program, kin_obj, traj):
    """Consistency check of the simulator (the fit was made on these same observations)."""
    obs = traj.objects[0].observations
    r = sp.simulate(fixture_program, [o.t for o in obs])
    p = kin_obj.fits[kf.IMPULSE].params
    for o, x, y in zip(obs, r.x_px, r.y_px):
        t = float(o.t)
        d = max(0.0, t - p["t_contact_s"])
        assert x == pytest.approx(p["x.q0"] + p["x.v0"] * t + 0.5 * p["x.a"] * t * t + p["x.dv1"] * d, abs=1e-6)
        assert y == pytest.approx(p["y.q0"] + p["y.v0"] * t + 0.5 * p["y.a"] * t * t + p["y.dv1"] * d, abs=1e-6)
        assert abs(x - o.x) <= 0.005 and abs(y - o.y) <= 0.005
    assert len(r.events) == 1
    assert r.events[0].t_s == pytest.approx(kin_obj.contact.t_contact_s, abs=1e-9)


def test_fixture_program_predicts_the_next_bounce(fixture_program):
    gt = ground_truth()  # test-only ground truth record; the program never sees it
    t_hit = math.sqrt(2 * (gt.GT_FLOOR - gt.GT_Y0) / gt.GT_A)
    t_next = t_hit + 2 * gt.GT_SCALE * gt.GT_A * t_hit / gt.GT_A
    assert t_next == pytest.approx(1.4795, abs=1e-4)
    r = sp.simulate(fixture_program, [Fraction(2)])
    assert len(r.events) >= 2
    assert r.events[1].t_s == pytest.approx(t_next, abs=1e-3)
    assert r.events[1].y_px == pytest.approx(gt.GT_FLOOR, abs=0.01)


# ------------------------------------------------------------------ refusals


def with_verdict(o, name, verdict):
    vs = dict(o.verdicts)
    vs[name] = replace(vs[name], verdict=verdict)
    return replace(o, verdicts=vs)


def axes_edit(o, axis, **changes):
    axes = tuple(replace(a, **changes) if a.axis == axis else a for a in o.contact.axes)
    return replace(o, contact=replace(o.contact, axes=axes))


REFUSALS = {
    "impulse not supported": (lambda o: with_verdict(o, "impulse", kf.INCONCLUSIVE), "impulse is"),
    "uniform acceleration not supported": (lambda o: with_verdict(o, "uniform_acceleration", kf.NOT_SUPPORTED),
                                           "uniform_acceleration is"),
    "drag supported": (lambda o: with_verdict(o, "linear_drag", kf.SUPPORTED), "linear drag is supported"),
    "inadequate": (lambda o: replace(o, adequacy="inadequate", adequacy_reasons=("chi2 tail",)), "inadequate"),
    "multiple contacts": (lambda o: replace(o, additional_discontinuity=kf.AdditionalDiscontinuity(1.0, (29, 30), 50.0)),
                          "multiple contacts"),
    "multimodal contact": (lambda o: replace(o, contact=replace(o.contact, profile_multimodal=True)), "multimodal"),
    "contact at search edge": (lambda o: replace(o, contact=replace(o.contact, interval_at_search_boundary=True)),
                               "edge of the searchable range"),
    "reversal on both axes": (lambda o: axes_edit(o, "x", reversal="right_to_left"), "exactly one axis"),
    "no reversal": (lambda o: axes_edit(o, "y", reversal=None), "exactly one axis"),
    "tangential impulse": (lambda o: axes_edit(o, "x", significant=True, delta_v_z=12.0), "not consistent with zero"),
    "restitution above 1": (lambda o: axes_edit(o, "y", v_after_px_s=-1.5 * o.contact.axes[1].v_before_px_s),
                            "outside [0, 1]"),
    "no contact": (lambda o: replace(o, contact=None), "no one-contact"),
    "other contact model": (lambda o: replace(o, contact=replace(o.contact, model=kf.DRAG_IMPULSE)), "comes from"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusals(kin_obj, name):
    edit, fragment = REFUSALS[name]
    with pytest.raises(sp.ProgramUnavailable) as info:
        sp.program_from_kinematics(edit(kin_obj))
    assert any(fragment in r for r in info.value.reasons), info.value.reasons


def test_refusal_lists_every_reason(kin_obj):
    o = with_verdict(with_verdict(kin_obj, "impulse", kf.INCONCLUSIVE), "linear_drag", kf.SUPPORTED)
    with pytest.raises(sp.ProgramUnavailable) as info:
        sp.program_from_kinematics(o)
    assert len(info.value.reasons) >= 2


def make_set(ys, n=61):
    times = frames(n)
    obs = [{"frame_index": i, "t": {"num": t.numerator, "den": t.denominator}, "x": 100 + 40 * float(t), "y": float(y)}
           for i, (t, y) in enumerate(zip(times, ys))]
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [{"object_id": "obj_01", "observations": obs}]})


def test_real_fit_without_contact_is_refused():
    t = np.array([float(x) for x in frames(31)])
    o = kf.analyze_kinematics(make_set(100 + 0.5 * 900 * t * t, 31), position_sigma_px=0.01).objects[0]
    with pytest.raises(sp.ProgramUnavailable) as info:
        sp.program_from_kinematics(o)
    assert any("impulse is" in r for r in info.value.reasons)


def test_real_fit_with_two_contacts_is_refused():
    prog = program(x0=100.0, vx0=40.0, y0=70.0, vy0=0.0, ay=2000.0, planes=(floor(450.0, 0.7),))
    r = sp.simulate(prog, frames(61))
    assert len(r.events) == 2
    o = kf.analyze_kinematics(make_set(r.y_px), position_sigma_px=0.01).objects[0]
    with pytest.raises(sp.ProgramUnavailable) as info:
        sp.program_from_kinematics(o)
    assert any("inadequate" in x or "multiple" in x or "impulse is" in x for x in info.value.reasons)


def test_type_errors():
    with pytest.raises(TypeError):
        sp.program_from_kinematics({"object_id": "obj_01"})
    with pytest.raises(TypeError):
        sp.simulate({"body": None}, [Fraction(0)])


# ------------------------------------------------------------------ isolation


def test_module_imports_no_api_or_interpretation_code():
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    imported, from_decompile = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            if node.module == "decompile":
                from_decompile |= {a.name for a in node.names}
    assert imported == {"__future__", "hashlib", "json", "math", "dataclasses", "fractions", "typing", "decompile",
                        "decompile.trajectories"}, imported
    assert from_decompile == {"kinematic_fit"}
    source = Path(sp.__file__).read_text(encoding="utf-8")
    for forbidden in ("meta_api", "scene_interpretation", "physics_review", "MuseSpark", "GT_", "test_trajectories"):
        assert forbidden not in source, forbidden
