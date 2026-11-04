"""Tests for decompile.kinematic_fit (B1c-1).

OFFLINE, deterministic. Every trajectory here is SYNTHETIC: either generated in this file
from closed-form motion (independent of the fitter's design matrices) or the existing
synthetic bounce fixture. The fixture's ground truth is read from tests/test_trajectories.py
by the recovery test only; the fitter never receives it.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import kinematic_fit as kf
from decompile import trajectories as tr

TESTS = Path(__file__).parent
BOUNCE = TESTS / "fixtures" / "synthetic_bounce_single.trajectories.json"
FPS30 = [Fraction(i, 30) for i in range(31)]


# ------------------------------------------------------------------ helpers


def make_set(times, xs, ys, object_id="obj_01") -> tr.TrajectorySet:
    obs = [{"frame_index": i, "t": {"num": t.numerator, "den": t.denominator}, "x": float(x), "y": float(y)}
           for i, (t, x, y) in enumerate(zip(times, xs, ys, strict=True))]
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [{"object_id": object_id, "observations": obs}],
    })


def secs(times) -> np.ndarray:
    return np.array([float(t) for t in times])


def ballistic(t, q0, v0, a, contacts=()):
    """Closed-form uniform acceleration with instantaneous velocity jumps (written per segment)."""
    out = []
    for ti in t:
        q, v, t_prev = q0, v0, 0.0
        for tc, dv in contacts:
            if ti <= tc:
                break
            q, v, t_prev = q + v * (tc - t_prev) + 0.5 * a * (tc - t_prev) ** 2, v + a * (tc - t_prev) + dv, tc
        d = ti - t_prev
        out.append(q + v * d + 0.5 * a * d * d)
    return np.array(out)


def drag_exact(t, q0, v0, g, k):
    """Exact solution of q'' = g - k q' (independent closed form)."""
    vt = g / k
    return q0 + vt * t + (v0 - vt) * (1 - np.exp(-k * t)) / k


def run(s, sigma=None) -> kf.ObjectKinematics:
    return kf.analyze_kinematics(s, position_sigma_px=sigma).objects[0]


def verdict(o: kf.ObjectKinematics, name: str) -> str:
    return o.verdicts[name].verdict


def load_ground_truth():
    """The fixture's test-only ground truth record (tests/test_trajectories.py)."""
    spec = importlib.util.spec_from_file_location("_trajectory_ground_truth", TESTS / "test_trajectories.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the existing fixture


@pytest.fixture(scope="module")
def fixture_traj() -> tr.TrajectorySet:
    return tr.load_trajectories(BOUNCE)


@pytest.fixture(scope="module")
def fixture_estimated(fixture_traj) -> kf.ObjectKinematics:
    return run(fixture_traj)


@pytest.fixture(scope="module")
def fixture_supplied(fixture_traj) -> kf.ObjectKinematics:
    return run(fixture_traj, 0.01 / math.sqrt(12))  # uniform rounding to 0.01 px


@pytest.mark.parametrize("mode", ["fixture_estimated", "fixture_supplied"])
def test_fixture_recovers_test_only_ground_truth(request, mode):
    o = request.getfixturevalue(mode)
    gt = load_ground_truth()
    t_hit = math.sqrt(2 * (gt.GT_FLOOR - gt.GT_Y0) / gt.GT_A)
    v_hit = gt.GT_A * t_hit
    t_apex = t_hit + gt.GT_SCALE * v_hit / gt.GT_A
    fit = o.fits[kf.IMPULSE]
    assert abs(fit.params["y.a"] - gt.GT_A) <= 0.05
    assert abs(o.contact.t_contact_s - t_hit) <= 1e-4
    y = next(a for a in o.contact.axes if a.axis == "y")
    assert abs(y.speed_ratio - gt.GT_SCALE) <= 1e-3
    assert abs(fit.params["x.v0"] - gt.GT_VX) <= 1e-6
    (apex,) = [tp for tp in o.turning_points if tp.axis == "y"]
    assert abs(apex.t_s - t_apex) <= 1e-3
    # The approximate profile interval contains the true contact time and is not at a search edge.
    lo, hi = o.contact.profile_interval_s
    assert lo <= o.contact.t_contact_s <= hi
    assert not o.contact.interval_at_search_boundary


@pytest.mark.parametrize("mode", ["fixture_estimated", "fixture_supplied"])
def test_fixture_verdicts(request, mode):
    o = request.getfixturevalue(mode)
    assert o.selected_model == kf.IMPULSE
    assert verdict(o, "constant_velocity") == kf.CONTRADICTED
    assert verdict(o, "uniform_acceleration") == kf.SUPPORTED
    assert verdict(o, "impulse") == kf.SUPPORTED
    assert verdict(o, "acceleration_continuity") == kf.SUPPORTED
    assert verdict(o, "linear_drag") == kf.NOT_SUPPORTED
    assert verdict(o, "surface_identity") == kf.NOT_EVALUATED
    assert verdict(o, "quadratic_drag") == kf.NOT_EVALUATED
    assert o.contact.gap_frames == (18, 19)
    x, y = o.contact.axes
    assert y.reversal == "down_to_up" and y.significant
    assert x.reversal is None and not x.significant  # no horizontal impulse
    # A smooth change would need ~31x the uniform acceleration inside the 1/30 s gap.
    assert y.ratio_to_uniform_accel > 30
    assert o.drag.k_per_s >= 0 and o.drag.k_upper_times_span < kf.DRAG_MAX_UNDETECTED_KT
    ua = o.verdicts["uniform_acceleration"].evidence
    assert 89.9 < ua["direction_deg_image"] < 90.1  # toward +y: down the image
    assert o.additional_discontinuity is None and o.outliers == ()


def test_fixture_apex_is_highest_image_point_not_an_event(fixture_estimated):
    (apex,) = fixture_estimated.turning_points
    assert apex.kind == "y_minimum"
    assert "highest point in the image" in apex.image_description
    assert apex.between_frames == (31, 32)
    assert "not an event" in apex.note
    assert apex.t_s != fixture_estimated.contact.t_contact_s


def test_fixture_adequacy_by_noise_mode(fixture_estimated, fixture_supplied):
    assert fixture_estimated.adequacy == "not_assessable"
    assert any("cannot be calibrated" in r for r in fixture_estimated.adequacy_reasons)
    assert fixture_supplied.adequacy == "adequate"
    assert fixture_supplied.fits[kf.IMPULSE].chi2_p_value > kf.ADEQUACY_P_MIN


def test_fixture_finite_differences_use_exact_timestamps(fixture_traj, fixture_estimated):
    fds = fixture_estimated.finite_differences
    obs = fixture_traj.objects[0].observations
    assert len(fds) == len(obs) - 1
    for fd, a, b in zip(fds, obs, obs[1:]):
        assert fd.dt_s == b.t - a.t and isinstance(fd.dt_s, Fraction)
        assert fd.vy_px_s == pytest.approx((b.y - a.y) / float(b.t - a.t), rel=1e-15)
    # Before the contact, second differences are ~2000 px/s^2; the two stencils that straddle
    # the contact show the impulse, not an acceleration.
    sds = {s.frame: s.ay_px_s2 for s in fixture_estimated.second_differences}
    assert all(abs(sds[f] - 2000) < 10 for f in range(1, 18))
    assert sds[18] < -20000 and sds[19] < -20000


# ------------------------------------------------------------------ exact recovery


def test_exact_constant_velocity():
    t = secs(FPS30)
    o = run(make_set(FPS30, 50 + 40 * t, 300 - 25 * t))
    fit = o.fits[kf.CV]
    assert fit.params["x.v0"] == pytest.approx(40, abs=1e-9)
    assert fit.params["y.v0"] == pytest.approx(-25, abs=1e-9)
    assert o.selected_model == kf.CV
    assert verdict(o, "constant_velocity") == kf.SUPPORTED
    assert verdict(o, "uniform_acceleration") == kf.NOT_SUPPORTED
    assert verdict(o, "impulse") == kf.NOT_SUPPORTED
    assert verdict(o, "stationary") == kf.CONTRADICTED
    assert fit.floored_axes == ("x", "y")  # exact data: the numerical floor is reported


def test_exact_constant_acceleration():
    t = secs(FPS30)
    o = run(make_set(FPS30, ballistic(t, 10, 30, -20), ballistic(t, 50, 10, 900)))
    fit = o.fits[kf.CA]
    assert fit.params["x.a"] == pytest.approx(-20, abs=1e-6)
    assert fit.params["y.a"] == pytest.approx(900, abs=1e-6)
    assert fit.params["y.v0"] == pytest.approx(10, abs=1e-7)
    assert o.selected_model == kf.CA
    assert verdict(o, "uniform_acceleration") == kf.SUPPORTED
    assert verdict(o, "constant_velocity") == kf.CONTRADICTED
    assert verdict(o, "impulse") == kf.NOT_SUPPORTED
    assert verdict(o, "linear_drag") == kf.NOT_SUPPORTED


def test_single_contact_bounce_recovery():
    t = secs(FPS30)
    tc, a = 0.4123, 1500.0
    v_before = a * tc
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 60, 0, a, [(tc, -1.6 * v_before)])), 0.01)
    assert verdict(o, "impulse") == kf.SUPPORTED
    assert abs(o.contact.t_contact_s - tc) < 1e-6
    y = o.contact.axes[1]
    assert y.v_before_px_s == pytest.approx(v_before, rel=1e-6)
    assert y.delta_v_px_s == pytest.approx(-1.6 * v_before, rel=1e-6)
    assert y.speed_ratio == pytest.approx(0.6, abs=1e-6)
    assert o.fits[kf.IMPULSE].params["y.a"] == pytest.approx(a, rel=1e-7)


def test_irregular_fraction_timestamps():
    steps = [Fraction(1, 30), Fraction(1, 60), Fraction(1, 24), Fraction(1, 50), Fraction(1, 25)]
    times = [Fraction(0)]
    for i in range(29):
        times.append(times[-1] + steps[i % len(steps)])
    xq = lambda tt: 7 + 33 * tt - Fraction(9, 2) * tt * tt  # noqa: E731  exact rational motion
    yq = lambda tt: 40 + 5 * tt + 400 * tt * tt  # noqa: E731
    o = run(make_set(times, [float(xq(tt)) for tt in times], [float(yq(tt)) for tt in times]))
    for fd, a, b in zip(o.finite_differences, times, times[1:]):
        assert fd.dt_s == b - a
        assert fd.vx_px_s == pytest.approx(float((xq(b) - xq(a)) / (b - a)), rel=1e-9)
        assert fd.vy_px_s == pytest.approx(float((yq(b) - yq(a)) / (b - a)), rel=1e-9)
    fit = o.fits[kf.CA]
    assert fit.params["x.a"] == pytest.approx(-9, abs=1e-6)
    assert fit.params["y.a"] == pytest.approx(800, abs=1e-6)
    assert verdict(o, "uniform_acceleration") == kf.SUPPORTED


def test_irregular_timestamps_bounce():
    steps = [Fraction(1, 30), Fraction(1, 45), Fraction(1, 20)]
    times = [Fraction(0)]
    for i in range(35):
        times.append(times[-1] + steps[i % 3])
    t = secs(times)
    o = run(make_set(times, 100 + 50 * t, ballistic(t, 80, 0, 1800, [(0.55, -1.75 * 1800 * 0.55)])), 0.01)
    assert verdict(o, "impulse") == kf.SUPPORTED
    assert abs(o.contact.t_contact_s - 0.55) < 1e-6


def test_linear_drag_recovery():
    t = secs(FPS30)
    k = 0.8
    xs = drag_exact(t, 100, 300, 0.0001, k)  # tiny drive on x keeps the closed form non-singular
    ys = drag_exact(t, 50, -200, 1500, k)
    o = run(make_set(FPS30, xs, ys))
    assert verdict(o, "linear_drag") == kf.SUPPORTED
    assert o.drag.k_per_s == pytest.approx(k, rel=1e-6)
    assert o.selected_model == kf.DRAG
    assert o.fits[kf.DRAG].params["y.a"] == pytest.approx(1500, rel=1e-6)
    # Regression: smooth drag curvature must not be mistaken for a velocity discontinuity
    # (impulse models are compared with the smooth drag model, not only with drag-free ones).
    assert verdict(o, "impulse") == kf.NOT_SUPPORTED
    assert o.additional_discontinuity is None and o.adequacy != "inadequate"
    assert o.turning_points == ()  # not computed when drag is supported
    assert "not uniform" in o.verdicts["uniform_acceleration"].reason


def test_drag_rate_is_never_negative():
    t = secs(FPS30)
    c = 0.5  # speed GROWS exponentially: "negative drag", physically excluded
    o = run(make_set(FPS30, 100 + 200 * (np.exp(c * t) - 1) / c, 200 + 0 * t))
    assert o.drag.k_per_s >= 0
    assert verdict(o, "linear_drag") != kf.SUPPORTED


# ------------------------------------------------------------------ reversals without surface identity


def test_floor_like_reversal_reports_image_direction_only():
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 100, 0, 1000, [(0.5, -800)])), 0.01)
    y = o.contact.axes[1]
    assert y.reversal == "down_to_up"
    assert verdict(o, "surface_identity") == kf.NOT_EVALUATED
    assert o.verdicts["surface_identity"].evidence["reversal"] == {"x": None, "y": "down_to_up"}


def test_ceiling_like_reversal():
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 400, -900, 1000, [(0.3, 1000)])), 0.01)
    assert verdict(o, "impulse") == kf.SUPPORTED
    assert o.contact.axes[1].reversal == "up_to_down"
    assert verdict(o, "surface_identity") == kf.NOT_EVALUATED


def test_wall_like_x_impulse():
    t = secs(FPS30)
    o = run(make_set(FPS30, ballistic(t, 100, 200, 0, [(0.5, -350)]), ballistic(t, 100, -300, 1000)), 0.01)
    assert verdict(o, "impulse") == kf.SUPPORTED
    x, y = o.contact.axes
    assert x.reversal == "right_to_left" and x.significant
    assert not y.significant  # the vertical motion is unaffected
    assert verdict(o, "acceleration_continuity") == kf.SUPPORTED


def test_no_surface_or_cause_words_in_labels(fixture_estimated):
    allowed_reversals = {None, "down_to_up", "up_to_down", "left_to_right", "right_to_left"}
    assert {a.reversal for a in fixture_estimated.contact.axes} <= allowed_reversals
    for tp in fixture_estimated.turning_points:
        assert tp.kind in {"x_minimum", "x_maximum", "y_minimum", "y_maximum"}
    names = set(fixture_estimated.verdicts)
    assert not names & {"gravity", "floor", "ceiling", "wall", "bounce", "collision"}
    assert "gravity" not in fixture_estimated.verdicts["uniform_acceleration"].reason.split("(e.g.")[0]


# ------------------------------------------------------------------ stationary, apex


def test_stationary_object():
    t = secs(FPS30)
    o = run(make_set(FPS30, 320 + 0 * t, 240 + 0 * t))
    assert verdict(o, "stationary") == kf.SUPPORTED
    assert verdict(o, "constant_velocity") == kf.SUPPORTED
    assert verdict(o, "uniform_acceleration") == kf.NOT_SUPPORTED
    assert verdict(o, "linear_drag") == kf.INCONCLUSIVE  # drag cannot act on an object at rest
    assert "not identifiable" in o.verdicts["linear_drag"].reason


def test_apex_without_contact_is_a_turning_point_not_an_impulse():
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 400, -600, 1000)), 0.01)
    assert verdict(o, "impulse") == kf.NOT_SUPPORTED
    (apex,) = o.turning_points
    assert apex.kind == "y_minimum" and apex.t_s == pytest.approx(0.6, abs=1e-6)
    assert apex.value_px == pytest.approx(400 - 600 ** 2 / 2000, abs=1e-6)


def test_release_from_rest_is_not_a_turning_point():
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 0 * t, ballistic(t, 50, 0, 1000)), 0.01)
    assert o.turning_points == ()


# ------------------------------------------------------------------ noise and inconclusive


def test_constant_velocity_with_noise():
    rng = np.random.default_rng(12345)
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t + rng.normal(0, .5, 31), 200 - 25 * t + rng.normal(0, .5, 31)), 0.5)
    assert o.adequacy == "adequate"
    assert verdict(o, "constant_velocity") == kf.SUPPORTED
    assert verdict(o, "uniform_acceleration") != kf.SUPPORTED
    assert verdict(o, "impulse") != kf.SUPPORTED


def test_weak_acceleration_in_noise_is_inconclusive():
    # 8 px/s^2 over 1 s under 0.5 px noise: Delta BIC lands between -2 and 10 for this seed.
    rng = np.random.default_rng(1)
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t + rng.normal(0, .5, 31), 200 - 25 * t + 4 * t * t + rng.normal(0, .5, 31)), 0.5)
    v = o.verdicts["uniform_acceleration"]
    assert v.verdict == kf.INCONCLUSIVE
    assert -kf.BIC_POSITIVE <= v.evidence["delta_bic"] <= kf.BIC_DECISIVE
    assert "between" in v.reason


def test_weak_bounce_in_noise_is_not_claimed():
    rng = np.random.default_rng(8)
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t + rng.normal(0, .5, 31),
                     ballistic(t, 100, 0, 800, [(0.5, -15)]) + rng.normal(0, .5, 31)), 0.5)
    assert verdict(o, "impulse") != kf.SUPPORTED
    assert verdict(o, "uniform_acceleration") == kf.SUPPORTED


# ------------------------------------------------------------------ insufficient data and edges


def test_three_observations_are_insufficient():
    o = run(make_set(FPS30[:3], [1, 2, 3], [5, 5, 5]))
    assert kf.CA not in o.fits  # 3 coefficients per axis leave no residual degrees of freedom
    for name in ("constant_velocity", "uniform_acceleration", "impulse", "stationary"):
        assert verdict(o, name) == kf.INCONCLUSIVE, name
    assert "insufficient observations" in o.verdicts["impulse"].reason


def test_five_observations_cannot_host_a_contact():
    t = secs(FPS30[:5])
    o = run(make_set(FPS30[:5], 100 + 40 * t, ballistic(t, 100, 0, 900)))
    assert kf.IMPULSE not in o.fits
    assert verdict(o, "impulse") == kf.INCONCLUSIVE
    assert verdict(o, "uniform_acceleration") == kf.SUPPORTED


@pytest.mark.parametrize("sigma", [None, 0.01])
def test_contact_too_close_to_the_start(sigma):
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 300, 600, 1000, [(0.05, -1100)])), sigma)
    assert verdict(o, "impulse") == kf.INCONCLUSIVE
    assert "edge of the searchable range" in o.verdicts["impulse"].reason
    assert o.contact.interval_at_search_boundary


def test_contact_too_close_to_the_end():
    t = secs(FPS30)
    o = run(make_set(FPS30, 100 + 40 * t, ballistic(t, 100, 0, 1000, [(0.96, -1700)])), 0.01)
    assert verdict(o, "impulse") == kf.INCONCLUSIVE


def test_multiple_contacts_flag_model_inadequacy():
    times = [Fraction(i, 30) for i in range(61)]
    t = secs(times)
    o = run(make_set(times, 100 + 40 * t, ballistic(t, 70, 0, 2000, [(0.6164, -2096), (1.5, -1500)])), 0.01)
    assert o.adequacy == "inadequate"
    assert o.additional_discontinuity is not None
    assert any("further velocity discontinuity" in r for r in o.adequacy_reasons)
    v = o.verdicts["impulse"]
    assert v.verdict == kf.INCONCLUSIVE and v.provisional == kf.SUPPORTED
    assert o.turning_points == ()  # nothing derived from an inadequate fit
    for v in o.verdicts.values():
        assert v.verdict in (kf.INCONCLUSIVE, kf.NOT_EVALUATED), v


@pytest.mark.parametrize("sigma", [0.2, None])
def test_outlier_is_flagged_and_blocks_conclusions(sigma):
    rng = np.random.default_rng(3)
    t = secs(FPS30)
    ys = ballistic(t, 100, -300, 1200) + rng.normal(0, .2, 31)
    ys[12] += 25.0
    o = run(make_set(FPS30, 100 + 40 * t + rng.normal(0, .2, 31), ys), sigma)
    assert ("y", 12) in {(a, f) for a, f, _ in o.outliers}
    assert o.adequacy == "inadequate"
    for v in o.verdicts.values():
        assert v.verdict in (kf.INCONCLUSIVE, kf.NOT_EVALUATED), v


# ------------------------------------------------------------------ statistics and bookkeeping


def test_parameter_and_observation_counts():
    t = secs(FPS30)
    s = make_set(FPS30, 100 + 40 * t, ballistic(t, 100, 0, 1000, [(0.5, -900)]))
    for sigma, n_noise in ((0.01, 0), (None, 2)):
        o = run(s, sigma)
        cv, ca, imp = o.fits[kf.CV], o.fits[kf.CA], o.fits[kf.IMPULSE]
        assert cv.n_scalar_obs == 62
        assert (cv.n_coefficients, cv.n_nonlinear, cv.n_params) == (4, 0, 4 + n_noise)
        assert (ca.n_coefficients, ca.n_params) == (6, 6 + n_noise)
        assert (imp.n_coefficients, imp.n_nonlinear, imp.n_params) == (8, 1, 9 + n_noise)
        for f in (cv, ca, imp):
            assert f.bic == pytest.approx(f.neg2_log_likelihood + f.n_params * math.log(62), rel=1e-12)


def test_per_axis_sigma_enters_chi_square_separately():
    rng = np.random.default_rng(5)
    t = secs(FPS30)
    s = make_set(FPS30, 100 + 40 * t + rng.normal(0, .1, 31), ballistic(t, 50, 0, 900) + rng.normal(0, 2, 31))
    o = run(s, (0.1, 2.0))
    ca = o.fits[kf.CA]
    assert ca.chi2 == pytest.approx(ca.rss_px2[0] / 0.1 ** 2 + ca.rss_px2[1] / 2.0 ** 2, rel=1e-12)
    assert ca.chi2_dof == 62 - 6
    assert o.adequacy == "adequate"
    common = run(s, 0.1)  # the wrong (common) sigma makes y look inadequate
    assert common.adequacy == "inadequate"


def test_estimated_mode_discloses_uncalibrated_adequacy():
    rng = np.random.default_rng(6)
    t = secs(FPS30)
    report = kf.analyze_kinematics(make_set(FPS30, 100 + 40 * t + rng.normal(0, .3, 31),
                                            200 + 10 * t + rng.normal(0, .3, 31)))
    assert report.noise.mode == "estimated" and report.noise.sigma_px is None
    o = report.objects[0]
    assert o.adequacy == "not_assessable"
    assert all(f.chi2 is None and f.chi2_p_value is None for f in o.fits.values())
    assert o.fits[kf.CV].floored_axes == ()  # real noise: the numerical floor is not used


def test_every_verdict_has_reason_and_numbers(fixture_estimated):
    for o in (fixture_estimated, run(make_set(FPS30[:3], [1, 2, 3], [5, 5, 5]))):
        for v in o.verdicts.values():
            assert v.verdict in (kf.SUPPORTED, kf.CONTRADICTED, kf.NOT_SUPPORTED, kf.INCONCLUSIVE, kf.NOT_EVALUATED)
            assert v.reason and v.evidence, v


def test_profile_interval_is_labelled_approximate(fixture_estimated):
    assert "approximate" in fixture_estimated.contact.note
    assert "Surface identity is not inferred" in fixture_estimated.contact.note


def test_report_metadata(fixture_traj):
    r = kf.analyze_kinematics(fixture_traj)
    assert r.method_version == kf.METHOD_VERSION
    assert r.thresholds["BIC_DECISIVE"] == 10.0 and r.thresholds["OUTLIER_Z"] == 5.0
    assert any("metres" in lim for lim in r.limitations)
    assert r.get("obj_01").object_id == "obj_01"
    with pytest.raises(KeyError):
        r.get("obj_99")


def test_multiple_objects_are_analysed_independently():
    t = secs(FPS30)
    data = make_set(FPS30, 100 + 40 * t, 300 - 25 * t).to_json_dict()
    other = make_set(FPS30, 100 + 0 * t, ballistic(t, 50, 0, 1000), object_id="obj_02").to_json_dict()
    data["objects"].append(other["objects"][0])
    r = kf.analyze_kinematics(tr.TrajectorySet.from_json_dict(data))
    assert [o.object_id for o in r.objects] == ["obj_01", "obj_02"]
    assert verdict(r.objects[0], "constant_velocity") == kf.SUPPORTED
    assert verdict(r.objects[1], "uniform_acceleration") == kf.SUPPORTED


def test_deterministic_repeated_runs(fixture_traj):
    a = kf.analyze_kinematics(fixture_traj)
    b = kf.analyze_kinematics(fixture_traj)
    assert a == b
    assert a.objects[0].fits[kf.IMPULSE].residuals_y == b.objects[0].fits[kf.IMPULSE].residuals_y


# ------------------------------------------------------------------ input validation and isolation


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "1", (1.0,), (1.0, 0.0), (1, 2, 3)])
def test_invalid_sigma_rejected(fixture_traj, bad):
    with pytest.raises(kf.KinematicFitError):
        kf.analyze_kinematics(fixture_traj, position_sigma_px=bad)


def test_non_trajectory_set_rejected(fixture_traj):
    with pytest.raises(kf.KinematicFitError):
        kf.analyze_kinematics(fixture_traj.to_json_dict())


def test_module_uses_trajectory_data_only():
    """No dependency on interpretations, prompts, the API client or test ground truth."""
    tree = ast.parse(Path(kf.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"__future__", "math", "dataclasses", "fractions", "typing", "numpy", "scipy.optimize",
                        "scipy.stats", "decompile.trajectories"}, imported
    source = Path(kf.__file__).read_text(encoding="utf-8")
    for forbidden in ("GT_", "ground_truth", "test_trajectories", "SceneInterpretation", "MuseSpark"):
        assert forbidden not in source, forbidden
