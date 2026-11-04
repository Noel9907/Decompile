"""P7: perspective-aware fitting (projective_fit), the 3-D scene program (projective_program), format-2
prediction (predict_projective) with evaluation, weighted kinematic fitting and the optional
observation fields. Exact synthetic trajectories come from the independent stress renderer's motion."""

from __future__ import annotations

import ast
import dataclasses
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import evaluate as ev
from decompile import holdout as ho
from decompile import kinematic_fit as kf
from decompile import predict as pd
from decompile import predict_projective as ppred
from decompile import projective_fit as pf
from decompile import projective_program as pp
from decompile import stress_render as sr
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
CAM = sr.Camera()
TRUE_HFOV = math.degrees(2 * math.atan((CAM.width / 2) / CAM.f_px))
SIG, SIG_D = 0.05, 0.05


def exact_traj(motion: sr.Motion, frames: range, *, radius=0.036, fps=30, noise=0.0, seed=1) -> tuple[tr.TrajectorySet, list]:
    segs = sr.contact_times(motion, radius, 3.0)
    g = np.array(motion.g)
    rng = np.random.default_rng(seed)
    obs = []
    for i in frames:
        t = Fraction(i, fps)
        P, _ = sr.state_at(segs, g, float(t))
        x, y, r = sr.project(CAM, P, radius)
        e = rng.normal(0, noise, 3) if noise else np.zeros(3)
        obs.append(tr.Observation(i, t, x + e[0], y + e[1], SIG, SIG, 2 * r + e[2], SIG_D))
    T = tr.TrajectorySet(True, "SYNTHETIC_NUMERICAL_GENERATOR", tr.CoordinateSystem(CAM.width, CAM.height),
                         (tr.ObjectTrajectory("obj_01", tuple(obs)),))
    return T, [s[0] for s in segs[1:]]


PITCHED = sr._thrown(z0=1.5, vz=0.9, pitch_deg=25.0, drop=0.45)
FLAT = sr._thrown()


@pytest.fixture(scope="module")
def pitched():
    return exact_traj(PITCHED, range(0, 19), noise=0.03)


def test_projective_model_is_supported_and_adequate_under_perspective(pitched):
    T, contacts = pitched
    rep = pf.analyze_projective(T)
    assert rep.selected == pf.PROJ1 and rep.adequacy == "adequate"
    assert rep.verdicts["perspective"]["verdict"] == "supported"
    assert rep.verdicts["second_discontinuity"]["verdict"] != "supported"
    assert abs(rep.fits[pf.PROJ1].t_contacts_s[0] - contacts[0]) < 2e-3
    img = rep.fits[pf.IMG1]
    assert img.chi2_p < 1e-3  # raw-pixel uniform acceleration cannot explain perspective motion


def test_no_false_perspective_claim_at_constant_depth():
    T, _ = exact_traj(FLAT, range(0, 20), noise=0.03)
    rep = pf.analyze_projective(T)
    assert rep.verdicts["perspective"]["verdict"] != "supported"
    assert rep.adequacy == "adequate"


def test_relative_noise_is_not_assessable_and_bad_input_is_refused(pitched):
    T, _ = pitched
    assert pf.analyze_projective(T, noise="relative").adequacy in ("not_assessable", "inadequate")
    bare = dataclasses.replace(T, objects=(dataclasses.replace(T.objects[0], observations=tuple(
        dataclasses.replace(o, diameter_px=None, sigma_diameter_px=None) for o in T.objects[0].observations)),))
    with pytest.raises(pf.ProjectiveFitError):
        pf.analyze_projective(bare)
    with pytest.raises(pf.ProjectiveFitError):
        pf.analyze_projective(T, noise="guess")


def test_program_reproduces_training_and_predicts_the_next_contact(pitched):
    T, contacts = pitched
    rep = pf.analyze_projective(T)
    obs = T.objects[0].observations
    prior = pp.CameraPrior(hfov_deg=TRUE_HFOV, hfov_range_deg=(TRUE_HFOV - 5, TRUE_HFOV + 5))
    prog = pp.program_from_fit(rep, width=CAM.width, prior=prior, t_last_rel=float(obs[-1].t - obs[0].t))
    sim = pp.simulate(prog, [o.t for o in obs])
    assert max(math.hypot(x - o.x, y - o.y) for x, y, o in zip(sim.x_px, sim.y_px, obs)) < 0.2
    future = pp.simulate(prog, [Fraction(i, 30) for i in range(19, 28)])
    nxt = [e.t_s for e in future.events if e.t_s > float(obs[-1].t)]
    assert nxt and abs(nxt[0] - contacts[1]) < 0.01
    assert 0 <= prog.restitution <= 1 and abs(np.linalg.norm(prog.normal) - 1) < 1e-12
    again = pp.program_from_json(pp.program_to_json(prog))
    assert pp.program_sha256(again) == pp.program_sha256(prog)
    assert abs(prog.provenance["implied_ball_diameter_m_if_g_9_81"] - 0.072) < 0.004  # gravity recovered


def test_program_refuses_two_contacts_in_training():
    T, contacts = exact_traj(PITCHED, range(0, 27), noise=0.03)  # contacts at frames ~9.3 and ~20.7
    rep = pf.analyze_projective(T)
    obs = T.objects[0].observations
    with pytest.raises(pp.ProgramUnavailable):
        pp.program_from_fit(rep, width=CAM.width, prior=pp.CameraPrior(), t_last_rel=float(obs[-1].t))


def test_format_2_prediction_is_frozen_and_evaluated(tmp_path):
    T, _ = exact_traj(PITCHED, range(0, 28), noise=0.03)
    split = ho.split_trajectories(T, Fraction(19, 28))
    assert split.inference.first_withheld_frame == 19
    path = tmp_path / "prediction.json"
    r = ppred.predict(split.inference.training, split.inference.schedule, output_path=path,
                      camera_prior=pp.CameraPrior(hfov_deg=TRUE_HFOV, hfov_range_deg=(50.0, 80.0)))
    assert r.artifact["prediction_format_version"] == 2
    assert r.artifact["objects"][0]["program_kind"] == pp.PROGRAM_KIND
    vault = tmp_path / "vault.json"
    ev.write_evaluation_vault(vault, split, ground_truth_contacts=None)
    rep = ev.evaluate(path, pd.sha256_path_for(path), vault)
    acc = rep.data["objects"][0]["withheld_accuracy"]
    assert acc["scene_program"]["rmse_px"] < 1.0
    assert min(v["rmse_px"] for k, v in acc.items() if k != "scene_program") > 10 * acc["scene_program"]["rmse_px"]
    with pytest.raises(FileExistsError):
        ppred.predict(split.inference.training, split.inference.schedule, output_path=path)


def test_format_2_program_tampering_is_detected(tmp_path):
    T, _ = exact_traj(PITCHED, range(0, 28), noise=0.03)
    split = ho.split_trajectories(T, Fraction(19, 28))
    art = ppred.build_prediction(split.inference.training, split.inference.schedule)
    art["objects"][0]["program"]["restitution"] *= 0.5
    path = tmp_path / "p.json"
    data = pd.canonical_bytes(art)
    path.write_bytes(data)
    import hashlib
    pd.sha256_path_for(path).write_text(hashlib.sha256(data).hexdigest() + "  p.json\n")
    vault = tmp_path / "vault.json"
    ev.write_evaluation_vault(vault, split, ground_truth_contacts=None)
    with pytest.raises(ev.EvaluationError):
        ev.evaluate(path, pd.sha256_path_for(path), vault)


def test_projective_inference_modules_never_see_withheld_data():
    for name in ("projective_fit", "projective_program", "predict_projective"):
        tree = ast.parse((ROOT / "decompile" / f"{name}.py").read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                for a in n.names}
        for forbidden in ("evaluate", "stress_render", "synth", "render", "overlay", "tracker"):
            assert not {m for m in mods if forbidden in m.split(".")[-1]}, (name, forbidden)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not {"vault", "WithheldVault", "ground_truth"} & names, name


# ------------------------------------------------------------------ weighted kinematic fitting


def test_equal_weights_reproduce_the_unweighted_fit():
    T, _ = exact_traj(FLAT, range(0, 20), noise=0.2, seed=3)
    a = kf.analyze_kinematics(T).objects[0]
    b = kf.analyze_kinematics(T, weights="per_observation_relative").objects[0]
    assert a.selected_model == b.selected_model
    assert abs(a.contact.t_contact_s - b.contact.t_contact_s) < 1e-6
    for m in a.fits:
        assert abs((a.fits[m].bic - a.fits[kf.CV].bic) - (b.fits[m].bic - b.fits[kf.CV].bic)) < 1e-6
    c = kf.analyze_kinematics(T, weights="per_observation").objects[0]
    d = kf.analyze_kinematics(T, position_sigma_px=SIG).objects[0]
    for m in c.fits:
        assert abs(c.fits[m].bic - d.fits[m].bic) <= 1e-5 * abs(d.fits[m].bic)  # contact-time search tolerance
    assert c.adequacy == d.adequacy


def test_weights_need_sigma_on_every_observation():
    T, _ = exact_traj(FLAT, range(0, 20))
    o = T.objects[0].observations
    bare = dataclasses.replace(T, objects=(dataclasses.replace(T.objects[0], observations=(
        dataclasses.replace(o[0], sigma_x=None, sigma_y=None),) + o[1:]),))
    with pytest.raises(kf.KinematicFitError):
        kf.analyze_kinematics(bare, weights="per_observation")
    with pytest.raises(kf.KinematicFitError):
        kf.analyze_kinematics(T, weights="per_observation", position_sigma_px=1.0)
    with pytest.raises(kf.KinematicFitError):
        kf.analyze_kinematics(T, weights="magic")


def test_weights_downweight_an_uncertain_frame():
    T, _ = exact_traj(FLAT, range(0, 20), noise=0.02)
    obs = list(T.objects[0].observations)
    obs[15] = dataclasses.replace(obs[15], y=obs[15].y + 6.0, sigma_y=20.0)  # a bad but honestly flagged frame
    T2 = dataclasses.replace(T, objects=(dataclasses.replace(T.objects[0], observations=tuple(obs)),))
    un = kf.analyze_kinematics(T2).objects[0]
    w = kf.analyze_kinematics(T2, weights="per_observation_relative").objects[0]
    assert un.adequacy == "inadequate"  # the bad frame breaks the unweighted fit
    assert w.adequacy != "inadequate"


# ------------------------------------------------------------------ optional observation fields


def test_optional_observation_fields_round_trip_and_validate():
    plain = tr.Observation(0, Fraction(0), 1.0, 2.0)
    assert plain.to_json_dict() == {"frame_index": 0, "t": {"num": 0, "den": 1}, "x": 1.0, "y": 2.0}
    T, _ = exact_traj(FLAT, range(0, 8))
    assert tr.TrajectorySet.from_json_dict(T.to_json_dict()) == T
    d = T.to_json_dict()
    d["objects"][0]["observations"][0].pop("sigma_y")
    with pytest.raises(tr.TrajectoryError):
        tr.TrajectorySet.from_json_dict(d)
    d = T.to_json_dict()
    d["objects"][0]["observations"][1]["diameter_px"] = -3.0
    with pytest.raises(tr.TrajectoryError):
        tr.TrajectorySet.from_json_dict(d)
    d = T.to_json_dict()
    d["objects"][0]["observations"][2]["sigma_z"] = 1.0
    with pytest.raises(tr.TrajectoryError):
        tr.TrajectorySet.from_json_dict(d)
