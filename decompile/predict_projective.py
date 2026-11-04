"""P7: frozen prediction with the PROJECTIVE scene program (prediction format version 2).

Same contract as decompile.predict (P3): sees ONLY the training TrajectorySet and the withheld
schedule (frame indices + exact timestamps, no positions); writes a canonical JSON artifact and its
SHA-256 atomically, never overwritten, BEFORE any evaluation. Differences from format 1:
  * the training observations must carry per-frame sigma and apparent diameter (P7 measurement);
  * the scene model is decompile.projective_program built from decompile.projective_fit (perspective-
    aware; camera field-of-view prior), recorded with program_kind "projective_scene_program";
  * the four baselines are EXACTLY the P3 baselines (decompile.predict.BASELINES, same windows); the
    post-contact window uses the projective fit's contact time;
  * the artifact records the projective fit (all models, BIC, verdicts, adequacy) and the predicted
    contact times under the ends of the pre-registered FOV range (sensitivity; never used to choose).
decompile.evaluate verifies format 2 programs with decompile.projective_program.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from decompile import predict as pd
from decompile import projective_fit as pf
from decompile import projective_program as pp
from decompile import trajectories as tr
from decompile.holdout import ScheduledFrame, schedule_to_json, sha256_canonical

PREDICTOR_VERSION = "decompile.predict_projective/1"
PREDICTION_FORMAT_VERSION = 2


def _fit_json(report: pf.ProjectiveReport) -> dict:
    return {
        "method_version": report.method_version, "noise": report.noise, "selected": report.selected,
        "adequacy": report.adequacy, "adequacy_reasons": list(report.adequacy_reasons),
        "verdicts": {k: {kk: vv for kk, vv in v.items()} for k, v in report.verdicts.items()},
        "principal_point": list(report.principal_point),
        "contact_gap_frames": list(report.contact_gap_frames or ()),
        "fits": {m: {"bic": f.bic, "n_params": f.n_params, "n_scalar_obs": f.n_scalar_obs, "chi2": f.chi2,
                     "chi2_dof": f.chi2_dof, "chi2_p": f.chi2_p, "rms_px": f.rms_px, "z_max": f.z_max,
                     "t_contacts_s": list(f.t_contacts_s), "params": f.params, "param_se": f.param_se}
                 for m, f in report.fits.items()},
    }


def _predict_object(training: tr.TrajectorySet, obj: tr.ObjectTrajectory, sched: list[ScheduledFrame],
                    prior: pp.CameraPrior, noise: str) -> dict:
    single = tr.TrajectorySet(training.synthetic, training.source, training.coordinate_system, (obj,))
    try:
        rep = pf.analyze_projective(single, noise=noise)
    except pf.ProjectiveFitError as exc:
        raise pd.PredictionError(f"{obj.object_id}: projective fit unavailable: {exc}") from None
    obs = obj.observations
    t_last = obs[-1].t
    width = training.coordinate_system.image_width
    try:
        program = pp.program_from_fit(rep, width=width, prior=prior, t_last_rel=float(t_last) - rep.t0)
    except pp.ProgramUnavailable as exc:
        raise pd.PredictionError(f"{obj.object_id}: no projective scene program: " + "; ".join(exc.reasons)) from None
    times = [s.t for s in sched]
    sim = pp.simulate(program, times)
    sim_train = pp.simulate(program, [o.t for o in obs])
    train_err = [math.hypot(x - o.x, y - o.y) for o, x, y in zip(obs, sim_train.x_px, sim_train.y_px)]
    if len(obs) < pd.BASELINE_WINDOW:
        raise pd.PredictionError(f"{obj.object_id}: {len(obs)} training observations; the last-{pd.BASELINE_WINDOW} "
                                 "baselines need at least that many")
    t_contact = rep.fits[pf.PROJ1].t_contacts_s[0]
    windows = {
        "constant_velocity_last8": obs[-pd.BASELINE_WINDOW:],
        "constant_acceleration_last8": obs[-pd.BASELINE_WINDOW:],
        "cubic_polynomial_all": obs,
        "post_contact_parabola": tuple(o for o in obs if float(o.t) > t_contact),
    }
    if len(windows["post_contact_parabola"]) < pd.MIN_POST_CONTACT_OBS:
        raise pd.PredictionError(f"{obj.object_id}: {len(windows['post_contact_parabola'])} training observations "
                                 f"after the fitted contact; the post-contact baseline needs {pd.MIN_POST_CONTACT_OBS}")
    predictions = {pd.SCENE_MODEL: {"x": list(sim.x_px), "y": list(sim.y_px)}}
    baselines, training_fit = {}, {pd.SCENE_MODEL: {"window": "all training observations", "n_obs": len(obs),
                                                    "rms_px": pd._rms(train_err), "max_px": max(train_err)}}
    for name, spec in pd.BASELINES.items():
        win = windows[name]
        wt = [o.t for o in win]
        cx = pd._polyfit(wt, [o.x for o in win], spec["degree"], t_last)
        cy = pd._polyfit(wt, [o.y for o in win], spec["degree"], t_last)
        predictions[name] = {"x": pd._polyval(cx, times, t_last), "y": pd._polyval(cy, times, t_last)}
        fx, fy = pd._polyval(cx, wt, t_last), pd._polyval(cy, wt, t_last)
        resid = [math.hypot(a - o.x, b - o.y) for o, a, b in zip(win, fx, fy)]
        baselines[name] = {"method": f"least-squares polynomial of degree {spec['degree']} per axis in tau = t - t_last (s)",
                           "degree": spec["degree"], "window": spec["window"],
                           "window_frames": [win[0].frame_index, win[-1].frame_index], "n_obs": len(win),
                           "t_ref": pd._fj(t_last), "coefficients": {"x": cx, "y": cy}, "models_contacts": False}
        training_fit[name] = {"window": spec["window"], "n_obs": len(win), "rms_px": pd._rms(resid), "max_px": max(resid)}
    events = [{"index": e.index, "t_s": e.t_s, "x_px": e.x_px, "y_px": e.y_px,
               "v_normal_before_diam_s": e.v_normal_before, "v_normal_after_diam_s": e.v_normal_after,
               "phase": "training" if e.t_s <= float(t_last) else "withheld"} for e in sim.events]
    sensitivity = {}
    for h in prior.hfov_range_deg:
        ps = pp.program_from_fit(rep, width=width, prior=prior, t_last_rel=float(t_last) - rep.t0, hfov_deg=h)
        ss = pp.simulate(ps, times)
        sensitivity[f"{h:g}"] = {"withheld_contacts_t_s": [e.t_s for e in ss.events if e.t_s > float(t_last)],
                                 "max_position_change_px": max(math.hypot(a - b, c - d) for a, b, c, d in
                                                               zip(ss.x_px, sim.x_px, ss.y_px, sim.y_px))}
    return {
        "object_id": obj.object_id, "training_last_frame": obs[-1].frame_index, "training_last_t": pd._fj(t_last),
        "n_training_observations": len(obs),
        "projective_fit": _fit_json(rep),
        "program_kind": pp.PROGRAM_KIND, "program": pp.program_to_json(program),
        "program_sha256": pp.program_sha256(program),
        "schedule": [{"frame_index": s.frame_index, "t": pd._fj(s.t)} for s in sched],
        "predictions": predictions, "predicted_events": events, "simulation_termination": sim.termination,
        "camera_prior_sensitivity": {"note": "predicted withheld contacts and largest position change at the ends of "
                                             "the pre-registered field-of-view range (not used for the prediction)",
                                     "hfov_deg": sensitivity},
        "baselines": baselines, "training_fit_consistency": training_fit,
    }


def build_prediction(training: tr.TrajectorySet, schedule: Sequence[ScheduledFrame], *,
                     camera_prior: pp.CameraPrior = pp.CameraPrior(), noise: str = "absolute",
                     experiment_notes: Sequence[str] = ()) -> dict:
    training = pd._validated_training(training)
    per_obj = pd._validated_schedule(schedule, training)
    if not all(isinstance(n, str) for n in experiment_notes):
        raise pd.PredictionError("experiment_notes must be strings")
    objects = [_predict_object(training, training.get(oid), per_obj[oid], camera_prior, noise)
               for oid in training.object_ids if oid in per_obj]
    return {
        "kind": pd.PREDICTION_KIND, "prediction_format_version": PREDICTION_FORMAT_VERSION,
        "predictor_version": PREDICTOR_VERSION,
        "versions": {"projective_fit": pf.METHOD_VERSION, "projective_program": pp.PROGRAM_VERSION},
        "inputs": {"training_sha256": sha256_canonical(training.to_json_dict()),
                   "schedule_sha256": sha256_canonical(schedule_to_json(tuple(schedule)))},
        "kinematics_noise": noise, "camera_prior": camera_prior.to_json_dict(),
        "baseline_specifications": pd.BASELINES, "objects": objects,
        "experiment_notes": list(experiment_notes), "excludes": list(pd.EXCLUDED),
        "note": ("Training-fit consistency is measured on training observations only and is not held-out "
                 "accuracy. Withheld accuracy is computed only by decompile.evaluate from separately held data."),
    }


def predict(training: tr.TrajectorySet, schedule: Sequence[ScheduledFrame], *, output_path: str | Path,
            camera_prior: pp.CameraPrior = pp.CameraPrior(), noise: str = "absolute",
            experiment_notes: Sequence[str] = ()) -> pd.PredictionResult:
    """Build, freeze and write the format-2 artifact and its SHA-256 (never overwritten)."""
    artifact = build_prediction(training, schedule, camera_prior=camera_prior, noise=noise,
                                experiment_notes=experiment_notes)
    data = pd.canonical_bytes(artifact)
    digest = hashlib.sha256(data).hexdigest()
    path = Path(output_path)
    sha_path = pd.sha256_path_for(path)
    if path.exists() or sha_path.exists():
        raise FileExistsError(path if path.exists() else sha_path)
    pd._write_new(path, data)
    try:
        pd._write_new(sha_path, f"{digest}  {path.name}\n".encode("ascii"))
    except BaseException as exc:
        path.unlink()
        if isinstance(exc, Exception):
            raise pd.PredictionWriteError(f"SHA-256 record not written ({type(exc).__name__}: {exc}); the artifact "
                                          "was removed") from exc
        raise
    return pd.PredictionResult(path, sha_path, digest, artifact)
