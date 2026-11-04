"""P3: predictive decompilation with a frozen prediction artifact.

predict(training, schedule, output_path=...) sees ONLY:
- the training TrajectorySet (positions of the training frames), and
- the prediction schedule (object ids, frame indices and EXACT timestamps of the withheld
  frames; no positions).
It never receives a vault, ground truth, a SyntheticScene or anything derived from withheld
positions, and it does not import the evaluator. Its output is a deterministic canonical
JSON artifact plus a separate SHA-256 file, both written atomically and never overwritten,
BEFORE any evaluation.

Per object:
1. analyze_kinematics(training)  (B1c-1, sigma estimated from the training data)
2. program_from_kinematics(...)   (P1; refuses unsupported fits)
3. simulate(program, schedule times) -> the scene-program prediction and its predicted
   contact events (numerical contact times from the simulator)
4. four baselines, fitted to training observations only, per axis by least squares in
   tau = t - t_last (seconds after the last training observation); windows fixed in advance:
     constant_velocity_last8      degree 1, the last 8 training observations
     constant_acceleration_last8  degree 2, the last 8 training observations
     cubic_polynomial_all         degree 3, all training observations
     post_contact_parabola        degree 2, the training observations strictly after the fitted
                                  contact time (no plane: it cannot predict a further contact)
   No baseline, window or degree is ever chosen from withheld errors.

The artifact also records training-fit consistency (each model against its own training
window). That is NOT held-out accuracy, which only decompile.evaluate can compute.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from decompile import kinematic_fit as kf
from decompile import scene_program as sp
from decompile import trajectories as tr
from decompile.holdout import ScheduledFrame, schedule_to_json, sha256_canonical

PREDICTOR_VERSION = "decompile.predict/1"
PREDICTION_FORMAT_VERSION = 1
PREDICTION_KIND = "prediction_artifact"
SCENE_MODEL = "scene_program"
BASELINE_WINDOW = 8
MIN_POST_CONTACT_OBS = 4  # degree 2 needs 3 coefficients; one more leaves a residual degree of freedom
BASELINES: dict[str, dict[str, Any]] = {
    "constant_velocity_last8": {"degree": 1, "window": "last 8 training observations"},
    "constant_acceleration_last8": {"degree": 2, "window": "last 8 training observations"},
    "cubic_polynomial_all": {"degree": 3, "window": "all training observations"},
    "post_contact_parabola": {"degree": 2,
                              "window": "training observations strictly after the fitted contact time"},
}
EXCLUDED = ("withheld positions", "ground truth", "full-dataset hashes", "evaluation results")


class PredictionError(ValueError):
    """Invalid inputs or an unsupported scene: no prediction was written."""


class PredictionWriteError(OSError):
    """The artifact could not be written completely; no valid frozen prediction exists."""


@dataclass(frozen=True)
class PredictionResult:
    artifact_path: Path
    sha256_path: Path
    sha256: str
    artifact: dict


# ------------------------------------------------------------------ helpers


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _fj(t: Fraction) -> dict:
    return {"num": t.numerator, "den": t.denominator}


def _write_new(path: Path, data: bytes) -> None:
    """Create `path` atomically (temp file, fsync, rename). Never overwrites; no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.name + ".sha256")


def _validated_training(training: Any) -> tr.TrajectorySet:
    if not isinstance(training, tr.TrajectorySet):
        raise PredictionError(f"training must be a TrajectorySet, got {type(training).__name__}")
    try:
        again = tr.TrajectorySet.from_json_dict(training.to_json_dict())
    except (tr.TrajectoryError, TypeError, ValueError) as exc:
        raise PredictionError(f"training data is invalid: {exc}") from None
    if again != training:
        raise PredictionError("training data does not survive validation unchanged")
    return again


def _validated_schedule(schedule: Any, training: tr.TrajectorySet) -> dict[str, list[ScheduledFrame]]:
    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise PredictionError("schedule must be a non-empty sequence of ScheduledFrame")
    clock: dict[int, Fraction] = {o.frame_index: o.t for obj in training.objects for o in obj.observations}
    last = {obj.object_id: obj.observations[-1] for obj in training.objects}
    per_obj: dict[str, list[ScheduledFrame]] = {}
    for s in schedule:
        if not isinstance(s, ScheduledFrame):
            raise PredictionError(f"schedule entries must be ScheduledFrame, got {type(s).__name__}")
        if not isinstance(s.t, Fraction) or isinstance(s.frame_index, bool) or not isinstance(s.frame_index, int):
            raise PredictionError(f"schedule entry {s!r} must have an int frame_index and an exact Fraction t")
        if s.object_id not in last:
            raise PredictionError(f"schedule object {s.object_id!r} is not in the training data")
        lo = last[s.object_id]
        if s.frame_index <= lo.frame_index or s.t <= lo.t:
            raise PredictionError(f"{s.object_id} frame {s.frame_index}: scheduled frames must follow the last "
                                  f"training frame {lo.frame_index} (t = {lo.t})")
        if s.frame_index in clock and clock[s.frame_index] != s.t:
            raise PredictionError(f"frame {s.frame_index}: t = {s.t} disagrees with t = {clock[s.frame_index]}")
        clock[s.frame_index] = s.t
        seq = per_obj.setdefault(s.object_id, [])
        if seq and (s.frame_index <= seq[-1].frame_index or s.t <= seq[-1].t):
            raise PredictionError(f"{s.object_id}: scheduled frames must be strictly increasing (frame {s.frame_index})")
        seq.append(s)
    ordered = sorted(clock.items())
    for (f0, t0), (f1, t1) in zip(ordered, ordered[1:]):
        if t1 <= t0:
            raise PredictionError(f"frames {f0} and {f1}: time does not increase with frame index")
    return per_obj


def _polyfit(times: Sequence[Fraction], values: Sequence[float], degree: int, t_ref: Fraction) -> list[float]:
    tau = np.array([float(t - t_ref) for t in times])
    A = np.vander(tau, degree + 1, increasing=True)
    coef, _, rank, _ = np.linalg.lstsq(A, np.asarray(values, dtype=float), rcond=None)
    if rank < degree + 1:
        raise PredictionError(f"baseline fit of degree {degree} is rank deficient")
    return [float(c) for c in coef]


def _polyval(coef: Sequence[float], times: Sequence[Fraction], t_ref: Fraction) -> list[float]:
    out = []
    for t in times:
        tau = float(t - t_ref)
        out.append(float(sum(c * tau ** k for k, c in enumerate(coef))))
    return out


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


# ------------------------------------------------------------------ prediction


def _predict_object(obj: tr.ObjectTrajectory, kin_obj: kf.ObjectKinematics, sched: list[ScheduledFrame]) -> dict:
    try:
        program = sp.program_from_kinematics(kin_obj)
    except sp.ProgramUnavailable as exc:
        raise PredictionError(f"{obj.object_id}: no scene program: " + "; ".join(exc.reasons)) from None
    obs = obj.observations
    times = [s.t for s in sched]
    t_last = obs[-1].t
    sim = sp.simulate(program, times)
    sim_train = sp.simulate(program, [o.t for o in obs])
    train_err = [math.hypot(x - o.x, y - o.y) for o, x, y in zip(obs, sim_train.x_px, sim_train.y_px)]

    if len(obs) < BASELINE_WINDOW:
        raise PredictionError(f"{obj.object_id}: {len(obs)} training observations; the last-{BASELINE_WINDOW} "
                              "baselines need at least that many")
    t_contact = float(kin_obj.contact.t_contact_s)
    windows = {
        "constant_velocity_last8": obs[-BASELINE_WINDOW:],
        "constant_acceleration_last8": obs[-BASELINE_WINDOW:],
        "cubic_polynomial_all": obs,
        "post_contact_parabola": tuple(o for o in obs if float(o.t) > t_contact),
    }
    if len(windows["post_contact_parabola"]) < MIN_POST_CONTACT_OBS:
        raise PredictionError(f"{obj.object_id}: {len(windows['post_contact_parabola'])} training observations "
                              f"after the fitted contact; the post-contact baseline needs {MIN_POST_CONTACT_OBS}")
    if len(obs) < BASELINES["cubic_polynomial_all"]["degree"] + 2:
        raise PredictionError(f"{obj.object_id}: too few observations for the cubic baseline")

    predictions = {SCENE_MODEL: {"x": list(sim.x_px), "y": list(sim.y_px)}}
    baselines = {}
    training_fit = {SCENE_MODEL: {"window": "all training observations", "n_obs": len(obs),
                                  "rms_px": _rms(train_err), "max_px": max(train_err)}}
    for name, spec in BASELINES.items():
        win = windows[name]
        wt = [o.t for o in win]
        cx = _polyfit(wt, [o.x for o in win], spec["degree"], t_last)
        cy = _polyfit(wt, [o.y for o in win], spec["degree"], t_last)
        predictions[name] = {"x": _polyval(cx, times, t_last), "y": _polyval(cy, times, t_last)}
        fx, fy = _polyval(cx, wt, t_last), _polyval(cy, wt, t_last)
        resid = [math.hypot(a - o.x, b - o.y) for o, a, b in zip(win, fx, fy)]
        baselines[name] = {
            "method": f"least-squares polynomial of degree {spec['degree']} per axis in tau = t - t_last (s)",
            "degree": spec["degree"], "window": spec["window"],
            "window_frames": [win[0].frame_index, win[-1].frame_index], "n_obs": len(win),
            "t_ref": _fj(t_last), "coefficients": {"x": cx, "y": cy},
            "models_contacts": False,
        }
        training_fit[name] = {"window": spec["window"], "n_obs": len(win), "rms_px": _rms(resid), "max_px": max(resid)}

    events = [{
        "index": e.index, "plane_index": e.plane_index, "axis": e.axis, "t_s": e.t_s, "t_rel_s": e.t_rel_s,
        "x_px": e.x_px, "y_px": e.y_px, "v_normal_before_px_s": e.v_normal_before_px_s,
        "v_normal_after_px_s": e.v_normal_after_px_s, "resting": e.resting,
        "phase": "training" if e.t_s <= float(t_last) else "withheld",
    } for e in sim.events]

    return {
        "object_id": obj.object_id,
        "training_last_frame": obs[-1].frame_index, "training_last_t": _fj(t_last),
        "n_training_observations": len(obs),
        "kinematics": {"verdicts": {k: v.verdict for k, v in sorted(kin_obj.verdicts.items())},
                       "adequacy": kin_obj.adequacy, "selected_model": kin_obj.selected_model,
                       "t_contact_s": t_contact, "contact_gap_frames": list(kin_obj.contact.gap_frames)},
        "program": sp.program_to_json(program), "program_sha256": sp.program_sha256(program),
        "schedule": [{"frame_index": s.frame_index, "t": _fj(s.t)} for s in sched],
        "predictions": predictions,
        "predicted_events": events,
        "simulation_termination": sim.termination,
        "baselines": baselines,
        "training_fit_consistency": training_fit,
    }


def build_prediction(training: tr.TrajectorySet, schedule: Sequence[ScheduledFrame], *,
                     experiment_notes: Sequence[str] = ()) -> dict:
    """The artifact content (pure: no file I/O). Raises PredictionError."""
    training = _validated_training(training)
    per_obj = _validated_schedule(schedule, training)
    if not all(isinstance(n, str) for n in experiment_notes):
        raise PredictionError("experiment_notes must be strings")
    kin = kf.analyze_kinematics(training)
    objects = [_predict_object(training.get(oid), kin.get(oid), per_obj[oid])
               for oid in training.object_ids if oid in per_obj]
    return {
        "kind": PREDICTION_KIND,
        "prediction_format_version": PREDICTION_FORMAT_VERSION,
        "predictor_version": PREDICTOR_VERSION,
        "versions": {"kinematic_fit": kf.METHOD_VERSION, "scene_program": sp.PROGRAM_VERSION},
        "inputs": {"training_sha256": sha256_canonical(training.to_json_dict()),
                   "schedule_sha256": sha256_canonical(schedule_to_json(tuple(schedule)))},
        "kinematics_noise": kin.noise.mode,
        "baseline_specifications": BASELINES,
        "objects": objects,
        "experiment_notes": list(experiment_notes),
        "excludes": list(EXCLUDED),
        "note": ("Training-fit consistency is measured on training observations only and is not held-out "
                 "accuracy. Withheld accuracy is computed only by decompile.evaluate from separately held data."),
    }


def predict(training: tr.TrajectorySet, schedule: Sequence[ScheduledFrame], *, output_path: str | Path,
            experiment_notes: Sequence[str] = ()) -> PredictionResult:
    """Build, freeze and write the prediction artifact and its SHA-256 (never overwritten)."""
    artifact = build_prediction(training, schedule, experiment_notes=experiment_notes)
    data = canonical_bytes(artifact)
    digest = hashlib.sha256(data).hexdigest()
    path = Path(output_path)
    sha_path = sha256_path_for(path)
    if path.exists() or sha_path.exists():
        raise FileExistsError(path if path.exists() else sha_path)
    _write_new(path, data)
    try:
        _write_new(sha_path, f"{digest}  {path.name}\n".encode("ascii"))
    except BaseException as exc:
        path.unlink()  # we created it a moment ago: never leave an artifact without its hash record
        if isinstance(exc, Exception):
            raise PredictionWriteError(f"SHA-256 record not written ({type(exc).__name__}: {exc}); the "
                                       "artifact was removed") from exc
        raise
    return PredictionResult(path, sha_path, digest, artifact)


def load_inference_inputs(path: str | Path) -> tuple[tr.TrajectorySet, tuple[ScheduledFrame, ...]]:
    """Read an inference-inputs file (decompile.holdout format: training + schedule; no positions withheld)."""
    from decompile.meta_api import strict_json_loads

    data = strict_json_loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(data, dict) or data.get("kind") != "inference_inputs":
        raise PredictionError("not an inference_inputs file")
    try:
        training = tr.TrajectorySet.from_json_dict(data["training"])
        schedule = tuple(ScheduledFrame(e["object_id"], e["frame_index"], tr.timestamp_from_json(e["t"]))
                         for e in data["schedule"])
    except (KeyError, TypeError, tr.TrajectoryError) as exc:
        raise PredictionError(f"invalid inference inputs: {exc}") from None
    if sha256_canonical(training.to_json_dict()) != data.get("training_sha256"):
        raise PredictionError("training data does not match its recorded training_sha256")
    if sha256_canonical(schedule_to_json(schedule)) != data.get("schedule_sha256"):
        raise PredictionError("schedule does not match its recorded schedule_sha256")
    return training, schedule
