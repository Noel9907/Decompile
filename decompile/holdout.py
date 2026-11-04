"""P2 (A7-lite): deterministic train/withheld split with an evaluation-only vault.

split_trajectories(traj, train_fraction=Fraction(7, 10)) splits on the SHARED FRAME CLOCK:
the sorted distinct frame indices of all objects; the first floor(train_fraction * N) frames
are training, the rest are withheld. Every object is split at the same frame index.

The result separates what inference may see from what only evaluation may see:

    HoldoutSplit.inference  InferenceInputs: the training TrajectorySet, the prediction
                            schedule (withheld frame indices and EXACT timestamps, no
                            positions), the split fraction, and hashes of those two only.
    HoldoutSplit.vault      WithheldVault: withheld positions (evaluation only) and a hash
                            covering them.

Timestamps of withheld frames are allowed for prediction (they are known from the video
container without looking at the image content); withheld positions are not. No hash of
the full data set, and nothing derived from withheld positions, appears in InferenceInputs.
The training TrajectorySet is rebuilt from the training observations alone (its warnings,
e.g. out-of-image positions, are recomputed from training data only).

Pure and deterministic: the input TrajectorySet is never mutated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from decompile import trajectories as tr

HOLDOUT_VERSION = "decompile.holdout/1"
MIN_TRAIN_OBS_PER_OBJECT = tr.MIN_OBSERVATIONS  # a training TrajectorySet needs >= 3 per object
MIN_WITHHELD_FRAMES = 1


class HoldoutError(ValueError):
    """Invalid split request."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScheduledFrame:
    object_id: str
    frame_index: int
    t: Fraction


@dataclass(frozen=True)
class WithheldObservation:
    object_id: str
    frame_index: int
    t: Fraction
    x: float
    y: float


@dataclass(frozen=True)
class InferenceInputs:
    """Everything inference may see. Contains no withheld position."""
    holdout_version: str
    train_fraction: Fraction
    n_frames: int
    n_train_frames: int
    first_withheld_frame: int
    training: tr.TrajectorySet
    schedule: tuple[ScheduledFrame, ...]
    training_sha256: str  # canonical JSON of the training TrajectorySet
    schedule_sha256: str  # canonical JSON of the schedule (frame indices and exact times)


@dataclass(frozen=True)
class WithheldVault:
    """Evaluation only."""
    holdout_version: str
    observations: tuple[WithheldObservation, ...]
    training_sha256: str  # binds the vault to the training data it was split from
    vault_sha256: str


@dataclass(frozen=True)
class HoldoutSplit:
    inference: InferenceInputs
    vault: WithheldVault


def _fraction_json(t: Fraction) -> dict:
    return {"num": t.numerator, "den": t.denominator}


def schedule_to_json(schedule: tuple[ScheduledFrame, ...]) -> list[dict]:
    return [{"object_id": s.object_id, "frame_index": s.frame_index, "t": _fraction_json(s.t)} for s in schedule]


def vault_to_json(vault: WithheldVault) -> dict:
    return {
        "kind": "withheld_vault", "holdout_version": vault.holdout_version, "training_sha256": vault.training_sha256,
        "observations": [{"object_id": o.object_id, "frame_index": o.frame_index, "t": _fraction_json(o.t),
                          "x": o.x, "y": o.y} for o in vault.observations],
    }


def inference_inputs_to_json(inp: InferenceInputs) -> dict:
    return {
        "kind": "inference_inputs", "holdout_version": inp.holdout_version,
        "train_fraction": _fraction_json(inp.train_fraction), "n_frames": inp.n_frames,
        "n_train_frames": inp.n_train_frames, "first_withheld_frame": inp.first_withheld_frame,
        "training": inp.training.to_json_dict(), "schedule": schedule_to_json(inp.schedule),
        "training_sha256": inp.training_sha256, "schedule_sha256": inp.schedule_sha256,
    }


def _check_fraction(f: Any) -> Fraction:
    if isinstance(f, bool) or not isinstance(f, (Fraction, int)):
        raise HoldoutError(f"train_fraction must be an exact Fraction, got {type(f).__name__} {f!r}")
    f = Fraction(f)
    if not 0 < f < 1:
        raise HoldoutError(f"train_fraction must be strictly between 0 and 1, got {f}")
    return f


def split_trajectories(traj: tr.TrajectorySet, train_fraction: Fraction = Fraction(7, 10)) -> HoldoutSplit:
    """Split `traj` on the shared frame clock. Raises HoldoutError on invalid requests."""
    if not isinstance(traj, tr.TrajectorySet):
        raise HoldoutError(f"expected a TrajectorySet, got {type(traj).__name__}")
    f = _check_fraction(train_fraction)
    frames = sorted({o.frame_index for obj in traj.objects for o in obj.observations})
    n = len(frames)
    n_train = math.floor(f * n)  # exact rational arithmetic
    if n - n_train < MIN_WITHHELD_FRAMES:
        raise HoldoutError(f"{n} frames x {f} leaves no withheld frame")
    if n_train < 1:
        raise HoldoutError(f"{n} frames x {f} leaves no training frame")
    cutoff = frames[n_train]  # first withheld frame index
    data = traj.to_json_dict()  # a fresh dict; the input object is untouched
    train_objects, schedule, withheld = [], [], []
    for obj, obj_json in zip(traj.objects, data["objects"]):
        keep = [ob for ob in obj_json["observations"] if ob["frame_index"] < cutoff]
        if len(keep) < MIN_TRAIN_OBS_PER_OBJECT:
            raise HoldoutError(f"{obj.object_id}: {len(keep)} training observation(s); need at least "
                               f"{MIN_TRAIN_OBS_PER_OBJECT}")
        train_objects.append({"object_id": obj.object_id, "observations": keep})
        for o in obj.observations:
            if o.frame_index >= cutoff:
                schedule.append(ScheduledFrame(obj.object_id, o.frame_index, o.t))
                withheld.append(WithheldObservation(obj.object_id, o.frame_index, o.t, o.x, o.y))
    training = tr.TrajectorySet.from_json_dict({
        "schema": data["schema"], "synthetic": data["synthetic"], "source": data["source"],
        "coordinate_system": data["coordinate_system"], "objects": train_objects,
    })
    training_sha = sha256_canonical(training.to_json_dict())
    sched = tuple(schedule)
    inference = InferenceInputs(HOLDOUT_VERSION, f, n, n_train, cutoff, training, sched, training_sha,
                                sha256_canonical(schedule_to_json(sched)))
    vault_wo_hash = WithheldVault(HOLDOUT_VERSION, tuple(withheld), training_sha, "")
    vault = WithheldVault(HOLDOUT_VERSION, tuple(withheld), training_sha, sha256_canonical(vault_to_json(vault_wo_hash)))
    return HoldoutSplit(inference, vault)
