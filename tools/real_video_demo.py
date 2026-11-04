"""P6: pre-registered VIDEO-IN predictive decompilation on a real (or stand-in) video. OFFLINE by default.

Three subcommands, run in this order:

  prereg   Hash the input video and write a pre-registration file (never overwritten) BEFORE any
           tracking: clip window and stride, first-frame seed box, ROI / exclusion boxes, train
           fraction, training-gap policy, tracking-gate thresholds, Muse Spark mode, disclosures.
  run      Verify the pre-registration and the video hash, then:
             measure  A1 probe/validate -> A2 frame export -> clip frames -> colour model from the
                      FIRST clip frame (a training frame) and the seed box -> per-frame classical
                      tracking of the clip frames -> tracking gate
             split    the pre-registered split of the CLIP frame list; the tracked set is split at the
                      same first withheld frame (exact fraction over tracked frames) ->
                      inference_inputs.json | evaluation_only/evaluation_vault.json
             spark    OPTIONAL advisory Muse Spark stage on the TRAINING trajectory only ->
                      inference/spark/spark_interpretation.json (separate artifact)
             predict  decompile.predict from inference_inputs.json alone -> frozen prediction
           and STOP. Nothing is evaluated by `run`.
  reveal   Evaluate the frozen prediction exactly once, draw the overlay and write the report.

Gates fixed by the pre-registration (never adjusted after looking at results):
  - every WITHHELD clip frame must be detected (else the run stops before freezing);
  - at most max_training_gaps TRAINING clip frames may fail detection; they are recorded as gaps,
    never interpolated, and simply absent from the training observations;
  - mask area within max_area_rel_dev of the median and fill ratio within fill_ratio, for every
    detected frame;
  - the training fit must yield a scene program (one contact in training with >= 4 later training
    observations, uniform acceleration, ...); otherwise the run stops before freezing.

AI / prediction independence: prediction.json depends only on the training observations, the
withheld timestamps and the pre-registered configuration. It is byte-identical whether the Muse
Spark stage is off, succeeds, is rejected or is unavailable. The Spark result lives only in
spark_interpretation.json; run_manifest.json links both by hash and records the Spark status.

Live Muse Spark calls: only when the pre-registration says spark.mode == "live" AND `run` is given
--confirm-live-call; one request, single attempt. Otherwise no request is made.

Evaluation claims (reveal): withheld "truth" is the classical tracker's measurement of the
withheld frames, NOT independently verified ground truth. Tracking accuracy is reported only
against independent manual annotations (--annotations); otherwise "not available". Contact timing
is "not_evaluated" (no independent event annotations). The number of downward-to-upward reversals
in the withheld tracker measurements is reported as a tracker-measured check only.

P7 options (pre-registration version 2; version-1 files and runs behave exactly as before):
  measurement.centre_method  "edge_ellipse" etc.: the colour detection is only the coarse stage; the centre,
                             its per-frame 1-sigma uncertainty and the apparent diameter come from the ball
                             outline (decompile.ball_measure). Unmeasurable outlines -> "edge_fit_rejected".
  physics.model              "projective": perspective-aware fit of (x, y, D) (decompile.projective_fit) and a
                             3-D scene program (decompile.projective_program) with a pre-registered camera
                             field-of-view prior; frozen with decompile.predict_projective (format 2).
  gates.max_withheld_gaps    withheld frames whose measurement failed may be absent from the evaluation
                             (at most this many; default 0 = every withheld frame must be measured).

SAM 3.1 backend (pre-registration version 3; versions 1 and 2 behave exactly as before):
  prereg-sam                 pins the video, the SAM prompt (ONE concept noun phrase, or box/point object prompts),
                             the object-selection rule, the outline measurement parameters, the gates and the physics.
  run --sam-result FILE      the SAM tracking result (decompile.sam_tracking JSON, produced by tools/sam_demo.py track
                             from THIS video and THIS prompt; verified) replaces the colour tracker: SAM gives identity
                             and the support region, the centre/diameter/sigma come from the outline in the original
                             pixels (decompile.sam_measure). Frames that are not "measured" are gaps, never interpolated.
                             This runner never calls SAM itself: the only live SAM call is `sam_demo.py track
                             --confirm-live-call`.

Exit codes: 0 done; 1 a gate/verification failed (nothing frozen, or nothing revealed);
2 usage error; 3 an output could not be written.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decompile import evaluate as ev  # noqa: E402
from decompile import frames as fr  # noqa: E402
from decompile import holdout as ho  # noqa: E402
from decompile import ball_measure as bm  # noqa: E402
from decompile import ingest  # noqa: E402
from decompile import kinematic_fit as kf  # noqa: E402
from decompile import meta_api  # noqa: E402
from decompile import overlay as ov  # noqa: E402
from decompile import predict as pd  # noqa: E402
from decompile import predict_projective as ppred  # noqa: E402
from decompile import sam_api as sa  # noqa: E402
from decompile import sam_measure as sm  # noqa: E402
from decompile import sam_tracking as st  # noqa: E402
from decompile import projective_fit as pfit  # noqa: E402
from decompile import projective_program as pprog  # noqa: E402
from decompile import spark_stage as ss  # noqa: E402
from decompile import tracker as tk  # noqa: E402
from decompile import trajectories as tr  # noqa: E402
from decompile.meta_api import strict_json_loads  # noqa: E402


def _load_p5_tool():
    spec = importlib.util.spec_from_file_location("video_predictive_demo_p5", ROOT / "tools" / "video_predictive_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p5 = _load_p5_tool()
stage_predict = _p5.stage_predict  # reads ONLY the inference-inputs file; fit gate; frozen prediction
GateError = _p5.GateError
interpret = _p5.interpret  # P4's fixed comparison statements

TOOL_VERSION = "tools.real_video_demo/1"
PREREG_KIND = "real_video_preregistration"
PREREG_VERSION = 1
PREREG_VERSION_P7 = 2
PREREG_VERSION_SAM = 3
SUPPORTED_PREREG_VERSIONS = (PREREG_VERSION, PREREG_VERSION_P7, PREREG_VERSION_SAM)
SAM_BACKEND = "sam_3_1"
SAM_DEFAULT_GATES = {"max_training_gaps": 0, "max_withheld_gaps": 0, "max_diameter_rel_dev": 0.5}
SAM_OBJECT_RULES = ("most_frames",)
SAM_DISCLOSURE = ("The clip window and the SAM prompt were chosen by a human looking at the video; the human could see "
                  "the whole clip, including the withheld part, before pre-registration.")
PHYSICS_MODELS = ("image_space", "projective")
PROVENANCES = {"real": False, "synthetic_stand_in": True}  # value: the TrajectorySet `synthetic` flag
SPARK_MODES = ("off", "live")
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "experiments"
DEFAULT_GATES = {"max_training_gaps": 0, "max_area_rel_dev": 0.5, "fill_ratio": [0.45, 0.95]}
MIN_CLIP_FRAMES = 12
EXIT_OK, EXIT_GATE, EXIT_USAGE, EXIT_WRITE = 0, 1, 2, 3

STANDARD_DISCLOSURES = (
    "The clip window, frame stride and seed box were chosen by a human looking at the video; the human could see "
    "the whole clip, including the withheld part, before pre-registration.",
)
LIMITATIONS = {
    "real": (
        "REAL VIDEO, one clip. This is a single demonstration, not a general real-world validation.",
        "Withheld positions are the classical tracker's measurements of the withheld frames, NOT independently "
        "verified ground truth; prediction errors include tracker measurement error.",
        "Tracking accuracy is reported only against independent manual annotations when supplied; otherwise it is "
        "not available.",
        "Contact timing is not evaluated: there are no independent event annotations.",
        "Image pixels and seconds; static camera assumed; perspective, lens distortion and rolling shutter are not "
        "modelled.",
        "Scene program v1: one body, exactly one contact inside the training frames, uniform acceleration, one "
        "axis-aligned INFERRED contact boundary, restitution in [0, 1], no drag.",
        "Point predictions only: parameter uncertainty is not propagated to prediction bands.",
    ),
    "synthetic_stand_in": (
        "SYNTHETIC stand-in video used to exercise the real-video runner; not real footage and not real-video "
        "validation.",
        "Withheld positions are the classical tracker's measurements of the withheld frames, not ground truth.",
        "Tracking accuracy is reported only against independent annotations when supplied.",
        "Contact timing is not evaluated: there are no independent event annotations.",
        "Point predictions only; image pixels and seconds.",
    ),
}


LIMITATIONS_P7 = {
    "real": (
        "REAL VIDEO, one clip. This is a single demonstration, not a general real-world validation.",
        "Withheld positions are the tracker's (outline-method) measurements of the withheld frames, NOT independently "
        "verified ground truth; prediction errors include measurement error.",
        "Tracking accuracy is reported only against independent manual annotations when supplied; otherwise it is "
        "not available.",
        "Contact timing is not evaluated: there are no independent event annotations.",
        "Perspective is modelled with a pinhole camera from the measured apparent diameter; the focal length enters "
        "only through the floor orientation and comes from a pre-registered field-of-view PRIOR, not a calibration. "
        "Lens distortion and rolling shutter are not modelled; the camera is assumed static.",
        "Projective scene program v1: one body, one contact inside the training frames, constant 3-D gravity, a "
        "plane floor perpendicular to gravity through the training contact, the training contact law reused, no "
        "drag, no spin dynamics.",
        "Point predictions only: parameter uncertainty is not propagated to prediction bands (the camera-prior "
        "sensitivity is reported separately).",
    ),
    "synthetic_stand_in": (
        "SYNTHETIC stand-in video used to exercise the real-video runner; not real footage and not real-video "
        "validation.",
        "Withheld positions are the tracker's measurements of the withheld frames, not ground truth.",
        "Tracking accuracy is reported only against independent annotations when supplied.",
        "Contact timing is not evaluated: there are no independent event annotations.",
        "Perspective modelled with a camera field-of-view prior; point predictions only.",
    ),
}


LIMITATIONS_SAM = {
    prov: (
        ("REAL VIDEO, one clip. This is a single demonstration, not a general real-world validation."
         if prov == "real" else
         "SYNTHETIC stand-in video used to exercise the runner; not real footage and not real-video validation."),
        "Object identity and support come from Meta SAM 3.1; the centre, apparent diameter and their uncertainties "
        "come from Decompile's outline measurement in the original pixels. Withheld positions are these "
        "measurements, NOT independently verified ground truth; prediction errors include measurement error.",
        "SAM masks are never treated as centres; frames whose outline could not be measured reliably are gaps "
        "(never interpolated).",
        "Tracking accuracy is reported only against independent manual annotations when supplied; otherwise it is "
        "not available.",
        "Contact timing is not evaluated: there are no independent event annotations.",
        "Static camera assumed; lens distortion and rolling shutter are not modelled. One body, one contact inside "
        "the training frames; point predictions only.",
    ) for prov in PROVENANCES
}


class UsageError(Exception):
    pass


class VerifyError(Exception):
    pass


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def _write_json(path: Path, obj) -> str:
    raw = pd.canonical_bytes(obj)
    ov.write_new(path, raw)
    return _sha_bytes(raw)


def _write_with_record(path: Path, obj) -> str:
    """Write canonical JSON plus a '<sha256>  <name>' record (both never overwritten)."""
    digest = _write_json(path, obj)
    ov.write_new(pd.sha256_path_for(path), f"{digest}  {path.name}\n".encode("ascii"))
    return digest


# ------------------------------------------------------------------ pre-registration


def make_preregistration(video: Path, *, seed_box, start_frame: int = 0, end_frame: int | None = None,
                         stride: int = 1, roi=None, exclude_boxes=(), object_id: str = "obj_01",
                         train_fraction: Fraction = Fraction(7, 10), gates: dict | None = None,
                         provenance: str = "real", spark_mode: str = "off", notes: str = "",
                         disclosures=(), centre_method: str = "colour_centroid", physics_model: str = "image_space",
                         fit_noise: str = "absolute", camera_prior: dict | None = None,
                         max_withheld_gaps: int = 0) -> dict:
    video = Path(video)
    if not video.is_file():
        raise UsageError(f"video not found: {video}")
    if provenance not in PROVENANCES:
        raise UsageError(f"provenance must be one of {sorted(PROVENANCES)}")
    if spark_mode not in SPARK_MODES:
        raise UsageError(f"spark mode must be one of {SPARK_MODES}")
    if not (isinstance(stride, int) and stride >= 1 and isinstance(start_frame, int) and start_frame >= 0):
        raise UsageError("stride must be >= 1 and start_frame >= 0")
    if end_frame is not None and end_frame <= start_frame:
        raise UsageError("end_frame must be greater than start_frame")
    f = Fraction(train_fraction)
    if not 0 < f < 1:
        raise UsageError("train_fraction must be strictly between 0 and 1")
    p7 = centre_method != "colour_centroid" or physics_model != "image_space" or max_withheld_gaps != 0
    if centre_method not in tk.CENTRE_METHODS:
        raise UsageError(f"centre_method must be one of {tk.CENTRE_METHODS}")
    if physics_model not in PHYSICS_MODELS:
        raise UsageError(f"physics_model must be one of {PHYSICS_MODELS}")
    if physics_model == "projective" and centre_method == "colour_centroid":
        raise UsageError("the projective model needs an edge centre method (per-frame sigma and apparent diameter)")
    if fit_noise not in pfit.NOISE_MODES:
        raise UsageError(f"fit_noise must be one of {pfit.NOISE_MODES}")
    if not (isinstance(max_withheld_gaps, int) and max_withheld_gaps >= 0):
        raise UsageError("max_withheld_gaps must be a non-negative int")
    base_gates = {**DEFAULT_GATES, "max_withheld_gaps": max_withheld_gaps} if p7 else dict(DEFAULT_GATES)
    g = {**base_gates, **(gates or {})}
    if set(g) != set(base_gates):
        raise UsageError(f"gates must have exactly the keys {sorted(base_gates)}")
    prior = None
    if physics_model == "projective":
        cp = camera_prior or {}
        try:
            prior = pprog.CameraPrior(hfov_deg=float(cp.get("hfov_deg", 66.0)),
                                      hfov_range_deg=tuple(float(v) for v in cp.get("hfov_range_deg", (50.0, 80.0))),
                                      **({"basis": cp["basis"]} if "basis" in cp else {}))
        except (ValueError, TypeError) as exc:
            raise UsageError(f"camera_prior: {exc}") from None
    # validate the boxes now (no frame is read)
    tk.TrackerConfig(object_id=object_id, roi=tuple(roi) if roi else None,
                     exclude_boxes=tuple(tuple(b) for b in exclude_boxes))
    tk._box(tuple(seed_box), "seed_box")
    data = {
        "kind": PREREG_KIND, "prereg_version": PREREG_VERSION_P7 if p7 else PREREG_VERSION, "tool_version": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "video": {"path": str(video.resolve()), "sha256": _sha_file(video), "size_bytes": video.stat().st_size},
        "video_provenance": provenance,
        "clip": {"start_frame": start_frame, "end_frame": end_frame, "stride": stride,
                 "semantics": "frame-table indices start_frame, start_frame+stride, ... < end_frame (exclusive; null = "
                              "to the end); the first clip frame is the seed frame"},
        "train_fraction": {"num": f.numerator, "den": f.denominator},
        "split_semantics": "the first floor(train_fraction * n_clip_frames) clip frames are training; the rest withheld",
        "tracker": {"object_id": object_id, "seed_box": list(seed_box), "roi": list(roi) if roi else None,
                    "exclude_boxes": [list(b) for b in exclude_boxes]},
        "gates": g, "spark": {"mode": spark_mode},
        "operator_notes": notes, "disclosures": list(STANDARD_DISCLOSURES) + list(disclosures),
    }
    if p7:
        data["measurement"] = {"centre_method": centre_method,
                               "edge_method_version": bm.METHOD_VERSION if centre_method != "colour_centroid" else None}
        data["physics"] = {"model": physics_model, "noise": fit_noise,
                           "camera_prior": prior.to_json_dict() if prior is not None else None,
                           "versions": ({"projective_fit": pfit.METHOD_VERSION, "projective_program": pprog.PROGRAM_VERSION,
                                         "predictor": ppred.PREDICTOR_VERSION} if physics_model == "projective" else
                                        {"kinematic_fit": kf.METHOD_VERSION})}
    return data


def make_preregistration_sam(video: Path, *, prompt: dict, object_selection="most_frames", start_frame: int = 0,
                             end_frame: int | None = None, stride: int = 1, object_id: str = "obj_01",
                             train_fraction: Fraction = Fraction(7, 10), gates: dict | None = None,
                             provenance: str = "real", spark_mode: str = "off", notes: str = "", disclosures=(),
                             outline_model: str = "ellipse", physics_model: str = "image_space",
                             fit_noise: str = "absolute", camera_prior: dict | None = None,
                             min_confidence: float = 0.5) -> dict:
    """Pre-registration version 3: SAM 3.1 identity/support + outline measurement. Reads no frame, calls no API."""
    video = Path(video)
    if not video.is_file():
        raise UsageError(f"video not found: {video}")
    if provenance not in PROVENANCES:
        raise UsageError(f"provenance must be one of {sorted(PROVENANCES)}")
    if spark_mode not in SPARK_MODES:
        raise UsageError(f"spark mode must be one of {SPARK_MODES}")
    if not (isinstance(stride, int) and stride >= 1 and isinstance(start_frame, int) and start_frame >= 0):
        raise UsageError("stride must be >= 1 and start_frame >= 0")
    if end_frame is not None and end_frame <= start_frame:
        raise UsageError("end_frame must be greater than start_frame")
    f = Fraction(train_fraction)
    if not 0 < f < 1:
        raise UsageError("train_fraction must be strictly between 0 and 1")
    try:
        sa.prompt_from_json(prompt)
    except (sa.PromptError, KeyError, TypeError) as exc:
        raise UsageError(f"prompt: {exc}") from None
    if not (object_selection in SAM_OBJECT_RULES or (isinstance(object_selection, dict) and set(object_selection) == {"id"})):
        raise UsageError(f"object_selection must be one of {SAM_OBJECT_RULES} or {{'id': <SAM object id>}}")
    if not tr.OBJECT_ID_RE.fullmatch(object_id):
        raise UsageError(f"object_id must match {tr.OBJECT_ID_RE.pattern}")
    if physics_model not in PHYSICS_MODELS:
        raise UsageError(f"physics_model must be one of {PHYSICS_MODELS}")
    if fit_noise not in pfit.NOISE_MODES:
        raise UsageError(f"fit_noise must be one of {pfit.NOISE_MODES}")
    g = {**SAM_DEFAULT_GATES, **(gates or {})}
    if set(g) != set(SAM_DEFAULT_GATES):
        raise UsageError(f"gates must have exactly the keys {sorted(SAM_DEFAULT_GATES)}")
    try:
        params = sm.SamMeasureParams(outline_model=outline_model, min_confidence=min_confidence)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    prior = None
    if physics_model == "projective":
        cp = camera_prior or {}
        try:
            prior = pprog.CameraPrior(hfov_deg=float(cp.get("hfov_deg", 66.0)),
                                      hfov_range_deg=tuple(float(v) for v in cp.get("hfov_range_deg", (50.0, 80.0))),
                                      **({"basis": cp["basis"]} if "basis" in cp else {}))
        except (ValueError, TypeError) as exc:
            raise UsageError(f"camera_prior: {exc}") from None
    return {
        "kind": PREREG_KIND, "prereg_version": PREREG_VERSION_SAM, "tool_version": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "video": {"path": str(video.resolve()), "sha256": _sha_file(video), "size_bytes": video.stat().st_size},
        "video_provenance": provenance,
        "clip": {"start_frame": start_frame, "end_frame": end_frame, "stride": stride,
                 "semantics": "frame-table indices start_frame, start_frame+stride, ... < end_frame (exclusive; null = "
                              "to the end)"},
        "train_fraction": {"num": f.numerator, "den": f.denominator},
        "split_semantics": "the first floor(train_fraction * n_clip_frames) clip frames are training; the rest withheld",
        "tracking": {"backend": SAM_BACKEND, "model": sa.SAM_MODEL, "client_version": sa.CLIENT_VERSION,
                     "prompt": prompt, "include_confidence": True, "object_selection": object_selection,
                     "trajectory_object_id": object_id,
                     "sam_frame_semantics": "SAM <Nf> = zero-based source frame index of the uploaded file (same SHA-256)"},
        "measurement": {"method_version": sm.METHOD_VERSION, "edge_method_version": bm.METHOD_VERSION,
                        "params": params.to_json_dict()},
        "physics": {"model": physics_model, "noise": fit_noise,
                    "camera_prior": prior.to_json_dict() if prior is not None else None,
                    "versions": ({"projective_fit": pfit.METHOD_VERSION, "projective_program": pprog.PROGRAM_VERSION,
                                  "predictor": ppred.PREDICTOR_VERSION} if physics_model == "projective" else
                                 {"kinematic_fit": kf.METHOD_VERSION})},
        "gates": g, "spark": {"mode": spark_mode},
        "operator_notes": notes, "disclosures": [SAM_DISCLOSURE] + list(disclosures),
    }


def load_preregistration(path: Path) -> tuple[dict, str]:
    path = Path(path)
    raw = path.read_bytes()
    rec = pd.sha256_path_for(path)
    if not rec.is_file():
        raise VerifyError(f"pre-registration has no SHA-256 record: {rec}")
    digest = _sha_bytes(raw)
    if rec.read_text(encoding="ascii").split()[0] != digest:
        raise VerifyError("pre-registration does not match its SHA-256 record (it was changed)")
    data = strict_json_loads(raw.decode("utf-8"))
    if data.get("kind") != PREREG_KIND or data.get("prereg_version") not in SUPPORTED_PREREG_VERSIONS:
        raise VerifyError("not a supported pre-registration file")
    return data, digest


# ------------------------------------------------------------------ measurement (video + pre-registration only)


def clip_frames(table: fr.FrameTable, clip: dict) -> list[int]:
    idx = [f.index for f in table.frames]
    end = clip["end_frame"] if clip["end_frame"] is not None else idx[-1] + 1
    sel = [i for i in idx if clip["start_frame"] <= i < end and (i - clip["start_frame"]) % clip["stride"] == 0]
    if len(sel) < MIN_CLIP_FRAMES:
        raise GateError("clip", [f"the clip has {len(sel)} frames; at least {MIN_CLIP_FRAMES} are needed"])
    return sel


def _failure_reason(d: tk.Detection) -> str:
    if d.status == tk.EDGE_REJECTED and d.edge is not None:
        return f"edge_fit_rejected: {d.edge.get('rejection_reason')}"
    if d.status == tk.MISSING and d.n_mask_px is not None:
        return "no target-coloured pixels" if d.n_mask_px == 0 else "target-coloured pixels below the minimum area"
    return d.status


def real_tracking_gate(result: tk.TrackingResult, training_frames: list[int], withheld_frames: list[int],
                       gates: dict) -> dict:
    """Pre-freeze gate from tracker diagnostics only (no positions, no ground truth)."""
    by = {d.frame_index: d for d in result.detections}
    problems, gaps = [], []
    max_wg = gates.get("max_withheld_gaps", 0)
    withheld_gaps = []
    for i in withheld_frames:
        d = by[i].detection
        if d.status != tk.DETECTED:
            if max_wg == 0:
                problems.append(f"withheld frame {i}: {_failure_reason(d)} (every withheld frame must be detected)")
            withheld_gaps.append({"frame_index": i, "status": d.status, "reason": _failure_reason(d)})
    if max_wg > 0 and len(withheld_gaps) > max_wg:
        problems.append(f"{len(withheld_gaps)} withheld frame(s) failed measurement; the pre-registered maximum is "
                        f"{max_wg}")
    for i in training_frames:
        d = by[i].detection
        if d.status != tk.DETECTED:
            gaps.append({"frame_index": i, "status": d.status, "reason": _failure_reason(d)})
    if len(gaps) > gates["max_training_gaps"]:
        problems.append(f"{len(gaps)} training frame(s) failed detection; the pre-registered maximum is "
                        f"{gates['max_training_gaps']}")
    if by[training_frames[0]].detection.status != tk.DETECTED:
        problems.append(f"the seed frame {training_frames[0]} was not detected")
    det = [d for d in result.detections if d.detection.status == tk.DETECTED]
    med = statistics.median(d.detection.area_px for d in det) if det else None
    lo, hi = gates["fill_ratio"]
    for fd in det:
        d = fd.detection
        if abs(d.area_px - med) > gates["max_area_rel_dev"] * med:
            problems.append(f"frame {fd.frame_index}: area {d.area_px} px deviates > {gates['max_area_rel_dev']:.0%} "
                            f"from the median {med}")
        if not lo <= d.fill_ratio <= hi:
            problems.append(f"frame {fd.frame_index}: fill ratio {d.fill_ratio:.3f} outside [{lo}, {hi}]")
    return {"passed": not problems, "problems": problems, "training_gaps": gaps, "gates": gates,
            "n_clip_frames": len(training_frames) + len(withheld_frames), "n_training_frames": len(training_frames),
            "n_withheld_frames": len(withheld_frames), "n_detected": len(det),
            "status_counts": result.status_counts(), "median_area_px": med,
            "area_px_range": [min(d.detection.area_px for d in det), max(d.detection.area_px for d in det)] if det else None,
            "fill_ratio_range": [min(d.detection.fill_ratio for d in det),
                                 max(d.detection.fill_ratio for d in det)] if det else None,
            "note": "Diagnostics only. Tracking ACCURACY needs an independent reference and is not assessed here.",
            **({"withheld_gaps": withheld_gaps} if "max_withheld_gaps" in gates else {})}


def stage_measure(video: Path, exp_dir: Path, prereg: dict) -> dict:
    """MP4 -> validated frames -> seeded tracker on the clip frames. Reads the video and the pre-registration only."""
    m = exp_dir / "measurement"
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    _write_json(m / "probe.json", {"kind": "video_probe", "info": dataclasses.asdict(info), "rejection_reasons": reasons})
    if reasons:
        raise GateError("video_validation", reasons)
    if info.sha256 != prereg["video"]["sha256"]:
        raise GateError("video_hash", ["the video does not match the pre-registered SHA-256"])
    frames_dir = m / "frames"
    table = fr.extract_frames(video, frames_dir)
    clip = clip_frames(table, prereg["clip"])
    f = Fraction(prereg["train_fraction"]["num"], prereg["train_fraction"]["den"])
    n_train = math.floor(f * len(clip))
    training_frames, withheld_frames = clip[:n_train], clip[n_train:]
    if not training_frames or not withheld_frames:
        raise GateError("split", ["the clip leaves no training or no withheld frame"])
    t = prereg["tracker"]
    seed_rgb = fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=clip[0]))
    try:
        config, derivation = tk.config_from_seed(
            seed_rgb, tuple(t["seed_box"]), object_id=t["object_id"], roi=tuple(t["roi"]) if t["roi"] else None,
            exclude_boxes=tuple(tuple(b) for b in t["exclude_boxes"]))
    except tk.SeedError as exc:
        raise GateError("seed", [str(exc)]) from None
    method = prereg.get("measurement", {}).get("centre_method", "colour_centroid")
    if method != "colour_centroid":  # P7: the seed colour model becomes the coarse stage of the outline method
        config = dataclasses.replace(config, centre_method=method)
    _write_json(m / "seed_derivation.json", {"kind": "seed_derivation", "seed_frame": clip[0], **derivation,
                                             "config": config.to_json_dict()})
    result = tk.track_extracted(frames_dir, config, expected_source_sha256=info.sha256, frames=clip)
    _write_json(m / "tracking_report.json", result.to_json_dict())
    gate = real_tracking_gate(result, training_frames, withheld_frames, prereg["gates"])
    _write_json(m / "tracking_gate.json", gate)
    if not gate["passed"]:
        raise GateError("tracking", gate["problems"])
    detected = dataclasses.replace(result, detections=tuple(d for d in result.detections
                                                           if d.detection.status == tk.DETECTED))
    traj = detected.to_trajectory_set(synthetic=PROVENANCES[prereg["video_provenance"]])
    tracked_path = m / "tracked_trajectories.json"
    _write_json(tracked_path, traj.to_json_dict())
    return {"tracked_path": tracked_path, "training_frames": training_frames, "withheld_frames": withheld_frames,
            "gate": gate, "config": config.to_json_dict(), "frames_manifest_sha256": _sha_file(frames_dir / fr.MANIFEST_NAME),
            "probe": {"sha256": info.sha256, "codec": info.codec, "width": info.width, "height": info.height,
                      "nominal_fps": info.nominal_fps, "duration_s": info.duration_s,
                      "decoded_frame_count": info.decoded_frame_count, "frame_count_status": info.frame_count_status},
            "extraction": {"frame_count": len(table.frames), "time_base": str(table.time_base),
                           "synthesized_timestamp_count": table.synthesized_timestamp_count,
                           "clip_first_frame": clip[0], "clip_last_frame": clip[-1], "clip_frames": len(clip)}}


def sam_tracking_gate(meas: sm.SamMeasurementResult, training_frames: list[int], withheld_frames: list[int],
                      gates: dict) -> dict:
    """Pre-freeze gate from measurement STATUSES and apparent diameters only (no positions, no ground truth)."""
    by = {f.frame_index: f for f in meas.frames}

    def gap(f):
        return {"frame_index": f.frame_index, "status": f.status, "reason": f.reason}

    training_gaps = [gap(by[i]) for i in training_frames if by[i].status != sm.MEASURED]
    withheld_gaps = [gap(by[i]) for i in withheld_frames if by[i].status != sm.MEASURED]
    problems = []
    if len(training_gaps) > gates["max_training_gaps"]:
        problems.append(f"{len(training_gaps)} training frame(s) were not measured; the pre-registered maximum is "
                        f"{gates['max_training_gaps']}")
    if len(withheld_gaps) > gates["max_withheld_gaps"]:
        problems.append(f"{len(withheld_gaps)} withheld frame(s) were not measured; the pre-registered maximum is "
                        f"{gates['max_withheld_gaps']}")
    measured = meas.measured()
    med = statistics.median(f.diameter_px for f in measured) if measured else None
    for f in measured:
        if abs(f.diameter_px - med) > gates["max_diameter_rel_dev"] * med:
            problems.append(f"frame {f.frame_index}: apparent diameter {f.diameter_px:.2f} px deviates > "
                            f"{gates['max_diameter_rel_dev']:.0%} from the median {med:.2f}")
    if not measured:
        problems.append("no frame was measured")
    return {"passed": not problems, "problems": problems, "training_gaps": training_gaps, "withheld_gaps": withheld_gaps,
            "gates": gates, "n_clip_frames": len(training_frames) + len(withheld_frames),
            "n_training_frames": len(training_frames), "n_withheld_frames": len(withheld_frames),
            "n_detected": len(measured), "status_counts": meas.status_counts(), "median_diameter_px": med,
            "area_px_range": ([min(f.mask_area_px for f in measured), max(f.mask_area_px for f in measured)]
                              if measured else None),
            "note": "Diagnostics only. Measurement ACCURACY needs an independent reference and is not assessed here."}


def load_sam_result(path: Path, prereg: dict) -> tuple[st.SamTrackingResult, str]:
    """Load a SAM tracking result and verify it belongs to the pre-registered video and prompt."""
    raw = Path(path).read_bytes()
    try:
        result = st.SamTrackingResult.from_json_dict(strict_json_loads(raw.decode("utf-8")))
    except (ValueError, KeyError, TypeError) as exc:
        raise VerifyError(f"SAM result {path}: {exc}") from None
    t = prereg["tracking"]
    problems = []
    if result.video.get("sha256") != prereg["video"]["sha256"]:
        problems.append("the SAM result was computed on a different video (SHA-256 mismatch)")
    if result.prompt != t["prompt"]:
        problems.append(f"the SAM result's prompt {result.prompt} differs from the pre-registered prompt {t['prompt']}")
    if result.model != t["model"]:
        problems.append(f"the SAM result's model {result.model!r} differs from {t['model']!r}")
    if problems:
        raise VerifyError("; ".join(problems))
    return result, _sha_bytes(raw)


def measure_params_from_json(p: dict) -> sm.SamMeasureParams:
    ep = dict(p["edge_params"])
    ep["inner_frac"] = tuple(ep["inner_frac"])
    return sm.SamMeasureParams(outline_model=p["outline_model"], min_confidence=p["min_confidence"],
                               min_mask_area_px=p["min_mask_area_px"], other_object_margin_px=p["other_object_margin_px"],
                               min_iou=p["min_iou"], diameter_ratio_range=tuple(p["diameter_ratio_range"]),
                               max_centre_offset_frac=p["max_centre_offset_frac"], edge_params=bm.EdgeParams(**ep))


def stage_measure_sam(video: Path, exp_dir: Path, prereg: dict, sam_result_path: Path) -> dict:
    """MP4 -> validated frames; SAM result (verified) -> outline measurement of the clip frames -> gate."""
    m = exp_dir / "measurement"
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    _write_json(m / "probe.json", {"kind": "video_probe", "info": dataclasses.asdict(info), "rejection_reasons": reasons})
    if reasons:
        raise GateError("video_validation", reasons)
    if info.sha256 != prereg["video"]["sha256"]:
        raise GateError("video_hash", ["the video does not match the pre-registered SHA-256"])
    result, sam_sha = load_sam_result(sam_result_path, prereg)
    frames_dir = m / "frames"
    table = fr.extract_frames(video, frames_dir)
    if (result.media["width"], result.media["height"]) != (table.width, table.height) or \
            result.media["frame_count"] != len(table.frames):
        raise GateError("sam_media", [f"SAM media {result.media} does not match the extracted frames "
                                      f"({table.width}x{table.height}, {len(table.frames)} frames)"])
    if not result.completed:
        raise GateError("sam_outcome", [f"the SAM response is {result.outcome.get('status')!r} "
                                        f"({result.outcome.get('reason')}); only a completed response is measured"])
    clip = clip_frames(table, prereg["clip"])
    f = Fraction(prereg["train_fraction"]["num"], prereg["train_fraction"]["den"])
    n_train = math.floor(f * len(clip))
    training_frames, withheld_frames = clip[:n_train], clip[n_train:]
    if not training_frames or not withheld_frames:
        raise GateError("split", ["the clip leaves no training or no withheld frame"])
    t = prereg["tracking"]
    try:
        sam_oid = sm.select_object(result, clip, t["object_selection"])
    except st.SamTrackingError as exc:
        raise GateError("sam_object", exc.problems) from None
    params = measure_params_from_json(prereg["measurement"]["params"])
    by_index = {fx.index: fx for fx in table.frames}

    def source(i):
        return by_index[i], fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=i))

    meas = sm.measure_track(result, source, clip, sam_object_id=sam_oid, trajectory_object_id=t["trajectory_object_id"],
                            params=params, sam_result_sha256=sam_sha)
    _write_json(m / "sam_measurement_report.json", meas.to_json_dict())
    gate = sam_tracking_gate(meas, training_frames, withheld_frames, prereg["gates"])
    _write_json(m / "tracking_gate.json", gate)
    if not gate["passed"]:
        raise GateError("tracking", gate["problems"])
    traj = meas.to_trajectory_set(synthetic=PROVENANCES[prereg["video_provenance"]])
    tracked_path = m / "tracked_trajectories.json"
    _write_json(tracked_path, traj.to_json_dict())
    return {"tracked_path": tracked_path, "training_frames": training_frames, "withheld_frames": withheld_frames,
            "gate": gate, "sam": {"result_path": str(Path(sam_result_path).resolve()), "result_sha256": sam_sha,
                                  "sam_object_id": sam_oid, "outcome": result.outcome, "usage": result.usage,
                                  "summary": result.summary()},
            "frames_manifest_sha256": _sha_file(frames_dir / fr.MANIFEST_NAME),
            "probe": {"sha256": info.sha256, "codec": info.codec, "width": info.width, "height": info.height,
                      "nominal_fps": info.nominal_fps, "duration_s": info.duration_s,
                      "decoded_frame_count": info.decoded_frame_count, "frame_count_status": info.frame_count_status},
            "extraction": {"frame_count": len(table.frames), "time_base": str(table.time_base),
                           "synthesized_timestamp_count": table.synthesized_timestamp_count,
                           "clip_first_frame": clip[0], "clip_last_frame": clip[-1], "clip_frames": len(clip)}}


# ------------------------------------------------------------------ split (measurement + evaluation side)


def stage_split(tracked_path: Path, exp_dir: Path, *, first_withheld_frame: int, notes: list[str]) -> dict:
    traj = tr.load_trajectories(tracked_path)
    frames = sorted({o.frame_index for obj in traj.objects for o in obj.observations})
    n_train = sum(1 for i in frames if i < first_withheld_frame)
    split = ho.split_trajectories(traj, Fraction(n_train, len(frames)))  # exact: floor(n_train/n * n) = n_train
    # the first TRACKED withheld frame: the pre-registered one, or (P7, withheld gaps allowed) the next measured one
    expected = min((i for i in frames if i >= first_withheld_frame), default=None)
    if split.inference.first_withheld_frame != expected:
        raise GateError("split", [f"tracked split starts at frame {split.inference.first_withheld_frame}, not at the "
                                  f"pre-registered first withheld frame {first_withheld_frame}"])
    inputs_path = exp_dir / "inference" / "inference_inputs.json"
    inputs_sha = _write_json(inputs_path, ho.inference_inputs_to_json(split.inference))
    vault_path = exp_dir / "evaluation_only" / "evaluation_vault.json"
    vault_sha = ev.write_evaluation_vault(vault_path, split, ground_truth_contacts=None, notes=notes)
    inf = split.inference
    return {"inputs_path": inputs_path, "inputs_sha256": inputs_sha, "vault_path": vault_path, "vault_sha256": vault_sha,
            "training_sha256": inf.training_sha256, "schedule_sha256": inf.schedule_sha256,
            "split": {"tracked_fraction": str(inf.train_fraction), "n_tracked_frames": inf.n_frames,
                      "n_train_tracked": inf.n_train_frames, "first_withheld_frame": inf.first_withheld_frame,
                      "n_withheld_tracked": len(inf.schedule)}}


# ------------------------------------------------------------------ advisory Muse Spark stage (training only)


def load_training_only(inputs_path: Path) -> tr.TrajectorySet:
    """The training TrajectorySet of an inference-inputs file, verified against its recorded hash."""
    data = strict_json_loads(Path(inputs_path).read_bytes().decode("utf-8"))
    training = tr.TrajectorySet.from_json_dict(data["training"])
    if ho.sha256_canonical(training.to_json_dict()) != data["training_sha256"]:
        raise VerifyError("training data does not match its recorded training_sha256")
    return training


def live_client_factory(evidence_dir: Path):
    def factory() -> meta_api.MuseSparkClient:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        return meta_api.MuseSparkClient(evidence_dir=evidence_dir)
    return factory


def stage_spark(inputs_path: Path, exp_dir: Path, *, client_factory=None) -> dict:
    """Advisory: training trajectory only; writes inference/spark/spark_interpretation.json; never raises for AI failures."""
    training = load_training_only(inputs_path)
    return ss.run_spark_stage(training, exp_dir / "inference" / "spark", client_factory=client_factory)


# ------------------------------------------------------------------ projective predict (P7; inference side)


def stage_predict_projective(inputs_path: Path, exp_dir: Path, notes: list[str], physics: dict) -> dict:
    """Reads ONLY the inference-inputs file; projective fit gate; frozen format-2 prediction."""
    training, schedule = pd.load_inference_inputs(inputs_path)
    cp = physics["camera_prior"]
    prior = pprog.CameraPrior(hfov_deg=cp["hfov_deg"], hfov_range_deg=tuple(cp["hfov_range_deg"]), basis=cp["basis"])
    objs, problems = [], []
    for obj in training.objects:
        single = tr.TrajectorySet(training.synthetic, training.source, training.coordinate_system, (obj,))
        entry = {"object_id": obj.object_id}
        try:
            rep_ = pfit.analyze_projective(single, noise=physics["noise"])
            entry.update({"selected_model": rep_.selected, "adequacy": rep_.adequacy,
                          "adequacy_reasons": list(rep_.adequacy_reasons),
                          "verdicts": {k: v["verdict"] for k, v in rep_.verdicts.items()},
                          "delta_bic": {k: v.get("delta_bic") for k, v in rep_.verdicts.items()},
                          "bic": {k: f.bic for k, f in rep_.fits.items()},
                          "chi2": {k: [f.chi2, f.chi2_dof, f.chi2_p] for k, f in rep_.fits.items()}})
            prog = pprog.program_from_fit(rep_, width=training.coordinate_system.image_width, prior=prior,
                                          t_last_rel=float(obj.observations[-1].t) - rep_.t0)
            entry["program"] = {"sha256": pprog.program_sha256(prog), "restitution": prog.restitution,
                                "gravity_magnitude_diameters_s2": prog.provenance["gravity_magnitude_diameters_s2"]}
        except pfit.ProjectiveFitError as exc:
            problems.append(f"{obj.object_id}: {exc}")
        except pprog.ProgramUnavailable as exc:
            problems += [f"{obj.object_id}: {r}" for r in exc.reasons]
        objs.append(entry)
    gate = {"kind": "projective_fit_gate", "method": pfit.METHOD_VERSION, "passed": not problems, "problems": problems,
            "objects": objs}
    gate_path = exp_dir / "inference" / "fit_gate.json"
    gate_sha = _write_json(gate_path, gate)
    if problems:
        raise GateError("training_fit", problems)
    try:
        r = ppred.predict(training, schedule, output_path=exp_dir / "inference" / "prediction.json", camera_prior=prior,
                          noise=physics["noise"], experiment_notes=notes)
    except pd.PredictionError as exc:
        raise GateError("prediction", [str(exc)]) from None
    return {"prediction_path": r.artifact_path, "sha256_path": r.sha256_path, "prediction_sha256": r.sha256,
            "fit_gate": gate, "fit_gate_path": gate_path, "fit_gate_sha256": gate_sha,
            "program_sha256": {o["object_id"]: o["program_sha256"] for o in r.artifact["objects"]}}


# ------------------------------------------------------------------ run (stops after freezing)


def run(prereg_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR, *, confirm_live_call: bool = False,
        spark_client_factory=None, experiment_id: str | None = None, sam_result: Path | None = None) -> dict:
    prereg, prereg_sha = load_preregistration(prereg_path)
    sam = prereg["prereg_version"] == PREREG_VERSION_SAM
    if sam and sam_result is None:
        raise UsageError("this pre-registration uses the SAM 3.1 backend: pass --sam-result (from sam_demo.py track)")
    if not sam and sam_result is not None:
        raise UsageError("--sam-result is only valid with a SAM (version 3) pre-registration")
    mode = prereg["spark"]["mode"]
    if spark_client_factory is None and mode == "live":
        if not confirm_live_call:
            raise UsageError("the pre-registration enables a LIVE Muse Spark call; pass --confirm-live-call to make it")
    video = Path(prereg["video"]["path"])
    if not video.is_file() or _sha_file(video) != prereg["video"]["sha256"]:
        raise VerifyError("the pre-registered video is missing or its SHA-256 changed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = "sam" if sam else "p6"
    experiment_id = experiment_id or f"{prefix}_{prereg['video_provenance']}_video_{stamp}_{uuid.uuid4().hex[:8]}"
    exp_dir = Path(output_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=False)
    label = "REAL VIDEO" if prereg["video_provenance"] == "real" else "SYNTHETIC stand-in VIDEO"
    # The prediction's notes must not depend on anything AI-related (spark mode, run id, pre-registration hash,
    # which covers the spark mode): only the label, a fixed sentence and the operator's disclosures. The
    # pre-registration hash and the experiment id are linked in run_manifest.json instead.
    if sam:
        notes = [f"SAM 3.1 {label} experiment. The predictor saw only outline measurements (SAM-guided identity and "
                 "support) of the training clip frames and the timestamps of the withheld frames."]
    else:
        notes = [f"P6 {label} experiment. The predictor saw only classical-tracker measurements of the training clip "
                 "frames and the timestamps of the withheld frames."]
    notes += list(prereg["disclosures"])
    m: dict = {"kind": "p6_run_manifest", "tool_version": TOOL_VERSION, "experiment_id": experiment_id, "label": label,
               "generated_utc": stamp, "preregistration": {"path": str(Path(prereg_path).resolve()), "sha256": prereg_sha},
               "video": prereg["video"], "status": "started", "evaluated": False}
    try:
        meas = stage_measure_sam(video, exp_dir, prereg, Path(sam_result)) if sam else stage_measure(video, exp_dir, prereg)
        m["measurement"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in meas.items()}
        sp_ = stage_split(meas["tracked_path"], exp_dir, first_withheld_frame=meas["withheld_frames"][0], notes=notes)
        m["split"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in sp_.items()}
        if spark_client_factory is not None:
            factory = spark_client_factory
        elif mode == "live":
            factory = live_client_factory(exp_dir / "inference" / "spark" / "api_evidence")
        else:
            factory = None
        spark = stage_spark(sp_["inputs_path"], exp_dir, client_factory=factory)
        m["spark"] = {"mode": mode if spark_client_factory is None else "injected", "status": spark["status"],
                      "artifact": {"path": str(spark["artifact_path"]), "sha256": spark["artifact_sha256"]},
                      "authority": ss.AUTHORITY}
        if prereg.get("physics", {}).get("model") == "projective":
            pr = stage_predict_projective(sp_["inputs_path"], exp_dir, notes, prereg["physics"])
        else:
            pr = stage_predict(sp_["inputs_path"], exp_dir, notes)
        m["predict"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in pr.items()}
        m["status"] = "prediction_frozen"
    except GateError as exc:
        m["status"] = "stopped_before_freeze"
        m["stop_reason"] = str(exc)
        _write_json(exp_dir / "run_manifest.json", m)
        raise
    _write_json(exp_dir / "run_manifest.json", m)
    return {"exp_dir": exp_dir, "manifest": m}


# ------------------------------------------------------------------ reveal (evaluation side; exactly once)


def withheld_reversals(ys: list[float]) -> int:
    """Downward-to-upward reversals (+y is down) in a sequence: tracker-measured bounce indications."""
    steps = [b - a for a, b in zip(ys, ys[1:])]
    return sum(1 for s0, s1 in zip(steps, steps[1:]) if s0 > 0 and s1 < 0)


def annotation_accuracy(annotations_path: Path, tracked: tr.TrajectorySet) -> dict:
    data = strict_json_loads(Path(annotations_path).read_bytes().decode("utf-8"))
    if data.get("kind") != "manual_annotations":
        raise VerifyError("annotations must be a manual_annotations file")
    obs = {o.frame_index: o for o in tracked.objects[0].observations}
    rows = [(a["frame_index"], obs[a["frame_index"]].x - a["x"], obs[a["frame_index"]].y - a["y"])
            for a in data["frames"] if a["frame_index"] in obs]
    if not rows:
        raise VerifyError("no annotated frame was tracked")
    stat = lambda v: {"rmse_px": math.sqrt(sum(e * e for e in v) / len(v)), "max_abs_px": max(abs(e) for e in v),  # noqa: E731
                      "mean_px": sum(v) / len(v), "n": len(v)}
    return {"status": "evaluated", "reference": "independent manual annotations", "annotator": data.get("annotator"),
            "method": data.get("method"), "annotations_sha256": _sha_file(annotations_path),
            "x": stat([r[1] for r in rows]), "y": stat([r[2] for r in rows]),
            "per_frame": [{"frame_index": f, "dx_px": dx, "dy_px": dy} for f, dx, dy in rows]}


def reveal(run_dir: Path, *, annotations: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "run_manifest.json").read_bytes())
    if m.get("status") != "prediction_frozen":
        raise VerifyError(f"run status is {m.get('status')!r}; only a frozen prediction can be revealed")
    if (run_dir / "reveal").exists():
        raise VerifyError("this run was already revealed (evaluation happens exactly once)")
    prereg, prereg_sha = load_preregistration(Path(m["preregistration"]["path"]))
    if prereg_sha != m["preregistration"]["sha256"]:
        raise VerifyError("the pre-registration changed after the run")
    pred = Path(m["predict"]["prediction_path"])
    before = _sha_file(pred)
    if before != m["predict"]["prediction_sha256"]:
        raise VerifyError("the frozen prediction changed")
    inputs, vault = Path(m["split"]["inputs_path"]), Path(m["split"]["vault_path"])
    if _sha_file(inputs) != m["split"]["inputs_sha256"] or _sha_file(vault) != m["split"]["vault_sha256"]:
        raise VerifyError("inference inputs or vault changed")
    artifact = strict_json_loads(pred.read_bytes().decode("utf-8"))
    training, _ = pd.load_inference_inputs(inputs)
    projective = prereg.get("physics", {}).get("model") == "projective"
    sam = prereg["prereg_version"] == PREREG_VERSION_SAM
    limitations = (LIMITATIONS_SAM if sam else LIMITATIONS_P7 if projective else LIMITATIONS)[prereg["video_provenance"]]
    reference = ("SAM-guided outline measurements of the withheld frames (NOT independently verified ground truth)" if sam
                 else "classical-tracker measurements of the withheld frames (NOT independently verified ground truth)")
    try:
        report = ev.evaluate(pred, pd.sha256_path_for(pred), vault, limitations=limitations)  # exactly once
    except ev.EvaluationError as exc:
        raise VerifyError(f"evaluate.{exc.check}: {exc}") from None
    chart = ov.build_chart_data(training, artifact, report.data)
    png = ov.render_png(chart)
    tracked = tr.load_trajectories(Path(m["measurement"]["tracked_path"]))
    accuracy = (annotation_accuracy(annotations, tracked) if annotations is not None else
                {"status": "not_available", "reason": "no independent reference measurements (manual annotations) "
                                                      "were supplied; tracker measurements are not ground truth"})
    if _sha_file(pred) != before:
        raise VerifyError("the prediction changed during the reveal")
    obj = report.data["objects"][0]
    interp = interpret(report.data)
    findings = {
        "kind": "p6_reveal_report", "label": m["label"], "experiment_id": m["experiment_id"],
        "prediction_sha256": before,
        "prediction_error_vs_withheld_tracker_measurements": {
            "reference": reference,
            "rmse_px": obj["baseline_comparison"]["rmse_px"], "ranking_by_rmse": obj["baseline_comparison"]["ranking_by_rmse"],
            "interpretation": interp["objects"][0]},
        "contact_timing": {"status": "not_evaluated", "reason": "no independent event annotations",
                           "predicted_withheld_contacts_t_s": [e["t_s"] for e in obj["predicted_events"]
                                                               if e["phase"] == "withheld"],
                           "tracker_measured_withheld_reversals": withheld_reversals(chart["withheld_truth"]["y"]),
                           "reversal_note": "downward-to-upward reversals in the withheld tracker measurements; a "
                                            "tracker-measured indication only, not an independent event annotation"},
        "tracking_accuracy": accuracy,
        "tracking_diagnostics": m["measurement"]["gate"],
        "fit_quality": m["predict"]["fit_gate"],
        "spark": m["spark"],
        "limitations": list(limitations),
    }
    rv = run_dir / "reveal"
    files = {"evaluation_report": rv / "evaluation_report.json", "overlay_png": rv / "overlay.png",
             "chart_data": rv / "chart_data.json", "reveal_report": rv / "reveal_report.json"}
    hashes = {"evaluation_report": _write_json(files["evaluation_report"], {**report.data, "label": m["label"]}),
              "chart_data": _write_json(files["chart_data"], chart),
              "reveal_report": _write_json(files["reveal_report"], findings)}
    ov.write_new(files["overlay_png"], png)
    hashes["overlay_png"] = _sha_bytes(png)
    summary = summary_text(findings, {k: str(v) for k, v in files.items()})
    ov.write_new(rv / "summary.md", summary.encode("utf-8"))
    hashes["summary"] = _sha_bytes(summary.encode("utf-8"))
    files["summary"] = rv / "summary.md"
    _write_json(rv / "reveal_manifest.json", {"kind": "p6_reveal_manifest", "evaluate_calls": 1, "api_requests_made": 0,
                                              "prediction_sha256_before": before, "prediction_sha256_after": _sha_file(pred),
                                              "outputs": {k: {"path": str(files[k]), "sha256": h} for k, h in hashes.items()}})
    return {"files": files, "hashes": hashes, "findings": findings}


def summary_text(f: dict, paths: dict) -> str:
    p = f["prediction_error_vs_withheld_tracker_measurements"]
    L = [f"# P6 predictive decompilation: {f['label']}", "", f"Experiment `{f['experiment_id']}`; frozen prediction "
         f"sha256 `{f['prediction_sha256']}`.", "", "## Prediction error against withheld TRACKER MEASUREMENTS", "",
         "These are classical-tracker measurements of the withheld frames, not independently verified ground truth.", "",
         "| model | Euclidean RMSE (px) |", "|---|---|"]
    L += [f"| {mm} | {p['rmse_px'][mm]!r} |" for mm in p["ranking_by_rmse"]]
    c = f["contact_timing"]
    L += ["", f"Ranking: {' < '.join(p['ranking_by_rmse'])}", "", "## Contact timing", "",
          f"- Status: {c['status']} ({c['reason']}).",
          f"- Predicted withheld contacts (s): {c['predicted_withheld_contacts_t_s']}",
          f"- Tracker-measured withheld reversals: {c['tracker_measured_withheld_reversals']} ({c['reversal_note']}).",
          "", "## Tracking", "", f"- Accuracy: {f['tracking_accuracy']['status']}"
          + (f" ({f['tracking_accuracy'].get('reason')})" if f["tracking_accuracy"].get("reason") else ""),
          f"- Diagnostics: {f['tracking_diagnostics']['status_counts']}; training gaps "
          f"{len(f['tracking_diagnostics']['training_gaps'])}; area {f['tracking_diagnostics']['area_px_range']} px",
          "", "## Fit quality (training only)", ""]
    for o in f["fit_quality"]["objects"]:
        L.append(f"- {o['object_id']}: {o['selected_model']}, adequacy {o['adequacy']}; program {o.get('program', {}).get('sha256')}")
    L += ["", "## Muse Spark (advisory; never changes the prediction)", "", f"- Status: {f['spark']['status']}; "
          f"artifact `{f['spark']['artifact']['path']}`", "", "## Limitations", ""] + [f"- {x}" for x in f["limitations"]]
    L += ["", "## Outputs", ""] + [f"- {k}: `{v}`" for k, v in paths.items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ CLI


def _box_arg(text: str) -> tuple[int, int, int, int]:
    try:
        vals = tuple(int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("a box is x0,y0,x1,y1 (ints)") from None
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("a box is x0,y0,x1,y1 (ints)")
    return vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="real_video_demo.py", description="P6 pre-registered real-video predictive "
                                "decompilation (OFFLINE unless a live Muse Spark call is pre-registered AND confirmed).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prereg", help="hash the video and write the pre-registration (never overwritten)")
    a.add_argument("--video", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--seed-box", type=_box_arg, required=True, help="x0,y0,x1,y1 around the ball in the first clip frame")
    a.add_argument("--start-frame", type=int, default=0)
    a.add_argument("--end-frame", type=int, default=None)
    a.add_argument("--stride", type=int, default=1)
    a.add_argument("--roi", type=_box_arg, default=None)
    a.add_argument("--exclude", type=_box_arg, action="append", default=[])
    a.add_argument("--max-training-gaps", type=int, default=DEFAULT_GATES["max_training_gaps"])
    a.add_argument("--max-area-rel-dev", type=float, default=DEFAULT_GATES["max_area_rel_dev"])
    a.add_argument("--centre-method", choices=tk.CENTRE_METHODS, default="colour_centroid",
                   help="P7: outline centre method (edge_ellipse recommended)")
    a.add_argument("--physics-model", choices=PHYSICS_MODELS, default="image_space",
                   help="P7: projective = perspective-aware fit and 3-D scene program")
    a.add_argument("--fit-noise", choices=("absolute", "relative"), default="absolute")
    a.add_argument("--hfov", type=float, default=66.0, help="P7: camera horizontal field-of-view prior (deg)")
    a.add_argument("--hfov-range", type=float, nargs=2, default=(50.0, 80.0), metavar=("LO", "HI"))
    a.add_argument("--max-withheld-gaps", type=int, default=0)
    a.add_argument("--provenance", choices=sorted(PROVENANCES), default="real")
    a.add_argument("--spark", choices=SPARK_MODES, default="off")
    a.add_argument("--note", default="")
    a.add_argument("--disclose", action="append", default=[])
    s_ = sub.add_parser("prereg-sam", help="pre-register a SAM 3.1 backend run (version 3; never overwritten)")
    s_.add_argument("--video", type=Path, required=True)
    s_.add_argument("--out", type=Path, required=True)
    s_.add_argument("--prompt-file", type=Path, required=True,
                    help="the plan JSON written by `sam_demo.py plan` (its 'prompt') or a bare prompt object")
    s_.add_argument("--sam-object", default="most_frames", help="'most_frames' or a SAM object id")
    s_.add_argument("--object-id", default="obj_01")
    s_.add_argument("--start-frame", type=int, default=0)
    s_.add_argument("--end-frame", type=int, default=None)
    s_.add_argument("--stride", type=int, default=1)
    s_.add_argument("--max-training-gaps", type=int, default=SAM_DEFAULT_GATES["max_training_gaps"])
    s_.add_argument("--max-withheld-gaps", type=int, default=SAM_DEFAULT_GATES["max_withheld_gaps"])
    s_.add_argument("--max-diameter-rel-dev", type=float, default=SAM_DEFAULT_GATES["max_diameter_rel_dev"])
    s_.add_argument("--outline-model", choices=("ellipse", "stadium", "circle"), default="ellipse")
    s_.add_argument("--min-confidence", type=float, default=0.5)
    s_.add_argument("--physics-model", choices=PHYSICS_MODELS, default="image_space")
    s_.add_argument("--fit-noise", choices=("absolute", "relative"), default="absolute")
    s_.add_argument("--hfov", type=float, default=66.0)
    s_.add_argument("--hfov-range", type=float, nargs=2, default=(50.0, 80.0), metavar=("LO", "HI"))
    s_.add_argument("--provenance", choices=sorted(PROVENANCES), default="real")
    s_.add_argument("--spark", choices=SPARK_MODES, default="off")
    s_.add_argument("--note", default="")
    s_.add_argument("--disclose", action="append", default=[])
    r = sub.add_parser("run", help="measure, split, (Spark), freeze the prediction; does NOT evaluate")
    r.add_argument("--prereg", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    r.add_argument("--confirm-live-call", action="store_true")
    r.add_argument("--sam-result", type=Path, default=None, help="SAM tracking result JSON (version-3 runs only)")
    v = sub.add_parser("reveal", help="evaluate a frozen run exactly once")
    v.add_argument("--run-dir", type=Path, required=True)
    v.add_argument("--annotations", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "prereg":
            if a.out.exists() or pd.sha256_path_for(a.out).exists():
                raise UsageError(f"{a.out} already exists (pre-registrations are never overwritten)")
            data = make_preregistration(a.video, seed_box=a.seed_box, start_frame=a.start_frame, end_frame=a.end_frame,
                                        stride=a.stride, roi=a.roi, exclude_boxes=a.exclude,
                                        gates={"max_training_gaps": a.max_training_gaps,
                                               "max_area_rel_dev": a.max_area_rel_dev},
                                        provenance=a.provenance, spark_mode=a.spark, notes=a.note, disclosures=a.disclose,
                                        centre_method=a.centre_method, physics_model=a.physics_model,
                                        fit_noise=a.fit_noise, max_withheld_gaps=a.max_withheld_gaps,
                                        camera_prior=({"hfov_deg": a.hfov, "hfov_range_deg": list(a.hfov_range)}
                                                      if a.physics_model == "projective" else None))
            digest = _write_with_record(a.out, data)
            print(f"pre-registration {a.out} sha256 {digest}; video sha256 {data['video']['sha256']}")
        elif a.cmd == "prereg-sam":
            if a.out.exists() or pd.sha256_path_for(a.out).exists():
                raise UsageError(f"{a.out} already exists (pre-registrations are never overwritten)")
            pdata = strict_json_loads(a.prompt_file.read_bytes().decode("utf-8"))
            prompt = pdata.get("prompt", pdata) if isinstance(pdata, dict) else pdata
            sel = a.sam_object if a.sam_object in SAM_OBJECT_RULES else {"id": str(a.sam_object)}
            data = make_preregistration_sam(
                a.video, prompt=prompt, object_selection=sel, start_frame=a.start_frame, end_frame=a.end_frame,
                stride=a.stride, object_id=a.object_id,
                gates={"max_training_gaps": a.max_training_gaps, "max_withheld_gaps": a.max_withheld_gaps,
                       "max_diameter_rel_dev": a.max_diameter_rel_dev},
                provenance=a.provenance, spark_mode=a.spark, notes=a.note, disclosures=a.disclose,
                outline_model=a.outline_model, physics_model=a.physics_model, fit_noise=a.fit_noise,
                min_confidence=a.min_confidence,
                camera_prior=({"hfov_deg": a.hfov, "hfov_range_deg": list(a.hfov_range)}
                              if a.physics_model == "projective" else None))
            digest = _write_with_record(a.out, data)
            print(f"pre-registration {a.out} sha256 {digest}; video sha256 {data['video']['sha256']}")
        elif a.cmd == "run":
            r = run(a.prereg, a.output_dir, confirm_live_call=a.confirm_live_call, sam_result=a.sam_result)
            mm = r["manifest"]
            print(f"prediction frozen (NOT evaluated): {r['exp_dir']}\n  prediction sha256 {mm['predict']['prediction_sha256']}"
                  f"\n  Muse Spark: {mm['spark']['status']}")
        else:
            out = reveal(a.run_dir, annotations=a.annotations)
            for k, v in out["files"].items():
                print(f"  {k}: {v}  sha256 {out['hashes'][k]}")
    except UsageError as exc:
        print(f"[USAGE] {exc}")
        return EXIT_USAGE
    except (GateError, VerifyError) as exc:
        print(f"[STOP] {exc}")
        return EXIT_GATE
    except OSError as exc:
        print(f"[FAIL] output not written ({type(exc).__name__}: {exc})")
        return EXIT_WRITE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
