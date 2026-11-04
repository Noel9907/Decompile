"""P5: VIDEO-IN predictive decompilation on a SYNTHETIC VIDEO (OFFLINE; no API request).

    generator side   decompile.synth scene -> decompile.render MP4            (ground truth stays here)
    measurement      MP4 -> A1 probe/validate -> A2 frame export -> decompile.tracker
                     -> CLASSICAL_TRACKER TrajectorySet (synthetic = true: the video is synthetic)
                     -> tracking quality gate (tracker diagnostics only; no ground truth)
    split            decompile.holdout 70/30 on the tracked frames -> inference_inputs.json (training +
                     withheld timestamp schedule) | evaluation_only/evaluation_vault.json (tracked withheld
                     positions + generator contact times)
    predict          reads ONLY inference_inputs.json -> training-fit gate (analyze_kinematics +
                     program_from_kinematics) -> decompile.predict.predict: frozen artifact + .sha256
    reveal           SHA-256 check -> decompile.evaluate.evaluate (exactly once; the only vault reader)
                     -> decompile.overlay PNG -> tracker accuracy against generator positions (after the
                     freeze) -> SHA-256 re-check -> report

Each stage function receives only the paths it is allowed to read: the measurement stage gets
the MP4 alone; the predict stage gets the inference-inputs file alone. If the tracking or the
training-fit gate fails, the run STOPS before any prediction is written (exit 1); gate
thresholds are fixed constants, never adjusted after looking at results.

Every file is written atomically and never overwritten, into a fresh directory
runs/experiments/<experiment_id>/. Exit codes: 0 done; 1 a gate or verification failed (no
prediction frozen, or no reveal written); 2 usage error; 3 an output could not be written.

Labelling: SYNTHETIC VIDEO. This is not real-video validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decompile import evaluate as ev  # noqa: E402
from decompile import frames as fr  # noqa: E402
from decompile import holdout as ho  # noqa: E402
from decompile import ingest  # noqa: E402
from decompile import kinematic_fit as kf  # noqa: E402
from decompile import overlay as ov  # noqa: E402
from decompile import predict as pd  # noqa: E402
from decompile import render as rd  # noqa: E402
from decompile import scene_program as sp  # noqa: E402
from decompile import synth  # noqa: E402
from decompile import tracker as tk  # noqa: E402
from decompile import trajectories as tr  # noqa: E402
from decompile.meta_api import strict_json_loads  # noqa: E402


def _load_p4_tool():
    spec = importlib.util.spec_from_file_location("predictive_demo_p4", ROOT / "tools" / "predictive_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interpret = _load_p4_tool().interpret  # P4's fixed comparison statements (no rounding)

TOOL_VERSION = "tools.video_predictive_demo/1"
LABEL = "SYNTHETIC VIDEO"
TRAIN_FRACTION = Fraction(7, 10)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "experiments"
EXIT_OK, EXIT_GATE, EXIT_USAGE, EXIT_WRITE = 0, 1, 2, 3

# Tracking quality gate (fixed in advance; tracker diagnostics only, never ground truth).
GATE_MAX_AREA_REL_DEV = 0.25  # every detection's mask area within 25% of the median area
GATE_FILL_RATIO = (0.6, 0.95)  # mask area / bounding-box area (a disc: about pi/4 = 0.785)

P2_DISCLOSURE = (
    "P2 development disclosure: before P3, while checking that the 70% training split supports a scene program, "
    "the program's simulated contact times through frame 49 (0.9, 1.44003, 1.60205 s) were printed. No withheld "
    "positions were used and nothing was tuned or changed afterwards; this evaluation is therefore not fully blind.")
P4_DISCLOSURE = (
    "P4 disclosure: the withheld generator trajectory of this same pre-registered scene was revealed in the P4 "
    "evaluation before P5 was built. The P5 video is a new rendering of that scene, so this experiment is NOT "
    "blind with respect to the scene's outcome.")
P5_NOTE = (
    "P5 SYNTHETIC VIDEO experiment: the predictor saw only classical-tracker measurements of the first 70% of the "
    "frames of a rendered MP4, plus the exact timestamps of the withheld frames. Renderer, tracker and gate "
    "settings were fixed before this run; they were developed on test-only scenes, and nothing is tuned after the "
    "reveal.")
P5_LIMITATIONS = (
    "SYNTHETIC VIDEO: a rendered MP4 of one scene from decompile.synth (the P2 pre-registered scene) with "
    "antialiasing, Gaussian blur, Gaussian pixel noise and H.264 (yuv420p) compression. This is NOT real-video "
    "validation and says nothing about real footage (lighting, motion blur, rolling shutter, lens distortion, "
    "camera motion, occlusion, clutter).",
    "Withheld 'truth' is the classical tracker's measurement of the withheld frames, not generator ground truth: "
    "the errors include tracking measurement noise. Tracker accuracy against generator positions is reported "
    "separately.",
    "Classical single-object colour tracker with a colour prompt fixed in advance: one clearly distinguishable ball "
    "on a neutral background. Not SAM 3.1 and not a Meta integration.",
    "Scene program v1 constraints: one body, exactly one contact inside the training frames, uniform acceleration, "
    "one axis-aligned INFERRED contact boundary, restitution in [0, 1], no drag. Scenes outside these are refused.",
    "Contact matching is chronological one-to-one without a time gate; timing errors are reported as measured.",
    "Point predictions only: parameter uncertainty is not propagated to prediction bands.",
    "Units are image pixels and seconds; nothing is calibrated to metres.",
)


class GateError(Exception):
    """A pre-freeze gate failed: no prediction was frozen."""

    def __init__(self, gate: str, problems: list[str]) -> None:
        super().__init__(f"{gate}: " + "; ".join(problems[:8]))
        self.gate, self.problems = gate, list(problems)


class VerifyError(Exception):
    """A reveal-time integrity check failed: nothing was evaluated or written."""


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    return _sha_bytes(Path(p).read_bytes())


def _write_json(path: Path, obj) -> str:
    raw = pd.canonical_bytes(obj)
    ov.write_new(path, raw)
    return _sha_bytes(raw)


# ------------------------------------------------------------------ generator side


def stage_render(exp_dir: Path, spec: synth.SceneSpec, settings: rd.RenderSettings) -> dict:
    """Generate the scene numerically and render it. Ground truth is written only to generator_side/."""
    gen = exp_dir / "generator_side"
    scene = synth.generate(spec)
    boundaries = [rd.Boundary(p.axis, p.position_px, p.keep_side) for p in spec.planes]
    video = exp_dir / "video" / "synthetic_video.mp4"
    result = rd.render_video(scene.trajectories, boundaries, video, settings)
    paths = {"source_trajectories": gen / "source_trajectories.json", "ground_truth": gen / "ground_truth.json",
             "render_manifest": gen / "render_manifest.json"}
    hashes = {"source_trajectories": _write_json(paths["source_trajectories"], scene.trajectories.to_json_dict()),
              "ground_truth": _write_json(paths["ground_truth"], synth.ground_truth_to_json(scene.ground_truth)),
              "render_manifest": _write_json(paths["render_manifest"], result.manifest)}
    return {"video": video, "video_sha256": result.video_sha256, "paths": paths, "hashes": hashes,
            "source_trajectory_sha256": result.manifest["source"]["trajectory_sha256"]}


# ------------------------------------------------------------------ measurement (video only)


def tracking_gate(info: ingest.VideoInfo, table: fr.FrameTable, result: tk.TrackingResult) -> dict:
    """Pre-freeze checks from the video and tracker diagnostics alone."""
    problems = []
    n = len(table.frames)
    if info.decoded_frame_count != n or len(result.detections) != n:
        problems.append(f"frame counts disagree: decoded {info.decoded_frame_count}, frame table {n}, "
                        f"tracked {len(result.detections)}")
    if [d.frame_index for d in result.detections] != [f.index for f in table.frames] or \
            [d.t for d in result.detections] != [f.t for f in table.frames]:
        problems.append("tracked frame indices/timestamps differ from the A2 frame table")
    if table.synthesized_timestamp_count:
        problems.append(f"{table.synthesized_timestamp_count} frame timestamp(s) were synthesized by FFmpeg")
    problems += [f"frame {d.frame_index}: {d.detection.status}" for d in result.failures]
    det = [d.detection for d in result.detections if d.detection.status == tk.DETECTED]
    median_area = statistics.median(d.area_px for d in det) if det else None
    if det:
        for fd in result.detections:
            d = fd.detection
            if d.status != tk.DETECTED:
                continue
            if abs(d.area_px - median_area) > GATE_MAX_AREA_REL_DEV * median_area:
                problems.append(f"frame {fd.frame_index}: area {d.area_px} px deviates > "
                                f"{GATE_MAX_AREA_REL_DEV:.0%} from the median {median_area}")
            if not GATE_FILL_RATIO[0] <= d.fill_ratio <= GATE_FILL_RATIO[1]:
                problems.append(f"frame {fd.frame_index}: fill ratio {d.fill_ratio:.3f} outside {GATE_FILL_RATIO}")
    return {"passed": not problems, "problems": problems,
            "thresholds": {"max_area_rel_dev": GATE_MAX_AREA_REL_DEV, "fill_ratio": list(GATE_FILL_RATIO)},
            "coverage": {"frames": n, "detected": len(det)}, "status_counts": result.status_counts(),
            "median_area_px": median_area,
            "area_px_range": [min(d.area_px for d in det), max(d.area_px for d in det)] if det else None,
            "fill_ratio_range": [min(d.fill_ratio for d in det), max(d.fill_ratio for d in det)] if det else None}


def stage_measure(video: Path, exp_dir: Path, config: tk.TrackerConfig, *, synthetic_video: bool) -> dict:
    """MP4 -> validated frames -> tracker -> CLASSICAL_TRACKER TrajectorySet. Reads nothing but the video."""
    m = exp_dir / "measurement"
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    _write_json(m / "probe.json", {"kind": "video_probe", "info": asdict(info), "rejection_reasons": reasons})
    if reasons:
        raise GateError("video_validation", reasons)
    frames_dir = m / "frames"
    table = fr.extract_frames(video, frames_dir)
    if table.source_sha256 != info.sha256:
        raise GateError("frame_extraction", ["the frame export does not belong to the probed video"])
    result = tk.track_extracted(frames_dir, config, expected_source_sha256=info.sha256)
    paths = {"probe": m / "probe.json", "frames_manifest": frames_dir / fr.MANIFEST_NAME,
             "tracking_report": m / "tracking_report.json", "tracking_gate": m / "tracking_gate.json",
             "tracked_trajectories": m / "tracked_trajectories.json"}
    hashes = {"tracking_report": _write_json(paths["tracking_report"], result.to_json_dict())}
    gate = tracking_gate(info, table, result)
    hashes["tracking_gate"] = _write_json(paths["tracking_gate"], gate)
    if not gate["passed"]:
        raise GateError("tracking", gate["problems"])
    traj = result.to_trajectory_set(synthetic=synthetic_video)
    hashes["tracked_trajectories"] = _write_json(paths["tracked_trajectories"], traj.to_json_dict())
    hashes["frames_manifest"] = _sha_file(paths["frames_manifest"])
    hashes["probe"] = _sha_file(paths["probe"])
    times = table.times_s()
    return {"paths": paths, "hashes": hashes, "gate": gate, "tracked_sha256": ho.sha256_canonical(traj.to_json_dict()),
            "probe": {"sha256": info.sha256, "container": info.container, "codec": info.codec,
                      "width": info.width, "height": info.height, "duration_s": info.duration_s,
                      "nominal_fps": info.nominal_fps, "avg_fps": info.avg_fps,
                      "reported_frame_count": info.reported_frame_count,
                      "decoded_frame_count": info.decoded_frame_count, "frame_count_status": info.frame_count_status,
                      "warnings": info.warnings, "accepted": True},
            "extraction": {"frame_count": len(table.frames), "time_base": str(table.time_base),
                           "first_pts": table.first_pts, "last_t": str(table.frames[-1].t),
                           "last_t_s": float(times[-1]), "rotation_deg": table.rotation_deg,
                           "synthesized_timestamp_count": table.synthesized_timestamp_count}}


# ------------------------------------------------------------------ split (measurement + evaluation side)


def stage_split(tracked_path: Path, ground_truth_path: Path, exp_dir: Path, notes: list[str]) -> dict:
    """70/30 split of the tracked frames; inference inputs and the evaluation-only vault go to different places."""
    traj = tr.load_trajectories(tracked_path)
    split = ho.split_trajectories(traj, TRAIN_FRACTION)
    inputs_path = exp_dir / "inference" / "inference_inputs.json"
    inputs_sha = _write_json(inputs_path, ho.inference_inputs_to_json(split.inference))
    gt = strict_json_loads(ground_truth_path.read_bytes().decode("utf-8"))
    if len(traj.objects) != 1:
        raise GateError("split", ["exactly one tracked object is supported"])
    oid = traj.objects[0].object_id  # the single tracked object is the single generated body
    contacts = [{"object_id": oid, "t_s": c["t_s"], "axis": c["axis"], "plane_index": c["plane_index"]}
                for c in gt["contacts"]]
    vault_path = exp_dir / "evaluation_only" / "evaluation_vault.json"
    vault_sha = ev.write_evaluation_vault(vault_path, split, ground_truth_contacts=contacts, notes=notes)
    inf = split.inference
    return {"inputs_path": inputs_path, "inputs_sha256": inputs_sha, "vault_path": vault_path, "vault_sha256": vault_sha,
            "training_sha256": inf.training_sha256, "schedule_sha256": inf.schedule_sha256,
            "split": {"train_fraction": str(inf.train_fraction), "n_frames": inf.n_frames,
                      "n_train_frames": inf.n_train_frames, "first_withheld_frame": inf.first_withheld_frame,
                      "n_withheld_frames": len(inf.schedule)}}


# ------------------------------------------------------------------ predict (inference inputs only)


def fit_gate(training: tr.TrajectorySet) -> dict:
    kin = kf.analyze_kinematics(training)
    objects, problems = [], []
    for o in kin.objects:
        entry = {"object_id": o.object_id, "verdicts": {k: v.verdict for k, v in sorted(o.verdicts.items())},
                 "adequacy": o.adequacy, "adequacy_reasons": list(o.adequacy_reasons),
                 "selected_model": o.selected_model, "noise_mode": kin.noise.mode}
        try:
            prog = sp.program_from_kinematics(o)
            pl = prog.planes[0]
            entry["program"] = {"sha256": sp.program_sha256(prog), "acceleration_px_s2": [prog.acceleration.ax,
                                                                                         prog.acceleration.ay],
                                "plane": {"axis": pl.axis, "position_px": pl.position_px, "keep_side": pl.keep_side,
                                          "restitution": pl.restitution},
                                "t_contact_s": float(o.contact.t_contact_s)}
        except sp.ProgramUnavailable as exc:
            problems += [f"{o.object_id}: {r}" for r in exc.reasons]
        if o.adequacy == "inadequate":
            problems.append(f"{o.object_id}: training fit inadequate")
        objects.append(entry)
    return {"passed": not problems, "problems": problems, "objects": objects}


def stage_predict(inputs_path: Path, exp_dir: Path, notes: list[str]) -> dict:
    """Reads ONLY the inference-inputs file; gates the training fit; freezes the prediction."""
    training, schedule = pd.load_inference_inputs(inputs_path)
    gate = fit_gate(training)
    gate_path = exp_dir / "inference" / "fit_gate.json"
    gate_sha = _write_json(gate_path, gate)
    if not gate["passed"]:
        raise GateError("training_fit", gate["problems"])
    try:
        r = pd.predict(training, schedule, output_path=exp_dir / "inference" / "prediction.json", experiment_notes=notes)
    except pd.PredictionError as exc:
        raise GateError("prediction", [str(exc)]) from None
    return {"prediction_path": r.artifact_path, "sha256_path": r.sha256_path, "prediction_sha256": r.sha256,
            "fit_gate": gate, "fit_gate_path": gate_path, "fit_gate_sha256": gate_sha,
            "program_sha256": {o["object_id"]: o["program_sha256"] for o in r.artifact["objects"]}}


# ------------------------------------------------------------------ reveal


def _axis_stats(errs: list[float]) -> dict:
    return {"rmse_px": math.sqrt(sum(e * e for e in errs) / len(errs)), "max_abs_px": max(abs(e) for e in errs),
            "mean_px": sum(errs) / len(errs), "n": len(errs)}


def tracking_accuracy(tracked: tr.TrajectorySet, generator: tr.TrajectorySet, first_withheld: int) -> dict:
    """Tracker measurement error against generator positions (evaluation side, after the freeze)."""
    obj, gen = tracked.objects[0], generator.objects[0]
    g = {o.frame_index: o for o in gen.observations}
    rows = []
    for o in obj.observations:
        ref = g[o.frame_index]
        if ref.t != o.t:
            raise VerifyError(f"frame {o.frame_index}: tracked t {o.t} != generator t {ref.t}")
        rows.append((o.frame_index, o.x - ref.x, o.y - ref.y))
    out = {}
    for name, sel in (("all", rows), ("training", [r for r in rows if r[0] < first_withheld]),
                      ("withheld", [r for r in rows if r[0] >= first_withheld])):
        out[name] = {"x": _axis_stats([r[1] for r in sel]), "y": _axis_stats([r[2] for r in sel]),
                     "euclidean_rmse_px": math.sqrt(sum(r[1] ** 2 + r[2] ** 2 for r in sel) / len(sel))}
    return {"kind": "tracking_accuracy", "reference": "generator positions (decompile.synth, rounded to 0.01 px)",
            "error": "tracked - generator (px)", "per_frame": [{"frame_index": f, "dx_px": dx, "dy_px": dy}
                                                               for f, dx, dy in rows], **out}


def stage_reveal(exp_dir: Path, *, prediction: Path, prediction_sha_path: Path, expected_prediction_sha256: str,
                 inputs_path: Path, vault_path: Path, tracked_path: Path, generator_trajectories_path: Path,
                 first_withheld: int) -> dict:
    before = _sha_file(prediction)
    if before != expected_prediction_sha256:
        raise VerifyError(f"prediction SHA-256 {before} != frozen {expected_prediction_sha256}")
    inputs_before, vault_before = _sha_file(inputs_path), _sha_file(vault_path)
    artifact = strict_json_loads(prediction.read_bytes().decode("utf-8"))
    training, _ = pd.load_inference_inputs(inputs_path)
    try:
        report = ev.evaluate(prediction, prediction_sha_path, vault_path, limitations=P5_LIMITATIONS)  # once
    except ev.EvaluationError as exc:
        raise VerifyError(f"evaluate.{exc.check}: {exc}") from None
    chart = ov.build_chart_data(training, artifact, report.data)
    png = ov.render_png(chart)
    accuracy = tracking_accuracy(tr.load_trajectories(tracked_path), tr.load_trajectories(generator_trajectories_path),
                                 first_withheld)
    after = _sha_file(prediction)
    if after != before or _sha_file(inputs_path) != inputs_before or _sha_file(vault_path) != vault_before:
        raise VerifyError("an input file changed during the reveal")
    interp = interpret(report.data)
    rv = exp_dir / "reveal"
    files = {"evaluation_report": rv / "evaluation_report.json", "overlay_png": rv / "overlay.png",
             "chart_data": rv / "chart_data.json", "tracking_accuracy": rv / "tracking_accuracy.json"}
    hashes = {"evaluation_report": _write_json(files["evaluation_report"], {**report.data, "label": LABEL,
                                                                            "interpretation": interp}),
              "chart_data": _write_json(files["chart_data"], chart),
              "tracking_accuracy": _write_json(files["tracking_accuracy"], accuracy)}
    ov.write_new(files["overlay_png"], png)
    hashes["overlay_png"] = _sha_bytes(png)
    return {"files": files, "hashes": hashes, "report": report.data, "interpretation": interp, "accuracy": accuracy,
            "prediction_sha256_before": before, "prediction_sha256_after": after, "evaluate_calls": 1}


# ------------------------------------------------------------------ summary and orchestration


def summary_text(m: dict) -> str:
    L = [f"# P5 predictive decompilation from video: {LABEL}", "",
         f"Experiment `{m['experiment_id']}`. **{LABEL}: not real-video validation.**", "",
         "## Video and frames", "",
         f"- Video `{m['video']['path']}` sha256 `{m['video']['sha256']}` (rendered by decompile.render)"]
    p, x = m["measurement"]["probe"], m["measurement"]["extraction"]
    L += [f"- A1 probe: {p['container']} / {p['codec']}, {p['width']}x{p['height']}, nominal {p['nominal_fps']} fps, "
          f"duration {p['duration_s']} s, decoded {p['decoded_frame_count']} frames (reported "
          f"{p['reported_frame_count']}, {p['frame_count_status']}), accepted",
          f"- A2 extraction: {x['frame_count']} frames, time base {x['time_base']}, last t = {x['last_t']} s, "
          f"synthesized timestamps {x['synthesized_timestamp_count']}", "", "## Tracking", ""]
    g = m["measurement"]["gate"]
    L += [f"- Status counts: {g['status_counts']}; mask area {g['area_px_range']} px (median {g['median_area_px']}); "
          f"fill ratio {g['fill_ratio_range']}; gate {'PASSED' if g['passed'] else 'FAILED'}"]
    if "reveal" in m:
        a = m["reveal"]["accuracy"]
        L += ["- Accuracy against generator positions (computed after the freeze):", "",
              "| frames | n | x RMSE | x max abs | x mean | y RMSE | y max abs | y mean |", "|---|---|---|---|---|---|---|---|"]
        for k in ("all", "training", "withheld"):
            ax, ay = a[k]["x"], a[k]["y"]
            L.append(f"| {k} | {ax['n']} | {ax['rmse_px']:.4f} | {ax['max_abs_px']:.4f} | {ax['mean_px']:+.4f} | "
                     f"{ay['rmse_px']:.4f} | {ay['max_abs_px']:.4f} | {ay['mean_px']:+.4f} |")
    s = m["split"]["split"]
    L += ["", "## Split and training fit", "",
          f"- {s['n_train_frames']} training / {s['n_withheld_frames']} withheld frames (fraction {s['train_fraction']}, "
          f"first withheld frame {s['first_withheld_frame']})",
          f"- Training sha256 `{m['split']['training_sha256']}`; schedule sha256 `{m['split']['schedule_sha256']}`"]
    if "predict" in m:
        for o in m["predict"]["fit_gate"]["objects"]:
            pr = o.get("program", {})
            L += [f"- {o['object_id']}: selected {o['selected_model']}, adequacy {o['adequacy']} (noise {o['noise_mode']}); "
                  f"verdicts {o['verdicts']}",
                  f"- Program `{pr.get('sha256')}`: a = {pr.get('acceleration_px_s2')} px/s^2, plane {pr.get('plane')}, "
                  f"fitted contact at t = {pr.get('t_contact_s')} s",
                  f"- Frozen prediction `{m['predict']['prediction_path']}` sha256 `{m['predict']['prediction_sha256']}`"]
    if "reveal" in m:
        r = m["reveal"]
        L += [f"- Prediction sha256 before/after the reveal: `{r['prediction_sha256_before']}` / "
              f"`{r['prediction_sha256_after']}`", "", "## Withheld prediction accuracy (truth = tracker measurement)", ""]
        for o, rep in zip(r["interpretation"]["objects"], r["report"]["objects"]):
            L += ["| model | x RMSE | y RMSE | Euclidean RMSE | mean | max |", "|---|---|---|---|---|---|"]
            for mm in o["ranking_by_rmse"]:
                q = rep["withheld_accuracy"][mm]
                L.append(f"| {mm} | {q['rmse_x_px']!r} | {q['rmse_y_px']!r} | {q['rmse_px']!r} | "
                         f"{q['mean_error_px']!r} | {q['max_error_px']!r} |")
            L += ["", f"Ranking: {' < '.join(o['ranking_by_rmse'])}"]
            for mm, won in o["beats"].items():
                L.append(f"- scene_program {'BEATS' if won else 'DOES NOT BEAT'} {mm}")
            c = o["contacts"]
            for pm in c.get("matched", []):
                L.append(f"- Withheld contact: predicted {pm['predicted_t_s']!r} s, actual (generator) "
                         f"{pm['actual_t_s']!r} s, error {pm['error_s']!r} s")
            L.append(f"- Missed {c.get('missed_actual_t_s')}; false {c.get('false_predicted_t_s')}; training contact "
                     f"excluded {c.get('excluded_training_contacts_t_s')}")
        L += ["", "## Disclosures", ""] + [f"- {d}" for d in r["report"]["disclosures"]]
        L += ["", "## Limitations", ""] + [f"- {d}" for d in r["report"]["limitations"]]
    return "\n".join(L) + "\n"


def run(output_dir: Path = DEFAULT_OUTPUT_DIR, *, spec: synth.SceneSpec = synth.PREREGISTERED_DEMO,
        settings: rd.RenderSettings = rd.RenderSettings(), config: tk.TrackerConfig = tk.TrackerConfig(),
        notes: list[str] | None = None, experiment_id: str | None = None) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_id = experiment_id or f"p5_synthetic_video_{stamp}_{uuid.uuid4().hex[:8]}"
    exp_dir = Path(output_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=False)
    notes = list(notes) if notes is not None else [P5_NOTE, P2_DISCLOSURE, P4_DISCLOSURE]
    m: dict = {"kind": "p5_run_manifest", "tool_version": TOOL_VERSION, "label": LABEL, "experiment_id": experiment_id,
               "generated_utc": stamp, "api_requests_made": 0, "status": "started", "experiment_notes": notes,
               "versions": {"render": rd.RENDER_VERSION, "tracker": tk.TRACKER_VERSION, "kinematic_fit": kf.METHOD_VERSION,
                            "scene_program": sp.PROGRAM_VERSION, "predictor": pd.PREDICTOR_VERSION,
                            "evaluator": ev.EVALUATOR_VERSION},
               "tracker_config": asdict(config)}
    try:
        g = stage_render(exp_dir, spec, settings)
        m["video"] = {"path": str(g["video"]), "sha256": g["video_sha256"]}
        m["generator_side"] = {k: {"path": str(p), "sha256": g["hashes"][k]} for k, p in g["paths"].items()}
        m["generator_side"]["source_trajectory_canonical_sha256"] = g["source_trajectory_sha256"]
        meas = stage_measure(g["video"], exp_dir, config, synthetic_video=True)
        m["measurement"] = {k: v for k, v in meas.items() if k != "paths"}
        m["measurement"]["paths"] = {k: str(p) for k, p in meas["paths"].items()}
        sp_ = stage_split(meas["paths"]["tracked_trajectories"], g["paths"]["ground_truth"], exp_dir, notes)
        m["split"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in sp_.items()}
        pr = stage_predict(sp_["inputs_path"], exp_dir, notes)
        m["predict"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in pr.items()}
        m["status"] = "prediction_frozen"
        rv = stage_reveal(exp_dir, prediction=pr["prediction_path"], prediction_sha_path=pr["sha256_path"],
                          expected_prediction_sha256=pr["prediction_sha256"], inputs_path=sp_["inputs_path"],
                          vault_path=sp_["vault_path"], tracked_path=meas["paths"]["tracked_trajectories"],
                          generator_trajectories_path=g["paths"]["source_trajectories"],
                          first_withheld=sp_["split"]["first_withheld_frame"])
        m["reveal"] = {"files": {k: {"path": str(p), "sha256": rv["hashes"][k]} for k, p in rv["files"].items()},
                       "prediction_sha256_before": rv["prediction_sha256_before"],
                       "prediction_sha256_after": rv["prediction_sha256_after"], "evaluate_calls": 1,
                       "accuracy": {k: rv["accuracy"][k] for k in ("all", "training", "withheld")},
                       "interpretation": rv["interpretation"]}
        m["status"] = "complete"
        summary = summary_text({**m, "reveal": {**m["reveal"], "report": rv["report"], "accuracy": rv["accuracy"]}})
        ov.write_new(exp_dir / "reveal" / "summary.md", summary.encode("utf-8"))
        m["reveal"]["files"]["summary"] = {"path": str(exp_dir / "reveal" / "summary.md"),
                                           "sha256": _sha_bytes(summary.encode("utf-8"))}
    except (GateError, VerifyError) as exc:
        m["status"] = "stopped_before_freeze" if m["status"] == "started" else "stopped_after_freeze"
        m["stop_reason"] = str(exc)
        _write_json(exp_dir / "run_manifest.json", m)
        raise
    _write_json(exp_dir / "run_manifest.json", m)
    return {"exp_dir": exp_dir, "manifest": m}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video_predictive_demo.py",
                                description="OFFLINE P5: render a SYNTHETIC VIDEO, track it, freeze a prediction from "
                                            "the first 70%, then evaluate it once. No API request is made.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="parent of the experiment directory")
    p.add_argument("--seed", type=int, default=rd.RenderSettings().seed, help="render noise seed (fix before running)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.seed < 0:
        print("[FAIL] --seed must be >= 0")
        return EXIT_USAGE
    print(f"P5 video-in predictive decompilation ({LABEL}; OFFLINE: no API request)")
    try:
        r = run(a.output_dir, settings=replace(rd.RenderSettings(), seed=a.seed))
    except (GateError, VerifyError) as exc:
        print(f"[STOP] {exc}")
        return EXIT_GATE
    except OSError as exc:
        print(f"[FAIL] output not written ({type(exc).__name__}: {exc})")
        return EXIT_WRITE
    m = r["manifest"]
    print(f"  experiment: {r['exp_dir']}")
    print(f"  prediction sha256: {m['predict']['prediction_sha256']}")
    for o in m["reveal"]["interpretation"]["objects"]:
        print(f"  {o['object_id']}: scene_program RMSE {o['scene_program_rmse_px']!r} px; best baseline "
              f"{o['best_baseline']} {o['best_baseline_rmse_px']!r} px")
    for k, v in m["reveal"]["files"].items():
        print(f"  {k}: {v['path']}  sha256 {v['sha256']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
