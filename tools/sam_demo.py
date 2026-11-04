#!/usr/bin/env python3
"""Decompile x SAM 3.1: video -> segmentation/tracking -> geometric measurement -> physics -> simulation.

Three subcommands, run in this order:

  plan         OFFLINE. Probe the local MP4 (exact frame table from ffprobe), build and validate the SAM prompt
               (one concept noun phrase, OR box/point object prompts on given source frames), and report what a
               live call would process and cost (documented price; nothing is sent). Writes plan.json.
  track        The ONLY live SAM call in Decompile, and only with --confirm-live-call. Uploads the video BYTES
               (POST /v1/files, purpose=user_data; an existing file id for the same SHA-256 is reused), streams
               the Responses API, decodes with the official meta_sam_parser, validates every frame index / box /
               mask against the exact frame table, and writes sam_tracking.json (+ raw stream and API evidence).
               Without the flag it prints the plan and makes NO request.
  reconstruct  OFFLINE. From the video and sam_tracking.json: per-frame outline measurement guided by the SAM
               mask (decompile.sam_measure), trajectory with exact PTS and per-frame sigma, event/kinematic
               analysis, the executable scene program (when the existing gates allow), deterministic simulation,
               an optional counterfactual (changed restitution / gravity), and judge-readable outputs:
               overlay.mp4, contact_sheet.png, trajectory.png, scene_program.json, report.json, report.md.

reconstruct FITS the program to the same observations it is compared with: reproducing them is a consistency
check of the physical model, not a prediction. Held-out prediction against baselines is done by
tools/real_video_demo.py (prereg-sam -> run --sam-result -> reveal).

Exit codes: 0 done; 1 a gate/verification failed; 2 usage error / live call not confirmed; 3 API failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decompile import frames as fr  # noqa: E402
from decompile import ingest  # noqa: E402
from decompile import kinematic_fit as kf  # noqa: E402
from decompile import projective_fit as pfit  # noqa: E402
from decompile import projective_program as pprog  # noqa: E402
from decompile import sam_api as sa  # noqa: E402
from decompile import sam_measure as sm  # noqa: E402
from decompile import sam_tracking as st  # noqa: E402
from decompile import scene_program as sp  # noqa: E402
from decompile.meta_api import strict_json_loads  # noqa: E402

TOOL_VERSION = "tools.sam_demo/1"
EXIT_OK, EXIT_GATE, EXIT_USAGE, EXIT_API = 0, 1, 2, 3
ACCEPTED_CODECS = ("h264", "hevc", "mpeg4")
MAX_TRAJECTORY_FRAMES = 300  # decompile.trajectories.MAX_OBSERVATIONS_PER_OBJECT


class UsageError(Exception):
    pass


class GateError(Exception):
    pass


def _write_json(path: Path, obj, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise UsageError(f"{path} already exists (outputs are never overwritten)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _fraction_json(t: Fraction) -> dict:
    return {"num": t.numerator, "den": t.denominator}


# ------------------------------------------------------------------ plan (offline)


def _parse_frame_box(text: str) -> tuple[int, tuple[int, int, int, int]]:
    try:
        f, box = text.split(":")
        vals = tuple(int(v) for v in box.split(","))
        assert len(vals) == 4
        return int(f), vals
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("a box prompt is FRAME:x0,y0,x1,y1 (half-open pixels)") from None


def _parse_frame_point(text: str) -> tuple[int, tuple[int, int, str]]:
    try:
        parts = text.split(":")
        f = int(parts[0])
        x, y = (int(v) for v in parts[1].split(","))
        label = "negative" if len(parts) > 2 and parts[2] in ("neg", "negative") else "positive"
        return f, (x, y, label)
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("a point prompt is FRAME:x,y[:neg]") from None


def build_prompt(table: fr.FrameTable, *, text: str | None, boxes=(), points=(), object_id: int = 1,
                 propagation: str = "both"):
    if text and (boxes or points):
        raise UsageError("a concept phrase and box/point prompts cannot be combined (SAM 3.1 answers 400)")
    if text:
        return sa.ConceptPrompt(text)
    if not boxes and not points:
        raise UsageError("give --prompt TEXT, or --box / --point object prompts")
    per_frame: dict[int, dict] = {}
    for f, b in boxes:
        if f in per_frame and per_frame[f].get("box") is not None:
            raise UsageError(f"one box per object per frame (frame {f})")
        per_frame.setdefault(f, {"box": None, "points": []})["box"] = b
    for f, p in points:
        per_frame.setdefault(f, {"box": None, "points": []})["points"].append(p)
    n = len(table.frames)
    for f in per_frame:
        if not 0 <= f < n:
            raise UsageError(f"prompt frame {f} is outside the video's frames 0..{n - 1}")
    objs = tuple((object_id, f, d["box"], tuple(d["points"])) for f, d in sorted(per_frame.items()))
    return sa.ObjectPrompt((table.width, table.height), objs, propagation)


def make_plan(video: Path, prompt_args: dict) -> dict:
    video = Path(video)
    if not video.is_file():
        raise UsageError(f"video not found: {video}")
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    if reasons:
        raise GateError("video rejected by validation: " + "; ".join(reasons))
    table = fr.read_frame_table(video, info)
    try:
        prompt = build_prompt(table, **prompt_args)
    except sa.PromptError as exc:
        raise UsageError(str(exc)) from None
    n = len(table.frames)
    warnings = []
    if info.codec not in ACCEPTED_CODECS:
        warnings.append(f"codec {info.codec!r}: SAM documents H.264, H.265 and MPEG-4 only")
    if video.suffix.lower() not in (".mp4", ".mov"):
        warnings.append("SAM documents MP4/MOV containers")
    if n > sa.MAX_FRAMES_PER_REQUEST:
        warnings.append(f"{n} frames exceed the documented {sa.MAX_FRAMES_PER_REQUEST} per request")
    if info.size_bytes > sa.UPLOAD_MAX_BYTES:
        warnings.append("file exceeds the 1 GiB upload limit")
    if n > MAX_TRAJECTORY_FRAMES:
        warnings.append(f"{n} frames: reconstruct/run must use a clip window of <= {MAX_TRAJECTORY_FRAMES} frames, "
                        "but SAM bills every frame of the uploaded file - trim the video first to save cost")
    if table.synthesized_timestamp_count:
        warnings.append(f"{table.synthesized_timestamp_count} frame timestamps were synthesized by FFmpeg (no pts)")
    return {"kind": "sam_call_plan", "tool_version": TOOL_VERSION, "live_call_made": False,
            "video": {"path": str(video.resolve()), "filename": video.name, "sha256": info.sha256,
                      "size_bytes": info.size_bytes, "duration_s": info.duration_s, "codec": info.codec,
                      "nominal_fps": info.nominal_fps},
            "media": {"source_frames": n, "displayed_width": table.width, "displayed_height": table.height,
                      "rotation_deg": table.rotation_deg, "time_base": str(table.time_base),
                      "first_t_s": float(table.frames[0].t), "last_t_s": float(table.frames[-1].t)},
            "prompt": prompt.to_json_dict(), "request_preview": sa.build_request(prompt, "<file id from upload>"),
            "expected_processed_frames": n, "cost": sa.cost_estimate(n), "warnings": warnings}


def plan_text(plan: dict) -> str:
    v, m, c = plan["video"], plan["media"], plan["cost"]
    p = plan["prompt"]
    ptxt = f"concept phrase {p['text']!r}" if p["kind"] == "concept" else f"object prompt {p['input_text']!r}"
    lines = [
        "SAM 3.1 call plan (NO request has been made)",
        f"  video            {v['filename']}  ({v['size_bytes']:,} bytes, sha256 {v['sha256'][:16]}...)",
        f"  duration         {v['duration_s']} s; codec {v['codec']}; nominal fps {v['nominal_fps']}",
        f"  source frames    {m['source_frames']} (displayed {m['displayed_width']}x{m['displayed_height']}, "
        f"rotation {m['rotation_deg']} deg; PTS {m['first_t_s']:.4f} .. {m['last_t_s']:.4f} s)",
        f"  prompt           {ptxt}",
        f"  processed frames {plan['expected_processed_frames']} expected; billed {c['expected_billed_frames']} "
        f"(minimum {c['billing_minimum_frames']} per video request)",
        f"  documented price USD {c['usd_per_1000_frames']} per 1,000 frames -> about USD {c['expected_cost_usd']:.4f} "
        f"({c['pricing_source']}); usage.video_frames_processed in the response is authoritative",
    ]
    lines += [f"  WARNING          {w}" for w in plan["warnings"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ track (the only live call)


def live_client(evidence_dir: Path) -> sa.SamClient:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    return sa.SamClient(evidence_dir=evidence_dir)


def track(plan_path: Path, out_dir: Path, *, confirm_live_call: bool, file_id: str | None = None,
          client_factory=None, log=print) -> dict | None:
    plan = strict_json_loads(Path(plan_path).read_bytes().decode("utf-8"))
    if plan.get("kind") != "sam_call_plan":
        raise UsageError("not a sam_demo plan file")
    video = Path(plan["video"]["path"])
    if not video.is_file() or sa.sha256_file(video) != plan["video"]["sha256"]:
        raise GateError("the planned video is missing or changed since planning")
    log(plan_text(plan))
    if not confirm_live_call:
        log("NOT CALLED: re-run with --confirm-live-call to upload the video and make this billed request.")
        return None
    out_dir = Path(out_dir)
    if (out_dir / "sam_tracking.json").exists():
        raise UsageError(f"{out_dir / 'sam_tracking.json'} already exists (never overwritten)")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = (client_factory or live_client)(out_dir / "api_evidence")
    prompt = sa.prompt_from_json(plan["prompt"])
    if file_id is not None:
        upload = {"file_id": file_id, "reused": True, "source": "--file-id"}
    else:
        rec = client.reuse_or_upload(video, out_dir / "upload_record.json")
        upload = rec.to_json_dict()
    log(f"file id {upload['file_id']} ({'reused' if upload['reused'] else 'uploaded'}); streaming SAM 3.1 ...")
    state = {"n": 0}

    def progress(snap):
        state["n"] += 1
        if state["n"] % 25 == 0:
            log(f"  ... {len(snap.records)} records (revision {snap.revision})")

    call = client.segment_video(upload["file_id"], prompt, on_snapshot=progress, raw_sse_path=out_dir / "raw_stream.sse")
    (out_dir / "raw_output.txt").write_text(call.result.raw_output, encoding="utf-8")
    table = fr.read_frame_table(video)
    try:
        result = st.from_parser_result(call.result, table, prompt=plan["prompt"],
                                       video={"sha256": plan["video"]["sha256"], "filename": video.name,
                                              "size_bytes": plan["video"]["size_bytes"]},
                                       usage=call.stream.usage, request=call.request,
                                       provenance={"tool_version": TOOL_VERSION, "client_version": sa.CLIENT_VERSION,
                                                   "response_id": call.stream.response_id, "upload": upload,
                                                   "started_utc": call.started_utc, "finished_utc": call.finished_utc,
                                                   "event_counts": call.stream.event_counts})
    except st.SamTrackingError as exc:
        _write_json(out_dir / "sam_validation_failure.json", {"problems": exc.problems})
        raise GateError("SAM output failed validation against the frame table: " + "; ".join(exc.problems[:5])) from None
    _write_json(out_dir / "sam_tracking.json", result.to_json_dict())
    summary = result.summary()
    log(f"SAM outcome {result.outcome['status']}; {summary['n_objects']} object(s) on "
        f"{summary['n_frames_with_records']} frame(s); usage {result.usage}")
    return {"result": result, "summary": summary, "path": out_dir / "sam_tracking.json"}


# ------------------------------------------------------------------ reconstruct (offline)


def _kinematics_summary(obj: kf.ObjectKinematics) -> dict:
    c = obj.contact
    return {"object_id": obj.object_id, "n_observations": obj.n_observations, "selected_model": obj.selected_model,
            "adequacy": obj.adequacy, "adequacy_reasons": list(obj.adequacy_reasons),
            "verdicts": {k: {"verdict": v.verdict, "reason": v.reason} for k, v in sorted(obj.verdicts.items())},
            "contact": None if c is None else {
                "model": c.model, "t_rel_s": c.t_contact_s, "t_s": float(obj.t_start_s) + c.t_contact_s,
                "between_frames": list(c.gap_frames), "interval_rel_s": list(c.profile_interval_s),
                "axes": [{"axis": a.axis, "v_before_px_s": a.v_before_px_s, "v_after_px_s": a.v_after_px_s,
                          "reversal": a.reversal, "significant": a.significant} for a in c.axes]},
            "additional_discontinuity": None if obj.additional_discontinuity is None else {
                "t_rel_s": obj.additional_discontinuity.t_contact_s,
                "between_frames": list(obj.additional_discontinuity.gap_frames),
                "delta_bic": obj.additional_discontinuity.delta_bic},
            "turning_points": [{"kind": t.kind, "t_s": t.t_s, "value_px": t.value_px} for t in obj.turning_points]}


def physics(traj, physics_model: str, hfov: float):
    """Fit, build the program, or explain why not. Returns (summary, program | None, simulate_fn | None, kind)."""
    obj_traj = traj.objects[0]
    if physics_model == "projective":
        try:
            rep = pfit.analyze_projective(traj, noise="absolute")
            prior = pprog.CameraPrior(hfov_deg=hfov)
            prog = pprog.program_from_fit(rep, width=traj.coordinate_system.image_width, prior=prior,
                                          t_last_rel=float(obj_traj.observations[-1].t) - rep.t0)
        except (pfit.ProjectiveFitError, pprog.ProgramUnavailable) as exc:
            reasons = getattr(exc, "reasons", None) or [str(exc)]
            return {"model": "projective", "program": None, "reasons": list(reasons)}, None, None
        summ = {"model": "projective", "selected": rep.selected, "adequacy": rep.adequacy,
                "adequacy_reasons": list(rep.adequacy_reasons), "program_sha256": pprog.program_sha256(prog),
                "restitution": prog.restitution, "gravity_magnitude_diameters_s2":
                    prog.provenance.get("gravity_magnitude_diameters_s2"),
                "contact_t_s": prog.provenance.get("contact_t_s"),
                "contact_between_measured_frames": prog.provenance.get("contact_gap_frames"),
                "verdicts": prog.provenance.get("verdicts"),
                "camera_hfov_prior_deg": prog.provenance.get("hfov_used_deg"),
                "implied_ball_diameter_m_if_g_9_81": prog.provenance.get("implied_ball_diameter_m_if_g_9_81")}
        return summ, prog, lambda p, times: pprog.simulate(p, times)
    rep = kf.analyze_kinematics(traj, weights="per_observation")
    obj = rep.objects[0]
    summ = {"model": "image_space", "kinematics": _kinematics_summary(obj)}
    try:
        prog = sp.program_from_kinematics(obj)
    except sp.ProgramUnavailable as exc:
        summ.update({"program": None, "reasons": list(exc.reasons)})
        return summ, None, None
    summ.update({"program_sha256": sp.program_sha256(prog), "restitution": prog.planes[0].restitution,
                 "acceleration_px_s2": [prog.acceleration.ax, prog.acceleration.ay],
                 "contact_boundary": {"axis": prog.planes[0].axis, "position_px": prog.planes[0].position_px,
                                      "keep_side": prog.planes[0].keep_side}})
    return summ, prog, lambda p, times: sp.simulate(p, times)


def counterfactual(prog, *, restitution: float | None, gravity_scale: float | None):
    if restitution is None and gravity_scale is None:
        return None
    if isinstance(prog, sp.SceneProgram):
        planes = tuple(dataclasses.replace(p, restitution=restitution) if restitution is not None else p
                       for p in prog.planes)
        acc = prog.acceleration if gravity_scale is None else sp.UniformAcceleration(
            prog.acceleration.ax * gravity_scale, prog.acceleration.ay * gravity_scale)
        prov = {**prog.provenance, "counterfactual": {"restitution": restitution, "gravity_scale": gravity_scale}}
        return sp.SceneProgram(prog.body, acc, planes, prov)
    g = prog.g if gravity_scale is None else tuple(v * gravity_scale for v in prog.g)
    return dataclasses.replace(prog, restitution=prog.restitution if restitution is None else restitution, g=g,
                               provenance={**prog.provenance, "counterfactual": {"restitution": restitution,
                                                                                 "gravity_scale": gravity_scale}})


def _sim_xy(sim) -> tuple[list[float], list[float]]:
    return list(sim.x_px), list(sim.y_px)


def _program_json(prog) -> dict:
    return sp.program_to_json(prog) if isinstance(prog, sp.SceneProgram) else pprog.program_to_json(prog)


def _events(sim) -> list[dict]:
    out = []
    for e in sim.events:
        out.append({"t_s": e.t_s, "x_px": e.x_px, "y_px": e.y_px})
    return out


def draw_frame(rgb: np.ndarray, *, frame_index: int, t: Fraction, fm: sm.FrameMeasurement, target_mask, other_masks,
               trail, sim_pt, cf_pt, cf_trail, diameter, label: str) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    for om in other_masks:
        img[om] = img[om] * 0.6 + np.array([160, 160, 160], np.float32) * 0.4
    if target_mask is not None:
        img[target_mask] = img[target_mask] * 0.6 + np.array([60, 200, 60], np.float32) * 0.4
    img = np.clip(img, 0, 255).astype(np.uint8)
    if target_mask is not None:
        cnts, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (60, 220, 60), 1, cv2.LINE_AA)
    S = 16  # sub-pixel drawing (shift 4); our pixel centres are at +0.5

    def P(x, y):
        return int(round((x - 0.5) * S)), int(round((y - 0.5) * S))

    for a, b in zip(cf_trail, cf_trail[1:]):
        cv2.line(img, P(*a), P(*b), (255, 200, 0), 1, cv2.LINE_AA, 4)
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, P(*a), P(*b), (0, 220, 255), 2, cv2.LINE_AA, 4)
    r = max(2.0, (diameter or 8.0) / 2.0)
    if sim_pt is not None:  # drawn 3 px outside the outline so both stay visible when they agree
        cv2.circle(img, P(*sim_pt), int((r + 3.0) * S), (255, 0, 255), 1, cv2.LINE_AA, 4)
    if cf_pt is not None:
        cv2.circle(img, P(*cf_pt), int(r * S), (255, 200, 0), 1, cv2.LINE_AA, 4)
    if fm.status == sm.MEASURED:
        o = fm.outline
        cv2.ellipse(img, P(fm.x, fm.y), (int(o["semi_major_px"] * S), int(o["semi_minor_px"] * S)), o["angle_deg"], 0, 360,
                    (0, 0, 255), 1, cv2.LINE_AA, 4)
        cx, cy = P(fm.x, fm.y)
        cv2.line(img, (cx - 4 * S, cy), (cx + 4 * S, cy), (255, 255, 255), 1, cv2.LINE_AA, 4)
        cv2.line(img, (cx, cy - 4 * S), (cx, cy + 4 * S), (255, 255, 255), 1, cv2.LINE_AA, 4)
        status = f"measured  sigma=({fm.sigma_x:.2f},{fm.sigma_y:.2f}) px  D={fm.diameter_px:.1f} px"
        colour = (80, 230, 80)
    elif fm.status == sm.UNCERTAIN:
        status, colour = f"UNCERTAIN (no coordinate): {fm.reason[:60]}", (0, 165, 255)
    else:
        status, colour = f"MISSING: {fm.reason[:60]}", (60, 60, 255)
    h, w = img.shape[:2]
    bar = np.zeros((44, w, 3), np.uint8)
    cv2.putText(bar, f"{label} | frame {frame_index}  t={float(t):.4f} s (PTS)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, status, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    legend = np.zeros((22, w, 3), np.uint8)
    cv2.putText(legend, "green=SAM mask  red=fitted outline  yellow=measured path  magenta=scene program  cyan=counterfactual",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
    out = np.vstack([bar, img, legend])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def encode_mp4(frames: list[np.ndarray], path: Path, fps: Fraction) -> None:
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-video_size", f"{w2}x{h2}", "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f[:h2, :w2]).tobytes() for f in frames), check=True)


def dense_times(times, n: int = 400) -> list[Fraction]:
    if not times:
        return []
    a, b = times[0], times[-1]
    return [a + (b - a) * Fraction(k, n - 1) for k in range(n)]


def plot_trajectory(path: Path, meas: sm.SamMeasurementResult, times, sim_xy, cf_xy, events, cf_events, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ms = meas.measured()
    tm = [float(f.t) for f in ms]
    ts = [float(t) for t in times]
    for ax, key, sk in ((axes[0], "x", "sigma_x"), (axes[1], "y", "sigma_y")):
        ax.errorbar(tm, [getattr(f, key) for f in ms], yerr=[3 * getattr(f, sk) for f in ms], fmt="o", ms=3, color="#1f6f3f",
                    ecolor="#7fbf7f", elinewidth=1, label="measured (SAM-guided outline), 3-sigma bars")
        bad = [f for f in meas.frames if f.status != sm.MEASURED]
        for f in bad:
            ax.axvline(float(f.t), color="#f0a030" if f.status == sm.UNCERTAIN else "#e04040", lw=0.6, alpha=0.6)
        if sim_xy is not None:
            ax.plot(ts, sim_xy[0 if key == "x" else 1], color="#c02cc0", lw=1.5, label="scene program (simulated)")
        if cf_xy is not None:
            ax.plot(ts, cf_xy[0 if key == "x" else 1], color="#1a9fd0", lw=1.3, ls="--", label="counterfactual")
        for e in events:
            ax.axvline(e["t_s"], color="#c02cc0", lw=0.8, ls=":")
        for e in cf_events:
            ax.axvline(e["t_s"], color="#1a9fd0", lw=0.8, ls=":")
        ax.set_ylabel(f"{key} (px, +y down)" if key == "y" else "x (px)")
        if key == "y":
            ax.invert_yaxis()
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("t (s, exact presentation timestamps)")
    axes[0].set_title(title, fontsize=10)
    fig.text(0.01, 0.005, "orange/red lines: frames with no coordinate (uncertain / missing) - never interpolated. "
             "Dotted: simulated contacts.", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def reconstruct(video: Path, sam_result_path: Path, out_dir: Path, *, sam_object="most_frames", start_frame: int = 0,
                end_frame: int | None = None, stride: int = 1, outline_model: str = "ellipse",
                physics_model: str = "image_space", hfov: float = 66.0, cf_restitution: float | None = None,
                cf_gravity_scale: float | None = None, provenance: str = "real", min_confidence: float = 0.5,
                write_video: bool = True, disclosures=()) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise UsageError(f"{out_dir} is not empty (outputs are never overwritten)")
    video = Path(video)
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    if reasons:
        raise GateError("video rejected: " + "; ".join(reasons))
    raw = Path(sam_result_path).read_bytes()
    result = st.SamTrackingResult.from_json_dict(strict_json_loads(raw.decode("utf-8")))
    if result.video.get("sha256") != info.sha256:
        raise GateError("the SAM result belongs to a different video (SHA-256 mismatch)")
    if not result.completed:
        raise GateError(f"the SAM response is {result.outcome['status']} ({result.outcome.get('reason')})")
    table = fr.extract_frames(video, out_dir / "frames")
    if (result.media["width"], result.media["height"], result.media["frame_count"]) != (table.width, table.height,
                                                                                       len(table.frames)):
        raise GateError("SAM media size / frame count differs from the extracted frames")
    end = end_frame if end_frame is not None else len(table.frames)
    clip = [f.index for f in table.frames if start_frame <= f.index < end and (f.index - start_frame) % stride == 0]
    if not 4 <= len(clip) <= MAX_TRAJECTORY_FRAMES:
        raise UsageError(f"the clip has {len(clip)} frames; use a window of 4..{MAX_TRAJECTORY_FRAMES}")
    sel = sam_object if sam_object == "most_frames" else {"id": str(sam_object)}
    try:
        oid = sm.select_object(result, clip, sel)
    except st.SamTrackingError as exc:
        raise GateError("; ".join(exc.problems)) from None
    by = {f.index: f for f in table.frames}
    rgb_cache: dict[int, np.ndarray] = {}

    def source(i):
        if i not in rgb_cache:
            rgb_cache[i] = fr.read_png_rgb(out_dir / "frames" / fr.PNG_PATTERN.format(index=i))
        return by[i], rgb_cache[i]

    params = sm.SamMeasureParams(outline_model=outline_model, min_confidence=min_confidence)
    meas = sm.measure_track(result, source, clip, sam_object_id=oid, params=params,
                            sam_result_sha256=hashlib.sha256(raw).hexdigest())
    _write_json(out_dir / "measurement" / "sam_measurement_report.json", meas.to_json_dict())
    counts = meas.status_counts()
    report: dict = {"kind": "sam_reconstruction_report", "tool_version": TOOL_VERSION,
                    "label": "REAL VIDEO" if provenance == "real" else "SYNTHETIC video",
                    "video": {"filename": video.name, "sha256": info.sha256, "frames": len(table.frames),
                              "displayed_size": [table.width, table.height], "time_base": str(table.time_base)},
                    "sam": {"model": result.model, "prompt": result.prompt, "object_id": oid,
                            "objects_in_result": list(result.object_ids()), "usage": result.usage,
                            "summary": result.summary()["objects"].get(oid)},
                    "clip": {"start_frame": clip[0], "end_frame_exclusive": end, "stride": stride, "frames": len(clip)},
                    "measurement": {"method": sm.METHOD_VERSION, "status_counts": counts,
                                    "median_sigma_px": (statistics.median(max(f.sigma_x, f.sigma_y) for f in meas.measured())
                                                        if meas.measured() else None),
                                    "median_diameter_px": (statistics.median(f.diameter_px for f in meas.measured())
                                                           if meas.measured() else None),
                                    "median_abs_mask_centroid_offset_px": (statistics.median(
                                        f.centre_offset_from_mask_px for f in meas.measured()) if meas.measured() else None)},
                    "claims": ["The scene program is FITTED to these observations; reproducing them is a consistency "
                               "check, not a prediction. Held-out prediction: tools/real_video_demo.py prereg-sam/run/reveal.",
                               "SAM provides identity and segmentation support only; centres come from the outline in the "
                               "original pixels."] + [str(d) for d in disclosures]}
    sim_xy = cf_xy = None
    events, cf_events = [], []
    prog = cf = simulate = None
    times = [by[i].t for i in clip]
    if len(meas.measured()) < 8:
        report["physics"] = {"program": None, "reasons": [f"only {len(meas.measured())} measured frames"]}
    else:
        traj = meas.to_trajectory_set(synthetic=provenance != "real")
        _write_json(out_dir / "measurement" / "trajectory.json", traj.to_json_dict())
        summ, prog, simulate = physics(traj, physics_model, hfov)
        report["physics"] = summ
        if prog is not None:
            t_first = traj.objects[0].observations[0].t
            times = [t for t in times if t >= t_first]
            sim = simulate(prog, times)
            sim_xy, events = _sim_xy(sim), _events(sim)
            mt = {f.frame_index: f for f in meas.measured()}
            res = [(sim.x_px[k] - mt[i].x, sim.y_px[k] - mt[i].y, mt[i]) for k, i in enumerate(i for i in clip if by[i].t >= t_first)
                   if i in mt]
            report["reconstruction_consistency"] = {
                "rms_px": math.sqrt(sum(dx * dx + dy * dy for dx, dy, _ in res) / len(res)),
                "rms_normalised": math.sqrt(sum((dx / f.sigma_x) ** 2 + (dy / f.sigma_y) ** 2 for dx, dy, f in res) / (2 * len(res))),
                "n": len(res), "events": events,
                "note": "residuals of the fitted program at the fitted observations (consistency, not prediction)"}
            _write_json(out_dir / "scene_program.json", _program_json(prog))
            cf = counterfactual(prog, restitution=cf_restitution, gravity_scale=cf_gravity_scale)
            if cf is not None:
                csim = simulate(cf, times)
                cf_xy, cf_events = _sim_xy(csim), _events(csim)
                _write_json(out_dir / "counterfactual_program.json", _program_json(cf))
                report["counterfactual"] = {"restitution": cf_restitution, "gravity_scale": cf_gravity_scale,
                                            "events": cf_events,
                                            "final_position_px": [float(csim.x_px[-1]), float(csim.y_px[-1])],
                                            "fitted_final_position_px": [float(sim.x_px[-1]), float(sim.y_px[-1])],
                                            "note": "a what-if simulation of the fitted program with changed physical "
                                                    "parameters; not observed, not validated"}
    title = (f"{report['label']}: SAM 3.1 ({result.prompt.get('text', 'object prompt')}) -> outline measurement -> "
             f"{report['physics'].get('model', '')} physics")
    if prog is not None:
        td = dense_times(times)
        dense = _sim_xy(simulate(prog, td))
        cf_dense = _sim_xy(simulate(cf, td)) if cf is not None else None
    else:
        td, dense, cf_dense = [], None, None
    plot_trajectory(out_dir / "trajectory.png", meas, td, dense, cf_dense, events, cf_events, title)
    # overlay frames
    W, H = table.width, table.height
    sim_at = {i: (sim_xy[0][k], sim_xy[1][k]) for k, i in enumerate(i for i in clip if by[i].t >= times[0])} if sim_xy else {}
    cf_at = {i: (cf_xy[0][k], cf_xy[1][k]) for k, i in enumerate(i for i in clip if by[i].t >= times[0])} if cf_xy else {}
    track_ = result.frames_of(oid)
    trail, cf_trail, rendered = [], [], []
    d_med = report["measurement"]["median_diameter_px"]
    for fm in meas.frames:
        i = fm.frame_index
        _, rgb = source(i)
        recs = result.at(i)
        target = track_[i].full_mask(W, H) if i in track_ and track_[i].has_mask else None
        others = [r.full_mask(W, H) for o, r in recs.items() if o != oid and r.has_mask]
        if fm.status == sm.MEASURED:
            trail.append((fm.x, fm.y))
        if i in cf_at:
            cf_trail.append(cf_at[i])
        rendered.append(draw_frame(rgb, frame_index=i, t=fm.t, fm=fm, target_mask=target, other_masks=others,
                                   trail=list(trail), sim_pt=sim_at.get(i), cf_pt=cf_at.get(i), cf_trail=list(cf_trail),
                                   diameter=fm.diameter_px or d_med, label=report["label"]))
    picks = np.linspace(0, len(rendered) - 1, min(8, len(rendered))).round().astype(int)
    tiles = [rendered[k] for k in picks]
    th, tw = tiles[0].shape[:2]
    cols = 4 if len(tiles) > 4 else len(tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, tile in enumerate(tiles):
        sheet[(k // cols) * th:(k // cols + 1) * th, (k % cols) * tw:(k % cols + 1) * tw] = tile
    cv2.imwrite(str(out_dir / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    if write_video:
        dts = [b.t - a.t for a, b in zip(table.frames, table.frames[1:])]
        fps = (1 / statistics.median(dts)) / stride if dts else Fraction(30)
        encode_mp4(rendered, out_dir / "overlay.mp4", Fraction(fps).limit_denominator(1001))
        report["overlay_video_note"] = "overlay.mp4 plays at the median frame rate for viewing; physics uses exact PTS"
    # the artifacts this run ACTUALLY wrote (the extracted frames/ directory is an input cache, not an artifact)
    written = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                     if p.is_file() and p.relative_to(out_dir).parts[0] != "frames")
    report["artifacts"] = written + ["report.json", "report.md"]
    _write_json(out_dir / "report.json", report)
    (out_dir / "report.md").write_text(report_md(report), encoding="utf-8")
    return report


def _fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def report_md(r: dict) -> str:
    m, p = r["measurement"], r.get("physics", {})
    c = m["status_counts"]
    L = [f"# Decompile reconstruction - {r['label']}", "",
         f"Video `{r['video']['filename']}` ({r['video']['frames']} frames, {r['video']['displayed_size'][0]}x"
         f"{r['video']['displayed_size'][1]}), clip frames {r['clip']['start_frame']}..{r['clip']['end_frame_exclusive'] - 1}"
         f" ({r['clip']['frames']} frames).", "",
         "## 1. Segmentation and tracking (Meta SAM 3.1)", "",
         f"- Prompt: `{json.dumps(r['sam']['prompt'])}`; tracked SAM object `{r['sam']['object_id']}` "
         f"(objects in result: {r['sam']['objects_in_result']}); billed usage: {r['sam']['usage']}", "",
         "## 2. Geometric measurement (Decompile)", "",
         f"- Measured {c['measured']}, uncertain {c['uncertain']}, missing {c['missing']} (uncertain/missing frames get "
         "NO coordinate and are never interpolated).",
         f"- Median per-frame 1-sigma: {_fmt(m['median_sigma_px'])} px; median apparent diameter "
         f"{_fmt(m['median_diameter_px'], 2)} px.",
         f"- Median distance between the outline centre and the SAM mask centroid: "
         f"{_fmt(m['median_abs_mask_centroid_offset_px'])} px (why mask centroids are not used as centres).", "",
         "## 3. Events and physical model", ""]
    if p.get("program") is None and "program_sha256" not in p:
        L += ["- No executable scene program: " + "; ".join(p.get("reasons", [])), ""]
    else:
        if p["model"] == "image_space":
            k = p["kinematics"]
            cc = k["contact"]
            L += [f"- Kinematic model: {k['selected_model']} (adequacy {k['adequacy']}).",
                  f"- Contact at t = {cc['t_s']:.4f} s (between frames {cc['between_frames']})." if cc else "- No contact.",
                  f"- Acceleration (px/s^2, +y down): {[round(v, 2) for v in p['acceleration_px_s2']]}; restitution "
                  f"e = {p['restitution']:.3f}; inferred contact boundary: {p['contact_boundary']['axis']} = "
                  f"{p['contact_boundary']['position_px']:.2f} px (body on the '{p['contact_boundary']['keep_side']}' side)."]
        else:
            ct = p.get("contact_t_s")
            L += [f"- Projective model {p['selected']} (adequacy {p['adequacy']}); verdicts {p.get('verdicts')}.",
                  (f"- Contact at t = {ct:.4f} s (between measured frames {p.get('contact_between_measured_frames')})."
                   if ct is not None else "- No contact."),
                  f"- Restitution e = {p['restitution']:.3f}; |g| = {_fmt(p['gravity_magnitude_diameters_s2'], 2)} ball "
                  f"diameters/s^2 (camera field-of-view PRIOR {p.get('camera_hfov_prior_deg')} deg, not a calibration)."]
        rc = r.get("reconstruction_consistency")
        if rc:
            L += [f"- Simulation vs measurements: RMS {rc['rms_px']:.3f} px ({rc['rms_normalised']:.2f} sigma) over "
                  f"{rc['n']} frames - {rc['note']}."]
        cf = r.get("counterfactual")
        if cf:
            L += ["", "## 4. Counterfactual", "",
                  f"- Restitution {cf['restitution']}, gravity scale {cf['gravity_scale']}: final position "
                  f"{[round(float(v), 1) for v in cf['final_position_px']]} px vs fitted "
                  f"{[round(float(v), 1) for v in cf['fitted_final_position_px']]} px. {cf['note']}."]
    L += ["", "## Claims and limits", ""] + [f"- {x}" for x in r["claims"]]
    L += ["", "Outputs (written by this run): " + ", ".join(r.get("artifacts", [])) + "."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sam_demo.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan", help="OFFLINE: probe the video, build the prompt, report frames and documented cost")
    a.add_argument("--video", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--prompt", default=None, help="ONE short noun phrase, e.g. 'tennis ball'")
    a.add_argument("--box", type=_parse_frame_box, action="append", default=[], help="FRAME:x0,y0,x1,y1 (half-open)")
    a.add_argument("--point", type=_parse_frame_point, action="append", default=[], help="FRAME:x,y[:neg]")
    a.add_argument("--propagation", choices=sa.PROPAGATION_DIRECTIONS, default="both")
    t = sub.add_parser("track", help="LIVE SAM call (billed) - only with --confirm-live-call")
    t.add_argument("--plan", type=Path, required=True)
    t.add_argument("--out-dir", type=Path, required=True)
    t.add_argument("--confirm-live-call", action="store_true")
    t.add_argument("--file-id", default=None, help="reuse an already uploaded file id (skips the upload)")
    r = sub.add_parser("reconstruct", help="OFFLINE: measurement -> physics -> simulation -> overlays and report")
    r.add_argument("--video", type=Path, required=True)
    r.add_argument("--sam-result", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--sam-object", default="most_frames")
    r.add_argument("--start-frame", type=int, default=0)
    r.add_argument("--end-frame", type=int, default=None)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--outline-model", choices=("ellipse", "stadium", "circle"), default="ellipse")
    r.add_argument("--physics-model", choices=("image_space", "projective"), default="image_space")
    r.add_argument("--hfov", type=float, default=66.0)
    r.add_argument("--min-confidence", type=float, default=0.5)
    r.add_argument("--counterfactual-restitution", type=float, default=None)
    r.add_argument("--counterfactual-gravity-scale", type=float, default=None)
    r.add_argument("--provenance", choices=("real", "synthetic"), default="real")
    r.add_argument("--no-video", action="store_true")
    r.add_argument("--disclose", action="append", default=[], help="a disclosure added to the report's claims")
    return p


def main(argv: list[str] | None = None, *, client_factory=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "plan":
            if a.out.exists():
                raise UsageError(f"{a.out} already exists")
            plan = make_plan(a.video, {"text": a.prompt, "boxes": a.box, "points": a.point, "propagation": a.propagation})
            _write_json(a.out, plan)
            print(plan_text(plan))
            print(f"plan written: {a.out}")
        elif a.cmd == "track":
            out = track(a.plan, a.out_dir, confirm_live_call=a.confirm_live_call, file_id=a.file_id,
                        client_factory=client_factory)
            if out is None:
                return EXIT_USAGE
            print(f"SAM tracking result: {out['path']}")
        else:
            rep = reconstruct(a.video, a.sam_result, a.out_dir, sam_object=a.sam_object, start_frame=a.start_frame,
                              end_frame=a.end_frame, stride=a.stride, outline_model=a.outline_model,
                              physics_model=a.physics_model, hfov=a.hfov, cf_restitution=a.counterfactual_restitution,
                              cf_gravity_scale=a.counterfactual_gravity_scale, provenance=a.provenance,
                              min_confidence=a.min_confidence, write_video=not a.no_video, disclosures=a.disclose)
            print(report_md(rep))
            print(f"outputs: {a.out_dir}")
    except UsageError as exc:
        print(f"[USAGE] {exc}")
        return EXIT_USAGE
    except (GateError, st.SamTrackingError) as exc:
        print(f"[STOP] {exc}")
        return EXIT_GATE
    except sa.SamAPIError as exc:
        print(f"[API] {exc.kind}: {exc}" + (f" (evidence: {exc.evidence_path})" if exc.evidence_path else ""))
        return EXIT_API
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
