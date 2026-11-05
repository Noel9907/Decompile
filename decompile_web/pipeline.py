"""Orchestration of the EXISTING Decompile pipeline for the web application.

Every scientific step is delegated:
  video probe / validation      decompile.ingest
  frame table (exact PTS)       decompile.frames
  SAM plan + documented cost    tools/sam_demo.py  make_plan
  SAM upload + segmentation     tools/sam_demo.py  track   (decompile.sam_api client; the only billed call)
  SAM result validation         decompile.sam_tracking
  measurement -> physics ->
  scene program -> simulation   tools/sam_demo.py  reconstruct
  counterfactual                tools/sam_demo.py  counterfactual + decompile.scene_program / projective_program

This module adds only: run bookkeeping, progress events, derived JSON views for the browser (mask contours,
per-frame tables) and a counterfactual overlay rendering of simulator output.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from decompile import frames as fr
from decompile import ingest
from decompile import projective_program as pprog
from decompile import sam_api as sa
from decompile import sam_tracking as st
from decompile import scene_program as sp
from decompile.meta_api import strict_json_loads

from . import WEB_VERSION
from .config import ROOT
from .source import sources as program_sources
from . import ask as ask_mod
from .security import CF_ID_RE, RECON_ID_RE, redact, safe_child
from .store import RunStore, utc_now, write_json_atomic


def _load_sam_demo():
    spec = importlib.util.spec_from_file_location("decompile_web_sam_demo", ROOT / "tools" / "sam_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load_sam_demo()

# Stage catalogue shown in the UI (key, title, which of the nine pipeline steps it belongs to).
TRACKING_STAGES = [
    ("ingest", "Video ingestion"),
    ("timestamps", "Frame table and timestamp validation"),
    ("sam_upload", "SAM upload"),
    ("sam_segmentation", "SAM 3.1 segmentation"),
    ("sam_validation", "Object tracking (SAM output validation)"),
]
RECON_STAGES = [
    ("frames", "Frame extraction"),
    ("measurement", "Geometric refinement (outline measurement)"),
    ("trajectory", "Trajectory construction"),
    ("physics", "Physical model fitting"),
    ("scene_program", "Scene program generation"),
    ("simulation", "Deterministic simulation"),
    ("rendering", "Rendering overlays and reports"),
    ("views", "Preparing interactive views"),
]
PHYSICS_MODELS = ("image_space", "projective")
OUTLINE_MODELS = ("ellipse", "stadium", "circle")
PROVENANCES = ("real", "synthetic")
ACCEPTED_UPLOAD_EXTENSIONS = (".mp4",)
MIN_RECON_FRAMES = 4
_ENGINE_LOCK = threading.Lock()  # matplotlib / reconstruct are not designed for concurrent use

LIMITATIONS = [
    "One body and one contact (bounce) inside the selected window: this is the only physics path implemented. "
    "Multi-object and multi-contact physics are not supported.",
    "Static camera assumed; lens distortion and rolling shutter are not modelled.",
    "Image-space model: pixels and seconds, an inferred axis-aligned contact boundary, uniform acceleration.",
    "Projective model: perspective from the measured apparent diameter; the focal length comes from a camera "
    "field-of-view PRIOR (not a calibration); units are ball diameters, so absolute metric scale is unknown.",
    "The scene program is FITTED to the same observations it is compared with: agreement is a consistency check, "
    "not a held-out prediction.",
    "Per-frame measurement uncertainty is reported, but parameter uncertainty is not propagated into the "
    "simulated trajectories (point simulations only).",
    "Counterfactual simulations are hypothetical what-if runs of the fitted program; they are not observed and "
    "not validated.",
]


class PipelineError(Exception):
    """A user-facing failure with a clear message (already redacted)."""

    def __init__(self, message: str, *, stage: str | None = None, details: list[str] | None = None) -> None:
        super().__init__(redact(message))
        self.stage = stage
        self.details = [redact(d) for d in (details or [])]


# ------------------------------------------------------------------ helpers


def _read_json(path: Path) -> Any:
    return strict_json_loads(Path(path).read_bytes().decode("utf-8"))


def _frac(d: dict) -> Fraction:
    return Fraction(int(d["num"]), int(d["den"]))


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_paths(obj: Any) -> Any:
    """Remove server filesystem paths from data sent to the browser."""
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items() if k not in ("path", "source_path", "result_path")}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    return obj


def empty_stages(catalogue) -> dict:
    return {k: {"key": k, "title": t, "status": "pending", "detail": None, "started_utc": None, "finished_utc": None}
            for k, t in catalogue}


class Ctx:
    """Run-scoped helper: stage bookkeeping + events."""

    def __init__(self, store: RunStore, run_id: str, recon_id: str | None = None) -> None:
        self.store, self.run_id, self.recon_id = store, run_id, recon_id

    @property
    def dir(self) -> Path:
        return self.store.run_dir(self.run_id)

    def _stages(self, run: dict) -> dict:
        if self.recon_id is None:
            return run["stages"]
        return next(r for r in run["reconstructions"] if r["id"] == self.recon_id)["stages"]

    def stage(self, key: str, status: str, detail: Any = None) -> None:
        with self.store.edit(self.run_id) as run:
            s = self._stages(run).get(key)
            if s is None:
                return
            if status == "progress":
                s["status"] = "running"
            else:
                s["status"] = {"started": "running"}.get(status, status)
            if status == "started":
                s["started_utc"] = utc_now()
            if status in ("done", "failed", "skipped"):
                s["finished_utc"] = utc_now()
            if detail is not None:
                s["detail"] = detail
        self.store.append_event(self.run_id, {"type": "stage", "recon_id": self.recon_id, "stage": key,
                                              "status": status, "detail": detail})

    def log(self, message: str, level: str = "info") -> None:
        self.store.append_event(self.run_id, {"type": "log", "recon_id": self.recon_id, "level": level,
                                              "message": message})


# ------------------------------------------------------------------ 1. ingestion


def new_run_record(run_id: str, display_name: str, size_bytes: int, provenance: str) -> dict:
    return {"id": run_id, "web_version": WEB_VERSION, "created_utc": utc_now(), "updated_utc": utc_now(),
            "original_filename": display_name, "upload_size_bytes": size_bytes, "provenance": provenance,
            "state": "ingesting", "video": None, "suitability": None, "rejection_reasons": [],
            "stages": empty_stages(TRACKING_STAGES), "plan": None,
            "sam": {"status": "not_started", "attempts": [], "result": None, "warnings": []},
            "reconstructions": [], "job": None}


def video_path(run_dir: Path) -> Path:
    return run_dir / "input" / "video.mp4"


def assess_suitability(info: ingest.VideoInfo, table: fr.FrameTable) -> dict:
    """What the CURRENT implementation can do with this video (honest, before any paid call)."""
    n = len(table.frames)
    checks, warnings = [], []

    def check(ok: bool, label: str, detail: str, level: str = "required"):
        checks.append({"ok": bool(ok), "label": label, "detail": detail, "level": level})

    check(True, "Decodable video", f"{info.codec}, {n} frames decoded, {table.width}x{table.height}")
    check(info.codec in demo.ACCEPTED_CODECS, "Codec accepted by SAM 3.1",
          f"{info.codec} (SAM documents {', '.join(demo.ACCEPTED_CODECS)})")
    check(n <= sa.MAX_FRAMES_PER_REQUEST, "Frames per SAM request", f"{n} of at most {sa.MAX_FRAMES_PER_REQUEST:,}")
    check(n >= MIN_RECON_FRAMES, "Enough frames to reconstruct", f"{n} frames (minimum {MIN_RECON_FRAMES})")
    check(n <= demo.MAX_TRAJECTORY_FRAMES, "Whole video fits one reconstruction window",
          f"{n} frames; a window may hold at most {demo.MAX_TRAJECTORY_FRAMES}"
          + ("" if n <= demo.MAX_TRAJECTORY_FRAMES else " - choose a window, but SAM still bills every frame of the "
             "file, so trimming the video first is cheaper"), level="advisory")
    if table.synthesized_timestamp_count:
        warnings.append(f"{table.synthesized_timestamp_count} frame timestamps were synthesized by FFmpeg (no PTS)")
    warnings += list(info.warnings)
    if info.has_audio:
        warnings.append("the audio track is ignored")
    required_ok = all(c["ok"] for c in checks if c["level"] == "required")
    return {"suitable": required_ok, "checks": checks, "warnings": warnings,
            "scope": "Supported reconstruction: ONE moving body with ONE contact (bounce) in the selected window, "
                     "filmed by a static camera. Whether the footage satisfies this can only be judged after "
                     "tracking and measurement; the physics gates will stop the run (without fabricating a "
                     "scene program) if it does not."}


def ingest_job(ctx: Ctx) -> None:
    try:
        _ingest(ctx)
    except PipelineError:
        raise
    except Exception as exc:
        with ctx.store.edit(ctx.run_id) as run:
            run["state"] = "rejected"
            run["rejection_reasons"] = [redact(f"ingestion failed: {type(exc).__name__}: {exc}")]
        for k in ("ingest", "timestamps"):
            if ctx.store.load(ctx.run_id)["stages"][k]["status"] == "running":
                ctx.stage(k, "failed", {"reason": redact(str(exc))})
        raise PipelineError(f"Ingestion failed: {exc}", stage="ingest") from None


def _ingest(ctx: Ctx) -> None:
    d = ctx.dir
    video = video_path(d)
    ctx.stage("ingest", "started")
    info = ingest.probe_video(video)
    reasons = ingest.validate(info)
    probe = {"kind": "video_probe", "info": _strip_paths(info.__dict__ | {"decode_errors": info.decode_errors}),
             "rejection_reasons": reasons}
    write_json_atomic(d / "input" / "probe.json", probe)
    summary = {"sha256": info.sha256, "size_bytes": info.size_bytes, "container": info.container, "codec": info.codec,
               "width": info.width, "height": info.height, "duration_s": info.duration_s,
               "nominal_fps": info.nominal_fps, "avg_fps": info.avg_fps, "decoded_frame_count": info.decoded_frame_count,
               "frame_count_status": info.frame_count_status, "has_audio": info.has_audio}
    with ctx.store.edit(ctx.run_id) as run:
        run["video"] = summary
    if reasons:
        ctx.stage("ingest", "failed", {"reasons": reasons})
        with ctx.store.edit(ctx.run_id) as run:
            run["state"] = "rejected"
            run["rejection_reasons"] = reasons
        raise PipelineError("The video was rejected by validation: " + "; ".join(reasons), stage="ingest",
                            details=reasons)
    ctx.stage("ingest", "done", {"codec": info.codec, "frames": info.decoded_frame_count})
    ctx.stage("timestamps", "started")
    try:
        table = fr.read_frame_table(video, info)
    except (fr.VideoRejectedError, fr.TimestampError, fr.FrameStreamError, ValueError) as exc:
        ctx.stage("timestamps", "failed", {"reason": str(exc)})
        with ctx.store.edit(ctx.run_id) as run:
            run["state"] = "rejected"
            run["rejection_reasons"] = [f"frame table: {exc}"]
        raise PipelineError(f"Frame timestamps could not be validated: {exc}", stage="timestamps") from None
    tj = table.to_json_dict()
    tj["source_path"] = "input/video.mp4"
    write_json_atomic(d / "input" / "frame_table.json", tj)
    # first frame as a poster / prompt canvas
    (d / "input" / "preview").mkdir(parents=True, exist_ok=True)
    for _, rgb in fr.iter_frames(video, table):
        fr.write_png_rgb(d / "input" / "preview" / fr.PNG_PATTERN.format(index=0), rgb)
        break
    suit = assess_suitability(info, table)
    times = [float(f.t) for f in table.frames]
    ctx.stage("timestamps", "done", {"frames": len(table.frames), "time_base": str(table.time_base),
                                     "synthesized": table.synthesized_timestamp_count})
    with ctx.store.edit(ctx.run_id) as run:
        run["video"].update({"frame_count": len(table.frames), "display_width": table.width,
                             "display_height": table.height, "rotation_deg": table.rotation_deg,
                             "time_base": str(table.time_base), "first_t_s": times[0], "last_t_s": times[-1],
                             "synthesized_timestamp_count": table.synthesized_timestamp_count})
        run["suitability"] = suit
        run["state"] = "uploaded"


def frame_table(run_dir: Path) -> dict:
    return _read_json(run_dir / "input" / "frame_table.json")


def preview_frame(run_dir: Path, index: int) -> Path:
    """PNG of one source frame (decoded in order up to ``index``; cached)."""
    out = run_dir / "input" / "preview" / fr.PNG_PATTERN.format(index=index)
    if out.is_file():
        return out
    table = fr.FrameTable.from_json_dict({**frame_table(run_dir), "source_path": str(video_path(run_dir))})
    if not 0 <= index < len(table.frames):
        raise PipelineError(f"frame {index} is outside 0..{len(table.frames) - 1}")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ts, rgb in fr.iter_frames(video_path(run_dir), table):
        if ts.index == index:
            fr.write_png_rgb(out, rgb)
            break
    return out


# ------------------------------------------------------------------ 2. plan (offline; no request)


def build_plan(store: RunStore, run_id: str, prompt: dict) -> dict:
    d = store.run_dir(run_id)
    kind = prompt.get("kind")
    if kind == "concept":
        args = {"text": str(prompt.get("text", "")).strip(), "boxes": [], "points": []}
        if not args["text"]:
            raise PipelineError("Enter a short concept phrase, e.g. 'tennis ball'.")
    elif kind == "objects":
        try:
            frame = int(prompt["frame"])
            box = prompt.get("box")
            boxes = [(frame, tuple(int(round(v)) for v in box))] if box else []
            points = [(frame, (int(round(p["x"])), int(round(p["y"])), "negative" if p.get("negative") else "positive"))
                      for p in prompt.get("points", [])]
        except (KeyError, TypeError, ValueError):
            raise PipelineError("An object prompt needs a frame and a box [x0, y0, x1, y1] and/or points.") from None
        if boxes and len(boxes[0][1]) != 4:
            raise PipelineError("A box is [x0, y0, x1, y1].")
        args = {"text": None, "boxes": boxes, "points": points,
                "propagation": prompt.get("propagation", "both")}
        if args["propagation"] not in sa.PROPAGATION_DIRECTIONS:
            raise PipelineError(f"propagation must be one of {sa.PROPAGATION_DIRECTIONS}")
    else:
        raise PipelineError("prompt.kind must be 'concept' or 'objects'")
    try:
        plan = demo.make_plan(video_path(d), args)
    except (demo.UsageError, demo.GateError, sa.PromptError) as exc:
        raise PipelineError(str(exc)) from None
    plan["video"]["path"] = "input/video.mp4"  # stored relative: the run directory is self-contained
    plan_path = d / "sam" / "plan.json"
    raw = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(raw)
    digest = _sha_bytes(raw)
    with store.edit(run_id) as run:
        run["plan"] = {"sha256": digest, "created_utc": utc_now(), "prompt": plan["prompt"], "cost": plan["cost"],
                       "media": plan["media"], "warnings": plan["warnings"],
                       "expected_processed_frames": plan["expected_processed_frames"],
                       "request_preview": plan["request_preview"], "live_call_made": False}
        if run["state"] in ("uploaded", "planned"):
            run["state"] = "planned"
    store.append_event(run_id, {"type": "plan", "sha256": digest, "cost": plan["cost"]})
    return store.load(run_id)["plan"]


# ------------------------------------------------------------------ 3. SAM tracking (the only billed call)


class _ObservedClient:
    """Wraps a decompile.sam_api.SamClient only to report progress; requests are made by the wrapped client."""

    def __init__(self, inner, ctx: Ctx, expected_frames: int) -> None:
        self._inner, self._ctx, self._n = inner, ctx, expected_frames
        self.sent = False

    def reuse_or_upload(self, path, record_path):
        self._ctx.stage("sam_upload", "started", {"bytes": Path(path).stat().st_size})
        rec = self._inner.reuse_or_upload(path, record_path)
        self._ctx.stage("sam_upload", "done", {"file_id_reused": rec.reused, "bytes": rec.bytes})
        return rec

    def segment_video(self, file_id, prompt, *, on_snapshot=None, raw_sse_path=None, **kw):
        self._ctx.stage("sam_segmentation", "started", {"expected_frames": self._n})
        state = {"n": 0, "last": -1}

        def snap(s):
            state["n"] += 1
            frames = len({getattr(r, "frame", None) for r in s.records} - {None})
            if frames != state["last"] and (state["n"] % 10 == 0 or frames - state["last"] >= 5):
                state["last"] = frames
                self._ctx.stage("sam_segmentation", "progress",
                                {"records": len(s.records), "frames_with_records": frames,
                                 "expected_frames": self._n})
            if on_snapshot is not None:
                on_snapshot(s)

        self.sent = True
        call = self._inner.segment_video(file_id, prompt, on_snapshot=snap, raw_sse_path=raw_sse_path, **kw)
        self._ctx.stage("sam_segmentation", "done", {"records": len(call.result.records),
                                                     "outcome": call.result.outcome.status,
                                                     "usage": call.stream.usage})
        self._ctx.stage("sam_validation", "started")
        return call


def check_confirmation(run: dict, *, plan_sha256: str, acknowledged_cost_usd: float, confirm: bool,
                       live_enabled: bool, max_cost_usd: float, api_key_configured: bool) -> None:
    if not confirm:
        raise PipelineError("Explicit confirmation is required before a billed SAM request.")
    if not live_enabled:
        raise PipelineError("Live SAM requests are disabled on this server (set DECOMPILE_LIVE_SAM=1 to enable). "
                            "No request was made.")
    if not api_key_configured:
        raise PipelineError("The server has no MODEL_API_KEY configured. No request was made.")
    plan = run.get("plan")
    if not plan:
        raise PipelineError("Create a SAM plan (prompt + cost estimate) first.")
    if plan_sha256 != plan["sha256"]:
        raise PipelineError("The confirmed plan is not the current plan; review the plan again.")
    cost = plan["cost"]["expected_cost_usd"]
    if not math.isclose(float(acknowledged_cost_usd), cost, rel_tol=0, abs_tol=1e-9):
        raise PipelineError("The acknowledged cost does not match the plan's estimate.")
    if cost > max_cost_usd:
        raise PipelineError(f"The estimated cost USD {cost:.4f} exceeds this server's limit USD {max_cost_usd:.2f}.")
    if run["sam"]["status"] in ("running",):
        raise PipelineError("A SAM request is already running for this run.")
    if run["sam"]["status"] == "completed":
        raise PipelineError("This run already has a completed SAM result; reconstruct from it (no new request).")


def tracking_job(ctx: Ctx, client_factory: Callable | None) -> None:
    """Runs sam_demo.track with confirmation already verified by the caller. Never retries a sent request."""
    d = ctx.dir
    run = ctx.store.load(ctx.run_id)
    n_attempt = len(run["sam"]["attempts"]) + 1
    out_dir = d / "sam" / f"attempt_{n_attempt}"
    attempt = {"n": n_attempt, "started_utc": utc_now(), "status": "running", "plan_sha256": run["plan"]["sha256"],
               "dir": out_dir.relative_to(d).as_posix()}
    with ctx.store.edit(ctx.run_id) as r:
        r["sam"]["status"] = "running"
        r["sam"]["source"] = "live"
        r["sam"]["attempts"].append(attempt)
        r["state"] = "tracking"
        for k in ("sam_upload", "sam_segmentation", "sam_validation"):
            r["stages"][k].update({"status": "pending", "detail": None, "started_utc": None, "finished_utc": None})
    expected = run["plan"]["expected_processed_frames"]
    holder: dict = {}

    def factory(evidence_dir):
        inner = (client_factory or demo.live_client)(evidence_dir)
        holder["client"] = _ObservedClient(inner, ctx, expected)
        return holder["client"]

    # sam_demo.track reads the plan's video path; give it this run's absolute path in a private copy.
    plan = _read_json(d / "sam" / "plan.json")
    plan["video"]["path"] = str(video_path(d))
    exec_plan = out_dir / "plan_resolved.json"
    exec_plan.parent.mkdir(parents=True, exist_ok=True)
    exec_plan.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    try:
        out = demo.track(exec_plan, out_dir, confirm_live_call=True, client_factory=factory,
                         log=lambda m: ctx.log(str(m)))
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, nothing is retried
        sent = holder.get("client") is not None and holder["client"].sent
        kind = getattr(exc, "kind", type(exc).__name__)
        msg = str(exc)
        billing = ("The segmentation request was sent; it may have been processed and billed. It was NOT retried."
                   if sent else "No segmentation request was sent.")
        for k in ("sam_upload", "sam_segmentation", "sam_validation"):
            st_ = ctx.store.load(ctx.run_id)["stages"][k]["status"]
            if st_ == "running":
                ctx.stage(k, "failed", {"reason": redact(msg)})
        with ctx.store.edit(ctx.run_id) as r:
            a = r["sam"]["attempts"][-1]
            a.update({"status": "failed", "finished_utc": utc_now(), "error_kind": kind, "error": redact(msg),
                      "request_sent": sent, "billing_note": billing})
            r["sam"]["status"] = "failed"
            r["state"] = "tracking_failed"
        raise PipelineError(f"SAM tracking failed ({kind}): {msg}. {billing}", stage="sam_segmentation") from None
    result = out["result"]
    _record_sam_result(ctx, out_dir / "sam_tracking.json", result, source="live")
    usage = result.usage or {}
    billed = usage.get("video_frames_processed")
    with ctx.store.edit(ctx.run_id) as r:
        a = r["sam"]["attempts"][-1]
        a.update({"status": "completed", "finished_utc": utc_now(), "request_sent": True, "usage": usage,
                  "actual_billed_frames": billed,
                  "actual_cost_usd": (round(billed * sa.PRICE_USD_PER_1000_FRAMES / 1000.0, 6)
                                      if isinstance(billed, (int, float)) else None),
                  "outcome": result.outcome})


def _record_sam_result(ctx: Ctx, path: Path, result: st.SamTrackingResult, *, source: str) -> None:
    summary = result.summary()
    ctx.stage("sam_validation", "done", {"objects": summary["n_objects"],
                                         "frames_with_records": summary["n_frames_with_records"],
                                         "outcome": result.outcome.get("status")})
    with ctx.store.edit(ctx.run_id) as r:
        r["sam"].update({"status": "completed" if result.completed else "incomplete", "source": source,
                         "result": {"path": path.relative_to(ctx.dir).as_posix(),
                                    "sha256": _sha_bytes(path.read_bytes()), "model": result.model,
                                    "prompt": result.prompt, "outcome": result.outcome, "usage": result.usage,
                                    "summary": summary, "provenance": _strip_paths(result.provenance)}})
        r["state"] = "tracked" if result.completed else "tracking_failed"
        if not result.completed:
            r["sam"]["warnings"].append(f"SAM outcome {result.outcome.get('status')!r} "
                                        f"({result.outcome.get('reason')}); only completed results are reconstructed.")


def import_sam_result(store: RunStore, run_id: str, raw: bytes) -> dict:
    """Attach an EXISTING SAM tracking result (e.g. from tools/sam_demo.py track) to this run. No request is made.
    The result must have been computed on byte-identical video (SHA-256) with the same frame table."""
    d = store.run_dir(run_id)
    run = store.load(run_id)
    if run["state"] in ("ingesting", "rejected"):
        raise PipelineError("Upload a valid video first.")
    if run["sam"]["status"] == "running":
        raise PipelineError("A SAM request is running for this run.")
    if run["sam"]["status"] == "completed":
        raise PipelineError("This run already has a completed SAM result.")
    try:
        result = st.SamTrackingResult.from_json_dict(strict_json_loads(raw.decode("utf-8")))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise PipelineError(f"Not a valid SAM tracking result ({st.SCHEMA}): {exc}") from None
    problems = []
    if result.video.get("sha256") != run["video"]["sha256"]:
        problems.append("it was computed on a different video (SHA-256 mismatch)")
    m = result.media
    if (m.get("width"), m.get("height"), m.get("frame_count")) != (run["video"]["display_width"],
                                                                    run["video"]["display_height"],
                                                                    run["video"]["frame_count"]):
        problems.append(f"its media {m} does not match the uploaded video's frame table")
    if not result.completed:
        problems.append(f"its outcome is {result.outcome.get('status')!r}; only completed results are accepted")
    if problems:
        raise PipelineError("The SAM result cannot be used: " + "; ".join(problems))
    n = len([p for p in (d / "sam").glob("imported_*")]) + 1 if (d / "sam").is_dir() else 1
    out = d / "sam" / f"imported_{n}" / "sam_tracking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    ctx = Ctx(store, run_id)
    for k in ("sam_upload", "sam_segmentation"):
        ctx.stage(k, "skipped", {"reason": "existing SAM result imported; no request made by this server"})
    ctx.stage("sam_validation", "started")
    _record_sam_result(ctx, out, result, source="imported")
    with store.edit(run_id) as r:
        r["sam"]["warnings"].append("This SAM result was imported, not requested by this server: no API call and no "
                                    "charge for this run. Its own provenance is shown below.")
    store.append_event(run_id, {"type": "sam_import", "sha256": _sha_bytes(raw)})
    return store.load(run_id)


def load_sam_result(run_dir: Path, run: dict) -> st.SamTrackingResult:
    if not run["sam"].get("result"):
        raise PipelineError("No SAM result for this run yet.")
    p = safe_child(run_dir, run["sam"]["result"]["path"])
    return st.SamTrackingResult.from_json_dict(_read_json(p))


def sam_tracks(run_dir: Path, run: dict) -> dict:
    """Per-object boxes / confidences per frame (no masks) for the object-selection view."""
    res = load_sam_result(run_dir, run)
    objs = {}
    for oid in res.object_ids():
        objs[oid] = [{"f": i, "box": list(r.box), "conf": r.mask_confidence if r.mask_confidence is not None
                      else r.box_confidence, "mask": r.has_mask} for i, r in sorted(res.frames_of(oid).items())]
    return {"objects": objs, "summary": res.summary()["objects"], "frame_count": res.media["frame_count"],
            "width": res.media["width"], "height": res.media["height"]}


# ------------------------------------------------------------------ 4. reconstruction (offline)


def validate_recon_params(run: dict, p: dict, object_ids: tuple[str, ...]) -> dict:
    n = run["video"]["frame_count"]
    try:
        start = int(p.get("start_frame", 0))
        end = p.get("end_frame")
        end = None if end in (None, "") else int(end)
        hfov = float(p.get("hfov_deg", 66.0))
        min_conf = float(p.get("min_confidence", 0.5))
    except (TypeError, ValueError):
        raise PipelineError("start_frame/end_frame must be integers; hfov_deg/min_confidence numbers") from None
    obj = str(p.get("sam_object", "most_frames"))
    if obj != "most_frames" and obj not in object_ids:
        raise PipelineError(f"SAM object {obj!r} is not in the result (objects: {list(object_ids)})")
    model = p.get("physics_model", "projective")
    outline = p.get("outline_model", "ellipse")
    if model not in PHYSICS_MODELS:
        raise PipelineError(f"physics_model must be one of {PHYSICS_MODELS}")
    if outline not in OUTLINE_MODELS:
        raise PipelineError(f"outline_model must be one of {OUTLINE_MODELS}")
    eff_end = n if end is None else end
    if not (0 <= start < eff_end <= n):
        raise PipelineError(f"the window must satisfy 0 <= start < end <= {n}")
    if not MIN_RECON_FRAMES <= eff_end - start <= demo.MAX_TRAJECTORY_FRAMES:
        raise PipelineError(f"the window must hold {MIN_RECON_FRAMES}..{demo.MAX_TRAJECTORY_FRAMES} frames")
    if not 20.0 <= hfov <= 140.0:
        raise PipelineError("hfov_deg must be within 20..140 degrees")
    if not 0.0 <= min_conf <= 1.0:
        raise PipelineError("min_confidence must be within 0..1")
    return {"sam_object": obj, "start_frame": start, "end_frame": end, "physics_model": model,
            "outline_model": outline, "hfov_deg": hfov, "min_confidence": min_conf}


def reconstruction_job(ctx: Ctx) -> None:
    d = ctx.dir
    rid = ctx.recon_id
    run = ctx.store.load(ctx.run_id)
    rec = next(r for r in run["reconstructions"] if r["id"] == rid)
    p = rec["params"]
    out = d / "recon" / rid
    with ctx.store.edit(ctx.run_id) as r:
        next(x for x in r["reconstructions"] if x["id"] == rid)["status"] = "running"
        r["state"] = "reconstructing"
    sam_path = safe_child(d, run["sam"]["result"]["path"])

    def on_stage(key, status, detail):
        if key == "physics" and status == "done":
            ctx.stage("physics", "done", detail)
            ctx.stage("scene_program", "done" if detail.get("program") else "failed",
                      None if detail.get("program") else {"reason": "no executable scene program (see physics)"})
            return
        ctx.stage(key, status, detail)
        if key == "physics" and status == "skipped":
            ctx.stage("scene_program", "skipped", detail)
            ctx.stage("simulation", "skipped", detail)

    try:
        with _ENGINE_LOCK:
            report = demo.reconstruct(video_path(d), sam_path, out, sam_object=p["sam_object"],
                                      start_frame=p["start_frame"], end_frame=p["end_frame"],
                                      outline_model=p["outline_model"], physics_model=p["physics_model"],
                                      hfov=p["hfov_deg"], provenance="real" if run["provenance"] == "real" else "synthetic",
                                      min_confidence=p["min_confidence"], write_video=True, on_stage=on_stage,
                                      disclosures=(["Run through the Decompile web application."]))
    except (demo.UsageError, demo.GateError, st.SamTrackingError, fr.VideoRejectedError) as exc:
        msg = "; ".join(exc.problems) if isinstance(exc, st.SamTrackingError) else str(exc)
        _fail_running_stages(ctx, msg)
        with ctx.store.edit(ctx.run_id) as r:
            x = next(x for x in r["reconstructions"] if x["id"] == rid)
            x.update({"status": "failed", "error": redact(msg), "finished_utc": utc_now(), "outcome": "stopped"})
            r["state"] = "tracked"
        raise PipelineError(f"Reconstruction stopped: {msg}", stage="reconstruction") from None
    except Exception as exc:
        _fail_running_stages(ctx, str(exc))
        with ctx.store.edit(ctx.run_id) as r:
            x = next(x for x in r["reconstructions"] if x["id"] == rid)
            x.update({"status": "failed", "error": redact(f"{type(exc).__name__}: {exc}"), "finished_utc": utc_now(),
                      "outcome": "error"})
            r["state"] = "tracked"
        raise
    if "program_sha256" not in report.get("physics", {}):
        for k in ("scene_program", "simulation"):
            if ctx.store.load(ctx.run_id)["reconstructions"][int(rid[1:]) - 1]["stages"][k]["status"] == "pending":
                ctx.stage(k, "skipped", {"reason": "no executable scene program"})
    ctx.stage("views", "started")
    analysis = build_analysis(d, rid)
    write_json_atomic(out / "web" / "analysis.json", analysis)
    write_json_atomic(out / "web" / "perception.json", build_perception(d, run, rid, report))
    has_program = analysis["program"] is not None
    outcome = "scene_program" if has_program else "measurements_only"
    with ctx.store.edit(ctx.run_id) as r:
        x = next(x for x in r["reconstructions"] if x["id"] == rid)
        x.update({"status": "completed", "finished_utc": utc_now(), "outcome": outcome,
                  "physics_status": analysis["physics_status"]})
        r["state"] = "reconstructed"
    write_json_atomic(out / "web" / "run_report.json", run_report(ctx.store, ctx.run_id, rid))
    ctx.stage("views", "done", {"outcome": outcome})


def _fail_running_stages(ctx: Ctx, msg: str) -> None:
    run = ctx.store.load(ctx.run_id)
    rec = next(r for r in run["reconstructions"] if r["id"] == ctx.recon_id)
    failed = False
    for k, s in rec["stages"].items():
        if s["status"] == "running" and not failed:
            ctx.stage(k, "failed", {"reason": redact(msg)})
            failed = True
        elif s["status"] in ("pending", "running"):
            ctx.stage(k, "skipped", {"reason": "an earlier stage stopped the reconstruction"})
    if not failed:
        first = next((k for k, s in rec["stages"].items() if s["status"] == "pending"), None)
        if first:
            ctx.stage(first, "failed", {"reason": redact(msg)})


def recon_dir(run_dir: Path, rid: str) -> Path:
    if not RECON_ID_RE.fullmatch(rid or ""):
        raise PipelineError("unknown reconstruction")
    p = run_dir / "recon" / rid
    if not p.is_dir():
        raise PipelineError("unknown reconstruction")
    return p


def _load_program(rdir: Path):
    p = rdir / "scene_program.json"
    if not p.is_file():
        return None, None, None
    data = _read_json(p)
    if data.get("kind") == pprog.PROGRAM_KIND:
        return data, pprog.program_from_json(data), pprog.simulate
    return data, sp.program_from_json(data), sp.simulate


def _sim_dict(sim, times) -> dict:
    out = {"t": [float(t) for t in times], "x": list(map(float, sim.x_px)), "y": list(map(float, sim.y_px)),
           "events": [{"t_s": float(e.t_s), "x_px": float(e.x_px), "y_px": float(e.y_px),
                       **({"resting": bool(e.resting)} if hasattr(e, "resting") else {})} for e in sim.events],
           "termination": getattr(sim, "termination", None)}
    if hasattr(sim, "diameter_px"):
        out["d"] = list(map(float, sim.diameter_px))
        # decompile.projective_program.simulate stops event handling when the body comes to rest but keeps
        # integrating gravity in the final segment, so samples after the rest time pass through the floor.
        # They are flagged (and not drawn by the UI) rather than altered here.
        if sim.termination == "resting" and sim.events:
            out["valid_until_s"] = float(sim.events[-1].t_s)
            out["validity_note"] = ("The projective simulator does not model resting contact: samples after the body "
                                    "comes to rest are not physically meaningful and are not shown.")
    return out


def _program_params(data: dict) -> dict:
    """The physical parameters a user may inspect / change, from the program JSON."""
    if data.get("kind") == pprog.PROGRAM_KIND:
        g = data["g"]
        return {"model": "projective", "restitution": data["restitution"],
                "gravity_vector_diameters_s2": g, "gravity_magnitude_diameters_s2": math.sqrt(sum(v * v for v in g)),
                "initial_position_diameters": data["p0"], "initial_velocity_diameters_s": data["v0"],
                "t0_s": float(_frac(data["t0"])), "focal_length_px": data["f_px"],
                "floor_normal": data["normal"], "tangential_impulse_ratio": data["k_tangential"],
                "units": data["provenance"].get("units")}
    b, a, planes = data["body"], data["acceleration"], data["planes"]
    return {"model": "image_space", "restitution": planes[0]["restitution"] if planes else None,
            "acceleration_px_s2": [a["ax"], a["ay"]], "gravity_magnitude_px_s2": math.hypot(a["ax"], a["ay"]),
            "initial_position_px": [b["x0"], b["y0"]], "initial_velocity_px_s": [b["vx0"], b["vy0"]],
            "t0_s": float(_frac(b["t0"])),
            "contact_boundary": planes[0] if planes else None, "units": {"length": "px", "time": "s"}}


def build_analysis(run_dir: Path, rid: str) -> dict:
    """Per-frame table + fitted simulation, read from the engine's outputs (nothing is re-estimated here)."""
    rdir = run_dir / "recon" / rid
    report = _read_json(rdir / "report.json")
    meas = _read_json(rdir / "measurement" / "sam_measurement_report.json")
    frames = []
    for f in meas["frames"]:
        frames.append({k: f.get(k) for k in ("frame_index", "status", "reason", "x", "y", "sigma_x", "sigma_y",
                                             "diameter_px", "sigma_diameter_px", "confidence", "mask_area_px",
                                             "mask_centroid", "iou_outline_mask", "centre_offset_from_mask_px",
                                             "outline")} | {"t_s": float(_frac(f["t"])), "t_exact": f["t"]})
    data, prog, simulate = _load_program(rdir)
    out = {"kind": "decompile_web_analysis", "recon_id": rid, "frames": frames, "report": report,
           "program": None, "program_parameters": None, "fitted": None, "fitted_dense": None,
           "physics_status": None}
    phys = report.get("physics", {})
    if prog is None:
        out["physics_status"] = {"status": "no_program", "reasons": phys.get("reasons") or
                                 ["the physics gates did not produce an executable scene program"]}
        return out
    traj = _read_json(rdir / "measurement" / "trajectory.json")
    t_first = _frac(traj["objects"][0]["observations"][0]["t"])
    ftimes = [_frac(f["t"]) for f in meas["frames"] if _frac(f["t"]) >= t_first]
    fsim = simulate(prog, ftimes)
    dense = demo.dense_times(ftimes, 400)
    out.update({"program": data, "program_parameters": _program_params(data),
                "fitted": {"frame_index": [f["frame_index"] for f in meas["frames"] if _frac(f["t"]) >= t_first],
                           **_sim_dict(fsim, ftimes)},
                "fitted_dense": _sim_dict(simulate(prog, dense), dense),
                "physics_status": {"status": "scene_program", "adequacy": phys.get("adequacy")
                                   or phys.get("kinematics", {}).get("adequacy")}})
    return out


def build_perception(run_dir: Path, run: dict, rid: str, report: dict) -> dict:
    """SAM boxes, confidences and mask contours (all objects) on the reconstruction window's frames."""
    res = load_sam_result(run_dir, run)
    W, H = res.media["width"], res.media["height"]
    clip = report["clip"]
    frames = range(clip["start_frame"], clip["end_frame_exclusive"])
    out = {}
    for i in frames:
        recs = res.at(i)
        objs = []
        for oid, r in sorted(recs.items()):
            contours = []
            if r.has_mask:
                try:
                    m = r.full_mask(W, H).astype(np.uint8)
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cs:
                        c = cv2.approxPolyDP(c, 0.6, True)
                        if len(c) >= 3:
                            contours.append([[int(p[0][0]), int(p[0][1])] for p in c])
                except st.SamTrackingError:
                    contours = []
            objs.append({"id": oid, "box": list(r.box), "box_confidence": r.box_confidence,
                         "mask_confidence": r.mask_confidence, "has_mask": r.has_mask, "contours": contours})
        out[str(i)] = objs
    return {"kind": "decompile_web_perception", "selected_object": report["sam"]["object_id"],
            "width": W, "height": H, "frames": out,
            "note": "SAM 3.1 boxes and mask outlines (identity and support only; centres come from Decompile's "
                    "outline measurement)."}


# ------------------------------------------------------------------ 5. counterfactuals


def _edit_program(prog, *, velocity_scale: float | None, remove_floor: bool):
    """Extra what-if edits on top of sam_demo.counterfactual: initial speed and floor removal (program edits only)."""
    if velocity_scale is None and not remove_floor:
        return prog
    extra = {"velocity_scale": velocity_scale, "floor_removed": bool(remove_floor)}
    cfp = dict(prog.provenance.get("counterfactual") or {})
    if isinstance(prog, pprog.ProjectiveProgram):
        return dataclasses.replace(
            prog, v0=tuple(v * velocity_scale for v in prog.v0) if velocity_scale is not None else prog.v0,
            plane_offset=-1.0e6 if remove_floor else prog.plane_offset,  # floor moved out of reach = no floor
            provenance={**prog.provenance, "counterfactual": {**cfp, **extra}})
    b = prog.body
    body = dataclasses.replace(b, vx0=b.vx0 * velocity_scale, vy0=b.vy0 * velocity_scale) if velocity_scale is not None else b
    return sp.SceneProgram(body, prog.acceleration, () if remove_floor else prog.planes,
                           {**prog.provenance, "counterfactual": {**cfp, **extra}})


def counterfactual(run_dir: Path, rid: str, *, restitution: float | None, gravity_scale: float | None,
                   extend_s: float = 0.0, velocity_scale: float | None = None, remove_floor: bool = False) -> dict:
    rdir = recon_dir(run_dir, rid)
    data, prog, simulate = _load_program(rdir)
    if prog is None:
        raise PipelineError("This reconstruction has no executable scene program, so there is nothing to simulate.")
    if restitution is not None and not 0.0 <= float(restitution) <= 1.0:
        raise PipelineError("restitution must be within [0, 1]")
    if gravity_scale is not None and not 0.1 <= float(gravity_scale) <= 3.0:
        raise PipelineError("gravity_scale must be within [0.1, 3]")
    if not 0.0 <= float(extend_s) <= 3.0:
        raise PipelineError("extend_s must be within [0, 3] seconds")
    if velocity_scale is not None and not 0.2 <= float(velocity_scale) <= 3.0:
        raise PipelineError("velocity_scale must be within [0.2, 3]")
    params = {"restitution": None if restitution is None else float(restitution),
              "gravity_scale": None if gravity_scale is None else float(gravity_scale),
              "extend_s": float(extend_s)}
    if velocity_scale is not None and float(velocity_scale) != 1.0:
        params["velocity_scale"] = float(velocity_scale)
    if remove_floor:
        params["remove_floor"] = True
    cf_id = "cf_" + _sha_bytes(json.dumps({"rid": rid, **params}, sort_keys=True).encode())[:16]
    path = rdir / "web" / "counterfactuals" / f"{cf_id}.json"
    if path.is_file():
        return _read_json(path)
    cf = demo.counterfactual(prog, restitution=params["restitution"], gravity_scale=params["gravity_scale"])
    if cf is None:  # no change requested: the counterfactual is the fitted program itself
        cf = prog
    cf = _edit_program(cf, velocity_scale=params.get("velocity_scale"), remove_floor=params.get("remove_floor", False))
    meas = _read_json(rdir / "measurement" / "sam_measurement_report.json")
    traj = _read_json(rdir / "measurement" / "trajectory.json")
    t_first = _frac(traj["objects"][0]["observations"][0]["t"])
    ftimes = [_frac(f["t"]) for f in meas["frames"] if _frac(f["t"]) >= t_first]
    findex = [f["frame_index"] for f in meas["frames"] if _frac(f["t"]) >= t_first]
    t_end = ftimes[-1] + Fraction(params["extend_s"]).limit_denominator(1000)
    dense = demo.dense_times([ftimes[0], t_end], 500)
    try:
        fitted_f, cf_f = simulate(prog, ftimes), simulate(cf, ftimes)
        fitted_d, cf_d = simulate(prog, dense), simulate(cf, dense)
    except (sp.SimulationError, ValueError) as exc:
        raise PipelineError(f"The simulator stopped: {exc}") from None
    cf_json = pprog.program_to_json(cf) if isinstance(cf, pprog.ProjectiveProgram) else sp.program_to_json(cf)
    title = _source_title(rdir)
    result = {"kind": "decompile_counterfactual", "id": cf_id, "recon_id": rid, "created_utc": utc_now(),
              "label": "HYPOTHETICAL - a what-if simulation of the fitted scene program with changed physical "
                       "parameters. Not observed and not validated.",
              "parameters": params, "fitted_parameters": _program_params(data),
              "counterfactual_parameters": _program_params(cf_json), "counterfactual_program": cf_json,
              "fitted_program_sha256": _sha_bytes(json.dumps(data, sort_keys=True).encode()),
              "frames": {"frame_index": findex, "fitted": _sim_dict(fitted_f, ftimes), "counterfactual": _sim_dict(cf_f, ftimes)},
              "dense": {"fitted": _sim_dict(fitted_d, dense), "counterfactual": _sim_dict(cf_d, dense)},
              "observed_until_s": float(ftimes[-1]),
              "source": program_sources(cf_json, {"title": f"HYPOTHETICAL counterfactual of {title}"})}
    write_json_atomic(path, result)
    return result


def _source_title(rdir: Path) -> str:
    rep = _read_json(rdir / "report.json")
    c = rep["clip"]
    try:  # the uploaded file's (sanitised) name, not the internal input/video.mp4
        name = json.loads((rdir.parent.parent / "run.json").read_text(encoding="utf-8"))["original_filename"]
    except (OSError, ValueError, KeyError):
        name = rep["video"]["filename"]
    return f"{name}, frames {c['start_frame']}-{c['end_frame_exclusive'] - 1}"


def ask_whatif(run_dir: Path, rid: str, question: str, client_factory) -> dict:
    """Muse Spark maps the question to supported parameters; the simulator computes the counterfactual."""
    rdir = recon_dir(run_dir, rid)
    data, prog, _ = _load_program(rdir)
    if prog is None:
        raise PipelineError("This reconstruction has no executable scene program, so there is nothing to simulate.")
    key = ask_mod.ask_key(rid, question)
    path = rdir / "web" / "asks" / f"ask_{key}.json"
    if path.is_file():
        rec = _read_json(path)
    else:
        rep = _read_json(rdir / "report.json")
        phys = rep.get("physics", {})
        params = _program_params(data)
        context = {"object": (rep.get("sam", {}).get("prompt") or {}).get("text", "the tracked object"),
                   "model": params["model"], "measured_restitution": round(params["restitution"], 4),
                   "gravity_in_video": "Earth (1 x); magnitude in the program's units "
                                       f"{params.get('gravity_magnitude_diameters_s2') or params.get('gravity_magnitude_px_s2'):.2f}",
                   "implied_object_diameter_m_if_earth_gravity": phys.get("implied_ball_diameter_m_if_g_9_81"),
                   "observed_contacts": 1, "observed_duration_s": round(rep["clip"]["frames"] / 30.0, 2)}
        try:
            interp = ask_mod.interpret(question, context, client_factory, rdir / "web" / "asks" / "api_evidence")
        except ask_mod.AskError as exc:
            raise PipelineError(str(exc)) from None
        rec = {"kind": "decompile_whatif", "key": key, "created_utc": utc_now(), "interpretation": interp}
        write_json_atomic(path, rec)
    it = rec["interpretation"]
    cf = None
    if it["supported"] and (it["restitution"] is not None or it["gravity_scale"] is not None
                            or it.get("velocity_scale") is not None or it.get("remove_floor")):
        cf = counterfactual(run_dir, rid, restitution=it["restitution"], gravity_scale=it["gravity_scale"],
                            extend_s=it["extend_s"], velocity_scale=it.get("velocity_scale"),
                            remove_floor=bool(it.get("remove_floor")))
    return {"interpretation": it, "counterfactual": cf}


def explain_fit(run_dir: Path, rid: str, client_factory) -> dict:
    """Muse Spark reviews the fitted numbers (commentary only; cached per reconstruction)."""
    rdir = recon_dir(run_dir, rid)
    path = rdir / "web" / "explain.json"
    if path.is_file():
        return _read_json(path)
    data, prog, _ = _load_program(rdir)
    if prog is None:
        raise PipelineError("This reconstruction has no scene program to explain.")
    rep = _read_json(rdir / "report.json")
    meas = _read_json(rdir / "measurement" / "sam_measurement_report.json")
    phys, m, rc = rep["physics"], rep["measurement"], rep.get("reconstruction_consistency") or {}
    flagged = [{"frame": f["frame_index"], "status": f["status"], "reason": f["reason"]}
               for f in meas["frames"] if f["status"] != "measured"]
    params = _program_params(data)
    context = {"object_named_by_user": (rep["sam"].get("prompt") or {}).get("text"),
               "physics_model": params["model"], "restitution": params["restitution"],
               "contact_time_s": phys.get("contact_t_s") or (phys.get("kinematics") or {}).get("contact", {}).get("t_s"),
               "contact_between_measured_frames": phys.get("contact_between_measured_frames"),
               "gravity_magnitude": params.get("gravity_magnitude_diameters_s2") or params.get("gravity_magnitude_px_s2"),
               "gravity_units": "ball diameters per s^2" if params["model"] == "projective" else "px per s^2",
               "implied_ball_diameter_m_if_earth_gravity": phys.get("implied_ball_diameter_m_if_g_9_81"),
               "camera": "focal length from a 66 deg field-of-view prior, not calibrated",
               "model_verdicts": phys.get("verdicts"),
               "verdict_glossary": {
                   "perspective": "supported = depth/perspective effects are present in the data and modelled",
                   "acceleration_change_at_contact": "not_supported = the data do NOT require different gravity before "
                                                     "and after the bounce (the expected, physically correct outcome); "
                                                     "this does not question that a bounce happened",
                   "second_discontinuity": "inconclusive = no clear evidence of a second bounce in the window"},
               "adequacy_gate": phys.get("adequacy"),
               "adequacy_meaning": "adequate = residuals are consistent with the per-frame measurement uncertainty",
               "fit_rms_px": rc.get("rms_px"), "fit_rms_in_sigma": rc.get("rms_normalised"), "fitted_frames": rc.get("n"),
               "window_frames": rep["clip"]["frames"], "measured_frames": m["status_counts"],
               "median_measurement_sigma_px": m.get("median_sigma_px"), "flagged_frames": flagged[:12],
               "note": "the model was fitted to these same frames (consistency check, not a held-out prediction)"}
    try:
        out = ask_mod.explain(context, client_factory, rdir / "web" / "asks" / "api_evidence")
    except ask_mod.AskError as exc:
        raise PipelineError(str(exc)) from None
    out["created_utc"] = utc_now()
    write_json_atomic(path, out)
    return out


def program_source_for(run_dir: Path, rid: str) -> dict:
    rdir = recon_dir(run_dir, rid)
    data, prog, _ = _load_program(rdir)
    if prog is None:
        raise PipelineError("This reconstruction has no executable scene program.")
    return {**program_sources(data, {"title": f"Decompiled from {_source_title(rdir)}"}),
            "filename": f"decompiled_{rid}.py"}


def get_counterfactual(run_dir: Path, rid: str, cf_id: str) -> dict:
    if not CF_ID_RE.fullmatch(cf_id or ""):
        raise PipelineError("unknown counterfactual")
    p = recon_dir(run_dir, rid) / "web" / "counterfactuals" / f"{cf_id}.json"
    if not p.is_file():
        raise PipelineError("unknown counterfactual")
    return _read_json(p)


def render_counterfactual_video(run_dir: Path, rid: str, cf_id: str) -> Path:
    """Draw the simulator's fitted and counterfactual positions on the source frames (clearly labelled)."""
    cf = get_counterfactual(run_dir, rid, cf_id)
    rdir = recon_dir(run_dir, rid)
    out = rdir / "web" / "counterfactuals" / f"{cf_id}.mp4"
    if out.is_file():
        return out
    analysis = _read_json(rdir / "web" / "analysis.json")
    by_meas = {f["frame_index"]: f for f in analysis["frames"]}
    fi = cf["frames"]["frame_index"]
    F, C = cf["frames"]["fitted"], cf["frames"]["counterfactual"]
    d_med = analysis["report"]["measurement"].get("median_diameter_px") or 20.0
    p = cf["parameters"]
    tag = (f"e={p['restitution']:.2f}" if p["restitution"] is not None else "e=fitted") + \
          (f"  g x{p['gravity_scale']:.2f}" if p["gravity_scale"] is not None else "")
    rendered, trail_f, trail_c = [], [], []
    S = 16

    def P(x, y):
        return int(round((x - 0.5) * S)), int(round((y - 0.5) * S))

    for k, i in enumerate(fi):
        rgb = fr.read_png_rgb(rdir / "frames" / fr.PNG_PATTERN.format(index=i))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img = (img.astype(np.float32) * 0.55).astype(np.uint8)  # dim the evidence under the hypothetical
        tk_ = F["t"][k]
        f_ok = tk_ <= F.get("valid_until_s", math.inf) + 1e-9  # no samples after a projective rest (see _sim_dict)
        c_ok = tk_ <= C.get("valid_until_s", math.inf) + 1e-9
        if f_ok:
            trail_f.append((F["x"][k], F["y"][k]))
        if c_ok:
            trail_c.append((C["x"][k], C["y"][k]))
        for a, b in zip(trail_f, trail_f[1:]):
            cv2.line(img, P(*a), P(*b), (112, 158, 25), 1, cv2.LINE_AA, 4)
        for a, b in zip(trail_c, trail_c[1:]):
            cv2.line(img, P(*a), P(*b), (38, 89, 217), 2, cv2.LINE_AA, 4)
        m = by_meas.get(i)
        if m and m["status"] == "measured":
            cx, cy = P(m["x"], m["y"])
            cv2.line(img, (cx - 5 * S, cy), (cx + 5 * S, cy), (120, 230, 120), 1, cv2.LINE_AA, 4)
            cv2.line(img, (cx, cy - 5 * S), (cx, cy + 5 * S), (120, 230, 120), 1, cv2.LINE_AA, 4)
        rf = (F.get("d") or [d_med] * len(fi))[k] / 2
        rc = (C.get("d") or [d_med] * len(fi))[k] / 2
        if f_ok:
            cv2.circle(img, P(*trail_f[-1]), int(max(2, rf) * S), (112, 158, 25), 1, cv2.LINE_AA, 4)
        if c_ok:
            cv2.circle(img, P(*trail_c[-1]), int(max(2, rc) * S), (38, 89, 217), 2, cv2.LINE_AA, 4)
        h, w = img.shape[:2]
        bar = np.zeros((40, w, 3), np.uint8)
        bar[:] = (20, 60, 90)
        cv2.putText(bar, f"HYPOTHETICAL counterfactual ({tag}) - not observed", (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (90, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"frame {i}  t={F['t'][k]:.4f} s   orange dashed=counterfactual  teal=fitted  green=measured",
                    (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
        rendered.append(cv2.cvtColor(np.vstack([bar, img]), cv2.COLOR_BGR2RGB))
    if not rendered:
        raise PipelineError("nothing to render")
    ts = cf["frames"]["fitted"]["t"]
    dts = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    fps = Fraction(1 / statistics.median(dts)).limit_denominator(1001) if dts else Fraction(30)
    tmp = out.with_name(out.stem + ".partial.mp4")
    with _ENGINE_LOCK:
        demo.encode_mp4(rendered, tmp, fps)
    tmp.replace(out)
    return out


# ------------------------------------------------------------------ 6. reports and artifacts


def run_report(store: RunStore, run_id: str, rid: str | None = None) -> dict:
    d = store.run_dir(run_id)
    run = store.load(run_id)
    recs = [r for r in run["reconstructions"] if r["status"] == "completed"]
    rec = next((r for r in run["reconstructions"] if r["id"] == rid), None) if rid else (recs[-1] if recs else None)
    rep = None
    if rec and (d / "recon" / rec["id"] / "report.json").is_file():
        rep = _read_json(d / "recon" / rec["id"] / "report.json")
    phys = (rep or {}).get("physics") or {}
    if rep is None:
        status, category = "not_reconstructed", None
    elif "program_sha256" in phys:
        status, category = "scene_program_fitted", "fitted_reconstruction"
    else:
        status, category = "stopped_before_scene_program", "measurements_only"
    sam_res = run["sam"].get("result") or {}
    obj = (rep or {}).get("sam", {}).get("object_id")
    return {
        "kind": "decompile_web_run_report", "web_version": WEB_VERSION, "generated_utc": utc_now(), "run_id": run_id,
        "reconstruction_id": rec["id"] if rec else None,
        "result_category": category,
        "result_category_note": ("A fitted reconstruction: the scene program was fitted to the same observations it is "
                                 "compared with (consistency check). It is not a held-out prediction. Counterfactuals "
                                 "are hypothetical.") if category == "fitted_reconstruction" else None,
        "input_video": {"filename": run["original_filename"], "provenance": run["provenance"],
                        **{k: (run["video"] or {}).get(k) for k in ("sha256", "size_bytes", "codec", "duration_s",
                                                                    "frame_count", "display_width", "display_height",
                                                                    "nominal_fps")}},
        "sam": {"source": run["sam"].get("source"), "status": run["sam"]["status"], "model": sam_res.get("model"),
                "prompt": sam_res.get("prompt"), "usage": sam_res.get("usage"),
                "attempts": [{k: a.get(k) for k in ("n", "status", "usage", "actual_billed_frames", "actual_cost_usd",
                                                    "request_sent", "error")} for a in run["sam"]["attempts"]],
                "estimated_cost": (run.get("plan") or {}).get("cost")},
        "selected_object": obj,
        "tracking": (sam_res.get("summary") or {}).get("objects", {}).get(obj) if obj else None,
        "reconstruction_parameters": rec["params"] if rec else None,
        "measurement": (rep or {}).get("measurement"),
        "physics": phys or None,
        "reconstruction_consistency": (rep or {}).get("reconstruction_consistency"),
        "status": status,
        "warnings": list(run["sam"].get("warnings", [])) + list((run.get("suitability") or {}).get("warnings", [])),
        "claims": (rep or {}).get("claims", []),
        "limitations": LIMITATIONS,
    }


ARTIFACT_RULES = [
    # (predicate on relative posix path, category, label)
    (lambda p: p == "input/video.mp4", "input", "Original video"),
    (lambda p: p == "input/probe.json", "input", "Video probe and validation"),
    (lambda p: p == "input/frame_table.json", "input", "Frame table (exact presentation timestamps)"),
    (lambda p: p == "sam/plan.json", "perception", "SAM call plan and cost estimate"),
    (lambda p: p.startswith("sam/") and p.endswith("/sam_tracking.json"), "perception", "SAM 3.1 tracking result"),
    (lambda p: p.startswith("sam/") and p.endswith("/raw_stream.sse"), "perception", "Raw SAM response stream"),
    (lambda p: p.startswith("sam/") and p.endswith("/raw_output.txt"), "perception", "Raw SAM output text"),
    (lambda p: p.startswith("sam/") and p.endswith("/upload_record.json"), "perception", "SAM upload record"),
    (lambda p: p.startswith("sam/") and "/api_evidence/" in p, "perception", "API evidence (redacted)"),
    (lambda p: p.startswith("sam/") and p.endswith("/sam_validation_failure.json"), "perception",
     "SAM validation failure"),
    (lambda p: p.endswith("/measurement/sam_measurement_report.json"), "measurement", "Measurement report (per frame)"),
    (lambda p: p.endswith("/measurement/trajectory.json"), "measurement", "Trajectory (exact PTS, per-frame sigma)"),
    (lambda p: p.endswith("/contact_sheet.png"), "measurement", "Segmentation and measurement contact sheet"),
    (lambda p: p.endswith("/trajectory.png"), "measurement", "Trajectory plot"),
    (lambda p: p.endswith("/overlay.mp4") and p.startswith("recon/"), "simulation",
     "Reconstruction overlay video (SAM mask, outline, scene program)"),
    (lambda p: p.endswith("/scene_program.json"), "physics", "Physical scene program"),
    (lambda p: p.endswith("/counterfactual_program.json"), "physics", "Counterfactual scene program (engine)"),
    (lambda p: "/web/counterfactuals/" in p and p.endswith(".json"), "counterfactual",
     "Counterfactual scene program and simulation (hypothetical)"),
    (lambda p: p.endswith("/web/explain.json"), "report", "Muse Spark review of the fit"),
    (lambda p: p.endswith("/web/heldout.json"), "physics", "Held-out prediction of the later frames"),
    (lambda p: "/web/asks/" in p and p.endswith(".json") and "/api_evidence/" not in p, "counterfactual",
     "Muse Spark what-if interpretation"),
    (lambda p: "/web/asks/api_evidence/" in p, "counterfactual", "Muse Spark API evidence (redacted)"),
    (lambda p: "/web/counterfactuals/" in p and p.endswith(".mp4") and ".partial" not in p, "counterfactual",
     "Counterfactual simulation video (hypothetical)"),
    (lambda p: p.endswith("/report.md") and p.startswith("recon/"), "report", "Human-readable report"),
    (lambda p: p.endswith("/report.json") and p.startswith("recon/") and "/web/" not in p, "report",
     "Engine reconstruction report"),
    (lambda p: p.endswith("/web/run_report.json"), "report", "Machine-readable run report"),
    (lambda p: p.endswith("/web/analysis.json"), "report", "Per-frame analysis table"),
    (lambda p: p.endswith("/web/perception.json"), "perception", "SAM boxes and mask outlines per frame"),
    (lambda p: p == "events.jsonl", "report", "Processing log"),
]


def list_artifacts(run_dir: Path) -> list[dict]:
    out = []
    if not run_dir.is_dir():
        return out
    for f in sorted(run_dir.rglob("*")):
        if not f.is_file() or f.name.endswith(".tmp") or ".tmp" in f.suffixes:
            continue
        rel = f.relative_to(run_dir).as_posix()
        if "/frames/" in rel or rel.startswith("input/preview/") or rel.endswith("plan_resolved.json") or "/heldout_measure/" in rel:
            continue
        for pred, cat, label in ARTIFACT_RULES:
            if pred(rel):
                parts = rel.split("/")
                scope = parts[1] if parts[0] == "recon" else (parts[1] if parts[0] == "sam" and len(parts) > 2 else None)
                out.append({"path": rel, "category": cat, "label": label, "size_bytes": f.stat().st_size,
                            "scope": scope, "media_type": _media_type(rel)})
                break
    return out


def _media_type(rel: str) -> str:
    ext = rel.rsplit(".", 1)[-1].lower()
    return {"mp4": "video/mp4", "png": "image/png", "json": "application/json", "md": "text/markdown; charset=utf-8",
            "sse": "text/plain; charset=utf-8", "txt": "text/plain; charset=utf-8",
            "jsonl": "application/x-ndjson"}.get(ext, "application/octet-stream")


def remove_partial_upload(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except OSError:
        pass


# ------------------------------------------------------------------ held-out prediction (frames the fit never saw)


def heldout_prediction(run_dir: Path, rid: str) -> dict:
    """Predict the video AFTER the fitted window with the fitted program, and compare with Decompile's measurement of
    those later frames (never used by the fit). Measurement-only reuse of sam_demo.reconstruct; nothing is refitted."""
    rdir = recon_dir(run_dir, rid)
    out_path = rdir / "web" / "heldout.json"
    if out_path.is_file():
        return _read_json(out_path)
    data, prog, simulate = _load_program(rdir)
    if prog is None:
        raise PipelineError("This reconstruction has no scene program, so it cannot predict anything.")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rec = next(r for r in run["reconstructions"] if r["id"] == rid)
    rep = _read_json(rdir / "report.json")
    n = run["video"]["frame_count"]
    start = rep["clip"]["end_frame_exclusive"]
    end = min(n, start + demo.MAX_TRAJECTORY_FRAMES)
    if end - start < MIN_RECON_FRAMES:
        raise PipelineError("There are fewer than 4 frames after the fitted window, so there is nothing left to predict.")
    mdir = rdir / "web" / "heldout_measure"
    if mdir.exists():
        shutil.rmtree(mdir)
    with _ENGINE_LOCK:
        try:
            demo.reconstruct(video_path(run_dir), safe_child(run_dir, run["sam"]["result"]["path"]), mdir,
                             sam_object=rep["sam"]["object_id"], start_frame=start, end_frame=end,
                             outline_model=rec["params"]["outline_model"], physics_model=rec["params"]["physics_model"],
                             hfov=rec["params"]["hfov_deg"], min_confidence=rec["params"]["min_confidence"],
                             provenance="real" if run["provenance"] == "real" else "synthetic", write_video=False)
        except (demo.UsageError, demo.GateError, st.SamTrackingError) as exc:
            raise PipelineError(f"The later frames could not be measured: {exc}") from None
    meas = _read_json(mdir / "measurement" / "sam_measurement_report.json")
    frames = [f for f in meas["frames"]]
    times = [_frac(f["t"]) for f in frames]
    sim = simulate(prog, times)
    sd = _sim_dict(sim, times)
    valid_until = sd.get("valid_until_s", math.inf)
    rows, errs = [], []
    for k, f in enumerate(frames):
        valid = sd["t"][k] <= valid_until + 1e-9
        row = {"frame_index": f["frame_index"], "t_s": sd["t"][k], "status": f["status"], "reason": f["reason"],
               "measured": [f["x"], f["y"]] if f["status"] == "measured" else None,
               "sigma": [f["sigma_x"], f["sigma_y"]] if f["status"] == "measured" else None,
               "predicted": [sd["x"][k], sd["y"][k]] if valid else None,
               "predicted_diameter_px": sd["d"][k] if valid and "d" in sd else None}
        if row["measured"] and row["predicted"]:
            row["error_px"] = math.hypot(row["predicted"][0] - row["measured"][0], row["predicted"][1] - row["measured"][1])
            errs.append(row["error_px"])
        rows.append(row)
    # dense predicted curve over the held-out span (for drawing)
    dense = demo.dense_times([times[0], times[-1]], 300)
    dd = _sim_dict(simulate(prog, dense), dense)
    srt = sorted(errs)
    # where SAM's track (identity data, independent of the fit) shows the next bounce: lowest point of the box centre
    observed_bounce = None
    try:
        res = load_sam_result(run_dir, run)
        pts = sorted((i, (r.box[1] + r.box[3]) / 2) for i, r in res.frames_of(rep["sam"]["object_id"]).items() if start <= i < end)
        for k in range(2, len(pts) - 2):
            y = pts[k][1]
            if y >= pts[k - 1][1] and y >= pts[k + 1][1] and y - pts[k - 2][1] > 2 and y - pts[k + 2][1] > 2:
                fi = pts[k][0]
                observed_bounce = {"frame": fi, "t_s": float(_frac(next(f["t"] for f in frames if f["frame_index"] == fi))),
                                   "source": "lowest point of the SAM 3.1 box track"}
                break
    except (PipelineError, StopIteration, KeyError):
        observed_bounce = None
    pred_next = next((e["t_s"] for e in dd["events"] if e["t_s"] > float(times[0])), None)
    result = {
        "kind": "decompile_heldout_prediction", "recon_id": rid, "created_utc": utc_now(),
        "fitted_window": [rep["clip"]["start_frame"], start - 1], "predicted_window": [start, end - 1],
        "frames": rows, "dense": dd,
        "summary": {"n_compared": len(errs), "median_error_px": srt[len(srt) // 2] if srt else None,
                    "mean_error_px": sum(errs) / len(errs) if errs else None, "max_error_px": max(errs) if errs else None,
                    "median_diameter_px": rep["measurement"].get("median_diameter_px"),
                    "predicted_contacts_s": [e["t_s"] for e in dd["events"]][:6],
                    "predicted_next_bounce_s": pred_next, "observed_next_bounce": observed_bounce,
                    "first_frames_error_px": [r.get("error_px") for r in rows if r.get("error_px") is not None][:3]},
        "label": ("Prediction: the program was fitted ONLY to frames "
                  f"{rep['clip']['start_frame']}-{start - 1}. Frames {start}-{end - 1} were never used by the fit. Their "
                  "positions are Decompile's own measurements of those frames (not independent ground truth)."),
    }
    write_json_atomic(out_path, result)
    return result


def _pre_bounce_errors(h: dict) -> list[float]:
    nb = h["summary"].get("predicted_next_bounce_s")
    return [f["error_px"] for f in h["frames"] if f.get("error_px") is not None and (nb is None or f["t_s"] < nb)]


def scoreboard(store: RunStore) -> list[dict]:
    """One row per run: the best reconstruction's fit and held-out prediction, or why the run was refused."""
    rows = []
    for run in store.list():
        row = {"run_id": run["id"], "filename": run["original_filename"], "created_utc": run["created_utc"],
               "provenance": run["provenance"], "status": None, "reason": None}
        recs = [r for r in run.get("reconstructions", []) if r.get("status") == "completed"]
        prog = next((r for r in reversed(recs) if r.get("outcome") == "scene_program"), None)
        if run["state"] == "rejected":
            row.update(status="refused", reason="video rejected: " + "; ".join(run.get("rejection_reasons", []))[:200])
        elif run["sam"]["status"] in ("failed", "interrupted"):
            row.update(status="refused", reason="SAM tracking failed")
        elif prog is None:
            if run.get("reconstructions"):
                last = run["reconstructions"][-1]
                why = last.get("error")
                if not why and last.get("status") == "completed":
                    try:
                        why = "; ".join(_read_json(store.run_dir(run["id"]) / "recon" / last["id"] / "web" / "analysis.json")
                                        ["physics_status"].get("reasons", []))
                    except (OSError, ValueError, KeyError, TypeError):
                        why = "no executable scene program"
                row.update(status="refused", reason=(why or "no executable scene program")[:240])
            else:
                row.update(status="pending", reason=run["state"])
        else:
            rdir = store.run_dir(run["id"]) / "recon" / prog["id"]
            rep = _read_json(rdir / "report.json")
            phys = rep["physics"]
            rc = rep.get("reconstruction_consistency") or {}
            row.update(status="decompiled", recon_id=prog["id"], model=phys.get("model"),
                       restitution=phys.get("restitution"), fit_rms_px=rc.get("rms_px"),
                       fitted_frames=[rep["clip"]["start_frame"], rep["clip"]["end_frame_exclusive"] - 1],
                       ball_diameter_cm=(phys.get("implied_ball_diameter_m_if_g_9_81") or 0) * 100 or None,
                       median_diameter_px=rep["measurement"].get("median_diameter_px"))
            hp = rdir / "web" / "heldout.json"
            if hp.is_file():
                h = _read_json(hp)
                pre = _pre_bounce_errors(h)
                ob, pb = h["summary"].get("observed_next_bounce"), h["summary"].get("predicted_next_bounce_s")
                row["prediction"] = {
                    "predicted_frames": h["predicted_window"], "n_compared": h["summary"]["n_compared"],
                    "pre_bounce_error_px": [min(pre), max(pre)] if pre else None,
                    "median_error_px": h["summary"]["median_error_px"],
                    "next_bounce_error_ms": abs(pb - ob["t_s"]) * 1000 if (ob and pb) else None,
                    "observed_bounce_frame": ob["frame"] if ob else None}
        rows.append(row)
    return rows
