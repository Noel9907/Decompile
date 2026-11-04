"""HTTP API (documented in web/API.md). Run with:  python -m decompile_web  (see README_WEB.md)."""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decompile import sam_api as sa

from . import WEB_VERSION
from . import pipeline as pl
from .config import Settings
from .jobs import JobConflict, JobRunner
from .security import sanitize_filename, safe_child
from .store import RunNotFound, RunStore, new_run_id

CHUNK = 1 << 20


class PromptIn(BaseModel):
    kind: str
    text: str | None = None
    frame: int | None = None
    box: list[float] | None = None
    points: list[dict] = Field(default_factory=list)
    propagation: str = "both"


class PlanIn(BaseModel):
    prompt: PromptIn


class ConfirmIn(BaseModel):
    confirm: bool
    plan_sha256: str
    acknowledged_cost_usd: float


class ReconIn(BaseModel):
    sam_object: str = "most_frames"
    start_frame: int = 0
    end_frame: int | None = None
    physics_model: str = "projective"
    outline_model: str = "ellipse"
    hfov_deg: float = 66.0
    min_confidence: float = 0.5


class CounterfactualIn(BaseModel):
    restitution: float | None = None
    gravity_scale: float | None = None
    extend_s: float = 0.0


def create_app(settings: Settings | None = None, *, sam_client_factory: Callable | None = None) -> FastAPI:
    """``sam_client_factory(evidence_dir) -> SamClient``: injected by tests (mock transport). Production uses the
    real client from tools/sam_demo.py, which reads MODEL_API_KEY from the server environment / .env."""
    settings = settings or Settings()
    settings.load_env_file()
    store = RunStore(settings.data_dir)
    jobs = JobRunner(store, settings.workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.recover()
        yield
        jobs.shutdown()

    app = FastAPI(title="Decompile", version=WEB_VERSION, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.store, app.state.jobs, app.state.settings = store, jobs, settings

    @app.exception_handler(pl.PipelineError)
    async def _pipeline_error(_, exc: pl.PipelineError):
        return JSONResponse(status_code=400, content={"detail": str(exc), "stage": exc.stage, "details": exc.details})

    @app.exception_handler(RunNotFound)
    async def _not_found(_, exc):
        return JSONResponse(status_code=404, content={"detail": "run not found"})

    @app.exception_handler(JobConflict)
    async def _conflict(_, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def _run(run_id: str) -> dict:
        return store.load(run_id)

    def _require_dir(run_id: str) -> Path:
        if not store.exists(run_id):
            raise RunNotFound(run_id)
        return store.run_dir(run_id)

    # ------------------------------------------------------------ meta

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": WEB_VERSION, **settings.public(),
                "pricing": {"usd_per_1000_frames": sa.PRICE_USD_PER_1000_FRAMES,
                            "billing_minimum_frames": sa.BILLING_MIN_FRAMES, "source": sa.PRICING_SOURCE},
                "accepted_extensions": list(pl.ACCEPTED_UPLOAD_EXTENSIONS),
                "max_window_frames": pl.demo.MAX_TRAJECTORY_FRAMES,
                "stages": {"tracking": pl.TRACKING_STAGES, "reconstruction": pl.RECON_STAGES},
                "limitations": pl.LIMITATIONS}

    # ------------------------------------------------------------ runs

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return [{k: r.get(k) for k in ("id", "created_utc", "updated_utc", "original_filename", "state", "provenance")}
                | {"frames": (r.get("video") or {}).get("frame_count"),
                   "reconstructions": len(r.get("reconstructions", [])),
                   "has_program": any(x.get("outcome") == "scene_program" for x in r.get("reconstructions", []))}
                for r in store.list()]

    @app.post("/api/runs", status_code=201)
    async def create_run(request: Request, file: UploadFile = File(...), provenance: str = Form("real")) -> dict:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.max_upload_bytes + 1024 * 1024:
            raise HTTPException(413, f"file too large (limit {settings.max_upload_bytes // (1024 * 1024)} MB)")
        if provenance not in pl.PROVENANCES:
            raise HTTPException(400, f"provenance must be one of {pl.PROVENANCES}")
        name = sanitize_filename(file.filename)
        ext = Path(name).suffix.lower()
        if ext not in pl.ACCEPTED_UPLOAD_EXTENSIONS:
            raise HTTPException(415, f"unsupported file type {ext or '(none)'}: upload an MP4 video")
        head = await file.read(12)
        if len(head) < 12 or head[4:8] != b"ftyp":
            raise HTTPException(415, "this file is not an MP4 container (no 'ftyp' header)")
        run_id = new_run_id()
        rd = store.run_dir(run_id)
        part = rd.parent / f".{run_id}.upload.part"
        size = len(head)
        try:
            with open(part, "wb") as f:
                f.write(head)
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"file too large (limit {settings.max_upload_bytes // (1024 * 1024)} MB)")
                    f.write(chunk)
        except BaseException:
            pl.remove_partial_upload(part)
            raise
        store.create(pl.new_run_record(run_id, name, size, provenance))
        (rd / "input").mkdir(parents=True, exist_ok=True)
        part.replace(pl.video_path(rd))
        store.append_event(run_id, {"type": "upload", "bytes": size, "filename": name})
        ctx = pl.Ctx(store, run_id)
        jobs.submit(run_id, "ingest", lambda: pl.ingest_job(ctx))
        return _run(run_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return _run(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = 0, once: bool = False):
        _require_dir(run_id)
        lei = request.headers.get("last-event-id")
        if lei and lei.isdigit():
            after = max(after, int(lei))

        async def gen():
            nonlocal after
            idle = 0.0
            yield "retry: 2000\n\n"
            while True:
                evs = await run_in_threadpool(store.read_events, run_id, after)
                for ev in evs:
                    after = ev["seq"]
                    yield f"id: {ev['seq']}\ndata: {json.dumps(ev)}\n\n"
                if once:
                    return
                if await request.is_disconnected():
                    return
                idle = 0.0 if evs else idle + 0.4
                if idle >= 15.0:
                    idle = 0.0
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/video")
    def original_video(run_id: str):
        p = pl.video_path(_require_dir(run_id))
        if not p.is_file():
            raise HTTPException(404, "no video")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/frame-table")
    def get_frame_table(run_id: str) -> dict:
        d = _require_dir(run_id)
        if not (d / "input" / "frame_table.json").is_file():
            raise HTTPException(404, "frame table not ready")
        t = pl.frame_table(d)
        return {"frame_count": t["frame_count"], "width": t["width"], "height": t["height"],
                "time_base": t["time_base"], "t": [str(f["t"]) for f in t["frames"]],
                "t_s": [float(Fraction(f["t"])) for f in t["frames"]]}

    @app.get("/api/runs/{run_id}/preview/{index}")
    def preview(run_id: str, index: int):
        d = _require_dir(run_id)
        if not (d / "input" / "frame_table.json").is_file():
            raise HTTPException(404, "not ready")
        p = pl.preview_frame(d, index)
        return FileResponse(p, media_type="image/png")

    # ------------------------------------------------------------ SAM

    @app.post("/api/runs/{run_id}/plan")
    def plan(run_id: str, body: PlanIn) -> dict:
        run = _run(run_id)
        if run["state"] not in ("uploaded", "planned", "tracking_failed"):
            raise HTTPException(409, f"cannot plan in state {run['state']!r}")
        if run["sam"]["status"] in ("running", "completed"):
            raise HTTPException(409, "this run already has a SAM result or a running request")
        return pl.build_plan(store, run_id, body.prompt.model_dump())

    @app.post("/api/runs/{run_id}/sam/confirm", status_code=202)
    def confirm_sam(run_id: str, body: ConfirmIn) -> dict:
        ctx = pl.Ctx(store, run_id)

        def check(run: dict) -> None:
            pl.check_confirmation(run, plan_sha256=body.plan_sha256, acknowledged_cost_usd=body.acknowledged_cost_usd,
                                  confirm=body.confirm, live_enabled=settings.live_sam_enabled or
                                  sam_client_factory is not None, max_cost_usd=settings.max_sam_cost_usd,
                                  api_key_configured=settings.api_key_configured() or sam_client_factory is not None)
            store.append_event(run_id, {"type": "confirmation", "plan_sha256": body.plan_sha256,
                                        "acknowledged_cost_usd": body.acknowledged_cost_usd})

        jobs.submit(run_id, "sam_tracking", lambda: pl.tracking_job(ctx, sam_client_factory), before=check)
        return _run(run_id)

    @app.post("/api/runs/{run_id}/sam/import")
    async def import_sam(run_id: str, file: UploadFile = File(...)) -> dict:
        _require_dir(run_id)
        raw = await file.read(settings.max_import_bytes + 1)
        if len(raw) > settings.max_import_bytes:
            raise HTTPException(413, "SAM result file too large")
        job = _run(run_id).get("job")
        if job and job.get("status") == "running":
            raise HTTPException(409, "a job is running for this run")
        return await run_in_threadpool(pl.import_sam_result, store, run_id, raw)

    @app.get("/api/runs/{run_id}/sam/tracks")
    def sam_tracks(run_id: str) -> dict:
        d = _require_dir(run_id)
        return pl.sam_tracks(d, _run(run_id))

    # ------------------------------------------------------------ reconstruction

    @app.post("/api/runs/{run_id}/reconstructions", status_code=202)
    def reconstruct(run_id: str, body: ReconIn) -> dict:
        d = _require_dir(run_id)
        run = _run(run_id)
        if run["sam"]["status"] != "completed":
            raise HTTPException(409, "a completed SAM tracking result is required first")
        ids = tuple(pl.sam_tracks(d, run)["objects"])
        params = pl.validate_recon_params(run, body.model_dump(), ids)
        holder = {}

        def before(r: dict) -> None:
            n = len(r["reconstructions"]) + 1
            holder["rid"] = f"r{n}"
            r["reconstructions"].append({"id": holder["rid"], "params": params, "status": "queued",
                                         "created_utc": pl.utc_now(), "finished_utc": None, "error": None,
                                         "outcome": None, "stages": pl.empty_stages(pl.RECON_STAGES)})

        def job():
            pl.reconstruction_job(pl.Ctx(store, run_id, holder["rid"]))

        jobs.submit(run_id, "reconstruction", job, before=before)
        return {"recon_id": holder["rid"], "run": _run(run_id)}

    @app.get("/api/runs/{run_id}/reconstructions/{rid}")
    def get_recon(run_id: str, rid: str) -> dict:
        d = _require_dir(run_id)
        rdir = pl.recon_dir(d, rid)
        rec = next((r for r in _run(run_id)["reconstructions"] if r["id"] == rid), None)
        a = rdir / "web" / "analysis.json"
        if not a.is_file():
            return {"reconstruction": rec, "analysis": None}
        return {"reconstruction": rec, "analysis": pl._read_json(a)}

    @app.get("/api/runs/{run_id}/reconstructions/{rid}/perception")
    def get_perception(run_id: str, rid: str) -> dict:
        p = pl.recon_dir(_require_dir(run_id), rid) / "web" / "perception.json"
        if not p.is_file():
            raise HTTPException(404, "not ready")
        return pl._read_json(p)

    @app.get("/api/runs/{run_id}/reconstructions/{rid}/frames/{index}")
    def recon_frame(run_id: str, rid: str, index: int):
        rdir = pl.recon_dir(_require_dir(run_id), rid)
        p = rdir / "frames" / pl.fr.PNG_PATTERN.format(index=index)
        if index < 0 or not p.is_file():
            raise HTTPException(404, "no such frame")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=86400, immutable"})

    @app.post("/api/runs/{run_id}/reconstructions/{rid}/counterfactuals")
    def make_counterfactual(run_id: str, rid: str, body: CounterfactualIn) -> dict:
        return pl.counterfactual(_require_dir(run_id), rid, restitution=body.restitution,
                                 gravity_scale=body.gravity_scale, extend_s=body.extend_s)

    @app.get("/api/runs/{run_id}/reconstructions/{rid}/counterfactuals/{cf_id}")
    def get_cf(run_id: str, rid: str, cf_id: str) -> dict:
        return pl.get_counterfactual(_require_dir(run_id), rid, cf_id)

    @app.post("/api/runs/{run_id}/reconstructions/{rid}/counterfactuals/{cf_id}/video")
    def render_cf(run_id: str, rid: str, cf_id: str) -> dict:
        d = _require_dir(run_id)
        p = pl.render_counterfactual_video(d, rid, cf_id)
        rel = p.relative_to(d).as_posix()
        store.append_event(run_id, {"type": "artifact", "path": rel})
        return {"path": rel}

    # ------------------------------------------------------------ reports and artifacts

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str, recon_id: str | None = None) -> dict:
        _require_dir(run_id)
        return pl.run_report(store, run_id, recon_id)

    @app.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> list[dict]:
        return pl.list_artifacts(_require_dir(run_id))

    @app.get("/api/runs/{run_id}/files/{rel:path}")
    def download(run_id: str, rel: str, download: bool = False):
        d = _require_dir(run_id)
        listed = {a["path"]: a for a in pl.list_artifacts(d)}
        if rel not in listed:
            raise HTTPException(404, "no such artifact")
        try:
            p = safe_child(d, rel)
        except ValueError:
            raise HTTPException(404, "no such artifact") from None
        name = f"{run_id}_{rel.replace('/', '_')}"
        return FileResponse(p, media_type=listed[rel]["media_type"],
                            filename=name if download else None,
                            content_disposition_type="attachment" if download else "inline")

    # ------------------------------------------------------------ frontend (built bundle)

    dist = settings.frontend_dist
    if dist.is_dir() and (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full:path}", include_in_schema=False)
        def spa(full: str):
            if full.startswith("api/"):
                raise HTTPException(404)
            try:
                f = safe_child(dist, full) if full else dist / "index.html"
            except ValueError:
                f = dist / "index.html"
            return FileResponse(f if f.is_file() else dist / "index.html")

    return app
