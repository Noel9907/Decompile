"""Decompile web application (decompile_web), OFFLINE.

Every SAM request goes to an httpx2 MockTransport injected through create_app(sam_client_factory=...); real network
access is forbidden for the whole module. The video is a test-only synthetic bounce with known truth.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

import sam_fake as fk
from decompile import sam_api as sa
from decompile import stress_render as sr
from decompile_web import pipeline as pl
from decompile_web.app import create_app
from decompile_web.config import Settings
from decompile_web.security import REDACTED, redact, safe_child, sanitize_filename
from decompile_web.store import RunStore

KEY = "sk-test-WEBAPP-SECRET-0123456789abcdef"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    fk.forbid_network(monkeypatch)


# ------------------------------------------------------------------ fixtures


def _encode(frames, path: Path) -> Path:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-video_size", f"{w}x{h}", "-framerate", "30/1", "-i", "pipe:0", "-an", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "1", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    base = [s for s in sr.stress_suite(30) if s.name == "two_tone_boundary"][0]
    v = sr.render(dataclasses.replace(base, noise_sigma=1.5))
    d = tmp_path_factory.mktemp("web_scene")
    return _encode(v.frames, d / "bounce.mp4"), v


def _sam_sse(v) -> bytes:
    size = (v.scene.camera.width, v.scene.camera.height)
    lines = []
    for t in v.truth:
        recs = [fk.disc_record(3, t.x + 1.2, t.y + 0.9, t.radius_px + 1.0, size, conf=0.93)]
        if t.frame_index < 4:
            recs.append(fk.disc_record(8, 600.0, 40.0, 6.0, size, conf=0.55))
        lines.append(fk.line(t.frame_index, recs))
    return fk.sse_bytes(fk.events(lines, usage={"video_frames_processed": 50}))


class FakeSam:
    """Records calls; responds like the documented API (upload + streamed Responses)."""

    def __init__(self, v, video: Path, *, status: int = 200) -> None:
        self.v, self.video, self.status = v, video, status
        self.calls: list[tuple[str, str]] = []
        self.constructed = 0

    def handler(self, request: httpx2.Request):
        self.calls.append((request.method, request.url.path))
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if request.url.path == "/v1/files":
            assert self.video.read_bytes()[:64] in request.read()
            return httpx2.Response(200, json={"id": "file-web-1", "bytes": self.video.stat().st_size,
                                              "purpose": "user_data", "status": "uploaded"})
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": f"boom {KEY}", "code": "server_error"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sam_sse(self.v))

    def factory(self, evidence_dir):
        self.constructed += 1
        return sa.SamClient(KEY, transport=httpx2.MockTransport(self.handler), evidence_dir=evidence_dir,
                            sleep=lambda s: None)


def make_client(tmp: Path, fake: FakeSam | None = None, **settings) -> TestClient:
    # env_file points nowhere: tests never load the real .env (and its API key) into the process
    s = Settings(data_dir=tmp / "webdata", frontend_dist=tmp / "no_dist", env_file=tmp / "no.env", **settings)
    return TestClient(create_app(s, sam_client_factory=fake.factory if fake else None))


def wait(c: TestClient, rid: str, timeout: float = 240.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{rid}").json()
        if r["job"] and r["job"]["status"] != "running":
            return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def upload(c: TestClient, video: Path, name: str = "bounce.mp4", provenance: str = "synthetic"):
    return c.post("/api/runs", files={"file": (name, video.read_bytes(), "video/mp4")}, data={"provenance": provenance})


def plan(c: TestClient, rid: str, text: str = "red ball") -> dict:
    r = c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "concept", "text": text}})
    assert r.status_code == 200, r.text
    return r.json()


def confirm(c: TestClient, rid: str, p: dict, **over):
    body = {"confirm": True, "plan_sha256": p["sha256"], "acknowledged_cost_usd": p["cost"]["expected_cost_usd"], **over}
    return c.post(f"/api/runs/{rid}/sam/confirm", json=body)


@pytest.fixture(scope="module")
def tracked(scene, tmp_path_factory):
    """One run taken through upload -> plan -> confirmed (mocked) SAM -> reconstruction with a program."""
    video, v = scene
    tmp = tmp_path_factory.mktemp("web_tracked")
    fake = FakeSam(v, video)
    os.environ.pop("MODEL_API_KEY", None)
    with make_client(tmp, fake) as c:
        r = upload(c, video)
        assert r.status_code == 201, r.text
        rid = r.json()["id"]
        wait(c, rid)
        p = plan(c, rid)
        assert confirm(c, rid, p).status_code == 202
        run = wait(c, rid)
        rc = c.post(f"/api/runs/{rid}/reconstructions", json={"end_frame": 21, "physics_model": "image_space"})
        assert rc.status_code == 202, rc.text
        run = wait(c, rid)
        yield c, rid, run, fake, tmp, video


# ------------------------------------------------------------------ upload validation


def test_upload_rejects_wrong_extension_bad_container_and_oversize(tmp_path, scene):
    video, _ = scene
    with make_client(tmp_path, max_upload_bytes=10_000) as c:
        r = c.post("/api/runs", files={"file": ("clip.mov", b"\x00" * 64, "video/quicktime")})
        assert r.status_code == 415 and "MP4" in r.json()["detail"]
        r = c.post("/api/runs", files={"file": ("clip.mp4", b"this is not a video at all", "video/mp4")})
        assert r.status_code == 415 and "ftyp" in r.json()["detail"]
        r = upload(c, video)  # larger than the 10 kB limit
        assert r.status_code == 413
        assert c.get("/api/runs").json() == []  # nothing was stored
        leftovers = [p for p in (tmp_path / "webdata").iterdir()]
        assert leftovers == [], leftovers


def test_corrupt_mp4_is_rejected_with_reasons_not_silently_accepted(tmp_path):
    junk = b"\x00\x00\x00\x18ftypisom" + os.urandom(4000)
    with make_client(tmp_path) as c:
        r = c.post("/api/runs", files={"file": ("../../evil name;rm -rf.mp4", junk, "video/mp4")})
        assert r.status_code == 201
        rid = r.json()["id"]
        assert r.json()["original_filename"] == "evil name_rm -rf.mp4"  # display only, sanitised
        run = wait(c, rid)
        assert run["state"] == "rejected" and run["rejection_reasons"]
        assert run["job"]["status"] == "failed" and run["stages"]["ingest"]["status"] == "failed"
        assert c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "concept", "text": "ball"}}).status_code == 409


def test_upload_creates_run_probe_frame_table_and_suitability(tracked, scene):
    c, rid, run, *_ = tracked
    _, v = scene
    assert run["video"]["frame_count"] == len(v.frames) and run["video"]["sha256"]
    assert run["suitability"]["suitable"] is True
    assert {ch["label"] for ch in run["suitability"]["checks"]} >= {"Codec accepted by SAM 3.1", "Enough frames to reconstruct"}
    assert run["stages"]["ingest"]["status"] == "done" and run["stages"]["timestamps"]["status"] == "done"
    ft = c.get(f"/api/runs/{rid}/frame-table").json()
    assert ft["frame_count"] == len(v.frames) and ft["t_s"][0] == 0.0
    assert c.get(f"/api/runs/{rid}/preview/0").headers["content-type"] == "image/png"
    assert c.get(f"/api/runs/{rid}/video").status_code == 200
    listed = c.get("/api/runs").json()
    assert any(x["id"] == rid for x in listed)
    assert c.get("/api/runs/web_20990101T000000Z_deadbeef").status_code == 404
    assert c.get("/api/runs/..%2F..%2Fetc").status_code == 404


# ------------------------------------------------------------------ paid API gate


def test_plan_reports_cost_and_minimum_billing_without_any_request(scene, tmp_path):
    video, v = scene
    fake = FakeSam(v, video)
    with make_client(tmp_path, fake) as c:
        rid = upload(c, video).json()["id"]
        wait(c, rid)
        p = plan(c, rid)
        assert p["cost"]["expected_billed_frames"] == max(50, len(v.frames))
        assert p["cost"]["billing_minimum_frames"] == 50 and p["live_call_made"] is False
        assert fake.calls == [] and fake.constructed == 0
        assert c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "concept", "text": ""}}).status_code == 400
        assert c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "magic"}}).status_code == 400
        bp = c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "objects", "frame": 2, "box": [10, 10, 60, 60]}})
        assert bp.status_code == 200 and bp.json()["prompt"]["kind"] == "objects"


def test_confirmation_gate_refuses_everything_but_an_explicit_matching_confirmation(scene, tmp_path):
    video, v = scene
    fake = FakeSam(v, video)
    with make_client(tmp_path, fake, max_sam_cost_usd=1.0) as c:
        rid = upload(c, video).json()["id"]
        wait(c, rid)
        no_plan = c.post(f"/api/runs/{rid}/sam/confirm", json={"confirm": True, "plan_sha256": "x", "acknowledged_cost_usd": 0.01})
        assert no_plan.status_code == 400 and "plan" in no_plan.json()["detail"]
        p = plan(c, rid)
        assert confirm(c, rid, p, confirm=False).status_code == 400
        assert confirm(c, rid, p, plan_sha256="0" * 64).status_code == 400
        assert confirm(c, rid, p, acknowledged_cost_usd=0.0).status_code == 400
        assert fake.calls == [] and fake.constructed == 0
        run = c.get(f"/api/runs/{rid}").json()
        assert run["sam"]["status"] == "not_started" and run["job"]["kind"] == "ingest"
    with make_client(tmp_path, fake, max_sam_cost_usd=0.001) as c:  # same data dir, lower limit
        r = confirm(c, rid, p)
        assert r.status_code == 400 and "limit" in r.json()["detail"]
        assert fake.calls == []


def test_live_sam_disabled_by_default_makes_no_request(scene, tmp_path, monkeypatch):
    video, _ = scene
    monkeypatch.setenv("MODEL_API_KEY", KEY)
    with make_client(tmp_path) as c:  # no injected factory, DECOMPILE_LIVE_SAM not set
        assert c.get("/api/health").json()["live_sam_enabled"] is False
        rid = upload(c, video).json()["id"]
        wait(c, rid)
        p = plan(c, rid)
        r = confirm(c, rid, p)
        assert r.status_code == 400 and "disabled" in r.json()["detail"] and "No request" in r.json()["detail"]
        assert c.get(f"/api/runs/{rid}").json()["sam"]["attempts"] == []
        health = c.get("/api/health").text
        assert KEY not in health


# ------------------------------------------------------------------ tracking, events, secrets


def test_confirmed_tracking_runs_the_existing_client_and_records_actual_usage(tracked):
    c, rid, run, fake, tmp, _ = tracked
    assert fake.calls[:2] == [("POST", "/v1/files"), ("POST", "/v1/responses")]
    assert sum(1 for m in fake.calls if m[1] == "/v1/responses") == 1
    assert run["sam"]["status"] == "completed" and run["sam"]["source"] == "live"
    a = run["sam"]["attempts"][0]
    assert a["status"] == "completed" and a["usage"] == {"video_frames_processed": 50}
    assert a["actual_billed_frames"] == 50 and a["actual_cost_usd"] == pytest.approx(0.01)
    for k in ("sam_upload", "sam_segmentation", "sam_validation"):
        assert run["stages"][k]["status"] == "done", k
    tracks = c.get(f"/api/runs/{rid}/sam/tracks").json()
    assert set(tracks["objects"]) == {"3", "8"}  # both SAM objects are offered for selection, no new call
    assert tracks["summary"]["3"]["n_frames"] > tracks["summary"]["8"]["n_frames"]


def test_progress_events_are_persisted_and_streamed_with_resume(tracked):
    c, rid, *_ = tracked
    r = c.get(f"/api/runs/{rid}/events?once=1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
    stages = [(e["stage"], e["status"]) for e in evs if e["type"] == "stage"]
    for s in ("ingest", "sam_upload", "sam_segmentation", "measurement", "physics", "simulation", "rendering"):
        assert (s, "done") in stages, s
    assert any(e["type"] == "confirmation" for e in evs)
    resumed = c.get(f"/api/runs/{rid}/events?once=1", headers={"Last-Event-ID": str(evs[-3]["seq"])})
    tail = [json.loads(line[6:]) for line in resumed.text.splitlines() if line.startswith("data: ")]
    assert [e["seq"] for e in tail] == [evs[-2]["seq"], evs[-1]["seq"]]


def test_no_secret_in_any_file_response_or_event(tracked):
    c, rid, run, fake, tmp, _ = tracked
    for p in (tmp / "webdata").rglob("*"):
        if p.is_file():
            assert KEY.encode() not in p.read_bytes(), p
    for path in (f"/api/runs/{rid}", f"/api/runs/{rid}/events?once=1", f"/api/runs/{rid}/report", "/api/health",
                 f"/api/runs/{rid}/artifacts"):
        assert KEY not in c.get(path).text, path
    evidence = [a["path"] for a in c.get(f"/api/runs/{rid}/artifacts").json() if "api_evidence" in a["path"]]
    assert evidence  # the client's redacted evidence is downloadable


def test_failed_sam_request_is_reported_not_retried_and_billing_is_explained(scene, tmp_path):
    video, v = scene
    fake = FakeSam(v, video, status=500)
    with make_client(tmp_path, fake) as c:
        rid = upload(c, video).json()["id"]
        wait(c, rid)
        assert confirm(c, rid, plan(c, rid)).status_code == 202
        run = wait(c, rid)
        assert run["state"] == "tracking_failed" and run["sam"]["status"] == "failed"
        a = run["sam"]["attempts"][0]
        assert a["status"] == "failed" and a["request_sent"] is True and "NOT retried" in a["billing_note"]
        assert sum(1 for m in fake.calls if m[1] == "/v1/responses") == 1
        assert run["stages"]["sam_segmentation"]["status"] == "failed"
        assert KEY not in json.dumps(run) and REDACTED in json.dumps(run)
        assert run["job"]["status"] == "failed" and "NOT retried" in run["job"]["error"]
        assert c.post(f"/api/runs/{rid}/reconstructions", json={}).status_code == 409


def test_interrupted_jobs_are_marked_on_restart(tmp_path):
    store = RunStore(tmp_path / "webdata")
    run = pl.new_run_record("web_20261105T000000Z_0badf00d", "x.mp4", 10, "real")
    run.update({"state": "tracking", "job": {"kind": "sam_tracking", "status": "running"}})
    run["sam"]["status"] = "running"
    run["stages"]["sam_segmentation"]["status"] = "running"
    store.create(run)
    with make_client(tmp_path) as c:
        r = c.get("/api/runs/web_20261105T000000Z_0badf00d").json()
    assert r["job"]["status"] == "interrupted" and r["sam"]["status"] == "interrupted"
    assert r["stages"]["sam_segmentation"]["status"] == "interrupted"
    assert any("NOT retried" in w for w in r["sam"]["warnings"])


# ------------------------------------------------------------------ reconstruction, counterfactuals, artifacts


def test_reconstruction_produces_program_measurements_and_views(tracked, scene):
    c, rid, run, *_ = tracked
    _, v = scene
    rec = run["reconstructions"][0]
    assert rec["status"] == "completed" and rec["outcome"] == "scene_program"
    assert all(s["status"] in ("done", "skipped") for s in rec["stages"].values()), rec["stages"]
    data = c.get(f"/api/runs/{rid}/reconstructions/r1").json()["analysis"]
    assert data["report"]["sam"]["object_id"] == "3"
    assert len(data["frames"]) == 21 and {f["status"] for f in data["frames"]} <= {"measured", "uncertain", "missing"}
    assert data["program_parameters"]["restitution"] == pytest.approx(0.65, abs=0.1)
    assert len(data["fitted"]["x"]) == len(data["fitted"]["frame_index"])
    per = c.get(f"/api/runs/{rid}/reconstructions/r1/perception").json()
    assert per["selected_object"] == "3" and per["frames"]["0"][0]["contours"]
    assert c.get(f"/api/runs/{rid}/reconstructions/r1/frames/0").headers["content-type"] == "image/png"
    assert c.get(f"/api/runs/{rid}/reconstructions/r9/perception").status_code == 400
    rep = c.get(f"/api/runs/{rid}/report").json()
    assert rep["status"] == "scene_program_fitted" and rep["result_category"] == "fitted_reconstruction"
    assert rep["selected_object"] == "3" and rep["limitations"]


def test_counterfactual_changes_only_requested_parameters_and_is_deterministic(tracked):
    c, rid, *_ = tracked
    url = f"/api/runs/{rid}/reconstructions/r1/counterfactuals"
    a = c.post(url, json={"restitution": 0.2, "extend_s": 0.5}).json()
    b = c.post(url, json={"restitution": 0.9, "gravity_scale": 0.5, "extend_s": 0.5}).json()
    again = c.post(url, json={"restitution": 0.2, "extend_s": 0.5}).json()
    assert a["id"] == again["id"] and a["dense"] == again["dense"]
    assert a["counterfactual_parameters"]["restitution"] == 0.2 and b["counterfactual_parameters"]["restitution"] == 0.9
    assert a["fitted_parameters"] == b["fitted_parameters"]
    ga, gb = a["fitted_parameters"]["acceleration_px_s2"], b["counterfactual_parameters"]["acceleration_px_s2"]
    assert gb == pytest.approx([x * 0.5 for x in ga])
    assert a["dense"]["fitted"]["y"] == b["dense"]["fitted"]["y"]  # the fitted program is unchanged
    assert a["dense"]["counterfactual"]["y"] != b["dense"]["counterfactual"]["y"]
    assert a["dense"]["fitted"]["t"][-1] == pytest.approx(a["observed_until_s"] + 0.5, abs=1e-3)
    assert "HYPOTHETICAL" in a["label"]
    for bad in ({"restitution": 1.5}, {"gravity_scale": 0.0}, {"extend_s": 10}):
        assert c.post(url, json=bad).status_code == 400, bad
    v = c.post(f"{url}/{a['id']}/video")
    assert v.status_code == 200
    arts = {x["path"] for x in c.get(f"/api/runs/{rid}/artifacts").json()}
    assert v.json()["path"] in arts and f"recon/r1/web/counterfactuals/{a['id']}.json" in arts


def test_missing_scene_program_is_reported_without_fabrication(tracked):
    c, rid, *_ = tracked
    r = c.post(f"/api/runs/{rid}/reconstructions", json={"end_frame": 9, "physics_model": "image_space"})
    assert r.status_code == 202
    run = wait(c, rid)
    rec = next(x for x in run["reconstructions"] if x["id"] == r.json()["recon_id"])
    assert rec["status"] == "completed" and rec["outcome"] == "measurements_only"
    assert rec["stages"]["scene_program"]["status"] in ("failed", "skipped")
    data = c.get(f"/api/runs/{rid}/reconstructions/{rec['id']}").json()["analysis"]
    assert data["program"] is None and data["fitted"] is None and data["physics_status"]["status"] == "no_program"
    assert data["physics_status"]["reasons"]
    cf = c.post(f"/api/runs/{rid}/reconstructions/{rec['id']}/counterfactuals", json={"restitution": 0.5})
    assert cf.status_code == 400 and "no executable scene program" in cf.json()["detail"]
    arts = {x["path"] for x in c.get(f"/api/runs/{rid}/artifacts").json()}
    assert f"recon/{rec['id']}/scene_program.json" not in arts and f"recon/{rec['id']}/measurement/sam_measurement_report.json" in arts
    rep = c.get(f"/api/runs/{rid}/report", params={"recon_id": rec["id"]}).json()
    assert rep["status"] == "stopped_before_scene_program" and rep["result_category"] == "measurements_only"


def test_reconstruction_parameters_are_validated(tracked):
    c, rid, *_ = tracked
    for bad in ({"start_frame": 10, "end_frame": 5}, {"end_frame": 2}, {"physics_model": "magic"},
                {"sam_object": "42"}, {"hfov_deg": 500}):
        r = c.post(f"/api/runs/{rid}/reconstructions", json=bad)
        assert r.status_code == 400, (bad, r.text)


def test_artifacts_list_only_existing_files_and_downloads_are_confined(tracked):
    c, rid, run, fake, tmp, video = tracked
    arts = c.get(f"/api/runs/{rid}/artifacts").json()
    paths = {a["path"] for a in arts}
    for need in ("input/video.mp4", "sam/plan.json", "recon/r1/scene_program.json", "recon/r1/overlay.mp4",
                 "recon/r1/trajectory.png", "recon/r1/contact_sheet.png", "recon/r1/report.md",
                 "recon/r1/web/run_report.json", "recon/r1/measurement/sam_measurement_report.json"):
        assert need in paths, need
    assert not any("/frames/" in p for p in paths) and "run.json" not in paths
    rd = tmp / "webdata" / rid
    for a in arts:
        assert (rd / a["path"]).is_file()
    d = c.get(f"/api/runs/{rid}/files/recon/r1/scene_program.json", params={"download": 1})
    assert d.status_code == 200 and "attachment" in d.headers["content-disposition"]
    assert json.loads(d.content) == json.loads((rd / "recon/r1/scene_program.json").read_bytes())
    rng = c.get(f"/api/runs/{rid}/files/input/video.mp4", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100
    assert rng.content == video.read_bytes()[:100]
    for bad in ("run.json", "../run.json", "input/../run.json", "..%2F..%2F.env", "C:/Windows/win.ini",
                "recon/r1/frames/frame_000000.png"):
        assert c.get(f"/api/runs/{rid}/files/{bad}").status_code == 404, bad


def test_import_of_existing_sam_result_is_verified_against_the_video(tracked, scene, tmp_path):
    c0, rid0, run0, fake, tmp0, video = tracked
    sam_bytes = (tmp0 / "webdata" / rid0 / run0["sam"]["result"]["path"]).read_bytes()
    with make_client(tmp_path) as c:
        rid = upload(c, video).json()["id"]
        wait(c, rid)
        bad = json.loads(sam_bytes)
        bad["video"]["sha256"] = "0" * 64
        r = c.post(f"/api/runs/{rid}/sam/import", files={"file": ("s.json", json.dumps(bad).encode(), "application/json")})
        assert r.status_code == 400 and "different video" in r.json()["detail"]
        r = c.post(f"/api/runs/{rid}/sam/import", files={"file": ("s.json", b"{not json", "application/json")})
        assert r.status_code == 400
        r = c.post(f"/api/runs/{rid}/sam/import", files={"file": ("s.json", sam_bytes, "application/json")})
        assert r.status_code == 200 and r.json()["sam"]["source"] == "imported" and r.json()["state"] == "tracked"
        assert r.json()["stages"]["sam_upload"]["status"] == "skipped"
        assert c.post(f"/api/runs/{rid}/sam/import", files={"file": ("s.json", sam_bytes, "application/json")}).status_code == 400


# ------------------------------------------------------------------ unit: security helpers


def test_redaction_and_filename_and_path_helpers(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", "abcd-SECRET-value-1234")
    assert redact({"m": "key abcd-SECRET-value-1234 here", "l": ["Bearer xyz123456789"]}) == \
        {"m": f"key {REDACTED} here", "l": [f"Bearer {REDACTED}"]}
    assert sanitize_filename("C:\\Users\\x\\$(whoami);.mp4") == "_(whoami)_.mp4"
    assert sanitize_filename("") == "video.mp4" and sanitize_filename("..") == "video.mp4"
    assert safe_child(tmp_path, "a/b.json") == (tmp_path / "a" / "b.json").resolve()
    for bad in ("../x", "/etc/passwd", "a/../../x", "C:/x", "a\\..\\..\\x", ""):
        with pytest.raises(ValueError):
            safe_child(tmp_path, bad)


def test_decompiled_source_is_runnable_and_reproduces_the_engine_program(tracked):
    """The 'decompiled source' shown in the UI must build the same program and simulate identically."""
    from fractions import Fraction

    from decompile import scene_program as sp
    from decompile_web.source import program_source

    c, rid, run, fake, tmp, _ = tracked
    src = c.get(f"/api/runs/{rid}/reconstructions/r1/source").json()
    assert "sp.SceneProgram(" in src["display"] and src["filename"] == "decompiled_r1.py"
    data = json.loads((tmp / "webdata" / rid / "recon/r1/scene_program.json").read_bytes())
    ns: dict = {}
    exec(compile(src["exact"], "decompiled.py", "exec"), ns)  # noqa: S102 - generated by our own code under test
    ref = sp.program_from_json(data)
    times = [ref.body.t0 + Fraction(k, 100) for k in range(100)]
    a, b = ns["sim"], sp.simulate(ref, times)
    assert a.x_px == b.x_px and a.y_px == b.y_px and len(a.events) == len(b.events)
    cf = c.post(f"/api/runs/{rid}/reconstructions/r1/counterfactuals", json={"restitution": 0.3}).json()
    assert "COUNTERFACTUAL" in cf["source"]["display"] and "HYPOTHETICAL" in cf["source"]["display"]
    ns2: dict = {}
    exec(compile(cf["source"]["exact"], "cf.py", "exec"), ns2)  # noqa: S102
    assert ns2["program"].planes[0].restitution == 0.3


def test_projective_source_round_trips(tmp_path):
    from fractions import Fraction

    from decompile import projective_program as pp
    from decompile_web.source import program_source

    prog = pp.ProjectiveProgram("obj_01", Fraction(20987, 22500), (424.0, 239.0), 652.9, (2.03, -0.62, 12.13),
                                (1.18, -20.05, -11.93), (2.42, 120.58, 74.64), (-0.0171, -0.8501, -0.5263), -12.149,
                                0.679, (-0.0075, 0.0113, -0.0181), {"contact_gap_frames": [39, 44], "hfov_used_deg": 66.0})
    data = pp.program_to_json(prog)
    ns: dict = {}
    exec(compile(program_source(data, exact=True), "p.py", "exec"), ns)  # noqa: S102
    times = [prog.t0 + Fraction(k, 100) for k in range(100)]
    ref = pp.simulate(prog, times)
    assert ns["sim"].x_px == ref.x_px and ns["sim"].y_px == ref.y_px and ns["sim"].diameter_px == ref.diameter_px
    assert "bounce in frames 39-44" in program_source(data)


class _FakeSpark:
    def __init__(self, parsed):
        self.parsed, self.calls = parsed, []

    def chat(self, messages, **kw):
        from decompile import meta_api
        self.calls.append((messages, kw))
        return meta_api.ChatResult(content=json.dumps(self.parsed), parsed=self.parsed, finish_reason="stop",
                                   model="muse-spark-test", response_id="r1", request_id=None, usage=meta_api.Usage(),
                                   latency_s=0.01, attempts=1, evidence_path=Path("x"))


def test_ask_maps_language_to_supported_parameters_and_simulates(tracked, tmp_path):
    from decompile_web import pipeline as pl

    c, rid, run, fake, tmp, _ = tracked
    rd = tmp / "webdata" / rid
    moon = _FakeSpark({"supported": True, "restitution": None, "gravity_scale": 0.165, "extend_s": 1.5,
                       "headline": "Moon gravity", "reasoning": "r", "assumptions": [], "unsupported_reason": None})
    out = pl.ask_whatif(rd, "r1", "What if this happened on the Moon?", lambda ev: moon)
    assert out["interpretation"]["gravity_scale"] == 0.165 and out["counterfactual"]["parameters"]["gravity_scale"] == 0.165
    assert moon.calls[0][1]["json_schema"]["additionalProperties"] is False
    again = pl.ask_whatif(rd, "r1", "what if this  happened on the moon?", lambda ev: (_ for _ in ()).throw(AssertionError))
    assert again["interpretation"] == out["interpretation"]  # cached: no second request
    wind = _FakeSpark({"supported": False, "restitution": 0.2, "gravity_scale": None, "extend_s": 0, "headline": "wind",
                       "reasoning": "r", "assumptions": [], "unsupported_reason": "needs drag"})
    out = pl.ask_whatif(rd, "r1", "add wind", lambda ev: wind)
    assert out["counterfactual"] is None and out["interpretation"]["restitution"] is None
    bad = _FakeSpark({"supported": True, "restitution": 7, "gravity_scale": None, "extend_s": 0, "headline": "x",
                      "reasoning": "r", "assumptions": [], "unsupported_reason": None})
    with pytest.raises(pl.PipelineError, match="outside"):
        pl.ask_whatif(rd, "r1", "bounce harder than possible", lambda ev: bad)
    with make_client(tmp_path) as c2:  # no key, no factory: refused without a request
        assert c2.post(f"/api/runs/{rid}/reconstructions/r1/ask", json={"question": "moon"}).status_code in (400, 404)


def test_new_whatif_knobs_change_the_program_and_its_source(tracked):
    c, rid, *_ = tracked
    url = f"/api/runs/{rid}/reconstructions/r1/counterfactuals"
    base = c.post(url, json={"extend_s": 0.5}).json()
    nofloor = c.post(url, json={"extend_s": 0.5, "remove_floor": True}).json()
    harder = c.post(url, json={"extend_s": 0.5, "velocity_scale": 2.0}).json()
    assert nofloor["counterfactual_program"]["planes"] == [] and "floor removed" in nofloor["source"]["display"]
    assert len(nofloor["dense"]["counterfactual"]["events"]) == 0 < len(base["dense"]["counterfactual"]["events"])
    assert max(nofloor["dense"]["counterfactual"]["y"]) > max(base["dense"]["counterfactual"]["y"])  # keeps falling (+y down)
    b0, b1 = base["counterfactual_program"]["body"], harder["counterfactual_program"]["body"]
    assert b1["vx0"] == pytest.approx(2 * b0["vx0"]) and b1["vy0"] == pytest.approx(2 * b0["vy0"])
    assert "thrown x2" in harder["source"]["display"]
    assert c.post(url, json={"velocity_scale": 9}).status_code == 400


def test_heldout_prediction_uses_only_frames_after_the_fitted_window(tracked):
    c, rid, *_ = tracked
    r = c.post(f"/api/runs/{rid}/reconstructions/r1/predict")
    assert r.status_code == 200, r.text
    h = r.json()
    assert h["fitted_window"] == [0, 20] and h["predicted_window"][0] == 21
    assert all(f["frame_index"] >= 21 for f in h["frames"])
    assert h["summary"]["n_compared"] > 0 and h["summary"]["median_error_px"] is not None
    assert "never used by the fit" in h["label"]


def test_explain_is_commentary_on_measured_numbers(tracked):
    from decompile_web import pipeline as pl

    c, rid, run, fake, tmp, _ = tracked
    review = {"headline": "h", "explanation": "e", "material_guess": "m", "plausibility": "p", "critique": ["c"],
              "confidence": "medium"}
    spark = _FakeSpark(review)
    out = pl.explain_fit(tmp / "webdata" / rid, "r1", lambda ev: spark)
    assert out["confidence"] == "medium" and "did not change" in out["authority"]
    sent = json.loads(spark.calls[0][0][1]["content"].split("\n", 1)[1])
    assert sent["restitution"] == pytest.approx(run_restitution(tmp, rid)) and "verdict_glossary" in sent
    again = pl.explain_fit(tmp / "webdata" / rid, "r1", lambda ev: (_ for _ in ()).throw(AssertionError))
    assert again["headline"] == "h"  # cached


def run_restitution(tmp, rid):
    return json.loads((tmp / "webdata" / rid / "recon/r1/scene_program.json").read_bytes())["planes"][0]["restitution"]


def test_scoreboard_reports_predictions_and_refusals(tracked):
    c, rid, *_ = tracked
    c.post(f"/api/runs/{rid}/reconstructions/r1/predict")
    rows = {r["run_id"]: r for r in c.get("/api/scoreboard").json()}
    row = rows[rid]
    assert row["status"] == "decompiled" and row["restitution"] == pytest.approx(0.65, abs=0.1)
    assert row["prediction"]["n_compared"] > 0
    assert all(r["status"] in ("decompiled", "refused", "pending") for r in rows.values())
