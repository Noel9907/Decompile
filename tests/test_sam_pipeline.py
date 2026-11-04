"""SAM 3.1 end to end, OFFLINE, on a test-only SYNTHETIC MP4 with exactly known centres:

plan (no request) -> track (mocked HTTP: upload + streamed Responses; official parser) -> reconstruct (outline
measurement, events, scene program, simulation, counterfactual, overlays) -> pre-registered run -> reveal.
SAM masks are deliberately biased; the pipeline must measure the centre from the pixels, not the mask.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import httpx2
import numpy as np
import pytest

import sam_fake as fk
from decompile import frames as fr
from decompile import sam_api as sa
from decompile import sam_measure as sm
from decompile import stress_render as sr

ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-test-SECRET-pipeline"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load("sam_demo_tool", ROOT / "tools" / "sam_demo.py")
rvd = _load("real_video_demo_sam", ROOT / "tools" / "real_video_demo.py")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    fk.forbid_network(monkeypatch)


def encode(frames, path: Path) -> Path:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-video_size", f"{w}x{h}", "-framerate", "30/1", "-i", "pipe:0", "-an", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "1",
           "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    base = [s for s in sr.stress_suite(30) if s.name == "two_tone_boundary"][0]
    v = sr.render(dataclasses.replace(base, noise_sigma=1.5))
    d = tmp_path_factory.mktemp("sam_e2e")
    return encode(v.frames, d / "synthetic_bounce.mp4"), v


def sam_sse(v, *, bias=(1.4, 1.1), grow=1.2) -> bytes:
    """Biased SAM-like masks for the ball (object 3) in every frame, plus a short-lived distractor (object 8)."""
    size = (v.scene.camera.width, v.scene.camera.height)
    lines = []
    for t in sorted(v.truth, key=lambda t: (t.frame_index % 4, t.frame_index)):  # out-of-order stream
        recs = [fk.disc_record(3, t.x + bias[0], t.y + bias[1], t.radius_px + grow, size, conf=0.93)]
        if t.frame_index < 4:
            recs.append(fk.disc_record(8, 600.0, 40.0, 6.0, size, conf=0.55))
        lines.append(fk.line(t.frame_index, recs))
    return fk.sse_bytes(fk.events(lines, usage={"video_frames_processed": 50}))


def mocked_client_factory(v, video: Path, calls: list):
    def handler(request: httpx2.Request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/files":
            body = request.read()
            assert video.read_bytes() in body and b"user_data" in body  # the actual bytes, not a path string
            return httpx2.Response(200, json={"id": "file-test-1", "bytes": video.stat().st_size, "purpose": "user_data",
                                              "status": "uploaded", "filename": video.name})
        assert request.url.path == "/v1/responses"
        req = json.loads(request.read())
        assert req["input"][0]["content"][1] == {"type": "input_video", "file_id": "file-test-1"}
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sam_sse(v))

    def factory(evidence_dir):
        return sa.SamClient(KEY, transport=httpx2.MockTransport(handler), evidence_dir=evidence_dir, sleep=lambda s: None)
    return factory


@pytest.fixture(scope="module")
def tracked(scene, tmp_path_factory):
    video, v = scene
    d = tmp_path_factory.mktemp("track")
    plan_path = d / "plan.json"
    assert demo.main(["plan", "--video", str(video), "--out", str(plan_path), "--prompt", "red ball"]) == 0
    calls = []
    out = demo.track(plan_path, d / "sam", confirm_live_call=True, client_factory=mocked_client_factory(v, video, calls),
                     log=lambda *a: None)
    assert calls == [("POST", "/v1/files"), ("POST", "/v1/responses")]
    return video, v, plan_path, out["path"]


def test_plan_reports_frames_size_duration_and_documented_cost_without_any_request(scene, tmp_path, capsys):
    video, v = scene
    plan_path = tmp_path / "plan.json"
    assert demo.main(["plan", "--video", str(video), "--out", str(plan_path), "--prompt", "red ball"]) == 0
    plan = json.loads(plan_path.read_text())
    assert plan["live_call_made"] is False and plan["media"]["source_frames"] == len(v.frames)
    assert plan["video"]["size_bytes"] == video.stat().st_size and plan["video"]["duration_s"] > 0
    assert plan["cost"]["expected_billed_frames"] == max(50, len(v.frames))
    assert plan["request_preview"]["input"][0]["content"][0] == {"type": "input_text", "text": "red ball"}
    text = capsys.readouterr().out
    assert "NO request" in text and video.name in text and "USD" in text
    # concept + box prompts are never combined
    assert demo.main(["plan", "--video", str(video), "--out", str(tmp_path / "p2.json"), "--prompt", "ball",
                      "--box", "0:10,10,40,40"]) == demo.EXIT_USAGE


def test_track_without_confirmation_makes_no_request(tracked, tmp_path):
    video, v, plan_path, _ = tracked
    def factory(_):
        raise AssertionError("a client must not even be constructed without confirmation")
    assert demo.main(["track", "--plan", str(plan_path), "--out-dir", str(tmp_path / "x")], client_factory=factory) == 2
    assert not (tmp_path / "x").exists()


def test_tracking_result_keeps_source_frames_and_exact_pts(tracked):
    video, v, _, path = tracked
    data = json.loads(Path(path).read_text())
    assert data["outcome"]["status"] == "completed" and data["usage"] == {"video_frames_processed": 50}
    table = fr.read_frame_table(video)
    for k, d in data["frame_times"].items():
        assert Fraction(d["num"], d["den"]) == table.frames[int(k)].t and d["pts"] == table.frames[int(k)].pts
    for p in Path(path).parent.rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text(encoding="utf-8", errors="replace")


def test_reconstruct_builds_a_program_simulates_and_explains(tracked, tmp_path):
    video, v, _, sam_path = tracked
    out = tmp_path / "recon"
    rep = demo.reconstruct(video, sam_path, out, end_frame=21, cf_restitution=0.3, provenance="synthetic")
    assert rep["sam"]["object_id"] == "3"  # most frames; the distractor is ignored
    c = rep["measurement"]["status_counts"]
    assert c["measured"] >= 18 and c["measured"] + c["uncertain"] + c["missing"] == 21
    # the masks were biased by ~1.8 px; the outline measurement is not
    report = json.loads((out / "measurement" / "sam_measurement_report.json").read_text())
    truth = {t.frame_index: t for t in v.truth}
    errs = [math.hypot(f["x"] - truth[f["frame_index"]].x, f["y"] - truth[f["frame_index"]].y)
            for f in report["frames"] if f["status"] == sm.MEASURED]
    assert np.median(errs) < 0.4
    assert rep["measurement"]["median_abs_mask_centroid_offset_px"] > 1.0
    phys = rep["physics"]
    assert phys["model"] == "image_space" and "program_sha256" in phys, phys
    assert abs(phys["restitution"] - 0.65) < 0.1  # the renderer's restitution
    k = phys["kinematics"]["contact"]
    assert abs(k["t_s"] - v.contacts_s[0]) < 1 / 30
    assert rep["reconstruction_consistency"]["rms_px"] < 1.0
    assert rep["counterfactual"]["restitution"] == 0.3
    for name in ("overlay.mp4", "contact_sheet.png", "trajectory.png", "scene_program.json", "report.json", "report.md",
                 "counterfactual_program.json"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert "consistency check, not a prediction" in (out / "report.md").read_text()
    md = (out / "report.md").read_text()
    assert "scene_program.json" in md and set(rep["artifacts"]) >= {"scene_program.json", "overlay.mp4", "report.md"}
    with pytest.raises(demo.UsageError):
        demo.reconstruct(video, sam_path, out)  # never overwrites


def test_preregistered_sam_run_freezes_and_reveals(tracked, tmp_path):
    video, v, plan_path, sam_path = tracked
    prompt = json.loads(plan_path.read_text())["prompt"]
    prereg = rvd.make_preregistration_sam(video, prompt=prompt, end_frame=21, provenance="synthetic_stand_in",
                                          gates={"max_training_gaps": 2, "max_withheld_gaps": 1}, notes="test-only")
    pp = tmp_path / "prereg.json"
    rvd._write_with_record(pp, prereg)
    r = rvd.run(pp, tmp_path / "runs", sam_result=Path(sam_path))
    m = r["manifest"]
    assert m["status"] == "prediction_frozen" and m["measurement"]["sam"]["sam_object_id"] == "3"
    traj = json.loads(Path(m["measurement"]["tracked_path"]).read_text())
    assert traj["source"] == "SAM_3_1" and traj["synthetic"] is True
    assert all("sigma_x" in o and "diameter_px" in o for o in traj["objects"][0]["observations"])
    out = rvd.reveal(r["exp_dir"])
    f = out["findings"]
    assert "SAM-guided outline" in f["prediction_error_vs_withheld_tracker_measurements"]["reference"]
    assert set(f["prediction_error_vs_withheld_tracker_measurements"]["rmse_px"]) >= {"scene_program"}


def test_runner_refuses_mismatched_sam_inputs(tracked, tmp_path):
    video, v, plan_path, sam_path = tracked
    other = rvd.make_preregistration_sam(video, prompt={"kind": "concept", "text": "blue cube"}, end_frame=21,
                                         provenance="synthetic_stand_in")
    pp = tmp_path / "other.json"
    rvd._write_with_record(pp, other)
    with pytest.raises(rvd.VerifyError, match="prompt"):
        rvd.run(pp, tmp_path / "runs", sam_result=Path(sam_path))
    with pytest.raises(rvd.UsageError, match="--sam-result"):
        rvd.run(pp, tmp_path / "runs")
    v1 = rvd.make_preregistration(video, seed_box=(10, 10, 40, 40), provenance="synthetic_stand_in")
    p1 = tmp_path / "v1.json"
    rvd._write_with_record(p1, v1)
    with pytest.raises(rvd.UsageError, match="only valid"):
        rvd.run(p1, tmp_path / "runs", sam_result=Path(sam_path))


def test_report_lists_only_artifacts_actually_written_when_no_program(tracked, tmp_path):
    """Regression: report.md used to list scene_program.json even when no program was produced."""
    video, v, _, sam_path = tracked
    out = tmp_path / "recon_no_contact"
    rep = demo.reconstruct(video, sam_path, out, end_frame=9, provenance="synthetic", write_video=False,
                           disclosures=["test-only disclosure"])  # frames 0-8: free fall, no contact
    assert rep["physics"].get("program_sha256") is None and rep["physics"]["reasons"]
    assert not (out / "scene_program.json").exists() and not (out / "counterfactual_program.json").exists()
    md = (out / "report.md").read_text()
    assert "scene_program.json" not in md and "overlay.mp4" not in md and "No executable scene program" in md
    assert "test-only disclosure" in md
    for name in rep["artifacts"]:
        assert (out / name).is_file(), name
    assert "trajectory.png" in rep["artifacts"] and "contact_sheet.png" in rep["artifacts"]
