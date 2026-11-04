"""P7: the real-video runner with pre-registration version 2 (outline measurement + projective physics),
exercised OFFLINE on a test-only synthetic stand-in MP4 rendered by decompile.stress_render."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import socket
import subprocess
from pathlib import Path

import pytest

from decompile import stress_render as sr

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("real_video_demo_p7", ROOT / "tools" / "real_video_demo.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def encode(frames, path: Path) -> Path:
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-video_size", f"{w}x{h}", "-framerate", "30/1", "-i", "pipe:0", "-an", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "1",
           "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture(scope="module")
def standin(tmp_path_factory):
    base = [s for s in sr.stress_suite(32) if s.name == "perspective_pitched"][0]
    scene = dataclasses.replace(base, ball=sr.Ball(specular=0.4), background=sr.Background(boundary_y=215.0),
                                exposure_s=1 / 60, noise_sigma=2.0)
    v = sr.render(scene)
    path = encode(v.frames, tmp_path_factory.mktemp("p7") / "standin.mp4")
    t = v.truth[2]
    pad = t.radius_px + 4
    box = [int(t.x - pad), int(t.y - pad), int(t.x + pad) + 1, int(t.y + pad) + 1]
    return path, box, v


def p7_prereg(tmp_path, standin, **kw):
    path, box, _ = standin
    args = dict(seed_box=box, start_frame=2, end_frame=30, provenance="synthetic_stand_in",
                centre_method="edge_ellipse", physics_model="projective",
                camera_prior={"hfov_deg": 66.0, "hfov_range_deg": [50.0, 80.0]},
                gates={"max_training_gaps": 3, "max_area_rel_dev": 1.0}, max_withheld_gaps=2, notes="test-only")
    args.update(kw)
    data = tool.make_preregistration(path, **args)
    p = tmp_path / "prereg.json"
    tool._write_with_record(p, data)
    return p, data


def test_version_1_preregistrations_are_unchanged(standin):
    path, box, _ = standin
    d = tool.make_preregistration(path, seed_box=box, provenance="synthetic_stand_in")
    assert d["prereg_version"] == 1 and "measurement" not in d and "physics" not in d
    assert set(d["gates"]) == set(tool.DEFAULT_GATES)


def test_version_2_records_the_method_and_refuses_inconsistent_options(tmp_path, standin):
    p, d = p7_prereg(tmp_path, standin)
    loaded, _ = tool.load_preregistration(p)
    assert loaded["prereg_version"] == 2 and loaded["measurement"]["centre_method"] == "edge_ellipse"
    assert loaded["physics"]["model"] == "projective" and loaded["physics"]["camera_prior"]["hfov_deg"] == 66.0
    assert loaded["gates"]["max_withheld_gaps"] == 2
    path, box, _ = standin
    with pytest.raises(tool.UsageError):  # projective needs per-frame sigma and diameter
        tool.make_preregistration(path, seed_box=box, physics_model="projective")
    with pytest.raises(tool.UsageError):
        tool.make_preregistration(path, seed_box=box, centre_method="edge_ellipse", physics_model="projective",
                                  camera_prior={"hfov_deg": 90.0, "hfov_range_deg": [50.0, 80.0]})


def test_projective_run_freezes_then_reveals_once(tmp_path, standin):
    p, _ = p7_prereg(tmp_path, standin)
    r = tool.run(p, tmp_path / "exp")
    m = r["manifest"]
    assert m["status"] == "prediction_frozen" and m["evaluated"] is False
    assert m["predict"]["fit_gate"]["kind"] == "projective_fit_gate" and m["predict"]["fit_gate"]["passed"]
    pred = json.loads(Path(m["predict"]["prediction_path"]).read_bytes())
    assert pred["prediction_format_version"] == 2 and pred["objects"][0]["program_kind"] == "projective_scene_program"
    training = json.loads(Path(m["split"]["inputs_path"]).read_bytes())["training"]
    assert all("sigma_x" in o and "diameter_px" in o for o in training["objects"][0]["observations"])
    assert not (r["exp_dir"] / "reveal").exists()
    out = tool.reveal(r["exp_dir"])
    f = out["findings"]
    rmse = f["prediction_error_vs_withheld_tracker_measurements"]["rmse_px"]
    assert rmse["scene_program"] < min(v for k, v in rmse.items() if k != "scene_program")
    assert any("Perspective is modelled" in x for x in f["limitations"]) or any("field-of-view" in x for x in f["limitations"])
    with pytest.raises(tool.VerifyError):
        tool.reveal(r["exp_dir"])


def test_withheld_gap_limit_is_enforced(tmp_path, standin):
    from decompile import tracker as tk
    result_gate = tool.real_tracking_gate
    fake = type("R", (), {})()
    det = lambda s: type("FD", (), {"detection": tk.Detection(s, None, None, 1, 400, 0, (0, 0, 20, 20), 0.78, None,
                                                               edge={"rejection_reason": "asymmetric outline"}),
                                     "frame_index": 0})()  # noqa: E731
    ok = lambda i: type("FD", (), {"frame_index": i, "detection": tk.Detection(tk.DETECTED, 1.0, 1.0, 1, 400, 0,
                                                                               (0, 0, 20, 20), 0.78, 1.0)})()  # noqa: E731
    dets = [ok(i) for i in range(10)]
    bad = det(tk.EDGE_REJECTED)
    bad.frame_index = 8
    dets[8] = bad
    fake.detections = dets
    fake.status_counts = lambda: {}
    gates = {**tool.DEFAULT_GATES, "max_withheld_gaps": 1}
    g = result_gate(fake, list(range(7)), [7, 8, 9], gates)
    assert g["passed"] and g["withheld_gaps"][0]["frame_index"] == 8
    g0 = result_gate(fake, list(range(7)), [7, 8, 9], {**tool.DEFAULT_GATES, "max_withheld_gaps": 0})
    assert not g0["passed"] and "every withheld frame must be detected" in g0["problems"][0]
