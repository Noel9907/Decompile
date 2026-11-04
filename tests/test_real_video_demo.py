"""OFFLINE tests for tools/real_video_demo.py (P6 pre-registered real-video runner).

No real footage is used: every video is a SYNTHETIC stand-in made here (provenance
"synthetic_stand_in"), with a BLUE ball, a static blue distractor and optional occlusions, so the
seed-derived colour model, ROI/exclusion, stride, gap policy and gates are exercised. Generator
positions are used only test-side. Sockets are blocked; Muse Spark only ever sees a fake transport.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import random
import socket
import subprocess
import textwrap
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import httpx2
import numpy as np
import pytest

from decompile import frames as fr
from decompile import meta_api as m
from decompile import render as rd
from decompile import scene_interpretation as si
from decompile import synth
from decompile import tracker as tk
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "real_video_demo.py"
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))
BLUE = (40, 90, 220)
DISTRACTOR = (560.0, 90.0, 14.0)  # static blue disc (x, y, r)
DISTRACTOR_BOX = (536, 66, 585, 115)
SEED_BOX = (86, 46, 115, 75)  # around the ball at frame 0: (100, 60), r = 12
FAKE_KEY = "sk-OFFLINE-FAKE-KEY-real-video-0123456789"


def _load_tool():
    spec = importlib.util.spec_from_file_location("real_video_demo_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


@pytest.fixture(scope="module", autouse=True)
def no_network():
    mp = pytest.MonkeyPatch()

    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    mp.setattr(socket.socket, "connect", refuse)
    mp.setattr(socket, "create_connection", refuse)
    yield
    mp.undo()


# ------------------------------------------------------------------ stand-in videos (test-only)


SETTINGS = replace(rd.RenderSettings(), ball_rgb=BLUE)


def _frame(k: int, x: float, y: float, *, occlude: bool) -> np.ndarray:
    b = [rd.Boundary("y", 440.0, "less")]
    img = rd.render_frame(640, 480, x, y, k, b, SETTINGS).astype(int)
    s0 = replace(SETTINGS, noise_sigma=0.0, ball_radius_px=DISTRACTOR[2])
    img += rd.render_frame(640, 480, DISTRACTOR[0], DISTRACTOR[1], k, b, s0).astype(int) - \
        rd.render_frame(640, 480, -100.0, -100.0, k, b, s0).astype(int)
    if occlude:  # a grey card covers the ball
        x0, y0 = int(x) - 20, int(y) - 20
        img[max(0, y0):y0 + 41, max(0, x0):x0 + 41] = SETTINGS.background_rgb
    return np.clip(img, 0, 255).astype(np.uint8)


def make_video(path: Path, occluded=()) -> Path:
    obs = synth.generate(ALT).trajectories.objects[0].observations
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-video_size", "640x480", "-framerate", "30/1", "-i", "pipe:0", "-an", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-threads", "1",
           "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
    data = b"".join(_frame(o.frame_index, o.x, o.y, occlude=o.frame_index in occluded).tobytes() for o in obs)
    subprocess.run(cmd, input=data, check=True)
    return path


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("standin")
    return {"clean": make_video(d / "clean.mp4"), "gap30": make_video(d / "gap30.mp4", occluded={30}),
            "gap50": make_video(d / "gap50.mp4", occluded={50})}


def prereg(tmp_path: Path, video: Path, name="prereg.json", **kw) -> Path:
    args = dict(seed_box=SEED_BOX, exclude_boxes=[DISTRACTOR_BOX], provenance="synthetic_stand_in",
                notes="test-only stand-in")
    args.update(kw)
    data = tool.make_preregistration(video, **args)
    path = tmp_path / name
    tool._write_with_record(path, data)
    return path


def p_dir(r, *parts):
    return r["exp_dir"].joinpath(*parts)


# ------------------------------------------------------------------ Muse Spark fake transport


class FakeServer:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, body = self.script.pop(0)
        return httpx2.Response(status, json=body)


def completion(content: str) -> dict:
    return {"id": "chatcmpl-fake-p6", "object": "chat.completion", "created": 0, "model": "muse-spark-1.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}


INTERP = {  # HANDWRITTEN test data (not model output)
    "schema_version": si.SCHEMA_VERSION,
    "observed_facts": [{"fact_id": "F1", "object_ids": ["obj_01"], "statement": "obj_01 is observed.",
                        "evidence": [{"object_id": "obj_01", "frame_index": 0}, {"object_id": "obj_01", "frame_index": 4}]}],
    "objects": [{"object_id": "obj_01", "motion_class": "insufficient_data", "motion_description": "Test data.",
                 "supporting_fact_ids": ["F1"]}],
    "physical_model_hypotheses": [], "event_hypotheses": [], "interaction_hypotheses": [],
    "ambiguities": [{"ambiguity_id": "A1", "about_ids": ["F1"], "description": "Test data.",
                     "alternatives": ["Any smooth motion."], "what_would_resolve": "More frames."}],
    "data_quality_notes": [],
}


def fake_factory(tmp_path: Path, script):
    server = FakeServer(script)

    def factory():
        return m.MuseSparkClient(evidence_dir=tmp_path / "fake_evidence", api_key=FAKE_KEY,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
                                 sleep=lambda s: None, rng=random.Random(0))
    return factory, server


# ------------------------------------------------------------------ the base run (Spark off)


@pytest.fixture(scope="module")
def base(tmp_path_factory, videos):
    d = tmp_path_factory.mktemp("base")
    return tool.run(prereg(d, videos["clean"]), d / "out", experiment_id="base")


def test_seeded_non_orange_tracking_freezes_without_evaluating(base):
    mm = base["manifest"]
    assert mm["status"] == "prediction_frozen" and mm["evaluated"] is False
    assert not p_dir(base, "reveal").exists() and p_dir(base, "inference", "prediction.json").is_file()
    cfg = mm["measurement"]["config"]
    assert 200 < cfg["hue_center_deg"] < 240 and cfg["hue_weighting"] and cfg["exclude_boxes"] == [list(DISTRACTOR_BOX)]
    tracked = tr.load_trajectories(p_dir(base, "measurement", "tracked_trajectories.json"))
    assert tracked.source == "CLASSICAL_TRACKER" and tracked.synthetic is True  # stand-in provenance
    gen = synth.generate(ALT).trajectories.objects[0].observations  # test-side reference only
    ex = [o.x - gen[o.frame_index].x for o in tracked.objects[0].observations]
    ey = [o.y - gen[o.frame_index].y for o in tracked.objects[0].observations]
    rms = lambda v: math.sqrt(sum(a * a for a in v) / len(v))  # noqa: E731
    print("seeded blue-ball tracking (px): x", rms(ex), max(map(abs, ex)), "y", rms(ey), max(map(abs, ey)))
    assert rms(ex) < 0.15 and rms(ey) < 0.15 and max(map(abs, ex + ey)) < 0.4


def test_extended_diagnostics_and_seed_record(base):
    rep = json.loads(p_dir(base, "measurement", "tracking_report.json").read_bytes())
    assert all("n_mask_px" in f and "largest_small_component_px" in f for f in rep["frames"])
    seed = json.loads(p_dir(base, "measurement", "seed_derivation.json").read_bytes())
    assert seed["seed_frame"] == 0 and seed["seed_box"] == list(SEED_BOX)


def test_without_exclusion_the_distractor_stops_the_run(tmp_path, videos):
    p = prereg(tmp_path, videos["clean"], exclude_boxes=[])
    with pytest.raises(tool.GateError) as info:
        tool.run(p, tmp_path / "out", experiment_id="noexcl")
    # the seed check on the first clip frame already sees the static distractor
    assert info.value.gate == "seed" and "ambiguous" in str(info.value)
    exp = tmp_path / "out" / "noexcl"
    assert not (exp / "inference").exists()
    assert json.loads((exp / "run_manifest.json").read_bytes())["status"] == "stopped_before_freeze"


def test_roi_also_excludes_the_distractor(tmp_path, videos):
    p = prereg(tmp_path, videos["clean"], exclude_boxes=[], roi=(0, 0, 500, 480))
    r = tool.run(p, tmp_path / "out", experiment_id="roi")
    assert r["manifest"]["status"] == "prediction_frozen" and r["manifest"]["measurement"]["config"]["roi"] == [0, 0, 500, 480]


def test_stride_keeps_original_indices_and_exact_timestamps(tmp_path, videos):
    r = tool.run(prereg(tmp_path, videos["clean"], stride=2), tmp_path / "out", experiment_id="stride")
    tracked = tr.load_trajectories(p_dir(r, "measurement", "tracked_trajectories.json"))
    obs = tracked.objects[0].observations
    assert [o.frame_index for o in obs] == list(range(0, 60, 2))
    assert all(o.t == Fraction(o.frame_index, 30) for o in obs)
    data = json.loads(p_dir(r, "inference", "inference_inputs.json").read_bytes())
    assert data["first_withheld_frame"] == 42 and all(s["frame_index"] % 2 == 0 for s in data["schedule"])


def test_training_gap_policy(tmp_path, videos):
    with pytest.raises(tool.GateError) as info:
        tool.run(prereg(tmp_path, videos["gap30"], name="a.json"), tmp_path / "out", experiment_id="gap0")
    assert "training frame(s) failed" in str(info.value)
    r = tool.run(prereg(tmp_path, videos["gap30"], name="b.json", gates={"max_training_gaps": 1}), tmp_path / "out",
                 experiment_id="gap1")
    gate = r["manifest"]["measurement"]["gate"]
    assert gate["training_gaps"] == [{"frame_index": 30, "status": "missing", "reason": "no target-coloured pixels"}]
    data = json.loads(p_dir(r, "inference", "inference_inputs.json").read_bytes())
    frames = [o["frame_index"] for o in data["training"]["objects"][0]["observations"]]
    assert 30 not in frames and frames == [i for i in range(42) if i != 30]  # absent, never interpolated
    assert data["first_withheld_frame"] == 42  # the pre-registered cutoff, unchanged by the gap


def test_every_withheld_frame_must_be_detected(tmp_path, videos):
    p = prereg(tmp_path, videos["gap50"], gates={"max_training_gaps": 5})
    with pytest.raises(tool.GateError) as info:
        tool.run(p, tmp_path / "out", experiment_id="w")
    assert "withheld frame 50" in str(info.value)
    assert not list((tmp_path / "out" / "w").rglob("prediction*.json"))


def test_seeded_detection_is_processing_order_independent(base):
    frames_dir = p_dir(base, "measurement", "frames")
    seed = json.loads(p_dir(base, "measurement", "seed_derivation.json").read_bytes())["config"]
    cfg = tk.TrackerConfig(**{**seed, "roi": None, "exclude_boxes": tuple(tuple(b) for b in seed["exclude_boxes"])})
    subset = list(range(0, 60, 3))
    shuffled = subset[:]
    random.Random(7).shuffle(shuffled)
    a = tk.track_extracted(frames_dir, cfg, frames=subset)
    b = tk.track_extracted(frames_dir, cfg, frames=shuffled)
    c = tk.track_extracted(frames_dir, cfg, frames=subset[:10])
    assert a.detections == b.detections and a.detections[:10] == c.detections


def test_predictions_are_byte_identical_with_spark_off_success_and_failure(tmp_path, videos, base):
    off = p_dir(base, "inference", "prediction.json").read_bytes()
    ok_factory, ok_server = fake_factory(tmp_path, [(200, completion(json.dumps(INTERP)))])
    # different pre-registered Spark modes and run ids: the prediction must not depend on either
    ok = tool.run(prereg(tmp_path, videos["clean"], name="ok.json", spark_mode="live"), tmp_path / "out", experiment_id="ok",
                  spark_client_factory=ok_factory)
    bad_factory, bad_server = fake_factory(tmp_path, [(503, {"error": {"message": "busy", "type": "server_error",
                                                                       "param": None, "code": "x"}})] * 3)
    bad = tool.run(prereg(tmp_path, videos["clean"], name="bad.json", spark_mode="live"), tmp_path / "out",
                   experiment_id="bad",
                   spark_client_factory=bad_factory)
    assert ok["manifest"]["spark"]["status"] == "succeeded" and len(ok_server.requests) == 1
    assert bad["manifest"]["spark"]["status"] == "unavailable" and len(bad_server.requests) == 1  # single attempt
    assert bad["manifest"]["status"] == "prediction_frozen"  # API failure does not block the pipeline
    assert base["manifest"]["spark"]["status"] == "disabled"
    assert p_dir(ok, "inference", "prediction.json").read_bytes() == off
    assert p_dir(bad, "inference", "prediction.json").read_bytes() == off
    text = off.decode("utf-8").lower()
    assert "spark" not in text and "muse" not in text and "interpretation" not in text
    for r in (base, ok, bad):
        art = Path(r["manifest"]["spark"]["artifact"]["path"])
        assert hashlib.sha256(art.read_bytes()).hexdigest() == r["manifest"]["spark"]["artifact"]["sha256"]
    # the Spark request held only training observations
    inputs = json.loads(p_dir(ok, "inference", "inference_inputs.json").read_bytes())
    vault = json.loads(p_dir(ok, "evaluation_only", "evaluation_vault.json").read_bytes())
    sent = ok_server.requests[0].content.decode("utf-8")
    for o in vault["holdout_vault"]["observations"]:
        assert repr(o["x"]) not in sent and repr(o["y"]) not in sent
    assert len(inputs["training"]["objects"][0]["observations"]) == 42


def test_live_mode_requires_explicit_confirmation(tmp_path, videos):
    p = prereg(tmp_path, videos["clean"], spark_mode="live")
    with pytest.raises(tool.UsageError, match="confirm-live-call"):
        tool.run(p, tmp_path / "out", experiment_id="live")
    assert not (tmp_path / "out" / "live").exists()


def test_preregistration_and_video_are_verified(tmp_path, videos):
    p = prereg(tmp_path, videos["clean"])
    data = json.loads(p.read_bytes())
    data["clip"]["stride"] = 3
    p.write_bytes(json.dumps(data).encode())
    with pytest.raises(tool.VerifyError, match="changed"):
        tool.run(p, tmp_path / "out", experiment_id="x")
    copy = tmp_path / "v.mp4"
    copy.write_bytes(videos["clean"].read_bytes())
    p2 = prereg(tmp_path, copy, name="p2.json")
    copy.write_bytes(copy.read_bytes() + b"\0")
    with pytest.raises(tool.VerifyError, match="SHA-256"):
        tool.run(p2, tmp_path / "out", experiment_id="y")


def _names(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
           {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


@pytest.mark.parametrize("fn", ["stage_measure", "real_tracking_gate", "clip_frames", "load_training_only",
                                "stage_spark"])
def test_inference_stages_cannot_reach_withheld_positions(fn):
    names = _names(getattr(tool, fn))
    for forbidden in ("vault", "vault_path", "write_evaluation_vault", "evaluate", "ev", "synth", "ground_truth",
                      "annotations", "schedule", "split_trajectories"):
        assert forbidden not in names, (fn, forbidden)


def test_reveal_evaluates_once_and_makes_no_ground_truth_claims(tmp_path, base):
    out = tool.reveal(base["exp_dir"])
    f = out["findings"]
    p = f["prediction_error_vs_withheld_tracker_measurements"]
    assert "NOT independently verified ground truth" in p["reference"]
    assert f["tracking_accuracy"]["status"] == "not_available"
    assert f["contact_timing"]["status"] == "not_evaluated"
    assert f["contact_timing"]["tracker_measured_withheld_reversals"] >= 1
    rep = json.loads(out["files"]["evaluation_report"].read_bytes())
    assert rep["limitations"] == list(tool.LIMITATIONS["synthetic_stand_in"])
    assert rep["objects"][0]["contacts"]["scene_program"]["status"] == "not_evaluated"
    man = json.loads((base["exp_dir"] / "reveal" / "reveal_manifest.json").read_bytes())
    assert man["evaluate_calls"] == 1 and man["prediction_sha256_before"] == man["prediction_sha256_after"]
    with pytest.raises(tool.VerifyError, match="already revealed"):
        tool.reveal(base["exp_dir"])


def test_annotation_accuracy_needs_an_independent_reference(tmp_path, base):
    tracked = tr.load_trajectories(p_dir(base, "measurement", "tracked_trajectories.json"))
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"kind": "manual_annotations", "annotator": "test", "method": "test values",
                               "frames": [{"frame_index": 0, "x": 100.0, "y": 60.0}]}), encoding="utf-8")
    acc = tool.annotation_accuracy(ann, tracked)
    assert acc["status"] == "evaluated" and acc["x"]["n"] == 1


def test_cli_prereg_never_overwrites(tmp_path, videos):
    out = tmp_path / "p.json"
    args = ["prereg", "--video", str(videos["clean"]), "--out", str(out), "--seed-box", "86,46,115,75",
            "--exclude", "536,66,585,115", "--provenance", "synthetic_stand_in"]
    assert tool.main(args) == tool.EXIT_OK and pd_record(out).is_file()
    assert tool.main(args) == tool.EXIT_USAGE


def pd_record(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def test_white_ball_seed_is_refused():
    img = np.full((120, 160, 3), 120, np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    img[(xx - 80) ** 2 + (yy - 60) ** 2 <= 100] = 245  # a white (colourless) ball
    with pytest.raises(tk.SeedError, match="too little colour"):
        tk.config_from_seed(img, (68, 48, 92, 72))
