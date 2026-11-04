"""OFFLINE tests for tools/video_predictive_demo.py (P5 video-in pipeline).

Every run uses a test-only scene (never the pre-registered experiment scene) in a pytest
temporary directory; the registered P3/P4 artifacts and vault are never read. Sockets are
blocked for the whole module.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import shutil
import socket
import textwrap
from pathlib import Path

import pytest

from decompile import evaluate as ev
from decompile import holdout as ho
from decompile import predict as pd
from decompile import synth
from decompile import tracker as tk
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "video_predictive_demo.py"
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))
NOTES = ["test-only SYNTHETIC VIDEO scene; not the registered experiment"]


def _load_tool():
    spec = importlib.util.spec_from_file_location("video_predictive_demo_tool", TOOL)
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


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    out = tmp_path_factory.mktemp("p5")
    calls = {"evaluate": 0}
    real = ev.evaluate

    def counting(*a, **k):
        calls["evaluate"] += 1
        return real(*a, **k)
    mp = pytest.MonkeyPatch()
    mp.setattr(tool.ev, "evaluate", counting)
    try:
        r = tool.run(out, spec=ALT, notes=NOTES, experiment_id="p5_test")
    finally:
        mp.undo()
    return r, calls


def _p(r, *parts):
    return r["exp_dir"].joinpath(*parts)


def test_run_completes_with_the_expected_outputs(done):
    r, calls = done
    m = r["manifest"]
    assert m["status"] == "complete" and m["label"] == "SYNTHETIC VIDEO" and m["api_requests_made"] == 0
    assert calls["evaluate"] == 1 and m["reveal"]["evaluate_calls"] == 1
    for k, v in m["reveal"]["files"].items():
        assert hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest() == v["sha256"], k
    assert Path(m["reveal"]["files"]["overlay_png"]["path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(_p(r, "run_manifest.json").read_bytes()) == json.loads(pd.canonical_bytes(m))
    summary = _p(r, "reveal", "summary.md").read_text(encoding="utf-8")
    assert "SYNTHETIC VIDEO" in summary and "not real-video validation" in summary


def test_video_ingest_and_frame_extraction(done):
    r, _ = done
    meas = r["manifest"]["measurement"]
    assert meas["probe"]["accepted"] and meas["probe"]["decoded_frame_count"] == ALT.n_frames
    assert meas["extraction"]["frame_count"] == ALT.n_frames and meas["extraction"]["synthesized_timestamp_count"] == 0
    table = json.loads(_p(r, "measurement", "frames", "frames.json").read_text())
    assert table["source_sha256"] == r["manifest"]["video"]["sha256"]
    assert meas["gate"]["passed"] and meas["gate"]["status_counts"]["detected"] == ALT.n_frames


def test_tracker_output_is_the_only_source_of_inference_observations(done):
    r, _ = done
    tracked = tr.load_trajectories(_p(r, "measurement", "tracked_trajectories.json"))
    assert tracked.source == "CLASSICAL_TRACKER" and tracked.synthetic is True
    training, schedule = pd.load_inference_inputs(_p(r, "inference", "inference_inputs.json"))
    assert training.source == "CLASSICAL_TRACKER" and training.synthetic is True
    expected = ho.split_trajectories(tracked, tool.TRAIN_FRACTION)
    assert training == expected.inference.training and schedule == expected.inference.schedule
    first = r["manifest"]["split"]["split"]["first_withheld_frame"]
    tobs = tracked.objects[0].observations
    assert training.objects[0].observations == tuple(o for o in tobs if o.frame_index < first)  # bit-identical
    gen = tr.load_trajectories(_p(r, "generator_side", "source_trajectories.json"))
    assert training.objects[0].observations != tuple(o for o in gen.objects[0].observations if o.frame_index < first)


def test_inference_inputs_hold_no_withheld_position_or_ground_truth(done):
    r, _ = done
    data = json.loads(_p(r, "inference", "inference_inputs.json").read_text())
    first = data["first_withheld_frame"]
    assert all(o["frame_index"] < first for o in data["training"]["objects"][0]["observations"])
    assert all(set(s) == {"object_id", "frame_index", "t"} for s in data["schedule"])
    text = json.dumps(data)
    for forbidden in ("contacts", "restitution", "ground_truth", "true_x_px", "holdout_vault"):
        assert forbidden not in text
    tracked = tr.load_trajectories(_p(r, "measurement", "tracked_trajectories.json"))
    withheld_x = {repr(o.x) for o in tracked.objects[0].observations if o.frame_index >= first}
    assert not any(x in text for x in withheld_x)


def test_frozen_prediction_hash_and_report_contents(done):
    r, _ = done
    m = r["manifest"]
    pred = Path(m["predict"]["prediction_path"])
    digest = hashlib.sha256(pred.read_bytes()).hexdigest()
    assert digest == m["predict"]["prediction_sha256"] == m["reveal"]["prediction_sha256_before"] \
        == m["reveal"]["prediction_sha256_after"]
    assert pd.sha256_path_for(pred).read_text(encoding="ascii").split()[0] == digest
    report = json.loads(Path(m["reveal"]["files"]["evaluation_report"]["path"]).read_bytes())
    assert report["prediction"]["sha256"] == digest and report["label"] == "SYNTHETIC VIDEO"
    assert report["limitations"] == list(tool.P5_LIMITATIONS)
    assert report["limitations"] != list(ev.LIMITATIONS)
    assert NOTES[0] in report["disclosures"]
    assert report["inputs"]["training_sha256"] == m["split"]["training_sha256"]
    o = report["objects"][0]
    assert o["contacts"]["scene_program"]["status"] == "evaluated"
    assert set(o["withheld_accuracy"]) == {"scene_program", *pd.BASELINES}
    acc = json.loads(Path(m["reveal"]["files"]["tracking_accuracy"]["path"]).read_bytes())
    assert acc["all"]["x"]["n"] == ALT.n_frames and acc["all"]["x"]["rmse_px"] < 0.1 and acc["all"]["y"]["rmse_px"] < 0.1


def test_default_notes_preserve_the_p2_disclosure():
    assert tool.P2_DISCLOSURE.startswith("P2 development disclosure") and "not fully blind" in tool.P2_DISCLOSURE
    src = inspect.getsource(tool.run)
    assert "P2_DISCLOSURE" in src and "P4_DISCLOSURE" in src


def test_reveal_never_regenerates_predictions_and_evaluates_once(done, tmp_path, monkeypatch):
    r, _ = done
    m = r["manifest"]

    def refuse(*a, **k):
        raise AssertionError("prediction during the reveal")
    monkeypatch.setattr(tool.pd, "predict", refuse)
    monkeypatch.setattr(tool.pd, "build_prediction", refuse)
    calls = []
    real = ev.evaluate
    monkeypatch.setattr(tool.ev, "evaluate", lambda *a, **k: calls.append(1) or real(*a, **k))
    pred = Path(m["predict"]["prediction_path"])
    out = tool.stage_reveal(tmp_path, prediction=pred, prediction_sha_path=pd.sha256_path_for(pred),
                            expected_prediction_sha256=m["predict"]["prediction_sha256"],
                            inputs_path=_p(r, "inference", "inference_inputs.json"),
                            vault_path=_p(r, "evaluation_only", "evaluation_vault.json"),
                            tracked_path=_p(r, "measurement", "tracked_trajectories.json"),
                            generator_trajectories_path=_p(r, "generator_side", "source_trajectories.json"),
                            first_withheld=m["split"]["split"]["first_withheld_frame"])
    assert len(calls) == 1 and out["prediction_sha256_before"] == out["prediction_sha256_after"]
    original = json.loads(Path(m["reveal"]["files"]["evaluation_report"]["path"]).read_bytes())
    assert json.loads(out["files"]["evaluation_report"].read_bytes()) == original  # deterministic


def test_tampered_prediction_is_refused(done, tmp_path):
    r, _ = done
    m = r["manifest"]
    pred = tmp_path / "prediction.json"
    shutil.copy(m["predict"]["prediction_path"], pred)
    shutil.copy(m["predict"]["sha256_path"], pd.sha256_path_for(pred))
    pred.write_bytes(pred.read_bytes().replace(b'"obj_01"', b'"obj_02"', 1))
    with pytest.raises(tool.VerifyError, match="frozen"):
        tool.stage_reveal(tmp_path, prediction=pred, prediction_sha_path=pd.sha256_path_for(pred),
                          expected_prediction_sha256=m["predict"]["prediction_sha256"],
                          inputs_path=_p(r, "inference", "inference_inputs.json"),
                          vault_path=_p(r, "evaluation_only", "evaluation_vault.json"),
                          tracked_path=_p(r, "measurement", "tracked_trajectories.json"),
                          generator_trajectories_path=_p(r, "generator_side", "source_trajectories.json"),
                          first_withheld=0)
    assert not (tmp_path / "reveal").exists()


def test_measurement_needs_only_the_video(done, tmp_path):
    r, _ = done
    video = tmp_path / "only" / "v.mp4"
    video.parent.mkdir()
    shutil.copy(r["manifest"]["video"]["path"], video)
    again = tool.stage_measure(video, tmp_path / "exp", tk.TrackerConfig(), synthetic_video=True)
    assert again["tracked_sha256"] == r["manifest"]["measurement"]["tracked_sha256"]


def test_prediction_needs_only_the_inference_inputs(done, tmp_path):
    r, _ = done
    inputs = tmp_path / "inputs.json"
    shutil.copy(_p(r, "inference", "inference_inputs.json"), inputs)
    again = tool.stage_predict(inputs, tmp_path / "exp", NOTES)
    assert again["prediction_sha256"] == r["manifest"]["predict"]["prediction_sha256"]  # deterministic, same inputs


def _function_names(fn) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
           {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


@pytest.mark.parametrize("fn", ["stage_measure", "tracking_gate", "stage_predict", "fit_gate"])
def test_inference_stages_cannot_reach_ground_truth_or_the_vault(fn):
    names = _function_names(getattr(tool, fn))
    for forbidden in ("synth", "rd", "render_video", "ground_truth", "vault", "vault_path", "write_evaluation_vault",
                      "evaluate", "split_trajectories", "generator_side", "evaluation_only"):
        assert forbidden not in names, (fn, forbidden)


def test_tracking_gate_stops_before_freezing(tmp_path):
    blind = tk.TrackerConfig(hue_center_deg=200.0)  # looks for a blue object: every frame is "missing"
    with pytest.raises(tool.GateError) as info:
        tool.run(tmp_path, spec=ALT, config=blind, notes=NOTES, experiment_id="gate")
    assert info.value.gate == "tracking"
    exp = tmp_path / "gate"
    assert not (exp / "inference").exists() and not list(exp.rglob("prediction*.json"))
    m = json.loads((exp / "run_manifest.json").read_bytes())
    assert m["status"] == "stopped_before_freeze"
    assert json.loads((exp / "measurement" / "tracking_gate.json").read_bytes())["passed"] is False


def test_training_fit_gate_stops_before_freezing(tmp_path):
    no_contact = synth.SceneSpec("obj_01", 30, 40, 100.0, 60.0, 60.0, 0.0, 0.0, 400.0)  # no plane: no contact
    with pytest.raises(tool.GateError) as info:
        tool.run(tmp_path, spec=no_contact, notes=NOTES, experiment_id="fit")
    assert info.value.gate == "training_fit"
    exp = tmp_path / "fit"
    assert not list(exp.rglob("prediction*.json")) and not (exp / "reveal").exists()
    assert json.loads((exp / "inference" / "fit_gate.json").read_bytes())["passed"] is False
    assert json.loads((exp / "run_manifest.json").read_bytes())["status"] == "stopped_before_freeze"


def test_cli_usage_error():
    assert tool.main(["--seed", "-1"]) == tool.EXIT_USAGE
