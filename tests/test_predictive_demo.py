"""OFFLINE tests for tools/predictive_demo.py (P4 reveal runner).

Never touches the registered experiment: every chain is a test-only synthetic scene (ALT)
in a temporary directory, passed with its own expected hashes. Sockets are blocked, and
predict() / build_prediction() raise if the runner calls them.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import socket
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import evaluate as ev
from decompile import holdout as ho
from decompile import predict as pd
from decompile import synth

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "predictive_demo.py"
REGISTERED = ROOT / "runs" / "predictions" / "20261104T070243Z_p3_preregistered_prediction.json"
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))


def _load_tool():
    spec = importlib.util.spec_from_file_location("predictive_demo_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


@pytest.fixture(autouse=True)
def guards(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("forbidden call during the reveal")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield refuse


def make_chain(d: Path, spec=ALT) -> dict:
    scene = synth.generate(spec)
    split = ho.split_trajectories(scene.trajectories, Fraction(7, 10))
    inputs = d / "inference_inputs.json"
    inputs.write_bytes(pd.canonical_bytes(ho.inference_inputs_to_json(split.inference)))
    r = pd.predict(split.inference.training, split.inference.schedule, output_path=d / "pred.json",
                   experiment_notes=["test-only chain"])
    contacts = [{"object_id": "obj_01", "t_s": c.t_s, "axis": c.axis, "plane_index": c.plane_index}
                for c in scene.ground_truth.contacts]
    ev.write_evaluation_vault(d / "vault.json", split, ground_truth_contacts=contacts)
    return {"prediction": d / "pred.json", "sha": d / "pred.json.sha256", "inputs": inputs, "vault": d / "vault.json",
            "pred_sha": r.sha256, "training": split.inference.training_sha256, "schedule": split.inference.schedule_sha256}


@pytest.fixture
def chain(tmp_path, monkeypatch, guards):
    c = make_chain(tmp_path)
    monkeypatch.setattr(pd, "predict", guards)  # the runner must never predict
    monkeypatch.setattr(pd, "build_prediction", guards)
    return c


def argv(c, out, **over):
    a = {"--prediction": c["prediction"], "--inference-inputs": c["inputs"], "--vault": c["vault"], "--output-dir": out,
         "--expected-prediction-sha256": c["pred_sha"], "--expected-training-sha256": c["training"],
         "--expected-schedule-sha256": c["schedule"]}
    a.update(over)
    return [str(x) for kv in a.items() for x in kv]


def runs(out: Path) -> list[Path]:
    return sorted(out.glob("*")) if out.exists() else []


def snapshot(c) -> dict:
    return {k: c[k].read_bytes() for k in ("prediction", "sha", "inputs", "vault")}


# ------------------------------------------------------------------ success


def test_reveal_end_to_end(chain, tmp_path, capsys):
    before = snapshot(chain)
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == tool.EXIT_OK
    (run_dir,) = runs(out)
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == ["chart_data.json", "evaluation_report.json", "overlay.png", "run_manifest.json", "summary.md"]
    report = json.loads((run_dir / "evaluation_report.json").read_bytes())
    expected = ev.evaluate(chain["prediction"], chain["sha"], chain["vault"]).data
    assert {k: v for k, v in report.items() if k != "interpretation"} == json.loads(pd.canonical_bytes(expected))
    manifest = json.loads((run_dir / "run_manifest.json").read_bytes())
    assert manifest["prediction"]["sha256_before"] == manifest["prediction"]["sha256_after"] == chain["pred_sha"]
    assert (manifest["api_requests_made"], manifest["predictions_made"], manifest["evaluate_calls"]) == (0, 0, 1)
    for k, v in manifest["outputs"].items():
        assert hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest() == v["sha256"]
    assert (run_dir / "overlay.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert snapshot(chain) == before  # prediction, record, inputs and vault byte-identical
    assert "summary" in capsys.readouterr().out


def test_interpretation_follows_the_fixed_rules(chain, tmp_path):
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == 0
    report = json.loads((runs(out)[0] / "evaluation_report.json").read_bytes())
    (o,) = report["interpretation"]["objects"]
    acc = report["objects"][0]["withheld_accuracy"]
    assert o["primary_metric"] == "euclidean_rmse_px"
    assert o["beats"] == {m: acc[pd.SCENE_MODEL]["rmse_px"] < acc[m]["rmse_px"] for m in pd.BASELINES}
    assert o["best_baseline"] == min(pd.BASELINES, key=lambda m: acc[m]["rmse_px"])
    assert o["error_growth"]["first_error_px"] == acc[pd.SCENE_MODEL]["first_error_px"]
    summary = (runs(out)[0] / "summary.md").read_text(encoding="utf-8")
    assert "Euclidean RMSE" in summary and "test-only chain" in summary and "not applicable" in summary


def test_report_is_deterministic(chain, tmp_path):
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == 0
    assert tool.main(argv(chain, out)) == 0
    a, b = runs(out)
    for name in ("evaluation_report.json", "chart_data.json", "overlay.png"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    strip = lambda p: (p / "summary.md").read_text(encoding="utf-8").split("## Outputs")[0]  # noqa: E731
    assert strip(a) == strip(b)  # only the listed output paths differ


# ------------------------------------------------------------------ verification before evaluation


def test_frozen_hash_is_verified_before_evaluation(chain, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "evaluate", lambda *a: calls.append(a))
    out = tmp_path / "out"
    assert tool.main(argv(chain, out, **{"--expected-prediction-sha256": "0" * 64})) == tool.EXIT_VERIFY
    assert calls == [] and runs(out) == []


def test_default_expected_hashes_are_the_frozen_p3_values():
    assert tool.FROZEN_PREDICTION_SHA256 == "fdd7128486f6e142867d68aba699cbfe968aca8c617b2093c06b43f32d3aa37a"
    assert tool.FROZEN_TRAINING_SHA256 == "53f01ebbba7df9379727108df99b13406192e27c5a98b1ee7cc72c0417134fa3"
    assert tool.FROZEN_SCHEDULE_SHA256 == "af8a269fe096191aad5aa06aa0596b893fb1b5220e2526690c11817c10cef4ed"
    parser = tool.build_parser()
    a = parser.parse_args(["--prediction", "p", "--inference-inputs", "i", "--vault", "v"])
    assert a.expected_prediction_sha256 == tool.FROZEN_PREDICTION_SHA256


def test_a_test_chain_is_refused_against_the_frozen_defaults(chain, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "evaluate", lambda *a: calls.append(a))
    a = ["--prediction", str(chain["prediction"]), "--inference-inputs", str(chain["inputs"]),
         "--vault", str(chain["vault"]), "--output-dir", str(tmp_path / "out")]
    assert tool.main(a) == tool.EXIT_VERIFY  # any other (e.g. regenerated) artifact is not the frozen one
    assert calls == []


def test_sha_record_mismatch(chain, tmp_path):
    chain["sha"].write_text("0" * 64 + "  pred.json\n", encoding="ascii")
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == tool.EXIT_VERIFY and runs(out) == []


def test_training_and_schedule_hash_checks(chain, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "evaluate", lambda *a: calls.append(a))
    out = tmp_path / "out"
    assert tool.main(argv(chain, out, **{"--expected-training-sha256": "1" * 64})) == tool.EXIT_VERIFY
    assert tool.main(argv(chain, out, **{"--expected-schedule-sha256": "2" * 64})) == tool.EXIT_VERIFY
    assert calls == [] and runs(out) == []


def test_inputs_from_another_experiment(chain, tmp_path):
    other = make_chain_raw(tmp_path / "other")
    out = tmp_path / "out"
    a = argv(chain, out, **{"--inference-inputs": other["inputs"], "--expected-training-sha256": other["training"],
                            "--expected-schedule-sha256": other["schedule"]})
    assert tool.main(a) == tool.EXIT_VERIFY and runs(out) == []


def make_chain_raw(d: Path) -> dict:
    d.mkdir()
    scene = synth.generate(synth.SceneSpec(**{**ALT.__dict__, "y0": 61.0}))
    split = ho.split_trajectories(scene.trajectories, Fraction(7, 10))
    inputs = d / "inference_inputs.json"
    inputs.write_bytes(pd.canonical_bytes(ho.inference_inputs_to_json(split.inference)))
    return {"inputs": inputs, "training": split.inference.training_sha256, "schedule": split.inference.schedule_sha256}


def test_evaluation_failure_writes_nothing(chain, tmp_path):
    data = json.loads(chain["vault"].read_bytes())
    data["holdout_vault"]["observations"][0]["y"] = 1.0
    raw = pd.canonical_bytes(data)
    chain["vault"].write_bytes(raw)
    pd.sha256_path_for(chain["vault"]).write_text(f"{hashlib.sha256(raw).hexdigest()}  vault.json\n", encoding="ascii")
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == tool.EXIT_VERIFY and runs(out) == []


@pytest.mark.parametrize("missing", ["prediction", "sha", "inputs", "vault"])
def test_missing_files(chain, tmp_path, missing):
    chain[missing].unlink()
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == tool.EXIT_VERIFY and runs(out) == []


# ------------------------------------------------------------------ writing


def test_write_failure_leaves_no_partial_output(chain, tmp_path, monkeypatch):
    real = tool.ov.write_new
    count = []

    def third_fails(path, data):
        count.append(path)
        if len(count) == 3:
            raise OSError("disk full (simulated)")
        return real(path, data)

    monkeypatch.setattr(tool.ov, "write_new", third_fails)
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == tool.EXIT_WRITE
    assert runs(out) == []


def test_existing_run_directory_is_never_reused(chain, tmp_path, monkeypatch):
    class FixedUUID:
        hex = "deadbeef" * 4

    class FixedDatetime:
        @staticmethod
        def now(tz):
            from datetime import datetime
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(tool.uuid, "uuid4", lambda: FixedUUID())
    monkeypatch.setattr(tool, "datetime", FixedDatetime)
    out = tmp_path / "out"
    assert tool.main(argv(chain, out)) == 0
    (run_dir,) = runs(out)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    assert tool.main(argv(chain, out)) == tool.EXIT_WRITE
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


# ------------------------------------------------------------------ CLI and isolation


def test_help_and_usage(capsys):
    with pytest.raises(SystemExit) as info:
        tool.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for word in ("--prediction", "--inference-inputs", "--vault", "OFFLINE", "Exit codes"):
        assert word in text
    with pytest.raises(SystemExit) as info:
        tool.main([])
    assert info.value.code == tool.EXIT_USAGE


def test_tool_never_predicts_or_calls_apis():
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"predict", "build_prediction", "generate", "split_trajectories", "MuseSparkClient", "chat"}
    assert [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "evaluate"]
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"openai", "httpx2", "socket", "requests", "urllib", "dotenv", "decompile.synth"}


def test_registered_prediction_is_untouched_by_the_suite():
    """The registered artifact (git-ignored, local) must never be regenerated or edited by tests."""
    if not REGISTERED.is_file():
        pytest.skip("the registered prediction is not present locally")
    assert hashlib.sha256(REGISTERED.read_bytes()).hexdigest() == tool.FROZEN_PREDICTION_SHA256
