"""Tests for decompile.predict (P3). OFFLINE, deterministic.

Prediction-side only. The pre-registered demo is used here ONLY through its training data
and timestamp schedule; no test reads or evaluates its withheld positions. Artifacts are
written to pytest temporary directories.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import holdout as ho
from decompile import kinematic_fit as kf
from decompile import predict as pd
from decompile import scene_program as sp
from decompile import synth
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))


@pytest.fixture(scope="module")
def demo_split():
    return ho.split_trajectories(synth.generate(synth.PREREGISTERED_DEMO).trajectories, Fraction(7, 10))


@pytest.fixture(scope="module")
def demo_artifact(demo_split):
    return pd.build_prediction(demo_split.inference.training, demo_split.inference.schedule)


def mutate_withheld(traj, cutoff, seed):
    rng = np.random.default_rng(seed)
    data = traj.to_json_dict()
    for obj in data["objects"]:
        for ob in obj["observations"]:
            if ob["frame_index"] >= cutoff:
                ob["x"], ob["y"] = float(rng.uniform(-1e4, 1e4)), float(rng.uniform(-1e4, 1e4))
    return tr.TrajectorySet.from_json_dict(data)


def run(split, path, **kw):
    return pd.predict(split.inference.training, split.inference.schedule, output_path=path, **kw)


# ------------------------------------------------------------------ leakage and determinism


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_prediction_is_invariant_to_withheld_positions(demo_split, tmp_path, seed):
    full = synth.generate(synth.PREREGISTERED_DEMO).trajectories
    other = ho.split_trajectories(mutate_withheld(full, 35, seed), Fraction(7, 10))
    a = run(demo_split, tmp_path / "a.json")
    b = run(other, tmp_path / "b.json")
    assert a.artifact_path.read_bytes() == b.artifact_path.read_bytes()
    assert a.sha256 == b.sha256
    oa, ob = a.artifact["objects"][0], b.artifact["objects"][0]
    assert oa["predictions"][pd.SCENE_MODEL] == ob["predictions"][pd.SCENE_MODEL]
    for name in pd.BASELINES:
        assert oa["predictions"][name] == ob["predictions"][name]
        assert oa["baselines"][name] == ob["baselines"][name]
    assert oa["predicted_events"] == ob["predicted_events"]


def test_serialisation_and_hash_are_deterministic(demo_split, tmp_path):
    a, b = run(demo_split, tmp_path / "a.json"), run(demo_split, tmp_path / "b.json")
    assert a.artifact_path.read_bytes() == b.artifact_path.read_bytes() and a.sha256 == b.sha256
    assert a.sha256 == hashlib.sha256(a.artifact_path.read_bytes()).hexdigest()
    assert a.sha256_path.read_text(encoding="ascii") == f"{a.sha256}  a.json\n"
    assert a.artifact_path.read_bytes() == pd.canonical_bytes(json.loads(a.artifact_path.read_bytes()))


def test_artifact_inputs_and_exclusions(demo_split, demo_artifact):
    inp = demo_split.inference
    assert demo_artifact["inputs"] == {"training_sha256": inp.training_sha256, "schedule_sha256": inp.schedule_sha256}
    text = json.dumps(demo_artifact)
    for forbidden in ("vault", "ground_truth", "withheld_vault", "full_sha256", "rmse"):
        assert forbidden not in text, forbidden

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert "observations" not in set(keys(demo_artifact))  # no raw position arrays of any kind
    assert demo_split.vault.vault_sha256 not in text
    assert set(demo_artifact) == {"kind", "prediction_format_version", "predictor_version", "versions", "inputs",
                                  "kinematics_noise", "baseline_specifications", "objects", "experiment_notes",
                                  "excludes", "note"}


def test_experiment_notes_recorded(demo_split):
    art = pd.build_prediction(demo_split.inference.training, demo_split.inference.schedule,
                              experiment_notes=["P2 disclosure: contact times were previewed"])
    assert art["experiment_notes"] == ["P2 disclosure: contact times were previewed"]


# ------------------------------------------------------------------ content


def test_exact_timestamps_and_object_ids(demo_split, demo_artifact):
    (obj,) = demo_artifact["objects"]
    assert obj["object_id"] == "obj_01"
    assert [(e["frame_index"], tr.timestamp_from_json(e["t"])) for e in obj["schedule"]] == [
        (s.frame_index, s.t) for s in demo_split.inference.schedule]
    assert obj["training_last_frame"] == 34 and obj["training_last_t"] == {"num": 17, "den": 15}
    for m in [pd.SCENE_MODEL, *pd.BASELINES]:
        assert len(obj["predictions"][m]["x"]) == len(obj["predictions"][m]["y"]) == 15


def test_scene_prediction_is_the_simulated_program(demo_split, demo_artifact):
    obj = demo_artifact["objects"][0]
    prog = sp.program_from_json(obj["program"])
    assert sp.program_sha256(prog) == obj["program_sha256"]
    kin = kf.analyze_kinematics(demo_split.inference.training)
    assert prog == sp.program_from_kinematics(kin.objects[0])
    r = sp.simulate(prog, [s.t for s in demo_split.inference.schedule])
    assert obj["predictions"][pd.SCENE_MODEL] == {"x": list(r.x_px), "y": list(r.y_px)}
    assert [e["t_s"] for e in obj["predicted_events"]] == [e.t_s for e in r.events]
    phases = [e["phase"] for e in obj["predicted_events"]]
    assert phases[0] == "training" and set(phases[1:]) <= {"withheld"}


def test_baselines_use_documented_training_windows(demo_split, demo_artifact):
    obs = demo_split.inference.training.objects[0].observations
    obj = demo_artifact["objects"][0]
    t_last = obs[-1].t
    t_contact = obj["kinematics"]["t_contact_s"]
    windows = {"constant_velocity_last8": obs[-8:], "constant_acceleration_last8": obs[-8:],
               "cubic_polynomial_all": obs, "post_contact_parabola": [o for o in obs if float(o.t) > t_contact]}
    assert [o.frame_index for o in windows["post_contact_parabola"]] == list(range(28, 35))
    times = [s.t for s in demo_split.inference.schedule]
    for name, win in windows.items():
        b = obj["baselines"][name]
        assert b["window_frames"] == [win[0].frame_index, win[-1].frame_index] and b["n_obs"] == len(win)
        assert b["models_contacts"] is False and b["t_ref"] == {"num": 17, "den": 15}
        tau = np.array([float(o.t - t_last) for o in win])
        for axis in ("x", "y"):
            vals = np.array([getattr(o, axis) for o in win])
            coef = np.polynomial.polynomial.polyfit(tau, vals, b["degree"])
            assert np.allclose(b["coefficients"][axis], coef, rtol=1e-9, atol=1e-6)
            pred = np.polynomial.polynomial.polyval(np.array([float(t - t_last) for t in times]), coef)
            assert np.allclose(obj["predictions"][name][axis], pred, rtol=1e-9, atol=1e-6)


def test_training_fit_consistency_is_training_only(demo_artifact):
    tfc = demo_artifact["objects"][0]["training_fit_consistency"]
    assert set(tfc) == {pd.SCENE_MODEL, *pd.BASELINES}
    assert tfc[pd.SCENE_MODEL]["n_obs"] == 35 and tfc[pd.SCENE_MODEL]["rms_px"] < 0.01
    assert "not held-out accuracy" in demo_artifact["note"]


# ------------------------------------------------------------------ refusals


def test_unsupported_scene_program_refused(tmp_path):
    spec = synth.SceneSpec("obj_01", 30, 40, 100.0, 60.0, 50.0, 0.0, 0.0, 300.0)  # no plane: no contact
    split = ho.split_trajectories(synth.generate(spec).trajectories, Fraction(7, 10))
    with pytest.raises(pd.PredictionError, match="no scene program"):
        run(split, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_insufficient_post_contact_training_data(tmp_path):
    spec = replace(ALT, n_frames=39)  # training frames 0..26; the contact is at ~0.796 s (frame 23.9)
    split = ho.split_trajectories(synth.generate(spec).trajectories, Fraction(7, 10))
    with pytest.raises(pd.PredictionError) as info:
        run(split, tmp_path / "p.json")
    assert "after the fitted contact" in str(info.value) or "no scene program" in str(info.value)
    assert list(tmp_path.iterdir()) == []


def test_invalid_training_rejected(demo_split):
    with pytest.raises(pd.PredictionError):
        pd.build_prediction(demo_split.inference.training.to_json_dict(), demo_split.inference.schedule)


@pytest.mark.parametrize("edit", [
    lambda s: (),
    lambda s: s[:1] + ("not a frame",),
    lambda s: s[:1] + (ho.ScheduledFrame("obj_99", 40, Fraction(40, 30)),),
    lambda s: (ho.ScheduledFrame("obj_01", 30, Fraction(30, 30)),) + s,
    lambda s: (s[1], s[0]) + s[2:],
    lambda s: (replace(s[0], t=0.5),) + s[1:],
    lambda s: (replace(s[0], t=Fraction(36, 30)),) + s[1:],
])
def test_invalid_schedules_rejected(demo_split, edit):
    with pytest.raises(pd.PredictionError):
        pd.build_prediction(demo_split.inference.training, edit(demo_split.inference.schedule))


def test_inputs_are_not_mutated(demo_split):
    t_before = copy.deepcopy(demo_split.inference.training.to_json_dict())
    s_before = tuple(demo_split.inference.schedule)
    pd.build_prediction(demo_split.inference.training, demo_split.inference.schedule)
    assert demo_split.inference.training.to_json_dict() == t_before and demo_split.inference.schedule == s_before


# ------------------------------------------------------------------ writing


def test_never_overwrites(demo_split, tmp_path):
    p = tmp_path / "p.json"
    p.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(demo_split, p)
    assert p.read_text(encoding="utf-8") == "keep" and not pd.sha256_path_for(p).exists()
    q = tmp_path / "q.json"
    pd.sha256_path_for(q).write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(demo_split, q)
    assert not q.exists()


def test_artifact_write_failure_leaves_nothing(demo_split, tmp_path, monkeypatch):
    def fail(src, dst):
        raise OSError("disk full (simulated)")
    monkeypatch.setattr(pd.os, "replace", fail)
    with pytest.raises(OSError):
        run(demo_split, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_sha_write_failure_removes_the_artifact(demo_split, tmp_path, monkeypatch):
    real = pd.os.replace
    calls = []

    def second_fails(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("disk full (simulated)")
        return real(src, dst)

    monkeypatch.setattr(pd.os, "replace", second_fails)
    with pytest.raises(pd.PredictionWriteError):
        run(demo_split, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []  # no artifact without its SHA-256 record


def test_interrupt_during_sha_write_removes_the_artifact(demo_split, tmp_path, monkeypatch):
    real = pd.os.replace
    calls = []

    def interrupt(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(pd.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(demo_split, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ inference-inputs file


def test_load_inference_inputs_round_trip_and_tamper(demo_split, tmp_path):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(ho.inference_inputs_to_json(demo_split.inference)), encoding="utf-8")
    training, schedule = pd.load_inference_inputs(path)
    assert training == demo_split.inference.training and schedule == demo_split.inference.schedule
    data = json.loads(path.read_text(encoding="utf-8"))
    data["training"]["objects"][0]["observations"][3]["y"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(pd.PredictionError, match="training_sha256"):
        pd.load_inference_inputs(path)


# ------------------------------------------------------------------ isolation


def test_predictor_never_touches_evaluation_code():
    tree = ast.parse((ROOT / "decompile/predict.py").read_text(encoding="utf-8"))
    imported, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    joined = " ".join(sorted(imported))
    for forbidden in ("evaluate", "synth", "split_trajectories", "WithheldVault", "vault_to_json"):
        assert forbidden not in joined, forbidden
    for forbidden in ("vault", "WithheldVault", "vault_to_json", "split_trajectories", "ground_truth", "GroundTruth",
                      "SyntheticScene", "evaluate"):
        assert forbidden not in names, forbidden
    holdout_names = {m.split(".", 2)[-1] for m in imported if m.startswith("decompile.holdout.")}
    assert holdout_names == {"ScheduledFrame", "schedule_to_json", "sha256_canonical"}
