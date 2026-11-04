"""Tests for decompile.holdout (P2, A7-lite) and the training-data leakage invariance.

OFFLINE, deterministic. Trajectories are SYNTHETIC (decompile.synth, pre-registered demo).
P2 establishes that the training data, its kinematic fit and the scene program built from
it do not depend on withheld positions. Prediction and evaluation arrive in P3; full
prediction invariance is not claimed here.
"""

from __future__ import annotations

import ast
import copy
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import holdout as ho
from decompile import kinematic_fit as kf
from decompile import scene_program as sp
from decompile import synth
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo():
    return synth.generate(synth.PREREGISTERED_DEMO)


@pytest.fixture(scope="module")
def split(demo):
    return ho.split_trajectories(demo.trajectories, synth.PREREGISTERED_TRAIN_FRACTION)


def mutate_withheld(traj: tr.TrajectorySet, cutoff: int, seed: int) -> tr.TrajectorySet:
    """Arbitrary withheld positions; timestamps, frame indices and object ids preserved."""
    rng = np.random.default_rng(seed)
    data = traj.to_json_dict()
    for obj in data["objects"]:
        for ob in obj["observations"]:
            if ob["frame_index"] >= cutoff:
                ob["x"], ob["y"] = float(rng.uniform(-1e4, 1e4)), float(rng.uniform(-1e4, 1e4))
    return tr.TrajectorySet.from_json_dict(data)


def two_objects(missing=()) -> tr.TrajectorySet:
    def obs(oid, k):
        return {"frame_index": k, "t": {"num": Fraction(k, 30).numerator, "den": Fraction(k, 30).denominator},
                "x": 10.0 + k, "y": 20.0 + (2 if oid == "obj_02" else 1) * k}
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {**tr.COORDINATE_CONSTANTS, "image_width": 640, "image_height": 480},
        "objects": [{"object_id": "obj_01", "observations": [obs("obj_01", k) for k in range(20)]},
                    {"object_id": "obj_02", "observations": [obs("obj_02", k) for k in range(20) if k not in missing]}],
    })


# ------------------------------------------------------------------ the pre-registered split


def test_demo_split_indices_and_timestamps(split, demo):
    inp = split.inference
    assert (inp.n_frames, inp.n_train_frames, inp.first_withheld_frame) == (50, 35, 35)
    assert inp.train_fraction == Fraction(7, 10)
    train_obs = inp.training.objects[0].observations
    assert [o.frame_index for o in train_obs] == list(range(35))
    assert train_obs[-1].t == Fraction(34, 30)
    assert [(s.frame_index, s.t) for s in inp.schedule] == [(f, Fraction(f, 30)) for f in range(35, 50)]
    assert all(isinstance(s.t, Fraction) for s in inp.schedule)
    # contacts (ground truth, test-only): the first is in training, the second and third are withheld
    c = demo.ground_truth.contacts
    assert c[0].t_s < float(inp.schedule[0].t) <= c[1].t_s


def test_withheld_coordinates_stay_associated(split, demo):
    full = {o.frame_index: o for o in demo.trajectories.objects[0].observations}
    assert len(split.vault.observations) == 15
    for w in split.vault.observations:
        o = full[w.frame_index]
        assert (w.object_id, w.t, w.x, w.y) == ("obj_01", o.t, o.x, o.y)
    assert [s.frame_index for s in split.inference.schedule] == [w.frame_index for w in split.vault.observations]


def test_inference_inputs_contain_no_withheld_position(split):
    inp_json = ho.inference_inputs_to_json(split.inference)
    assert set(inp_json["schedule"][0]) == {"object_id", "frame_index", "t"}
    train_frames = {o["frame_index"] for o in inp_json["training"]["objects"][0]["observations"]}
    assert train_frames.isdisjoint({w.frame_index for w in split.vault.observations})
    text = ho.canonical_json(inp_json)
    assert "vault" not in text and "withheld_vault" not in text
    assert "ScheduledFrame" not in text
    assert {f.name for f in ho.ScheduledFrame.__dataclass_fields__.values()} == {"object_id", "frame_index", "t"}
    assert not any(k in inp_json for k in ("vault_sha256", "full_sha256", "observations"))


def test_training_contains_no_ground_truth(split):
    data = split.inference.training.to_json_dict()
    assert set(data) == {"schema", "synthetic", "source", "coordinate_system", "objects"}
    text = str(ho.inference_inputs_to_json(split.inference)).lower()
    for word in ("restitution", "contact", "truth", "plane", "rk4", "450.0, 'keep"):
        assert word not in text


def test_hashes(split):
    inp = split.inference
    assert inp.training_sha256 == ho.sha256_canonical(inp.training.to_json_dict())
    assert inp.schedule_sha256 == ho.sha256_canonical(ho.schedule_to_json(inp.schedule))
    assert split.vault.training_sha256 == inp.training_sha256
    assert split.vault.vault_sha256 == ho.sha256_canonical(ho.vault_to_json(
        ho.WithheldVault(split.vault.holdout_version, split.vault.observations, split.vault.training_sha256, "")))
    assert len({inp.training_sha256, inp.schedule_sha256, split.vault.vault_sha256}) == 3


# ------------------------------------------------------------------ leakage invariance


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_training_and_fitted_program_ignore_withheld_positions(split, demo, seed):
    mutated = mutate_withheld(demo.trajectories, split.inference.first_withheld_frame, seed)
    assert mutated != demo.trajectories  # the full data really differs
    other = ho.split_trajectories(mutated, synth.PREREGISTERED_TRAIN_FRACTION)
    a, b = split.inference, other.inference
    # training data: byte-identical canonical JSON and identical hash
    assert ho.canonical_json(a.training.to_json_dict()) == ho.canonical_json(b.training.to_json_dict())
    assert a.training_sha256 == b.training_sha256 and a.schedule_sha256 == b.schedule_sha256
    assert ho.canonical_json(ho.inference_inputs_to_json(a)) == ho.canonical_json(ho.inference_inputs_to_json(b))
    # the vault records the change
    assert other.vault.vault_sha256 != split.vault.vault_sha256
    # fitted results and scene programs are identical
    ka, kb = kf.analyze_kinematics(a.training), kf.analyze_kinematics(b.training)
    assert ka == kb
    pa, pb = sp.program_from_kinematics(ka.objects[0]), sp.program_from_kinematics(kb.objects[0])
    assert sp.program_sha256(pa) == sp.program_sha256(pb)
    assert pa == pb


def test_training_hash_is_sensitive_to_training_values(split, demo):
    data = demo.trajectories.to_json_dict()
    data["objects"][0]["observations"][10]["y"] += 0.01
    other = ho.split_trajectories(tr.TrajectorySet.from_json_dict(data), Fraction(7, 10))
    assert other.inference.training_sha256 != split.inference.training_sha256


def test_demo_training_supports_a_scene_program(split):
    """Only the training data is used here; nothing about withheld performance is examined."""
    prog = sp.program_from_kinematics(kf.analyze_kinematics(split.inference.training).objects[0])
    assert prog.planes[0].axis == "y" and prog.planes[0].keep_side == "less"
    assert prog.provenance["fitted_span"]["t_last"] == {"num": 17, "den": 15}  # 34/30: training only


# ------------------------------------------------------------------ general behaviour


def test_multiple_objects_share_the_split_frame():
    s = ho.split_trajectories(two_objects(missing=(3, 15, 16)), Fraction(7, 10))
    assert s.inference.first_withheld_frame == 14
    t1, t2 = s.inference.training.objects
    assert [o.frame_index for o in t1.observations] == list(range(14))
    assert [o.frame_index for o in t2.observations] == [k for k in range(14) if k != 3]
    assert {(w.object_id, w.frame_index) for w in s.vault.observations} == (
        {("obj_01", k) for k in range(14, 20)} | {("obj_02", k) for k in range(14, 20) if k not in (15, 16)})
    assert [(x.object_id, x.frame_index) for x in s.inference.schedule] == [
        (w.object_id, w.frame_index) for w in s.vault.observations]


@pytest.mark.parametrize("bad", [0, 1, Fraction(3, 2), -1, Fraction(0)])
def test_invalid_fractions(demo, bad):
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(demo.trajectories, bad)


@pytest.mark.parametrize("bad", [0.7, True, "7/10", None])
def test_non_exact_fractions_rejected(demo, bad):
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(demo.trajectories, bad)


def test_insufficient_training_data():
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(two_objects(), Fraction(1, 20))  # 1 training frame
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(two_objects(missing=(0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)), Fraction(7, 10))
    small = tr.TrajectorySet.from_json_dict({**two_objects().to_json_dict(), "objects": [
        {"object_id": "obj_01", "observations": two_objects().to_json_dict()["objects"][0]["observations"][:3]}]})
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(small, Fraction(7, 10))  # floor(2.1) = 2 training observations


def test_non_trajectory_set_rejected(demo):
    with pytest.raises(ho.HoldoutError):
        ho.split_trajectories(demo.trajectories.to_json_dict())


def test_input_is_not_mutated(demo):
    before = copy.deepcopy(demo.trajectories.to_json_dict())
    ho.split_trajectories(demo.trajectories, Fraction(7, 10))
    assert demo.trajectories.to_json_dict() == before


def test_split_is_deterministic(demo):
    assert ho.split_trajectories(demo.trajectories) == ho.split_trajectories(demo.trajectories)


def test_floor_uses_exact_arithmetic():
    s = ho.split_trajectories(two_objects(), Fraction(7, 10))  # 20 * 7/10 = 14 exactly
    assert s.inference.n_train_frames == 14 == math.floor(Fraction(7, 10) * 20)


# ------------------------------------------------------------------ import boundaries


def _module_references(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    return imported, names


@pytest.mark.parametrize("module", ["decompile/kinematic_fit.py", "decompile/scene_program.py"])
def test_inference_modules_do_not_depend_on_holdout_or_ground_truth(module):
    imported, names = _module_references(ROOT / module)
    joined = " ".join(sorted(imported))
    for forbidden in ("holdout", "synth", "evaluate", "vault"):
        assert forbidden not in joined, (module, forbidden)
    for forbidden in ("split_trajectories", "WithheldVault", "ground_truth", "GroundTruth", "vault"):
        assert forbidden not in names, (module, forbidden)


def test_holdout_imports_only_trajectories():
    imported, _ = _module_references(ROOT / "decompile/holdout.py")
    assert {m for m in imported if m.startswith("decompile")} == {"decompile", "decompile.trajectories"}
