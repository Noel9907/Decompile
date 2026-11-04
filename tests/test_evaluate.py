"""Tests for decompile.evaluate (P3). OFFLINE, deterministic.

IMPORTANT: these tests never evaluate the pre-registered demo. They use a different,
test-only synthetic scene (ALT), so no withheld result of the registered experiment is
revealed by running the suite. Every artifact lives in a pytest temporary directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import evaluate as ev
from decompile import holdout as ho
from decompile import predict as pd
from decompile import synth
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))


def gt_contacts(scene):
    return [{"object_id": scene.ground_truth.spec.object_id, "t_s": c.t_s, "axis": c.axis, "plane_index": c.plane_index}
            for c in scene.ground_truth.contacts]


@pytest.fixture(scope="module")
def alt_scene():
    return synth.generate(ALT)


@pytest.fixture(scope="module")
def alt_split(alt_scene):
    return ho.split_trajectories(alt_scene.trajectories, Fraction(7, 10))


@pytest.fixture
def chain(tmp_path, alt_scene, alt_split):
    pred = pd.predict(alt_split.inference.training, alt_split.inference.schedule, output_path=tmp_path / "pred.json",
                      experiment_notes=["test-only scene; not the registered experiment"])
    ev.write_evaluation_vault(tmp_path / "vault.json", alt_split, ground_truth_contacts=gt_contacts(alt_scene),
                              notes=["vault note"])
    return tmp_path / "pred.json", tmp_path / "pred.json.sha256", tmp_path / "vault.json"


def rewrite(path: Path, edit, *, rehash=True):
    data = json.loads(path.read_bytes())
    edit(data)
    raw = pd.canonical_bytes(data)
    path.write_bytes(raw)
    if rehash:
        pd.sha256_path_for(path).write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n", encoding="ascii")


def refused(chain_paths, check):
    with pytest.raises(ev.EvaluationError) as info:
        ev.evaluate(*chain_paths)
    assert info.value.check == check, str(info.value)
    return info.value


# ------------------------------------------------------------------ metrics


def test_metrics_match_an_independent_computation(chain, alt_split):
    report = ev.evaluate(*chain).data
    (obj,) = report["objects"]
    art = json.loads(chain[0].read_bytes())["objects"][0]
    truth = [(o.x, o.y) for o in alt_split.vault.observations]
    for model in [pd.SCENE_MODEL, *pd.BASELINES]:
        px, py = art["predictions"][model]["x"], art["predictions"][model]["y"]
        dx = [a - t[0] for a, t in zip(px, truth)]
        dy = [b - t[1] for b, t in zip(py, truth)]
        e = [math.hypot(a, b) for a, b in zip(dx, dy)]
        m = obj["withheld_accuracy"][model]
        assert [f["dx_px"] for f in m["per_frame"]] == dx and [f["dy_px"] for f in m["per_frame"]] == dy
        assert m["rmse_x_px"] == pytest.approx(math.sqrt(sum(v * v for v in dx) / len(dx)))
        assert m["rmse_y_px"] == pytest.approx(math.sqrt(sum(v * v for v in dy) / len(dy)))
        assert m["rmse_px"] == pytest.approx(math.sqrt(sum(v * v for v in e) / len(e)))
        assert m["mean_error_px"] == pytest.approx(sum(e) / len(e)) and m["max_error_px"] == max(e)
        assert (m["first_error_px"], m["last_error_px"]) == (e[0], e[-1])
    ranking = obj["baseline_comparison"]["ranking_by_rmse"]
    assert sorted(ranking) == sorted([pd.SCENE_MODEL, *pd.BASELINES])
    rm = obj["baseline_comparison"]["rmse_px"]
    assert [rm[m] for m in ranking] == sorted(rm.values())


def test_horizon_is_exact_time_after_the_last_training_frame(chain, alt_split):
    obj = ev.evaluate(*chain).data["objects"][0]
    t_last = alt_split.inference.training.objects[0].observations[-1].t
    rows = obj["withheld_accuracy"][pd.SCENE_MODEL]["per_frame"]
    assert [tr.timestamp_from_json(r["horizon_s"]) for r in rows] == [s.t - t_last for s in alt_split.inference.schedule]
    assert tr.timestamp_from_json(rows[0]["horizon_s"]) == Fraction(1, 30)


def test_contact_metrics_on_the_test_scene(chain):
    c = ev.evaluate(*chain).data["objects"][0]["contacts"]
    s = c[pd.SCENE_MODEL]
    assert s["status"] == "evaluated" and s["n_actual"] == 1 and s["n_predicted"] == 1
    assert s["n_missed"] == 0 and s["n_false"] == 0
    assert len(s["excluded_training_contacts_t_s"]) == 1  # the training contact is not a withheld success
    assert s["matched"][0]["abs_error_s"] < 1e-3
    for b in pd.BASELINES:
        assert c[b] == {"status": "not_applicable", "reason": "this baseline does not model contacts"}


def test_match_contacts_one_to_one_chronological():
    m = ev.match_contacts([1.1, 0.5], [0.52, 1.0, 1.6])
    assert [(p["predicted_t_s"], p["actual_t_s"]) for p in m["matched"]] == [(0.5, 0.52), (1.1, 1.0)]
    assert m["missed_actual_t_s"] == [1.6] and m["false_predicted_t_s"] == [] and m["n_missed"] == 1
    m = ev.match_contacts([0.5, 1.0, 1.4], [0.6])
    assert m["n_false"] == 2 and m["false_predicted_t_s"] == [1.0, 1.4] and m["matched"][0]["error_s"] == pytest.approx(-0.1)
    m = ev.match_contacts([], [])
    assert m["matched"] == [] and m["mean_abs_error_s"] is None and m["max_abs_error_s"] is None


def test_vault_without_ground_truth_contacts(tmp_path, alt_split):
    pd.predict(alt_split.inference.training, alt_split.inference.schedule, output_path=tmp_path / "p.json")
    ev.write_evaluation_vault(tmp_path / "v.json", alt_split)
    c = ev.evaluate(tmp_path / "p.json", tmp_path / "p.json.sha256", tmp_path / "v.json").data["objects"][0]["contacts"]
    assert c[pd.SCENE_MODEL]["status"] == "not_evaluated"


def test_report_sections_disclosures_and_limitations(chain):
    r = ev.evaluate(*chain).data
    assert set(r["sections"]) == {"training_fit_consistency", "withheld_accuracy", "baseline_comparison"}
    obj = r["objects"][0]
    assert set(obj["training_fit_consistency"]) == {pd.SCENE_MODEL, *pd.BASELINES}
    assert r["disclosures"] == ["test-only scene; not the registered experiment", "vault note"]
    assert any("real videos" in lim for lim in r["limitations"])
    assert [c["check"] for c in r["integrity_checks"]] == [
        "prediction_sha256", "prediction_format", "vault_sha256", "inputs", "alignment", "programs"]
    assert r["prediction"]["sha256"] == hashlib.sha256(chain[0].read_bytes()).hexdigest()


def test_evaluation_is_deterministic_and_read_only(chain):
    before = [p.read_bytes() for p in (*chain, pd.sha256_path_for(chain[2]))]
    a, b = ev.evaluate(*chain), ev.evaluate(*chain)
    assert a == b
    assert [p.read_bytes() for p in (*chain, pd.sha256_path_for(chain[2]))] == before


def test_write_evaluation_report_never_overwrites(chain, tmp_path):
    r = ev.evaluate(*chain)
    path = tmp_path / "report.json"
    digest = ev.write_evaluation_report(r, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        ev.write_evaluation_report(r, path)


# ------------------------------------------------------------------ integrity failures


def test_altered_prediction_file(chain):
    raw = chain[0].read_bytes()
    chain[0].write_bytes(raw.replace(b'"phase":"withheld"', b'"phase":"WITHHELD"', 1))
    refused(chain, "prediction_sha256")


def test_altered_sha_record(chain):
    chain[1].write_text("0" * 64 + "  pred.json\n", encoding="ascii")
    refused(chain, "prediction_sha256")
    chain[1].write_text("not a record", encoding="ascii")
    refused(chain, "prediction_sha256")


def test_sha_record_naming_another_file(chain):
    digest = hashlib.sha256(chain[0].read_bytes()).hexdigest()
    chain[1].write_text(f"{digest}  other.json\n", encoding="ascii")
    refused(chain, "prediction_sha256")


def test_incomplete_artifact_without_sha_record(chain):
    chain[1].unlink()
    err = refused(chain, "prediction_sha256")
    assert "not valid" in str(err)


def test_malformed_and_non_canonical_json(chain):
    for content in (b"{bad json", json.dumps(json.loads(chain[0].read_bytes()), indent=2).encode()):
        chain[0].write_bytes(content)
        chain[1].write_text(f"{hashlib.sha256(content).hexdigest()}  pred.json\n", encoding="ascii")
        refused(chain, "prediction_sha256")


def test_wrong_format_version(chain):
    rewrite(chain[0], lambda d: d.update(prediction_format_version=99))
    refused(chain, "prediction_format")


def test_wrong_training_hash(tmp_path, alt_split):
    other_split = ho.split_trajectories(synth.generate(synth.SceneSpec(**{**ALT.__dict__, "y0": 61.0})).trajectories,
                                        Fraction(7, 10))
    pd.predict(alt_split.inference.training, alt_split.inference.schedule, output_path=tmp_path / "p.json")
    ev.write_evaluation_vault(tmp_path / "v.json", other_split)
    refused((tmp_path / "p.json", tmp_path / "p.json.sha256", tmp_path / "v.json"), "inputs")


def test_wrong_schedule_hash(chain):
    rewrite(chain[0], lambda d: d["inputs"].update(schedule_sha256="0" * 64))
    refused(chain, "inputs")


@pytest.mark.parametrize("edit", [
    lambda o: o["schedule"].pop(),                                   # missing timestamp
    lambda o: o["schedule"].append({"frame_index": 99, "t": {"num": 33, "den": 10}}),  # extra timestamp
    lambda o: o["schedule"].__setitem__(1, o["schedule"][0]),        # duplicate timestamp
    lambda o: o["schedule"][0].update(t={"num": 43, "den": 31}),     # shifted time
    lambda o: o["predictions"]["cubic_polynomial_all"]["y"].pop(),   # incomplete model series
])
def test_alignment_failures(chain, edit):
    rewrite(chain[0], lambda d: edit(d["objects"][0]))
    refused(chain, "alignment")


def test_missing_object_predictions(chain):
    rewrite(chain[0], lambda d: d.update(objects=[]))
    refused(chain, "alignment")


def test_tampered_program(chain):
    rewrite(chain[0], lambda d: d["objects"][0]["program"]["planes"][0].update(restitution=0.9))
    refused(chain, "programs")


def test_tampered_vault(chain):
    rewrite(chain[2], lambda d: d["holdout_vault"]["observations"][0].update(y=1.0))
    refused(chain, "vault_sha256")
    rewrite(chain[2], lambda d: None, rehash=False)
    pd.sha256_path_for(chain[2]).write_text("0" * 64 + "  vault.json\n", encoding="ascii")
    refused(chain, "vault_sha256")


def test_vault_never_overwritten(tmp_path, alt_split):
    ev.write_evaluation_vault(tmp_path / "v.json", alt_split)
    with pytest.raises(FileExistsError):
        ev.write_evaluation_vault(tmp_path / "v.json", alt_split)


# ------------------------------------------------------------------ the evaluator is the only vault reader


def test_only_the_evaluator_reads_vault_data():
    for path in sorted((ROOT / "decompile").glob("*.py")):
        if path.name == "evaluate.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported_from_decompile = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                                   and n.module == "decompile" for a in n.names}
        assert "decompile.evaluate" not in mods and "evaluate" not in imported_from_decompile, path.name
        for forbidden in ("evaluation_vault", "holdout_vault", "write_evaluation_vault", "EvaluationError"):
            assert forbidden not in source, (path.name, forbidden)


# ------------------------------------------------------------------ experiment-specific limitations (P5)


P3_LIMITATIONS_TEXT = (
    "Synthetic data from decompile.synth: one pre-registered scene, positions rounded to 0.01 px and no other "
    "noise. This says nothing about performance on real videos.",
    "Contact matching is chronological one-to-one without a time gate; timing errors are reported as measured.",
    "Point predictions only: parameter uncertainty is not propagated to prediction bands.",
    "Units are image pixels and seconds; nothing is calibrated to metres.",
)


def test_default_limitations_are_the_unchanged_p3_text(chain):
    assert ev.LIMITATIONS == P3_LIMITATIONS_TEXT
    default = ev.evaluate(*chain)
    explicit_none = ev.evaluate(*chain, limitations=None)
    assert default.data["limitations"] == list(P3_LIMITATIONS_TEXT)
    assert pd.canonical_bytes(default.data) == pd.canonical_bytes(explicit_none.data)


def test_supplied_limitations_are_recorded_and_nothing_else_changes(chain):
    custom = ["SYNTHETIC VIDEO experiment (test).", "Truth is tracker measurement."]
    default = ev.evaluate(*chain).data
    report = ev.evaluate(*chain, limitations=custom).data
    assert report["limitations"] == custom
    assert {k: v for k, v in report.items() if k != "limitations"} ==         {k: v for k, v in default.items() if k != "limitations"}
    assert pd.canonical_bytes(report) == pd.canonical_bytes(ev.evaluate(*chain, limitations=tuple(custom)).data)


@pytest.mark.parametrize("bad", ["one string", b"bytes", [], (), ["ok", ""], ["ok", "   "], ["ok", 3], [None], 7])
def test_invalid_limitations_are_rejected(chain, bad):
    with pytest.raises(ValueError):
        ev.evaluate(*chain, limitations=bad)
