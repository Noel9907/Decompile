"""Tests for decompile.overlay (P4). OFFLINE, deterministic.

Never touches the registered experiment: a test-only synthetic scene (ALT) is predicted,
vaulted and evaluated in a temporary directory. The overlay itself receives only the
training data, the prediction artifact and the evaluation report.
"""

from __future__ import annotations

import ast
import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import evaluate as ev
from decompile import holdout as ho
from decompile import overlay as ov
from decompile import predict as pd
from decompile import synth

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))
NOTE = "Disclosure: test-only scene; contact times previewed during development (test note)."


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    d = tmp_path_factory.mktemp("overlay_chain")
    scene = synth.generate(ALT)
    split = ho.split_trajectories(scene.trajectories, Fraction(7, 10))
    pd.predict(split.inference.training, split.inference.schedule, output_path=d / "pred.json", experiment_notes=[NOTE])
    contacts = [{"object_id": "obj_01", "t_s": c.t_s, "axis": c.axis, "plane_index": c.plane_index}
                for c in scene.ground_truth.contacts]
    ev.write_evaluation_vault(d / "vault.json", split, ground_truth_contacts=contacts)
    report = ev.evaluate(d / "pred.json", d / "pred.json.sha256", d / "vault.json").data
    artifact = json.loads((d / "pred.json").read_bytes())
    return {"scene": scene, "split": split, "artifact": artifact, "report": report}


@pytest.fixture(scope="module")
def chart(chain):
    return ov.build_chart_data(chain["split"].inference.training, chain["artifact"], chain["report"])


# ------------------------------------------------------------------ chart data


def test_training_and_withheld_coordinates(chart, chain):
    obs = chain["split"].inference.training.objects[0].observations
    assert chart["training"] == {"t_s": [float(o.t) for o in obs], "x": [o.x for o in obs], "y": [o.y for o in obs]}
    vault = chain["split"].vault.observations  # test-side only: the overlay never sees this
    assert chart["withheld_truth"]["t_s"] == [float(w.t) for w in vault]
    assert max(abs(a - w.x) for a, w in zip(chart["withheld_truth"]["x"], vault)) <= 1e-9
    assert max(abs(a - w.y) for a, w in zip(chart["withheld_truth"]["y"], vault)) <= 1e-9
    assert "prediction - dx_px" in chart["withheld_truth"]["derivation"]


def test_every_prediction_series_is_the_frozen_artifact(chart, chain):
    art = chain["artifact"]["objects"][0]["predictions"]
    assert set(chart["predictions"]) == {pd.SCENE_MODEL, *pd.BASELINES}
    for m, series in chart["predictions"].items():
        assert series["x"] == art[m]["x"] and series["y"] == art[m]["y"]
    assert set(chart["baseline_curves"]) == set(pd.BASELINES)
    sc = chart["scene_program_curve"]
    assert sc["t_s"][0] == 0.0 and sc["t_s"][-1] == chart["withheld_truth"]["t_s"][-1]


def test_split_marker_position(chart, chain):
    s = chart["split"]
    inp = chain["split"].inference
    t_last, t_first = inp.training.objects[0].observations[-1].t, inp.schedule[0].t
    assert s["t_split"] == {"num": ((t_last + t_first) / 2).numerator, "den": ((t_last + t_first) / 2).denominator}
    assert s["t_last_training_s"] < s["t_split_s"] < s["t_first_withheld_s"]
    assert (s["n_training"], s["n_withheld"]) == (42, 18)


def test_contacts(chart, chain):
    phases = [c["phase"] for c in chart["predicted_contacts"]]
    assert phases == ["training", "withheld"]
    gt = [c.t_s for c in chain["scene"].ground_truth.contacts]
    assert chart["actual_withheld_contacts_t_s"] == [gt[1]]
    assert chart["excluded_training_contacts_t_s"] == [gt[0]]


def test_disclosure_and_caption(chart):
    assert NOTE in chart["disclosures"]
    assert "saw only the training frames" in chart["caption"]


def test_report_and_artifact_are_not_mutated(chain):
    rep, art = copy.deepcopy(chain["report"]), copy.deepcopy(chain["artifact"])
    ov.build_chart_data(chain["split"].inference.training, chain["artifact"], chain["report"])
    assert chain["report"] == rep and chain["artifact"] == art


def test_inconsistent_inputs_are_refused(chain):
    training = chain["split"].inference.training
    bad = copy.deepcopy(chain["report"])
    bad["objects"][0]["withheld_accuracy"]["cubic_polynomial_all"]["per_frame"][3]["dx_px"] += 0.5
    with pytest.raises(ov.OverlayError, match="disagrees"):
        ov.build_chart_data(training, chain["artifact"], bad)
    bad = copy.deepcopy(chain["report"])
    bad["objects"][0]["withheld_accuracy"][pd.SCENE_MODEL]["per_frame"][0]["t"] = {"num": 1, "den": 7}
    with pytest.raises(ov.OverlayError, match="frame times"):
        ov.build_chart_data(training, chain["artifact"], bad)
    art = copy.deepcopy(chain["artifact"])
    art["objects"][0]["program"]["planes"][0]["restitution"] = 0.9
    with pytest.raises(ov.OverlayError, match="program"):
        ov.build_chart_data(training, art, chain["report"])
    art = copy.deepcopy(chain["artifact"])
    art["objects"][0]["predictions"][pd.SCENE_MODEL]["y"][0] += 1.0
    with pytest.raises(ov.OverlayError):
        ov.build_chart_data(training, art, chain["report"])


# ------------------------------------------------------------------ figure


def test_figure_contents(chart):
    fig = ov.draw_figure(chart)
    try:
        ax_y, ax_xy, ax_e = fig.axes[:3]
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        for m in [pd.SCENE_MODEL, *pd.BASELINES]:
            assert ov.STYLES[m]["label"] in labels
        for needed in ("training observations (seen by the predictor)", "withheld truth (revealed after freezing)",
                       "train / withheld split", "predicted withheld contact", "actual withheld contact time"):
            assert needed in labels
        split_lines = [ln for ln in ax_y.get_lines() if list(ln.get_xdata()) == [chart["split"]["t_split_s"]] * 2]
        assert split_lines  # the split marker sits at t_split
        train_line = next(ln for ln in ax_y.get_lines() if ln.get_label().startswith("training observations"))
        truth_line = next(ln for ln in ax_y.get_lines() if ln.get_label().startswith("withheld truth"))
        assert list(train_line.get_ydata()) == chart["training"]["y"]
        assert list(truth_line.get_ydata()) == chart["withheld_truth"]["y"]
        assert train_line.get_marker() != truth_line.get_marker()
        assert ax_y.yaxis_inverted() and ax_xy.yaxis_inverted()
        assert "(s)" in ax_y.get_xlabel() and "px" in ax_y.get_ylabel() and "px" in ax_xy.get_xlabel()
        texts = " ".join(t.get_text() for t in fig.texts)
        assert NOTE in texts and "saw only the training frames" in texts
        assert "SYNTHETIC" in fig._suptitle.get_text()
        assert len(ax_e.get_lines()) == 5
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_png_is_deterministic(chart):
    a, b = ov.render_png(chart), ov.render_png(chart)
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and a == b


# ------------------------------------------------------------------ writing


def test_write_new_never_overwrites_and_leaves_no_partial(tmp_path, monkeypatch):
    p = tmp_path / "x.png"
    ov.write_new(p, b"one")
    with pytest.raises(FileExistsError):
        ov.write_new(p, b"two")
    assert p.read_bytes() == b"one"

    def fail(src, dst):
        raise OSError("disk full (simulated)")
    monkeypatch.setattr(ov.os, "replace", fail)
    with pytest.raises(OSError):
        ov.write_new(tmp_path / "y.png", b"data")
    assert sorted(x.name for x in tmp_path.iterdir()) == ["x.png"]


# ------------------------------------------------------------------ isolation


def test_overlay_cannot_read_the_vault():
    tree = ast.parse((ROOT / "decompile/overlay.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    from_decompile = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "decompile"
                      for a in n.names}
    assert from_decompile == {"scene_program", "trajectories"}
    assert not {m for m in mods if m and ("evaluate" in m or "holdout" in m or "synth" in m or "predict" in m)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("read_bytes", "read_text", "load", "loads", "strict_json_loads", "vault", "evaluate"):
        assert forbidden not in names, forbidden
