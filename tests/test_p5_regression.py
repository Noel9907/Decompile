"""P5 regression guard (OFFLINE): the P5 video pipeline must keep producing byte-identical outputs.

The golden hashes below were computed with the P5 code (commit efb3045) BEFORE any P6 change,
on a test-only synthetic scene (never the registered experiment scene), in a temporary
directory. Any P6 change to the tracker, renderer, holdout, fitting, prediction, evaluation or
overlay that alters P5 behaviour with default settings fails this test.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from decompile import synth

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))
GOLDEN = {
    "video/synthetic_video.mp4": "27719bd3f5a9efccf761b052307a5664aa975f42350e9f0d52d15af914e00b70",
    "measurement/tracked_trajectories.json": "eb8f395e1f8f3cece583a820fe536ca5701bc50da42844658fcbfa99ffbae922",
    "measurement/tracking_report.json": "db4735fba491f4007ffd6d5acba64b3e2f5d153194fa0497a73b3b02322bf736",
    "measurement/tracking_gate.json": "cce4142146d3f9def43235cbee89d00776508642abce05d06be45680a3c9c34e",
    "inference/inference_inputs.json": "80b053eaef27eef3da8ea463b8a7deb138f2792dd9ca210bd4779320d842cbaf",
    "inference/fit_gate.json": "f0aa7b066d83d67b0492c1a3a37b16950e0180c312d1508dca687bed86c07083",
    "inference/prediction.json": "5a0d0bb00c43d975e124a306dedde76a68e6ad7c399ae44af45e15ff537f3a98",
    "reveal/chart_data.json": "c391e6eefd698e32bd78d7ef512c743ffb730c0ab617fdc386a15699da9e5416",
    "reveal/overlay.png": "3fa335d605efd00d7340cc623a3c0c34df2fd5d26bbcdde8376f1ee6bde121c9",
    "reveal/tracking_accuracy.json": "1abf74d6bbe6313fe53c747f2994f657c5552f4b0f059b9038a2b0d576016d50",
}


def _load_tool():
    spec = importlib.util.spec_from_file_location("p5_tool_regression", ROOT / "tools" / "video_predictive_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def p5_run(tmp_path_factory):
    tool = _load_tool()
    r = tool.run(tmp_path_factory.mktemp("p5_regression"), spec=ALT, notes=["P5 regression golden (test-only scene)"],
                 experiment_id="g")
    return r["exp_dir"]


@pytest.mark.parametrize("rel", sorted(GOLDEN))
def test_p5_outputs_are_byte_identical_to_the_pre_p6_golden_run(p5_run, rel):
    assert hashlib.sha256((p5_run / rel).read_bytes()).hexdigest() == GOLDEN[rel], rel
