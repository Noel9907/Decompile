"""Tests for decompile.spark_stage (P6b). OFFLINE: every request goes to an injected fake transport.

The training data is a test-only synthetic scene split 70/30 by decompile.holdout; withheld
coordinates exist here only so the tests can prove they never reach the model or the artifact.
"""

from __future__ import annotations

import ast
import json
import random
from fractions import Fraction
from pathlib import Path

import httpx2
import pytest

from decompile import holdout as ho
from decompile import meta_api as m
from decompile import scene_interpretation as si
from decompile import scene_prompt as spr
from decompile import spark_stage as ss
from decompile import synth

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 60, 100.0, 60.0, 50.0, 0.0, 0.0, 1200.0, planes=(synth.Plane("y", 440.0, "less", 0.5),))
FAKE_KEY = "sk-OFFLINE-FAKE-KEY-spark-stage-0123456789"


def completion(content: str) -> dict:
    return {"id": "chatcmpl-fake-p6b", "object": "chat.completion", "created": 0, "model": "muse-spark-1.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 900, "total_tokens": 1900,
                      "completion_tokens_details": {"reasoning_tokens": 400}}}


class FakeServer:
    """Replays scripted (status, body) replies and records every request (test double, not Meta output)."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        if not self.script:
            raise AssertionError("more requests than scripted")
        status, body = self.script.pop(0)
        return httpx2.Response(status, json=body, headers={"x-request-id": "req-fake-p6b"})

    def text(self) -> str:
        return "\n".join(r.content.decode("utf-8") for r in self.requests)


def interpretation(frames=(0, 3)) -> dict:
    """HANDWRITTEN minimal interpretation (test data, not model output)."""
    return {
        "schema_version": si.SCHEMA_VERSION,
        "observed_facts": [{"fact_id": "F1", "object_ids": ["obj_01"], "statement": "obj_01 is observed in these frames.",
                            "evidence": [{"object_id": "obj_01", "frame_index": f} for f in frames]}],
        "objects": [{"object_id": "obj_01", "motion_class": "insufficient_data", "motion_description": "Test data.",
                     "supporting_fact_ids": ["F1"]}],
        "physical_model_hypotheses": [], "event_hypotheses": [], "interaction_hypotheses": [],
        "ambiguities": [{"ambiguity_id": "A1", "about_ids": ["F1"], "description": "Test data.",
                         "alternatives": ["Any smooth motion."], "what_would_resolve": "More frames."}],
        "data_quality_notes": [],
    }


@pytest.fixture(scope="module")
def split():
    return ho.split_trajectories(synth.generate(ALT).trajectories, Fraction(7, 10))


@pytest.fixture
def make_client(tmp_path):
    https = []

    def _make(script):
        server = FakeServer(script)
        http = httpx2.Client(transport=httpx2.MockTransport(server))
        https.append(http)
        client = m.MuseSparkClient(evidence_dir=tmp_path / "api_evidence", api_key=FAKE_KEY, http_client=http,
                                   sleep=lambda s: None, rng=random.Random(0))
        return client, server

    yield _make
    for h in https:
        h.close()


def withheld_values(split) -> list[str]:
    return [repr(o.x) for o in split.vault.observations] + [repr(o.y) for o in split.vault.observations]


def test_disabled_makes_no_request_and_writes_the_artifact(tmp_path, split):
    r = ss.run_spark_stage(split.inference.training, tmp_path / "spark")
    assert r["status"] == ss.DISABLED and r["artifact_path"].is_file()
    art = json.loads(r["artifact_path"].read_bytes())
    assert art["status"] == "disabled" and art["input"]["training_sha256"] == split.inference.training_sha256
    assert art["versions"]["prompt"] == spr.PROMPT_VERSION and "interpretation" not in art


def test_success_sends_only_the_training_trajectory(tmp_path, split, make_client):
    client, server = make_client([(200, completion(json.dumps(interpretation())))])
    r = ss.run_spark_stage(split.inference.training, tmp_path / "spark", client=client)
    assert r["status"] == ss.SUCCEEDED and len(server.requests) == 1
    body = json.loads(server.requests[0].content)
    user = body["messages"][1]["content"]
    payload = spr.canonical_json(spr.prompt_payload(split.inference.training))
    assert payload in user  # exactly the training payload is the data block
    sent = server.text()
    for v in withheld_values(split):
        assert v not in sent, "a withheld coordinate reached the model"
    for frame in {o.frame_index for o in split.vault.observations}:
        assert f'"frame_index":{frame},' not in user
    art = r["artifact"]
    assert art["physics_review"]["assessment"] in ("no_contradictions_found", "concerns_found", "contradictions_found")
    assert art["kinematics_summary"][0]["object_id"] == "obj_01" and art["evidence"]["exists"] and art["sidecar"]["exists"]
    assert art["model"]["response_provenance"] == "injected"
    text = r["artifact_path"].read_text(encoding="utf-8")
    for v in withheld_values(split):
        assert v not in text


def test_server_error_is_unavailable_after_exactly_one_attempt(tmp_path, split, make_client):
    assert m.MAX_ATTEMPTS > 1  # the module default would retry a 503
    client, server = make_client([(503, {"error": {"message": "busy", "type": "server_error", "param": None,
                                                   "code": "x"}})] * 3)
    r = ss.run_spark_stage(split.inference.training, tmp_path / "spark", client=client)
    assert r["status"] == ss.UNAVAILABLE and len(server.requests) == 1
    assert r["artifact"]["error"]["type"] == "ServerError" and m.MAX_ATTEMPTS > 1  # restored afterwards
    assert FAKE_KEY not in r["artifact_path"].read_text(encoding="utf-8")


def test_citing_a_withheld_frame_is_rejected(tmp_path, split, make_client):
    withheld_frame = split.vault.observations[0].frame_index
    client, _ = make_client([(200, completion(json.dumps(interpretation(frames=(0, withheld_frame)))))])
    r = ss.run_spark_stage(split.inference.training, tmp_path / "spark", client=client)
    assert r["status"] == ss.REJECTED and r["artifact"]["validation_errors"]


def test_missing_key_and_unexpected_errors_are_unavailable(tmp_path, split, monkeypatch):
    monkeypatch.delenv(m.API_KEY_ENV, raising=False)
    r = ss.run_spark_stage(split.inference.training, tmp_path / "a",
                           client_factory=lambda: m.MuseSparkClient(evidence_dir=tmp_path / "ev"))
    assert r["status"] == ss.UNAVAILABLE and r["artifact"]["error"]["type"] == "MissingAPIKeyError"

    def boom():
        raise RuntimeError("secret-ish detail")
    r = ss.run_spark_stage(split.inference.training, tmp_path / "b", client_factory=boom)
    assert r["status"] == ss.UNAVAILABLE and "secret-ish" not in r["artifact_path"].read_text(encoding="utf-8")


def test_artifact_is_never_overwritten(tmp_path, split):
    ss.run_spark_stage(split.inference.training, tmp_path / "spark")
    with pytest.raises(FileExistsError):
        ss.run_spark_stage(split.inference.training, tmp_path / "spark")


def test_spark_stage_cannot_reach_withheld_data_or_the_prediction():
    tree = ast.parse((ROOT / "decompile" / "spark_stage.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                for a in n.names} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    joined = " ".join(sorted(imported))
    for forbidden in ("holdout", "evaluate", "synth", "render", "predict", "scene_program", "overlay"):
        assert forbidden not in joined, forbidden
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("schedule", "vault", "ground_truth", "WithheldVault", "split_trajectories"):
        assert forbidden not in names, forbidden
