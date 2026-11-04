"""Tests for decompile.scene_interpreter (B1b-2b).

OFFLINE ONLY. Every test injects a fake HTTP transport (httpx2.MockTransport) into the
real MuseSparkClient, or stubs client.chat directly. No network call is made and no
response here came from Meta or from any model.

Replayed response content is either the HANDWRITTEN interpretation fixture (written by
hand as validator test data, NOT model output) or a hand-made mutation of it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path

import httpx2
import pytest

from decompile import meta_api as m
from decompile import scene_interpretation as si
from decompile import scene_interpreter as sint
from decompile import scene_prompt as sp
from decompile import trajectories as tr

FIXTURES = Path(__file__).parent / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"
HANDWRITTEN = FIXTURES / "handwritten_interpretation_bounce_single.json"

FAKE_KEY = "sk-OFFLINE-FAKE-KEY-interpreter-0123456789"


# ------------------------------------------------------------------ fake transport


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-fake-b1b2",
        "object": "chat.completion",
        "created": 0,
        "model": "muse-spark-1.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 900, "total_tokens": 1900,
                  "completion_tokens_details": {"reasoning_tokens": 400}},
    }


def ok(content: str, finish_reason: str = "stop"):
    return 200, completion(content, finish_reason), {"x-request-id": "req-fake-b1b2"}


def http_error(status: int, message: str = "boom", code: str = "x"):
    return status, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}, {}


class FakeServer:
    """Replays scripted (status, body, headers) replies and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("fake server received more requests than scripted")
        status, body, headers = self.script.pop(0)
        return httpx2.Response(status, json=body, headers=headers)

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def make_client(tmp_path):
    https = []

    def _make(script):
        server = FakeServer(script)
        http = httpx2.Client(transport=httpx2.MockTransport(server))
        https.append(http)
        client = m.MuseSparkClient(
            evidence_dir=tmp_path / "api_evidence", api_key=FAKE_KEY, http_client=http,
            sleep=lambda s: None, rng=random.Random(0),
        )
        return client, server

    yield _make
    for http in https:
        http.close()


# ------------------------------------------------------------------ data


@pytest.fixture(scope="module")
def traj() -> tr.TrajectorySet:
    return tr.load_trajectories(BOUNCE)


@pytest.fixture
def golden() -> dict:
    """HANDWRITTEN test data (not model output), replayed by the fake transport."""
    return copy.deepcopy(json.loads(HANDWRITTEN.read_text(encoding="utf-8"))["interpretation"])


def ob(k: int, x: float, y: float) -> dict:
    t = Fraction(k, 30)
    return {"frame_index": k, "t": {"num": t.numerator, "den": t.denominator}, "x": x, "y": y}


def two_object_set() -> tr.TrajectorySet:
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [
            {"object_id": "obj_01", "observations": [ob(k, 100 + 10 * k, 50) for k in range(4)]},
            {"object_id": "obj_02", "observations": [ob(k, 200, 50) for k in range(4)]},
        ],
    })


def minimal_interpretation(object_ids=("obj_01", "obj_02")) -> dict:
    """HANDWRITTEN minimal interpretation for the two-object set (test data)."""
    return {
        "schema_version": si.SCHEMA_VERSION,
        "observed_facts": [
            {"fact_id": f"F{i + 1}", "object_ids": [oid], "statement": f"{oid} is observed in frames 0 to 3.",
             "evidence": [{"object_id": oid, "frame_index": 0}, {"object_id": oid, "frame_index": 3}]}
            for i, oid in enumerate(object_ids)
        ],
        "objects": [
            {"object_id": oid, "motion_class": "insufficient_data", "motion_description": "Few observations.",
             "supporting_fact_ids": [f"F{i + 1}"]}
            for i, oid in enumerate(object_ids)
        ],
        "physical_model_hypotheses": [],
        "event_hypotheses": [],
        "interaction_hypotheses": [],
        "ambiguities": [
            {"ambiguity_id": "A1", "about_ids": ["F1"], "description": "Too few frames.",
             "alternatives": ["Any smooth motion."], "what_would_resolve": "More frames."}
        ],
        "data_quality_notes": [],
    }


def run(client, traj, tmp_path, **kwargs):
    return sint.interpret_scene(traj, client, sidecar_dir=tmp_path / "scene_interpretation", **kwargs)


def sidecars(tmp_path) -> list[Path]:
    d = tmp_path / "scene_interpretation"
    return sorted(d.glob("*.json")) if d.exists() else []


def only_sidecar(tmp_path) -> dict:
    files = sidecars(tmp_path)
    assert len(files) == 1, files
    return json.loads(files[0].read_text(encoding="utf-8"))


def all_files_text(tmp_path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file())


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ success


def test_successful_interpretation(make_client, traj, golden, tmp_path):
    client, server = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    assert isinstance(result, sint.SceneInterpretationResult)
    assert isinstance(result.interpretation, si.SceneInterpretation)
    assert len(server.requests) == 1  # chat() exactly once
    assert [o.object_id for o in result.interpretation.objects] == ["obj_01"]
    assert result.prompt_version == sp.PROMPT_VERSION
    assert result.requested_model == m.DEFAULT_MODEL
    assert result.model == "muse-spark-1.3"
    assert result.response_id == "chatcmpl-fake-b1b2"
    assert result.request_id == "req-fake-b1b2"
    assert result.attempts == 1
    assert result.usage.total_tokens == 1900 and result.usage.reasoning_tokens == 400
    assert result.evidence_path.is_file() and result.sidecar_path.is_file()
    assert result.sidecar_path.parent == tmp_path / "scene_interpretation"
    assert result.warnings == ()


def test_request_uses_developer_and_user_messages(make_client, traj, golden, tmp_path):
    client, server = make_client([ok(json.dumps(golden))])
    run(client, traj, tmp_path)
    (body,) = server.bodies()
    expected = sp.build_scene_prompt(traj)
    assert body["messages"] == expected.messages
    assert [msg["role"] for msg in body["messages"]] == ["developer", "user"]
    user = body["messages"][1]["content"]
    assert sp.TRAJECTORY_OPEN in user and sp.TRAJECTORY_CLOSE in user
    assert "synthetic" not in json.dumps(body["messages"]).lower()


def test_request_settings_and_strict_schema(make_client, traj, golden, tmp_path):
    client, server = make_client([ok(json.dumps(golden))])
    run(client, traj, tmp_path)
    (body,) = server.bodies()
    assert body["max_completion_tokens"] == 8192
    assert body["reasoning_effort"] == "low"
    assert body["model"] == m.DEFAULT_MODEL
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == si.SCHEMA_NAME
    assert rf["json_schema"]["schema"] == si.build_interpretation_schema(traj)


def test_token_settings_are_configurable(make_client, traj, golden, tmp_path):
    client, server = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path, max_completion_tokens=4096, reasoning_effort="medium")
    (body,) = server.bodies()
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "medium"
    side = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert side["call"]["max_completion_tokens"] == 4096 and side["call"]["reasoning_effort"] == "medium"


def test_defaults():
    assert sint.DEFAULT_MAX_COMPLETION_TOKENS == 8192
    assert sint.DEFAULT_REASONING_EFFORT == "low"


def test_schema_object_id_enum_is_dynamic(make_client, golden, tmp_path):
    two = two_object_set()
    client, server = make_client([ok(json.dumps(minimal_interpretation()))])
    result = run(client, two, tmp_path)
    (body,) = server.bodies()
    schema = body["response_format"]["json_schema"]["schema"]
    enum = schema["properties"]["objects"]["items"]["properties"]["object_id"]["enum"]
    assert enum == ["obj_01", "obj_02"]
    assert [o.object_id for o in result.interpretation.objects] == ["obj_01", "obj_02"]
    assert only_sidecar(tmp_path)["input_object_ids"] == ["obj_01", "obj_02"]


def test_hashes_match_exactly_what_was_sent(make_client, traj, golden, tmp_path):
    content = json.dumps(golden)
    client, server = make_client([ok(content)])
    result = run(client, traj, tmp_path)
    (body,) = server.bodies()
    user = body["messages"][1]["content"]
    start = user.index(sp.TRAJECTORY_OPEN) + len(sp.TRAJECTORY_OPEN) + 1
    end = user.index(sp.TRAJECTORY_CLOSE) - 1
    embedded = user[start:end]
    assert result.input_sha256 == hashlib.sha256(embedded.encode("utf-8")).hexdigest()
    assert result.messages_sha256 == sp.sha256_text(sp.canonical_json(body["messages"]))
    assert result.developer_prompt_sha256 == sp.sha256_text(body["messages"][0]["content"])
    assert result.schema_sha256 == sp.sha256_text(sp.canonical_json(body["response_format"]["json_schema"]["schema"]))
    assert result.response_content_sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
    side = only_sidecar(tmp_path)
    for key in ("input_sha256", "messages_sha256", "developer_prompt_sha256", "schema_sha256",
                "response_content_sha256"):
        assert side[key] == getattr(result, key), key
    assert side["api_evidence_sha256"] == sha256_file(result.evidence_path)


def test_response_content_hash_binds_to_api_evidence(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    raw_content = evidence["attempts"][-1]["response_json"]["choices"][0]["message"]["content"]
    assert sp.sha256_text(raw_content) == only_sidecar(tmp_path)["response_content_sha256"]


def test_event_windows_are_exact_fractions(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    (window,) = result.event_windows
    assert window == si.EventWindow("E1", 18, 20, Fraction(3, 5), Fraction(2, 3), (19,))
    assert isinstance(window.t_after, Fraction) and isinstance(window.t_before, Fraction)
    assert window.t_after == traj.frame_time(18) and window.t_before == traj.frame_time(20)
    (side_window,) = only_sidecar(tmp_path)["event_windows"]
    assert side_window == {
        "event_id": "E1", "after_frame_index": 18, "before_frame_index": 20,
        "t_after": {"num": 3, "den": 5}, "t_before": {"num": 2, "den": 3},
        "interior_frame_indices": [19],
    }
    assert Fraction(side_window["t_before"]["num"], side_window["t_before"]["den"]) - Fraction(
        side_window["t_after"]["num"], side_window["t_after"]["den"]) == Fraction(1, 15)


# ------------------------------------------------------------------ sidecar provenance


def test_sidecar_provenance_and_metadata(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    side = only_sidecar(tmp_path)
    assert side["sidecar_version"] == sint.SIDECAR_VERSION
    assert side["kind"] == sint.SIDECAR_KIND
    assert side["status"] == sint.STATUS_ACCEPTED
    assert side["prompt_version"] == sp.PROMPT_VERSION
    assert side["schema_version"] == si.SCHEMA_VERSION
    assert side["input_object_ids"] == ["obj_01"]
    assert side["input_provenance"] == {"synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
                                        "sent_to_model": False}
    assert side["api_evidence_path"] == str(result.evidence_path)
    assert side["call"] == {"purpose": sint.PURPOSE, "requested_model": m.DEFAULT_MODEL,
                            "max_completion_tokens": 8192, "reasoning_effort": "low"}
    resp = side["response"]
    assert resp["model"] == "muse-spark-1.3"
    assert resp["response_id"] == "chatcmpl-fake-b1b2" and resp["request_id"] == "req-fake-b1b2"
    assert resp["usage"]["total_tokens"] == 1900 and resp["attempts"] == 1
    assert isinstance(resp["latency_s"], float)
    assert side["validation"] == {"ok": True, "errors": [], "warnings": []}
    assert side["api_error"] is None
    assert side["physical_correctness_note"] == sint.PHYSICAL_CORRECTNESS_NOTE
    assert "does not establish" in side["physical_correctness_note"]


def test_sidecar_does_not_duplicate_the_interpretation(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    run(client, traj, tmp_path)
    text = sidecars(tmp_path)[0].read_text(encoding="utf-8")
    side = json.loads(text)
    for key in ("observed_facts", "physical_model_hypotheses", "event_hypotheses", "ambiguities"):
        assert key not in side
    assert golden["observed_facts"][0]["statement"] not in text
    assert golden["physical_model_hypotheses"][0]["rationale"] not in text


def test_injected_transport_is_labelled_injected_not_meta(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    side = only_sidecar(tmp_path)
    assert result.response_provenance == "injected"
    assert side["response_provenance"] == "injected"
    assert "NOT a Meta API result" in side["response_provenance_note"]
    evidence = json.loads(result.evidence_path.read_text(encoding="utf-8"))
    assert evidence["transport"] == "injected"


@pytest.mark.parametrize("transport, expected", [("network", "network"), ("injected", "injected"),
                                                 ("something_else", "unknown")])
def test_evidence_info_reads_transport(tmp_path, transport, expected):
    path = tmp_path / "ev.json"
    path.write_text(json.dumps({"transport": transport}), encoding="utf-8")
    digest, provenance = sint._evidence_info(path)
    assert provenance == expected
    assert digest == sha256_file(path)


def test_evidence_info_tolerates_missing_or_garbage(tmp_path):
    assert sint._evidence_info(None) == (None, "unknown")
    assert sint._evidence_info(tmp_path / "missing.json") == (None, "unknown")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{not json")
    assert sint._evidence_info(bad) == (sha256_file(bad), "unknown")


# ------------------------------------------------------------------ semantic rejection


def test_semantic_validation_failure(make_client, traj, golden, tmp_path):
    golden["physical_model_hypotheses"][0]["rationale"] += " Gravity is about 9.81 m/s^2."
    client, server = make_client([ok(json.dumps(golden))])
    with pytest.raises(sint.InterpretationRejectedError) as info:
        run(client, traj, tmp_path)
    err = info.value
    assert isinstance(err, si.SceneInterpretationError)  # existing except-clauses still catch it
    assert any("numerical physical-parameter estimate" in e for e in err.errors)
    assert len(server.requests) == 1  # no repair or retry at the interpretation layer
    assert err.evidence_path.is_file() and err.sidecar_path.is_file()
    side = only_sidecar(tmp_path)
    assert side["status"] == sint.STATUS_REJECTED
    assert side["validation"]["ok"] is False
    assert side["validation"]["errors"] == list(err.errors)
    assert side["event_windows"] == []
    assert side["response_content_sha256"] is not None
    assert side["api_evidence_path"] == str(err.evidence_path)


def test_rejection_for_missing_object_coverage(make_client, tmp_path):
    two = two_object_set()
    data = minimal_interpretation()
    data["objects"] = data["objects"][:1]
    client, _ = make_client([ok(json.dumps(data))])
    with pytest.raises(sint.InterpretationRejectedError) as info:
        run(client, two, tmp_path)
    assert any("no interpretation for input objects ['obj_02']" in e for e in info.value.errors)
    assert only_sidecar(tmp_path)["status"] == sint.STATUS_REJECTED


def test_warnings_propagate_to_result_and_sidecar(make_client, traj, golden, tmp_path):
    golden["ambiguities"] = []  # nothing references ambiguity ids; gives a warning, not an error
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    assert any("ambiguities: none recorded" in w for w in result.warnings)
    side = only_sidecar(tmp_path)
    assert side["status"] == sint.STATUS_ACCEPTED
    assert side["validation"]["warnings"] == list(result.warnings)


def test_trajectory_warnings_recorded_in_sidecar(make_client, tmp_path):
    s = tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": False, "source": "CLASSICAL_TRACKER",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 64, "image_height": 48},
        "objects": [{"object_id": "obj_01", "observations": [ob(k, 100.0, 10.0) for k in range(3)]}],
    })
    data = minimal_interpretation(("obj_01",))
    data["observed_facts"][0]["evidence"] = [{"object_id": "obj_01", "frame_index": 0},
                                             {"object_id": "obj_01", "frame_index": 2}]
    data["observed_facts"][0]["statement"] = "obj_01 is observed in frames 0 to 2."
    client, _ = make_client([ok(json.dumps(data))])
    run(client, s, tmp_path)
    side = only_sidecar(tmp_path)
    assert side["trajectory_warnings"] == list(s.warnings) and s.warnings
    assert side["input_provenance"] == {"synthetic": False, "source": "CLASSICAL_TRACKER", "sent_to_model": False}


# ------------------------------------------------------------------ API failures


def assert_api_failure(tmp_path, info, api_error_type):
    err = info.value
    assert isinstance(err.api_error, api_error_type)
    assert err.__cause__ is err.api_error
    assert err.evidence_path is not None and err.evidence_path.is_file()
    assert err.sidecar_path.is_file()
    side = only_sidecar(tmp_path)
    assert side["status"] == sint.STATUS_API_FAILED
    assert side["api_error"]["error_class"] == api_error_type.__name__
    assert side["api_evidence_path"] == str(err.evidence_path)
    assert side["api_evidence_sha256"] == sha256_file(err.evidence_path)
    assert side["response_provenance"] == "injected"
    assert side["validation"] is None and side["event_windows"] == []
    assert side["response"] is None and side["response_content_sha256"] is None
    return side


def test_schema_failure(make_client, traj, tmp_path):
    client, server = make_client([ok(json.dumps({"schema_version": si.SCHEMA_VERSION}))])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    assert_api_failure(tmp_path, info, m.StructuredOutputError)
    assert len(server.requests) == 1


def test_non_json_content_fails(make_client, traj, tmp_path):
    client, _ = make_client([ok("I think it is a bouncing ball.")])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    assert_api_failure(tmp_path, info, m.StructuredOutputError)


def test_unknown_object_id_is_a_schema_failure(make_client, traj, golden, tmp_path):
    golden["objects"][0]["object_id"] = "obj_99"
    client, _ = make_client([ok(json.dumps(golden))])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    assert_api_failure(tmp_path, info, m.StructuredOutputError)


def test_truncated_response_fails_loudly(make_client, traj, golden, tmp_path):
    partial = json.dumps(golden)[:500]
    client, _ = make_client([ok(partial, finish_reason="length")])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    side = assert_api_failure(tmp_path, info, m.IncompleteResponseError)
    assert side["api_error"]["finish_reason"] == "length"


@pytest.mark.parametrize("status, exc_type", [(400, m.InvalidRequestError), (504, m.GatewayTimeoutError),
                                              (401, m.AuthenticationError), (403, m.PermissionDeniedError)])
def test_http_errors(make_client, traj, tmp_path, status, exc_type):
    client, server = make_client([http_error(status, code=f"code_{status}")])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    side = assert_api_failure(tmp_path, info, exc_type)
    assert len(server.requests) == 1  # never retried
    assert side["api_error"]["status"] == status
    assert side["api_error"]["error_code"] == f"code_{status}"


def test_rate_limit_retries_belong_to_b1a_only(make_client, traj, tmp_path):
    client, server = make_client([http_error(429)] * m.MAX_ATTEMPTS)
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    assert_api_failure(tmp_path, info, m.RateLimitError)
    assert len(server.requests) == m.MAX_ATTEMPTS  # B1a policy; no extra retries added here


# ------------------------------------------------------------------ credential redaction


def test_credentials_redacted_everywhere(make_client, traj, tmp_path):
    client, _ = make_client([http_error(400, message=f"bad key {FAKE_KEY}")])
    with pytest.raises(sint.InterpretationAPIError) as info:
        run(client, traj, tmp_path)
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in str(info.value.api_error)
    assert FAKE_KEY not in all_files_text(tmp_path)
    assert m.REDACTED in only_sidecar(tmp_path)["api_error"]["message"]


def test_credentials_absent_on_success(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden))])
    result = run(client, traj, tmp_path)
    assert FAKE_KEY not in all_files_text(tmp_path)
    assert FAKE_KEY not in repr(result)


# ------------------------------------------------------------------ sidecar write failures


def failing_write(path, data):
    raise OSError("disk full (simulated)")


def test_sidecar_write_failure_after_success_keeps_result(make_client, traj, golden, tmp_path, monkeypatch):
    monkeypatch.setattr(sint, "write_json_atomic", failing_write)
    client, _ = make_client([ok(json.dumps(golden))])
    with pytest.raises(sint.SidecarWriteError) as info:
        run(client, traj, tmp_path)
    err = info.value
    assert err.status == sint.STATUS_ACCEPTED
    assert isinstance(err.result, sint.SceneInterpretationResult)
    assert err.result.event_windows[0].t_after == Fraction(3, 5)
    assert err.api_error is None
    assert isinstance(err.__cause__, OSError)
    assert err.evidence_path is not None and err.evidence_path.is_file()  # B1a evidence still written
    assert sidecars(tmp_path) == []


def test_sidecar_write_failure_after_api_failure_keeps_api_error(make_client, traj, tmp_path, monkeypatch):
    monkeypatch.setattr(sint, "write_json_atomic", failing_write)
    client, _ = make_client([http_error(504)])
    with pytest.raises(sint.SidecarWriteError) as info:
        run(client, traj, tmp_path)
    err = info.value
    assert err.status == sint.STATUS_API_FAILED
    assert isinstance(err.api_error, m.GatewayTimeoutError)
    assert err.result is None
    assert "GatewayTimeoutError" in str(err)
    assert isinstance(err.__cause__, OSError)
    assert err.evidence_path.is_file()


def test_sidecar_write_failure_after_rejection_keeps_errors(make_client, traj, golden, tmp_path, monkeypatch):
    monkeypatch.setattr(sint, "write_json_atomic", failing_write)
    golden["event_hypotheses"][0]["after_frame_index"] = 10
    client, _ = make_client([ok(json.dumps(golden))])
    with pytest.raises(sint.SidecarWriteError) as info:
        run(client, traj, tmp_path)
    err = info.value
    assert err.status == sint.STATUS_REJECTED
    assert any("too wide" in e for e in err.errors)
    assert err.result is None and err.api_error is None


def test_sidecar_never_overwrites_an_existing_file(make_client, traj, golden, tmp_path, monkeypatch):
    target = tmp_path / "scene_interpretation" / "existing.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"keep": true}', encoding="utf-8")
    monkeypatch.setattr(sint, "_new_sidecar_path", lambda d, now: target)
    client, _ = make_client([ok(json.dumps(golden))])
    with pytest.raises(sint.SidecarWriteError) as info:
        run(client, traj, tmp_path)
    assert isinstance(info.value.__cause__, FileExistsError)
    assert info.value.result is not None
    assert target.read_text(encoding="utf-8") == '{"keep": true}'


def test_one_sidecar_per_call_and_no_temp_files(make_client, traj, golden, tmp_path):
    client, _ = make_client([ok(json.dumps(golden)), ok(json.dumps(golden))])
    a = run(client, traj, tmp_path)
    b = run(client, traj, tmp_path)
    assert a.sidecar_path != b.sidecar_path
    assert len(sidecars(tmp_path)) == 2
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix == ".tmp"]
    assert leftovers == []
    assert sorted(p.name for p in (tmp_path / "scene_interpretation").iterdir()) == sorted(
        p.name for p in sidecars(tmp_path))


# ------------------------------------------------------------------ interrupts and pre-call errors


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_interrupts_propagate_unchanged(make_client, traj, tmp_path, monkeypatch, exc_type):
    client, server = make_client([])
    original = exc_type()

    def interrupted_chat(*args, **kwargs):
        raise original

    monkeypatch.setattr(client, "chat", interrupted_chat)
    with pytest.raises(exc_type) as info:
        run(client, traj, tmp_path)
    assert info.value is original
    side = only_sidecar(tmp_path)
    assert side["status"] == sint.STATUS_INTERRUPTED
    assert side["api_error"]["error_class"] == exc_type.__name__


def test_interrupt_with_failing_sidecar_still_propagates(make_client, traj, tmp_path, monkeypatch):
    client, _ = make_client([])
    monkeypatch.setattr(client, "chat", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(sint, "write_json_atomic", failing_write)
    with pytest.raises(KeyboardInterrupt) as info:
        run(client, traj, tmp_path)
    assert any("sidecar not written" in note for note in getattr(info.value, "__notes__", []))


@pytest.mark.parametrize("kwargs", [{"max_completion_tokens": 0}, {"max_completion_tokens": True},
                                    {"max_completion_tokens": 10.5}, {"reasoning_effort": "none"}])
def test_invalid_settings_rejected_before_any_call(make_client, traj, tmp_path, kwargs):
    client, server = make_client([])
    with pytest.raises(ValueError):
        run(client, traj, tmp_path, **kwargs)
    assert server.requests == []
    assert sidecars(tmp_path) == []
    assert not (tmp_path / "api_evidence").exists()


def test_invalid_trajectory_set_rejected_before_any_call(make_client, traj, tmp_path):
    from dataclasses import replace
    client, server = make_client([])
    with pytest.raises(tr.TrajectoryError):
        run(client, replace(traj, objects=()), tmp_path)
    with pytest.raises(TypeError):
        run(client, traj.to_json_dict(), tmp_path)
    assert server.requests == [] and sidecars(tmp_path) == []


def test_sidecar_dir_is_required(make_client, traj):
    client, _ = make_client([])
    with pytest.raises(TypeError):
        sint.interpret_scene(traj, client)  # no default: tests can never write into runs/
