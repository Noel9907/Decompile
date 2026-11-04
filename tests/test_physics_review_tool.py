"""OFFLINE tests for tools/physics_review.py (B1c-3).

No network: an autouse fixture makes every socket connection and every MuseSparkClient
construction fail. Test chains (sidecar + B1a evidence) are produced by the real
interpret_scene() through an INJECTED fake HTTP transport that replays the committed
B1b-3 response content (tests/fixtures/live_muse_spark_b1b3_response.json); their evidence
is therefore labelled transport "injected", never presented as a new Meta result. Reports
are written to pytest temporary directories only.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import random
import socket
from pathlib import Path

import httpx2
import pytest

from decompile import meta_api as m
from decompile import scene_interpreter as sint
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "physics_review.py"
FIXTURES = ROOT / "tests" / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"
LIVE = FIXTURES / "live_muse_spark_b1b3_response.json"
REAL_SIDECAR = ROOT / "runs/scene_interpretation/20261104T035731330917Z_scene-interpretation_a984acc386ff.json"
REAL_EVIDENCE = ROOT / "runs/api_evidence/20261104T035731331122Z_b1b-scene-interpretation_8d6c991cecf5.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("physics_review_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


@pytest.fixture
def no_client(monkeypatch):
    """Applied while the TOOL runs: constructing an API client is a test failure."""
    def refuse(*args, **kwargs):
        raise AssertionError("the offline tool must not create an API client")

    monkeypatch.setattr(m, "MuseSparkClient", refuse)


# ------------------------------------------------------------------ building real offline chains


def make_chain(tmp_path: Path, name: str = "chain") -> tuple[Path, Path]:
    """Real interpret_scene() + injected transport replaying the committed live response content."""
    content = json.loads(LIVE.read_text(encoding="utf-8"))["content"]
    body = {"id": "chatcmpl-replayed-offline", "object": "chat.completion", "created": 0, "model": "muse-spark-1.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    http = httpx2.Client(transport=httpx2.MockTransport(lambda req: httpx2.Response(200, json=body)))
    try:
        client = m.MuseSparkClient(evidence_dir=tmp_path / name / "api_evidence", api_key="sk-OFFLINE-FAKE-b1c3",
                                   http_client=http, sleep=lambda s: None, rng=random.Random(0))
        result = sint.interpret_scene(tr.load_trajectories(BOUNCE), client,
                                      sidecar_dir=tmp_path / name / "scene_interpretation")
    finally:
        http.close()
    return result.sidecar_path, result.evidence_path


@pytest.fixture
def chain(tmp_path):
    return make_chain(tmp_path)


def rewrite_json(path: Path, edit) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_content(sidecar: Path, evidence: Path, content: str) -> None:
    """Replace the response content and re-hash every link, so only the content itself is wrong."""
    rewrite_json(evidence, lambda d: d["attempts"][-1]["response_json"]["choices"][0]["message"].update(content=content))
    rewrite_json(sidecar, lambda d: d.update(api_evidence_sha256=sha(evidence),
                                             response_content_sha256=hashlib.sha256(content.encode()).hexdigest()))


def run(argv: list[str], capsys) -> tuple[int, str]:
    code = tool.main([str(a) for a in argv])
    return code, capsys.readouterr().out


def args(sidecar, out, trajectory=BOUNCE, evidence=None):
    a = ["--sidecar", sidecar, "--trajectory", trajectory, "--output-dir", out]
    if evidence is not None:
        a += ["--evidence", evidence]
    return a


def reports(out: Path) -> list[Path]:
    return sorted(out.glob("*")) if out.exists() else []


def assert_refused(code, text, out: Path, check: str):
    assert code == tool.EXIT_VERIFICATION, text
    assert f"[FAIL] {check}:" in text, text
    assert "nothing was written" in text
    assert reports(out) == []  # no report, no temporary file


# ------------------------------------------------------------------ success


def test_valid_chain_writes_report(chain, tmp_path, capsys, no_client):
    sidecar, evidence = chain
    out = tmp_path / "out"
    code, text = run(args(sidecar, out), capsys)
    assert code == tool.EXIT_OK, text
    (path,) = reports(out)
    assert path.suffix == ".json" and "_physics-review_" in path.name
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["report_format_version"] == 1 and report["kind"] == "physics_review_report"
    assert report["offline"] is True and report["api_requests_made"] == 0
    assert report["summary"]["counts"] == {"error": 0, "warning": 4, "info": 11}
    assert report["summary"]["assessment"] == "concerns_found"
    assert all(c["passed"] for c in report["integrity_checks"])
    assert [c["check"] for c in report["integrity_checks"]] == [
        "sidecar", "evidence_resolve", "evidence_sha256", "evidence", "request_binding", "response_sha256",
        "response_json", "input_sha256", "schema_sha256", "interpretation"]
    assert report["response"]["response_provenance"] == "injected"  # an offline replay, labelled as such
    assert report["sources"]["evidence"]["sha256"] == sha(evidence)
    assert report["sources"]["sidecar"]["sha256"] == sha(sidecar)
    assert report["sources"]["evidence"]["path_matches_recorded"] is True
    assert f"report: {path}" in text and "4 warning(s)" in text


def test_report_contents(chain, tmp_path, capsys):
    sidecar, _ = chain
    out = tmp_path / "out"
    assert run(args(sidecar, out), capsys)[0] == 0
    report = json.loads(reports(out)[0].read_text(encoding="utf-8"))
    kin = report["kinematics"]
    assert kin["method_version"] == "decompile.kinematic_fit/1"
    assert kin["noise"]["mode"] == "estimated"
    obj = kin["objects"][0]
    assert obj["t_start_s"] == {"num": 0, "den": 1}  # exact rational
    assert obj["finite_differences"][0]["dt_s"] == {"num": 1, "den": 30}
    assert obj["contact"]["gap_t"] == [{"num": 3, "den": 5}, {"num": 19, "den": 30}]
    assert set(obj["fits"]) >= {"constant_velocity", "constant_acceleration", "impulse_uniform_acceleration",
                                "linear_drag", "linear_drag_impulse"}
    fit = obj["fits"]["impulse_uniform_acceleration"]
    assert {"bic", "rss_px2", "params", "param_se", "residuals_y", "n_params"} <= set(fit)
    assert "covariance" not in fit
    assert obj["turning_points"][0]["kind"] == "y_minimum"
    rev = report["physics_review"]
    assert rev["method_version"] == "decompile.physics_review/1"
    assert len(rev["findings"]) == 15 and rev["limitations"]
    assert report["sources"]["trajectory"]["file_sha256"] == sha(BOUNCE)
    text = reports(out)[0].read_text(encoding="utf-8").lower()
    for secret in ("sk-offline-fake", "bearer", "authorization", "api_key"):
        assert secret not in text


def test_report_is_deterministic_apart_from_timestamp(chain, tmp_path, capsys):
    sidecar, _ = chain
    out = tmp_path / "out"
    assert run(args(sidecar, out), capsys)[0] == 0
    assert run(args(sidecar, out), capsys)[0] == 0
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in reports(out))
    assert "generated_utc" in a and "generated_utc" in b
    a.pop("generated_utc"), b.pop("generated_utc")
    assert a == b


def test_explicit_relocated_evidence_is_accepted_by_hash(chain, tmp_path, capsys):
    sidecar, evidence = chain
    moved = tmp_path / "elsewhere" / evidence.name
    moved.parent.mkdir()
    moved.write_bytes(evidence.read_bytes())
    out = tmp_path / "out"
    code, text = run(args(sidecar, out, evidence=moved), capsys)
    assert code == 0, text
    report = json.loads(reports(out)[0].read_text(encoding="utf-8"))
    assert report["sources"]["evidence"]["path_matches_recorded"] is False
    assert report["sources"]["evidence"]["path"] == str(moved)


# ------------------------------------------------------------------ provenance and integrity failures


def test_evidence_file_hash_mismatch(chain, tmp_path, capsys):
    sidecar, evidence = chain
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "evidence_sha256")


def test_response_content_hash_mismatch(chain, tmp_path, capsys):
    sidecar, evidence = chain
    rewrite_json(evidence, lambda d: d["attempts"][-1]["response_json"]["choices"][0]["message"].update(
        content=d["attempts"][-1]["response_json"]["choices"][0]["message"]["content"].replace("F1", "F1", 1) + " "))
    rewrite_json(sidecar, lambda d: d.update(api_evidence_sha256=sha(evidence)))  # file hash fixed up, content not
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "response_sha256")


def test_sidecar_with_another_chains_evidence(tmp_path, capsys):
    sidecar1, _ = make_chain(tmp_path, "one")
    _, evidence2 = make_chain(tmp_path, "two")
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar1, out, evidence=evidence2), capsys), out, "evidence_sha256")
    rewrite_json(sidecar1, lambda d: d.update(api_evidence_path=str(evidence2)))  # sidecar points elsewhere
    assert_refused(*run(args(sidecar1, out), capsys), out, "evidence_sha256")


def test_trajectory_input_hash_mismatch(chain, tmp_path, capsys):
    sidecar, _ = chain
    data = json.loads(BOUNCE.read_text(encoding="utf-8"))
    data["objects"][0]["observations"][5]["x"] += 0.5
    other = tmp_path / "other.trajectories.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out, trajectory=other), capsys), out, "input_sha256")


def test_trajectory_provenance_mismatch(chain, tmp_path, capsys):
    sidecar, _ = chain
    data = json.loads(BOUNCE.read_text(encoding="utf-8"))
    data.update(synthetic=False, source="CLASSICAL_TRACKER")  # same model input, different provenance
    other = tmp_path / "relabelled.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    code, text = run(args(sidecar, out, trajectory=other), capsys)
    assert_refused(code, text, out, "input_sha256")
    assert "provenance" in text


def test_request_binding_mismatch(chain, tmp_path, capsys):
    sidecar, evidence = chain
    rewrite_json(evidence, lambda d: d["request"]["messages"][0].update(content="tampered"))
    rewrite_json(sidecar, lambda d: d.update(api_evidence_sha256=sha(evidence)))
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "request_binding")


def test_response_id_mismatch(chain, tmp_path, capsys):
    sidecar, _ = chain
    rewrite_json(sidecar, lambda d: d["response"].update(response_id="chatcmpl-other"))
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "evidence")


def test_transport_label_mismatch(chain, tmp_path, capsys):
    sidecar, _ = chain
    rewrite_json(sidecar, lambda d: d.update(response_provenance="network"))  # cannot relabel a replay as real
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "evidence")


def test_non_accepted_sidecar_refused(chain, tmp_path, capsys):
    sidecar, _ = chain
    rewrite_json(sidecar, lambda d: d.update(status="rejected_semantic_validation"))
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "sidecar")


# ------------------------------------------------------------------ malformed inputs


def test_malformed_sidecar_json(tmp_path, capsys):
    bad = tmp_path / "sidecar.json"
    bad.write_text("{not json", encoding="utf-8")
    out = tmp_path / "out"
    assert_refused(*run(args(bad, out), capsys), out, "sidecar")


def test_sidecar_missing_keys(tmp_path, capsys):
    bad = tmp_path / "sidecar.json"
    bad.write_text(json.dumps({"kind": "scene_interpretation_validation"}), encoding="utf-8")
    out = tmp_path / "out"
    assert_refused(*run(args(bad, out), capsys), out, "sidecar")


def test_malformed_evidence_json(chain, tmp_path, capsys):
    sidecar, evidence = chain
    evidence.write_text("{broken", encoding="utf-8")
    rewrite_json(sidecar, lambda d: d.update(api_evidence_sha256=sha(evidence)))
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "evidence")


def test_malformed_model_response_json(chain, tmp_path, capsys):
    sidecar, evidence = chain
    set_content(sidecar, evidence, '{"schema_version": NaN}')
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out), capsys), out, "response_json")


def test_schema_validation_failure(chain, tmp_path, capsys):
    sidecar, evidence = chain
    set_content(sidecar, evidence, json.dumps({"schema_version": "decompile.scene_interpretation/1"}))
    out = tmp_path / "out"
    code, text = run(args(sidecar, out), capsys)
    assert_refused(code, text, out, "interpretation")
    assert "schema:" in text


def test_semantic_validation_failure(chain, tmp_path, capsys):
    sidecar, evidence = chain
    data = json.loads(json.loads(LIVE.read_text(encoding="utf-8"))["content"])
    data["physical_model_hypotheses"][0]["rationale"] += " Gravity is about 9.81 m/s^2."
    set_content(sidecar, evidence, json.dumps(data))
    out = tmp_path / "out"
    code, text = run(args(sidecar, out), capsys)
    assert_refused(code, text, out, "interpretation")
    assert "numerical physical-parameter estimate" in text


@pytest.mark.parametrize("which", ["sidecar", "evidence", "trajectory"])
def test_missing_files(chain, tmp_path, capsys, which):
    sidecar, evidence = chain
    out = tmp_path / "out"
    missing = tmp_path / "does_not_exist.json"
    if which == "sidecar":
        a, check = args(missing, out), "sidecar"
    elif which == "evidence":
        evidence.unlink()
        a, check = args(sidecar, out), "evidence_resolve"
    else:
        a, check = args(sidecar, out, trajectory=missing), "trajectory"
    assert_refused(*run(a, capsys), out, check)


def test_invalid_trajectory(chain, tmp_path, capsys):
    sidecar, _ = chain
    bad = tmp_path / "bad.trajectories.json"
    bad.write_text(json.dumps({"schema": "decompile.trajectories/1", "objects": []}), encoding="utf-8")
    out = tmp_path / "out"
    assert_refused(*run(args(sidecar, out, trajectory=bad), capsys), out, "trajectory")


# ------------------------------------------------------------------ writing


def test_write_failure_leaves_no_partial_report(chain, tmp_path, capsys, monkeypatch):
    sidecar, _ = chain
    out = tmp_path / "out"

    def failing_replace(src, dst):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(m.os, "replace", failing_replace)  # inside the real atomic writer
    code, text = run(args(sidecar, out), capsys)
    assert code == tool.EXIT_WRITE, text
    assert "no partial report" in text
    assert reports(out) == []  # the temporary file was removed too


def test_interrupt_during_write_leaves_no_partial_report(chain, tmp_path, monkeypatch):
    sidecar, _ = chain
    out = tmp_path / "out"

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(m.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tool.main([str(a) for a in args(sidecar, out)])
    assert reports(out) == []


def test_existing_report_is_never_overwritten(chain, tmp_path, capsys, monkeypatch):
    sidecar, _ = chain
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "existing.json"
    existing.write_text('{"keep": true}', encoding="utf-8")
    monkeypatch.setattr(tool, "new_report_path", lambda output_dir, now=None: existing)
    code, text = run(args(sidecar, out), capsys)
    assert code == tool.EXIT_WRITE and "FileExistsError" in text
    assert existing.read_text(encoding="utf-8") == '{"keep": true}'
    assert reports(out) == [existing]


# ------------------------------------------------------------------ CLI


def test_help(capsys):
    with pytest.raises(SystemExit) as info:
        tool.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for word in ("--sidecar", "--trajectory", "--evidence", "--output-dir", "OFFLINE", "Exit codes"):
        assert word in text
    assert "--live" not in text


def test_usage_errors_exit_2(capsys):
    with pytest.raises(SystemExit) as info:
        tool.main([])
    assert info.value.code == tool.EXIT_USAGE
    with pytest.raises(SystemExit) as info:
        tool.main(["--sidecar", "a", "--trajectory", "b", "--position-sigma-px", "x"])
    assert info.value.code == tool.EXIT_USAGE


def test_supplied_sigma_is_recorded(chain, tmp_path, capsys):
    sidecar, _ = chain
    out = tmp_path / "out"
    assert run(args(sidecar, out) + ["--position-sigma-px", "0.0029,0.0029"], capsys)[0] == 0
    noise = json.loads(reports(out)[0].read_text(encoding="utf-8"))["kinematics"]["noise"]
    assert noise["mode"] == "supplied" and noise["sigma_px"] == [0.0029, 0.0029]


def test_tool_never_creates_a_client_or_imports_network_libraries():
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert not imported & {"openai", "httpx", "httpx2", "socket", "requests", "urllib", "dotenv"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"MuseSparkClient", "chat", "interpret_scene", "load_dotenv"}
    assert "--live" not in TOOL.read_text(encoding="utf-8")


# ------------------------------------------------------------------ the real B1b-3 artifacts (if present locally)


@pytest.mark.skipif(not (REAL_SIDECAR.is_file() and REAL_EVIDENCE.is_file()),
                    reason="the local B1b-3 run artifacts (git-ignored runs/) are not present")
def test_real_b1b3_artifacts_replay(tmp_path, capsys, no_client):
    out = tmp_path / "out"
    before = (sha(REAL_SIDECAR), sha(REAL_EVIDENCE))
    code, text = run(args(REAL_SIDECAR, out, evidence=REAL_EVIDENCE), capsys)
    assert code == 0, text
    report = json.loads(reports(out)[0].read_text(encoding="utf-8"))
    assert report["summary"]["counts"] == {"error": 0, "warning": 4, "info": 11}
    assert report["summary"]["assessment"] == "concerns_found"
    assert report["response"]["response_provenance"] == "network"
    assert report["response"]["response_content_sha256"] == json.loads(
        LIVE.read_text(encoding="utf-8"))["response_content_sha256"]
    assert (sha(REAL_SIDECAR), sha(REAL_EVIDENCE)) == before  # read-only
