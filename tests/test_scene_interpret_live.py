"""OFFLINE tests for tools/scene_interpret_live.py (B1b-3 runner).

No network. These check the runner's safety rails only: a dry run sends nothing, the
single-attempt cap really yields one request, and a repeat live run is refused.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import httpx2
import pytest

from decompile import meta_api as m

RUNNER_PATH = Path(__file__).resolve().parent.parent / "tools" / "scene_interpret_live.py"


@pytest.fixture
def runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("scene_interpret_live", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.meta_api is m  # the runner patches the real B1a module object
    monkeypatch.setattr(m, "MAX_ATTEMPTS", m.MAX_ATTEMPTS)  # restored after each test
    return module


@pytest.fixture
def no_client(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a client must not be created in this test")

    monkeypatch.setattr(m, "MuseSparkClient", refuse)


def test_runner_configuration(runner):
    assert runner.MODEL == "muse-spark-1.3"
    assert runner.REASONING_EFFORT == "low"
    assert runner.MAX_COMPLETION_TOKENS == 8192
    assert runner.EXPECTED_ENDPOINT == "https://api.meta.ai/v1/chat/completions"
    assert runner.EXPECTED_ENDPOINT == f"{m.BASE_URL}/chat/completions"
    assert runner.FIXTURE.name == "synthetic_bounce_single.trajectories.json"
    assert runner.SIDECAR_DIR.parts[-2:] == ("runs", "scene_interpretation")
    assert runner.EVIDENCE_DIR.parts[-2:] == ("runs", "api_evidence")


def test_dry_run_sends_nothing(runner, no_client, capsys):
    assert runner.main([]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "no request sent" in out
    assert m.MAX_ATTEMPTS == 3


def test_live_run_refused_when_a_sidecar_exists(runner, no_client, tmp_path, monkeypatch, capsys):
    (tmp_path / "earlier.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "SIDECAR_DIR", tmp_path)
    assert runner.main(["--live"]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert m.MAX_ATTEMPTS == 3  # refused before anything was configured


@pytest.mark.parametrize("status", [429, 500, 503])
def test_single_attempt_cap_sends_exactly_one_request(runner, tmp_path, status):
    runner.force_single_attempt()
    assert m.MAX_ATTEMPTS == 1
    requests = []

    def server(request):
        requests.append(request)
        return httpx2.Response(status, json={"error": {"message": "busy", "type": "server_error"}})

    http = httpx2.Client(transport=httpx2.MockTransport(server))
    try:
        client = m.MuseSparkClient(evidence_dir=tmp_path, api_key="sk-OFFLINE-FAKE-runner", http_client=http,
                                   sleep=lambda s: None, rng=random.Random(0))
        with pytest.raises(m.MetaAPIError) as info:
            client.chat([{"role": "user", "content": "hi"}], purpose="unit", max_completion_tokens=8)
    finally:
        http.close()
    assert len(requests) == 1  # normally retryable, but NOT retried under the cap
    evidence = m.strict_json_loads(info.value.evidence_path.read_text(encoding="utf-8"))
    assert evidence["retry_policy"]["max_attempts"] == 1
    assert len(evidence["attempts"]) == 1
    assert evidence["attempts"][0]["retry"] is False
    assert runner.evidence_attempts(info.value.evidence_path) == 1
