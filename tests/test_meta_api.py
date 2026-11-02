"""B1a OFFLINE unit tests for decompile.meta_api.

Every test here uses a fake HTTP transport (httpx2.MockTransport) injected into the
real openai SDK client. No network call is made and no response here came from Meta.
These tests prove our client logic only; they are NOT integration evidence. The live
check is tools/meta_smoke.py.
"""

from __future__ import annotations

import json
import random
import ssl
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import httpx2
import openai
import pytest

from decompile import meta_api as m

FAKE_KEY = "sk-OFFLINE-FAKE-KEY-0123456789abcdef"

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "sum": {"type": "integer"},
        "colors": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["status", "sum", "colors"],
    "additionalProperties": False,
}


def completion(content="hello", finish_reason="stop", usage=None, **extra):
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": "muse-spark-1.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "total_tokens": 33,
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }
    body.update(extra)
    return body


def error_body(status, code="x", etype="invalid_request_error", message="boom"):
    return {"error": {"message": message, "type": etype, "param": None, "code": code}}


class FakeServer:
    """Replays a scripted list of responses/exceptions and records every request."""

    def __init__(self, script, on_request=None):
        self.script = list(script)
        self.requests: list[httpx2.Request] = []
        self.on_request = on_request  # called as on_request(request, n) before replying

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request(request, len(self.requests))
        if not self.script:
            raise AssertionError("fake server received more requests than scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("simulated transport failure", request=request)
        status, body, headers = item
        if isinstance(body, (dict, list)):
            return httpx2.Response(status, json=body, headers=headers)
        return httpx2.Response(status, content=body.encode(), headers=headers)

    def bodies(self):
        return [json.loads(r.content) for r in self.requests]


def ok(body=None, headers=None):
    return (200, body if body is not None else completion(), {"x-request-id": "req-1", **(headers or {})})


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(script, on_request=None, on_sleep=None, **kwargs):
        server = FakeServer(script, on_request=on_request)
        http = httpx2.Client(transport=httpx2.MockTransport(server))
        clients.append(http)
        sleeps: list[float] = []

        def _sleep(delay):
            sleeps.append(delay)
            if on_sleep is not None:
                on_sleep(delay)

        client = m.MuseSparkClient(
            evidence_dir=tmp_path / "evidence",
            api_key=kwargs.pop("api_key", FAKE_KEY),
            http_client=http,
            sleep=_sleep,
            rng=random.Random(0),
            **kwargs,
        )
        return client, server, sleeps

    yield _make
    for http in clients:
        http.close()


def call(client, **kwargs):
    kwargs.setdefault("purpose", "unit")
    kwargs.setdefault("max_completion_tokens", 64)
    return client.chat([{"role": "user", "content": "hi"}], **kwargs)


def evidence_files(tmp_path):
    return sorted((tmp_path / "evidence").glob("*.json"))


def load_only_evidence_in(directory):
    files = sorted(directory.glob("*.json"))
    assert len(files) == 1, files
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


def load_only_evidence(tmp_path):
    files = evidence_files(tmp_path)
    assert len(files) == 1, files
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


# ------------------------------------------------------------------ credentials


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv(m.API_KEY_ENV, raising=False)
    with pytest.raises(m.MissingAPIKeyError):
        m.MuseSparkClient(evidence_dir=tmp_path)
    with pytest.raises(m.MissingAPIKeyError):
        m.MuseSparkClient(evidence_dir=tmp_path, api_key="   ")


def test_key_from_environment_is_sent_as_bearer(monkeypatch, tmp_path):
    monkeypatch.setenv(m.API_KEY_ENV, FAKE_KEY)
    server = FakeServer([ok()])
    with httpx2.Client(transport=httpx2.MockTransport(server)) as http:
        client = m.MuseSparkClient(evidence_dir=tmp_path, http_client=http)
        call(client)
    assert server.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert str(server.requests[0].url) == "https://api.meta.ai/v1/chat/completions"


def test_explicit_key_overrides_environment(monkeypatch, make_client):
    monkeypatch.setenv(m.API_KEY_ENV, "sk-ENV-KEY")
    client, server, _ = make_client([ok()])
    call(client)
    assert server.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"


def test_repr_never_contains_key(make_client):
    client, _, _ = make_client([])
    assert FAKE_KEY not in repr(client) and m.REDACTED in repr(client)


def test_key_never_written_to_evidence_or_errors(make_client, tmp_path):
    # Server echoes the key in its error message: it must be scrubbed everywhere.
    client, _, _ = make_client([(401, error_body(401, "invalid_api_key", "authentication_error", f"bad key {FAKE_KEY}"), {})])
    with pytest.raises(m.AuthenticationError) as info:
        call(client)
    assert FAKE_KEY not in str(info.value)
    path, record = load_only_evidence(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert FAKE_KEY not in text
    assert "authorization" not in text.lower()
    assert m.REDACTED in text


def test_key_scrubbed_on_success_path(make_client, tmp_path):
    client, _, _ = make_client([ok(completion(content=f"echo {FAKE_KEY}"))])
    call(client)
    path, _ = load_only_evidence(tmp_path)
    assert FAKE_KEY not in path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ endpoint


def test_defaults_are_meta_endpoint_and_muse_spark():
    assert m.BASE_URL == "https://api.meta.ai/v1"
    assert m.DEFAULT_MODEL == "muse-spark-1.3"


def test_openai_environment_cannot_redirect_endpoint(monkeypatch, tmp_path):
    # OpenAI SDK environment variables must not make the client fall back to OpenAI.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-be-used")
    server = FakeServer([ok()])
    with httpx2.Client(transport=httpx2.MockTransport(server)) as http:
        client = m.MuseSparkClient(evidence_dir=tmp_path, api_key=FAKE_KEY, http_client=http)
        assert str(client._client.base_url).rstrip("/") == "https://api.meta.ai/v1"
        result = call(client)
    request = server.requests[0]
    assert request.url.host == "api.meta.ai"
    assert str(request.url) == "https://api.meta.ai/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(request.content)["model"] == "muse-spark-1.3"
    _, record = load_only_evidence_in(tmp_path)
    assert record["endpoint"] == "https://api.meta.ai/v1/chat/completions"
    assert result.model == "muse-spark-1.3"


@pytest.mark.parametrize("bad", [None, "", "   ", "http://api.meta.ai/v1", "api.meta.ai/v1"])
def test_base_url_must_be_explicit_https(monkeypatch, tmp_path, bad):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(ValueError, match="base_url"):
        m.MuseSparkClient(evidence_dir=tmp_path, api_key=FAKE_KEY, base_url=bad)


@pytest.mark.parametrize("bad", [None, "", "  ", " muse-spark-1.3", "muse-spark-1.3 ", 5])
def test_model_must_be_explicit(tmp_path, bad):
    with pytest.raises(ValueError, match="model"):
        m.MuseSparkClient(evidence_dir=tmp_path, api_key=FAKE_KEY, model=bad)


# ------------------------------------------------------------- request shape


def test_request_uses_max_completion_tokens_and_default_low_reasoning(make_client):
    client, server, _ = make_client([ok()])
    call(client, max_completion_tokens=512)
    body = server.bodies()[0]
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body
    assert body["reasoning_effort"] == "low"
    assert body["model"] == "muse-spark-1.3"
    assert "response_format" not in body


def test_reasoning_effort_override(make_client):
    client, server, _ = make_client([ok()])
    call(client, reasoning_effort="high")
    assert server.bodies()[0]["reasoning_effort"] == "high"


@pytest.mark.parametrize("bad", ["none", "max", "LOW", ""])
def test_invalid_reasoning_effort_rejected_before_request(make_client, bad):
    client, server, _ = make_client([])
    with pytest.raises(ValueError):
        call(client, reasoning_effort=bad)
    assert server.requests == []


def test_structured_request_uses_strict_json_schema(make_client):
    client, server, _ = make_client([ok(completion(content=json.dumps({"status": "ok", "sum": 42, "colors": ["red"]})))])
    result = call(client, json_schema=SIMPLE_SCHEMA, schema_name="smoke")
    rf = server.bodies()[0]["response_format"]
    assert rf == {"type": "json_schema", "json_schema": {"name": "smoke", "schema": SIMPLE_SCHEMA, "strict": True}}
    assert result.parsed == {"status": "ok", "sum": 42, "colors": ["red"]}


def test_invalid_schema_rejected_before_request(make_client, tmp_path):
    client, server, _ = make_client([])
    bad = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}  # no additionalProperties
    with pytest.raises(m.SchemaError):
        call(client, json_schema=bad)
    assert server.requests == []
    assert evidence_files(tmp_path) == []


# ------------------------------------------------------------- response parsing


def test_success_parses_content_usage_and_ids(make_client, tmp_path):
    client, _, _ = make_client([ok()])
    result = call(client)
    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.response_id == "chatcmpl-fake"
    assert result.request_id == "req-1"
    assert result.usage == m.Usage(prompt_tokens=11, completion_tokens=22, reasoning_tokens=7, total_tokens=33)
    assert result.attempts == 1
    assert result.evidence_path == evidence_files(tmp_path)[0]


def test_usage_missing_fields_are_none(make_client):
    client, _, _ = make_client([ok(completion(usage={"prompt_tokens": 3}))])
    usage = call(client).usage
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens is None and usage.reasoning_tokens is None and usage.total_tokens is None


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls", None, "something_new"])
def test_finish_reason_other_than_stop_fails(make_client, finish_reason):
    client, _, _ = make_client([ok(completion(finish_reason=finish_reason))])
    with pytest.raises(m.IncompleteResponseError) as info:
        call(client)
    assert info.value.finish_reason == finish_reason


@pytest.mark.parametrize("content", ["", "   \n", None])
def test_empty_content_fails(make_client, content):
    client, _, _ = make_client([ok(completion(content=content))])
    with pytest.raises(m.EmptyResponseError):
        call(client)


@pytest.mark.parametrize(
    "body",
    ["not json at all", {"choices": []}, {"choices": [{"finish_reason": "stop"}]}, {"no": "choices"}],
)
def test_malformed_success_body_fails(make_client, body):
    client, _, _ = make_client([(200, body, {})])
    with pytest.raises(m.MalformedResponseError):
        call(client)


def test_structured_invalid_json_fails(make_client):
    client, _, _ = make_client([ok(completion(content="{not json"))])
    with pytest.raises(m.StructuredOutputError):
        call(client, json_schema=SIMPLE_SCHEMA)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ok", "sum": 42},  # missing key
        {"status": "ok", "sum": 42, "colors": ["r"], "extra": 1},  # additional key
        {"status": "bad", "sum": 42, "colors": ["r"]},  # enum
        {"status": "ok", "sum": 4.2, "colors": ["r"]},  # integer
        {"status": "ok", "sum": True, "colors": ["r"]},  # bool is not integer
        {"status": "ok", "sum": 42, "colors": []},  # minItems
        ["not", "an", "object"],
    ],
)
def test_structured_schema_violation_fails(make_client, tmp_path, payload):
    client, _, _ = make_client([ok(completion(content=json.dumps(payload)))])
    with pytest.raises(m.StructuredOutputError):
        call(client, json_schema=SIMPLE_SCHEMA)
    _, record = load_only_evidence(tmp_path)
    assert record["outcome"]["error_class"] == "StructuredOutputError"


# ------------------------------------------------------------------ retry policy


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_then_success(make_client, tmp_path, status):
    client, server, sleeps = make_client([(status, error_body(status), {}), ok()])
    result = call(client)
    assert result.attempts == 2
    assert len(server.requests) == 2
    assert len(sleeps) == 1
    _, record = load_only_evidence(tmp_path)
    assert [a["http_status"] for a in record["attempts"]] == [status, 200]
    assert record["attempts"][0]["retry"] is True
    assert record["outcome"]["ok"] is True


@pytest.mark.parametrize(
    "status, exc",
    [
        (400, m.InvalidRequestError),
        (401, m.AuthenticationError),
        (402, m.BillingError),
        (403, m.PermissionDeniedError),
        (404, m.InvalidRequestError),
        (413, m.InvalidRequestError),
        (504, m.GatewayTimeoutError),
    ],
)
def test_non_retryable_status_single_attempt(make_client, tmp_path, status, exc):
    client, server, sleeps = make_client([(status, error_body(status), {"retry-after": "1"})])
    with pytest.raises(exc) as info:
        call(client)
    assert info.value.status == status
    assert len(server.requests) == 1
    assert sleeps == []
    _, record = load_only_evidence(tmp_path)
    assert len(record["attempts"]) == 1
    assert record["attempts"][0]["retry"] is False
    assert record["outcome"] == {"ok": False, "error_class": exc.__name__, "error_message": str(info.value)}
    assert info.value.evidence_path is not None


def test_retries_capped_at_three_attempts(make_client, tmp_path):
    client, server, sleeps = make_client([(503, error_body(503, "service_overloaded", "server_error"), {})] * 3)
    with pytest.raises(m.ServerError) as info:
        call(client)
    assert len(server.requests) == m.MAX_ATTEMPTS == 3
    assert len(sleeps) == 2
    assert info.value.error_code == "service_overloaded"
    _, record = load_only_evidence(tmp_path)
    assert [a["http_status"] for a in record["attempts"]] == [503, 503, 503]
    assert record["attempts"][-1]["retry_reason"] == "attempts exhausted"


def test_rate_limit_exhausted_raises_rate_limit_error(make_client):
    client, server, _ = make_client([(429, error_body(429, "rate_limit_exceeded", "rate_limit_error"), {})] * 3)
    with pytest.raises(m.RateLimitError):
        call(client)
    assert len(server.requests) == 3


def test_retry_after_seconds_honoured(make_client):
    client, _, sleeps = make_client([(429, error_body(429), {"retry-after": "15"}), ok()])
    call(client)
    assert sleeps == [15.0]  # backoff for attempt 1 is <= 1.0 s, so Retry-After dominates


def test_retry_after_http_date_honoured(make_client):
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    client, _, sleeps = make_client([(503, error_body(503), {"retry-after": when}), ok()])
    call(client)
    assert 25.0 <= sleeps[0] <= 30.0


def test_retry_after_above_limit_gives_up(make_client, tmp_path):
    client, server, sleeps = make_client([(503, error_body(503), {"retry-after": "120"})])
    with pytest.raises(m.ServerError):
        call(client)
    assert len(server.requests) == 1 and sleeps == []
    _, record = load_only_evidence(tmp_path)
    assert "exceeds" in record["attempts"][0]["retry_reason"]


def test_backoff_is_exponential_with_jitter(make_client):
    client, _, sleeps = make_client([(500, error_body(500), {})] * 3)
    with pytest.raises(m.ServerError):
        call(client)
    # attempt 1: base 0.5 + U(0, 0.5); attempt 2: base 1.0 + U(0, 1.0)
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


NOT_SENT_CAUSES = [httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout]


class SubclassedConnectError(httpx2.ConnectError):
    pass


class SubclassedConnectTimeout(httpx2.ConnectTimeout):
    pass


class SubclassedPoolTimeout(httpx2.PoolTimeout):
    pass


AMBIGUOUS_CAUSES = [
    httpx2.ReadTimeout,
    httpx2.WriteTimeout,
    httpx2.ReadError,
    httpx2.WriteError,
    httpx2.RemoteProtocolError,
    httpx2.ProxyError,
    httpx2.UnsupportedProtocol,
    SubclassedConnectError,
    SubclassedConnectTimeout,
    SubclassedPoolTimeout,
]


@pytest.mark.parametrize("exc_type", NOT_SENT_CAUSES)
def test_not_sent_transport_errors_retried_then_succeed(make_client, tmp_path, exc_type):
    client, server, sleeps = make_client([exc_type, ok()])
    result = call(client)
    assert result.attempts == 2 and len(server.requests) == 2 and len(sleeps) == 1
    _, record = load_only_evidence(tmp_path)
    first = record["attempts"][0]
    assert first["state"] == "transport_error"
    assert first["exception"]["type"] in ("APIConnectionError", "APITimeoutError")
    assert first["cause_type"].endswith("." + exc_type.__name__)
    assert first["transport_classification"] == "not_sent_retryable"
    assert first["request_possibly_sent"] is False
    assert first["retry"] is True
    assert "http_status" not in first


@pytest.mark.parametrize("exc_type", NOT_SENT_CAUSES)
def test_not_sent_transport_errors_exhausted(make_client, tmp_path, exc_type):
    client, server, sleeps = make_client([exc_type] * 3)
    with pytest.raises(m.TransportError) as info:
        call(client)
    assert info.value.request_possibly_sent is False
    assert len(server.requests) == 3 and len(sleeps) == 2
    _, record = load_only_evidence(tmp_path)
    assert len(record["attempts"]) == 3
    assert record["attempts"][-1]["retry_reason"] == "attempts exhausted"


@pytest.mark.parametrize(
    "cause",
    AMBIGUOUS_CAUSES + [ssl.SSLError("simulated TLS failure")],
    ids=lambda c: c.__name__ if isinstance(c, type) else type(c).__name__,
)
def test_ambiguous_transport_errors_not_retried(make_client, tmp_path, cause):
    client, server, sleeps = make_client([cause, ok()])
    with pytest.raises(m.TransportError) as info:
        call(client)
    assert info.value.request_possibly_sent is True
    assert len(server.requests) == 1 and sleeps == []
    _, record = load_only_evidence(tmp_path)
    assert record["state"] == "failed"
    (attempt,) = record["attempts"]
    assert attempt["transport_classification"] == "ambiguous_not_retried"
    assert attempt["request_possibly_sent"] is True
    assert attempt["retry"] is False and attempt["retry_reason"] == "not retryable"


def _sdk_wrapped(cause: BaseException) -> openai.APIConnectionError:
    request = httpx2.Request("POST", "https://api.meta.ai/v1/chat/completions")
    try:
        try:
            raise cause
        except BaseException as inner:
            raise openai.APIConnectionError(request=request) from inner
    except openai.APIConnectionError as wrapped:
        return wrapped


def test_classification_uses_exact_types_only():
    request = httpx2.Request("POST", "https://api.meta.ai/v1/chat/completions")
    for exc_type in NOT_SENT_CAUSES:
        retryable, cause_type = m.classify_transport_error(_sdk_wrapped(exc_type("x", request=request)))
        assert retryable is True and cause_type.endswith(exc_type.__name__)
    # ProxyError (request not sent, but outside the approved list) and subclasses of the
    # retryable types must NOT be retried.
    for exc_type in (httpx2.ProxyError, SubclassedConnectError, SubclassedConnectTimeout, SubclassedPoolTimeout):
        retryable, _ = m.classify_transport_error(_sdk_wrapped(exc_type("x", request=request)))
        assert retryable is False, exc_type
    # No cause at all: ambiguous.
    assert m.classify_transport_error(openai.APIConnectionError(request=request)) == (False, None)


def test_sdk_does_not_retry_on_its_own(make_client):
    # Three scripted 500s and MAX_ATTEMPTS == 3: our loop sends exactly three requests.
    # If the SDK also retried internally (max_retries > 0) it would send extra requests
    # and the fake server would fail with "more requests than scripted".
    client, server, _ = make_client([(500, error_body(500), {}), (500, error_body(500), {}), (500, error_body(500), {})])
    with pytest.raises(m.ServerError):
        call(client)
    assert len(server.requests) == 3
    assert client._client.max_retries == 0


# ------------------------------------------------------------------- evidence


def test_evidence_record_contents(make_client, tmp_path):
    headers = {"set-cookie": "secret=1", "x-ratelimit-remaining-requests": "2999", "x-internal": "nope"}
    client, _, _ = make_client([ok(headers=headers)])
    call(client, purpose="Plain Text / smoke", max_completion_tokens=128)
    path, record = load_only_evidence(tmp_path)
    assert "_Plain-Text-smoke_" in path.name
    assert record["transport"] == "injected"
    assert record["request"]["max_completion_tokens"] == 128
    assert record["request"]["messages"] == [{"role": "user", "content": "hi"}]
    attempt = record["attempts"][0]
    assert attempt["http_status"] == 200
    assert attempt["request_id"] == "req-1"
    assert set(attempt["response_headers"]) <= set(m.SAFE_RESPONSE_HEADERS)
    assert attempt["response_headers"]["x-ratelimit-remaining-requests"] == "2999"
    assert "set-cookie" not in json.dumps(record) and "x-internal" not in json.dumps(record)
    assert json.loads(attempt["response_body_raw"])["id"] == "chatcmpl-fake"
    assert isinstance(attempt["latency_s"], float)
    assert record["outcome"]["usage"]["reasoning_tokens"] == 7


def test_evidence_filenames_unique_and_no_temp_files(make_client, tmp_path):
    client, _, _ = make_client([ok(), ok(), ok()])
    for _ in range(3):
        call(client, purpose="same")
    files = evidence_files(tmp_path)
    assert len(files) == 3 and len({f.name for f in files}) == 3
    assert list((tmp_path / "evidence").glob("*.tmp")) == []


def test_write_json_atomic_refuses_overwrite(tmp_path):
    target = tmp_path / "a.json"
    m.write_json_atomic(target, {"x": 1})
    with pytest.raises(FileExistsError):
        m.write_json_atomic(target, {"x": 2})
    assert json.loads(target.read_text()) == {"x": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


# ------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("15", 15.0),
        ("0.5", 0.5),
        ("-3", 0.0),
        ("garbage", None),
        ("nan", None),
        ("NaN", None),
        ("inf", None),
        ("-inf", None),
        ("Infinity", None),
        ("1e999", None),
        ("-1e999", None),
    ],
)
def test_parse_retry_after(value, expected):
    assert m.parse_retry_after(value) == expected


def test_strict_schema_accepts_valid():
    m.check_strict_schema(SIMPLE_SCHEMA)
    with_ref = {
        "type": "object",
        "properties": {"p": {"$ref": "#/$defs/point"}, "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["p", "maybe"],
        "additionalProperties": False,
        "$defs": {
            "point": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            }
        },
    }
    m.check_strict_schema(with_ref)
    assert m.validate_instance({"p": {"x": 1, "y": 2.5}, "maybe": None}, with_ref) == []
    assert m.validate_instance({"p": {"x": "1", "y": 2}, "maybe": 3}, with_ref) != []


def _obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _nested(depth):
    node = {"type": "string"}
    for _ in range(depth):
        node = _obj({"a": node})
    return node


@pytest.mark.parametrize(
    "schema, fragment",
    [
        ({"type": "array", "items": {"type": "string"}}, "type 'object'"),
        ({"anyOf": [_obj({})]}, "root"),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}, "additionalProperties"),
        ({"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False}, "required"),
        (_obj({"a": {"oneOf": [{"type": "string"}]}}), "oneOf"),
        (_obj({"a": {"allOf": [{"type": "string"}]}}), "oneOf/allOf"),
        (_obj({"a": {"type": "string", "pattern": "^x$"}}), "unsupported keyword"),
        (_obj({"a": {"type": "object"}}), "additionalProperties"),
        (_obj({"a": {"type": "array"}}), "items"),
        (_nested(11), "depth"),
        ({**_obj({"n": {"$ref": "#/$defs/n"}}), "$defs": {"n": _obj({"n": {"$ref": "#/$defs/n"}})}}, "recursive"),
        (_obj({"a": {"$ref": "http://example.com/x"}}), "local"),
        (_obj({"a": {"enum": list(range(1001))}}), "enum values"),
    ],
)
def test_strict_schema_rejections(schema, fragment):
    with pytest.raises(m.SchemaError) as info:
        m.check_strict_schema(schema)
    assert fragment in str(info.value)


def test_strict_schema_depth_limit_boundary():
    m.check_strict_schema(_nested(9))  # 9 objects + leaf = depth 10


# ------------------------------------------------------------ input validation (Fix 1)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "5", None])
def test_max_completion_tokens_rejects_bool_and_non_positive(make_client, tmp_path, bad):
    client, server, _ = make_client([])
    with pytest.raises(ValueError, match="max_completion_tokens"):
        call(client, max_completion_tokens=bad)
    assert server.requests == [] and evidence_files(tmp_path) == []


@pytest.mark.parametrize("bad", ["", "  ", " muse-spark-1.3", "muse-spark-1.3 ", 5])
def test_per_call_model_override_validated(make_client, tmp_path, bad):
    client, server, _ = make_client([])
    with pytest.raises(ValueError, match="model"):
        call(client, model=bad)
    assert server.requests == [] and evidence_files(tmp_path) == []


def test_per_call_model_none_uses_default_and_override_is_sent(make_client):
    client, server, _ = make_client([ok(), ok()])
    call(client, model=None)
    call(client, model="muse-spark-1.2")
    assert [b["model"] for b in server.bodies()] == ["muse-spark-1.3", "muse-spark-1.2"]


@pytest.mark.parametrize("bad", [0, -1, 0.0, -0.5, float("inf"), float("nan"), True, "10", None])
def test_timeout_must_be_positive_finite(tmp_path, bad):
    with pytest.raises(ValueError, match="timeout_s"):
        m.MuseSparkClient(evidence_dir=tmp_path, api_key=FAKE_KEY, timeout_s=bad)


@pytest.mark.parametrize("good", [0.5, 1, 120])
def test_timeout_accepts_positive_finite(tmp_path, good):
    client = m.MuseSparkClient(evidence_dir=tmp_path, api_key=FAKE_KEY, timeout_s=good)
    assert client._client.timeout == float(good)


# ------------------------------------------------------------ HTTP status (Fix 2)


@pytest.mark.parametrize("status", [201, 202, 204])
def test_non_200_success_status_is_typed_failure(make_client, tmp_path, status):
    body = "" if status == 204 else completion()
    client, server, sleeps = make_client([(status, body, {"x-request-id": "req-s"})])
    with pytest.raises(m.UnexpectedStatusError) as info:
        call(client)
    assert info.value.status == status
    assert len(server.requests) == 1 and sleeps == []
    _, record = load_only_evidence(tmp_path)
    assert record["state"] == "failed"
    assert record["outcome"]["error_class"] == "UnexpectedStatusError"
    (attempt,) = record["attempts"]
    assert attempt["http_status"] == status and attempt["request_id"] == "req-s"
    assert "response_body_raw" in attempt


def test_redirect_status_is_not_followed_or_retried(make_client, tmp_path):
    client, server, _ = make_client([(302, "", {"location": "https://api.openai.com/v1/chat/completions"})])
    with pytest.raises(m.InvalidRequestError) as info:
        call(client)
    assert info.value.status == 302
    assert len(server.requests) == 1
    assert [str(r.url) for r in server.requests] == ["https://api.meta.ai/v1/chat/completions"]


# ------------------------------------------------------------ strict JSON (Fix 4)

BAD_JSON = {
    "nan-top": '{"v": NaN}',
    "inf-top": '{"v": Infinity}',
    "neginf-top": '{"v": -Infinity}',
    "overflow-top": '{"v": 1e999}',
    "nan-nested": '{"a": {"b": [1, NaN]}}',
    "inf-nested": '{"a": [{"b": Infinity}]}',
    "neginf-nested": '{"a": {"b": {"c": -Infinity}}}',
    "overflow-nested": '{"a": [[-1e999]]}',
    "dup-top": '{"a": 1, "a": 2}',
    "dup-nested": '{"a": {"b": 1, "b": 1}}',
    "dup-in-array": '{"a": [{"k": "x", "k": "y"}]}',
}


@pytest.mark.parametrize("text", BAD_JSON.values(), ids=BAD_JSON.keys())
def test_strict_json_loads_rejects(text):
    with pytest.raises(ValueError):
        m.strict_json_loads(text)


def test_strict_json_loads_accepts_valid_json():
    text = '{"a": [1, 2.5, -3e10, 12345678901234567890, true, null], "b": {"c": "NaN", "d": "Infinity"}}'
    assert m.strict_json_loads(text) == {
        "a": [1, 2.5, -3e10, 12345678901234567890, True, None],
        "b": {"c": "NaN", "d": "Infinity"},
    }


@pytest.mark.parametrize("text", BAD_JSON.values(), ids=BAD_JSON.keys())
def test_structured_output_strict_json(make_client, tmp_path, text):
    # Strict parsing happens before schema validation, so any shape is enough here.
    client, _, _ = make_client([ok(completion(content=text))])
    with pytest.raises(m.StructuredOutputError, match="strict JSON"):
        call(client, json_schema=SIMPLE_SCHEMA)
    path, record = load_only_evidence(tmp_path)
    m.strict_json_loads(path.read_text(encoding="utf-8"))  # evidence itself stays strict JSON


def _body_text(**overrides) -> str:
    return json.dumps(completion(**overrides))


BAD_BODIES = {
    "nan-top": lambda: '{"x": NaN, ' + _body_text()[1:],
    "inf-nested-usage": lambda: _body_text().replace('"prompt_tokens": 11', '"prompt_tokens": Infinity'),
    "overflow-nested": lambda: _body_text().replace('"created": 0', '"created": 1e999'),
    "dup-top": lambda: '{"id": "dup", ' + _body_text()[1:],
    "dup-nested-choice": lambda: _body_text().replace('"finish_reason": "stop"', '"finish_reason": "stop", "finish_reason": "stop"'),
}


@pytest.mark.parametrize("make_body", BAD_BODIES.values(), ids=BAD_BODIES.keys())
def test_response_body_strict_json(make_client, tmp_path, make_body):
    text = make_body()
    client, _, _ = make_client([(200, text, {"content-type": "application/json"})])
    with pytest.raises(m.MalformedResponseError, match="strict JSON"):
        call(client)
    path, record = load_only_evidence(tmp_path)
    attempt = record["attempts"][0]
    assert attempt["response_json"] is None and attempt["response_json_error"]
    assert attempt["response_body_raw"] == text  # raw body preserved as a string
    m.strict_json_loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ Retry-After (refinement 3)


INVALID_RETRY_AFTER = ["inf", "-inf", "nan", "NaN", "Infinity", "-Infinity", "1e999", "-1e999"]


@pytest.mark.parametrize("header", INVALID_RETRY_AFTER)
def test_non_finite_retry_after_ignored_in_delay_and_evidence(make_client, tmp_path, header):
    client, _, sleeps = make_client([(429, error_body(429), {"retry-after": header}), ok()])
    call(client)
    assert 0.5 <= sleeps[0] <= 1.0  # plain backoff, the invalid header contributed nothing
    path, record = load_only_evidence(tmp_path)
    first = record["attempts"][0]
    assert first["retry_after_s"] is None
    assert first["retry_after_header_invalid"] is True
    assert "retry-after" not in first["response_headers"]
    text = path.read_text(encoding="utf-8")
    assert header not in text  # the raw invalid value appears nowhere in the evidence
    m.strict_json_loads(text)


@pytest.mark.parametrize("header", INVALID_RETRY_AFTER)
def test_invalid_retry_after_absent_from_every_evidence_version(make_client, tmp_path, header):
    # Snapshot the evidence at each request and each sleep, and after exhaustion.
    texts: list[str] = []

    def grab(*_args):
        texts.extend(p.read_text(encoding="utf-8") for p in evidence_files(tmp_path))

    script = [(503, error_body(503), {"retry-after": header})] * 3
    client, server, _ = make_client(script, on_request=grab, on_sleep=grab)
    with pytest.raises(m.ServerError):
        call(client)
    grab()
    assert len(server.requests) == 3 and len(texts) == 6
    for text in texts:
        assert header not in text
    record = m.strict_json_loads(texts[-1])
    assert all(a["retry_after_header_invalid"] is True for a in record["attempts"])
    assert all("retry-after" not in a["response_headers"] for a in record["attempts"])


def test_valid_retry_after_is_recorded(make_client, tmp_path):
    client, _, sleeps = make_client([(429, error_body(429), {"retry-after": "15"}), ok()])
    call(client)
    _, record = load_only_evidence(tmp_path)
    first = record["attempts"][0]
    assert first["response_headers"]["retry-after"] == "15"
    assert first["retry_after_header_invalid"] is False
    assert first["retry_after_s"] == 15.0 and sleeps == [15.0]
    assert "retry_after_header_invalid" not in record["attempts"][1]  # header absent there


def test_retry_after_exactly_at_limit_is_honoured(make_client):
    client, _, sleeps = make_client([(503, error_body(503), {"retry-after": "60"}), ok()])
    call(client)
    assert sleeps == [60.0]


def test_mixed_retryable_then_fatal_stops(make_client, tmp_path):
    client, server, sleeps = make_client([(503, error_body(503), {}), (400, error_body(400), {}), ok()])
    with pytest.raises(m.InvalidRequestError):
        call(client)
    assert len(server.requests) == 2 and len(sleeps) == 1
    _, record = load_only_evidence(tmp_path)
    assert [a["http_status"] for a in record["attempts"]] == [503, 400]


# ------------------------------------------------------------ evidence lifecycle (Fix 5)


def _snapshot(tmp_path) -> dict:
    files = sorted((tmp_path / "evidence").glob("*.json"))
    assert len(files) == 1, files
    return m.strict_json_loads(files[0].read_text(encoding="utf-8"))


def test_initial_record_exists_before_request_with_attempt_in_flight(make_client, tmp_path):
    seen = []
    client, _, _ = make_client([ok()], on_request=lambda req, n: seen.append(_snapshot(tmp_path)))
    call(client)
    (first,) = seen
    assert first["evidence_version"] == 2 and first["evidence_id"]
    assert first["state"] == "in_progress" and first["outcome"] is None
    assert len(first["attempts"]) == 1
    assert first["attempts"][0]["attempt"] == 1
    assert first["attempts"][0]["state"] == "in_flight"
    assert "http_status" not in first["attempts"][0]
    assert "UNKNOWN" in first["in_flight_semantics"]
    final = _snapshot(tmp_path)
    assert final["evidence_id"] == first["evidence_id"]
    assert final["state"] == "succeeded"
    assert final["attempts"][0]["state"] == "response_received"


def test_each_attempt_persisted_in_flight_then_outcome_before_sleep(make_client, tmp_path):
    events = []
    client, _, _ = make_client(
        [(503, error_body(503), {}), httpx2.ConnectError, ok()],
        on_request=lambda req, n: events.append(("request", n, _snapshot(tmp_path))),
        on_sleep=lambda d: events.append(("sleep", d, _snapshot(tmp_path))),
    )
    call(client)
    kinds = [(e[0], len(e[2]["attempts"])) for e in events]
    assert kinds == [("request", 1), ("sleep", 1), ("request", 2), ("sleep", 2), ("request", 3)]

    def states(snap):
        return [a["state"] for a in snap["attempts"]]

    assert states(events[0][2]) == ["in_flight"]
    # Outcome of attempt 1 is on disk before the first sleep.
    assert states(events[1][2]) == ["response_received"]
    assert events[1][2]["attempts"][0]["http_status"] == 503
    assert events[1][2]["attempts"][0]["sleep_s"] == round(events[1][1], 4)
    # Attempt 2 is persisted in flight before its request.
    assert states(events[2][2]) == ["response_received", "in_flight"]
    # Outcome of attempt 2 (transport) is on disk before the second sleep.
    assert states(events[3][2]) == ["response_received", "transport_error"]
    assert events[3][2]["attempts"][1]["transport_classification"] == "not_sent_retryable"
    assert states(events[4][2]) == ["response_received", "transport_error", "in_flight"]
    final = _snapshot(tmp_path)
    assert final["state"] == "succeeded"
    assert states(final) == ["response_received", "transport_error", "response_received"]
    assert all(snap["state"] == "in_progress" for _, _, snap in events)


def test_key_redacted_in_every_evidence_version(make_client, tmp_path):
    texts = []

    def grab(*_):
        files = sorted((tmp_path / "evidence").glob("*.json"))
        texts.append(files[0].read_text(encoding="utf-8"))

    client, _, _ = make_client(
        [(503, error_body(503, message=f"overloaded {FAKE_KEY}"), {}), ok(completion(content=f"x {FAKE_KEY}"))],
        on_request=grab,
        on_sleep=grab,
    )
    client.chat(
        [{"role": "user", "content": f"secret {FAKE_KEY}"}], purpose="redact", max_completion_tokens=8
    )
    grab()
    assert len(texts) == 4  # initial, outcome-before-sleep, attempt-2 in flight, final
    for text in texts:
        assert FAKE_KEY not in text
        assert "authorization" not in text.lower()


def _fail_replace_after(monkeypatch, successes: int) -> None:
    real = m.replace_json_atomic
    calls = {"n": 0}

    def flaky(path, data, evidence_id):
        calls["n"] += 1
        if calls["n"] > successes:
            raise OSError("simulated disk full")
        return real(path, data, evidence_id)

    monkeypatch.setattr(m, "replace_json_atomic", flaky)


def test_initial_evidence_failure_sends_no_request(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    server = FakeServer([ok()])
    with httpx2.Client(transport=httpx2.MockTransport(server)) as http:
        client = m.MuseSparkClient(evidence_dir=blocker / "evidence", api_key=FAKE_KEY, http_client=http)
        with pytest.raises(m.EvidenceWriteError) as info:
            call(client)
    err = info.value
    assert err.stage == "initial" and err.api_error is None and err.result is None
    assert isinstance(err.__cause__, OSError)
    assert "no request was sent" in str(err)
    assert server.requests == []


def test_final_evidence_failure_on_success_keeps_result(make_client, tmp_path, monkeypatch):
    _fail_replace_after(monkeypatch, successes=0)
    client, server, _ = make_client([ok()])
    with pytest.raises(m.EvidenceWriteError) as info:
        call(client)
    err = info.value
    assert err.stage == "final" and err.api_error is None
    assert err.result is not None and err.result.content == "hello"
    assert isinstance(err.__cause__, OSError)
    assert "call succeeded" in str(err)
    assert len(server.requests) == 1
    # The last record that did reach disk shows attempt 1 in flight: outcome unknown.
    on_disk = _snapshot(tmp_path)
    assert on_disk["state"] == "in_progress"
    assert on_disk["attempts"][0]["state"] == "in_flight"


def test_final_evidence_failure_on_api_error_keeps_original(make_client, monkeypatch):
    _fail_replace_after(monkeypatch, successes=0)
    client, _, _ = make_client(
        [(401, error_body(401, "invalid_api_key", "authentication_error", f"bad key {FAKE_KEY}"), {})]
    )
    with pytest.raises(m.EvidenceWriteError) as info:
        call(client)
    err = info.value
    assert err.stage == "final"
    assert isinstance(err.api_error, m.AuthenticationError) and err.api_error.status == 401
    assert isinstance(err.__cause__, OSError)
    # The original error is also preserved in the exception chain: the filesystem error
    # was raised while the API error was being handled.
    assert err.__cause__.__context__ is err.api_error
    assert "AuthenticationError" in str(err)
    assert FAKE_KEY not in str(err) and FAKE_KEY not in str(err.api_error)


def test_attempt_outcome_evidence_failure_stops_before_sleep(make_client, monkeypatch):
    _fail_replace_after(monkeypatch, successes=0)
    client, server, sleeps = make_client([(503, error_body(503), {}), ok()])
    with pytest.raises(m.EvidenceWriteError) as info:
        call(client)
    err = info.value
    assert err.stage == "attempt_outcome"
    assert isinstance(err.api_error, m.ServerError)
    assert len(server.requests) == 1 and sleeps == []


def test_attempt_start_evidence_failure_sends_no_retry(make_client, monkeypatch):
    _fail_replace_after(monkeypatch, successes=1)
    client, server, sleeps = make_client([(503, error_body(503), {}), ok()])
    with pytest.raises(m.EvidenceWriteError) as info:
        call(client)
    err = info.value
    assert err.stage == "attempt_start"
    assert isinstance(err.api_error, m.ServerError)
    assert "retry request was not sent" in str(err)
    assert len(server.requests) == 1 and len(sleeps) == 1


def test_keyboard_interrupt_preserved_when_final_evidence_fails(make_client, monkeypatch):
    _fail_replace_after(monkeypatch, successes=1)  # outcome write succeeds, final fails

    def interrupt(_delay):
        raise KeyboardInterrupt

    client, server, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=interrupt)
    with pytest.raises(KeyboardInterrupt) as info:
        call(client)
    notes = getattr(info.value, "__notes__", [])
    assert any("evidence write failed at stage 'final'" in n for n in notes)
    assert len(server.requests) == 1


def test_interrupt_during_request_marks_attempt_outcome_unknown(make_client, tmp_path):
    def interrupt(req, n):
        raise KeyboardInterrupt

    client, _, _ = make_client([ok()], on_request=interrupt)
    with pytest.raises(KeyboardInterrupt):
        call(client)
    record = _snapshot(tmp_path)
    assert record["state"] == "failed"
    assert record["outcome"]["error_class"] == "KeyboardInterrupt"
    assert record["attempts"][0]["state"] == "interrupted_outcome_unknown"


def test_replace_json_atomic_guards(tmp_path):
    target = tmp_path / "rec.json"
    with pytest.raises(FileNotFoundError):
        m.replace_json_atomic(target, {"evidence_id": "a"}, "a")
    m.write_json_atomic(target, {"evidence_id": "a", "v": 1})
    with pytest.raises(ValueError, match="mismatch"):
        m.replace_json_atomic(target, {"evidence_id": "b", "v": 2}, "b")
    with pytest.raises(ValueError, match="different evidence_id"):
        m.replace_json_atomic(target, {"evidence_id": "b", "v": 2}, "a")
    m.replace_json_atomic(target, {"evidence_id": "a", "v": 3}, "a")
    assert json.loads(target.read_text()) == {"evidence_id": "a", "v": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["rec.json"]  # no temp files left


def test_evidence_writer_rejects_non_finite_numbers(tmp_path):
    with pytest.raises(ValueError):
        m.write_json_atomic(tmp_path / "x.json", {"v": float("nan")})
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- unexpected exceptions (Fix B)


def _exposed_text(exc: BaseException) -> str:
    """Everything a caller can see: the formatted traceback plus str/repr/args/notes of
    every exception reachable through __cause__ / __context__."""
    import traceback

    parts = ["".join(traceback.format_exception(exc))]
    seen: set[int] = set()
    stack = [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        parts += [str(e), repr(e), repr(e.args), *getattr(e, "__notes__", [])]
        parts += [repr(getattr(e, "api_error", None))]
        stack += [e.__cause__, e.__context__, getattr(e, "api_error", None)]
    return "\n".join(parts)


def _raise_with_key(_delay):
    try:
        raise ValueError(f"inner detail {FAKE_KEY}")
    except ValueError as inner:
        raise RuntimeError(f"boom {FAKE_KEY}") from inner


def test_unexpected_exception_is_sanitized(make_client, tmp_path):
    client, server, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=_raise_with_key)
    with pytest.raises(m.UnexpectedClientError) as info:
        call(client)
    err = info.value
    assert isinstance(err, m.MetaAPIError)
    assert err.original_type == "builtins.RuntimeError"
    assert "RuntimeError" in str(err) and "boom" in str(err) and m.REDACTED in str(err)
    assert err.__cause__ is None and err.__context__ is None
    assert FAKE_KEY not in _exposed_text(err)
    assert len(server.requests) == 1
    path, record = load_only_evidence(tmp_path)
    assert err.evidence_path == path
    text = path.read_text(encoding="utf-8")
    assert FAKE_KEY not in text
    assert record["state"] == "failed"
    assert record["outcome"]["error_class"] == "RuntimeError"
    assert record["outcome"]["raised_as"] == "UnexpectedClientError"
    assert record["attempts"][0]["state"] == "response_received"


def test_unexpected_exception_while_in_flight_marks_outcome_unknown(make_client, tmp_path, monkeypatch):
    client, _, _ = make_client([ok()])

    def explode(**_kwargs):
        raise KeyError(FAKE_KEY)  # str(KeyError) is the repr of the key

    monkeypatch.setattr(client._client.chat.completions.with_raw_response, "create", explode)
    with pytest.raises(m.UnexpectedClientError) as info:
        call(client)
    err = info.value
    assert err.original_type == "builtins.KeyError"
    assert FAKE_KEY not in _exposed_text(err)
    path, record = load_only_evidence(tmp_path)
    assert FAKE_KEY not in path.read_text(encoding="utf-8")
    assert record["attempts"][0]["state"] == "interrupted_outcome_unknown"


def test_unexpected_exception_with_failed_final_write_stays_sanitized(make_client, monkeypatch):
    _fail_replace_after(monkeypatch, successes=1)  # attempt_outcome ok, final write fails
    client, _, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=_raise_with_key)
    with pytest.raises(m.EvidenceWriteError) as info:
        call(client)
    err = info.value
    assert err.stage == "final"
    assert isinstance(err.api_error, m.UnexpectedClientError)
    assert err.api_error.original_type == "builtins.RuntimeError"
    assert err.api_error.__cause__ is None and err.api_error.__context__ is None
    assert FAKE_KEY not in _exposed_text(err)


def test_unprintable_unexpected_exception(make_client):
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError(FAKE_KEY)

    def raiser(_delay):
        raise Unprintable()

    client, _, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=raiser)
    with pytest.raises(m.UnexpectedClientError) as info:
        call(client)
    assert "<unprintable Unprintable>" in str(info.value)
    assert FAKE_KEY not in _exposed_text(info.value)


def test_system_exit_is_preserved(make_client, tmp_path):
    original = SystemExit(3)

    def leave(_delay):
        raise original

    client, server, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=leave)
    with pytest.raises(SystemExit) as info:
        call(client)
    assert info.value is original and info.value.code == 3
    assert len(server.requests) == 1
    _, record = load_only_evidence(tmp_path)
    assert record["outcome"]["error_class"] == "SystemExit"
    assert "raised_as" not in record["outcome"]


def test_keyboard_interrupt_is_the_same_object(make_client):
    original = KeyboardInterrupt()

    def interrupt(_delay):
        raise original

    client, _, _ = make_client([(503, error_body(503), {}), ok()], on_sleep=interrupt)
    with pytest.raises(KeyboardInterrupt) as info:
        call(client)
    assert info.value is original
