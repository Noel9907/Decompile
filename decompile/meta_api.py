"""B1a: reliable Muse Spark client for the Meta Model API.

One entry point, MuseSparkClient.chat(), wraps the OpenAI-compatible Chat Completions
endpoint (https://api.meta.ai/v1, documented at dev.meta.ai/docs) with:

- explicit credentials: the key comes from the constructor argument or the
  MODEL_API_KEY process environment variable. This module never reads .env files.
- a project-owned retry policy (the SDK's own retries are disabled):
    retried:      HTTP 429, 500, 503, and transport failures that prove the request
                  was never sent (httpx2 ConnectError / ConnectTimeout / PoolTimeout,
                  exact types only, see classify_transport_error)
    never:        HTTP 400, 401, 402, 403, 404, 413, 504 (and any other status), and
                  every ambiguous transport failure (read/write timeouts and errors,
                  protocol errors, proxy errors, TLS errors, unknown causes)
    at most MAX_ATTEMPTS total attempts; Retry-After is honoured; otherwise
    exponential backoff with jitter.
- strict structured output: response_format type=json_schema, strict=true. The schema
  is checked locally against Meta's documented strict subset before any request, and
  the response is validated against it after.
- fail-loud parsing: HTTP status other than 200, finish_reason other than "stop",
  empty content, a malformed body or a schema-violating structured response all
  raise. JSON is parsed strictly (no NaN/Infinity, no duplicate keys). Undocumented
  behaviour is never guessed at.
- evidence: every call writes one JSON record, created before the first request and
  atomically replaced as the call progresses. Each attempt is persisted as
  "in_flight" before its request is sent and with its outcome before any retry or
  sleep. An attempt still "in_flight" after a crash means its outcome is UNKNOWN; it
  does not confirm that the request was delivered. Authorization headers are never
  recorded and the API key is scrubbed from every string written. A Retry-After header
  is recorded only when it parses to a finite delay. An unexpected ordinary exception
  is replaced by a sanitized UnexpectedClientError. KeyboardInterrupt and SystemExit
  propagate unchanged. If evidence cannot
  be written, EvidenceWriteError is raised and keeps the original API error or the
  successful result.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import httpx2
import openai

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
API_KEY_ENV = "MODEL_API_KEY"

MAX_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 500, 503})
# Documented as non-retryable (dev.meta.ai/docs/error-handling); 504 = gateway_timeout,
# "Do not simply retry". Any status not in RETRY_STATUSES is also not retried.
NO_RETRY_STATUSES = frozenset({400, 401, 402, 403, 404, 413, 504})
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
# If the server asks us to wait longer than this, give up instead of sleeping.
MAX_RETRY_AFTER_S = 60.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_REASONING_EFFORT = "low"
# Accepted on Chat Completions per dev.meta.ai/docs/protocols/chat-completions ("none" -> 400).
REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})

# Response headers safe to persist. Anything else (cookies, auth echoes) is dropped.
SAFE_RESPONSE_HEADERS = (
    "content-type",
    "date",
    "retry-after",
    "x-request-id",
    "request-id",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
)

REDACTED = "[REDACTED]"


# --------------------------------------------------------------------------- errors


class MetaAPIError(Exception):
    """Base class. evidence_path points at the evidence record when one was written."""

    def __init__(self, message: str, *, evidence_path: Path | None = None) -> None:
        super().__init__(message)
        self.evidence_path = evidence_path


class MissingAPIKeyError(MetaAPIError):
    pass


class SchemaError(MetaAPIError):
    """The caller's JSON schema is outside Meta's documented strict subset (no request sent)."""


class MetaAPIStatusError(MetaAPIError):
    def __init__(
        self,
        message: str,
        *,
        status: int,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        evidence_path: Path | None = None,
    ) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.status = status
        self.error_type = error_type
        self.error_code = error_code
        self.request_id = request_id


class InvalidRequestError(MetaAPIStatusError):  # 400, 404, 413, other 4xx
    pass


class AuthenticationError(MetaAPIStatusError):  # 401
    pass


class BillingError(MetaAPIStatusError):  # 402
    pass


class PermissionDeniedError(MetaAPIStatusError):  # 403
    pass


class RateLimitError(MetaAPIStatusError):  # 429 after retries exhausted
    pass


class GatewayTimeoutError(MetaAPIStatusError):  # 504, never retried
    pass


class ServerError(MetaAPIStatusError):  # other 5xx
    pass


class TransportError(MetaAPIError):
    """Transport-level failure (no HTTP response).

    request_possibly_sent is False only when the cause proves the request was never
    sent (connect / pool-acquisition failures); otherwise the outcome is ambiguous and
    the request may have been received and billed.
    """

    def __init__(
        self,
        message: str,
        *,
        request_possibly_sent: bool,
        cause_type: str | None,
        evidence_path: Path | None = None,
    ) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.request_possibly_sent = request_possibly_sent
        self.cause_type = cause_type


class ResponseError(MetaAPIError):
    """HTTP 200 but the response is unusable."""


class MalformedResponseError(ResponseError):
    pass


class UnexpectedStatusError(ResponseError):
    """A non-error HTTP status other than 200 (e.g. 201, 202, 204) reached the success path."""

    def __init__(self, message: str, *, status: int, evidence_path: Path | None = None) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.status = status


class IncompleteResponseError(ResponseError):
    def __init__(self, message: str, *, finish_reason: Any, evidence_path: Path | None = None) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.finish_reason = finish_reason


class EmptyResponseError(ResponseError):
    pass


class StructuredOutputError(ResponseError):
    """Content is not valid JSON or does not conform to the requested schema."""


class UnexpectedClientError(MetaAPIError):
    """Sanitized replacement for an unexpected ordinary Exception raised during a call.

    The original exception is NOT attached (neither __cause__ nor __context__), because
    its message or chain may contain the API key. Only its qualified type name and its
    redacted message are kept; the evidence record holds the same redacted information.
    """

    def __init__(self, message: str, *, original_type: str, evidence_path: Path | None = None) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.original_type = original_type


class EvidenceWriteError(MetaAPIError):
    """Evidence could not be persisted. Never hides what happened to the API call.

    stage:      "initial"         before the first request (no request was sent)
                "attempt_outcome" after an attempt, before any retry/sleep
                "attempt_start"   before a retry request (that request was not sent)
                "final"           after the call finished
    api_error:  the original MetaAPIError of the call/attempt, if there was one
    result:     the ChatResult of a call that succeeded but could not be recorded
    __cause__:  the underlying filesystem/serialisation error
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        evidence_path: Path | None,
        api_error: MetaAPIError | None = None,
        result: Any | None = None,
    ) -> None:
        super().__init__(message, evidence_path=evidence_path)
        self.stage = stage
        self.api_error = api_error
        self.result = result


_STATUS_ERRORS: dict[int, type[MetaAPIStatusError]] = {
    401: AuthenticationError,
    402: BillingError,
    403: PermissionDeniedError,
    429: RateLimitError,
    504: GatewayTimeoutError,
}


def _status_error_class(status: int) -> type[MetaAPIStatusError]:
    if status in _STATUS_ERRORS:
        return _STATUS_ERRORS[status]
    return ServerError if status >= 500 else InvalidRequestError


# ------------------------------------------------------------------ schema subset

# Meta's strict subset (dev.meta.ai/docs/structured-output): root is a plain object
# without anyOf/oneOf/allOf/enum/not; oneOf/allOf nowhere; every object sets
# additionalProperties: false and lists every property in required; nesting depth
# <= 10; <= 5000 properties; <= 120000 chars of strings; <= 1000 enum values;
# <= 200000 nodes after $ref inlining; no recursive $ref.
#
# Meta does not publish a keyword allowlist. We additionally restrict schemas to the
# keywords our local validator (validate_instance) enforces, so a "conforms" verdict
# is never claimed for a keyword we did not check.
MAX_DEPTH = 10
MAX_PROPERTIES = 5000
MAX_STRING_CHARS = 120_000
MAX_ENUM_VALUES = 1000
MAX_EXPANDED_NODES = 200_000

_ANNOTATION_KEYWORDS = {"title", "description", "$defs", "definitions"}
_VALIDATED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "$ref",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}
_ALLOWED_KEYWORDS = _ANNOTATION_KEYWORDS | _VALIDATED_KEYWORDS
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _resolve_ref(root: dict, ref: str) -> dict:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise SchemaError(f"only local '#/...' $ref values are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaError(f"unresolvable $ref {ref!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaError(f"$ref {ref!r} does not point at a schema object")
    return node


def check_strict_schema(schema: dict) -> None:
    """Raise SchemaError if `schema` is outside the documented strict subset."""
    if not isinstance(schema, dict):
        raise SchemaError("schema must be a JSON object")
    for kw in ("anyOf", "oneOf", "allOf", "enum", "not"):
        if kw in schema:
            raise SchemaError(f"root schema must not use {kw!r}")
    if schema.get("type") != "object":
        raise SchemaError("root schema must have type 'object'")

    counts = {"properties": 0, "enum": 0, "chars": 0, "nodes": 0}

    def walk(node: Any, path: str, depth: int, ref_stack: tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            raise SchemaError(f"{path}: schema must be an object")
        counts["nodes"] += 1
        if counts["nodes"] > MAX_EXPANDED_NODES:
            raise SchemaError(f"schema exceeds {MAX_EXPANDED_NODES} nodes after $ref expansion")
        if depth > MAX_DEPTH:
            raise SchemaError(f"{path}: nesting depth exceeds {MAX_DEPTH}")
        unknown = set(node) - _ALLOWED_KEYWORDS
        if "oneOf" in node or "allOf" in node:
            raise SchemaError(f"{path}: oneOf/allOf are not supported")
        if unknown:
            raise SchemaError(f"{path}: unsupported keyword(s) {sorted(unknown)}")
        for text_kw in ("title", "description"):
            if isinstance(node.get(text_kw), str):
                counts["chars"] += len(node[text_kw])

        if "$ref" in node:
            ref = node["$ref"]
            if ref in ref_stack:
                raise SchemaError(f"{path}: recursive $ref {ref!r} is not supported")
            walk(_resolve_ref(schema, ref), f"{path}->{ref}", depth, ref_stack + (ref,))
            return

        types = node.get("type")
        type_list = types if isinstance(types, list) else [types] if types is not None else []
        for t in type_list:
            if t not in _JSON_TYPES:
                raise SchemaError(f"{path}: unknown type {t!r}")
        if not type_list and not any(k in node for k in ("enum", "const", "anyOf")):
            raise SchemaError(f"{path}: schema must declare type, enum, const or anyOf")

        if "enum" in node:
            if not isinstance(node["enum"], list) or not node["enum"]:
                raise SchemaError(f"{path}: enum must be a non-empty list")
            counts["enum"] += len(node["enum"])
            counts["chars"] += sum(len(v) for v in node["enum"] if isinstance(v, str))
        if isinstance(node.get("const"), str):
            counts["chars"] += len(node["const"])

        if "object" in type_list or "properties" in node:
            props = node.get("properties", {})
            if not isinstance(props, dict):
                raise SchemaError(f"{path}: properties must be an object")
            if node.get("additionalProperties") is not False:
                raise SchemaError(f"{path}: objects must set additionalProperties: false")
            required = node.get("required", [])
            if not isinstance(required, list) or set(required) != set(props) or len(required) != len(props):
                raise SchemaError(f"{path}: required must list every key in properties exactly once")
            counts["properties"] += len(props)
            for name, sub in props.items():
                counts["chars"] += len(name)
                walk(sub, f"{path}.{name}", depth + 1, ref_stack)

        if "array" in type_list or "items" in node:
            if "items" not in node:
                raise SchemaError(f"{path}: arrays must define items")
            walk(node["items"], f"{path}[]", depth + 1, ref_stack)

        if "anyOf" in node:
            if not isinstance(node["anyOf"], list) or not node["anyOf"]:
                raise SchemaError(f"{path}: anyOf must be a non-empty list")
            for i, sub in enumerate(node["anyOf"]):
                walk(sub, f"{path}.anyOf[{i}]", depth, ref_stack)

    walk(schema, "$", 1, ())
    for defs_kw in ("$defs", "definitions"):
        for name in schema.get(defs_kw, {}) or {}:
            counts["chars"] += len(name)

    if counts["properties"] > MAX_PROPERTIES:
        raise SchemaError(f"schema has {counts['properties']} properties (max {MAX_PROPERTIES})")
    if counts["enum"] > MAX_ENUM_VALUES:
        raise SchemaError(f"schema has {counts['enum']} enum values (max {MAX_ENUM_VALUES})")
    if counts["chars"] > MAX_STRING_CHARS:
        raise SchemaError(f"schema string content is {counts['chars']} chars (max {MAX_STRING_CHARS})")


def _type_matches(value: Any, t: str) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    return False


def validate_instance(instance: Any, schema: dict, *, _root: dict | None = None, _path: str = "$") -> list[str]:
    """Validate `instance` against a schema accepted by check_strict_schema.

    Returns a list of violations (empty = conforms).
    """
    root = schema if _root is None else _root
    if "$ref" in schema:
        return validate_instance(instance, _resolve_ref(root, schema["$ref"]), _root=root, _path=_path)
    errors: list[str] = []

    if "anyOf" in schema:
        if not any(not validate_instance(instance, s, _root=root, _path=_path) for s in schema["anyOf"]):
            errors.append(f"{_path}: matches no anyOf branch")
            return errors

    types = schema.get("type")
    if types is not None:
        type_list = types if isinstance(types, list) else [types]
        if not any(_type_matches(instance, t) for t in type_list):
            return [f"{_path}: expected type {types}, got {type(instance).__name__}"]

    if "enum" in schema and not any(instance == v and type(instance) is type(v) for v in schema["enum"]):
        errors.append(f"{_path}: {instance!r} not in enum")
    if "const" in schema and not (instance == schema["const"] and type(instance) is type(schema["const"])):
        errors.append(f"{_path}: {instance!r} != const {schema['const']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{_path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{_path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{_path}: string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{_path}: string longer than {schema['maxLength']}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{_path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{_path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors += validate_instance(item, schema["items"], _root=root, _path=f"{_path}[{i}]")
    if isinstance(instance, dict) and ("properties" in schema or "required" in schema):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{_path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in props:
                    errors.append(f"{_path}: unexpected key {key!r}")
        for key, sub in props.items():
            if key in instance:
                errors += validate_instance(instance[key], sub, _root=root, _path=f"{_path}.{key}")
    return errors


# ------------------------------------------------------------------ strict JSON


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON number {name!r} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999 overflows to inf
        raise ValueError(f"JSON number {text!r} is not finite")
    return value


def _unique_keys_object(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    """json.loads that rejects NaN, Infinity, -Infinity, overflowing floats and duplicate keys.

    Applies at every nesting level. Raises ValueError (json.JSONDecodeError is a subclass).
    """
    return json.loads(
        text,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_keys_object,
    )


# ------------------------------------------------------------ transport errors

# Causes that prove the request was never sent. Matched by EXACT type (not isinstance),
# so subclasses and unknown variants are treated as ambiguous. Established from the
# installed sources (httpx2 2.16.0 / httpcore2 2.16.0 / openai 3.42.0):
#   ConnectError, ConnectTimeout: raised only by httpcore2 _backends connect_tcp /
#       start_tls / connect_unix_socket while opening a connection, before any bytes
#       of the HTTP request are written.
#   PoolTimeout: raised only while waiting for a pool connection
#       (wait_for_connection), before connection.handle_request() sends anything.
# The openai SDK wraps these as APIConnectionError/APITimeoutError "from err", so the
# original httpx2 exception is the wrapper's __cause__.
RETRYABLE_TRANSPORT_CAUSES: tuple[type[BaseException], ...] = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.PoolTimeout,
)


def classify_transport_error(exc: BaseException) -> tuple[bool, str | None]:
    """Return (retryable, cause_type) for an SDK APIConnectionError/APITimeoutError.

    retryable is True only when __cause__ is exactly one of RETRYABLE_TRANSPORT_CAUSES.
    Everything else (read/write timeouts and errors, protocol, proxy and TLS errors,
    subclasses, a missing cause) is ambiguous: the request may have been sent.
    """
    cause = exc.__cause__
    if cause is None:
        return False, None
    cause_type = f"{type(cause).__module__}.{type(cause).__qualname__}"
    return type(cause) in RETRYABLE_TRANSPORT_CAUSES, cause_type


# ------------------------------------------------------------------ results / evidence


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ChatResult:
    content: str
    parsed: Any | None
    finish_reason: str
    model: str | None
    response_id: str | None
    request_id: str | None
    usage: Usage
    latency_s: float
    attempts: int
    evidence_path: Path


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_usage(body: dict) -> Usage:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    details = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    return Usage(
        prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
        completion_tokens=_int_or_none(usage.get("completion_tokens")),
        reasoning_tokens=_int_or_none(details.get("reasoning_tokens")) if isinstance(details, dict) else None,
        cached_tokens=_int_or_none(prompt_details.get("cached_tokens")) if isinstance(prompt_details, dict) else None,
        total_tokens=_int_or_none(usage.get("total_tokens")),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_headers(headers: Any) -> dict[str, str]:
    out = {}
    for name in SAFE_RESPONSE_HEADERS:
        value = headers.get(name) if headers is not None else None
        if value is not None:
            out[name] = value
    return out


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP-date)."""
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        # Rejects NaN, +/-inf and overflowing literals such as "1e999" (float() -> inf).
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (when - (now or _utc_now())).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:48] or "call"


def _atomic_replace_with_json(path: Path, data: Any) -> None:
    """Serialise to a unique temp file in the same directory, fsync, then os.replace."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            # allow_nan=False: evidence is strict JSON; a non-finite number is a bug.
            json.dump(data, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path: Path, data: Any) -> None:
    """Create `path` atomically with JSON content. Refuses to overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    _atomic_replace_with_json(path, data)


def replace_json_atomic(path: Path, data: dict, evidence_id: str) -> None:
    """Atomically replace an existing evidence record that we created.

    Refuses if the file is missing or carries a different evidence_id, so an update can
    never clobber an unrelated file.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        current = strict_json_loads(fh.read())
    if not isinstance(current, dict) or current.get("evidence_id") != evidence_id:
        raise ValueError(f"refusing to replace {path.name}: evidence_id mismatch")
    if data.get("evidence_id") != evidence_id:
        raise ValueError("new evidence record has a different evidence_id")
    _atomic_replace_with_json(path, data)


def _scrub(obj: Any, secret: str) -> Any:
    """Replace every occurrence of `secret` in strings (keys included) with [REDACTED]."""
    if isinstance(obj, str):
        return obj.replace(secret, REDACTED) if secret else obj
    if isinstance(obj, dict):
        return {_scrub(k, secret): _scrub(v, secret) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v, secret) for v in obj]
    return obj


# ------------------------------------------------------------------------- client


IN_FLIGHT_SEMANTICS = (
    "An attempt whose state is 'in_flight' has an UNKNOWN outcome: the request may or may "
    "not have been delivered, processed or billed. It is not evidence of delivery."
)


def _safe_str(exc: BaseException) -> str:
    """str(exc), tolerating exceptions whose __str__ itself fails."""
    try:
        return str(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


def _check_model(model: Any) -> str:
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError(f"model must be a non-empty string without surrounding whitespace, got {model!r}")
    return model


def _check_timeout(timeout_s: Any) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ValueError(f"timeout_s must be a positive finite number, got {timeout_s!r}")
    if (isinstance(timeout_s, float) and not math.isfinite(timeout_s)) or timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive finite number, got {timeout_s!r}")
    return float(timeout_s)


class MuseSparkClient:
    """Thin, fail-loud Chat Completions client for Muse Spark.

    api_key:      explicit key, else os.environ["MODEL_API_KEY"]. .env files are not read here.
    evidence_dir: directory for per-call evidence records (e.g. runs/api_evidence).
    http_client:  optional httpx2.Client, used by offline tests to inject a fake transport.
    sleep / rng:  injectable for deterministic offline tests.
    """

    def __init__(
        self,
        *,
        evidence_dir: str | Path,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        http_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        key = (key or "").strip()
        if not key:
            raise MissingAPIKeyError(
                f"no API key: pass api_key= or set the {API_KEY_ENV} environment variable"
            )
        # The SDK substitutes its own default endpoint (or OPENAI_BASE_URL) when base_url is
        # None/empty. Require an explicit https URL so that can never happen silently.
        if not isinstance(base_url, str) or not base_url.strip().startswith("https://"):
            raise ValueError(f"base_url must be an explicit https:// URL, got {base_url!r}")
        _check_model(model)
        timeout = _check_timeout(timeout_s)
        self.__key = key
        self.model = model
        self.base_url = base_url.strip()
        self.evidence_dir = Path(evidence_dir)
        self._sleep = sleep
        self._transport = "network" if http_client is None else "injected"
        self._rng = rng or random.Random()
        self._client = openai.OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,  # retries are ours, see _run
            http_client=http_client,
        )
        # Defence in depth: confirm the SDK resolved exactly the endpoint we asked for
        # (environment variables or SDK defaults must not override it).
        resolved = str(self._client.base_url).rstrip("/")
        if resolved != self.base_url.rstrip("/"):
            raise ValueError(f"SDK resolved base_url {resolved!r}, expected {self.base_url!r}")

    def __repr__(self) -> str:
        return f"MuseSparkClient(model={self.model!r}, base_url={self.base_url!r}, api_key={REDACTED})"

    # -- public API ---------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str,
        max_completion_tokens: int,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        json_schema: dict | None = None,
        schema_name: str = "response",
        model: str | None = None,
    ) -> ChatResult:
        """Send one chat completion. Raises a MetaAPIError subclass on any failure."""
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0
        ):
            raise ValueError("max_completion_tokens must be a positive int (bool is not accepted)")
        request_model = _check_model(self.model if model is None else model)
        if json_schema is not None:
            check_strict_schema(json_schema)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", schema_name):
                raise SchemaError("schema_name must match [A-Za-z0-9_-]{1,64}")

        request: dict[str, Any] = {
            "model": request_model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        }
        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            }

        started = _utc_now()
        evidence_path = self._new_evidence_path(started, purpose)
        record: dict[str, Any] = {
            "evidence_version": 2,
            "evidence_id": uuid.uuid4().hex,
            "kind": "chat_completion_call",
            # "network" = SDK default transport (a real API call); "injected" = caller-supplied
            # http_client, e.g. the offline fake transport in tests. Injected records are not
            # integration evidence.
            "transport": self._transport,
            "purpose": purpose,
            "created_utc": started.isoformat(),
            "endpoint": f"{self.base_url.rstrip('/')}/chat/completions",
            "request": request,
            "retry_policy": {
                "max_attempts": MAX_ATTEMPTS,
                "retry_statuses": sorted(RETRY_STATUSES),
                "no_retry_statuses": sorted(NO_RETRY_STATUSES),
                "retryable_transport_causes": [
                    f"{t.__module__}.{t.__qualname__}" for t in RETRYABLE_TRANSPORT_CAUSES
                ],
                "sdk_max_retries": 0,
            },
            "in_flight_semantics": IN_FLIGHT_SEMANTICS,
            "state": "in_progress",
            "attempts": [],
            "outcome": None,
        }
        try:
            result = self._run(request, json_schema, record, evidence_path)
        except EvidenceWriteError:
            # Raised inside _run; already carries the attempt's api_error and is redacted.
            raise
        except MetaAPIError as exc:
            exc.evidence_path = evidence_path
            exc.args = tuple(self._redact(a) if isinstance(a, str) else a for a in exc.args)
            record["state"] = "failed"
            record["outcome"] = {
                "ok": False,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
            }
            # On failure this raises EvidenceWriteError(api_error=exc). exc also stays in
            # the chain: EvidenceWriteError.__cause__ (the OSError) has exc as __context__.
            self._persist(record, evidence_path, stage="final", api_error=exc)
            raise
        except Exception as exc:  # ordinary unexpected failure (a bug, an unmapped library error)
            # Its message or chain may contain the API key, so the caller gets a sanitized
            # replacement. It is raised AFTER this block so the original is attached neither
            # as __cause__ nor as __context__.
            original_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            message = self._redact(f"unexpected {original_type}: {_safe_str(exc)}")
            unexpected = UnexpectedClientError(message, original_type=original_type, evidence_path=evidence_path)
            self._mark_failed(record, type(exc).__name__, self._redact(_safe_str(exc)))
            record["outcome"]["raised_as"] = UnexpectedClientError.__name__
        except BaseException as exc:  # KeyboardInterrupt, SystemExit, ...: re-raised unchanged
            self._mark_failed(record, type(exc).__name__, self._redact(_safe_str(exc)))
            try:
                self._persist(record, evidence_path, stage="final")
            except EvidenceWriteError as write_exc:
                # Never replace or swallow the original (e.g. KeyboardInterrupt, SystemExit).
                exc.add_note(str(write_exc))
            raise
        else:
            unexpected = None
        if unexpected is not None:
            # Outside the except block: no implicit __context__. On a write failure this
            # raises EvidenceWriteError(api_error=unexpected), equally free of the original.
            self._persist(record, evidence_path, stage="final", api_error=unexpected)
            raise unexpected
        record["state"] = "succeeded"
        record["outcome"] = {
            "ok": True,
            "finish_reason": result.finish_reason,
            "usage": asdict(result.usage),
            "response_id": result.response_id,
            "request_id": result.request_id,
            "total_latency_s": result.latency_s,
            "attempts": result.attempts,
        }
        self._persist(record, evidence_path, stage="final", result=result)
        return result

    # -- evidence -----------------------------------------------------------------

    def _new_evidence_path(self, started: datetime, purpose: str) -> Path:
        stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
        return self.evidence_dir / f"{stamp}_{_slug(purpose)}_{uuid.uuid4().hex[:12]}.json"

    def _redact(self, text: str) -> str:
        return _scrub(text, self.__key)

    @staticmethod
    def _mark_failed(record: dict, error_class: str, redacted_message: str) -> None:
        """Mark the call failed; attempts still in flight get an unknown outcome."""
        for entry in record["attempts"]:
            if entry.get("state") == "in_flight":
                entry["state"] = "interrupted_outcome_unknown"
        record["state"] = "failed"
        record["outcome"] = {"ok": False, "error_class": error_class, "error_message": redacted_message}

    def _persist(
        self,
        record: dict,
        path: Path,
        *,
        stage: str,
        api_error: MetaAPIError | None = None,
        result: ChatResult | None = None,
    ) -> None:
        """Atomically create (stage "initial") or replace the evidence record.

        Any failure becomes EvidenceWriteError carrying the original API outcome.
        """
        try:
            data = _scrub(record, self.__key)
            if stage == "initial":
                write_json_atomic(path, data)
            else:
                replace_json_atomic(path, data, record["evidence_id"])
        except (OSError, ValueError, TypeError) as write_exc:
            if api_error is not None:
                original = f"{type(api_error).__name__}: {api_error}"
            elif result is not None:
                original = f"call succeeded (response_id={result.response_id}) but was not recorded"
            elif stage == "initial":
                original = "no request was sent"
            else:
                original = "none recorded"
            if stage == "attempt_start":
                original += "; the retry request was not sent"
            message = (
                f"evidence write failed at stage {stage!r} "
                f"({type(write_exc).__name__}: {write_exc}); original outcome: {original}"
            )
            raise EvidenceWriteError(
                self._redact(message),
                stage=stage,
                evidence_path=path,
                api_error=api_error,
                result=result,
            ) from write_exc

    # -- retries ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
        return base + self._rng.uniform(0, base)

    def _retry_delay(self, entry: dict, attempt: int, retryable: bool, retry_after: float | None) -> float | None:
        """Record the retry decision in `entry`; return the delay, or None to stop."""
        entry["retry_after_s"] = retry_after
        if not retryable or attempt >= MAX_ATTEMPTS:
            entry["retry"] = False
            entry["retry_reason"] = "not retryable" if not retryable else "attempts exhausted"
            return None
        if retry_after is not None and retry_after > MAX_RETRY_AFTER_S:
            entry["retry"] = False
            entry["retry_reason"] = f"Retry-After {retry_after:.1f}s exceeds {MAX_RETRY_AFTER_S:.0f}s limit"
            return None
        delay = self._backoff(attempt)
        if retry_after is not None:
            delay = max(delay, retry_after)
        entry["retry"] = True
        entry["sleep_s"] = round(delay, 4)
        return delay

    def _run(self, request: dict, json_schema: dict | None, record: dict, evidence_path: Path) -> ChatResult:
        t0 = time.monotonic()
        previous_error: MetaAPIError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            entry: dict[str, Any] = {"attempt": attempt, "state": "in_flight", "started_utc": _utc_now().isoformat()}
            record["attempts"].append(entry)
            # Persist the in-flight attempt BEFORE its request is sent.
            self._persist(
                record,
                evidence_path,
                stage="initial" if attempt == 1 else "attempt_start",
                api_error=previous_error,
            )
            a0 = time.monotonic()
            try:
                raw = self._client.chat.completions.with_raw_response.create(**request)
            except openai.APIStatusError as exc:
                entry["latency_s"] = round(time.monotonic() - a0, 4)
                entry["state"] = "response_received"
                self._record_http(entry, exc.response)
                status = exc.status_code
                err = _error_envelope(entry.get("response_json"))
                retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
                retryable = status in RETRY_STATUSES and status not in NO_RETRY_STATUSES
                error: MetaAPIError = _status_error_class(status)(
                    self._redact(
                        f"HTTP {status}"
                        + (f" {err.get('type')}" if err.get("type") else "")
                        + (f" ({err.get('code')})" if err.get("code") else "")
                        + (f": {err.get('message')}" if err.get("message") else "")
                    ),
                    status=status,
                    error_type=err.get("type"),
                    error_code=err.get("code"),
                    request_id=entry.get("request_id"),
                )
            except openai.APIConnectionError as exc:  # includes APITimeoutError
                entry["latency_s"] = round(time.monotonic() - a0, 4)
                entry["state"] = "transport_error"
                retryable, cause_type = classify_transport_error(exc)
                entry["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                entry["cause_type"] = cause_type
                entry["transport_classification"] = "not_sent_retryable" if retryable else "ambiguous_not_retried"
                entry["request_possibly_sent"] = not retryable
                retry_after = None
                error = TransportError(
                    self._redact(f"{type(exc).__name__} (cause: {cause_type}): {exc}"),
                    request_possibly_sent=not retryable,
                    cause_type=cause_type,
                )
            else:
                entry["latency_s"] = round(time.monotonic() - a0, 4)
                entry["state"] = "response_received"
                self._record_http(entry, raw.http_response)
                entry["retry"] = False
                status = raw.http_response.status_code
                if status != 200:
                    raise UnexpectedStatusError(
                        f"HTTP {status} reached the success path; only 200 is accepted", status=status
                    )
                return self._parse_success(
                    entry, json_schema, attempt, round(time.monotonic() - t0, 4), evidence_path
                )

            delay = self._retry_delay(entry, attempt, retryable, retry_after)
            if delay is None:
                raise error from None
            # Persist this attempt's outcome BEFORE sleeping or retrying.
            self._persist(record, evidence_path, stage="attempt_outcome", api_error=error)
            self._sleep(delay)
            previous_error = error
        raise AssertionError("unreachable")  # pragma: no cover

    # -- parsing ------------------------------------------------------------------

    @staticmethod
    def _record_http(entry: dict, response: Any) -> None:
        entry["http_status"] = response.status_code
        headers = _safe_headers(response.headers)
        if "retry-after" in headers:
            # Only a Retry-After that parses to a finite delay is recorded. An invalid value
            # (NaN, inf, 1e999, garbage) is dropped from evidence entirely; only the fact
            # that it was invalid is kept, never its contents.
            invalid = parse_retry_after(headers["retry-after"]) is None
            if invalid:
                del headers["retry-after"]
            entry["retry_after_header_invalid"] = invalid
        entry["response_headers"] = headers
        entry["request_id"] = response.headers.get("x-request-id") or response.headers.get("request-id")
        text = response.text
        entry["response_body_raw"] = text
        try:
            entry["response_json"] = strict_json_loads(text)
        except ValueError as exc:
            entry["response_json"] = None
            entry["response_json_error"] = str(exc)[:300]

    def _parse_success(
        self, entry: dict, json_schema: dict | None, attempts: int, latency_s: float, evidence_path: Path
    ) -> ChatResult:
        body = entry.get("response_json")
        if not isinstance(body, dict):
            detail = entry.get("response_json_error")
            raise MalformedResponseError(
                "HTTP 200 body is not a strict JSON object" + (f": {detail}" if detail else "")
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise MalformedResponseError("expected exactly one choice in response")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError("choice has no message object")
        finish_reason = choice.get("finish_reason")
        if finish_reason != "stop":
            raise IncompleteResponseError(
                f"finish_reason is {finish_reason!r}, expected 'stop'", finish_reason=finish_reason
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise EmptyResponseError("response content is empty")

        parsed = None
        if json_schema is not None:
            try:
                parsed = strict_json_loads(content)
            except ValueError as exc:
                raise StructuredOutputError(f"content is not valid strict JSON: {exc}") from None
            violations = validate_instance(parsed, json_schema)
            entry["schema_violations"] = violations
            if violations:
                raise StructuredOutputError("response violates schema: " + "; ".join(violations[:10]))

        return ChatResult(
            content=content,
            parsed=parsed,
            finish_reason=finish_reason,
            model=body.get("model") if isinstance(body.get("model"), str) else None,
            response_id=body.get("id") if isinstance(body.get("id"), str) else None,
            request_id=entry.get("request_id"),
            usage=parse_usage(body),
            latency_s=latency_s,
            attempts=attempts,
            evidence_path=evidence_path,
        )


def _error_envelope(body: Any) -> dict:
    """Extract {"message","type","param","code"} from Meta's documented error envelope."""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return {k: v for k, v in body["error"].items() if isinstance(v, (str, int)) or v is None}
    return {}
