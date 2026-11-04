"""B1b-2b: Muse Spark scene interpretation of a validated trajectory set.

One entry point, interpret_scene(), connects the B1b-2a prompt, the B1b-1 schema and
validator, and the B1a client:

    TrajectorySet
      -> build_scene_prompt()              developer + user messages, input hash
      -> build_interpretation_schema()     strict schema, object_id enum from the input
      -> MuseSparkClient.chat()            exactly once; B1a evidence record written
      -> validate_interpretation()         schema + semantic rules, exact event windows
      -> parse_interpretation()            SceneInterpretation
      -> validation sidecar                atomic, one per call, never overwritten
      -> SceneInterpretationResult         or a typed error

There is no repair or retry at this layer. The B1a client applies its own transport
retry policy; an invalid interpretation is reported, not re-requested.

A validation sidecar is written for every outcome once chat() has been entered:
accepted, rejected by semantic validation, API call failed, or interrupted. It binds
the outcome to the B1a evidence record (path and SHA-256) and to the model output
(response_content_sha256) without duplicating the interpretation. It records
`response_provenance` copied from the B1a evidence `transport` field: "injected" means
a caller-supplied HTTP transport (an offline test double), never a Meta API result.

Passing validation means the output is well formed and consistent with the cited
observations. It does not establish that any hypothesis is physically correct.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from decompile import scene_interpretation as si
from decompile.meta_api import (
    REASONING_EFFORTS,
    ChatResult,
    EvidenceWriteError,
    IncompleteResponseError,
    MetaAPIError,
    MetaAPIStatusError,
    MuseSparkClient,
    TransportError,
    Usage,
    strict_json_loads,
    write_json_atomic,
)
from decompile.scene_prompt import build_scene_prompt, canonical_json, sha256_text
from decompile.trajectories import TrajectorySet, timestamp_to_json

SIDECAR_VERSION = 1
SIDECAR_KIND = "scene_interpretation_validation"
PURPOSE = "b1b-scene-interpretation"
DEFAULT_MAX_COMPLETION_TOKENS = 8192
DEFAULT_REASONING_EFFORT = "low"

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected_semantic_validation"
STATUS_API_FAILED = "api_call_failed"
STATUS_INTERRUPTED = "interrupted"

PHYSICAL_CORRECTNESS_NOTE = (
    "Passing schema and semantic validation means the interpretation is well formed and "
    "consistent with the cited observations. It does not establish that any hypothesis is "
    "physically correct; physical parameters are fitted and tested by the numerical pipeline."
)

PROVENANCE_NOTES = {
    "network": "B1a evidence reports the SDK default network transport: a real Meta API call.",
    "injected": "B1a evidence reports a caller-injected HTTP transport (offline test double). "
    "This is NOT a Meta API result.",
    "unknown": "B1a evidence was unavailable or unreadable; the response provenance is unknown.",
}


# ------------------------------------------------------------------ errors


class SceneInterpreterError(Exception):
    """Base for interpret_scene failures that are not semantic rejections."""

    def __init__(self, message: str, *, evidence_path: Path | None, sidecar_path: Path | None) -> None:
        super().__init__(message)
        self.evidence_path = evidence_path
        self.sidecar_path = sidecar_path


class InterpretationAPIError(SceneInterpreterError):
    """MuseSparkClient.chat() raised. api_error is the original (already redacted) MetaAPIError.

    Schema-violating, truncated (finish_reason != "stop") and empty responses arrive here,
    as StructuredOutputError / IncompleteResponseError / EmptyResponseError from B1a.
    """

    def __init__(self, message: str, *, api_error: MetaAPIError, evidence_path: Path | None,
                 sidecar_path: Path | None) -> None:
        super().__init__(message, evidence_path=evidence_path, sidecar_path=sidecar_path)
        self.api_error = api_error


class InterpretationRejectedError(si.SceneInterpretationError):
    """The response conformed to the schema but failed semantic validation.

    A subclass of the B1b-1 SceneInterpretationError (errors, warnings), with the paths of
    the B1a evidence record and the validation sidecar.
    """

    def __init__(self, errors: list[str], warnings: list[str], *, evidence_path: Path,
                 sidecar_path: Path) -> None:
        super().__init__(errors, warnings)
        self.evidence_path = evidence_path
        self.sidecar_path = sidecar_path


class SidecarWriteError(SceneInterpreterError):
    """The validation sidecar could not be written. Never hides the interpretation outcome.

    status:       the outcome that was being recorded (STATUS_*)
    api_error:    the original MetaAPIError, for STATUS_API_FAILED
    result:       the SceneInterpretationResult that would have been returned, for
                  STATUS_ACCEPTED (its sidecar_path names the file that was NOT written)
    errors / warnings: the validation findings, for STATUS_REJECTED
    __cause__:    the underlying filesystem or serialisation error
    """

    def __init__(self, message: str, *, status: str, evidence_path: Path | None, sidecar_path: Path,
                 api_error: MetaAPIError | None = None, result: "SceneInterpretationResult | None" = None,
                 errors: tuple[str, ...] = (), warnings: tuple[str, ...] = ()) -> None:
        super().__init__(message, evidence_path=evidence_path, sidecar_path=sidecar_path)
        self.status = status
        self.api_error = api_error
        self.result = result
        self.errors = tuple(errors)
        self.warnings = tuple(warnings)


# ------------------------------------------------------------------ result


@dataclass(frozen=True)
class SceneInterpretationResult:
    interpretation: si.SceneInterpretation
    evidence_path: Path
    sidecar_path: Path
    response_provenance: str  # "network" | "injected" | "unknown", from B1a evidence
    prompt_version: str
    developer_prompt_sha256: str
    messages_sha256: str
    input_sha256: str
    schema_sha256: str
    response_content_sha256: str
    requested_model: str
    model: str | None
    response_id: str | None
    request_id: str | None
    usage: Usage
    attempts: int
    latency_s: float

    @property
    def event_windows(self) -> tuple[si.EventWindow, ...]:
        return self.interpretation.event_windows

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.interpretation.warnings


# ------------------------------------------------------------------ helpers


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_sidecar_path(sidecar_dir: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    return sidecar_dir / f"{stamp}_scene-interpretation_{uuid.uuid4().hex[:12]}.json"


def _evidence_info(evidence_path: Path | None) -> tuple[str | None, str]:
    """(SHA-256 of the evidence file bytes, transport) from a B1a evidence record.

    Never raises: an absent or unreadable record gives (None or hash, "unknown").
    """
    if evidence_path is None:
        return None, "unknown"
    try:
        raw = Path(evidence_path).read_bytes()
    except OSError:
        return None, "unknown"
    digest = hashlib.sha256(raw).hexdigest()
    try:
        record = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return digest, "unknown"
    transport = record.get("transport") if isinstance(record, dict) else None
    return digest, transport if transport in ("network", "injected") else "unknown"


def _window_json(window: si.EventWindow) -> dict:
    return {
        "event_id": window.event_id,
        "after_frame_index": window.after_frame_index,
        "before_frame_index": window.before_frame_index,
        "t_after": timestamp_to_json(window.t_after),
        "t_before": timestamp_to_json(window.t_before),
        "interior_frame_indices": list(window.interior_frame_indices),
    }


def _api_error_json(exc: MetaAPIError) -> dict:
    """Sanitized error details. Messages of B1a errors are already redacted by the client."""
    out: dict[str, Any] = {"error_class": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, MetaAPIStatusError):
        out.update(status=exc.status, error_type=exc.error_type, error_code=exc.error_code,
                   request_id=exc.request_id)
    if isinstance(exc, IncompleteResponseError):
        out["finish_reason"] = exc.finish_reason if isinstance(exc.finish_reason, (str, type(None))) else repr(
            exc.finish_reason)
    if isinstance(exc, TransportError):
        out.update(request_possibly_sent=exc.request_possibly_sent, cause_type=exc.cause_type)
    if isinstance(exc, EvidenceWriteError):
        out["stage"] = exc.stage
        out["wrapped_api_error_class"] = type(exc.api_error).__name__ if exc.api_error is not None else None
        out["call_succeeded_but_unrecorded"] = exc.result is not None
    return out


def _check_call_settings(max_completion_tokens: Any, reasoning_effort: Any) -> None:
    """Reject bad settings before the API call is entered (same rules as B1a chat())."""
    if (isinstance(max_completion_tokens, bool) or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0):
        raise ValueError("max_completion_tokens must be a positive int (bool is not accepted)")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")


# ------------------------------------------------------------------ main entry point


def interpret_scene(
    traj: TrajectorySet,
    client: MuseSparkClient,
    *,
    sidecar_dir: str | Path,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> SceneInterpretationResult:
    """Interpret `traj` with Muse Spark: one chat() call, validation, one sidecar.

    Raises before any API call (no sidecar): TypeError / TrajectoryError for an invalid
    trajectory set, ValueError for invalid call settings.
    After chat() is entered, exactly one sidecar is written and one of these happens:
      returns SceneInterpretationResult       (status "accepted")
      raises InterpretationRejectedError      (status "rejected_semantic_validation")
      raises InterpretationAPIError           (status "api_call_failed")
      re-raises KeyboardInterrupt/SystemExit  (status "interrupted", best effort)
      raises SidecarWriteError                (the sidecar itself could not be written)
    """
    prompt = build_scene_prompt(traj)  # re-validates traj; raises before any call
    schema = si.build_interpretation_schema(traj)
    schema_sha256 = sha256_text(canonical_json(schema))
    _check_call_settings(max_completion_tokens, reasoning_effort)
    sidecar_dir = Path(sidecar_dir)
    requested_model = client.model

    started = _utc_now()
    sidecar_path = _new_sidecar_path(sidecar_dir, started)
    record: dict[str, Any] = {
        "sidecar_version": SIDECAR_VERSION,
        "kind": SIDECAR_KIND,
        "created_utc": started.isoformat(),
        "status": None,
        "physical_correctness_note": PHYSICAL_CORRECTNESS_NOTE,
        "response_provenance": "unknown",
        "response_provenance_note": PROVENANCE_NOTES["unknown"],
        "prompt_version": prompt.prompt_version,
        "developer_prompt_sha256": prompt.developer_prompt_sha256,
        "messages_sha256": prompt.messages_sha256,
        "input_sha256": prompt.input_sha256,
        "schema_version": si.SCHEMA_VERSION,
        "schema_name": si.SCHEMA_NAME,
        "schema_sha256": schema_sha256,
        "input_object_ids": list(traj.object_ids),
        # Provenance of the input. Deliberately NOT sent to the model (see scene_prompt).
        "input_provenance": {"synthetic": traj.synthetic, "source": traj.source, "sent_to_model": False},
        "trajectory_warnings": list(traj.warnings),
        "call": {
            "purpose": PURPOSE,
            "requested_model": requested_model,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        },
        "api_evidence_path": None,
        "api_evidence_sha256": None,
        "response_content_sha256": None,
        "response": None,
        "api_error": None,
        "validation": None,
        "event_windows": [],
    }

    def write(status: str) -> None:
        record["status"] = status
        write_json_atomic(sidecar_path, record)

    def note_evidence(evidence_path: Path | None) -> None:
        digest, provenance = _evidence_info(evidence_path)
        record["api_evidence_path"] = str(evidence_path) if evidence_path is not None else None
        record["api_evidence_sha256"] = digest
        record["response_provenance"] = provenance
        record["response_provenance_note"] = PROVENANCE_NOTES[provenance]

    # -- the single API call ---------------------------------------------------------
    try:
        chat: ChatResult = client.chat(
            prompt.messages,
            purpose=PURPOSE,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            json_schema=schema,
            schema_name=si.SCHEMA_NAME,
        )
    except MetaAPIError as exc:
        evidence_path = exc.evidence_path
        note_evidence(evidence_path)
        record["api_error"] = _api_error_json(exc)
        try:
            write(STATUS_API_FAILED)
        except (OSError, ValueError, TypeError) as write_exc:
            raise SidecarWriteError(
                f"sidecar write failed ({type(write_exc).__name__}: {write_exc}); "
                f"original outcome: {type(exc).__name__}: {exc}",
                status=STATUS_API_FAILED, evidence_path=evidence_path, sidecar_path=sidecar_path, api_error=exc,
            ) from write_exc
        raise InterpretationAPIError(
            f"Muse Spark call failed: {type(exc).__name__}: {exc}",
            api_error=exc, evidence_path=evidence_path, sidecar_path=sidecar_path,
        ) from exc
    except (KeyboardInterrupt, SystemExit) as exc:  # recorded best effort, re-raised unchanged
        record["api_error"] = {"error_class": type(exc).__name__, "message": None}
        try:
            write(STATUS_INTERRUPTED)
        except Exception as write_exc:
            exc.add_note(f"scene interpretation sidecar not written: {type(write_exc).__name__}: {write_exc}")
        raise

    # -- validation ------------------------------------------------------------------
    evidence_path = chat.evidence_path
    note_evidence(evidence_path)
    content_sha256 = sha256_text(chat.content)
    record["response_content_sha256"] = content_sha256
    record["response"] = {
        "model": chat.model,
        "response_id": chat.response_id,
        "request_id": chat.request_id,
        "finish_reason": chat.finish_reason,
        "usage": asdict(chat.usage),
        "attempts": chat.attempts,
        "latency_s": chat.latency_s,
    }
    report = si.validate_interpretation(chat.parsed, traj)
    record["validation"] = {"ok": report.ok, "errors": list(report.errors), "warnings": list(report.warnings)}

    if not report.ok:
        try:
            write(STATUS_REJECTED)
        except (OSError, ValueError, TypeError) as write_exc:
            raise SidecarWriteError(
                f"sidecar write failed ({type(write_exc).__name__}: {write_exc}); original outcome: "
                f"interpretation rejected with {len(report.errors)} validation error(s)",
                status=STATUS_REJECTED, evidence_path=evidence_path, sidecar_path=sidecar_path,
                errors=report.errors, warnings=report.warnings,
            ) from write_exc
        raise InterpretationRejectedError(
            list(report.errors), list(report.warnings), evidence_path=evidence_path, sidecar_path=sidecar_path
        )

    interpretation = si.parse_interpretation(chat.parsed, traj)
    record["event_windows"] = [_window_json(w) for w in interpretation.event_windows]
    result = SceneInterpretationResult(
        interpretation=interpretation,
        evidence_path=evidence_path,
        sidecar_path=sidecar_path,
        response_provenance=record["response_provenance"],
        prompt_version=prompt.prompt_version,
        developer_prompt_sha256=prompt.developer_prompt_sha256,
        messages_sha256=prompt.messages_sha256,
        input_sha256=prompt.input_sha256,
        schema_sha256=schema_sha256,
        response_content_sha256=content_sha256,
        requested_model=requested_model,
        model=chat.model,
        response_id=chat.response_id,
        request_id=chat.request_id,
        usage=chat.usage,
        attempts=chat.attempts,
        latency_s=chat.latency_s,
    )
    try:
        write(STATUS_ACCEPTED)
    except (OSError, ValueError, TypeError) as write_exc:
        raise SidecarWriteError(
            f"sidecar write failed ({type(write_exc).__name__}: {write_exc}); original outcome: "
            f"interpretation accepted (response_id={chat.response_id}) but not recorded",
            status=STATUS_ACCEPTED, evidence_path=evidence_path, sidecar_path=sidecar_path, result=result,
            warnings=interpretation.warnings,
        ) from write_exc
    return result
