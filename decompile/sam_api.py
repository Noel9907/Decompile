"""SAM 3.1 client for the Meta Model API: video upload + streamed Responses segmentation (decompile.sam_api/1).

Contract (dev.meta.ai/docs/sam/overview, /sam/segmenting, /sam/reading-segmentation, /cookbook/sam-api-basics,
/api-reference/files/upload-file, /pricing-rate-limits; read 2026-11-04):

  base URL     https://api.meta.ai/v1            auth  Authorization: Bearer $MODEL_API_KEY
  upload       POST /files  multipart: purpose=user_data, file=<the MP4 BYTES>  -> {"id": "file-...", ...}
  inference    POST /responses  {"model": "sam-3.1", "stream": true, "metadata": {...},
                                 "input": [{"type": "message", "role": "user", "content": [
                                     {"type": "input_text", "text": <concept phrase OR object prompt>},
                                     {"type": "input_video", "file_id": <id>}]}]}
  metadata     include_confidence "true"/"false" (strings); propagation_direction "both"/"forward"/"backward"
               (object prompts only)
  prompts      EITHER one short noun phrase (concept) OR box/point object prompts built with the official
               meta_sam_parser.build_object_prompt; the two are never combined (the API answers 400).
  output       one output_text line per emitted frame; decoded ONLY with the official meta_sam_parser
               (parse_responses_stream + video_segmentation_format). Frame indices are zero-based source
               frames, may skip and need not ascend.
  billing      video: USD 0.20 per 1,000 processed frames, minimum 50 frames per video request;
               usage.video_frames_processed reports the billed frames.

Safety rules of this module:
  * No request is ever made unless a SamClient method is called; tests inject an httpx2 transport.
  * The API key comes from the constructor or the MODEL_API_KEY environment variable; it is never logged,
    printed or written (every evidence string is scrubbed).
  * Retries only when the request provably did not reach the server (connect errors) or on HTTP 429
    (honouring Retry-After). A billed video request is NEVER retried after it was sent (5xx, read
    timeouts and broken streams are reported, not retried), so a failure cannot double-bill.
  * Terminal outcomes are explicit: completed / incomplete (kept, flagged) / failed / refused / error.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx2
import meta_sam_parser as msp

from decompile import meta_api

CLIENT_VERSION = "decompile.sam_api/1"
BASE_URL = meta_api.BASE_URL
API_KEY_ENV = meta_api.API_KEY_ENV
SAM_MODEL = "sam-3.1"
FILE_PURPOSE = "user_data"
UPLOAD_MAX_BYTES = 1_073_741_824  # Files API limit (1 GiB)
MAX_FRAMES_PER_REQUEST = 15_000
MAX_PROMPTED_OBJECTS = 16
BILLING_MIN_FRAMES = 50
PRICE_USD_PER_1000_FRAMES = 0.20
PRICING_SOURCE = "https://dev.meta.ai/docs/pricing-rate-limits (SAM pricing; read 2026-11-04)"
PROPAGATION_DIRECTIONS = ("both", "forward", "backward")
MAX_CONCEPT_WORDS = 6
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 0.5
MAX_RETRY_AFTER_S = 60.0
CONNECT_TIMEOUT_S = 15.0
UPLOAD_TIMEOUT_S = 600.0
STREAM_READ_TIMEOUT_S = 300.0  # longest silence tolerated between two stream chunks
REDACTED = "[REDACTED]"

OUTCOME_COMPLETED, OUTCOME_INCOMPLETE = "completed", "incomplete"


# ------------------------------------------------------------------ errors


class SamAPIError(Exception):
    """Base class. `kind` is a stable machine-readable category."""

    kind = "sam_error"

    def __init__(self, message: str, *, evidence_path: Path | None = None) -> None:
        super().__init__(message)
        self.evidence_path = evidence_path


class MissingAPIKeyError(SamAPIError):
    kind = "missing_api_key"


class PromptError(SamAPIError, ValueError):
    kind = "invalid_prompt"


class UploadError(SamAPIError):
    kind = "upload_error"


class SamHTTPError(SamAPIError):
    kind = "http_error"

    def __init__(self, message: str, *, status: int, code: str | None = None, evidence_path: Path | None = None):
        super().__init__(message, evidence_path=evidence_path)
        self.status, self.code = status, code


class SamTransportError(SamAPIError):
    kind = "transport_error"

    def __init__(self, message: str, *, sent: bool | None, evidence_path: Path | None = None):
        super().__init__(message, evidence_path=evidence_path)
        self.sent = sent  # False: provably never sent; None: unknown (may have been processed and billed)


class SamStreamError(SamAPIError):
    """The response stream ended in failure: failed / refused / error event / lane or parser violation."""

    kind = "stream_error"

    def __init__(self, message: str, *, reason: str, evidence_path: Path | None = None):
        super().__init__(message, evidence_path=evidence_path)
        self.reason = reason  # response_failed | response_refusal | response_error | lane | parser | source


# ------------------------------------------------------------------ prompts


@dataclass(frozen=True)
class ConceptPrompt:
    """One short noun phrase (e.g. "red ball"). Lists, negations and questions are refused locally."""

    text: str

    def __post_init__(self) -> None:
        t = self.text.strip() if isinstance(self.text, str) else ""
        if not t:
            raise PromptError("a concept prompt must be a non-empty noun phrase")
        if any(c in t for c in ",;+\n<>|"):
            raise PromptError("a concept prompt is ONE noun phrase: no lists (',', ';', '+') or prompt tokens")
        if len(t.split()) > MAX_CONCEPT_WORDS:
            raise PromptError(f"a concept prompt should be a short noun phrase (<= {MAX_CONCEPT_WORDS} words)")
        object.__setattr__(self, "text", t)

    def input_text(self) -> str:
        return self.text

    def metadata(self) -> dict[str, str]:
        return {}

    def to_json_dict(self) -> dict:
        return {"kind": "concept", "text": self.text}


@dataclass(frozen=True)
class ObjectPrompt:
    """Box / point prompts for objects the user has already located (pixel coordinates of the DISPLAYED media).

    boxes are half-open (x0, y0, x1, y1); the official builder writes SAM's inclusive x2 = x1 - 1, y2 = y1 - 1.
    objects: ((object_id, frame, box | None, ((x, y, "positive"|"negative"), ...)), ...)
    """

    size: tuple[int, int]  # displayed (width, height), after rotation
    objects: tuple[tuple[int, int, tuple[int, int, int, int] | None, tuple[tuple[int, int, str], ...]], ...]
    propagation_direction: str = "both"

    def __post_init__(self) -> None:
        if self.propagation_direction not in PROPAGATION_DIRECTIONS:
            raise PromptError(f"propagation_direction must be one of {PROPAGATION_DIRECTIONS}")
        ids = {o[0] for o in self.objects}
        if len(ids) > MAX_PROMPTED_OBJECTS:
            raise PromptError(f"at most {MAX_PROMPTED_OBJECTS} prompted objects")
        if self.propagation_direction == "backward" and not any(o[1] > 0 for o in self.objects):
            raise PromptError("backward propagation needs an object prompt on a frame after the first")
        self.input_text()  # validates every coordinate with the official builder

    def _prompt_objects(self) -> list[msp.PromptObject]:
        by_id: dict[int, list[msp.FramePrompt]] = {}
        for oid, frame, box, points in self.objects:
            fp = msp.FramePrompt(frame=frame, box=msp.PromptBox(*box) if box is not None else None,
                                 points=tuple(msp.PromptPoint(x, y, label) for x, y, label in points))
            by_id.setdefault(oid, []).append(fp)
        return [msp.PromptObject(id=oid, prompts=tuple(fps)) for oid, fps in by_id.items()]

    def input_text(self) -> str:
        try:
            return msp.build_object_prompt(size=tuple(self.size), objects=self._prompt_objects())
        except msp.ObjectPromptError as exc:
            raise PromptError(f"object prompt: {exc}") from None

    def metadata(self) -> dict[str, str]:
        return {"propagation_direction": self.propagation_direction}

    def to_json_dict(self) -> dict:
        return {"kind": "objects", "size": list(self.size), "propagation_direction": self.propagation_direction,
                "objects": [{"id": oid, "frame": fr, "box": list(b) if b is not None else None,
                             "points": [list(p) for p in pts]} for oid, fr, b, pts in self.objects],
                "input_text": self.input_text()}


def prompt_from_json(d: dict) -> ConceptPrompt | ObjectPrompt:
    if d.get("kind") == "concept":
        return ConceptPrompt(d["text"])
    if d.get("kind") == "objects":
        return ObjectPrompt(tuple(d["size"]), tuple((o["id"], o["frame"], tuple(o["box"]) if o["box"] else None,
                                                     tuple(tuple(p) for p in o["points"])) for o in d["objects"]),
                            d.get("propagation_direction", "both"))
    raise PromptError(f"unknown prompt kind {d.get('kind')!r}")


def build_request(prompt: ConceptPrompt | ObjectPrompt, file_id: str, *, include_confidence: bool = True,
                  model: str = SAM_MODEL) -> dict:
    """The exact documented Responses request body for one video segmentation."""
    if not isinstance(file_id, str) or not file_id:
        raise PromptError("file_id must be a non-empty string from POST /v1/files")
    metadata = {"include_confidence": "true" if include_confidence else "false", **prompt.metadata()}
    return {"model": model, "stream": True, "metadata": metadata,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt.input_text()},
                {"type": "input_video", "file_id": file_id}]}]}


def cost_estimate(n_frames: int) -> dict:
    """Documented per-frame price; billed frames = max(frames processed, 50). Not a quote."""
    billed = max(int(n_frames), BILLING_MIN_FRAMES)
    return {"frames_in_video": int(n_frames), "billing_minimum_frames": BILLING_MIN_FRAMES,
            "expected_billed_frames": billed, "usd_per_1000_frames": PRICE_USD_PER_1000_FRAMES,
            "expected_cost_usd": round(billed * PRICE_USD_PER_1000_FRAMES / 1000.0, 6), "pricing_source": PRICING_SOURCE,
            "caveat": "assumes SAM processes every frame once; the response's usage.video_frames_processed is "
                      "authoritative"}


# ------------------------------------------------------------------ SSE + official parser


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict]:
    """Server-sent events -> JSON event dicts. Multi-line data fields are joined; '[DONE]' ends the stream."""
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                payload = "\n".join(data)
                data = []
                if payload.strip() == "[DONE]":
                    return
                yield meta_api.strict_json_loads(payload)
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:][1:] if line[5:6] == " " else line[5:])
    if data:
        payload = "\n".join(data)
        if payload.strip() != "[DONE]":
            yield meta_api.strict_json_loads(payload)


@dataclass
class StreamLog:
    """What the stream adapter observed (event counts, usage, response id) next to the parser result."""

    event_counts: dict[str, int] = field(default_factory=dict)
    response_id: str | None = None
    usage: dict | None = None
    n_snapshots: int = 0
    last_revision: int = 0

    def observe(self, ev: Any) -> None:
        t = ev.get("type") if isinstance(ev, dict) else getattr(ev, "type", None)
        self.event_counts[str(t)] = self.event_counts.get(str(t), 0) + 1
        resp = ev.get("response") if isinstance(ev, dict) else None
        if isinstance(resp, dict):
            if isinstance(resp.get("id"), str):
                self.response_id = resp["id"]
            if isinstance(resp.get("usage"), dict):
                self.usage = resp["usage"]


def parse_video_events(events: Iterable[Any], *, log: StreamLog | None = None,
                       on_snapshot: Callable[[msp.VideoSegmentationSnapshot], None] | None = None
                       ) -> msp.VideoSegmentationResult:
    """Run the OFFICIAL parser over Responses events. Snapshots are cumulative (never accumulated here).

    Returns the final result (outcome completed or incomplete). Raises SamStreamError for failed, refused,
    error-event, lane, parser or source failures.
    """
    log = log if log is not None else StreamLog()

    async def source():
        for ev in events:
            log.observe(ev)
            yield ev

    async def main():
        stream = msp.parse_responses_stream(source(), msp.video_segmentation_format())
        async for snap in stream:
            log.n_snapshots += 1
            log.last_revision = snap.revision
            if on_snapshot is not None:
                on_snapshot(snap)
        return await stream.final_result()

    try:
        return asyncio.run(main())
    except msp.ResponsesStreamFailedError as exc:
        raise SamStreamError(f"SAM reported the response as failed: {exc}", reason="response_failed") from None
    except msp.ResponsesStreamRefusalError as exc:
        raise SamStreamError(f"SAM refused the request: {exc}", reason="response_refusal") from None
    except msp.ResponsesStreamEventError as exc:
        raise SamStreamError(f"the stream reported an error: {exc}", reason="response_error") from None
    except msp.ResponsesStreamLaneError as exc:
        raise SamStreamError(f"invalid output-text lane: {exc}", reason="lane") from None
    except msp.ResponsesStreamParserError as exc:
        raise SamStreamError(f"parser failure: {exc}", reason="parser") from None
    except msp.ResponsesStreamSourceError as exc:
        cause = exc.__cause__ or exc
        if isinstance(cause, SamAPIError):
            raise cause from None
        raise SamStreamError(f"the event source failed: {type(cause).__name__}: {cause}", reason="source") from None
    except msp.ResponsesStreamError as exc:
        raise SamStreamError(f"stream error: {type(exc).__name__}: {exc}", reason="stream") from None


# ------------------------------------------------------------------ the client


@dataclass(frozen=True)
class UploadRecord:
    file_id: str
    bytes: int
    filename: str
    sha256: str
    purpose: str
    status: str | None
    created_at: int | None
    reused: bool = False

    def to_json_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SegmentationCall:
    """A finished video segmentation request: the official parser result plus what the transport observed."""

    result: msp.VideoSegmentationResult
    request: dict
    stream: StreamLog
    started_utc: str
    finished_utc: str
    evidence_path: Path | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class SamClient:
    """Explicit, single-purpose SAM 3.1 client. Constructing it makes no request."""

    def __init__(self, api_key: str | None = None, *, base_url: str = BASE_URL, model: str = SAM_MODEL,
                 transport: httpx2.BaseTransport | None = None, evidence_dir: str | Path | None = None,
                 sleep: Callable[[float], None] = time.sleep, stream_read_timeout_s: float = STREAM_READ_TIMEOUT_S):
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingAPIKeyError(f"no API key: pass api_key or set {API_KEY_ENV}")
        self._key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        self._sleep = sleep
        self._read_timeout = float(stream_read_timeout_s)

    def __repr__(self) -> str:
        return f"SamClient(base_url={self.base_url!r}, model={self.model!r}, api_key={REDACTED})"

    # -- plumbing

    def _client(self, read_timeout: float) -> httpx2.Client:
        timeout = httpx2.Timeout(connect=CONNECT_TIMEOUT_S, read=read_timeout, write=UPLOAD_TIMEOUT_S, pool=CONNECT_TIMEOUT_S)
        kw = {"transport": self._transport} if self._transport is not None else {}
        return httpx2.Client(base_url=self.base_url, timeout=timeout,
                             headers={"Authorization": f"Bearer {self._key}"}, **kw)

    def _scrub(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace(self._key, REDACTED)
        if isinstance(obj, dict):
            return {self._scrub(k): self._scrub(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._scrub(v) for v in obj]
        return obj

    def _evidence(self, name: str, record: dict) -> Path | None:
        if self.evidence_dir is None:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._scrub(record), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def _error_from_response(self, status: int, body: bytes, evidence: Path | None) -> SamHTTPError:
        code = message = None
        try:
            err = json.loads(body.decode("utf-8", "replace")).get("error", {})
            code, message = err.get("code"), err.get("message")
        except (ValueError, AttributeError):
            pass
        text = self._scrub(message or body[:300].decode("utf-8", "replace"))
        return SamHTTPError(f"HTTP {status}: {text}", status=status, code=code, evidence_path=evidence)

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float | None:
        if attempt >= MAX_ATTEMPTS:
            return None
        ra = meta_api.parse_retry_after(retry_after)
        if ra is not None:
            return ra if ra <= MAX_RETRY_AFTER_S else None
        return BACKOFF_BASE_S * (2 ** (attempt - 1)) * (1.0 + random.random())

    # -- files

    def upload_video(self, path: str | Path, *, expected_sha256: str | None = None) -> UploadRecord:
        """Upload the LOCAL file's bytes (POST /v1/files, purpose=user_data)."""
        path = Path(path)
        if not path.is_file():
            raise UploadError(f"not a file: {path}")
        size = path.stat().st_size
        if size == 0 or size > UPLOAD_MAX_BYTES:
            raise UploadError(f"file size {size} bytes is outside (0, {UPLOAD_MAX_BYTES}]")
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise UploadError("the file does not match the expected SHA-256")
        stamp = _now()
        rec: dict = {"kind": "sam_upload_evidence", "client_version": CLIENT_VERSION, "started_utc": stamp,
                     "file": {"name": path.name, "bytes": size, "sha256": digest}, "attempts": []}
        ev_name = f"upload_{stamp.replace(':', '').replace('.', '')}.json"
        evidence = self._evidence(ev_name, rec)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            entry = {"attempt": attempt, "started_utc": _now()}
            rec["attempts"].append(entry)
            try:
                with self._client(UPLOAD_TIMEOUT_S) as c, open(path, "rb") as fh:
                    resp = c.post("/files", data={"purpose": FILE_PURPOSE}, files={"file": (path.name, fh, "video/mp4")})
            except (httpx2.ConnectError, httpx2.ConnectTimeout) as exc:
                entry["outcome"] = f"not_sent: {type(exc).__name__}"
                self._evidence(ev_name, rec)
                delay = self._retry_delay(attempt, None)
                if delay is None:
                    raise SamTransportError(f"upload could not connect: {type(exc).__name__}", sent=False,
                                            evidence_path=evidence) from None
                self._sleep(delay)
                continue
            except httpx2.HTTPError as exc:
                entry["outcome"] = f"transport_error: {type(exc).__name__}"
                self._evidence(ev_name, rec)
                raise SamTransportError(f"upload failed in transit ({type(exc).__name__}); it may or may not have "
                                        "been stored", sent=None, evidence_path=evidence) from None
            entry["status"] = resp.status_code
            if resp.status_code == 429:
                delay = self._retry_delay(attempt, resp.headers.get("retry-after"))
                self._evidence(ev_name, rec)
                if delay is None:
                    raise self._error_from_response(429, resp.content, evidence)
                self._sleep(delay)
                continue
            if resp.status_code != 200:
                self._evidence(ev_name, rec)
                raise self._error_from_response(resp.status_code, resp.content, evidence)
            try:
                body = meta_api.strict_json_loads(resp.content.decode("utf-8"))
            except ValueError as exc:
                raise UploadError(f"malformed upload response: {exc}", evidence_path=evidence) from None
            if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not body["id"]:
                raise UploadError("upload response has no file id", evidence_path=evidence)
            if isinstance(body.get("bytes"), int) and body["bytes"] != size:
                raise UploadError(f"server stored {body['bytes']} bytes, local file has {size}", evidence_path=evidence)
            out = UploadRecord(body["id"], size, path.name, digest, body.get("purpose", FILE_PURPOSE),
                               body.get("status"), body.get("created_at"))
            rec["response"] = body
            rec["finished_utc"] = _now()
            self._evidence(ev_name, rec)
            return out
        raise SamTransportError("upload attempts exhausted", sent=None, evidence_path=evidence)

    def get_file(self, file_id: str) -> dict:
        with self._client(CONNECT_TIMEOUT_S * 2) as c:
            try:
                resp = c.get(f"/files/{file_id}")
            except httpx2.HTTPError as exc:
                raise SamTransportError(f"GET /files failed: {type(exc).__name__}", sent=None) from None
        if resp.status_code != 200:
            raise self._error_from_response(resp.status_code, resp.content, None)
        return meta_api.strict_json_loads(resp.content.decode("utf-8"))

    def reuse_or_upload(self, path: str | Path, record_path: str | Path) -> UploadRecord:
        """Reuse the file id recorded for this exact file (same SHA-256 and size, still on the server); else upload."""
        path, record_path = Path(path), Path(record_path)
        digest = sha256_file(path)
        if record_path.is_file():
            old = json.loads(record_path.read_text(encoding="utf-8"))
            if old.get("sha256") == digest and old.get("bytes") == path.stat().st_size:
                try:
                    info = self.get_file(old["file_id"])
                except SamHTTPError as exc:
                    if exc.status != 404:
                        raise
                    info = None
                if info is not None and info.get("bytes") in (None, old["bytes"]):
                    return UploadRecord(**{**old, "reused": True})
        rec = self.upload_video(path, expected_sha256=digest)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(rec.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return rec

    # -- segmentation

    def segment_video(self, file_id: str, prompt: ConceptPrompt | ObjectPrompt, *, include_confidence: bool = True,
                      on_snapshot: Callable[[msp.VideoSegmentationSnapshot], None] | None = None,
                      raw_sse_path: str | Path | None = None) -> SegmentationCall:
        """Stream one video segmentation and decode it with the official parser.

        A completed OR incomplete outcome returns a SegmentationCall (check result.outcome); failed, refused,
        error and HTTP failures raise. raw_sse_path: every received SSE line is appended there (no secrets).
        """
        request = build_request(prompt, file_id, include_confidence=include_confidence, model=self.model)
        started = _now()
        rec: dict = {"kind": "sam_segmentation_evidence", "client_version": CLIENT_VERSION, "started_utc": started,
                     "request": request, "attempts": []}
        ev_name = f"segment_{started.replace(':', '').replace('.', '')}.json"
        evidence = self._evidence(ev_name, rec)
        log = StreamLog()
        sse_out = open(raw_sse_path, "w", encoding="utf-8", newline="\n") if raw_sse_path is not None else None
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                entry = {"attempt": attempt, "started_utc": _now()}
                rec["attempts"].append(entry)
                client = self._client(self._read_timeout)
                try:
                    try:
                        cm = client.stream("POST", "/responses", json=request, headers={"Accept": "text/event-stream"})
                        resp = cm.__enter__()
                    except (httpx2.ConnectError, httpx2.ConnectTimeout) as exc:
                        entry["outcome"] = f"not_sent: {type(exc).__name__}"
                        self._evidence(ev_name, rec)
                        delay = self._retry_delay(attempt, None)
                        if delay is None:
                            raise SamTransportError(f"could not connect: {type(exc).__name__}", sent=False,
                                                    evidence_path=evidence) from None
                        self._sleep(delay)
                        continue
                    except httpx2.HTTPError as exc:
                        entry["outcome"] = f"transport_error: {type(exc).__name__}"
                        self._evidence(ev_name, rec)
                        raise SamTransportError(f"request failed in transit ({type(exc).__name__}); it may have been "
                                                "processed and billed; NOT retried", sent=None,
                                                evidence_path=evidence) from None
                    try:
                        entry["status"] = resp.status_code
                        if resp.status_code == 429:
                            resp.read()
                            delay = self._retry_delay(attempt, resp.headers.get("retry-after"))
                            self._evidence(ev_name, rec)
                            if delay is None:
                                raise self._error_from_response(429, resp.content, evidence)
                            self._sleep(delay)
                            continue
                        if resp.status_code != 200:
                            resp.read()
                            self._evidence(ev_name, rec)
                            raise self._error_from_response(resp.status_code, resp.content, evidence)

                        def lines():
                            try:
                                for line in resp.iter_lines():
                                    if sse_out is not None:
                                        sse_out.write(line + "\n")
                                    yield line
                            except httpx2.HTTPError as exc:
                                raise SamTransportError(f"the stream broke ({type(exc).__name__}); the request may "
                                                        "have been billed; NOT retried", sent=None,
                                                        evidence_path=evidence) from None

                        try:
                            result = parse_video_events(iter_sse_events(lines()), log=log, on_snapshot=on_snapshot)
                        except SamAPIError as exc:
                            entry["outcome"] = f"{exc.kind}: {self._scrub(str(exc))}"
                            rec["stream"] = log.__dict__
                            self._evidence(ev_name, rec)
                            exc.evidence_path = evidence
                            raise
                    finally:
                        cm.__exit__(None, None, None)
                finally:
                    client.close()
                finished = _now()
                entry["outcome"] = result.outcome.status
                rec.update({"finished_utc": finished, "stream": log.__dict__,
                            "outcome": {"status": result.outcome.status,
                                        "reason": getattr(result.outcome, "reason", None),
                                        "detail": getattr(result.outcome, "detail", None)},
                            "n_records": len(result.records), "n_diagnostics": len(result.diagnostics),
                            "raw_output_sha256": hashlib.sha256(result.raw_output.encode("utf-8")).hexdigest()})
                self._evidence(ev_name, rec)
                return SegmentationCall(result, request, log, started, finished, evidence)
            raise SamTransportError("attempts exhausted", sent=None, evidence_path=evidence)
        finally:
            if sse_out is not None:
                sse_out.close()


def finite_or_none(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
