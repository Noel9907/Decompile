"""P6b: training-only Muse Spark interpretation and deterministic physics review (ADVISORY).

run_spark_stage(training, out_dir, client=...) connects the existing B1a-B1c pieces:

    training TrajectorySet (and NOTHING else)
      -> decompile.scene_interpreter.interpret_scene()   B1b prompt + strict schema + one chat() call
                                                         + semantic validation + validation sidecar
      -> decompile.kinematic_fit.analyze_kinematics()     B1c-1, on the same training set
      -> decompile.physics_review.review_interpretation() B1c-2, deterministic findings
      -> <out_dir>/spark_interpretation.json              ONE separate AI artifact

Separation from prediction (mandatory): this module never touches the prediction, the scene
program or any of their inputs. The frozen prediction artifact is built by decompile.predict from
the inference inputs alone and is byte-identical whether this stage is disabled, succeeds, is
rejected or is unavailable. The run manifest links the two artifacts by hash.

Input discipline: the function receives one TrajectorySet, the training observations. It is
never given the withheld schedule, the evaluation vault, ground truth, full-clip information or
anything derived from withheld frames. The prompt (B1b-2a) sends only coordinate_system and
objects; the synthetic/source labels are withheld from the model.

Statuses (artifact "status"):
    disabled     no client: no request was made
    succeeded    the response passed strict schema and semantic validation and was reviewed
    rejected     the response conformed to the schema but failed semantic validation
    unavailable  the call failed (missing key, HTTP/transport error, schema violation, truncated
                 or empty response, unexpected client error) or its sidecar could not be written
A failure never raises out of run_spark_stage (KeyboardInterrupt/SystemExit still propagate):
the deterministic pipeline continues.

Single attempt: the call runs inside single_attempt(), which caps B1a's retry loop at one
attempt for its duration (B1a reads meta_api.MAX_ATTEMPTS at call time; the SDK's own retries are
already disabled), so a 429/5xx/connect failure is reported, never retried.

The artifact holds the validated interpretation, the review findings, provenance hashes and the
paths/hashes of the B1a evidence record and B1b validation sidecar (which hold the raw response).
Passing validation and review does not establish that any hypothesis is physically true.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import uuid
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from decompile import kinematic_fit as kf
from decompile import meta_api
from decompile import physics_review as prv
from decompile import scene_interpretation as si
from decompile import scene_interpreter as sint
from decompile import scene_prompt as spr
from decompile import trajectories as tr

STAGE_VERSION = "decompile.spark_stage/1"
ARTIFACT_NAME = "spark_interpretation.json"
DISABLED, SUCCEEDED, REJECTED, UNAVAILABLE = "disabled", "succeeded", "rejected", "unavailable"
STATUSES = (DISABLED, SUCCEEDED, REJECTED, UNAVAILABLE)
AUTHORITY = ("advisory only: Muse Spark hypotheses are checked by the deterministic physics reviewer and never "
             "change the kinematic fit, the scene program or the prediction")


# ------------------------------------------------------------------ helpers


def _to_json(obj: Any) -> Any:
    """JSON form of B1b/B1c dataclasses. Fractions stay exact; non-finite floats are labelled."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.name != "covariance"}
    if isinstance(obj, Fraction):
        return {"num": obj.numerator, "den": obj.denominator}
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"non_finite": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return _to_json(obj.item())
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _write_new(path: Path, data: bytes) -> None:
    """Atomic create; never overwrites; no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "xb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _file_ref(path: Path | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
            "exists": p.is_file()}


@contextmanager
def single_attempt():
    """Cap B1a's retry loop at one attempt for the duration of the block (restored afterwards)."""
    saved = meta_api.MAX_ATTEMPTS
    meta_api.MAX_ATTEMPTS = 1
    try:
        yield
    finally:
        meta_api.MAX_ATTEMPTS = saved


def _validated_training(training: Any) -> tr.TrajectorySet:
    if not isinstance(training, tr.TrajectorySet):
        raise TypeError(f"training must be a TrajectorySet, got {type(training).__name__}")
    return tr.TrajectorySet.from_json_dict(training.to_json_dict())


# ------------------------------------------------------------------ the stage


def run_spark_stage(training: tr.TrajectorySet, out_dir: str | Path, *, client: meta_api.MuseSparkClient | None = None,
                    client_factory: Callable[[], meta_api.MuseSparkClient] | None = None,
                    max_completion_tokens: int = sint.DEFAULT_MAX_COMPLETION_TOKENS,
                    reasoning_effort: str = sint.DEFAULT_REASONING_EFFORT) -> dict:
    """Interpret and review the TRAINING trajectory; write <out_dir>/spark_interpretation.json.

    client=None and client_factory=None means disabled (no request). client_factory is called inside
    the failure handling, so e.g. a missing API key is recorded as "unavailable". Returns {"status", "artifact_path", "artifact_sha256",
    "artifact"}. Never raises for API or validation failures.
    """
    training = _validated_training(training)
    out_dir = Path(out_dir)
    prompt = spr.build_scene_prompt(training)
    base = {
        "kind": "spark_interpretation", "stage_version": STAGE_VERSION, "authority": AUTHORITY,
        "input": {"what": "training trajectory only (coordinate_system + objects); no withheld schedule, vault, "
                          "ground truth or full-clip information",
                  "training_sha256": hashlib.sha256(canonical_bytes(training.to_json_dict())).hexdigest(),
                  "prompt_input_sha256": prompt.input_sha256,
                  "object_ids": list(training.object_ids),
                  "n_observations": sum(len(o.observations) for o in training.objects)},
        "versions": {"prompt": spr.PROMPT_VERSION, "interpretation_schema": si.SCHEMA_VERSION,
                     "kinematic_fit": kf.METHOD_VERSION, "physics_review": prv.METHOD_VERSION},
        "developer_prompt_sha256": prompt.developer_prompt_sha256,
        "request": {"max_completion_tokens": max_completion_tokens, "reasoning_effort": reasoning_effort,
                    "max_attempts": 1},
        "physical_correctness_note": ("Validation and review check consistency with the training data only; they do "
                                      "not establish that any hypothesis is physically true."),
    }
    if client is not None and client_factory is not None:
        raise ValueError("give client or client_factory, not both")
    if client is None and client_factory is None:
        record = {**base, "status": DISABLED, "reason": "Muse Spark stage disabled: no request was made"}
        return _finish(out_dir, record)

    try:
        if client is None:
            client = client_factory()
        with single_attempt():
            result = sint.interpret_scene(training, client, sidecar_dir=out_dir / "sidecars",
                                          max_completion_tokens=max_completion_tokens,
                                          reasoning_effort=reasoning_effort)
    except sint.InterpretationRejectedError as exc:
        record = {**base, "status": REJECTED, "validation_errors": list(exc.errors),
                  "validation_warnings": list(exc.warnings),
                  "evidence": _file_ref(exc.evidence_path), "sidecar": _file_ref(exc.sidecar_path)}
        return _finish(out_dir, record)
    except sint.SceneInterpreterError as exc:  # API failure or sidecar write failure
        api = getattr(exc, "api_error", None)
        record = {**base, "status": UNAVAILABLE,
                  "error": {"type": type(api).__name__ if api is not None else type(exc).__name__,
                            "message": str(api if api is not None else exc)[:500]},
                  "evidence": _file_ref(exc.evidence_path), "sidecar": _file_ref(exc.sidecar_path)}
        return _finish(out_dir, record)
    except meta_api.MetaAPIError as exc:  # raised before chat() (e.g. a missing key)
        record = {**base, "status": UNAVAILABLE, "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
                  "evidence": _file_ref(getattr(exc, "evidence_path", None))}
        return _finish(out_dir, record)
    except Exception as exc:  # noqa: BLE001  never let the advisory stage stop the deterministic pipeline
        record = {**base, "status": UNAVAILABLE,
                  "error": {"type": type(exc).__name__, "message": "unexpected error in the Muse Spark stage"}}
        return _finish(out_dir, record)

    kin = kf.analyze_kinematics(training)
    review = prv.review_interpretation(result.interpretation, kin)
    record = {
        **base, "status": SUCCEEDED,
        "model": {"requested": result.requested_model, "returned": result.model, "response_id": result.response_id,
                  "request_id": result.request_id, "response_provenance": result.response_provenance},
        "hashes": {"messages_sha256": result.messages_sha256, "input_sha256": result.input_sha256,
                   "schema_sha256": result.schema_sha256, "response_content_sha256": result.response_content_sha256},
        "usage": _to_json(result.usage), "attempts": result.attempts, "latency_s": result.latency_s,
        "evidence": _file_ref(result.evidence_path), "sidecar": _file_ref(result.sidecar_path),
        "interpretation": _to_json(result.interpretation),
        "physics_review": _to_json(review),
        "kinematics_summary": [{"object_id": o.object_id, "selected_model": o.selected_model, "adequacy": o.adequacy,
                                "verdicts": {k: v.verdict for k, v in sorted(o.verdicts.items())}}
                               for o in kin.objects],
    }
    return _finish(out_dir, record)


def _finish(out_dir: Path, record: dict) -> dict:
    assert record["status"] in STATUSES
    raw = canonical_bytes(record)
    path = out_dir / ARTIFACT_NAME
    _write_new(path, raw)
    return {"status": record["status"], "artifact_path": path, "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            "artifact": record}
