"""B1c-3 OFFLINE: replay a B1b scene interpretation through the deterministic physics review.

    B1b sidecar + its B1a API evidence record + the trajectory input
      -> verified model response (every link checked by SHA-256)
      -> SceneInterpretation (strict JSON, schema and semantic validation)
      -> analyze_kinematics() (B1c-1)
      -> review_interpretation() (B1c-2)
      -> runs/physics_review/<stamp>_physics-review_<id>.json (atomic, never overwritten)

OFFLINE ONLY: no API client is created and no request is made. The response is read from
the B1a evidence record that the sidecar names, and only if every check passes:

    sidecar           strict JSON, kind/version, status "accepted"
    evidence          resolved from the sidecar (or --evidence); file SHA-256 == sidecar api_evidence_sha256
    evidence content  strict JSON chat_completion_call, state "succeeded", exactly one HTTP 200 attempt,
                      transport == sidecar response_provenance, response id and model == sidecar
    request binding   evidence request messages hash == sidecar messages_sha256, developer prompt hash
                      == developer_prompt_sha256, request schema hash == schema_sha256
    response          SHA-256 of choices[0].message.content == sidecar response_content_sha256
    trajectory        parsed by decompile.trajectories; canonical model input hash == sidecar input_sha256;
                      that exact input is embedded in the evidence user message; object ids and
                      synthetic/source provenance == sidecar
    schema            build_interpretation_schema(trajectory) hash == sidecar schema_sha256
    interpretation    strict JSON, then schema + semantic validation (decompile.scene_interpretation)

Any failure stops the run before anything is written (exit code 1). A write failure exits
with code 3 and leaves no partial report. Exit code 2 is a command-line usage error.

Usage:
    .venv\\Scripts\\python tools\\physics_review.py --sidecar SIDECAR.json --trajectory TRAJ.json
        [--evidence EVIDENCE.json] [--output-dir DIR] [--position-sigma-px S | SX,SY]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import sys
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decompile import kinematic_fit as kf  # noqa: E402
from decompile import physics_review as pr  # noqa: E402
from decompile import scene_interpretation as si  # noqa: E402
from decompile.meta_api import strict_json_loads, write_json_atomic  # noqa: E402
from decompile.scene_prompt import build_scene_prompt, canonical_json, sha256_text  # noqa: E402
from decompile.trajectories import TrajectoryError, load_trajectories  # noqa: E402

REPORT_FORMAT_VERSION = 1
REPORT_KIND = "physics_review_report"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "physics_review"
SIDECAR_KIND = "scene_interpretation_validation"
SIDECAR_VERSION = 1
EVIDENCE_KIND = "chat_completion_call"

EXIT_OK, EXIT_VERIFICATION, EXIT_USAGE, EXIT_WRITE = 0, 1, 2, 3

REQUIRED_SIDECAR_KEYS = (
    "sidecar_version", "kind", "status", "response_provenance", "prompt_version", "developer_prompt_sha256",
    "messages_sha256", "input_sha256", "schema_version", "schema_sha256", "input_object_ids", "input_provenance",
    "api_evidence_path", "api_evidence_sha256", "response_content_sha256", "response",
)


class ReplayError(Exception):
    """A provenance, integrity, parsing or validation check failed."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


# ------------------------------------------------------------------ helpers


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_file(path: Path, check: str) -> Any:
    if not path.is_file():
        raise ReplayError(check, f"file not found: {path}")
    try:
        return strict_json_loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReplayError(check, f"{path} is not strict UTF-8 JSON: {exc}") from None


def to_json(obj: Any) -> Any:
    """JSON form of the B1c result dataclasses. Fractions stay exact; non-finite floats are labelled."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # `covariance` is an internal numerical cache of ModelFit (compare=False); its standard errors
        # are already reported in param_se.
        return {f.name: to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.name != "covariance"}
    if isinstance(obj, Fraction):
        return {"num": obj.numerator, "den": obj.denominator}
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else {"non_finite": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return to_json(obj.item())
    raise TypeError(f"cannot serialise {type(obj).__name__}")


# ------------------------------------------------------------------ verification and review


def build_report(sidecar_path: Path, trajectory_path: Path, evidence_path: Path | None = None,
                 position_sigma_px: float | tuple[float, float] | None = None) -> dict:
    """Run every check, the kinematic fit and the review. Raises ReplayError; writes nothing."""
    checks: list[dict] = []

    def passed(name: str, detail: str) -> None:
        checks.append({"check": name, "passed": True, "detail": detail})

    # 1. sidecar
    sidecar_path = Path(sidecar_path)
    sidecar = _load_json_file(sidecar_path, "sidecar")
    if not isinstance(sidecar, dict):
        raise ReplayError("sidecar", "top level is not a JSON object")
    missing = [k for k in REQUIRED_SIDECAR_KEYS if k not in sidecar]
    if missing:
        raise ReplayError("sidecar", f"missing keys {missing}")
    if sidecar["kind"] != SIDECAR_KIND or sidecar["sidecar_version"] != SIDECAR_VERSION:
        raise ReplayError("sidecar", f"not a {SIDECAR_KIND} v{SIDECAR_VERSION} sidecar "
                                     f"(kind={sidecar['kind']!r}, version={sidecar['sidecar_version']!r})")
    if sidecar["status"] != "accepted":
        raise ReplayError("sidecar", f"status is {sidecar['status']!r}; only an accepted interpretation can be reviewed")
    sidecar_sha = sha256_file(sidecar_path)
    passed("sidecar", f"{SIDECAR_KIND} v{SIDECAR_VERSION}, status accepted")

    # 2-3. evidence: resolve and verify the file hash recorded in the sidecar
    recorded = sidecar["api_evidence_path"]
    if not isinstance(recorded, str) or not recorded:
        raise ReplayError("evidence_resolve", "sidecar records no api_evidence_path")
    ev_path = Path(evidence_path) if evidence_path is not None else Path(recorded)
    if not ev_path.is_file():
        raise ReplayError("evidence_resolve", f"evidence file not found: {ev_path}")
    path_matches = ev_path.resolve() == Path(recorded).resolve() if Path(recorded).exists() else False
    passed("evidence_resolve", f"{ev_path} ({'recorded path' if path_matches else 'explicit path; identity checked by hash'})")
    ev_sha = sha256_file(ev_path)
    if ev_sha != sidecar["api_evidence_sha256"]:
        raise ReplayError("evidence_sha256", f"evidence file SHA-256 {ev_sha} != sidecar api_evidence_sha256 "
                                             f"{sidecar['api_evidence_sha256']}: not the evidence record this sidecar names")
    passed("evidence_sha256", ev_sha)

    # 4. evidence content
    ev = _load_json_file(ev_path, "evidence")
    if not isinstance(ev, dict) or ev.get("kind") != EVIDENCE_KIND:
        raise ReplayError("evidence", f"not a {EVIDENCE_KIND} evidence record")
    if ev.get("state") != "succeeded":
        raise ReplayError("evidence", f"evidence state is {ev.get('state')!r}, not 'succeeded'")
    if ev.get("transport") != sidecar["response_provenance"]:
        raise ReplayError("evidence", f"evidence transport {ev.get('transport')!r} != sidecar response_provenance "
                                      f"{sidecar['response_provenance']!r}")
    attempts = ev.get("attempts")
    ok_attempts = [a for a in attempts if a.get("http_status") == 200] if isinstance(attempts, list) else []
    if len(ok_attempts) != 1 or attempts[-1] is not ok_attempts[0]:
        raise ReplayError("evidence", "expected exactly one HTTP 200 attempt, as the final attempt")
    body = ok_attempts[0].get("response_json")
    try:
        content = body["choices"][0]["message"]["content"]
        response_id, model = body["id"], body["model"]
    except (TypeError, KeyError, IndexError):
        raise ReplayError("evidence", "the HTTP 200 attempt has no choices[0].message.content") from None
    if not isinstance(content, str):
        raise ReplayError("evidence", "response content is not a string")
    side_resp = sidecar["response"] if isinstance(sidecar["response"], dict) else {}
    if response_id != side_resp.get("response_id") or model != side_resp.get("model"):
        raise ReplayError("evidence", f"response id/model ({response_id!r}, {model!r}) != sidecar "
                                      f"({side_resp.get('response_id')!r}, {side_resp.get('model')!r})")
    request = ev.get("request") if isinstance(ev.get("request"), dict) else {}
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ReplayError("request_binding", "evidence request does not hold the two prompt messages")
    if sha256_text(canonical_json(messages)) != sidecar["messages_sha256"]:
        raise ReplayError("request_binding", "evidence request messages do not hash to sidecar messages_sha256")
    if sha256_text(messages[0].get("content", "")) != sidecar["developer_prompt_sha256"]:
        raise ReplayError("request_binding", "developer prompt does not hash to sidecar developer_prompt_sha256")
    try:
        request_schema = request["response_format"]["json_schema"]["schema"]
    except (TypeError, KeyError):
        raise ReplayError("request_binding", "evidence request has no strict json_schema") from None
    if sha256_text(canonical_json(request_schema)) != sidecar["schema_sha256"]:
        raise ReplayError("request_binding", "evidence request schema does not hash to sidecar schema_sha256")
    passed("evidence", f"transport {ev['transport']}, one HTTP 200 attempt, response {response_id}, model {model}")
    passed("request_binding", "messages, developer prompt and schema hashes match the sidecar")

    # 5. response content hash
    content_sha = sha256_text(content)
    if content_sha != sidecar["response_content_sha256"]:
        raise ReplayError("response_sha256", f"response content SHA-256 {content_sha} != sidecar "
                                             f"response_content_sha256 {sidecar['response_content_sha256']}")
    passed("response_sha256", content_sha)

    # 6. strict JSON
    try:
        data = strict_json_loads(content)
    except ValueError as exc:
        raise ReplayError("response_json", f"response content is not strict JSON: {exc}") from None
    passed("response_json", "strict JSON")

    # 8-9. trajectory and the model input it produced
    trajectory_path = Path(trajectory_path)
    if not trajectory_path.is_file():
        raise ReplayError("trajectory", f"file not found: {trajectory_path}")
    try:
        traj = load_trajectories(trajectory_path)
    except TrajectoryError as exc:
        raise ReplayError("trajectory", str(exc)) from None
    prompt = build_scene_prompt(traj)
    if prompt.input_sha256 != sidecar["input_sha256"]:
        raise ReplayError("input_sha256", f"trajectory model-input SHA-256 {prompt.input_sha256} != sidecar "
                                          f"input_sha256 {sidecar['input_sha256']}: not the input of this interpretation")
    if prompt.input_json not in str(messages[1].get("content", "")):
        raise ReplayError("input_sha256", "the evidence user message does not embed this trajectory's model input")
    if list(traj.object_ids) != sidecar["input_object_ids"]:
        raise ReplayError("input_sha256", f"object ids {list(traj.object_ids)} != sidecar {sidecar['input_object_ids']}")
    prov = sidecar["input_provenance"]
    if not isinstance(prov, dict) or prov.get("synthetic") != traj.synthetic or prov.get("source") != traj.source:
        raise ReplayError("input_sha256", "trajectory synthetic/source provenance differs from the sidecar")
    passed("input_sha256", f"{prompt.input_sha256}; embedded in the evidence request; provenance matches")
    schema = si.build_interpretation_schema(traj)
    if sha256_text(canonical_json(schema)) != sidecar["schema_sha256"]:
        raise ReplayError("schema_sha256", "schema built from this trajectory does not hash to sidecar schema_sha256")
    passed("schema_sha256", sidecar["schema_sha256"])

    # 7. schema and semantic validation
    validation = si.validate_interpretation(data, traj)
    if not validation.ok:
        raise ReplayError("interpretation", f"{len(validation.errors)} validation error(s): "
                                            + "; ".join(validation.errors[:10]))
    interp = si.parse_interpretation(data, traj)
    passed("interpretation", f"schema and semantic validation passed ({len(validation.warnings)} warning(s))")

    # 10-11. kinematics and review
    try:
        kin = kf.analyze_kinematics(traj, position_sigma_px=position_sigma_px)
    except kf.KinematicFitError as exc:
        raise ReplayError("kinematics", str(exc)) from None
    review = pr.review_interpretation(interp, kin)

    return {
        "report_format_version": REPORT_FORMAT_VERSION,
        "kind": REPORT_KIND,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "offline": True,
        "api_requests_made": 0,
        "sources": {
            "sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha},
            "evidence": {"path": str(ev_path), "sha256": ev_sha, "recorded_path": recorded,
                         "path_matches_recorded": path_matches},
            "trajectory": {"path": str(trajectory_path), "file_sha256": sha256_file(trajectory_path),
                           "input_sha256": prompt.input_sha256, "synthetic": traj.synthetic, "source": traj.source},
        },
        "response": {
            "response_id": response_id, "model": model, "response_provenance": ev["transport"],
            "prompt_version": sidecar["prompt_version"], "developer_prompt_sha256": sidecar["developer_prompt_sha256"],
            "messages_sha256": sidecar["messages_sha256"], "input_sha256": sidecar["input_sha256"],
            "schema_version": sidecar["schema_version"], "schema_sha256": sidecar["schema_sha256"],
            "response_content_sha256": content_sha,
        },
        "integrity_checks": checks,
        "interpretation_validation": {"errors": list(validation.errors), "warnings": list(validation.warnings)},
        "kinematics": to_json(kin),
        "physics_review": to_json(review),
        "summary": {"counts": dict(review.counts), "assessment": review.assessment,
                    "assessment_note": review.assessment_note},
        "physical_correctness_note": (
            "This report checks an interpretation against kinematic fits of the same trajectory. Passing the "
            "checks does not establish that the interpretation, or any physical cause, is true."),
    }


def new_report_path(output_dir: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(output_dir) / f"{stamp}_physics-review_{uuid.uuid4().hex[:12]}.json"


def write_report(report: dict, output_dir: Path) -> Path:
    """Atomic create (temp file, fsync, rename); refuses to overwrite; no partial file on failure."""
    path = new_report_path(output_dir)
    write_json_atomic(path, report)
    return path


# ------------------------------------------------------------------ CLI


def _sigma(text: str) -> float | tuple[float, float]:
    parts = text.split(",")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'sx,sy', got {text!r}") from None
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return (values[0], values[1])
    raise argparse.ArgumentTypeError(f"expected a number or 'sx,sy', got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="physics_review.py",
        description="OFFLINE replay of a B1b scene interpretation through the deterministic physics review. "
                    "Every input is verified by SHA-256 against the sidecar; no API request is ever made.",
        epilog="Exit codes: 0 report written; 1 a provenance/integrity/validation check failed (nothing written); "
               "2 command-line usage error; 3 the report could not be written (no partial file).")
    p.add_argument("--sidecar", required=True, type=Path, help="B1b scene-interpretation sidecar (runs/scene_interpretation/*.json)")
    p.add_argument("--trajectory", required=True, type=Path, help="trajectory JSON the interpretation was made from")
    p.add_argument("--evidence", type=Path, default=None,
                   help="B1a API evidence record; default: the path recorded in the sidecar. Its SHA-256 must match the sidecar.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"directory for the report (default: {DEFAULT_OUTPUT_DIR.relative_to(ROOT)})")
    p.add_argument("--position-sigma-px", type=_sigma, default=None,
                   help="measurement sigma in px, 'S' or 'SX,SY'; default: estimated per axis from the data")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("B1c-3 physics review (OFFLINE: no API request is made)")
    try:
        report = build_report(args.sidecar, args.trajectory, args.evidence, args.position_sigma_px)
    except ReplayError as exc:
        print(f"[FAIL] {exc}\n       nothing was written")
        return EXIT_VERIFICATION
    for c in report["integrity_checks"]:
        print(f"  [PASS] {c['check']}: {c['detail']}")
    try:
        path = write_report(report, args.output_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"[FAIL] report not written ({type(exc).__name__}: {exc}); no partial report was left")
        return EXIT_WRITE
    s = report["summary"]
    print(f"  response: {report['response']['response_id']} ({report['response']['model']}, "
          f"provenance {report['response']['response_provenance']})")
    print(f"  findings: {s['counts']['error']} error(s), {s['counts']['warning']} warning(s), "
          f"{s['counts']['info']} info; assessment {s['assessment']}")
    for f in report["physics_review"]["findings"]:
        if f["severity"] != pr.INFO:
            print(f"    [{f['severity']}] {f['code']} {f['subject_id']}: {f['message'][:160]}")
    print(f"  report: {path}")
    print(f"  report sha256: {sha256_file(path)}")
    print(f"  note: {s['assessment_note']}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
