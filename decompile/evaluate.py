"""P3: held-out evaluation of a frozen prediction artifact (the ONLY reader of vault data).

Evaluation-only inputs:
- the frozen prediction artifact written by decompile.predict and its separate SHA-256 file;
- an evaluation vault (written by write_evaluation_vault): the withheld positions from the
  decompile.holdout split, optionally the generator's ground-truth contacts, and notes.

evaluate(prediction_path, prediction_sha256_path, vault_path) refuses (EvaluationError)
unless every check passes:
  prediction_sha256  the artifact bytes hash to the separate SHA-256 record (which names it)
  prediction_format  strict canonical JSON; kind/version; every number finite
  vault_sha256       the vault file hashes to its own SHA-256 record; the holdout vault hash
                     recomputes from its content
  inputs             training_sha256 and schedule_sha256 of the prediction equal the vault's;
                     the vault's withheld frames reproduce that schedule hash
  alignment          one prediction per vault object; identical frame indices and exact times;
                     no missing, extra or duplicate entries; every model series complete
  programs           each embedded scene program re-hashes to its recorded program_sha256
It only reads: the artifact, its SHA-256 and the vault are never modified.

Metrics per object and model (scene_program and each baseline): per-frame x, y and
Euclidean errors (prediction - truth, px), x/y/Euclidean RMSE, mean and maximum Euclidean
error, error against horizon (t - last training time, exact), first and last withheld errors.

Contacts (scene_program only; baselines do not model contacts: "not_applicable"):
predicted events after the last training time and up to the last scheduled time are matched
one-to-one, chronologically (k-th with k-th), with ground-truth contacts in the same window;
unmatched predictions are false contacts, unmatched truths are missed. The contact made
during the training period is excluded.

The report separates training-fit consistency (copied from the artifact), withheld
predictive accuracy, and baseline comparisons. A synthetic experiment says nothing about
performance on real videos.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from decompile import holdout as ho
from decompile import predict as pd
from decompile import scene_program as sp
from decompile.meta_api import strict_json_loads
from decompile.trajectories import timestamp_from_json

EVALUATOR_VERSION = "decompile.evaluate/1"
VAULT_KIND = "evaluation_vault"
VAULT_FORMAT_VERSION = 1
REPORT_KIND = "evaluation_report"

LIMITATIONS = (
    "Synthetic data from decompile.synth: one pre-registered scene, positions rounded to 0.01 px and no other "
    "noise. This says nothing about performance on real videos.",
    "Contact matching is chronological one-to-one without a time gate; timing errors are reported as measured.",
    "Point predictions only: parameter uncertainty is not propagated to prediction bands.",
    "Units are image pixels and seconds; nothing is calibrated to metres.",
)


class EvaluationError(ValueError):
    """A provenance, integrity or alignment check failed; nothing was evaluated."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


def _fj(t: Fraction) -> dict:
    return {"num": t.numerator, "den": t.denominator}


def _sha_record(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="ascii")
    parts = text.strip().split("  ")
    if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
        raise ValueError(f"{path.name} is not a '<sha256>  <filename>' record")
    return parts[0], parts[1]


def _write_new(path: Path, data: bytes) -> None:
    pd._write_new(path, data)  # same atomic, no-overwrite writer


# ------------------------------------------------------------------ vault


def write_evaluation_vault(path: str | Path, split: ho.HoldoutSplit, *, ground_truth_contacts: Sequence[dict] | None = None,
                           notes: Sequence[str] = ()) -> str:
    """Write the evaluation-only vault (canonical JSON) and its SHA-256 record. Returns the SHA-256.

    ground_truth_contacts: [{"object_id", "t_s", "axis", "plane_index"}, ...] (e.g. from decompile.synth).
    """
    contacts = []
    for c in ground_truth_contacts or ():
        contacts.append({"object_id": str(c["object_id"]), "t_s": float(c["t_s"]), "axis": str(c["axis"]),
                         "plane_index": int(c["plane_index"])})
    data = {
        "kind": VAULT_KIND, "vault_format_version": VAULT_FORMAT_VERSION, "evaluator_version": EVALUATOR_VERSION,
        "training_sha256": split.inference.training_sha256, "schedule_sha256": split.inference.schedule_sha256,
        "holdout_vault": ho.vault_to_json(split.vault), "holdout_vault_sha256": split.vault.vault_sha256,
        "ground_truth_contacts": contacts if ground_truth_contacts is not None else None,
        "notes": [str(n) for n in notes],
    }
    raw = pd.canonical_bytes(data)
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(path)
    sha_path = pd.sha256_path_for(path)
    if path.exists() or sha_path.exists():
        raise FileExistsError(path if path.exists() else sha_path)
    _write_new(path, raw)
    try:
        _write_new(sha_path, f"{digest}  {path.name}\n".encode("ascii"))
    except BaseException:
        path.unlink()
        raise
    return digest


def _load_verified(path: Path, sha_path: Path, check: str) -> tuple[bytes, Any, str]:
    if not path.is_file():
        raise EvaluationError(check, f"file not found: {path}")
    if not sha_path.is_file():
        raise EvaluationError(check, f"SHA-256 record not found: {sha_path} (an artifact without it is not valid)")
    raw = path.read_bytes()
    try:
        recorded, name = _sha_record(sha_path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise EvaluationError(check, str(exc)) from None
    digest = hashlib.sha256(raw).hexdigest()
    if name != path.name:
        raise EvaluationError(check, f"SHA-256 record names {name!r}, not {path.name!r}")
    if digest != recorded:
        raise EvaluationError(check, f"SHA-256 {digest} != recorded {recorded}: the file was altered")
    try:
        data = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvaluationError(check, f"not strict JSON: {exc}") from None
    if pd.canonical_bytes(data) != raw:
        raise EvaluationError(check, "not in canonical form (re-serialised content differs)")
    return raw, data, digest


# ------------------------------------------------------------------ metrics


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def position_metrics(pred_x, pred_y, true_x, true_y, times: Sequence[Fraction], t_last: Fraction) -> dict:
    dx = [p - t for p, t in zip(pred_x, true_x)]
    dy = [p - t for p, t in zip(pred_y, true_y)]
    de = [math.hypot(a, b) for a, b in zip(dx, dy)]
    return {
        "per_frame": [{"t": _fj(t), "horizon_s": _fj(t - t_last), "dx_px": a, "dy_px": b, "error_px": e}
                      for t, a, b, e in zip(times, dx, dy, de)],
        "rmse_x_px": _rms(dx), "rmse_y_px": _rms(dy), "rmse_px": _rms(de),
        "mean_error_px": sum(de) / len(de), "max_error_px": max(de),
        "first_error_px": de[0], "last_error_px": de[-1],
    }


def match_contacts(predicted: Sequence[float], actual: Sequence[float]) -> dict:
    """One-to-one chronological matching (k-th predicted with k-th actual)."""
    p, a = sorted(predicted), sorted(actual)
    n = min(len(p), len(a))
    pairs = [{"predicted_t_s": p[k], "actual_t_s": a[k], "error_s": p[k] - a[k], "abs_error_s": abs(p[k] - a[k])}
             for k in range(n)]
    return {"matched": pairs, "missed_actual_t_s": a[n:], "false_predicted_t_s": p[n:],
            "n_predicted": len(p), "n_actual": len(a), "n_missed": len(a) - n, "n_false": len(p) - n,
            "mean_abs_error_s": (sum(x["abs_error_s"] for x in pairs) / n) if n else None,
            "max_abs_error_s": max((x["abs_error_s"] for x in pairs), default=None)}


def _finite_numbers(obj: Any) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_finite_numbers(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_finite_numbers(v) for v in obj)
    return True


# ------------------------------------------------------------------ evaluation


@dataclass(frozen=True)
class EvaluationReport:
    data: dict

    @property
    def summary(self) -> dict:
        return self.data["summary"]


def _check_limitations(limitations: Sequence[str] | None) -> list[str]:
    """None -> the default LIMITATIONS (unchanged P3 text); otherwise a non-empty sequence of non-empty strings."""
    if limitations is None:
        return list(LIMITATIONS)
    if isinstance(limitations, (str, bytes)) or not isinstance(limitations, Sequence) or not limitations:
        raise ValueError("limitations must be None or a non-empty sequence of strings")
    out = []
    for item in limitations:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"every limitation must be a non-empty string, got {item!r}")
        out.append(item)
    return out


def evaluate(prediction_path: str | Path, prediction_sha256_path: str | Path, vault_path: str | Path, *,
             limitations: Sequence[str] | None = None) -> EvaluationReport:
    """Verify the frozen prediction against the vault and compute held-out metrics. Reads only.

    limitations: experiment-specific limitation statements for the report. None (default) keeps
    the module's LIMITATIONS text exactly. Validated before any file is read (ValueError).
    """
    report_limitations = _check_limitations(limitations)
    checks: list[dict] = []
    ok = lambda name, detail: checks.append({"check": name, "passed": True, "detail": detail})  # noqa: E731
    prediction_path, prediction_sha256_path, vault_path = Path(prediction_path), Path(prediction_sha256_path), Path(vault_path)

    _, pred, pred_sha = _load_verified(prediction_path, prediction_sha256_path, "prediction_sha256")
    ok("prediction_sha256", pred_sha)
    if not isinstance(pred, dict) or pred.get("kind") != pd.PREDICTION_KIND \
            or pred.get("prediction_format_version") != pd.PREDICTION_FORMAT_VERSION:
        raise EvaluationError("prediction_format", "not a prediction artifact of the supported format version")
    if not _finite_numbers(pred):
        raise EvaluationError("prediction_format", "non-finite number in the artifact")
    ok("prediction_format", f"{pred['predictor_version']}, format v{pred['prediction_format_version']}")

    _, vault, vault_sha = _load_verified(vault_path, pd.sha256_path_for(vault_path), "vault_sha256")
    if not isinstance(vault, dict) or vault.get("kind") != VAULT_KIND or vault.get("vault_format_version") != VAULT_FORMAT_VERSION:
        raise EvaluationError("vault_sha256", "not an evaluation vault of the supported format version")
    hv = vault["holdout_vault"]
    if ho.sha256_canonical(hv) != vault["holdout_vault_sha256"]:
        raise EvaluationError("vault_sha256", "holdout vault content does not match holdout_vault_sha256")
    ok("vault_sha256", vault_sha)

    inputs = pred.get("inputs", {})
    if inputs.get("training_sha256") != vault["training_sha256"] or hv.get("training_sha256") != vault["training_sha256"]:
        raise EvaluationError("inputs", "training_sha256 of the prediction does not match the vault")
    if inputs.get("schedule_sha256") != vault["schedule_sha256"]:
        raise EvaluationError("inputs", "schedule_sha256 of the prediction does not match the vault")
    vault_obs = [(o["object_id"], o["frame_index"], timestamp_from_json(o["t"]), o["x"], o["y"]) for o in hv["observations"]]
    sched_json = [{"object_id": o, "frame_index": f, "t": _fj(t)} for o, f, t, _, _ in vault_obs]
    if ho.sha256_canonical(sched_json) != vault["schedule_sha256"]:
        raise EvaluationError("inputs", "the vault's withheld frames do not reproduce schedule_sha256")
    ok("inputs", f"training {vault['training_sha256']}, schedule {vault['schedule_sha256']}")

    truth: dict[str, list[tuple[int, Fraction, float, float]]] = {}
    for o, f, t, x, y in vault_obs:
        truth.setdefault(o, []).append((f, t, x, y))
    objs = pred.get("objects")
    if not isinstance(objs, list):
        raise EvaluationError("alignment", "artifact has no object list")
    ids = [o.get("object_id") for o in objs]
    if len(set(ids)) != len(ids):
        raise EvaluationError("alignment", f"duplicate object predictions {ids}")
    if set(ids) != set(truth):
        raise EvaluationError("alignment", f"predicted objects {sorted(map(str, ids))} != withheld objects {sorted(truth)}")
    models = [pd.SCENE_MODEL] + list(pred["baseline_specifications"])
    for o in objs:
        want = [(f, t) for f, t, _, _ in truth[o["object_id"]]]
        try:
            got = [(e["frame_index"], timestamp_from_json(e["t"])) for e in o["schedule"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError("alignment", f"{o['object_id']}: malformed schedule: {exc}") from None
        if len(set(got)) != len(got):
            raise EvaluationError("alignment", f"{o['object_id']}: duplicate predicted timestamps")
        if got != want:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            raise EvaluationError("alignment", f"{o['object_id']}: predicted frames/times differ from the withheld "
                                               f"frames (missing {missing[:5]}, extra {extra[:5]})")
        for m in models:
            series = o.get("predictions", {}).get(m)
            if not isinstance(series, dict) or len(series.get("x", ())) != len(want) or len(series.get("y", ())) != len(want):
                raise EvaluationError("alignment", f"{o['object_id']}: model {m!r} predictions are incomplete")
    ok("alignment", f"{len(objs)} object(s), {len(vault_obs)} withheld frame(s), models {models}")

    for o in objs:
        try:
            prog = sp.program_from_json(o["program"])
        except (sp.ProgramError, KeyError, TypeError) as exc:
            raise EvaluationError("programs", f"{o['object_id']}: invalid program: {exc}") from None
        if sp.program_sha256(prog) != o["program_sha256"]:
            raise EvaluationError("programs", f"{o['object_id']}: program does not match program_sha256")
    ok("programs", "every embedded scene program matches its recorded hash")

    # ---- metrics
    gt_contacts = vault.get("ground_truth_contacts")
    per_object = []
    for o in objs:
        oid = o["object_id"]
        rows = truth[oid]
        times = [t for _, t, _, _ in rows]
        tx, ty = [x for *_, x, _ in rows], [y for *_, y in rows]
        t_last = timestamp_from_json(o["training_last_t"])
        if not t_last < times[0]:
            raise EvaluationError("alignment", f"{oid}: last training time is not before the first withheld time")
        accuracy = {m: position_metrics(o["predictions"][m]["x"], o["predictions"][m]["y"], tx, ty, times, t_last)
                    for m in models}
        lo, hi = float(t_last), float(times[-1])
        predicted = [e["t_s"] for e in o["predicted_events"] if lo < e["t_s"] <= hi]
        if gt_contacts is None:
            contacts = {pd.SCENE_MODEL: {"status": "not_evaluated", "reason": "the vault holds no ground-truth contacts"}}
        else:
            actual = [c["t_s"] for c in gt_contacts if c["object_id"] == oid and lo < c["t_s"] <= hi]
            excluded = [c["t_s"] for c in gt_contacts if c["object_id"] == oid and c["t_s"] <= lo]
            contacts = {pd.SCENE_MODEL: {"status": "evaluated", "window_s": [lo, hi],
                                         "excluded_training_contacts_t_s": excluded, **match_contacts(predicted, actual)}}
        for b in pred["baseline_specifications"]:
            contacts[b] = {"status": "not_applicable", "reason": "this baseline does not model contacts"}
        ranking = sorted(models, key=lambda m: (accuracy[m]["rmse_px"], m))
        scene_rmse = accuracy[pd.SCENE_MODEL]["rmse_px"]
        comparison = {"ranking_by_rmse": ranking, "rmse_px": {m: accuracy[m]["rmse_px"] for m in models},
                      "rmse_ratio_to_scene_program": {m: (accuracy[m]["rmse_px"] / scene_rmse if scene_rmse > 0 else None)
                                                      for m in models}}
        per_object.append({
            "object_id": oid, "program_sha256": o["program_sha256"],
            "training_last_t": o["training_last_t"], "n_withheld": len(rows),
            "training_fit_consistency": o["training_fit_consistency"],
            "withheld_accuracy": accuracy, "baseline_comparison": comparison, "contacts": contacts,
            "predicted_events": o["predicted_events"],
        })

    disclosures = list(pred.get("experiment_notes", [])) + [n for n in vault.get("notes", [])
                                                           if n not in pred.get("experiment_notes", [])]
    summary = {
        "objects": [{"object_id": p["object_id"],
                     "rmse_px": p["baseline_comparison"]["rmse_px"],
                     "ranking_by_rmse": p["baseline_comparison"]["ranking_by_rmse"],
                     "scene_program_contacts": {k: p["contacts"][pd.SCENE_MODEL].get(k)
                                                for k in ("n_predicted", "n_actual", "n_missed", "n_false",
                                                          "mean_abs_error_s", "max_abs_error_s")}}
                    for p in per_object],
    }
    return EvaluationReport({
        "kind": REPORT_KIND, "evaluator_version": EVALUATOR_VERSION,
        "predictor_version": pred["predictor_version"], "versions": pred.get("versions", {}),
        "prediction": {"path": str(prediction_path), "sha256": pred_sha, "sha256_path": str(prediction_sha256_path)},
        "vault": {"path": str(vault_path), "sha256": vault_sha},
        "inputs": {"training_sha256": vault["training_sha256"], "schedule_sha256": vault["schedule_sha256"]},
        "integrity_checks": checks,
        "model_specifications": {"scene_program": "decompile.scene_program fitted on training data only",
                                 **pred["baseline_specifications"]},
        "sections": {"training_fit_consistency": "each model against its own training window (not held-out)",
                     "withheld_accuracy": "predictions against withheld positions (held-out)",
                     "baseline_comparison": "withheld RMSE of every model, ranked"},
        "objects": per_object, "summary": summary,
        "disclosures": disclosures, "limitations": report_limitations,
    })


def write_evaluation_report(report: EvaluationReport, path: str | Path) -> str:
    raw = pd.canonical_bytes(report.data)
    _write_new(Path(path), raw)
    return hashlib.sha256(raw).hexdigest()
