"""P4: reveal the withheld future of a FROZEN prediction, evaluate it once, and draw the overlay.

OFFLINE: no API request, no prediction. The frozen artifact is only read (its SHA-256 is
checked before and after). Steps:

  1. the prediction file SHA-256 must equal --expected-prediction-sha256 (default: the frozen
     P3 artifact) and its .sha256 record;
  2. the inference inputs (training + schedule) are loaded and their recorded hashes verified;
     training and schedule hashes must equal the expected values and the artifact's inputs;
  3. decompile.evaluate.evaluate(...) is called exactly once (the only reader of the vault);
  4. decompile.overlay builds the chart from the training data, the artifact and the report
     (never from the vault) and renders the PNG;
  5. a fresh run directory receives evaluation_report.json, overlay.png, chart_data.json,
     summary.md and run_manifest.json, each written atomically and never overwritten. A write
     failure removes everything this run wrote.

Interpretation rules (fixed before the reveal): the primary metric is the Euclidean RMSE
over the withheld frames; "beats" means a strictly smaller Euclidean RMSE; error growth over
the horizon is the sign of the least-squares slope of the scene program's Euclidean error
against horizon, reported together with the first and last withheld errors.

Exit codes: 0 done; 1 a verification/evaluation check failed (nothing written);
2 usage error; 3 an output could not be written (nothing left behind).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from decompile import evaluate as ev  # noqa: E402
from decompile import overlay as ov  # noqa: E402
from decompile import predict as pd  # noqa: E402  (load_inference_inputs and constants only)
from decompile.meta_api import strict_json_loads  # noqa: E402

TOOL_VERSION = "tools.predictive_demo/1"
FROZEN_PREDICTION_SHA256 = "fdd7128486f6e142867d68aba699cbfe968aca8c617b2093c06b43f32d3aa37a"
FROZEN_TRAINING_SHA256 = "53f01ebbba7df9379727108df99b13406192e27c5a98b1ee7cc72c0417134fa3"
FROZEN_SCHEDULE_SHA256 = "af8a269fe096191aad5aa06aa0596b893fb1b5220e2526690c11817c10cef4ed"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "reveal"
EXIT_OK, EXIT_VERIFY, EXIT_USAGE, EXIT_WRITE = 0, 1, 2, 3


class DemoError(Exception):
    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slope(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0


def interpret(report: dict) -> dict:
    """Pre-defined comparison statements for each object (no rounding of the numbers)."""
    out = []
    for obj in report["objects"]:
        acc = obj["withheld_accuracy"]
        scene = acc[pd.SCENE_MODEL]["rmse_px"]
        baselines = [m for m in acc if m != pd.SCENE_MODEL]
        best = min(baselines, key=lambda m: (acc[m]["rmse_px"], m))
        rows = acc[pd.SCENE_MODEL]["per_frame"]
        hz = [r["horizon_s"]["num"] / r["horizon_s"]["den"] for r in rows]
        err = [r["error_px"] for r in rows]
        slope = _slope(hz, err)
        c = obj["contacts"][pd.SCENE_MODEL]
        out.append({
            "object_id": obj["object_id"], "primary_metric": "euclidean_rmse_px",
            "scene_program_rmse_px": scene,
            "beats": {m: scene < acc[m]["rmse_px"] for m in baselines},
            "baseline_rmse_px": {m: acc[m]["rmse_px"] for m in baselines},
            "best_baseline": best, "best_baseline_rmse_px": acc[best]["rmse_px"],
            "ranking_by_rmse": obj["baseline_comparison"]["ranking_by_rmse"],
            "contacts": c,
            "error_growth": {"slope_px_per_s": slope, "grows": slope > 0, "first_error_px": err[0],
                             "last_error_px": err[-1], "max_error_px": max(err),
                             "max_error_at_horizon_s": hz[err.index(max(err))]},
        })
    return {"rules": "primary metric Euclidean RMSE over withheld frames; 'beats' = strictly smaller; growth = sign "
                     "of the least-squares slope of error against horizon", "objects": out}


def summary_text(report: dict, interp: dict, paths: dict) -> str:
    lines = ["# Predictive decompilation: reveal of the frozen prediction (SYNTHETIC scene)", ""]
    lines += [f"- Prediction: `{report['prediction']['path']}` sha256 `{report['prediction']['sha256']}`",
              f"- Training sha256 `{report['inputs']['training_sha256']}`; schedule sha256 `{report['inputs']['schedule_sha256']}`",
              f"- Vault sha256 `{report['vault']['sha256']}`",
              f"- Integrity checks: {', '.join(c['check'] for c in report['integrity_checks'])} (all passed)", ""]
    for o, rep in zip(interp["objects"], report["objects"]):
        lines += [f"## {o['object_id']}  (program `{rep['program_sha256']}`)", "",
                  "| model | x RMSE | y RMSE | Euclidean RMSE | mean | max | first frame | last frame |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in o["ranking_by_rmse"]:
            a = rep["withheld_accuracy"][m]
            lines.append(f"| {m} | {a['rmse_x_px']!r} | {a['rmse_y_px']!r} | {a['rmse_px']!r} | {a['mean_error_px']!r} "
                         f"| {a['max_error_px']!r} | {a['first_error_px']!r} | {a['last_error_px']!r} |")
        lines += ["", f"Primary metric: Euclidean RMSE (px). Ranking: {' < '.join(o['ranking_by_rmse'])}."]
        for m, won in o["beats"].items():
            lines.append(f"- scene_program {'BEATS' if won else 'DOES NOT BEAT'} {m}: "
                         f"{o['scene_program_rmse_px']!r} vs {o['baseline_rmse_px'][m]!r} px")
        lines.append(f"- Best baseline: {o['best_baseline']} ({o['best_baseline_rmse_px']!r} px)")
        c = o["contacts"]
        if c.get("status") == "evaluated":
            for p in c["matched"]:
                lines.append(f"- Withheld contact predicted at {p['predicted_t_s']!r} s, actual {p['actual_t_s']!r} s, "
                             f"error {p['error_s']!r} s")
            lines.append(f"- Missed contacts: {c['missed_actual_t_s']}; false contacts: {c['false_predicted_t_s']}; "
                         f"training-period contact excluded: {c['excluded_training_contacts_t_s']}")
        else:
            lines.append(f"- Contacts: {c.get('status')} ({c.get('reason')})")
        lines.append("- Baseline contact metrics: not applicable (baselines do not model contacts)")
        g = o["error_growth"]
        lines.append(f"- Error over the horizon: first {g['first_error_px']!r} px, last {g['last_error_px']!r} px, max "
                     f"{g['max_error_px']!r} px at {g['max_error_at_horizon_s']!r} s; least-squares slope "
                     f"{g['slope_px_per_s']!r} px/s ({'grows' if g['grows'] else 'does not grow'}).")
        lines.append("")
    lines += ["## Disclosures", ""] + [f"- {d}" for d in report["disclosures"]]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    lines += ["", "## Outputs", ""] + [f"- {k}: `{v}`" for k, v in paths.items()]
    return "\n".join(lines) + "\n"


def run(prediction: Path, prediction_sha: Path, inputs: Path, vault: Path, output_dir: Path, *,
        expected_prediction: str, expected_training: str, expected_schedule: str) -> dict:
    # 1. the frozen prediction
    for p in (prediction, prediction_sha, inputs, vault):
        if not Path(p).is_file():
            raise DemoError("inputs", f"file not found: {p}")
    before = sha256_file(prediction)
    if before != expected_prediction:
        raise DemoError("frozen_prediction", f"prediction SHA-256 {before} != expected frozen {expected_prediction}")
    record = prediction_sha.read_text(encoding="ascii").split()
    if not record or record[0] != before:
        raise DemoError("frozen_prediction", "the .sha256 record does not match the prediction file")
    artifact = strict_json_loads(prediction.read_bytes().decode("utf-8"))
    # 2. training and schedule
    try:
        training, schedule = pd.load_inference_inputs(inputs)
    except (pd.PredictionError, ValueError) as exc:
        raise DemoError("training_inputs", str(exc)) from None
    from decompile.holdout import schedule_to_json, sha256_canonical
    t_sha = sha256_canonical(training.to_json_dict())
    s_sha = sha256_canonical(schedule_to_json(schedule))
    if (t_sha, s_sha) != (expected_training, expected_schedule):
        raise DemoError("training_inputs", f"training/schedule hashes ({t_sha}, {s_sha}) != expected "
                                           f"({expected_training}, {expected_schedule})")
    if artifact["inputs"] != {"training_sha256": t_sha, "schedule_sha256": s_sha}:
        raise DemoError("training_inputs", "the artifact was not made from these inference inputs")
    vault_sha_before = sha256_file(vault)
    inputs_sha_before = sha256_file(inputs)
    # 3. the one evaluation
    try:
        report = ev.evaluate(prediction, prediction_sha, vault)
    except ev.EvaluationError as exc:
        raise DemoError(f"evaluate.{exc.check}", str(exc)) from None
    # 4. chart from permitted inputs only
    try:
        chart = ov.build_chart_data(training, artifact, report.data)
    except ov.OverlayError as exc:
        raise DemoError("overlay", str(exc)) from None
    png = ov.render_png(chart)
    after = sha256_file(prediction)
    if after != before or sha256_file(vault) != vault_sha_before or sha256_file(inputs) != inputs_sha_before:
        raise DemoError("integrity", "an input file changed during the run")
    interp = interpret(report.data)

    # 5. outputs
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(output_dir) / f"{stamp}_p4_reveal_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    files = {"evaluation_report": out / "evaluation_report.json", "overlay_png": out / "overlay.png",
             "chart_data": out / "chart_data.json", "summary": out / "summary.md", "manifest": out / "run_manifest.json"}
    report_bytes = pd.canonical_bytes({**report.data, "interpretation": interp})
    chart_bytes = pd.canonical_bytes(chart)
    summary = summary_text(report.data, interp, {k: str(v) for k, v in files.items()}).encode("utf-8")
    hashes = {k: hashlib.sha256(b).hexdigest() for k, b in (("evaluation_report", report_bytes), ("overlay_png", png),
                                                           ("chart_data", chart_bytes), ("summary", summary))}
    manifest = {"kind": "p4_reveal_manifest", "tool_version": TOOL_VERSION, "generated_utc": stamp,
                "api_requests_made": 0, "predictions_made": 0, "evaluate_calls": 1,
                "prediction": {"path": str(prediction), "sha256_before": before, "sha256_after": after},
                "inference_inputs": {"path": str(inputs), "sha256": inputs_sha_before,
                                     "training_sha256": t_sha, "schedule_sha256": s_sha},
                "vault": {"path": str(vault), "sha256": vault_sha_before},
                "outputs": {k: {"path": str(files[k]), "sha256": h} for k, h in hashes.items()}}
    written: list[Path] = []
    try:
        for key, data in (("evaluation_report", report_bytes), ("overlay_png", png), ("chart_data", chart_bytes),
                          ("summary", summary), ("manifest", pd.canonical_bytes(manifest))):
            ov.write_new(files[key], data)
            written.append(files[key])
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"run_dir": out, "files": files, "hashes": hashes, "manifest": manifest, "interpretation": interp,
            "report": report.data}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predictive_demo.py",
        description="OFFLINE reveal: verify a frozen prediction, evaluate it once against the withheld vault and "
                    "draw the prediction-versus-reality overlay. Never predicts and never calls an API.",
        epilog="Exit codes: 0 done; 1 verification/evaluation failure (nothing written); 2 usage error; "
               "3 write failure (nothing left behind).")
    p.add_argument("--prediction", required=True, type=Path, help="frozen prediction artifact (runs/predictions/*.json)")
    p.add_argument("--prediction-sha256", type=Path, default=None, help="its SHA-256 record (default: <prediction>.sha256)")
    p.add_argument("--inference-inputs", required=True, type=Path, help="training + schedule file the prediction was made from")
    p.add_argument("--vault", required=True, type=Path, help="evaluation vault (read only by decompile.evaluate)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="parent directory for the run directory")
    p.add_argument("--expected-prediction-sha256", default=FROZEN_PREDICTION_SHA256)
    p.add_argument("--expected-training-sha256", default=FROZEN_TRAINING_SHA256)
    p.add_argument("--expected-schedule-sha256", default=FROZEN_SCHEDULE_SHA256)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    sha = a.prediction_sha256 or pd.sha256_path_for(a.prediction)
    print("P4 reveal (OFFLINE: no prediction, no API request)")
    try:
        r = run(a.prediction, sha, a.inference_inputs, a.vault, a.output_dir,
                expected_prediction=a.expected_prediction_sha256, expected_training=a.expected_training_sha256,
                expected_schedule=a.expected_schedule_sha256)
    except DemoError as exc:
        print(f"[FAIL] {exc}\n       nothing was written")
        return EXIT_VERIFY
    except OSError as exc:
        print(f"[FAIL] output not written ({type(exc).__name__}: {exc}); nothing was left behind")
        return EXIT_WRITE
    for o in r["interpretation"]["objects"]:
        print(f"  {o['object_id']}: scene_program RMSE {o['scene_program_rmse_px']!r} px; best baseline "
              f"{o['best_baseline']} {o['best_baseline_rmse_px']!r} px; beats {o['beats']}")
    print(f"  prediction sha256 before/after: {r['manifest']['prediction']['sha256_before']} / "
          f"{r['manifest']['prediction']['sha256_after']}")
    for k, p in r["files"].items():
        print(f"  {k}: {p}" + (f"  sha256 {r['hashes'][k]}" if k in r["hashes"] else ""))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
