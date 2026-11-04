"""P4: deterministic prediction-versus-reality overlay (PNG).

The overlay never reads an evaluation vault. It receives:
- the training TrajectorySet (what the predictor saw),
- the frozen prediction artifact (decompile.predict), and
- the evaluation report data (decompile.evaluate).
Withheld truth is DERIVED from the report: decompile.evaluate documents
dx_px = prediction - truth, so truth = prediction - dx. The derivation is cross-checked
across every model in the report (all must give the same truth to TRUTH_CHECK_PX) and the
report's frame times must equal the artifact's schedule; otherwise OverlayError.

build_chart_data(...) returns every plotted number (deterministic, JSON-ready); render_png
draws three panels:
  1. y(t), image coordinates (+y is DOWN: the axis is inverted so the plot looks like the image);
  2. the x-y path (inverted y);
  3. Euclidean error against prediction horizon, every model.
The title names the data provenance (chart["data_label"], derived from the training
TrajectorySet): "SYNTHETIC VIDEO" for tracker measurements of a synthetic video, "SYNTHETIC
scene" for generator trajectories, "REAL VIDEO" for tracker measurements of a real video.
Layout: the plots, the legend and the caption/disclosure block are three separate horizontal
bands. The legend and the disclosure block are measured after a first draw and the figure
height is set from those measurements, so neither overlaps the other or the plots, however
many disclosures there are.
The split boundary is drawn halfway between the last training frame and the first withheld
frame. Training data, withheld truth, the frozen scene-program prediction and the four
baselines use distinct styles. Predicted contacts (from the artifact) are marked at their
predicted positions; actual withheld contacts appear as vertical lines at their times (the
report holds their times, not positions). The experiment disclosures are printed under the
figure.
"""

from __future__ import annotations

import io
import math
import os
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decompile import scene_program as sp  # noqa: E402
from decompile import trajectories as tr  # noqa: E402

OVERLAY_VERSION = "decompile.overlay/1"
SCENE_MODEL = "scene_program"
TRUTH_CHECK_PX = 1e-9  # derived truth must agree across models to this tolerance
DENSE_POINTS = 400
TRACKER_SOURCES = ("CLASSICAL_TRACKER", "SAM_3_1")
# Figure layout (inches): a fixed plot band; the legend and disclosure bands are measured.
FIG_WIDTH_IN = 18.0
PLOT_BAND_IN = 6.2
TITLE_BAND_IN = 0.45
BAND_GAP_IN = 0.18
EDGE_PAD_IN = 0.08
CAPTION = ("The predictor saw only the training frames (left of the split) and the timestamps of the withheld "
           "frames. Withheld positions were revealed afterwards, for evaluation only.")
STYLES = {
    SCENE_MODEL: dict(color="#c0392b", lw=2.4, ls="-", label="physical scene program (frozen prediction)"),
    "constant_velocity_last8": dict(color="#2471a3", lw=1.4, ls="--", label="baseline: constant velocity (last 8)"),
    "constant_acceleration_last8": dict(color="#7d3c98", lw=1.4, ls="-.", label="baseline: constant acceleration (last 8)"),
    "cubic_polynomial_all": dict(color="#b9770e", lw=1.4, ls=":", label="baseline: cubic polynomial (all training)"),
    "post_contact_parabola": dict(color="#148f77", lw=1.4, ls=(0, (5, 2, 1, 2)), label="baseline: post-contact parabola"),
}


class OverlayError(ValueError):
    """Inconsistent inputs: nothing is drawn."""


def _t(d: dict) -> Fraction:
    return tr.timestamp_from_json(d)


def data_label(training: tr.TrajectorySet) -> str:
    """Provenance label for the figure title, from the data the predictor actually saw."""
    if not training.synthetic:
        return "REAL VIDEO"
    return "SYNTHETIC VIDEO" if training.source in TRACKER_SOURCES else "SYNTHETIC scene"


def build_chart_data(training: tr.TrajectorySet, artifact: dict, report: dict, object_id: str | None = None) -> dict:
    """Every number the figure shows, derived from permitted inputs only."""
    if not isinstance(training, tr.TrajectorySet):
        raise OverlayError("training must be a TrajectorySet")
    objs = {o["object_id"]: o for o in artifact["objects"]}
    reps = {o["object_id"]: o for o in report["objects"]}
    oid = object_id or artifact["objects"][0]["object_id"]
    if oid not in objs or oid not in reps or oid not in training.object_ids:
        raise OverlayError(f"object {oid!r} missing from training, artifact or report")
    if report["prediction"]["sha256"] is None or report["inputs"]["training_sha256"] != artifact["inputs"]["training_sha256"]:
        raise OverlayError("report and artifact refer to different training data")
    art, rep = objs[oid], reps[oid]
    times = [_t(e["t"]) for e in art["schedule"]]
    models = list(rep["withheld_accuracy"])
    if models[0] != SCENE_MODEL or set(models) != set(art["predictions"]):
        raise OverlayError("report models do not match the artifact predictions")
    truth_x = truth_y = None
    for m in models:
        rows = rep["withheld_accuracy"][m]["per_frame"]
        if [_t(r["t"]) for r in rows] != times:
            raise OverlayError(f"report frame times for {m} differ from the artifact schedule")
        px, py = art["predictions"][m]["x"], art["predictions"][m]["y"]
        tx = [p - r["dx_px"] for p, r in zip(px, rows)]  # dx = prediction - truth
        ty = [p - r["dy_px"] for p, r in zip(py, rows)]
        if truth_x is None:
            truth_x, truth_y = tx, ty
        elif max(abs(a - b) for a, b in zip(tx + ty, truth_x + truth_y)) > TRUTH_CHECK_PX:
            raise OverlayError(f"withheld truth derived from {m} disagrees with {models[0]}")
    obs = training.get(oid).observations
    t_last, t_first = obs[-1].t, times[0]
    split = (t_last + t_first) / 2

    prog = sp.program_from_json(art["program"])
    if sp.program_sha256(prog) != art["program_sha256"]:
        raise OverlayError("embedded program does not match program_sha256")
    t0 = prog.body.t0
    dense_t = [t0 + (times[-1] - t0) * Fraction(i, DENSE_POINTS) for i in range(DENSE_POINTS + 1)]
    dense = sp.simulate(prog, dense_t)
    check = sp.simulate(prog, times)
    if list(check.x_px) != art["predictions"][SCENE_MODEL]["x"] or list(check.y_px) != art["predictions"][SCENE_MODEL]["y"]:
        raise OverlayError("re-simulating the frozen program does not reproduce the frozen prediction")
    dense_b = [t_last + (times[-1] - t_last) * Fraction(i, 100) for i in range(101)]
    baseline_curves = {}
    for name, b in art["baselines"].items():
        tref = _t(b["t_ref"])
        ev = lambda c, tt: sum(ci * float(tt - tref) ** k for k, ci in enumerate(c))  # noqa: E731
        baseline_curves[name] = {"t_s": [float(t) for t in dense_b],
                                 "x": [ev(b["coefficients"]["x"], t) for t in dense_b],
                                 "y": [ev(b["coefficients"]["y"], t) for t in dense_b]}
    contacts = rep["contacts"][SCENE_MODEL]
    actual = []
    if contacts.get("status") == "evaluated":
        actual = sorted([p["actual_t_s"] for p in contacts["matched"]] + list(contacts["missed_actual_t_s"]))
    return {
        "overlay_version": OVERLAY_VERSION, "object_id": oid, "data_label": data_label(training),
        "prediction_sha256": report["prediction"]["sha256"], "program_sha256": art["program_sha256"],
        "training": {"t_s": [float(o.t) for o in obs], "x": [o.x for o in obs], "y": [o.y for o in obs]},
        "withheld_truth": {"t_s": [float(t) for t in times], "x": truth_x, "y": truth_y,
                           "derivation": "prediction - dx_px / dy_px from the evaluation report"},
        "predictions": {m: {"t_s": [float(t) for t in times], "x": art["predictions"][m]["x"],
                            "y": art["predictions"][m]["y"]} for m in models},
        "scene_program_curve": {"t_s": [float(t) for t in dense_t], "x": list(dense.x_px), "y": list(dense.y_px)},
        "baseline_curves": baseline_curves,
        "errors": {m: {"horizon_s": [float(_t(r["horizon_s"])) for r in rep["withheld_accuracy"][m]["per_frame"]],
                       "error_px": [r["error_px"] for r in rep["withheld_accuracy"][m]["per_frame"]]} for m in models},
        "split": {"t_last_training_s": float(t_last), "t_first_withheld_s": float(t_first), "t_split_s": float(split),
                  "t_split": {"num": split.numerator, "den": split.denominator},
                  "n_training": len(obs), "n_withheld": len(times)},
        "predicted_contacts": [{"t_s": e["t_s"], "x_px": e["x_px"], "y_px": e["y_px"], "phase": e["phase"]}
                               for e in art["predicted_events"]],
        "actual_withheld_contacts_t_s": actual,
        "excluded_training_contacts_t_s": list(contacts.get("excluded_training_contacts_t_s", [])),
        "rmse_px": {m: rep["withheld_accuracy"][m]["rmse_px"] for m in models},
        "disclosures": list(report.get("disclosures", [])),
        "caption": CAPTION,
    }


def draw_figure(chart: dict):
    """Draw the three-panel figure (a matplotlib Figure; close it after use)."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "decompile"})
    fig, (ax_y, ax_xy, ax_e) = plt.subplots(1, 3, figsize=(18, 8.0), dpi=130,
                                            gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    tr_, wt = chart["training"], chart["withheld_truth"]
    split = chart["split"]["t_split_s"]
    t_end = wt["t_s"][-1]

    # panel 1: y(t)
    ax_y.axvspan(split, t_end + 0.02, color="#eeeeee", zorder=0, label="withheld period (30%)")
    ax_y.axvline(split, color="black", lw=1.2, ls="--", zorder=1, label="train / withheld split")
    sc = chart["scene_program_curve"]
    ax_y.plot(sc["t_s"], sc["y"], color=STYLES[SCENE_MODEL]["color"], lw=0.9, alpha=0.55, zorder=2,
              label="scene program trajectory (fitted on training, extended forward)")
    for name, curve in chart["baseline_curves"].items():
        s = STYLES[name]
        ax_y.plot(curve["t_s"], curve["y"], color=s["color"], lw=s["lw"], ls=s["ls"], zorder=3, label=s["label"])
    p = chart["predictions"][SCENE_MODEL]
    ax_y.plot(p["t_s"], p["y"], color=STYLES[SCENE_MODEL]["color"], lw=STYLES[SCENE_MODEL]["lw"], marker="o", ms=4,
              zorder=5, label=STYLES[SCENE_MODEL]["label"])
    ax_y.plot(tr_["t_s"], tr_["y"], ls="none", marker="o", ms=4.5, color="black", zorder=6,
              label="training observations (seen by the predictor)")
    ax_y.plot(wt["t_s"], wt["y"], ls="none", marker="s", ms=6, mfc="none", mec="#1e8449", mew=1.6, zorder=7,
              label="withheld truth (revealed after freezing)")
    for c in chart["predicted_contacts"]:
        withheld = c["phase"] == "withheld"
        ax_y.plot([c["t_s"]], [c["y_px"]], ls="none", marker="X" if withheld else "P", ms=10,
                  color=STYLES[SCENE_MODEL]["color"] if withheld else "#555555", zorder=8)
    for i, t in enumerate(chart["actual_withheld_contacts_t_s"]):
        ax_y.axvline(t, color="#1e8449", lw=1.0, ls=(0, (1, 1)), zorder=4,
                     label="actual withheld contact time" if i == 0 else None)
    ax_y.plot([], [], ls="none", marker="X", ms=9, color=STYLES[SCENE_MODEL]["color"], label="predicted withheld contact")
    ax_y.plot([], [], ls="none", marker="P", ms=9, color="#555555", label="contact fitted from training")
    ax_y.invert_yaxis()
    ax_y.set_xlabel("time t (s)")
    ax_y.set_ylabel("y (px, image coordinates: +y is down)")
    ax_y.set_title("Vertical position over time")
    ax_y.text(split, 1.01, f"  split: {chart['split']['n_training']} training | {chart['split']['n_withheld']} withheld frames",
              transform=ax_y.get_xaxis_transform(), fontsize=8, va="bottom")
    handles, labels = ax_y.get_legend_handles_labels()
    lo = min(min(tr_["y"]), min(wt["y"]))
    hi = max(max(tr_["y"]), max(wt["y"]))
    pad = 0.35 * (hi - lo)
    ax_y.set_ylim(hi + pad, lo - pad)  # inverted, clipped so diverging baselines do not flatten the data

    # panel 2: x-y path
    ax_xy.plot(sc["x"], sc["y"], color=STYLES[SCENE_MODEL]["color"], lw=0.9, alpha=0.55)
    for name, curve in chart["baseline_curves"].items():
        s = STYLES[name]
        ax_xy.plot(curve["x"], curve["y"], color=s["color"], lw=s["lw"], ls=s["ls"])
    ax_xy.plot(p["x"], p["y"], color=STYLES[SCENE_MODEL]["color"], lw=STYLES[SCENE_MODEL]["lw"], marker="o", ms=4)
    ax_xy.plot(tr_["x"], tr_["y"], ls="none", marker="o", ms=4.5, color="black")
    ax_xy.plot(wt["x"], wt["y"], ls="none", marker="s", ms=6, mfc="none", mec="#1e8449", mew=1.6)
    for c in chart["predicted_contacts"]:
        withheld = c["phase"] == "withheld"
        ax_xy.plot([c["x_px"]], [c["y_px"]], ls="none", marker="X" if withheld else "P", ms=10,
                   color=STYLES[SCENE_MODEL]["color"] if withheld else "#555555")
    xs = tr_["x"] + wt["x"]
    ax_xy.set_xlim(min(xs) - 15, max(xs) + 15)
    ax_xy.set_ylim(hi + pad, lo - pad)
    ax_xy.set_xlabel("x (px, +x is right)")
    ax_xy.set_ylabel("y (px, +y is down)")
    ax_xy.set_title("Trajectory in the image plane")

    # panel 3: error against horizon
    for m, e in chart["errors"].items():
        s = STYLES[m]
        ax_e.plot(e["horizon_s"], [max(v, 1e-6) for v in e["error_px"]], color=s["color"], lw=s["lw"], ls=s["ls"],
                  marker="o", ms=3, label=f"{s['label'].split(' (')[0]}: RMSE {chart['rmse_px'][m]:.4g} px")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("prediction horizon (s after the last training frame)")
    ax_e.set_ylabel("Euclidean error (px, log scale)")
    ax_e.set_title("Withheld error against horizon")
    ax_e.legend(fontsize=7, loc="center right")

    label = chart.get("data_label", "SYNTHETIC scene")
    fig.suptitle(f"Predictive decompilation: frozen prediction versus withheld reality ({label})", fontsize=12,
                 y=0.995)
    notes = [chart["caption"]] + [f"Disclosure: {d}" for d in chart["disclosures"]]
    legend = fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5, frameon=False)
    notes_text = fig.text(0.01, 0.0, "\n".join(_wrap(n, 230) for n in notes), fontsize=7.2, va="bottom", ha="left",
                          linespacing=1.35)
    _layout(fig, legend, notes_text)
    return fig


def _layout(fig, legend, notes_text) -> None:
    """Stack three bands bottom-up: disclosures, legend, plots. Sizes in inches, measured after a draw."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_h = legend.get_window_extent(renderer).height / fig.dpi
    notes_h = notes_text.get_window_extent(renderer).height / fig.dpi
    height = EDGE_PAD_IN + notes_h + BAND_GAP_IN + legend_h + BAND_GAP_IN + PLOT_BAND_IN + TITLE_BAND_IN
    fig.set_size_inches(FIG_WIDTH_IN, height)
    notes_text.set_position((0.01, EDGE_PAD_IN / height))
    legend_top = (EDGE_PAD_IN + notes_h + BAND_GAP_IN + legend_h) / height
    legend.set_bbox_to_anchor((0.5, legend_top), transform=fig.transFigure)
    fig.tight_layout(rect=(0, legend_top + BAND_GAP_IN / height, 1, 1 - TITLE_BAND_IN / height))


def render_png(chart: dict) -> bytes:
    """Draw the figure to PNG bytes (deterministic for identical chart data)."""
    fig = draw_figure(chart)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def _wrap(text: str, width: int) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return "\n".join(lines)


def write_new(path: str | Path, data: bytes) -> None:
    """Atomic create (temp file, fsync, rename); never overwrites; no partial file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def finite(values) -> bool:
    return all(math.isfinite(v) for v in values)
