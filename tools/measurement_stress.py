#!/usr/bin/env python3
"""P7: compare ball-centre measurement methods on the independent synthetic stress suite.

Renders decompile.stress_render.stress_suite() (exact centres known), measures every frame with:
  M0 colour_centroid  the P5/P6 tracker (colour mask, chroma-weighted centroid)
  M1 mask_ellipse     an ellipse fitted (cv2.fitEllipse) to the contour of the same colour mask
  M2 edge_circle      decompile.ball_measure outline, robust circle fit (tracker centre_method)
  M3 edge_ellipse     decompile.ball_measure outline, robust ellipse fit (tracker centre_method)
and reports per scene and method: detections/rejections, x/y bias, x/y RMSE, max error, RMSE by
background region (top / bottom / straddling the boundary), diameter error, and for methods
with an uncertainty the calibration of (error / sigma).

The colour model of every method is derived ONCE per scene with tracker.config_from_seed on
frame 0 and a box around the ball (as a human would draw it). The harness is the only place
where measurements meet the truth. Usage: python tools/measurement_stress.py [--out DIR] [--frames N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decompile import ball_measure as bm  # noqa: E402
from decompile import stress_render as sr  # noqa: E402
from decompile import tracker as tk  # noqa: E402

METHODS = ("colour_centroid", "mask_ellipse", "edge_circle", "edge_ellipse", "edge_stadium")


def seed_config(video: sr.StressVideo) -> tk.TrackerConfig:
    t0 = video.truth[0]
    pad = t0.radius_px + 4 + t0.blur_px / 2
    box = (int(math.floor(t0.x - pad)), int(math.floor(t0.y - pad)), int(math.ceil(t0.x + pad)), int(math.ceil(t0.y + pad)))
    try:
        cfg, _ = tk.config_from_seed(video.frames[0], box, object_id="obj_01")
        return cfg
    except tk.SeedError as exc:
        # Fallback (recorded in the report): a fixed red colour model. For the edge methods the colour
        # model is only the coarse stage; the colour centroid is then measured with this model too.
        print(f"  [{video.scene.name}] seed refused ({exc}); using the fixed red fallback colour model")
        return tk.TrackerConfig(object_id="obj_01", hue_center_deg=356.0, hue_tolerance_deg=22.0, min_chroma=45,
                                weight_floor_chroma=25.0, min_area_px=40, max_area_px=4000, edge_margin_px=2)


def mask_ellipse(rgb: np.ndarray, cfg: tk.TrackerConfig) -> tuple[float, float] | None:
    det = tk.detect(rgb, cfg)
    if det.status != tk.DETECTED:
        return None
    chroma, hue = tk._chroma_hue(rgb)
    dh = np.abs((hue - cfg.hue_center_deg + 180.0) % 360.0 - 180.0)
    mask = ((chroma >= cfg.min_chroma) & (dh <= cfg.hue_tolerance_deg)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=len)
    if len(c) < 5:
        return None
    (ex, ey), _, _ = cv2.fitEllipse(c.astype(np.float32))
    return ex + 0.5, ey + 0.5  # contour points are pixel indices; pixel centres are at +0.5


def measure_scene(video: sr.StressVideo) -> dict:
    base = seed_config(video)
    cfgs = {m: tk.TrackerConfig(**{**{k: getattr(base, k) for k in ("object_id", "hue_center_deg", "hue_tolerance_deg",
                                                                     "min_chroma", "weight_floor_chroma", "min_area_px",
                                                                     "max_area_px", "edge_margin_px")},
                                   "hue_weighting": True, "centre_method": m})
            for m in ("colour_centroid", "edge_circle", "edge_ellipse", "edge_stadium")}
    rows = []
    for rgb, tru in zip(video.frames, video.truth):
        row = {"frame": tru.frame_index, "truth": [tru.x, tru.y], "radius_px": tru.radius_px, "bg": tru.background_at_centre,
               "blur_px": tru.blur_px, "occluded": tru.occluded_fraction}
        for m in ("colour_centroid", "edge_circle", "edge_ellipse", "edge_stadium"):
            d = tk.detect(rgb, cfgs[m])
            row[m] = None if d.status != tk.DETECTED else {
                "x": d.x, "y": d.y, "sigma_x": d.sigma_x, "sigma_y": d.sigma_y, "diameter_px": d.diameter_px}
            if d.status != tk.DETECTED:
                row[m + "_status"] = d.status if d.edge is None else f"{d.status}: {d.edge.get('rejection_reason')}"
        me = mask_ellipse(rgb, cfgs["colour_centroid"])
        row["mask_ellipse"] = None if me is None else {"x": me[0], "y": me[1], "sigma_x": None, "sigma_y": None,
                                                      "diameter_px": None}
        rows.append(row)
    return {"scene": video.scene.name, "frames": rows}


def _stats(errs: list[tuple[float, float]]) -> dict:
    if not errs:
        return {"n": 0}
    a = np.asarray(errs)
    mag = np.hypot(a[:, 0], a[:, 1])
    return {"n": len(errs), "bias_x": float(a[:, 0].mean()), "bias_y": float(a[:, 1].mean()),
            "rmse_x": float(np.sqrt((a[:, 0] ** 2).mean())), "rmse_y": float(np.sqrt((a[:, 1] ** 2).mean())),
            "max_err": float(mag.max())}


def summarise(result: dict) -> dict:
    out = {}
    for m in METHODS:
        errs, by_bg, zs, derr = [], {}, [], []
        n_rej = 0
        for r in result["frames"]:
            v = r[m]
            if v is None:
                n_rej += 1
                continue
            e = (v["x"] - r["truth"][0], v["y"] - r["truth"][1])
            errs.append(e)
            by_bg.setdefault(r["bg"], []).append(e)
            if v["sigma_x"]:
                zs.append((e[0] / v["sigma_x"], e[1] / v["sigma_y"]))
            if v["diameter_px"]:
                derr.append(v["diameter_px"] - 2 * r["radius_px"])
        s = _stats(errs)
        s["rejected"] = n_rej
        s["by_background"] = {k: _stats(v) for k, v in sorted(by_bg.items())}
        if zs:
            z = np.asarray(zs)
            s["z_rms"] = [float(np.sqrt((z[:, 0] ** 2).mean())), float(np.sqrt((z[:, 1] ** 2).mean()))]
            s["frac_within_2sigma"] = float((np.abs(z) <= 2).all(axis=1).mean())
        if derr:
            s["diameter_bias_px"] = float(np.mean(derr))
            s["diameter_rmse_px"] = float(np.sqrt(np.mean(np.square(derr))))
        out[m] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "measurement_stress")
    ap.add_argument("--frames", type=int, default=36)
    a = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = a.out / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {"kind": "measurement_stress_report", "renderer": sr.STRESS_RENDER_VERSION,
              "edge_method": bm.METHOD_VERSION, "edge_params": bm.EdgeParams().to_json_dict(), "scenes": {}}
    print(f"{'scene':22s} {'method':16s} {'n':>3s} {'rej':>3s} {'bias_x':>7s} {'bias_y':>7s} {'rmse_x':>7s} "
          f"{'rmse_y':>7s} {'max':>6s} {'strad_rmse':>10s} {'z_rms x,y':>11s} {'in2s':>5s} {'d_bias':>6s}")
    for scene in sr.stress_suite(a.frames):
        video = sr.render(scene)
        res = measure_scene(video)
        summ = summarise(res)
        report["scenes"][scene.name] = {"summary": summ, "frames": res["frames"], "contacts_s": list(video.contacts_s)}
        for m in METHODS:
            s = summ[m]
            st = s["by_background"].get("straddle", {})
            strad = math.hypot(st["rmse_x"], st["rmse_y"]) if st.get("n") else float("nan")
            z = s.get("z_rms")
            print(f"{scene.name:22s} {m:16s} {s['n']:3d} {s['rejected']:3d} "
                  + (f"{s['bias_x']:7.3f} {s['bias_y']:7.3f} {s['rmse_x']:7.3f} {s['rmse_y']:7.3f} {s['max_err']:6.2f}"
                     if s["n"] else " " * 38)
                  + f" {strad:10.3f} " + (f"{z[0]:5.2f},{z[1]:5.2f}" if z else " " * 11)
                  + (f" {s['frac_within_2sigma']:5.2f}" if z else "      ")
                  + (f" {s['diameter_bias_px']:6.2f}" if "diameter_bias_px" in s else ""))
    (out_dir / "measurement_stress_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {out_dir / 'measurement_stress_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
