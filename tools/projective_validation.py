#!/usr/bin/env python3
"""P7: held-out validation of the projective physics model on synthetic stress scenes (truth known).

For each scene: render (decompile.stress_render), measure every frame with the edge-ellipse tracker,
split by time into training (one contact) and withheld frames (containing the next contact), fit
ONLY the training observations, and predict the withheld frames with
  projective   decompile.projective_fit + decompile.projective_program (camera FOV prior)
  image_space  the P5 path: decompile.kinematic_fit + decompile.scene_program (if its gates allow)
  4 baselines  the frozen P3 baseline definitions (decompile.predict.BASELINES)
then score positions against the EXACT renderer centres and the predicted next contact against
the renderer's contact time. This is synthetic validation of the method, not a real experiment.
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import measurement_stress as ms  # noqa: E402
from decompile import kinematic_fit as kf  # noqa: E402
from decompile import predict as pd  # noqa: E402
from decompile import projective_fit as pf  # noqa: E402
from decompile import projective_program as pp  # noqa: E402
from decompile import scene_program as sp  # noqa: E402
from decompile import stress_render as sr  # noqa: E402
from decompile import tracker as tk  # noqa: E402
from decompile import trajectories as tr  # noqa: E402

# (scene, last training frame, last withheld frame): fixed before any result was seen; each training
# window holds exactly one contact with >= 4 observations after it, each withheld window the next contact.
CASES = (("perspective_pitched", 18, 27), ("realistic_combined", 21, 31), ("receding_size", 19, 29),
         ("baseline_flat", 19, 29), ("two_tone_boundary", 19, 29))


def hfov_of(cam: sr.Camera) -> float:
    return math.degrees(2 * math.atan((cam.width / 2) / cam.f_px))


def run_case(name: str, last_train: int, last_held: int, colour: bool = False) -> dict:
    scene = [s for s in sr.stress_suite(max(last_held + 1, 36)) if s.name == name][0]
    v = sr.render(scene)
    base = ms.seed_config(v)
    common = {k: getattr(base, k) for k in ("object_id", "hue_center_deg", "hue_tolerance_deg", "min_chroma",
                                            "weight_floor_chroma", "min_area_px", "max_area_px", "edge_margin_px")}
    cfg = tk.TrackerConfig(**common, hue_weighting=True, centre_method="colour_centroid" if colour else "edge_ellipse")
    obs = []
    for i in range(0, last_train + 1):
        d = tk.detect(v.frames[i], cfg)
        if d.status == "detected":
            obs.append(tr.Observation(i, v.truth[i].t, d.x, d.y, d.sigma_x, d.sigma_y, d.diameter_px, d.sigma_diameter_px))
    cam = scene.camera
    T = tr.TrajectorySet(True, "CLASSICAL_TRACKER", tr.CoordinateSystem(cam.width, cam.height),
                         (tr.ObjectTrajectory("obj_01", tuple(obs)),))
    held = [v.truth[i] for i in range(last_train + 1, last_held + 1)]
    times = [h.t for h in held]
    t_last = obs[-1].t
    lo, hi = float(t_last), float(times[-1])
    true_contacts = [c for c in v.contacts_s if lo < c <= hi]
    out = {"scene": name, "n_train": len(obs), "n_withheld": len(held), "true_withheld_contacts": true_contacts,
           "models": {}}

    def score(label, xs, ys, contacts=None, note=None):
        err = [math.hypot(x - h.x, y - h.y) for x, y, h in zip(xs, ys, held)]
        out["models"][label] = {"rmse_px": math.sqrt(sum(e * e for e in err) / len(err)), "max_px": max(err),
                                "pred_contacts": contacts, "note": note}

    # projective
    if not colour:
        try:
            rep = pf.analyze_projective(T, noise="absolute")
            prior = pp.CameraPrior(hfov_deg=66.0)
            prog = pp.program_from_fit(rep, width=cam.width, prior=prior, t_last_rel=float(t_last) - rep.t0)
            sim = pp.simulate(prog, times)
            score("projective", sim.x_px, sim.y_px, [e.t_s for e in sim.events if lo < e.t_s <= hi],
                  f"perspective {rep.verdicts['perspective']['verdict']} (dBIC {rep.verdicts['perspective']['delta_bic']:.1f}),"
                  f" adequacy {rep.adequacy}; true hfov {hfov_of(cam):.1f} deg, prior 66")
            sens = {}
            for h in prior.hfov_range_deg:
                ps = pp.program_from_fit(rep, width=cam.width, prior=prior, t_last_rel=float(t_last) - rep.t0, hfov_deg=h)
                ss = pp.simulate(ps, times)
                err = [math.hypot(x - q.x, y - q.y) for x, y, q in zip(ss.x_px, ss.y_px, held)]
                sens[h] = (math.sqrt(sum(e * e for e in err) / len(err)), [round(e.t_s, 4) for e in ss.events if lo < e.t_s <= hi])
            out["projective_hfov_sensitivity"] = sens
        except (pp.ProgramUnavailable, pf.ProjectiveFitError) as exc:
            out["models"]["projective"] = {"rmse_px": None, "note": f"unavailable: {exc}"}
    # image space (P5 path)
    kin = kf.analyze_kinematics(T).objects[0]
    try:
        prog2 = sp.program_from_kinematics(kin)
        sim2 = sp.simulate(prog2, times)
        score("image_space", sim2.x_px, sim2.y_px, [e.t_s for e in sim2.events if lo < e.t_s <= hi],
              f"kinematic_fit adequacy {kin.adequacy}")
    except sp.ProgramUnavailable as exc:
        out["models"]["image_space"] = {"rmse_px": None, "note": "unavailable: " + "; ".join(exc.reasons)[:160]}
    # baselines
    o = T.objects[0].observations
    tc = float(kin.contact.t_contact_s) if kin.contact else float(o[len(o) // 2].t)
    wins = {"constant_velocity_last8": o[-8:], "constant_acceleration_last8": o[-8:], "cubic_polynomial_all": o,
            "post_contact_parabola": tuple(q for q in o if float(q.t) > tc)}
    for b, spec in pd.BASELINES.items():
        w = wins[b]
        if len(w) < spec["degree"] + 1:
            continue
        cx = pd._polyfit([q.t for q in w], [q.x for q in w], spec["degree"], t_last)
        cy = pd._polyfit([q.t for q in w], [q.y for q in w], spec["degree"], t_last)
        score(b, pd._polyval(cx, times, t_last), pd._polyval(cy, times, t_last))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    for name, a, b in CASES:
        r = run_case(name, a, b)
        print(f"\n== {name}: train {r['n_train']} obs, withheld {r['n_withheld']}, true withheld contacts "
              f"{[round(c, 4) for c in r['true_withheld_contacts']]}")
        for m, s in sorted(r["models"].items(), key=lambda kv: (kv[1]["rmse_px"] is None, kv[1]["rmse_px"] or 0)):
            if s["rmse_px"] is None:
                print(f"   {m:28s} {s['note']}")
            else:
                pc = s.get("pred_contacts")
                print(f"   {m:28s} RMSE {s['rmse_px']:8.3f} px  max {s['max_px']:8.3f}"
                      + (f"  contacts {[round(c, 4) for c in pc]}" if pc is not None else "") + (f"  [{s['note']}]" if s.get("note") else ""))
        if "projective_hfov_sensitivity" in r:
            print("   projective hfov sensitivity:", {h: (round(v[0], 3), v[1]) for h, v in r["projective_hfov_sensitivity"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
