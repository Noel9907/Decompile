"""P7: executable 3-D scene program built from a projective fit (decompile.projective_program/1).

State (camera frame, units of BALL DIAMETERS d; X right, Y down, Z forward):
    P(t) = P0 + V0 (t - t0) + g (t - t0)^2 / 2 between contacts, with
    Xt = X / d, Yt = Y / d from decompile.projective_fit, and Zt = Z / d = f_px * W (W = 1 / D).
Image: x = cx + f * Xt / Zt = cx + Xt / W, y = cy + Yt / W, apparent diameter D = f / Zt.

Only the floor ORIENTATION needs the focal length f: monocular video fixes Xt, Yt and W but
not the depth scale f. f comes from a pre-registered camera prior (horizontal field of view):
    f_px = (image_width / 2) / tan(hfov / 2).
Prediction sensitivity to the prior is reported (hfov range), never tuned.

Physical assumptions (each stated in the program provenance):
  * one constant gravity vector g (fitted; the projective fit tests this);
  * a plane floor perpendicular to g, through the ball-centre position at the training contact
    (the centre is one radius above the floor at every contact);
  * contact law = the law measured at the training contact: the normal velocity reverses with
    restitution e = -v_n+ / v_n-, and the tangential velocity changes by k_t |dv_n|, where the
    vector k_t = dv_t / |dv_n| is the tangential impulse per unit normal impulse observed at the
    training contact (sliding friction with a constant direction; k_t = 0 is frictionless). The
    training contact is therefore reproduced exactly;
  * no drag, no spin dynamics, no other obstacles.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np

from decompile import kinematic_fit as kf
from decompile import projective_fit as pf

PROGRAM_VERSION = "decompile.projective_program/1"
PROGRAM_KIND = "projective_scene_program"
MAX_EVENTS = 64
RESTING_SPEED = 1e-6


class ProgramUnavailable(ValueError):
    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class CameraPrior:
    hfov_deg: float = 66.0
    hfov_range_deg: tuple[float, float] = (50.0, 80.0)
    basis: str = ("typical smartphone main camera (about 26 mm equivalent) recording 16:9 video; a pre-registered "
                  "prior, not a calibration")

    def __post_init__(self) -> None:
        lo, hi = self.hfov_range_deg
        if not (0 < lo <= self.hfov_deg <= hi < 180):
            raise ValueError("need 0 < hfov_lo <= hfov <= hfov_hi < 180")

    def f_px(self, width: int, hfov_deg: float | None = None) -> float:
        h = self.hfov_deg if hfov_deg is None else hfov_deg
        return (width / 2.0) / math.tan(math.radians(h) / 2.0)

    def to_json_dict(self) -> dict:
        return {"hfov_deg": self.hfov_deg, "hfov_range_deg": list(self.hfov_range_deg), "basis": self.basis}


@dataclass(frozen=True)
class ProjectiveProgram:
    object_id: str
    t0: Fraction
    principal_point: tuple[float, float]
    f_px: float
    p0: tuple[float, float, float]  # ball diameters
    v0: tuple[float, float, float]
    g: tuple[float, float, float]
    normal: tuple[float, float, float]  # unit, opposite to g (points away from the floor)
    plane_offset: float  # n . P at contact (ball centre one radius above the floor)
    restitution: float
    k_tangential: tuple[float, float, float]
    provenance: dict


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def program_to_json(p: ProjectiveProgram) -> dict:
    return {"kind": PROGRAM_KIND, "program_version": PROGRAM_VERSION, "object_id": p.object_id,
            "t0": {"num": p.t0.numerator, "den": p.t0.denominator}, "principal_point": list(p.principal_point),
            "f_px": p.f_px, "p0": list(p.p0), "v0": list(p.v0), "g": list(p.g), "normal": list(p.normal),
            "plane_offset": p.plane_offset, "restitution": p.restitution, "k_tangential": list(p.k_tangential),
            "provenance": p.provenance}


def program_from_json(d: dict) -> ProjectiveProgram:
    if not isinstance(d, dict) or d.get("kind") != PROGRAM_KIND or d.get("program_version") != PROGRAM_VERSION:
        raise ValueError("not a projective scene program of the supported version")
    t3 = lambda v: tuple(float(a) for a in v)  # noqa: E731
    return ProjectiveProgram(d["object_id"], Fraction(d["t0"]["num"], d["t0"]["den"]), tuple(d["principal_point"]),
                             float(d["f_px"]), t3(d["p0"]), t3(d["v0"]), t3(d["g"]), t3(d["normal"]),
                             float(d["plane_offset"]), float(d["restitution"]), t3(d["k_tangential"]), d["provenance"])


def program_sha256(p: ProjectiveProgram) -> str:
    return hashlib.sha256(_canon(program_to_json(p)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ simulation


@dataclass(frozen=True)
class Contact:
    index: int
    t_s: float
    x_px: float
    y_px: float
    v_normal_before: float  # ball diameters / s
    v_normal_after: float


@dataclass(frozen=True)
class Simulation:
    x_px: tuple[float, ...]
    y_px: tuple[float, ...]
    diameter_px: tuple[float, ...]
    events: tuple[Contact, ...]
    termination: str


def _project(p: ProjectiveProgram, P: np.ndarray) -> tuple[float, float, float]:
    cx, cy = p.principal_point
    Z = P[2]
    return cx + p.f_px * P[0] / Z, cy + p.f_px * P[1] / Z, p.f_px / Z


def _segments(p: ProjectiveProgram, t_end: float):
    n, g = np.array(p.normal), np.array(p.g)
    kt = np.array(p.k_tangential)
    P, V, t = np.array(p.p0), np.array(p.v0), float(p.t0)
    segs = [(t, P, V)]
    events: list[Contact] = []
    termination = "end_of_horizon"
    for k in range(MAX_EVENTS):
        c = float(n @ P) - p.plane_offset
        b = float(n @ V)
        a = 0.5 * float(n @ g)
        roots = []
        if abs(a) > 1e-15:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        elif b < 0:
            roots = [-c / b]
        roots = sorted(r for r in roots if r > 1e-9 and float(n @ (V + g * r)) < 0)
        if not roots or t + roots[0] > t_end:
            break
        tau = roots[0]
        P = P + V * tau + 0.5 * g * tau * tau
        Vin = V + g * tau
        vn = float(n @ Vin)
        dvn = -(1.0 + p.restitution) * vn
        V = Vin + dvn * n + kt * abs(dvn)
        t += tau
        x, y, _ = _project(p, P)
        events.append(Contact(k, t, x, y, vn, float(n @ V)))
        segs.append((t, P, V))
        if abs(float(n @ V)) < RESTING_SPEED:
            termination = "resting"
            break
    else:
        termination = "event_limit"
    return segs, tuple(events), termination


def simulate(p: ProjectiveProgram, times: Sequence[Fraction]) -> Simulation:
    ts = [float(t) for t in times]
    t_end = max(ts) if ts else float(p.t0)
    segs, events, term = _segments(p, t_end)
    g = np.array(p.g)
    xs, ys, ds = [], [], []
    for t in ts:
        t_s, P, V = segs[0]
        for s in segs:
            if s[0] <= t:
                t_s, P, V = s
        tau = t - t_s
        Q = P + V * tau + 0.5 * g * tau * tau
        x, y, d = _project(p, Q)
        xs.append(x), ys.append(y), ds.append(d)
    return Simulation(tuple(xs), tuple(ys), tuple(ds), events, term)


# ------------------------------------------------------------------ from a projective fit


def _vec(params: dict, name: str, f: float) -> np.ndarray:
    return np.array([params[f"Xt.{name}"], params[f"Yt.{name}"], f * params[f"W.{name}"]])


def program_from_fit(report: pf.ProjectiveReport, *, width: int, prior: CameraPrior, t_last_rel: float,
                     hfov_deg: float | None = None) -> ProjectiveProgram:
    """Build the 3-D program from the projective one-contact fit, or raise ProgramUnavailable with every reason."""
    reasons: list[str] = []
    fit = report.fits.get(pf.PROJ1)
    img = report.fits.get(pf.IMG1)
    if fit is None or img is None:
        raise ProgramUnavailable(["no projective one-contact fit"])
    if img.bic < fit.bic - kf.BIC_DECISIVE:
        reasons.append(f"image-space uniform acceleration is very strongly preferred over the projective model "
                       f"(Delta BIC = {fit.bic - img.bic:.2f})")
    if report.adequacy == "inadequate":
        reasons.append("the projective fit is inadequate: " + "; ".join(report.adequacy_reasons))
    f = prior.f_px(width, hfov_deg)
    prm = fit.params
    tc_rel = fit.t_contacts_s[0] - report.t0
    if not 0 < tc_rel < t_last_rel:
        reasons.append("the contact time lies outside the fitted span")
    g = _vec(prm, "a", f)
    gn = float(np.linalg.norm(g))
    if not gn > 0:
        raise ProgramUnavailable(reasons + ["zero fitted acceleration"])
    n = -g / gn
    v_minus = _vec(prm, "v0", f) + g * tc_rel
    v_plus = v_minus + _vec(prm, "dv1", f)
    vn_m, vn_p = float(n @ v_minus), float(n @ v_plus)
    if not (vn_m < 0 < vn_p):
        reasons.append(f"no reversal of the velocity normal to the floor at the contact (before {vn_m:.4g}, after "
                       f"{vn_p:.4g} diameters/s)")
    e = -vn_p / vn_m if vn_m != 0 else math.nan
    if not (math.isfinite(e) and 0.0 <= e <= 1.0):
        reasons.append(f"restitution {e!r} is outside [0, 1]")
    if reasons:
        raise ProgramUnavailable(reasons)
    dv = v_plus - v_minus
    dvn = float(n @ dv)
    dvt = dv - dvn * n
    kt = dvt / abs(dvn)
    p0 = np.array([prm["Xt.q0"], prm["Yt.q0"], f * prm["W.q0"]])
    pc = p0 + _vec(prm, "v0", f) * tc_rel + 0.5 * g * tc_rel ** 2
    t0 = report.t0_exact if isinstance(report.t0_exact, Fraction) else Fraction(report.t0)
    prov = {
        "source": pf.METHOD_VERSION, "fit_model": pf.PROJ1, "noise": report.noise, "fit_adequacy": report.adequacy,
        "verdicts": {k: v["verdict"] for k, v in report.verdicts.items()},
        "contact_t_s": fit.t_contacts_s[0], "contact_gap_frames": list(report.contact_gap_frames or ()),
        "camera_prior": prior.to_json_dict(), "hfov_used_deg": prior.hfov_deg if hfov_deg is None else hfov_deg,
        "units": "ball diameters (X, Y from the measured apparent diameter; Z = f_px / D)",
        "gravity_magnitude_diameters_s2": gn,
        "implied_ball_diameter_m_if_g_9_81": 9.81 / gn,
        "restitution_basis": "-v_n+ / v_n- of the fitted 3-D velocities at the training contact, n opposite to g",
        "tangential_impulse_per_normal_impulse": float(np.linalg.norm(kt)),
        "assumptions": ["one constant gravity vector", "plane floor perpendicular to gravity through the ball centre "
                        "at the training contact", "contact law measured at the training contact (normal restitution "
                        "and tangential impulse proportional to the normal impulse)", "no drag, no spin dynamics"],
        "note": "Reproducing the fitted observations is a consistency check, not held-out validation.",
    }
    return ProjectiveProgram(report.object_id, t0, tuple(report.principal_point), f, tuple(map(float, p0)),
                             tuple(map(float, _vec(prm, "v0", f))), tuple(map(float, g)), tuple(map(float, n)),
                             float(n @ pc), float(e), tuple(map(float, kt)), prov)
