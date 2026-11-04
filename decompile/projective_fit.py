"""P7: perspective-aware kinematic fitting from image position AND apparent diameter (decompile.projective_fit/1).

Why: a pinhole camera images a ball at depth Z at x = cx + f X / Z, y = cy + f Y / Z with apparent
diameter D = f d / Z (d = true diameter). When Z changes (a ball thrown away from the camera, or
falling while the camera looks down at the floor) image-space motion is NOT uniformly accelerated
even though the 3-D motion is: image velocity and image gravity both scale with 1/Z. Fitting
uniform acceleration in raw pixels then leaves systematic residuals and invents extra "events".

Key observation (no calibration needed): with the measured diameter D,
    Xt = (x - cx) / D = X / d,   Yt = (y - cy) / D = Y / d,   W = 1 / D = Z / (f d)
are the ball's 3-D coordinates in ball diameters (depth scaled by the unknown focal length f).
Under constant 3-D gravity with impulsive contacts, Xt(t), Yt(t) and W(t) are each piecewise
quadratic in t with SHARED contact times and one constant acceleration vector. An error in the
principal point (cx, cy) adds a multiple of W to Xt / Yt, which is itself piecewise quadratic, so
it is absorbed exactly by the coefficients: only the image centre is used, and it does not matter.

Models, all fitted to the SAME observations (x, y, D) with per-frame sigma (sigma_x, sigma_y,
sigma_diameter_px from decompile.ball_measure), so BIC comparisons are fair:
    image_one_contact        x, y, D each piecewise quadratic, one shared velocity discontinuity
    projective_one_contact   Xt, Yt, W piecewise quadratic, one shared contact; x = cx + Xt / W,
                             y = cy + Yt / W, D = 1 / W   (same number of parameters as above)
    projective_split_accel   + a different acceleration after the contact (tests constant gravity)
    projective_two_contacts  + a second velocity discontinuity (tests one-contact adequacy)
Noise: "absolute" (sigmas are absolute 1-sigma errors: -2 ln L = chi^2, adequacy by the chi^2 tail)
or "relative" (sigmas are relative weights; one scale per channel profiled out; adequacy
"not_assessable"). Thresholds are those of decompile.kinematic_fit (BIC 10 / 2, chi^2 p 1e-3,
outlier |z| 5); none were tuned on real data. The fit can return inadequate or inconclusive.

Inputs: a TrajectorySet whose observations ALL carry sigma_x, sigma_y and diameter_px (+ sigma).
Nothing else: no frames, no withheld schedule, no vault, no ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import chi2 as _chi2

from decompile import kinematic_fit as kf
from decompile.trajectories import ObjectTrajectory, TrajectorySet

METHOD_VERSION = "decompile.projective_fit/1"
IMG1, PROJ1, PROJ_SPLIT, PROJ2 = ("image_one_contact", "projective_one_contact", "projective_split_accel",
                                  "projective_two_contacts")
CHANNELS = ("x", "y", "D")
NOISE_MODES = ("absolute", "relative")
GRID = 13


class ProjectiveFitError(ValueError):
    pass


@dataclass(frozen=True)
class PFit:
    model: str
    t_contacts_s: tuple[float, ...]  # absolute times
    params: dict[str, float]  # channel.coef (channel = Xt, Yt, W for projective; x, y, D for image)
    param_se: dict[str, float]
    n_params: int
    n_scalar_obs: int
    neg2ll: float
    bic: float
    chi2: float | None
    chi2_dof: int | None
    chi2_p: float | None
    rms_px: dict[str, float]  # per channel (x, y, D), unweighted
    z_max: float  # largest |standardised residual|
    residuals: dict[str, tuple[float, ...]]  # x, y, D residuals (px)
    theta: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ProjectiveReport:
    method_version: str
    noise: str
    object_id: str
    principal_point: tuple[float, float]
    fits: dict[str, PFit]
    selected: str
    verdicts: dict[str, dict]
    adequacy: str  # "adequate" | "inadequate" | "not_assessable"
    adequacy_reasons: tuple[str, ...]
    contact_gap_frames: tuple[int, int] | None
    t0: float
    t0_exact: Any = None  # Fraction: the exact time of the first observation


# ------------------------------------------------------------------ data


class _Data:
    def __init__(self, obj: ObjectTrajectory, cx: float, cy: float) -> None:
        obs = obj.observations
        miss = [o.frame_index for o in obs if None in (o.sigma_x, o.sigma_y, o.diameter_px, o.sigma_diameter_px)]
        if miss:
            raise ProjectiveFitError(f"every observation needs sigma_x, sigma_y, diameter_px and sigma_diameter_px; "
                                     f"missing on frames {miss[:10]}")
        self.frames = [o.frame_index for o in obs]
        self.t0_exact = obs[0].t
        self.t0 = float(obs[0].t)
        self.t = np.array([float(o.t) - self.t0 for o in obs])
        self.Y = np.column_stack([[o.x for o in obs], [o.y for o in obs], [o.diameter_px for o in obs]]).astype(float)
        self.S = np.column_stack([[o.sigma_x for o in obs], [o.sigma_y for o in obs],
                                  [o.sigma_diameter_px for o in obs]]).astype(float)
        self.n = len(obs)
        self.cx, self.cy = cx, cy


def _basis(t: np.ndarray, contacts: tuple[float, ...], split_at: float | None) -> tuple[np.ndarray, list[str]]:
    cols, names = [np.ones_like(t), t, 0.5 * t * t], ["q0", "v0", "a"]
    for j, tc in enumerate(contacts):
        cols.append(np.maximum(t - tc, 0.0))
        names.append(f"dv{j + 1}")
    if split_at is not None:
        d = np.maximum(t - split_at, 0.0)
        cols.append(0.5 * d * d)
        names.append("da")
    return np.column_stack(cols), names


def _predict(theta: np.ndarray, B: np.ndarray, projective: bool, cx: float, cy: float) -> np.ndarray:
    k = B.shape[1]
    c = B @ theta.reshape(3, k).T  # (n, 3)
    if not projective:
        return c
    W = c[:, 2]
    W = np.where(np.abs(W) < 1e-9, 1e-9, W)
    return np.column_stack([cx + c[:, 0] / W, cy + c[:, 1] / W, 1.0 / W])


def _linear_init(d: _Data, B: np.ndarray, projective: bool) -> np.ndarray:
    Y, S = d.Y, d.S
    if projective:
        D = Y[:, 2]
        obs = np.column_stack([(Y[:, 0] - d.cx) / D, (Y[:, 1] - d.cy) / D, 1.0 / D])
        sd = np.column_stack([np.hypot(S[:, 0] / D, (Y[:, 0] - d.cx) * S[:, 2] / D ** 2),
                              np.hypot(S[:, 1] / D, (Y[:, 1] - d.cy) * S[:, 2] / D ** 2), S[:, 2] / D ** 2])
    else:
        obs, sd = Y, S
    th = []
    for c in range(3):
        w = 1.0 / sd[:, c]
        coef, *_ = np.linalg.lstsq(B * w[:, None], obs[:, c] * w, rcond=None)
        th.append(coef)
    return np.concatenate(th)


def _fit_fixed(d: _Data, contacts: tuple[float, ...], split_at: float | None, projective: bool, noise: str,
               theta0: np.ndarray | None = None):
    B, names = _basis(d.t, contacts, split_at)
    if np.linalg.matrix_rank(B) < B.shape[1] or d.n - B.shape[1] < 1:
        return None
    th0 = _linear_init(d, B, projective) if theta0 is None or theta0.size != 3 * B.shape[1] else theta0
    if projective:
        def res(th):
            return ((d.Y - _predict(th, B, True, d.cx, d.cy)) / d.S).ravel()
        sol = least_squares(res, th0, method="lm", x_scale="jac")
        th = sol.x
        J = sol.jac
    else:
        th = _linear_init(d, B, False)  # weighted LS is exact for the linear image-space model
        J = None
    r = (d.Y - _predict(th, B, projective, d.cx, d.cy)) / d.S
    rss = np.sum(r * r, axis=0)
    if noise == "absolute":
        neg2ll = float(rss.sum())
    else:
        neg2ll = float(d.n * np.sum(np.log(np.maximum(rss / d.n, 1e-18))))
    return th, B, names, r, rss, neg2ll, J


def _objective(d, contacts, split_at, projective, noise):
    out = _fit_fixed(d, contacts, split_at, projective, noise)
    return math.inf if out is None else out[5]


def _gaps(d: _Data, exclude: tuple[int, ...] = ()) -> list[int]:
    m = kf.MIN_SEGMENT_OBS
    return [g for g in range(m - 1, d.n - m) if g not in exclude]


def _search_gap(d: _Data, g: int, fixed: tuple[float, ...], split: bool, projective: bool, noise: str):
    lo, hi = d.t[g], d.t[g + 1]

    def f(tc):
        cs = tuple(sorted(fixed + (tc,)))
        return _objective(d, cs, tc if split else None, projective, noise)

    xs = np.linspace(lo, hi, GRID)[1:-1]
    vals = [f(x) for x in xs]
    k = int(np.argmin(vals))
    a, b = (xs[k - 1] if k > 0 else lo), (xs[k + 1] if k + 1 < len(xs) else hi)
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-7})
    best = (float(r.x), float(r.fun)) if r.fun <= vals[k] else (float(xs[k]), float(vals[k]))
    return best


def _best_contact(d: _Data, fixed: tuple[float, ...], split: bool, projective: bool, noise: str,
                  exclude: tuple[int, ...] = ()):
    best = None
    for g in _gaps(d, exclude):
        tc, v = _search_gap(d, g, fixed, split, projective, noise)
        if best is None or v < best[2]:
            best = (g, tc, v)
    return best


def _make_fit(d: _Data, model: str, contacts: tuple[float, ...], split_at: float | None, projective: bool,
              noise: str) -> PFit | None:
    out = _fit_fixed(d, contacts, split_at, projective, noise)
    if out is None:
        return None
    th, B, names, r, rss, neg2ll, J = out
    k = B.shape[1]
    n_nonlin = len(contacts) + (1 if split_at is not None and split_at not in contacts else 0)
    n_noise = 3 if noise == "relative" else 0
    p = 3 * k + n_nonlin + n_noise
    N = 3 * d.n
    bic = neg2ll + p * math.log(N)
    chi2 = chi2_dof = chi2_p = None
    if noise == "absolute":
        chi2 = float(rss.sum())
        chi2_dof = N - (3 * k + n_nonlin)
        chi2_p = float(_chi2.sf(chi2, chi2_dof)) if chi2_dof > 0 else None
    # parameter standard errors (local, conditional on the contact times)
    chans = ("Xt", "Yt", "W") if projective else ("x", "y", "D")
    if J is None:
        Ws = []
        for c in range(3):
            Ws.append(B / d.S[:, c][:, None])
        covs = [np.linalg.pinv(Wc.T @ Wc) for Wc in Ws]
        scale = (rss / max(1, d.n - k)) if noise == "relative" else np.ones(3)
        se = {f"{chans[c]}.{nm}": float(math.sqrt(max(covs[c][j, j] * scale[c], 0.0)))
              for c in range(3) for j, nm in enumerate(names)}
    else:
        cov = np.linalg.pinv(J.T @ J)
        if noise == "relative":
            cov = cov * float(rss.sum() / max(1, N - 3 * k))
        se = {f"{chans[c]}.{nm}": float(math.sqrt(max(cov[c * k + j, c * k + j], 0.0)))
              for c in range(3) for j, nm in enumerate(names)}
    params = {f"{chans[c]}.{nm}": float(th[c * k + j]) for c in range(3) for j, nm in enumerate(names)}
    resid_px = d.Y - _predict(th, B, projective, d.cx, d.cy)
    zs = r if noise == "absolute" else r / np.sqrt(np.maximum(rss / d.n, 1e-18))
    return PFit(model, tuple(d.t0 + c for c in contacts), params, se, p, N, neg2ll, bic, chi2, chi2_dof, chi2_p,
                {ch: float(np.sqrt(np.mean(resid_px[:, i] ** 2))) for i, ch in enumerate(CHANNELS)},
                float(np.max(np.abs(zs))),
                {ch: tuple(float(v) for v in resid_px[:, i]) for i, ch in enumerate(CHANNELS)}, theta=(th, names))


def _nested(delta: float, simple: str, complex_: str) -> tuple[str, str]:
    if delta > kf.BIC_DECISIVE:
        return "supported", f"{complex_} is very strongly preferred over {simple} (Delta BIC = {delta:.2f})"
    if delta < -kf.BIC_POSITIVE:
        return "not_supported", f"{simple} is preferred over {complex_} (Delta BIC = {delta:.2f})"
    return "inconclusive", f"Delta BIC = {delta:.2f} between -{kf.BIC_POSITIVE} and {kf.BIC_DECISIVE}"


def analyze_projective(traj: TrajectorySet, *, noise: str = "absolute",
                       principal_point: tuple[float, float] | None = None) -> ProjectiveReport:
    """Fit the image-space and projective models to the (single) object of `traj`. Training data only."""
    if noise not in NOISE_MODES:
        raise ProjectiveFitError(f"noise must be one of {NOISE_MODES}")
    if len(traj.objects) != 1:
        raise ProjectiveFitError("exactly one object is supported")
    obj = traj.objects[0]
    cs = traj.coordinate_system
    cx, cy = principal_point if principal_point is not None else (cs.image_width / 2.0, cs.image_height / 2.0)
    d = _Data(obj, float(cx), float(cy))
    if d.n < 2 * kf.MIN_SEGMENT_OBS + 2:
        raise ProjectiveFitError(f"need at least {2 * kf.MIN_SEGMENT_OBS + 2} observations")
    fits: dict[str, PFit] = {}
    gi = _best_contact(d, (), False, False, noise)
    gp = _best_contact(d, (), False, True, noise)
    fits[IMG1] = _make_fit(d, IMG1, (gi[1],), None, False, noise)
    fits[PROJ1] = _make_fit(d, PROJ1, (gp[1],), None, True, noise)
    tc = gp[1]
    fits[PROJ_SPLIT] = _make_fit(d, PROJ_SPLIT, (tc,), tc, True, noise)
    g2 = _best_contact(d, (tc,), False, True, noise, exclude=(gp[0],))
    if g2 is not None:
        f2 = _make_fit(d, PROJ2, tuple(sorted((tc, g2[1]))), None, True, noise)
        if f2 is not None:
            fits[PROJ2] = f2
    verdicts: dict[str, dict] = {}
    dp = fits[IMG1].bic - fits[PROJ1].bic
    # Same parameter count: Delta BIC is the likelihood difference; "supported" = the projective model explains the
    # data very much better than image-space uniform acceleration.
    v, r = _nested(dp, IMG1, PROJ1)
    verdicts["perspective"] = {"verdict": v, "reason": r, "delta_bic": dp}
    ds = fits[PROJ1].bic - fits[PROJ_SPLIT].bic
    v, r = _nested(ds, PROJ1, PROJ_SPLIT)
    verdicts["acceleration_change_at_contact"] = {"verdict": v, "reason": r, "delta_bic": ds}
    extra = None
    if PROJ2 in fits:
        d2 = fits[PROJ1].bic - fits[PROJ2].bic
        verdicts["second_discontinuity"] = {"verdict": _nested(d2, PROJ1, PROJ2)[0], "delta_bic": d2,
                                            "t_s": max(fits[PROJ2].t_contacts_s, key=lambda t: abs(t - (d.t0 + tc))),
                                            "gap_frames": (d.frames[g2[0]], d.frames[g2[0] + 1])}
        if d2 > kf.BIC_DECISIVE:
            extra = verdicts["second_discontinuity"]
    selected = min((IMG1, PROJ1), key=lambda m: fits[m].bic)
    reasons: list[str] = []
    sel = fits[selected]
    if noise == "absolute" and sel.chi2_p is not None and sel.chi2_p < kf.ADEQUACY_P_MIN:
        reasons.append(f"{selected}: chi^2 = {sel.chi2:.4g} on {sel.chi2_dof} dof, p = {sel.chi2_p:.3g} < "
                       f"{kf.ADEQUACY_P_MIN}")
    if sel.z_max > kf.OUTLIER_Z:
        reasons.append(f"{selected}: largest standardised residual {sel.z_max:.1f} > {kf.OUTLIER_Z}")
    if selected == PROJ1 and extra is not None:
        reasons.append(f"a further velocity discontinuity near t = {extra['t_s']:.6g} s (frames {extra['gap_frames']}) "
                       f"is very strongly preferred (Delta BIC = {extra['delta_bic']:.2f})")
    if selected == PROJ1 and ds > kf.BIC_DECISIVE:
        reasons.append(f"the 3-D acceleration differs after the contact (Delta BIC = {ds:.2f}): not one constant gravity")
    if reasons:
        adequacy = "inadequate"
    elif noise == "relative":
        adequacy = "not_assessable"
        reasons.append("noise scale profiled out (relative weights): absolute adequacy not assessable")
    else:
        adequacy = "adequate"
    return ProjectiveReport(METHOD_VERSION, noise, obj.object_id, (cx, cy), fits, selected, verdicts, adequacy,
                            tuple(reasons), (d.frames[gp[0]], d.frames[gp[0] + 1]), d.t0, d.t0_exact)
