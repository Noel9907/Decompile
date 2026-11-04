"""Background-independent ball-centre measurement from edge evidence (method decompile.ball_measure/1).

Why: the P5/P6 colour-weighted centroid depends on which edge pixels pass a colour threshold.
Blended edge pixels pass more often against a background of similar hue (a pink wall behind a
red ball) than against a different one (a grey-green floor), so the centroid shifts towards the
similar background when the ball straddles a background boundary. Seam highlights punch holes
in the mask and shading changes the weights. This module instead locates the ball's OUTLINE.

Method (per frame, independent; no other frame, no temporal smoothing, no future frames):
 1. Start from a coarse centre (x0, y0), a radius guess r0 and the largest distance `extent`
    from (x0, y0) to the coarse ball region (all from a coarse detection, e.g. the colour tracker).
 2. Cast N rays from the current centre. Along each ray sample the RGB image (bilinear, pixel
    centres at +0.5) from 0 to beyond the coarse extent. A ray stops at the image border or at
    a pixel outside the allowed region (ROI / exclusion boxes); if its outer background window
    is not fully available, the ray is dropped.
 3. Per ray, estimate the ball colour I just INSIDE the outline and the background colour O just
    OUTSIDE it, LOCALLY on that ray (two passes: far-window levels locate the transition, then
    levels re-estimated right at the ends of the transition ramp). Project the profile onto
    (I - O): p(r) = ((c(r) - O).(I - O)) / |I - O|^2 (1 inside, 0 outside).
 4. The edge on that ray is the OUTERMOST crossing of p = 0.5 (scanning inwards from outside), by
    linear interpolation. Because I and O are local, the 50% crossing marks 50% ball coverage
    whatever the background colour is on that side (wall, floor, or a boundary between them).
    Interior seam highlights and shading cannot move the outermost crossing. Rays with
    |I - O| below min_contrast carry no edge evidence and are dropped.
 5. Fit the outline by robust geometric least squares (soft-L1), reject outliers beyond
    max(outlier_z * robust sigma, min_outlier_px), and refit on inliers. Default model "stadium":
    a disc of radius r swept along a segment of half-length L (direction phi) -- exactly the
    outline of a ball moving uniformly during the exposure (L = 0: a sharp disc). Its centre is
    the ball position at mid-exposure and 2 r is the ball diameter. "ellipse" and "circle" are
    kept for comparison. Repeat from the fitted centre (n_iter passes) so rays are near-radial.
 6. Symmetry test: every model above is centrally symmetric, and so is the outline of an
    unoccluded ball under linear blur. An outline cut by an occluder or the frame edge, or bent
    by a bounce DURING the exposure, is not. The inlier residuals as a function of ray angle are
    regressed on cos/sin(k theta), k = 3 and 5; the amplitude A_odd of these odd harmonics
    measures the asymmetry the symmetric model could not absorb.
 7. Uncertainty (per frame): covariance of (x, y) from the inlier fit, (J^T J)^+ s^2, inflated for
    correlated neighbouring rays (effective number of independent edge samples = covered
    perimeter / corr_len_px), plus (asym_gain * A_odd)^2 for the unexplained asymmetry, plus a
    SYSTEMATIC floor sys_floor_px, all in quadrature on each axis. The floor and gain are
    properties of the method version, set from the independent synthetic stress suite
    (decompile.stress_render) BEFORE any real-video use; they are never tuned on real data.
 8. Quality gates -> status "rejected" with a reason (never a silently biased coordinate):
    too few inlier edge points, a large angular gap in edge coverage, an asymmetric outline
    (A_odd > max_asym_px: occluded / truncated / bounce within the exposure), implausible
    shape parameters, or a centre far from the coarse detection.

The module imports numpy, scipy and decompile.frames only. It never sees timestamps, other
frames, the renderer, ground truth, a vault or any physics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

METHOD_VERSION = "decompile.ball_measure/1"
MEASURED, REJECTED = "measured", "rejected"


@dataclass(frozen=True)
class EdgeParams:
    model: str = "ellipse"  # "ellipse" (default) | "stadium" | "circle"
    n_rays: int = 120
    step_px: float = 0.25
    inner_frac: tuple[float, float] = (0.30, 0.60)  # pass-1 inside window, fraction of r0
    outer_gap_px: float = 3.0  # pass-1 outside window starts this far beyond the coarse extent
    outer_len_px: float = 5.0
    ramp_hi: float = 0.85  # pass 2: ends of the transition ramp
    ramp_lo: float = 0.15
    local_win_px: float = 2.0  # pass 2: width of the local level windows
    min_contrast: float = 18.0  # |I - O| in RGB grey levels
    soft_l1_px: float = 0.5
    outlier_z: float = 3.0
    min_outlier_px: float = 0.75
    min_inlier_frac: float = 0.45
    max_gap_deg: float = 110.0
    max_axis_ratio: float = 6.0
    n_iter: int = 2
    corr_len_px: float = 2.0
    max_asym_px: float = 0.45  # reject: odd-harmonic outline asymmetry above this
    asym_gain: float = 3.0  # sigma term per px of odd-harmonic asymmetry
    sys_floor_px: float = 0.12  # systematic floor, from the synthetic stress suite (see module doc)
    diam_sys_rel: float = 0.02  # relative systematic diameter error under blur (stress suite: -1.2 .. -2.7 %)

    def __post_init__(self) -> None:
        if self.model not in ("stadium", "ellipse", "circle"):
            raise ValueError("model must be 'stadium', 'ellipse' or 'circle'")
        if self.n_rays < 16 or not 0 < self.step_px <= 1:
            raise ValueError("need n_rays >= 16 and 0 < step_px <= 1")

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["inner_frac"] = list(self.inner_frac)
        return d


@dataclass(frozen=True)
class BallMeasurement:
    status: str
    reason: str | None
    x: float | None
    y: float | None
    sigma_x: float | None  # 1-sigma, includes the systematic floor
    sigma_y: float | None
    cov_xy: float | None
    semi_major_px: float | None
    semi_minor_px: float | None
    angle_deg: float | None  # major-axis / blur direction, image degrees (0 = +x, 90 = +y down)
    diameter_px: float | None  # stadium: 2 r; ellipse: minor axis; circle: 2 r
    sigma_diameter_px: float | None  # statistical + relative systematic (diam_sys_rel)
    n_rays: int
    n_edges: int
    n_inliers: int
    max_gap_deg: float | None
    rms_residual_px: float | None
    asymmetry_px: float | None = None  # odd-harmonic (k = 3, 5) amplitude of the outline residuals
    blur_half_length_px: float | None = None  # stadium only

    def to_json_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ helpers


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear RGB samples at image coordinates (pixel centres at +0.5)."""
    coords = np.stack([ys.ravel() - 0.5, xs.ravel() - 0.5])
    out = np.stack([ndimage.map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(3)], axis=-1)
    return out.reshape(xs.shape + (3,))


def _crossing(p: np.ndarray, step: float) -> float | None:
    """Outermost r where p crosses 0.5 going inwards (p sampled at r = k * step)."""
    above = p >= 0.5
    idx = np.nonzero(above)[0]
    if idx.size == 0:
        return None
    j = int(idx[-1])
    if j + 1 >= p.size:
        return None  # still "inside" at the far end: no outline on this ray
    pj, pk = float(p[j]), float(p[j + 1])
    if pj == pk:
        return j * step
    return (j + (pj - 0.5) / (pj - pk)) * step


def _ellipse_resid(params: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    cx, cy, a, b, phi = params
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = px - cx, py - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    a, b = abs(a) + 1e-9, abs(b) + 1e-9
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    d = np.hypot(u, v)
    return d * (1.0 - 1.0 / np.maximum(rho, 1e-9))


def _circle_resid(params: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    cx, cy, r = params
    return np.hypot(px - cx, py - cy) - abs(r)


def _stadium_resid(params: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    cx, cy, r, L, phi = params
    ux, uy = math.cos(phi), math.sin(phi)
    dx, dy = px - cx, py - cy
    s = np.clip(dx * ux + dy * uy, -abs(L), abs(L))
    return np.hypot(dx - s * ux, dy - s * uy) - abs(r)


def _odd_asymmetry(theta: np.ndarray, resid: np.ndarray) -> float:
    """Amplitude of the k = 3 and k = 5 harmonics of the residuals versus ray angle (px)."""
    cols = [np.ones_like(theta)]
    for k in (1, 2, 3, 4, 5):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    A = np.stack(cols, axis=1)
    if theta.size <= A.shape[1] + 2:
        return float("inf")
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    odd = coef[[5, 6, 9, 10]]  # cos3, sin3, cos5, sin5
    return float(math.sqrt(float(odd @ odd)))


def _max_gap_deg(angles: np.ndarray) -> float:
    a = np.sort(np.mod(angles, 2 * math.pi))
    if a.size == 0:
        return 360.0
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return float(np.degrees(gaps.max()))


def _rejected(reason: str, n_rays: int, n_edges: int = 0, n_in: int = 0, gap: float | None = None) -> BallMeasurement:
    return BallMeasurement(REJECTED, reason, None, None, None, None, None, None, None, None, None, None,
                           n_rays, n_edges, n_in, gap, None)


# ------------------------------------------------------------------ one pass of edge points


def edge_points(img: np.ndarray, cx: float, cy: float, r0: float, extent: float, params: EdgeParams,
                allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge points (x, y) and their ray angles from the current centre. img: float RGB."""
    h, w = img.shape[:2]
    step = params.step_px
    r_end = extent + params.outer_gap_px + params.outer_len_px
    rs = np.arange(0.0, r_end + step / 2, step)
    th = np.arange(params.n_rays) * (2 * math.pi / params.n_rays)
    xs = cx + np.cos(th)[:, None] * rs[None, :]
    ys = cy + np.sin(th)[:, None] * rs[None, :]
    inside_img = (xs >= 0.5) & (xs <= w - 0.5) & (ys >= 0.5) & (ys <= h - 0.5)
    if allowed is not None:
        ok = np.zeros_like(inside_img)
        jj = np.clip(np.floor(xs).astype(int), 0, w - 1)
        ii = np.clip(np.floor(ys).astype(int), 0, h - 1)
        ok = allowed[ii, jj] > 0
        inside_img &= ok
    prof = _sample(img, xs, ys)
    in_lo, in_hi = (int(round(f * r0 / step)) for f in params.inner_frac)
    out_lo = int(round((extent + params.outer_gap_px) / step))
    lw = max(1, int(round(params.local_win_px / step)))
    pts_x, pts_y, angs = [], [], []
    for k in range(params.n_rays):
        valid = inside_img[k]
        bad = np.nonzero(~valid)[0]
        n_ok = int(bad[0]) if bad.size else rs.size  # ray truncated at the first invalid sample
        if n_ok < rs.size:
            continue  # the outer background window is not available on this ray
        c = prof[k]
        I = np.median(c[in_lo:max(in_hi, in_lo + 1)], axis=0)
        O = np.median(c[out_lo:], axis=0)
        d = I - O
        nd2 = float(d @ d)
        if nd2 < params.min_contrast ** 2:
            continue
        p = (c - O) @ d / nd2
        r1 = _crossing(p, step)
        if r1 is None:
            continue
        j = int(r1 / step)
        # pass 2: local levels at the ends of the transition ramp around the crossing
        hi = j
        while hi > 0 and p[hi] < params.ramp_hi:
            hi -= 1
        lo = j + 1
        while lo < p.size - 1 and p[lo] > params.ramp_lo:
            lo += 1
        if hi > 0 and lo + lw <= p.size:
            I2 = np.median(c[max(0, hi - lw):hi + 1], axis=0)
            O2 = np.median(c[lo:lo + lw + 1], axis=0)
            d2 = I2 - O2
            n2 = float(d2 @ d2)
            if n2 >= params.min_contrast ** 2:
                p2 = (c - O2) @ d2 / n2
                p2[:hi] = np.maximum(p2[:hi], 0.5)  # only the ramp region may host the crossing
                r2 = _crossing(p2[:lo + lw + 1], step)
                if r2 is not None:
                    r1 = r2
        pts_x.append(cx + math.cos(th[k]) * r1)
        pts_y.append(cy + math.sin(th[k]) * r1)
        angs.append(th[k])
    return np.asarray(pts_x), np.asarray(pts_y), np.asarray(angs)


# ------------------------------------------------------------------ the measurement


def _fit(px: np.ndarray, py: np.ndarray, guess: tuple[float, float, float], params: EdgeParams):
    cx, cy, r = guess
    if params.model == "circle":
        f, x0 = _circle_resid, np.array([cx, cy, r])
    elif params.model == "ellipse":
        f, x0 = _ellipse_resid, np.array([cx, cy, r, r * 0.98, 0.0])
    else:
        # initialise the stadium from a robust ellipse: radius = minor axis, half-length = a - b
        e = least_squares(_ellipse_resid, np.array([cx, cy, r, r * 0.98, 0.0]), args=(px, py), loss="soft_l1",
                          f_scale=params.soft_l1_px, method="trf")
        ea, eb, ephi = abs(e.x[2]), abs(e.x[3]), float(e.x[4])
        if eb > ea:
            ea, eb, ephi = eb, ea, ephi + math.pi / 2
        f, x0 = _stadium_resid, np.array([e.x[0], e.x[1], eb, max(ea - eb, 0.05), ephi])
    res = least_squares(f, x0, args=(px, py), loss="soft_l1", f_scale=params.soft_l1_px, method="trf")
    r_all = f(res.x, px, py)
    s_rob = 1.4826 * float(np.median(np.abs(r_all - np.median(r_all)))) if r_all.size else 0.0
    thr = max(params.outlier_z * s_rob, params.min_outlier_px)
    inl = np.abs(r_all) <= thr
    if inl.sum() < len(x0) + 2:
        return None
    res2 = least_squares(f, res.x, args=(px[inl], py[inl]), method="lm" if inl.sum() > len(x0) else "trf")
    return res2, inl


def measure_ball(rgb: np.ndarray, x0: float, y0: float, r0: float, *, extent: float | None = None,
                 allowed: np.ndarray | None = None, params: EdgeParams = EdgeParams()) -> BallMeasurement:
    """Measure the ball centre in ONE RGB uint8 image from a coarse detection. Pure, deterministic."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an RGB uint8 array of shape (H, W, 3)")
    if not (all(math.isfinite(v) for v in (x0, y0, r0)) and r0 > 1.0):
        raise ValueError("need a finite coarse centre and r0 > 1 px")
    extent = max(float(extent) if extent is not None else 1.6 * r0, r0)
    img = rgb.astype(np.float64)
    cx, cy, r = float(x0), float(y0), float(r0)
    fit = inl = None
    px = py = angs = np.empty(0)
    for _ in range(max(1, params.n_iter)):
        px, py, angs = edge_points(img, cx, cy, r0, extent, params, allowed)
        if px.size < max(8, int(params.min_inlier_frac * params.n_rays)):
            return _rejected(f"too few edge points ({px.size} of {params.n_rays} rays)", params.n_rays, int(px.size))
        out = _fit(px, py, (cx, cy, r), params)
        if out is None:
            return _rejected("robust outline fit failed", params.n_rays, int(px.size))
        fit, inl = out
        cx, cy = float(fit.x[0]), float(fit.x[1])
        if params.model == "circle":
            r = abs(float(fit.x[2]))
        elif params.model == "ellipse":
            r = math.sqrt(abs(fit.x[2] * fit.x[3]))
        else:
            r = abs(float(fit.x[2]))
    n_in = int(inl.sum())
    gap = _max_gap_deg(angs[inl])
    if n_in < params.min_inlier_frac * params.n_rays:
        return _rejected(f"too few inlier edge points ({n_in} of {params.n_rays} rays)", params.n_rays, int(px.size), n_in, gap)
    if gap > params.max_gap_deg:
        return _rejected(f"edge coverage has a {gap:.0f} deg gap (> {params.max_gap_deg:.0f}): outline occluded or "
                         "truncated", params.n_rays, int(px.size), n_in, gap)
    p = fit.x
    blur_l = None
    if params.model == "circle":
        a = b = abs(float(p[2]))
        phi, d_idx, diam = 0.0, 2, 2 * abs(float(p[2]))
    elif params.model == "ellipse":
        a, b, phi = abs(float(p[2])), abs(float(p[3])), float(p[4])
        d_idx = 3 if abs(p[3]) <= abs(p[2]) else 2
        if b > a:
            a, b, phi = b, a, phi + math.pi / 2
        diam = 2 * b
    else:
        b, blur_l, phi = abs(float(p[2])), abs(float(p[3])), float(p[4])
        a, d_idx, diam = b + blur_l, 2, 2 * b
    if b < 2.0 or a / b > params.max_axis_ratio:
        return _rejected(f"implausible outline shape (half-extents {a:.2f}, {b:.2f} px)", params.n_rays, int(px.size),
                         n_in, gap)
    if math.hypot(cx - x0, cy - y0) > extent:
        return _rejected("fitted centre is outside the coarse detection", params.n_rays, int(px.size), n_in, gap)
    resid = fit.fun
    theta = np.arctan2(py[inl] - cy, px[inl] - cx)
    asym = _odd_asymmetry(theta, resid)
    if not asym <= params.max_asym_px:
        return _rejected(f"asymmetric outline (odd-harmonic amplitude {asym:.2f} px > {params.max_asym_px}): occluded, "
                         "truncated or a bounce during the exposure", params.n_rays, int(px.size), n_in, gap)
    dof = max(1, n_in - p.size)
    s2 = float(resid @ resid) / dof
    J = fit.jac
    cov = np.linalg.pinv(J.T @ J, rcond=1e-10) * s2
    perimeter = n_in * (2 * math.pi * math.sqrt(a * b) / params.n_rays)
    n_eff = max(p.size + 1.0, min(float(n_in), perimeter / params.corr_len_px))
    cov = cov * (n_in / n_eff)
    extra2 = params.sys_floor_px ** 2 + (params.asym_gain * asym) ** 2
    sx = math.sqrt(max(cov[0, 0], 0.0) + extra2)
    sy = math.sqrt(max(cov[1, 1], 0.0) + extra2)
    sd = math.sqrt(4 * max(cov[d_idx, d_idx], 0.0) + (params.diam_sys_rel * diam) ** 2)
    ang = math.degrees(phi) % 180.0
    return BallMeasurement(MEASURED, None, cx, cy, sx, sy, float(cov[0, 1]), a, b, ang, diam, sd,
                           params.n_rays, int(px.size), n_in, gap, float(math.sqrt(s2)), asym, blur_l)
