"""B1c-1: deterministic kinematic fitting of 2-D trajectories.

Pure numerical analysis of a validated TrajectorySet. This module never sees a scene
interpretation or any model output; B1c-2 (physics_review) compares interpretations
with what is computed here.

Coordinates and units: image pixels and seconds. Origin top-left, +x to the right,
+y DOWN the image (so increasing y is downward image motion). Nothing is converted to
metres and no physical cause is asserted: a uniform acceleration is reported as such
(it may or may not be gravity), a velocity discontinuity is an impulse/contact
candidate (no surface identity is inferred), and a turning point such as an apex is a
kinematic feature, not an event.

Time: exact Fraction timestamps are differenced exactly and converted to float seconds
(relative to the first observation) only at the numerical fitting boundary, so
irregular frame timing is handled exactly.

Models (all fitted jointly on x and y, which share any contact time and drag rate;
each axis has its own coefficients). With d = max(0, t - t_c):

    constant_velocity               q = q0 + v0 t
    constant_acceleration           q = q0 + v0 t + a t^2/2
    impulse_constant_velocity       q = q0 + v0 t + dv d
    impulse_uniform_acceleration    q = q0 + v0 t + a t^2/2 + dv d
    impulse_split_acceleration      ... + da d^2/2        (diagnostic: is a the same after contact?)
    two_impulses_uniform_acceleration                     (diagnostic: is one contact enough?)
    linear_drag, linear_drag_impulse (constant acceleration / one contact, plus drag k >= 0)
        q'' = a - k q', solved exactly: q = q0 + v0 t phi1(kt) + a t^2 phi2(kt) + dv d phi1(kd)
        with phi1(z) = (1 - e^-z)/z and phi2(z) = (z - 1 + e^-z)/z^2 (stable as z -> 0)

Position is continuous at a contact and the same acceleration acts before and after it:
a bounce is a velocity jump, never a change of the acceleration. For fixed contact time
and drag rate every model is linear in its coefficients (exact least squares); the
contact time is profiled over every admissible gap between samples and the drag rate is
profiled over k >= 0.

Statistics (see NoiseModel):
- supplied sigma (common or per axis, px): -2 ln L = chi^2 = sum r^2 / sigma^2;
  BIC = chi^2 + p ln N, with N scalar observations (2 per sample) and p all fitted
  parameters (coefficients + contact times + drag rate). Absolute adequacy is tested
  with the chi^2 tail probability.
- estimated sigma (per axis, profiled out): -2 ln L = sum_axis n ln(RSS_axis / n);
  p additionally counts the two noise parameters. Absolute adequacy cannot be
  calibrated from the same residuals and is reported as "not_assessable".
- P7 per-observation weights (weights=..., needs sigma_x/sigma_y on every observation, e.g.
  from the outline measurement of decompile.ball_measure):
    "per_observation"           sigma_x/sigma_y of each observation are ABSOLUTE 1-sigma
                                measurement errors: weighted least squares, -2 ln L = chi^2,
                                absolute adequacy by the chi^2 tail probability.
    "per_observation_relative"  sigma_x/sigma_y give only RELATIVE precision; the per-axis scale
                                is profiled out as in the estimated mode (weighted RSS), so a
                                mis-calibrated absolute uncertainty cannot make a model look
                                adequate. Absolute adequacy is "not_assessable".
  With weights=None (default) every result is numerically identical to the unweighted fitter.
- A numerical variance floor (NUMERICAL_FLOOR_REL x coordinate scale) keeps ln(RSS)
  finite for exact synthetic data. It is a numerical safeguard, not a measurement
  uncertainty, and every use of it is reported.

Verdict logic is explicit and conservative (see the threshold constants): an added
effect is "supported" only with very strong evidence (Delta BIC > 10), declared
"not_supported" only when positively disfavoured (Delta BIC < -2) and identifiable,
and "inconclusive" otherwise. If the selected model is inadequate (chi^2 tail, outliers,
or a further velocity discontinuity), every verdict is downgraded to "inconclusive"
and the verdict it would otherwise have had is kept as `provisional`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2 as _chi2

from decompile.trajectories import ObjectTrajectory, TrajectorySet

METHOD_VERSION = "decompile.kinematic_fit/1"
AXES = ("x", "y")
# Image-direction words for the sign of a velocity component (+y is DOWN the image).
DIRECTION_WORDS = {"x": {1: "right", -1: "left"}, "y": {1: "down", -1: "up"}}

# ------------------------------------------------------------------ thresholds (documented)

# Delta BIC scale of Kass & Raftery (1995), dimensionless: > 10 "very strong", 2-6 "positive".
# Claiming an added effect requires very strong evidence; declining one requires positive
# evidence against it (plus identifiability where stated). Between the two: inconclusive.
BIC_DECISIVE = 10.0
BIC_POSITIVE = 2.0
# |estimate| / standard error for "significantly non-zero" (about 0.3% two-sided false alarm
# under Gaussian errors), dimensionless.
Z_SIGNIFICANT = 3.0
# Observations needed on each side of a contact to determine that segment's position,
# velocity and acceleration (3 coefficients).
MIN_SEGMENT_OBS = 3
# Profile-likelihood interval for the contact time: Delta(-2 ln L) <= 1, about 68% if the
# Gaussian / regular asymptotics held. Approximate: the profile is non-smooth at sample times.
PROFILE_DELTA = 1.0
# One-sided 95% likelihood-ratio bound for the drag rate at the k >= 0 boundary:
# chi-bar-squared critical value = chi2.ppf(0.90, 1).
DRAG_UPPER_DELTA = float(_chi2.ppf(0.90, 1))
# Drag is "not identifiable" when the data cannot exclude k*T > this value: a velocity loss of
# about 1 - e^-0.1 = 9.5% over the observed span T. Dimensionless.
DRAG_MAX_UNDETECTED_KT = 0.1
# Largest drag rate searched, as k*T (velocity decays by e^-20 over the span: fully stopped).
DRAG_MAX_KT = 20.0
DRAG_MIN_KT = 1e-6  # smallest positive grid point, as k*T
DRAG_GRID = 40
# Chi^2 tail probability below which a supplied-sigma fit is inadequate (dimensionless).
ADEQUACY_P_MIN = 1e-3
# |residual| / scale above which an observation is an outlier candidate: P(|z|>5) = 5.7e-7
# under Gaussian errors, so < 1e-3 expected false flags for the 1200 coordinates allowed.
OUTLIER_Z = 5.0
MAD_TO_SIGMA = 1.4826  # Gaussian consistency factor of the median absolute deviation
# Numerical variance floor, relative to max(1 px, max |coordinate|): far above float64
# round-off of these fits (~1e-13) and far below any real measurement precision (>~1e-3 px).
NUMERICAL_FLOOR_REL = 1e-9
CONTACT_GRID = 17  # deterministic grid points per gap before bounded refinement

# Model names
CV = "constant_velocity"
CA = "constant_acceleration"
IMPULSE_CV = "impulse_constant_velocity"
IMPULSE = "impulse_uniform_acceleration"
IMPULSE_SPLIT = "impulse_split_acceleration"
IMPULSE2 = "two_impulses_uniform_acceleration"
DRAG = "linear_drag"
DRAG_IMPULSE = "linear_drag_impulse"

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
NOT_SUPPORTED = "not_supported"
INCONCLUSIVE = "inconclusive"
NOT_EVALUATED = "not_evaluated"

THRESHOLDS = {
    "BIC_DECISIVE": BIC_DECISIVE, "BIC_POSITIVE": BIC_POSITIVE, "Z_SIGNIFICANT": Z_SIGNIFICANT,
    "MIN_SEGMENT_OBS": MIN_SEGMENT_OBS, "PROFILE_DELTA": PROFILE_DELTA, "DRAG_UPPER_DELTA": DRAG_UPPER_DELTA,
    "DRAG_MAX_UNDETECTED_KT": DRAG_MAX_UNDETECTED_KT, "DRAG_MAX_KT": DRAG_MAX_KT,
    "ADEQUACY_P_MIN": ADEQUACY_P_MIN, "OUTLIER_Z": OUTLIER_Z, "NUMERICAL_FLOOR_REL": NUMERICAL_FLOOR_REL,
}

LIMITATIONS = (
    "Pixels and seconds only; no calibration to metres and no physical cause is inferred.",
    "At most one contact is fitted; a further velocity discontinuity is detected and reported as "
    "model inadequacy, not fitted.",
    "Contacts are instantaneous velocity jumps with continuous position; finite-duration or deforming "
    "contact, spin and friction effects are not modelled separately.",
    "Errors are assumed independent and Gaussian with one standard deviation per axis.",
    "Contact-time intervals are approximate profile-likelihood intervals; parameter standard errors are "
    "local approximations conditional on the fitted contact time and drag rate.",
    "Linear drag shares one rate k >= 0 between axes. It is fitted on both the constant-acceleration and the "
    "one-contact model and decided within the family selected by the impulse verdict. Quadratic drag is not "
    "evaluated.",
    "Turning points are computed from the drag-free fit and are not reported when linear drag is supported.",
    "Surface identity (floor, ceiling, wall) is never inferred; only the image direction of a velocity "
    "reversal is reported.",
    "Objects are analysed independently; interactions between objects are not evaluated.",
)


class KinematicFitError(ValueError):
    """Invalid input to analyze_kinematics."""


# ------------------------------------------------------------------ data structures


@dataclass(frozen=True)
class NoiseModel:
    mode: str  # "supplied" | "estimated"
    sigma_px: tuple[float, float] | None  # (x, y) measurement standard deviation when supplied
    note: str


@dataclass(frozen=True)
class FiniteDifference:
    frame_a: int
    frame_b: int
    dt_s: Fraction  # exact
    vx_px_s: float
    vy_px_s: float


@dataclass(frozen=True)
class SecondDifference:
    frame: int
    ax_px_s2: float
    ay_px_s2: float


@dataclass(frozen=True)
class ModelFit:
    model: str
    n_scalar_obs: int
    n_params: int  # coefficients + nonlinear + noise parameters
    n_coefficients: int
    n_nonlinear: int
    n_noise_params: int
    params: dict[str, float]  # "x.q0", "y.a", "y.dv1", ...; plus "t_contact_s", "k_per_s"
    param_se: dict[str, float]  # local, conditional on the nonlinear parameters
    rss_px2: tuple[float, float]
    rms_px: tuple[float, float]
    max_abs_residual_px: float
    neg2_log_likelihood: float  # up to an additive constant common to all models
    bic: float
    chi2: float | None  # supplied sigma only
    chi2_dof: int | None
    chi2_p_value: float | None
    floored_axes: tuple[str, ...]  # axes whose variance hit the numerical floor
    residuals_x: tuple[float, ...]
    residuals_y: tuple[float, ...]
    # Local covariance (unit-variance matrix, per-axis variance, coefficient names); internal.
    covariance: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class AxisImpulse:
    axis: str
    v_before_px_s: float
    v_before_se: float
    v_after_px_s: float
    delta_v_px_s: float
    delta_v_se: float
    delta_v_z: float
    significant: bool
    speed_ratio: float | None  # |v_after| / |v_before|, when v_before is significant
    reversal: str | None  # image direction, e.g. "down_to_up"
    smooth_equivalent_accel_px_s2: float  # delta_v / gap width: what a smooth change would need
    ratio_to_uniform_accel: float | None


@dataclass(frozen=True)
class ContactEstimate:
    model: str
    t_contact_s: float
    gap_frames: tuple[int, int]
    gap_t: tuple[Fraction, Fraction]
    profile_interval_s: tuple[float, float]
    interval_at_search_boundary: bool
    profile_multimodal: bool
    axes: tuple[AxisImpulse, ...]
    note: str


@dataclass(frozen=True)
class TurningPoint:
    axis: str
    kind: str  # "y_minimum" (highest image point), "y_maximum", "x_minimum", "x_maximum"
    image_description: str
    t_s: float
    approx_se_s: float
    value_px: float
    between_frames: tuple[int, int]
    note: str


@dataclass(frozen=True)
class DragResult:
    base_model: str
    k_per_s: float
    k_upper_95_per_s: float | None  # None: not bounded within the searched range
    k_upper_times_span: float | None
    fit: ModelFit


@dataclass(frozen=True)
class AdditionalDiscontinuity:
    t_contact_s: float
    gap_frames: tuple[int, int]
    delta_bic: float


@dataclass(frozen=True)
class Verdict:
    name: str
    verdict: str
    reason: str
    evidence: dict[str, Any]
    provisional: str | None = None  # the verdict before an inadequacy downgrade


@dataclass(frozen=True)
class ObjectKinematics:
    object_id: str
    n_observations: int
    t_start_s: Fraction
    span_s: float
    finite_differences: tuple[FiniteDifference, ...]
    second_differences: tuple[SecondDifference, ...]
    fits: dict[str, ModelFit]
    selected_model: str
    contact: ContactEstimate | None
    drag: DragResult | None
    additional_discontinuity: AdditionalDiscontinuity | None
    outliers: tuple[tuple[str, int, float], ...]  # (axis, frame_index, standardized residual)
    adequacy: str  # "adequate" | "inadequate" | "not_assessable"
    adequacy_reasons: tuple[str, ...]
    turning_points: tuple[TurningPoint, ...]
    verdicts: dict[str, Verdict]


@dataclass(frozen=True)
class KinematicsReport:
    method_version: str
    noise: NoiseModel
    objects: tuple[ObjectKinematics, ...]
    thresholds: dict[str, float]
    limitations: tuple[str, ...]

    def get(self, object_id: str) -> ObjectKinematics:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        raise KeyError(object_id)


# ------------------------------------------------------------------ numerics


def _phi(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """phi1(z) = (1-e^-z)/z, phi2(z) = (z-1+e^-z)/z^2; series for small |z|."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    em = np.expm1(-zs)
    f1 = np.where(small, 1 - z / 2 + z * z / 6 - z ** 3 / 24, -em / zs)
    f2 = np.where(small, 0.5 - z / 6 + z * z / 24 - z ** 3 / 120, (zs + em) / (zs * zs))
    return f1, f2


def _design(t: np.ndarray, *, accel: bool, contacts: tuple[float, ...] = (), k: float = 0.0,
            split_at: float | None = None) -> tuple[np.ndarray, list[str]]:
    f1, f2 = _phi(k * t)
    cols, names = [np.ones_like(t), t * f1], ["q0", "v0"]
    if accel:
        cols.append(t * t * f2)
        names.append("a")
    for j, tc in enumerate(contacts):
        d = np.maximum(t - tc, 0.0)
        g1, _ = _phi(k * d)
        cols.append(d * g1)
        names.append(f"dv{j + 1}")
    if split_at is not None:
        d = np.maximum(t - split_at, 0.0)
        cols.append(0.5 * d * d)
        names.append("da")
    return np.column_stack(cols), names


WEIGHT_MODES = ("per_observation", "per_observation_relative")


class _Ctx:
    """One object's data and noise settings."""

    def __init__(self, obj: ObjectTrajectory, sigma: tuple[float, float] | None, weights: str | None = None) -> None:
        obs = obj.observations
        self.frames = [o.frame_index for o in obs]
        self.t0 = obs[0].t
        self.t_exact = [o.t - self.t0 for o in obs]  # exact Fractions, relative
        self.t = np.array([float(x) for x in self.t_exact])  # fitting boundary
        self.Q = np.column_stack([[o.x for o in obs], [o.y for o in obs]]).astype(float)
        self.n = len(obs)
        self.N = 2 * self.n
        self.sigma = sigma
        scale = np.maximum(1.0, np.max(np.abs(self.Q), axis=0))
        self.floor_var = (NUMERICAL_FLOOR_REL * scale) ** 2
        self.weights = weights
        self.S: np.ndarray | None = None  # per-observation sigma (n, 2) in the weighted modes
        if weights is not None:
            if weights not in WEIGHT_MODES:
                raise KinematicFitError(f"weights must be one of {WEIGHT_MODES}")
            if sigma is not None:
                raise KinematicFitError("give either position_sigma_px or weights, not both")
            missing = [o.frame_index for o in obs if o.sigma_x is None or o.sigma_y is None]
            if missing:
                raise KinematicFitError(f"weights={weights!r} needs sigma_x/sigma_y on every observation; missing on "
                                        f"frames {missing[:10]}")
            self.S = np.column_stack([[o.sigma_x for o in obs], [o.sigma_y for o in obs]]).astype(float)
            self.floor_var = self.floor_var / np.square(np.median(self.S, axis=0))

    @property
    def absolute(self) -> bool:
        """True when the noise level is known (supplied sigma or absolute per-observation sigma)."""
        return self.sigma is not None or self.weights == "per_observation"

    @property
    def estimated(self) -> bool:
        return not self.absolute

    def _wfit(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Weighted least squares per axis: coefficients (ncols, 2), whitened RSS (2,), whitened designs."""
        coefs, rss, designs = [], [], []
        for i in range(2):
            w = 1.0 / self.S[:, i]
            Aw = A * w[:, None]
            c, *_ = np.linalg.lstsq(Aw, self.Q[:, i] * w, rcond=None)
            r = (self.Q[:, i] - A @ c) * w
            coefs.append(c)
            rss.append(float(r @ r))
            designs.append(Aw)
        return np.column_stack(coefs), np.array(rss), designs

    def rss(self, A: np.ndarray) -> np.ndarray:
        if self.S is not None:
            return self._wfit(A)[1]
        coef, *_ = np.linalg.lstsq(A, self.Q, rcond=None)
        r = self.Q - A @ coef
        return np.sum(r * r, axis=0)

    def neg2ll(self, rss: np.ndarray) -> float:
        if self.S is not None and self.absolute:
            return float(np.sum(rss))  # whitened: chi^2
        if self.sigma is not None:
            return float(np.sum(rss / np.square(self.sigma)))
        var = np.maximum(rss / self.n, self.floor_var)
        return float(self.n * np.sum(np.log(var)))

    def objective(self, A: np.ndarray) -> float:
        return self.neg2ll(self.rss(A))


def _finish(ctx: _Ctx, model: str, A: np.ndarray, names: list[str], nonlinear: dict[str, float]) -> ModelFit | None:
    """Exact least-squares fit for fixed nonlinear parameters, or None if not evaluable."""
    ncols = A.shape[1]
    if ctx.n - ncols < 1:
        return None  # no residual degrees of freedom
    if ctx.S is not None:
        return _finish_weighted(ctx, model, A, names, nonlinear)
    coef, _, rank, _ = np.linalg.lstsq(A, ctx.Q, rcond=None)
    if rank < ncols:
        return None  # coefficients not identifiable
    resid = ctx.Q - A @ coef
    rss = np.sum(resid * resid, axis=0)
    n_noise = 2 if ctx.estimated else 0
    p = 2 * ncols + len(nonlinear) + n_noise
    neg2ll = ctx.neg2ll(rss)
    bic = neg2ll + p * math.log(ctx.N)
    floored: tuple[str, ...] = ()
    if ctx.sigma is not None:
        s2 = np.square(np.asarray(ctx.sigma, dtype=float))
        chi2 = float(np.sum(rss / s2))
        dof = ctx.N - (2 * ncols + len(nonlinear))
        pval = float(_chi2.sf(chi2, dof)) if dof > 0 else None
    else:
        unbiased = rss / (ctx.n - ncols)
        floored = tuple(ax for i, ax in enumerate(AXES) if rss[i] / ctx.n <= ctx.floor_var[i])
        s2 = np.maximum(unbiased, ctx.floor_var)
        chi2, dof, pval = None, None, None
    cov_unit = np.linalg.inv(A.T @ A)
    params: dict[str, float] = {}
    se: dict[str, float] = {}
    for i, ax in enumerate(AXES):
        for j, name in enumerate(names):
            params[f"{ax}.{name}"] = float(coef[j, i])
            se[f"{ax}.{name}"] = float(math.sqrt(max(cov_unit[j, j], 0.0) * s2[i]))
    params.update(nonlinear)
    return ModelFit(
        model=model, n_scalar_obs=ctx.N, n_params=p, n_coefficients=2 * ncols, n_nonlinear=len(nonlinear),
        n_noise_params=n_noise, params=params,
        param_se=se, rss_px2=(float(rss[0]), float(rss[1])),
        rms_px=(math.sqrt(rss[0] / ctx.n), math.sqrt(rss[1] / ctx.n)),
        max_abs_residual_px=float(np.max(np.abs(resid))), neg2_log_likelihood=neg2ll, bic=bic,
        chi2=chi2, chi2_dof=dof, chi2_p_value=pval, floored_axes=floored,
        residuals_x=tuple(float(v) for v in resid[:, 0]), residuals_y=tuple(float(v) for v in resid[:, 1]),
        covariance=(cov_unit, s2, tuple(names)),
    )


def _finish_weighted(ctx: _Ctx, model: str, A: np.ndarray, names: list[str], nonlinear: dict[str, float]) -> ModelFit | None:
    """P7: weighted least squares with per-observation sigma (see the module docstring)."""
    ncols = A.shape[1]
    if np.linalg.matrix_rank(A) < ncols:
        return None
    coef, rss_w, designs = ctx._wfit(A)
    resid = ctx.Q - A @ coef
    n_noise = 0 if ctx.absolute else 2
    p = 2 * ncols + len(nonlinear) + n_noise
    neg2ll = ctx.neg2ll(rss_w)
    bic = neg2ll + p * math.log(ctx.N)
    floored: tuple[str, ...] = ()
    if ctx.absolute:
        s2 = np.ones(2)
        chi2 = float(np.sum(rss_w))
        dof = ctx.N - (2 * ncols + len(nonlinear))
        pval = float(_chi2.sf(chi2, dof)) if dof > 0 else None
    else:
        unbiased = rss_w / (ctx.n - ncols)
        floored = tuple(ax for i, ax in enumerate(AXES) if rss_w[i] / ctx.n <= ctx.floor_var[i])
        s2 = np.maximum(unbiased, ctx.floor_var)
        chi2, dof, pval = None, None, None
    cov_units = [np.linalg.inv(Aw.T @ Aw) for Aw in designs]
    params: dict[str, float] = {}
    se: dict[str, float] = {}
    for i, ax in enumerate(AXES):
        for j, name in enumerate(names):
            params[f"{ax}.{name}"] = float(coef[j, i])
            se[f"{ax}.{name}"] = float(math.sqrt(max(cov_units[i][j, j], 0.0) * s2[i]))
    params.update(nonlinear)
    rms = np.sqrt(np.sum(resid * resid, axis=0) / ctx.n)
    return ModelFit(
        model=model, n_scalar_obs=ctx.N, n_params=p, n_coefficients=2 * ncols, n_nonlinear=len(nonlinear),
        n_noise_params=n_noise, params=params, param_se=se, rss_px2=(float(rss_w[0]), float(rss_w[1])),
        rms_px=(float(rms[0]), float(rms[1])), max_abs_residual_px=float(np.max(np.abs(resid))),
        neg2_log_likelihood=neg2ll, bic=bic, chi2=chi2, chi2_dof=dof, chi2_p_value=pval, floored_axes=floored,
        residuals_x=tuple(float(v) for v in resid[:, 0]), residuals_y=tuple(float(v) for v in resid[:, 1]),
        covariance=(cov_units, s2, tuple(names)),
    )


def _lin_se(fit: ModelFit, axis: str, weights: dict[str, float]) -> float:
    """Local standard error of a linear combination of one axis's coefficients."""
    cov_unit, s2, names = fit.covariance
    i = AXES.index(axis)
    if isinstance(cov_unit, list):  # weighted fit: one covariance per axis
        cov_unit = cov_unit[i]
    g = np.array([weights.get(n, 0.0) for n in names])
    return float(math.sqrt(max(g @ cov_unit @ g, 0.0) * s2[i]))


# ------------------------------------------------------------------ 1-D profile search


@dataclass
class _Search:
    best_value: float
    best_x: float
    best_gap: int
    samples: list[tuple[float, float]]  # (x, objective), sorted by x
    lo: float
    hi: float


def _search(objective: Callable[[float], float], gaps: list[tuple[int, float, float]]) -> _Search | None:
    """Grid + bounded refinement in each gap [a, b]; deterministic."""
    samples: list[tuple[float, float]] = []
    best: tuple[float, float, int] | None = None
    for gap, a, b in gaps:
        xs = np.linspace(a, b, CONTACT_GRID)
        vals = [objective(float(x)) for x in xs]
        samples += list(zip((float(x) for x in xs), vals))
        i = int(np.argmin(vals))
        lo, hi = float(xs[max(i - 1, 0)]), float(xs[min(i + 1, len(xs) - 1)])
        cand = (vals[i], float(xs[i]))
        if hi > lo:
            res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-13 * max(1.0, abs(b))})
            if res.fun < cand[0]:
                cand = (float(res.fun), float(res.x))
                samples.append((cand[1], cand[0]))
        if best is None or cand[0] < best[0]:
            best = (cand[0], cand[1], gap)
    if best is None:
        return None
    samples.sort()
    return _Search(best[0], best[1], best[2], samples, gaps[0][1], gaps[-1][2])


def _profile_interval(objective: Callable[[float], float], s: _Search, delta: float) -> tuple[float, float, bool, bool]:
    """Contiguous region {x : objective(x) <= min + delta} around the minimum."""
    thr = s.best_value + delta
    xs = [x for x, _ in s.samples]
    vs = [v for _, v in s.samples]
    i0 = min(range(len(xs)), key=lambda i: (abs(xs[i] - s.best_x), xs[i]))
    f = lambda x: objective(x) - thr  # noqa: E731

    def walk(step: int) -> tuple[float, bool]:
        i = i0
        while 0 <= i + step < len(xs) and vs[i + step] <= thr:
            i += step
        if not 0 <= i + step < len(xs):
            return xs[i], True
        a, b = sorted((xs[i], xs[i + step]))
        try:
            return float(brentq(f, a, b, xtol=1e-14)), False
        except ValueError:
            return xs[i], False

    lo, at_lo = walk(-1)
    hi, at_hi = walk(+1)
    multimodal = any(v <= thr and not (lo <= x <= hi) for x, v in s.samples)
    return lo, hi, at_lo or at_hi, multimodal


def _contact_gaps(ctx: _Ctx, exclude: tuple[int, ...] = ()) -> list[tuple[int, float, float]]:
    """Gaps i (between samples i, i+1) leaving >= MIN_SEGMENT_OBS samples on each side."""
    out = []
    for i in range(MIN_SEGMENT_OBS - 1, ctx.n - MIN_SEGMENT_OBS):
        if any(abs(i - g) < MIN_SEGMENT_OBS for g in exclude):
            continue
        out.append((i, float(ctx.t[i]), float(ctx.t[i + 1])))
    return out


# ------------------------------------------------------------------ per-object analysis


def _fit_contact_model(ctx: _Ctx, model: str, accel: bool, gaps):
    if not gaps:
        return None, None, None

    def obj(tc: float) -> float:
        return ctx.objective(_design(ctx.t, accel=accel, contacts=(tc,))[0])

    s = _search(obj, gaps)
    if s is None:
        return None, None, None
    A, names = _design(ctx.t, accel=accel, contacts=(s.best_x,))
    fit = _finish(ctx, model, A, names, {"t_contact_s": s.best_x})
    return fit, s, obj


def _drag(ctx: _Ctx, name: str, base_model: str, *, accel: bool,
          contact_gap: tuple[int, float, float] | None) -> DragResult | None:
    """Profile the drag rate k >= 0; the contact time is re-optimised within its gap for every k."""
    span = float(ctx.t[-1])
    if span <= 0:
        return None

    def best_tc(k: float) -> tuple[float, float | None]:
        if contact_gap is None:
            return ctx.objective(_design(ctx.t, accel=accel, k=k)[0]), None
        s = _search(lambda tc: ctx.objective(_design(ctx.t, accel=accel, contacts=(tc,), k=k)[0]), [contact_gap])
        return s.best_value, s.best_x

    obj = lambda k: best_tc(k)[0]  # noqa: E731
    ks = [0.0] + list(np.geomspace(DRAG_MIN_KT / span, DRAG_MAX_KT / span, DRAG_GRID))
    vals = [obj(float(k)) for k in ks]
    i = int(np.argmin(vals))
    k_hat, v_hat = float(ks[i]), vals[i]
    lo, hi = float(ks[max(i - 1, 0)]), float(ks[min(i + 1, len(ks) - 1)])
    if hi > lo:
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12 / span})
        if res.fun < v_hat:
            k_hat, v_hat = float(res.x), float(res.fun)
    thr = v_hat + DRAG_UPPER_DELTA
    k_up = None
    prev = k_hat
    for k, v in zip(ks, vals):
        if k <= k_hat:
            continue
        if v >= thr:
            try:
                k_up = float(brentq(lambda kk: obj(kk) - thr, prev, float(k), xtol=1e-14 / span))
            except ValueError:
                k_up = float(k)
            break
        prev = float(k)
    _, tc = best_tc(k_hat)
    contacts = () if tc is None else (tc,)
    A, names = _design(ctx.t, accel=accel, contacts=contacts, k=k_hat)
    nonlinear = {"k_per_s": k_hat}
    if tc is not None:
        nonlinear["t_contact_s"] = tc
    fit = _finish(ctx, name, A, names, nonlinear)
    if fit is None:
        return None
    return DragResult(base_model, k_hat, k_up, None if k_up is None else k_up * span, fit)


def _finite_differences(obj: ObjectTrajectory) -> tuple[tuple[FiniteDifference, ...], tuple[SecondDifference, ...]]:
    obs = obj.observations
    fds = []
    for a, b in zip(obs, obs[1:]):
        dt = b.t - a.t
        fds.append(FiniteDifference(a.frame_index, b.frame_index, dt, (b.x - a.x) / float(dt), (b.y - a.y) / float(dt)))
    sds = []
    for i in range(1, len(obs) - 1):
        span = float(obs[i + 1].t - obs[i - 1].t)
        sds.append(SecondDifference(obs[i].frame_index, 2 * (fds[i].vx_px_s - fds[i - 1].vx_px_s) / span,
                                    2 * (fds[i].vy_px_s - fds[i - 1].vy_px_s) / span))
    return tuple(fds), tuple(sds)


def _axis_impulses(ctx: _Ctx, fit: ModelFit, gap: int, accel: bool) -> tuple[AxisImpulse, ...]:
    tc = fit.params["t_contact_s"]
    width = float(ctx.t[gap + 1] - ctx.t[gap])
    out = []
    for ax in AXES:
        v0, a = fit.params[f"{ax}.v0"], fit.params.get(f"{ax}.a", 0.0)
        dv, dv_se = fit.params[f"{ax}.dv1"], fit.param_se[f"{ax}.dv1"]
        vb = v0 + a * tc
        vb_se = _lin_se(fit, ax, {"v0": 1.0, "a": tc if accel else 0.0})
        va = vb + dv
        z = abs(dv) / dv_se if dv_se > 0 else math.inf
        vb_sig = vb_se == 0 or abs(vb) > Z_SIGNIFICANT * vb_se
        va_se = _lin_se(fit, ax, {"v0": 1.0, "a": tc if accel else 0.0, "dv1": 1.0})
        va_sig = va_se == 0 or abs(va) > Z_SIGNIFICANT * va_se
        reversal = None
        if vb_sig and va_sig and vb != 0 and va != 0 and math.copysign(1, vb) != math.copysign(1, va):
            words = DIRECTION_WORDS[ax]
            reversal = f"{words[int(math.copysign(1, vb))]}_to_{words[int(math.copysign(1, va))]}"
        a_se = fit.param_se.get(f"{ax}.a", 0.0)
        a_sig = accel and a != 0 and (a_se == 0 or abs(a) > Z_SIGNIFICANT * a_se)
        smooth = dv / width
        out.append(AxisImpulse(
            axis=ax, v_before_px_s=vb, v_before_se=vb_se, v_after_px_s=va, delta_v_px_s=dv, delta_v_se=dv_se,
            delta_v_z=z, significant=z > Z_SIGNIFICANT,
            speed_ratio=abs(va) / abs(vb) if vb_sig and vb != 0 else None, reversal=reversal,
            smooth_equivalent_accel_px_s2=smooth,
            ratio_to_uniform_accel=abs(smooth) / abs(a) if a_sig else None,
        ))
    return tuple(out)


def _turning_points(ctx: _Ctx, fit: ModelFit) -> tuple[TurningPoint, ...]:
    tc = fit.params.get("t_contact_s")
    bounds = [0.0, float(ctx.t[-1])] if tc is None else [0.0, tc, float(ctx.t[-1])]
    out = []
    for ax in AXES:
        a, a_se = fit.params.get(f"{ax}.a"), fit.param_se.get(f"{ax}.a")
        if a is None or a == 0 or not (a_se == 0 or abs(a) > Z_SIGNIFICANT * a_se):
            continue
        for seg in range(len(bounds) - 1):
            after = seg == 1
            v_off = fit.params[f"{ax}.v0"] + (fit.params[f"{ax}.dv1"] if after else 0.0)
            ts = -v_off / a
            if not bounds[seg] < ts < bounds[seg + 1]:
                continue
            q = (fit.params[f"{ax}.q0"] + fit.params[f"{ax}.v0"] * ts + 0.5 * a * ts * ts
                 + (fit.params[f"{ax}.dv1"] * (ts - tc) if after else 0.0))
            grad = {"v0": -1.0 / a, "a": v_off / (a * a)}
            if after:
                grad["dv1"] = -1.0 / a
            se = _lin_se(fit, ax, grad)
            # Report only turning points resolved by the data: at least one observation on each
            # side within the segment, and Z_SIGNIFICANT standard errors from the segment bounds
            # (a release from rest at the first sample is not a resolved turning point).
            lo_b, hi_b = bounds[seg], bounds[seg + 1]
            inside = ctx.t[(ctx.t >= lo_b) & (ctx.t <= hi_b)]
            if not (np.any(inside < ts) and np.any(inside > ts)):
                continue
            if min(ts - lo_b, hi_b - ts) <= Z_SIGNIFICANT * se:
                continue
            if a > 0:
                kind = f"{ax}_minimum"
                desc = "highest point in the image (smallest y; +y is down)" if ax == "y" else "leftmost point"
            else:
                kind = f"{ax}_maximum"
                desc = "lowest point in the image (largest y; +y is down)" if ax == "y" else "rightmost point"
            k = int(np.searchsorted(ctx.t, ts))
            out.append(TurningPoint(
                axis=ax, kind=kind, image_description=desc, t_s=float(ctx.t0) + ts, approx_se_s=se,
                value_px=float(q), between_frames=(ctx.frames[k - 1], ctx.frames[k]),
                note="smooth kinematic turning point of the fitted model; not an event or contact. "
                     "Standard error is a local delta-method approximation conditional on the contact time.",
            ))
    return tuple(out)


def _outliers(ctx: _Ctx, fit: ModelFit) -> tuple[tuple[str, int, float], ...]:
    out = []
    for i, (ax, res) in enumerate(zip(AXES, (fit.residuals_x, fit.residuals_y))):
        r = np.asarray(res)
        if ctx.S is not None:  # P7 weighted: standardise each residual by its own sigma
            r = r / ctx.S[:, i]
            scale = 1.0 if ctx.absolute else max(MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r)))),
                                                 math.sqrt(ctx.floor_var[i]))
        elif ctx.sigma is not None:
            scale = ctx.sigma[i]
        else:
            scale = max(MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r)))), math.sqrt(ctx.floor_var[i]))
        for j, v in enumerate(r):
            z = abs(v) / scale
            if z > OUTLIER_Z:
                out.append((ax, ctx.frames[j], float(z)))
    return tuple(out)


def _v(name: str, verdict: str, reason: str, **evidence: Any) -> Verdict:
    return Verdict(name, verdict, reason, evidence)


def _nested(d: float | None, *, simple_name: str, complex_name: str) -> tuple[str, str]:
    """Verdict on the added structure of `complex` over `simple` from Delta BIC = BIC_s - BIC_c."""
    if d is None:
        return INCONCLUSIVE, "a model needed for the comparison could not be evaluated (insufficient data)"
    if d > BIC_DECISIVE:
        return SUPPORTED, f"{complex_name} is very strongly preferred over {simple_name} (Delta BIC = {d:.2f} > {BIC_DECISIVE})"
    if d < -BIC_POSITIVE:
        return NOT_SUPPORTED, f"{simple_name} is preferred over {complex_name} (Delta BIC = {d:.2f} < -{BIC_POSITIVE})"
    return INCONCLUSIVE, f"Delta BIC = {d:.2f} is between -{BIC_POSITIVE} and {BIC_DECISIVE}"


def _analyze_object(obj: ObjectTrajectory, sigma: tuple[float, float] | None,
                    weights: str | None = None) -> ObjectKinematics:
    ctx = _Ctx(obj, sigma, weights)
    fds, sds = _finite_differences(obj)
    fits: dict[str, ModelFit] = {}
    drags: dict[str, DragResult] = {}

    def add(fit: ModelFit | None) -> None:
        if fit is not None:
            fits[fit.model] = fit

    def bic(name: str) -> float | None:
        return fits[name].bic if name in fits else None

    def best_of(*names: str) -> str | None:
        avail = [n for n in names if n in fits]
        return min(avail, key=lambda n: (fits[n].bic, names.index(n))) if avail else None

    # ---- smooth family: no velocity discontinuity
    for model, accel in ((CV, False), (CA, True)):
        A, names = _design(ctx.t, accel=accel)
        add(_finish(ctx, model, A, names, {}))
    if CA in fits:
        dr = _drag(ctx, DRAG, CA, accel=True, contact_gap=None)
        if dr is not None:
            drags[DRAG] = dr
            add(dr.fit)

    # ---- contact family: one velocity discontinuity (impulse / contact candidate)
    gaps = _contact_gaps(ctx)
    searches: dict[str, tuple[_Search, Callable[[float], float]]] = {}
    gap_of: dict[str, tuple[int, float, float]] = {}
    for model, accel in ((IMPULSE_CV, False), (IMPULSE, True)):
        fit, s, f = _fit_contact_model(ctx, model, accel, gaps)
        add(fit)
        if fit is not None:
            searches[model] = (s, f)
            gap_of[model] = next(gp for gp in gaps if gp[0] == s.best_gap)
    if IMPULSE in fits:
        dr = _drag(ctx, DRAG_IMPULSE, IMPULSE, accel=True, contact_gap=gap_of[IMPULSE])
        if dr is not None:
            drags[DRAG_IMPULSE] = dr
            add(dr.fit)
            gap_of[DRAG_IMPULSE] = gap_of[IMPULSE]

    # ---- velocity discontinuity: best smooth model vs best contact model (drag on both sides)
    no_contact = best_of(CV, CA, DRAG)
    with_contact = best_of(IMPULSE_CV, IMPULSE, DRAG_IMPULSE)
    contact: ContactEstimate | None = None
    if with_contact is not None:
        search_model = IMPULSE_CV if with_contact == IMPULSE_CV else IMPULSE
        s, f = searches[search_model]
        lo, hi, at_edge, multi = _profile_interval(f, s, PROFILE_DELTA)
        g = gap_of[with_contact]
        cfit = fits[with_contact]
        contact = ContactEstimate(
            model=with_contact, t_contact_s=float(ctx.t0) + cfit.params["t_contact_s"],
            gap_frames=(ctx.frames[g[0]], ctx.frames[g[0] + 1]),
            gap_t=(obj.observations[g[0]].t, obj.observations[g[0] + 1].t),
            profile_interval_s=(float(ctx.t0) + lo, float(ctx.t0) + hi),
            interval_at_search_boundary=at_edge, profile_multimodal=multi,
            axes=_axis_impulses(ctx, cfit, g[0], with_contact != IMPULSE_CV),
            note="velocity discontinuity with continuous position and unchanged acceleration: an impulse/"
                 "contact candidate. Surface identity is not inferred. The interval is an approximate "
                 f"profile-likelihood interval (Delta(-2 ln L) <= {PROFILE_DELTA}) of the {search_model} model.",
        )

    verdicts: dict[str, Verdict] = {}
    if not gaps:
        verdicts["impulse"] = _v("impulse", INCONCLUSIVE,
                                 f"insufficient observations for a contact search: need at least "
                                 f"{2 * MIN_SEGMENT_OBS} ({MIN_SEGMENT_OBS} on each side of a contact)",
                                 n_observations=ctx.n)
    elif no_contact is None or with_contact is None:
        verdicts["impulse"] = _v("impulse", INCONCLUSIVE, "a model needed for the comparison could not be evaluated",
                                 n_observations=ctx.n)
    else:
        d = fits[no_contact].bic - fits[with_contact].bic
        sig = [a.axis for a in contact.axes if a.significant]
        evidence = dict(delta_bic=d, no_contact_model=no_contact, contact_model=with_contact,
                        t_contact_s=contact.t_contact_s, gap_frames=contact.gap_frames,
                        profile_interval_s=contact.profile_interval_s,
                        delta_v_px_s={a.axis: a.delta_v_px_s for a in contact.axes},
                        delta_v_z={a.axis: a.delta_v_z for a in contact.axes},
                        reversal={a.axis: a.reversal for a in contact.axes})
        verdict, reason = _nested(d, simple_name=no_contact, complex_name=with_contact)
        if verdict == SUPPORTED and contact.interval_at_search_boundary:
            verdict, reason = INCONCLUSIVE, ("the contact-time interval reaches the edge of the searchable range "
                                             f"(contacts need {MIN_SEGMENT_OBS} observations on each side); " + reason)
        elif verdict == SUPPORTED and not sig:
            verdict, reason = INCONCLUSIVE, f"no axis has |delta_v| > {Z_SIGNIFICANT} standard errors; " + reason
        elif verdict == SUPPORTED:
            reason += f"; significant velocity change on axis {sig}"
        verdicts["impulse"] = _v("impulse", verdict, reason, **evidence)
    impulse_ok = verdicts["impulse"].verdict == SUPPORTED

    # ---- linear drag, decided within the family chosen above
    if impulse_ok:
        base, drag_name = best_of(IMPULSE_CV, IMPULSE), DRAG_IMPULSE
    else:
        base, drag_name = best_of(CV, CA), DRAG
    drag = drags.get(drag_name)
    if drag is None or base is None:
        verdicts["linear_drag"] = _v("linear_drag", INCONCLUSIVE, "drag model not evaluable (insufficient data)",
                                     n_observations=ctx.n)
    else:
        d = fits[base].bic - drag.fit.bic
        ev = dict(delta_bic=d, base_model=base, drag_model=drag_name, k_per_s=drag.k_per_s,
                  k_upper_95_per_s=drag.k_upper_95_per_s, k_upper_times_span=drag.k_upper_times_span,
                  span_s=float(ctx.t[-1]))
        if d > BIC_DECISIVE:
            vd, rs = SUPPORTED, f"linear drag is very strongly preferred over {base} (Delta BIC = {d:.2f} > {BIC_DECISIVE})"
        elif drag.k_upper_times_span is None or drag.k_upper_times_span > DRAG_MAX_UNDETECTED_KT:
            vd, rs = INCONCLUSIVE, (f"drag is not identifiable: the data cannot exclude k*T > {DRAG_MAX_UNDETECTED_KT} "
                                    f"(about {100 * (1 - math.exp(-DRAG_MAX_UNDETECTED_KT)):.1f}% velocity loss over "
                                    "the span) at the one-sided 95% level")
        elif d < -BIC_POSITIVE:
            vd, rs = NOT_SUPPORTED, (f"{base} without drag is preferred (Delta BIC = {d:.2f}); one-sided 95% "
                                     f"upper bound k*T = {drag.k_upper_times_span:.3g}")
        else:
            vd, rs = INCONCLUSIVE, f"Delta BIC = {d:.2f} is between -{BIC_POSITIVE} and {BIC_DECISIVE}"
        verdicts["linear_drag"] = _v("linear_drag", vd, rs, **ev)
    drag_ok = verdicts["linear_drag"].verdict == SUPPORTED

    # ---- uniform acceleration (cause not identified)
    simple, complex_ = (IMPULSE_CV, IMPULSE) if impulse_ok else (CV, CA)
    d = None if bic(simple) is None or bic(complex_) is None else bic(simple) - bic(complex_)
    verdict, reason = _nested(d, simple_name=simple, complex_name=complex_)
    ev: dict[str, Any] = dict(delta_bic=d, compared=(simple, complex_))
    if complex_ in fits:
        ax_, ay_ = fits[complex_].params["x.a"], fits[complex_].params["y.a"]
        ev.update(a_px_s2={"x": ax_, "y": ay_},
                  a_se_px_s2={"x": fits[complex_].param_se["x.a"], "y": fits[complex_].param_se["y.a"]},
                  magnitude_px_s2=math.hypot(ax_, ay_),
                  direction_deg_image=math.degrees(math.atan2(ay_, ax_)),
                  direction_convention="0 deg = +x (right), 90 deg = +y (down the image)")
    reason += ". A uniform acceleration is reported; its cause (e.g. gravity) is not identified by the fit"
    if drag_ok:
        reason += ("; linear drag is supported, so the acceleration is not uniform: the value reported is the "
                   "drag-free fit, see linear_drag for the velocity-independent part")
    verdicts["uniform_acceleration"] = _v("uniform_acceleration", verdict, reason, **ev)
    accel_ok = verdicts["uniform_acceleration"].verdict == SUPPORTED

    # ---- constant velocity and stationarity
    alt = best_of(CA, IMPULSE_CV, IMPULSE, DRAG, DRAG_IMPULSE)
    if CV not in fits or alt is None:
        verdicts["constant_velocity"] = _v("constant_velocity", INCONCLUSIVE,
                                           "insufficient data to compare constant velocity with any alternative",
                                           n_observations=ctx.n)
    else:
        d = fits[CV].bic - fits[alt].bic
        if d > BIC_DECISIVE:
            vd, rs = CONTRADICTED, f"{alt} is very strongly preferred (Delta BIC = {d:.2f} > {BIC_DECISIVE})"
        elif d < -BIC_POSITIVE:
            vd, rs = SUPPORTED, f"constant velocity is preferred over {alt} (Delta BIC = {d:.2f} < -{BIC_POSITIVE})"
        else:
            vd, rs = INCONCLUSIVE, f"Delta BIC = {d:.2f} versus {alt} is between -{BIC_POSITIVE} and {BIC_DECISIVE}"
        verdicts["constant_velocity"] = _v("constant_velocity", vd, rs, delta_bic=d, best_alternative=alt,
                                           v_px_s={a: fits[CV].params[f"{a}.v0"] for a in AXES})
    cvv = verdicts["constant_velocity"].verdict
    if cvv == SUPPORTED:
        z = {a: (abs(fits[CV].params[f"{a}.v0"]) / fits[CV].param_se[f"{a}.v0"]
                 if fits[CV].param_se[f"{a}.v0"] > 0 else (0.0 if fits[CV].params[f"{a}.v0"] == 0 else math.inf))
             for a in AXES}
        moving = [a for a in AXES if z[a] > Z_SIGNIFICANT]
        verdicts["stationary"] = _v(
            "stationary", CONTRADICTED if moving else SUPPORTED,
            (f"significant constant velocity on axis {moving}" if moving
             else f"velocity within {Z_SIGNIFICANT} standard errors of zero on both axes"),
            v_px_s={a: fits[CV].params[f"{a}.v0"] for a in AXES}, v_z=z)
    elif cvv == CONTRADICTED:
        verdicts["stationary"] = _v("stationary", CONTRADICTED, "constant velocity (including rest) is contradicted",
                                    delta_bic=verdicts["constant_velocity"].evidence["delta_bic"])
    else:
        verdicts["stationary"] = _v("stationary", INCONCLUSIVE, "constant velocity itself is inconclusive",
                                    n_observations=ctx.n)

    # ---- acceleration continuity through the contact
    if impulse_ok and accel_ok and not drag_ok and IMPULSE in gap_of:
        g = gap_of[IMPULSE]
        s = _search(lambda tc: ctx.objective(_design(ctx.t, accel=True, contacts=(tc,), split_at=tc)[0]), [g])
        A, names = _design(ctx.t, accel=True, contacts=(s.best_x,), split_at=s.best_x)
        split_fit = _finish(ctx, IMPULSE_SPLIT, A, names, {"t_contact_s": s.best_x})
        add(split_fit)
        d = None if split_fit is None else fits[IMPULSE].bic - split_fit.bic
        if d is None:
            vd, rs = INCONCLUSIVE, "split-acceleration model not evaluable"
        elif d > BIC_DECISIVE:
            vd, rs = CONTRADICTED, f"the acceleration differs after the contact (Delta BIC = {d:.2f} > {BIC_DECISIVE})"
        elif d < -BIC_POSITIVE:
            vd, rs = SUPPORTED, f"one acceleration before and after the contact is preferred (Delta BIC = {d:.2f})"
        else:
            vd, rs = INCONCLUSIVE, f"Delta BIC = {d:.2f} is between -{BIC_POSITIVE} and {BIC_DECISIVE}"
        ev = dict(delta_bic=d)
        if split_fit is not None:
            ev.update(delta_a_px_s2={a: split_fit.params[f"{a}.da"] for a in AXES},
                      delta_a_se={a: split_fit.param_se[f"{a}.da"] for a in AXES})
        verdicts["acceleration_continuity"] = _v("acceleration_continuity", vd, rs, **ev)
    else:
        verdicts["acceleration_continuity"] = _v(
            "acceleration_continuity", NOT_EVALUATED,
            "requires a supported impulse and a supported uniform acceleration, without supported drag",
            impulse=verdicts["impulse"].verdict, uniform_acceleration=verdicts["uniform_acceleration"].verdict,
            linear_drag=verdicts["linear_drag"].verdict)

    verdicts["quadratic_drag"] = _v("quadratic_drag", NOT_EVALUATED, "quadratic drag is not implemented in B1c-1",
                                    n_observations=ctx.n)
    verdicts["surface_identity"] = _v(
        "surface_identity", NOT_EVALUATED,
        "a centroid trajectory alone cannot identify a surface (floor, ceiling, wall); only the image direction "
        "of a velocity reversal is reported", reversal=None if contact is None else {a.axis: a.reversal for a in contact.axes})

    # ---- selected model
    candidates = [n for n in (CV, CA, DRAG, IMPULSE_CV, IMPULSE, DRAG_IMPULSE) if n in fits]
    selected = min(candidates, key=lambda n: (fits[n].bic, candidates.index(n)))

    # ---- a further velocity discontinuity (one-contact adequacy), relative to the selected contact model
    extra = None
    if impulse_ok and selected in (IMPULSE_CV, IMPULSE, DRAG_IMPULSE):
        g1 = gap_of[selected]
        tc1 = fits[selected].params["t_contact_s"]
        k_sel = fits[selected].params.get("k_per_s", 0.0)
        accel_c = selected != IMPULSE_CV
        gaps2 = _contact_gaps(ctx, exclude=(g1[0],))
        if gaps2:
            def f2(tc: float) -> float:
                return ctx.objective(_design(ctx.t, accel=accel_c, contacts=tuple(sorted((tc1, tc))), k=k_sel)[0])

            s2 = _search(f2, gaps2)
            A, names = _design(ctx.t, accel=accel_c, contacts=tuple(sorted((tc1, s2.best_x))), k=k_sel)
            nonlinear = {"t_contact_s": tc1, "t_contact2_s": s2.best_x}
            if "k_per_s" in fits[selected].params:
                nonlinear["k_per_s"] = k_sel
            fit2 = _finish(ctx, IMPULSE2, A, names, nonlinear)
            if fit2 is not None:
                fits[IMPULSE2] = fit2
                d2 = fits[selected].bic - fit2.bic
                if d2 > BIC_DECISIVE:
                    gi = s2.best_gap
                    extra = AdditionalDiscontinuity(float(ctx.t0) + s2.best_x, (ctx.frames[gi], ctx.frames[gi + 1]), d2)

    # ---- adequacy and outliers
    outliers = _outliers(ctx, fits[selected])
    reasons: list[str] = []
    if ctx.absolute:
        p = fits[selected].chi2_p_value
        if p is not None and p < ADEQUACY_P_MIN:
            reasons.append(f"selected model {selected}: chi^2 = {fits[selected].chi2:.4g} on {fits[selected].chi2_dof} "
                           f"dof, p = {p:.3g} < {ADEQUACY_P_MIN}")
    if outliers:
        reasons.append(f"{len(outliers)} outlier candidate(s) with |residual| > {OUTLIER_Z} x scale: "
                       + ", ".join(f"{a} frame {f} ({z:.1f})" for a, f, z in outliers[:10]))
    if extra is not None:
        reasons.append(f"a further velocity discontinuity near t = {extra.t_contact_s:.6g} s (frames "
                       f"{extra.gap_frames}) is very strongly preferred (Delta BIC = {extra.delta_bic:.2f}); "
                       "the one-contact model is inadequate")
    if reasons:
        adequacy = "inadequate"
    elif not ctx.absolute:
        adequacy = "not_assessable"
        reasons.append("sigma was estimated from the same residuals, so absolute fit adequacy cannot be calibrated")
    else:
        adequacy = "adequate"
    if adequacy == "inadequate":
        for name, v in list(verdicts.items()):
            if v.verdict in (SUPPORTED, CONTRADICTED, NOT_SUPPORTED):
                verdicts[name] = Verdict(name, INCONCLUSIVE, "model inadequate (" + "; ".join(reasons) + "); "
                                         f"provisional verdict {v.verdict}: {v.reason}", v.evidence, v.verdict)

    # Turning points only from an adequate, drag-free fit whose uniform acceleration is supported.
    turning: tuple[TurningPoint, ...] = ()
    structural = IMPULSE if impulse_ok else CA
    if (structural in fits and verdicts["uniform_acceleration"].verdict == SUPPORTED
            and verdicts["linear_drag"].verdict != SUPPORTED):
        turning = _turning_points(ctx, fits[structural])

    return ObjectKinematics(
        object_id=obj.object_id, n_observations=ctx.n, t_start_s=ctx.t0, span_s=float(ctx.t[-1]),
        finite_differences=fds, second_differences=sds, fits=fits, selected_model=selected, contact=contact,
        drag=drag, additional_discontinuity=extra, outliers=outliers, adequacy=adequacy,
        adequacy_reasons=tuple(reasons), turning_points=turning, verdicts=verdicts,
    )


# ------------------------------------------------------------------ public entry point


def _noise(position_sigma_px: Any) -> tuple[NoiseModel, tuple[float, float] | None]:
    if position_sigma_px is None:
        return NoiseModel("estimated", None,
                          "per-axis sigma profiled out of the likelihood; absolute fit adequacy is not assessable. "
                          "A numerical variance floor (NUMERICAL_FLOOR_REL) is applied only to keep ln(RSS) finite "
                          "and is not a measurement uncertainty."), None
    values = position_sigma_px if isinstance(position_sigma_px, tuple) else (position_sigma_px, position_sigma_px)
    if len(values) != 2:
        raise KinematicFitError("position_sigma_px must be a number or an (x, y) pair")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise KinematicFitError(f"position_sigma_px must be positive and finite, got {v!r}")
    sigma = (float(values[0]), float(values[1]))
    return NoiseModel("supplied", sigma, "measurement standard deviation supplied by the caller (px, per axis)"), sigma


def analyze_kinematics(traj: TrajectorySet, *, position_sigma_px: float | tuple[float, float] | None = None,
                       weights: str | None = None) -> KinematicsReport:
    """Fit the kinematic models to every object of `traj` and return explicit verdicts.

    position_sigma_px: measurement standard deviation in px, common (a number) or per axis
    ((sigma_x, sigma_y)); None estimates it per axis from the data.
    weights (P7): None, "per_observation" or "per_observation_relative" (see the module docstring);
    uses the sigma_x/sigma_y of every observation. Not combinable with position_sigma_px.
    """
    if not isinstance(traj, TrajectorySet):
        raise KinematicFitError(f"expected a TrajectorySet, got {type(traj).__name__}")
    if weights is not None:
        if weights not in WEIGHT_MODES:
            raise KinematicFitError(f"weights must be one of {WEIGHT_MODES}")
        if position_sigma_px is not None:
            raise KinematicFitError("give either position_sigma_px or weights, not both")
        noise, sigma = NoiseModel(weights, None, (
            "per-observation 1-sigma measurement errors (sigma_x, sigma_y of each observation) used as ABSOLUTE "
            "weights; chi^2 adequacy" if weights == "per_observation" else
            "per-observation sigma_x/sigma_y used as RELATIVE weights; per-axis scale profiled out; absolute "
            "adequacy not assessable")), None
    else:
        noise, sigma = _noise(position_sigma_px)
    objects = tuple(_analyze_object(o, sigma, weights) for o in traj.objects)
    return KinematicsReport(METHOD_VERSION, noise, objects, dict(THRESHOLDS), LIMITATIONS)
