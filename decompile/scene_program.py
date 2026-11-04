"""P1 (A9-lite): executable scene program and deterministic event-driven simulator.

A SceneProgram is the smallest executable description of a single-body scene:

    Body                 initial position and velocity of one object at an exact time t0
    UniformAcceleration  one constant acceleration (ax, ay)
    AxisPlane            zero or more axis-aligned contact boundaries with a restitution e

Units and coordinates: pixels and seconds, image coordinates with the origin at the top
left, +x to the right and +y DOWN the image. Nothing is converted to metres.

Planes are INFERRED CONTACT BOUNDARIES: a plane is placed where a fitted velocity
discontinuity happened, at the fitted position of the contact. It is not a visually
observed surface, and calling it a floor, ceiling or wall would be an interpretation the
trajectory does not support. keep_side says which side of the boundary the body occupies:
"less" means the body stays at coordinate <= position_px.

simulate(program, times) runs the program forward, event by event:
- between contacts the motion is the closed form q + v t + a t^2/2;
- the next contact with each plane is the smallest positive root of the gap equation,
  found with a numerically stable quadratic formula; after a contact with a plane, the
  return time to that same plane is computed directly as -2 w / alpha, so a zero-time
  repeated contact cannot happen;
- at a contact the normal velocity becomes -e times its value and the tangential velocity
  is unchanged;
- if the next flight would last less than MIN_FLIGHT_S, the body comes to rest on the
  plane (normal velocity and normal acceleration become zero; it may still slide along the
  plane); contacts with two planes within SIMULTANEOUS_TOL_S are rejected as ambiguous;
  more than max_events contacts before the last requested time raise SimulationLimitError.

Query times are exact Fractions (>= t0) and are returned exactly; positions are evaluated
in float seconds relative to t0. Contact times are generally irrational and are reported as
float seconds.

program_from_kinematics(obj) builds a program from a B1c-1 kinematic fit of ONE object,
and refuses (ProgramUnavailable, with every reason) unless the fit supports exactly one
impulse with a reversal on one axis, a uniform acceleration, no drag, an adequate or
not-independently-assessable fit, a restitution in [0, 1], and a tangential velocity change
consistent with zero.

IMPORTANT: a program built from a fit reproduces that fit at the fitted observations by
construction. That is a consistency check of the simulator, not held-out predictive
validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Sequence

from decompile import kinematic_fit as kf
from decompile.trajectories import timestamp_from_json, timestamp_to_json

PROGRAM_VERSION = "decompile.scene_program/1"
UNITS = {"length": "px", "time": "s"}
COORDINATE_SYSTEM = {"origin": "top_left", "x_axis": "right", "y_axis": "down"}
AXES = ("x", "y")
KEEP_SIDES = ("less", "greater")
PLANE_BASIS = ("inferred contact boundary: placed at the fitted position of a fitted velocity discontinuity; "
               "not a visually observed surface")

# A flight shorter than this after a contact means the body is at rest on the plane. Seconds;
# far below any video frame interval (>~1e-3 s): the rest of such a bounce is shorter than
# a microsecond and its height alpha*T^2/8 is negligible (2.5e-10 px at 2000 px/s^2).
MIN_FLIGHT_S = 1e-6
# Contacts with two different planes closer in time than this are simultaneous (e.g. a
# corner) and rejected as ambiguous. Seconds; far below a frame interval.
SIMULTANEOUS_TOL_S = 1e-9
# Default cap on contacts before the last requested time (an elastic bounce never rests).
MAX_EVENTS = 10_000
# A position this far beyond a plane (px) means the region was left: a numerical bug guard.
POSITION_TOL_PX = 1e-6


class ProgramError(ValueError):
    """A scene program or its JSON form is invalid."""


class ProgramUnavailable(ValueError):
    """A kinematic fit does not support building a scene program. `reasons` lists every cause."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("no scene program: " + "; ".join(self.reasons))


class SimulationError(RuntimeError):
    """The simulation cannot continue deterministically."""


class SimultaneousContactError(SimulationError):
    pass


class SimulationLimitError(SimulationError):
    pass


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProgramError(f"{name} must be a finite number, got {value!r}")
    return float(value)


# ------------------------------------------------------------------ program


@dataclass(frozen=True)
class Body:
    object_id: str
    t0: Fraction
    x0: float
    y0: float
    vx0: float
    vy0: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ProgramError("object_id must be a non-empty string")
        if not isinstance(self.t0, Fraction) or self.t0 < 0:
            raise ProgramError(f"t0 must be a non-negative Fraction, got {self.t0!r}")
        for name in ("x0", "y0", "vx0", "vy0"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))


@dataclass(frozen=True)
class UniformAcceleration:
    ax: float
    ay: float

    def __post_init__(self) -> None:
        for name in ("ax", "ay"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))


@dataclass(frozen=True)
class AxisPlane:
    axis: str
    position_px: float
    keep_side: str
    restitution: float
    basis: str = PLANE_BASIS

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ProgramError(f"axis must be one of {AXES}, got {self.axis!r}")
        if self.keep_side not in KEEP_SIDES:
            raise ProgramError(f"keep_side must be one of {KEEP_SIDES}, got {self.keep_side!r}")
        object.__setattr__(self, "position_px", _finite("position_px", self.position_px))
        e = _finite("restitution", self.restitution)
        if not 0.0 <= e <= 1.0:
            raise ProgramError(f"restitution must be in [0, 1], got {e}")
        object.__setattr__(self, "restitution", e)
        if not isinstance(self.basis, str) or not self.basis:
            raise ProgramError("plane basis must be a non-empty string")

    @property
    def sign(self) -> int:
        """+1 if the body stays at coordinate <= position (normal coordinate u = s (q - P) <= 0)."""
        return 1 if self.keep_side == "less" else -1


@dataclass(frozen=True)
class SceneProgram:
    body: Body
    acceleration: UniformAcceleration
    planes: tuple[AxisPlane, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = PROGRAM_VERSION

    def __post_init__(self) -> None:
        if self.version != PROGRAM_VERSION:
            raise ProgramError(f"version must be {PROGRAM_VERSION!r}, got {self.version!r}")
        if not isinstance(self.body, Body) or not isinstance(self.acceleration, UniformAcceleration):
            raise ProgramError("body and acceleration must be Body and UniformAcceleration")
        object.__setattr__(self, "planes", tuple(self.planes))
        if not all(isinstance(p, AxisPlane) for p in self.planes):
            raise ProgramError("planes must be AxisPlane instances")
        if not isinstance(self.provenance, dict):
            raise ProgramError("provenance must be a JSON object")
        try:
            _canonical(self.provenance)
        except (TypeError, ValueError) as exc:
            raise ProgramError(f"provenance is not finite JSON: {exc}") from None
        q0 = {"x": self.body.x0, "y": self.body.y0}
        for i, p in enumerate(self.planes):
            u = p.sign * (q0[p.axis] - p.position_px)
            if u >= 0:
                raise ProgramError(f"planes[{i}]: the body must start strictly on its keep side "
                                   f"({p.axis} = {q0[p.axis]} vs {p.keep_side} than {p.position_px})")


# ------------------------------------------------------------------ JSON


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def program_to_json(program: SceneProgram) -> dict:
    """Deterministic JSON form; t0 stays an exact rational."""
    b, a = program.body, program.acceleration
    return {
        "program_version": program.version,
        "units": dict(UNITS),
        "coordinate_system": dict(COORDINATE_SYSTEM),
        "body": {"object_id": b.object_id, "t0": timestamp_to_json(b.t0), "x0": b.x0, "y0": b.y0,
                 "vx0": b.vx0, "vy0": b.vy0},
        "acceleration": {"ax": a.ax, "ay": a.ay},
        "planes": [{"axis": p.axis, "position_px": p.position_px, "keep_side": p.keep_side,
                    "restitution": p.restitution, "basis": p.basis} for p in program.planes],
        "provenance": json.loads(_canonical(program.provenance)),
    }


def _keys(obj: Any, expected: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise ProgramError(f"{where} must be a JSON object")
    if set(obj) != expected:
        raise ProgramError(f"{where}: expected keys {sorted(expected)}, got {sorted(obj)}")
    return obj


def program_from_json(data: Any) -> SceneProgram:
    """Strict inverse of program_to_json. Raises ProgramError on any deviation."""
    d = _keys(data, {"program_version", "units", "coordinate_system", "body", "acceleration", "planes", "provenance"},
              "program")
    if d["units"] != UNITS or d["coordinate_system"] != COORDINATE_SYSTEM:
        raise ProgramError(f"units/coordinate_system must be {UNITS} / {COORDINATE_SYSTEM}")
    b = _keys(d["body"], {"object_id", "t0", "x0", "y0", "vx0", "vy0"}, "body")
    try:
        t0 = timestamp_from_json(b["t0"])
    except ValueError as exc:
        raise ProgramError(f"body.t0: {exc}") from None
    a = _keys(d["acceleration"], {"ax", "ay"}, "acceleration")
    if not isinstance(d["planes"], list):
        raise ProgramError("planes must be a list")
    planes = tuple(
        AxisPlane(p["axis"], p["position_px"], p["keep_side"], p["restitution"], p["basis"])
        for p in (_keys(p, {"axis", "position_px", "keep_side", "restitution", "basis"}, f"planes[{i}]")
                  for i, p in enumerate(d["planes"])))
    return SceneProgram(Body(b["object_id"], t0, b["x0"], b["y0"], b["vx0"], b["vy0"]),
                        UniformAcceleration(a["ax"], a["ay"]), planes, d["provenance"], d["program_version"])


def program_sha256(program: SceneProgram) -> str:
    """SHA-256 of the canonical JSON form (sorted keys, no whitespace, finite numbers only)."""
    return hashlib.sha256(_canonical(program_to_json(program)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ simulation


@dataclass(frozen=True)
class ContactEvent:
    index: int
    plane_index: int
    axis: str
    t_s: float  # absolute seconds (float: contact times are generally irrational)
    t_rel_s: float  # seconds after t0
    x_px: float
    y_px: float
    v_normal_before_px_s: float
    v_normal_after_px_s: float
    resting: bool  # the body came to rest on this plane at this contact


@dataclass(frozen=True)
class SimulationResult:
    program_sha256: str
    times: tuple[Fraction, ...]  # exactly as requested
    x_px: tuple[float, ...]
    y_px: tuple[float, ...]
    vx_px_s: tuple[float, ...]
    vy_px_s: tuple[float, ...]
    events: tuple[ContactEvent, ...]
    termination: str  # "end" | "resting"
    rest_time_s: float | None
    note: str = ("Velocities at an exact contact time are the pre-contact velocities. Contact times are float "
                 "seconds; query times are exact.")


def _time_to_contact(u: float, w: float, alpha: float) -> float | None:
    """Smallest tau > 0 with u + w tau + alpha tau^2 / 2 = 0, for u < 0 (strictly inside)."""
    if alpha == 0.0:
        return -u / w if w > 0 else None
    disc = w * w - 2.0 * alpha * u
    if disc < 0:
        return None
    q = -0.5 * (w + math.copysign(math.sqrt(disc), w))
    roots = [r for r in ((q / (0.5 * alpha)) if q != 0 else None, (u / q) if q != 0 else None)
             if r is not None and r > 0 and math.isfinite(r)]
    return min(roots) if roots else None


def simulate(program: SceneProgram, times: Sequence[Fraction], *, max_events: int = MAX_EVENTS) -> SimulationResult:
    """Run `program` and return positions and velocities at the exact `times` (each >= t0)."""
    if not isinstance(program, SceneProgram):
        raise TypeError(f"expected a SceneProgram, got {type(program).__name__}")
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0:
        raise ValueError("max_events must be a non-negative int")
    exact: list[Fraction] = []
    for t in times:
        if isinstance(t, bool) or not isinstance(t, (Fraction, int)):
            raise TypeError(f"times must be exact Fractions (or ints), got {type(t).__name__}")
        t = Fraction(t)
        if t < program.body.t0:
            raise ValueError(f"time {t} is before the program start t0 = {program.body.t0}")
        exact.append(t)
    t0 = program.body.t0
    rel = [float(t - t0) for t in exact]
    t_end = max(rel, default=0.0)

    acc = [program.acceleration.ax, program.acceleration.ay]
    q = [program.body.x0, program.body.y0]
    v = [program.body.vx0, program.body.vy0]
    planes = program.planes
    frozen: set[int] = set()  # axes on which the body rests on a plane
    on_plane: int | None = None  # plane of the last contact (exact zero gap)
    t = 0.0
    segments: list[tuple[float, float, list[float], list[float], list[float]]] = []
    events: list[ContactEvent] = []
    rest_time = None

    while True:
        a_eff = [0.0 if i in frozen else acc[i] for i in range(2)]
        cands: list[tuple[float, int]] = []
        for j, p in enumerate(planes):
            i = AXES.index(p.axis)
            if i in frozen:
                continue
            s = p.sign
            u, w, al = s * (q[i] - p.position_px), s * v[i], s * a_eff[i]
            if j == on_plane:
                tau = (-2.0 * w / al) if (al > 0 and w < 0) else None
            elif u > POSITION_TOL_PX:
                raise SimulationError(f"the body is {u:.3g} px beyond plane {j}")
            elif u >= 0:
                tau = 0.0 if w > 0 else None
            else:
                tau = _time_to_contact(u, w, al)
            if tau is not None:
                cands.append((tau, j))
        cands.sort()
        if not cands or t + cands[0][0] > t_end:
            segments.append((t, math.inf, list(q), list(v), a_eff))
            break
        tau, j = cands[0]
        if len(cands) > 1 and cands[1][0] - tau <= SIMULTANEOUS_TOL_S:
            raise SimultaneousContactError(
                f"contacts with planes {j} and {cands[1][1]} at t0 + {t + tau:.12g} s are simultaneous "
                f"(within {SIMULTANEOUS_TOL_S} s); the response is ambiguous")
        if len(events) >= max_events:
            raise SimulationLimitError(f"more than {max_events} contacts before t0 + {t_end:.12g} s")
        segments.append((t, t + tau, list(q), list(v), a_eff))
        for i in range(2):
            q[i], v[i] = q[i] + v[i] * tau + 0.5 * a_eff[i] * tau * tau, v[i] + a_eff[i] * tau
        t += tau
        p = planes[j]
        i = AXES.index(p.axis)
        q[i] = p.position_px  # exactly on the boundary
        vn_before = v[i]
        v[i] = -p.restitution * vn_before
        al = p.sign * a_eff[i]
        w_after = p.sign * v[i]
        resting = (al > 0 and 2.0 * abs(w_after) / al < MIN_FLIGHT_S) or (al <= 0 and w_after == 0.0)
        if resting:
            v[i] = 0.0
            frozen.add(i)
            rest_time = float(t0) + t
        events.append(ContactEvent(len(events), j, p.axis, float(t0) + t, t, q[0], q[1], vn_before, v[i], resting))
        on_plane = j

    xs, ys, vxs, vys = [], [], [], []
    for tr in rel:
        for (ts, te, q0, v0, a0) in segments:
            if tr <= te:
                d = tr - ts
                xs.append(q0[0] + v0[0] * d + 0.5 * a0[0] * d * d)
                ys.append(q0[1] + v0[1] * d + 0.5 * a0[1] * d * d)
                vxs.append(v0[0] + a0[0] * d)
                vys.append(v0[1] + a0[1] * d)
                break
    return SimulationResult(program_sha256(program), tuple(exact), tuple(xs), tuple(ys), tuple(vxs), tuple(vys),
                            tuple(events), "resting" if rest_time is not None else "end", rest_time)


# ------------------------------------------------------------------ from a kinematic fit


def _fraction_json(t: Fraction) -> dict:
    return {"num": t.numerator, "den": t.denominator}


def _num(x: float) -> float | None:
    """Finite float, or None for a non-finite diagnostic (e.g. z = inf when a standard error is 0)."""
    return float(x) if math.isfinite(x) else None


def program_from_kinematics(obj: kf.ObjectKinematics) -> SceneProgram:
    """Build a one-plane program from a B1c-1 fit, or raise ProgramUnavailable with every reason."""
    if not isinstance(obj, kf.ObjectKinematics):
        raise TypeError(f"expected an ObjectKinematics, got {type(obj).__name__}")
    reasons: list[str] = []
    v = obj.verdicts
    for name in ("impulse", "uniform_acceleration"):
        if v[name].verdict != kf.SUPPORTED:
            reasons.append(f"{name} is {v[name].verdict!r}, not supported")
    if v["linear_drag"].verdict == kf.SUPPORTED:
        reasons.append("linear drag is supported; drag is not part of scene program v1")
    if obj.adequacy == "inadequate":
        reasons.append("the kinematic fit is inadequate: " + "; ".join(obj.adequacy_reasons))
    elif obj.adequacy not in ("adequate", "not_assessable"):
        reasons.append(f"unknown fit adequacy {obj.adequacy!r}")
    if obj.additional_discontinuity is not None:
        reasons.append("a further velocity discontinuity was detected (multiple contacts)")
    c = obj.contact
    fit = obj.fits.get(kf.IMPULSE)
    if c is None or fit is None:
        reasons.append("no one-contact uniform-acceleration fit")
        raise ProgramUnavailable(reasons)
    if c.model != kf.IMPULSE:
        reasons.append(f"the contact estimate comes from {c.model!r}, not {kf.IMPULSE!r}")
    if c.profile_multimodal:
        reasons.append("the contact-time profile is multimodal (ambiguous contact)")
    if c.interval_at_search_boundary:
        reasons.append("the contact-time interval reaches the edge of the searchable range")
    reversing = [a for a in c.axes if a.reversal is not None]
    if len(reversing) != 1:
        reasons.append(f"a contact boundary needs a velocity reversal on exactly one axis; found {len(reversing)}")
        raise ProgramUnavailable(reasons)
    normal = reversing[0]
    tangential = next(a for a in c.axes if a.axis != normal.axis)
    if tangential.significant:
        reasons.append(f"tangential velocity change on {tangential.axis} is {tangential.delta_v_px_s:.6g} px/s "
                       f"(z = {tangential.delta_v_z:.3g} > {kf.Z_SIGNIFICANT}): not consistent with zero")
    if not normal.significant:
        reasons.append(f"the normal velocity change on {normal.axis} is not significant")
    vb, va = normal.v_before_px_s, normal.v_after_px_s
    e = -va / vb if vb != 0 else math.nan
    if not (math.isfinite(e) and 0.0 <= e <= 1.0):
        reasons.append(f"restitution -v_after/v_before = {e!r} is outside [0, 1]")

    i = AXES.index(normal.axis)
    p = fit.params
    tc = p["t_contact_s"]  # seconds after the object's first observation
    pos = p[f"{normal.axis}.q0"] + p[f"{normal.axis}.v0"] * tc + 0.5 * p[f"{normal.axis}.a"] * tc * tc
    keep = "less" if vb > 0 else "greater"
    start = p[f"{normal.axis}.q0"]
    if not math.isfinite(pos) or not ((start < pos) if keep == "less" else (start > pos)):
        reasons.append(f"contact geometry invalid: boundary at {pos!r} px is not ahead of the start {start} px")
    t_last = obj.t_start_s + sum((fd.dt_s for fd in obj.finite_differences), Fraction(0))
    if not 0 < tc < float(t_last - obj.t_start_s):
        reasons.append("the contact time lies outside the fitted span")
    if reasons:
        raise ProgramUnavailable(reasons)

    provenance = {
        "source": kf.METHOD_VERSION,
        "object_id": obj.object_id,
        "fit_model": kf.IMPULSE,
        "fit_adequacy": obj.adequacy,
        "fit_adequacy_note": ("absolute fit adequacy was not independently assessed (sigma estimated from the "
                              "same residuals)" if obj.adequacy == "not_assessable"
                              else "chi-square adequacy test passed with the supplied sigma"),
        "verdicts": {name: obj.verdicts[name].verdict for name in sorted(obj.verdicts)},
        "fitted_span": {"t_first": _fraction_json(obj.t_start_s), "t_last": _fraction_json(t_last),
                        "n_observations": obj.n_observations},
        "contact": {"t_rel_s": tc, "gap_frames": list(c.gap_frames),
                    "gap_t": [_fraction_json(c.gap_t[0]), _fraction_json(c.gap_t[1])],
                    "profile_interval_s": [_num(x) for x in c.profile_interval_s]},
        "plane_basis": PLANE_BASIS,
        "restitution_basis": "-v_after / v_before of the fitted normal velocities at the contact",
        "normal_axis": {"axis": normal.axis, "v_before_px_s": vb, "v_before_se": _num(normal.v_before_se),
                        "v_after_px_s": va, "delta_v_se": _num(normal.delta_v_se), "reversal": normal.reversal},
        "tangential_axis": {"axis": tangential.axis, "delta_v_px_s": tangential.delta_v_px_s,
                            "delta_v_z": _num(tangential.delta_v_z),
                            "treatment": "carried through the contact unchanged (change consistent with zero)"},
        "drag": obj.verdicts["linear_drag"].verdict,
        "note": "Reproducing the fitted observations is a consistency check, not held-out validation.",
    }
    body = Body(obj.object_id, obj.t_start_s, p["x.q0"], p["y.q0"], p["x.v0"], p["y.v0"])
    accel = UniformAcceleration(p["x.a"], p["y.a"])
    return SceneProgram(body, accel, (AxisPlane(normal.axis, pos, keep, e),), provenance)
