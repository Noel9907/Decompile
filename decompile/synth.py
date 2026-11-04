"""P2 (A4-lite): independent numerical ground-truth generator for synthetic trajectories.

INDEPENDENCE: this generator shares no code or method with the fitter
(decompile.kinematic_fit) or the simulator (decompile.scene_program). It integrates the
equations of motion numerically:

    dq/dt = v,   dv/dt = a - k v        (k = 0: no drag)

with the classical 4th-order Runge-Kutta method on a fine fixed step (frame interval /
substeps, so every frame time is an exact step boundary). Plane crossings are detected
after each step and located by bracketing and bisection on the step length (each trial
re-integrates the step with RK4); the normal velocity is then reflected with the
restitution and the rest of the step is integrated. Contact times are therefore numerical
results, never inserted from a formula.

Outputs are kept apart:
- SyntheticScene.trajectories: a TrajectorySet with positions rounded to 0.01 px and
  exact Fraction timestamps (frame / fps). It carries no ground truth.
- SyntheticScene.ground_truth: parameters, contacts, unrounded positions and method
  details, for evaluation and tests only. Never pass it to inference code.

Labelling: the generated set carries source "SYNTHETIC_NUMERICAL_GENERATOR" and
synthetic = true, distinct from the closed-form "SYNTHETIC_ANALYTIC_FIXTURE" fixtures.
GroundTruth.generator_method records the numerical method.

PREREGISTERED_DEMO is the fixed, pre-registered scene of the first predictive
demonstration (P3). It must not be tuned on prediction results.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction

from decompile import trajectories as tr

GENERATOR_VERSION = "decompile.synth/1"
GENERATOR_METHOD = "RK4 fixed step; plane crossings by bisection on the step length; restitution reflection"
SOURCE_LABEL = "SYNTHETIC_NUMERICAL_GENERATOR"  # numerically generated, not an analytic fixture
DEFAULT_SUBSTEPS = 2000
BISECTION_ITERATIONS = 80
# After a contact the body rests if its normal speed is below this (px/s): a bounce that low
# rises less than v^2 / (2a), i.e. < 1e-9 px at 500 px/s^2: physically invisible.
REST_SPEED_PX_S = 1e-3
MAX_CONTACTS = 10_000


class SynthError(ValueError):
    """Invalid generator specification or a generation failure."""


@dataclass(frozen=True)
class Plane:
    axis: str  # "x" | "y"
    position_px: float
    keep_side: str  # "less": the body stays at coordinate <= position_px; "greater": >=
    restitution: float

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y") or self.keep_side not in ("less", "greater"):
            raise SynthError(f"invalid plane axis/side {self.axis!r}/{self.keep_side!r}")
        if not (math.isfinite(self.position_px) and 0.0 <= self.restitution <= 1.0):
            raise SynthError("plane position must be finite and restitution in [0, 1]")


@dataclass(frozen=True)
class SceneSpec:
    object_id: str
    fps: int
    n_frames: int
    x0: float
    y0: float
    vx0: float
    vy0: float
    ax: float
    ay: float
    planes: tuple[Plane, ...] = ()
    drag_k_per_s: float = 0.0
    image_width: int = 640
    image_height: int = 480
    round_decimals: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "planes", tuple(self.planes))
        if not (isinstance(self.fps, int) and self.fps > 0 and isinstance(self.n_frames, int)
                and self.n_frames >= tr.MIN_OBSERVATIONS):
            raise SynthError("fps must be a positive int and n_frames an int >= 3")
        for name in ("x0", "y0", "vx0", "vy0", "ax", "ay", "drag_k_per_s"):
            if not math.isfinite(getattr(self, name)):
                raise SynthError(f"{name} must be finite")
        if self.drag_k_per_s < 0:
            raise SynthError("drag_k_per_s must be >= 0")


@dataclass(frozen=True)
class GroundTruthContact:
    index: int
    plane_index: int
    axis: str
    t_s: float
    between_frames: tuple[int, int]  # frames bracketing the contact (equal if it lands on a frame)
    x_px: float
    y_px: float
    v_normal_before_px_s: float
    v_normal_after_px_s: float
    resting: bool


@dataclass(frozen=True)
class GroundTruth:
    generator_version: str
    generator_method: str
    spec: SceneSpec
    substeps: int
    step_s: Fraction
    contacts: tuple[GroundTruthContact, ...]
    rest_time_s: float | None
    true_x_px: tuple[float, ...]  # unrounded, per frame
    true_y_px: tuple[float, ...]


@dataclass(frozen=True)
class SyntheticScene:
    trajectories: tr.TrajectorySet  # model input: no ground truth inside
    ground_truth: GroundTruth  # evaluation and tests only


# Pre-registered by the project owner before any withheld evaluation; never tuned on results.
# Disclosure (P2 development): while checking that the 70% training split supports a scene
# program, the program's simulated contact times through frame 49 (0.9, 1.44003, 1.60205 s)
# were printed. No withheld positions were used and nothing was changed afterwards, but the
# P3 evaluation of this scene is therefore not fully blind.
PREREGISTERED_DEMO = SceneSpec(
    object_id="obj_01", fps=30, n_frames=50, x0=100.0, y0=45.0, vx0=60.0, vy0=0.0, ax=0.0, ay=1000.0,
    planes=(Plane("y", 450.0, "less", 0.3),),
)
PREREGISTERED_TRAIN_FRACTION = Fraction(7, 10)


# ------------------------------------------------------------------ integration


def _deriv(state: list[float], acc: tuple[float, float], k: float, frozen: set[int]) -> list[float]:
    x, y, vx, vy = state
    ax = 0.0 if 0 in frozen else acc[0] - k * vx
    ay = 0.0 if 1 in frozen else acc[1] - k * vy
    return [vx, vy, ax, ay]


def _rk4(state: list[float], h: float, acc, k: float, frozen: set[int]) -> list[float]:
    k1 = _deriv(state, acc, k, frozen)
    k2 = _deriv([s + 0.5 * h * d for s, d in zip(state, k1)], acc, k, frozen)
    k3 = _deriv([s + 0.5 * h * d for s, d in zip(state, k2)], acc, k, frozen)
    k4 = _deriv([s + h * d for s, d in zip(state, k3)], acc, k, frozen)
    out = [s + h / 6.0 * (a + 2 * b + 2 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
    for i in frozen:  # resting on a plane: the normal coordinate and velocity stay fixed
        out[i], out[i + 2] = state[i], 0.0
    return out


def _penetration(state: list[float], p: Plane) -> float:
    """> 0 when the body is beyond the plane."""
    q = state[0] if p.axis == "x" else state[1]
    return (q - p.position_px) if p.keep_side == "less" else (p.position_px - q)


def generate(spec: SceneSpec, *, substeps: int = DEFAULT_SUBSTEPS) -> SyntheticScene:
    """Integrate `spec` numerically and return the rounded TrajectorySet plus separate ground truth."""
    if not isinstance(spec, SceneSpec):
        raise TypeError(f"expected a SceneSpec, got {type(spec).__name__}")
    if not isinstance(substeps, int) or substeps < 1:
        raise SynthError("substeps must be a positive int")
    for j, p in enumerate(spec.planes):
        if _penetration([spec.x0, spec.y0, 0, 0], p) >= 0:
            raise SynthError(f"planes[{j}]: the body must start strictly on its keep side")
    step = Fraction(1, spec.fps * substeps)
    h = float(step)
    acc = (spec.ax, spec.ay)
    state = [spec.x0, spec.y0, spec.vx0, spec.vy0]
    frozen: set[int] = set()
    contacts: list[GroundTruthContact] = []
    rest_time = None
    xs, ys = [state[0]], [state[1]]
    n_steps = (spec.n_frames - 1) * substeps
    for n in range(n_steps):
        t_start = float(step * n)
        remaining = h
        while remaining > 0.0:
            trial = _rk4(state, remaining, acc, spec.drag_k_per_s, frozen)
            crossed = [(j, p) for j, p in enumerate(spec.planes)
                       if (0 if p.axis == "x" else 1) not in frozen and _penetration(trial, p) > 0]
            if not crossed:
                state, remaining = trial, 0.0
                continue
            if len(crossed) > 1:
                raise SynthError(f"two planes crossed within one step near t = {t_start:.9g} s (ambiguous)")
            j, p = crossed[0]
            lo, hi = 0.0, remaining  # penetration(lo) <= 0 < penetration(hi)
            for _ in range(BISECTION_ITERATIONS):
                mid = 0.5 * (lo + hi)
                if _penetration(_rk4(state, mid, acc, spec.drag_k_per_s, frozen), p) > 0:
                    hi = mid
                else:
                    lo = mid
            state = _rk4(state, lo, acc, spec.drag_k_per_s, frozen)
            i = 0 if p.axis == "x" else 1
            state[i] = p.position_px
            v_before = state[i + 2]
            state[i + 2] = -p.restitution * v_before
            resting = abs(state[i + 2]) < REST_SPEED_PX_S
            if resting:
                state[i + 2] = 0.0
                frozen.add(i)
            t_contact = t_start + (h - remaining) + lo
            f_lo = math.floor(t_contact * spec.fps + 1e-9)
            f_hi = math.ceil(t_contact * spec.fps - 1e-9)
            contacts.append(GroundTruthContact(len(contacts), j, p.axis, t_contact, (f_lo, f_hi), state[0], state[1],
                                               v_before, state[i + 2], resting))
            if resting and rest_time is None:
                rest_time = t_contact
            if len(contacts) > MAX_CONTACTS:
                raise SynthError("too many contacts (Zeno sequence without rest)")
            remaining -= lo
        if (n + 1) % substeps == 0:
            xs.append(state[0])
            ys.append(state[1])

    times = [Fraction(f, spec.fps) for f in range(spec.n_frames)]
    obs = [{"frame_index": f, "t": tr.timestamp_to_json(t), "x": round(x, spec.round_decimals),
            "y": round(y, spec.round_decimals)} for f, (t, x, y) in enumerate(zip(times, xs, ys))]
    traj = tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": SOURCE_LABEL,
        "coordinate_system": {**tr.COORDINATE_CONSTANTS, "image_width": spec.image_width,
                              "image_height": spec.image_height},
        "objects": [{"object_id": spec.object_id, "observations": obs}],
    })
    gt = GroundTruth(GENERATOR_VERSION, GENERATOR_METHOD, spec, substeps, step, tuple(contacts), rest_time,
                     tuple(xs), tuple(ys))
    return SyntheticScene(traj, gt)


def ground_truth_to_json(gt: GroundTruth) -> dict:
    """JSON form of the ground truth, for an evaluation-only file. Never part of model input."""
    spec = asdict(gt.spec)
    data = {
        "kind": "synthetic_ground_truth", "generator_version": gt.generator_version,
        "generator_method": gt.generator_method, "spec": spec, "substeps": gt.substeps,
        "step_s": tr.timestamp_to_json(gt.step_s),
        "contacts": [asdict(c) for c in gt.contacts], "rest_time_s": gt.rest_time_s,
        "true_x_px": list(gt.true_x_px), "true_y_px": list(gt.true_y_px),
    }
    return json.loads(json.dumps(data, allow_nan=False))
