"""B1b-1: trajectory input format `decompile.trajectories/1`.

A trajectory set holds per-object observations (frame index, exact timestamp, pixel
centroid). It is the input to scene interpretation and, later, the output of a tracker.

Timestamps are exact rationals, matching A2's `FrameTimestamp.t` (a Fraction). On disk
a timestamp is {"num": <int>, "den": <int>}, in lowest terms, with den > 0 and num >= 0.
They are never converted to floating point: ordering and event windows use
fractions.Fraction.

Coordinates are pixels with the origin at the top-left; x grows rightwards and
**y grows downwards** (so a falling object has increasing y).

Loading is strict and fail-loud:
- strict JSON: no NaN/Infinity, no duplicate keys (meta_api.strict_json_loads)
- unknown keys are rejected at every level, so ground truth, expected answers or free
  text can never travel inside a trajectory file into a prompt
- all problems are collected and raised together as one TrajectoryError
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from decompile.meta_api import strict_json_loads

SCHEMA = "decompile.trajectories/1"

# source -> the `synthetic` value that source requires, or None if it accepts either.
# The `synthetic` flag records the provenance of the underlying data (for a tracker: of the
# video it tracked), not the algorithm that produced the observations.
#   SYNTHETIC_ANALYTIC_FIXTURE     closed-form test fixtures (e.g. synthetic_bounce_single): always synthetic
#   SYNTHETIC_NUMERICAL_GENERATOR  numerically integrated scenes from decompile.synth: always synthetic
#   CLASSICAL_TRACKER              decompile.tracker on a video; synthetic iff the video is synthetic
#   SAM_3_1                        SAM 3.1 tracking of a video; synthetic iff the video is synthetic
SOURCES: dict[str, bool | None] = {
    "SYNTHETIC_ANALYTIC_FIXTURE": True,
    "SYNTHETIC_NUMERICAL_GENERATOR": True,
    "CLASSICAL_TRACKER": None,
    "SAM_3_1": None,
}

COORDINATE_CONSTANTS = {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down"}

OBJECT_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
MIN_OBSERVATIONS = 3
MAX_OBSERVATIONS_PER_OBJECT = 300
MAX_TOTAL_OBSERVATIONS = 600
MAX_OBJECTS = 8
MAX_INT64 = 2**63 - 1  # A2 pts/time-base values are int64

_TOP_KEYS = {"schema", "synthetic", "source", "coordinate_system", "objects"}
_COORD_KEYS = set(COORDINATE_CONSTANTS) | {"image_width", "image_height"}
_OBJECT_KEYS = {"object_id", "observations"}
_OBS_KEYS = {"frame_index", "t", "x", "y"}
_TIME_KEYS = {"num", "den"}


class TrajectoryError(ValueError):
    """A trajectory set is invalid. `reasons` lists every problem found."""

    def __init__(self, reasons: list[str], source: str | None = None) -> None:
        self.reasons = list(reasons)
        self.source = source
        where = f"{source}: " if source else ""
        super().__init__(where + "invalid trajectory set: " + "; ".join(self.reasons))


# ------------------------------------------------------------------ timestamps


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def timestamp_to_json(t: Fraction) -> dict[str, int]:
    """Exact JSON form of a timestamp. Fraction is always in lowest terms with den > 0."""
    if not isinstance(t, Fraction):
        raise TypeError(f"timestamp must be a Fraction, got {type(t).__name__}")
    return {"num": t.numerator, "den": t.denominator}


def _parse_timestamp(value: Any, path: str, reasons: list[str]) -> Fraction | None:
    if not isinstance(value, dict):
        reasons.append(f"{path}: timestamp must be an object {{num, den}}")
        return None
    _check_keys(value, _TIME_KEYS, path, reasons)
    num, den = value.get("num"), value.get("den")
    ok = True
    if not _is_int(num):
        reasons.append(f"{path}.num: must be an integer (bool not accepted), got {num!r}")
        ok = False
    if not _is_int(den):
        reasons.append(f"{path}.den: must be an integer (bool not accepted), got {den!r}")
        ok = False
    if not ok:
        return None
    if den <= 0:
        reasons.append(f"{path}.den: must be positive, got {den}")
        return None
    if num < 0:
        reasons.append(f"{path}.num: must be >= 0, got {num}")
        return None
    if num > MAX_INT64 or den > MAX_INT64:
        reasons.append(f"{path}: num/den must not exceed 2**63-1")
        return None
    if gcd(num, den) != 1:
        reasons.append(f"{path}: {num}/{den} is not in lowest terms (expected {Fraction(num, den)})")
        return None
    return Fraction(num, den)


def timestamp_from_json(value: Any) -> Fraction:
    """Parse one {"num", "den"} timestamp; raise TrajectoryError if invalid."""
    reasons: list[str] = []
    t = _parse_timestamp(value, "t", reasons)
    if reasons:
        raise TrajectoryError(reasons)
    assert t is not None
    return t


# ------------------------------------------------------------------ data structures


@dataclass(frozen=True)
class Observation:
    frame_index: int
    t: Fraction
    x: float
    y: float

    def to_json_dict(self) -> dict:
        return {"frame_index": self.frame_index, "t": timestamp_to_json(self.t), "x": self.x, "y": self.y}


@dataclass(frozen=True)
class ObjectTrajectory:
    object_id: str
    observations: tuple[Observation, ...]

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(o.frame_index for o in self.observations)

    def observation_at(self, frame_index: int) -> Observation | None:
        for o in self.observations:
            if o.frame_index == frame_index:
                return o
        return None


@dataclass(frozen=True)
class CoordinateSystem:
    image_width: int
    image_height: int

    def to_json_dict(self) -> dict:
        return {**COORDINATE_CONSTANTS, "image_width": self.image_width, "image_height": self.image_height}


@dataclass(frozen=True)
class TrajectorySet:
    synthetic: bool
    source: str
    coordinate_system: CoordinateSystem
    objects: tuple[ObjectTrajectory, ...]
    warnings: tuple[str, ...] = field(default=(), compare=False)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(o.object_id for o in self.objects)

    def get(self, object_id: str) -> ObjectTrajectory:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        raise KeyError(object_id)

    def frame_time(self, frame_index: int) -> Fraction:
        """Exact timestamp of an observed frame (the clock is shared by all objects)."""
        for o in self.objects:
            obs = o.observation_at(frame_index)
            if obs is not None:
                return obs.t
        raise KeyError(frame_index)

    def to_json_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "synthetic": self.synthetic,
            "source": self.source,
            "coordinate_system": self.coordinate_system.to_json_dict(),
            "objects": [
                {"object_id": o.object_id, "observations": [ob.to_json_dict() for ob in o.observations]}
                for o in self.objects
            ],
        }

    @classmethod
    def from_json_dict(cls, data: Any, *, source_name: str | None = None) -> "TrajectorySet":
        return _parse_set(data, source_name)


# ------------------------------------------------------------------ parsing


def _check_keys(obj: dict, allowed: set[str], path: str, reasons: list[str]) -> None:
    extra = sorted(set(obj) - allowed)
    missing = sorted(allowed - set(obj))
    if extra:
        reasons.append(f"{path}: unknown keys {extra}")
    if missing:
        reasons.append(f"{path}: missing keys {missing}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_set(data: Any, source_name: str | None) -> TrajectorySet:
    reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        raise TrajectoryError(["top level must be a JSON object"], source_name)
    _check_keys(data, _TOP_KEYS, "$", reasons)

    if data.get("schema") != SCHEMA:
        reasons.append(f"$.schema: must be {SCHEMA!r}, got {data.get('schema')!r}")

    synthetic = data.get("synthetic")
    source = data.get("source")
    if not isinstance(synthetic, bool):
        reasons.append(f"$.synthetic: must be a boolean, got {synthetic!r}")
    if source not in SOURCES:
        reasons.append(f"$.source: must be one of {sorted(SOURCES)}, got {source!r}")
    elif isinstance(synthetic, bool) and SOURCES[source] is not None and SOURCES[source] != synthetic:
        reasons.append(f"$.synthetic: {synthetic} contradicts source {source!r}")

    coords = _parse_coords(data.get("coordinate_system"), reasons)

    objects_raw = data.get("objects")
    objects: list[ObjectTrajectory] = []
    if not isinstance(objects_raw, list):
        reasons.append("$.objects: must be an array")
        objects_raw = []
    elif not 1 <= len(objects_raw) <= MAX_OBJECTS:
        reasons.append(f"$.objects: must contain 1..{MAX_OBJECTS} objects, got {len(objects_raw)}")

    seen_ids: set[str] = set()
    total = 0
    for i, obj in enumerate(objects_raw):
        parsed = _parse_object(obj, f"$.objects[{i}]", reasons)
        if parsed is None:
            continue
        if parsed.object_id in seen_ids:
            reasons.append(f"$.objects[{i}].object_id: duplicate {parsed.object_id!r}")
        seen_ids.add(parsed.object_id)
        total += len(parsed.observations)
        objects.append(parsed)
    if total > MAX_TOTAL_OBSERVATIONS:
        reasons.append(f"$.objects: {total} observations in total (max {MAX_TOTAL_OBSERVATIONS})")

    _check_shared_clock(objects, reasons)

    if coords is not None:
        for o in objects:
            for ob in o.observations:
                if not (0 <= ob.x <= coords.image_width and 0 <= ob.y <= coords.image_height):
                    warnings.append(
                        f"{o.object_id} frame {ob.frame_index}: ({ob.x}, {ob.y}) lies outside the image"
                    )

    if reasons:
        raise TrajectoryError(reasons, source_name)
    assert coords is not None and isinstance(synthetic, bool) and isinstance(source, str)
    return TrajectorySet(synthetic, source, coords, tuple(objects), tuple(warnings))


def _parse_coords(value: Any, reasons: list[str]) -> CoordinateSystem | None:
    path = "$.coordinate_system"
    if not isinstance(value, dict):
        reasons.append(f"{path}: must be an object")
        return None
    _check_keys(value, _COORD_KEYS, path, reasons)
    ok = True
    for key, expected in COORDINATE_CONSTANTS.items():
        if key in value and value[key] != expected:
            reasons.append(f"{path}.{key}: must be {expected!r}, got {value[key]!r}")
            ok = False
    dims = {}
    for key in ("image_width", "image_height"):
        v = value.get(key)
        if not _is_int(v) or v <= 0:
            reasons.append(f"{path}.{key}: must be a positive integer, got {v!r}")
            ok = False
        dims[key] = v
    if not ok or set(value) != _COORD_KEYS:
        return None
    return CoordinateSystem(dims["image_width"], dims["image_height"])


def _parse_object(value: Any, path: str, reasons: list[str]) -> ObjectTrajectory | None:
    if not isinstance(value, dict):
        reasons.append(f"{path}: must be an object")
        return None
    _check_keys(value, _OBJECT_KEYS, path, reasons)
    object_id = value.get("object_id")
    ok = True
    if not isinstance(object_id, str) or not OBJECT_ID_RE.fullmatch(object_id):
        reasons.append(f"{path}.object_id: must match {OBJECT_ID_RE.pattern}, got {object_id!r}")
        ok = False
    obs_raw = value.get("observations")
    if not isinstance(obs_raw, list):
        reasons.append(f"{path}.observations: must be an array")
        return None
    if not MIN_OBSERVATIONS <= len(obs_raw) <= MAX_OBSERVATIONS_PER_OBJECT:
        reasons.append(
            f"{path}.observations: must contain {MIN_OBSERVATIONS}..{MAX_OBSERVATIONS_PER_OBJECT} "
            f"observations, got {len(obs_raw)}"
        )
        ok = False
    observations: list[Observation] = []
    for j, ob in enumerate(obs_raw):
        parsed = _parse_observation(ob, f"{path}.observations[{j}]", reasons)
        if parsed is None:
            ok = False
            continue
        if observations:
            prev = observations[-1]
            if parsed.frame_index <= prev.frame_index:
                reasons.append(f"{path}.observations[{j}].frame_index: not strictly increasing")
                ok = False
            if parsed.t <= prev.t:
                reasons.append(f"{path}.observations[{j}].t: not strictly increasing")
                ok = False
        observations.append(parsed)
    if not ok or set(value) != _OBJECT_KEYS:
        return None
    return ObjectTrajectory(object_id, tuple(observations))


def _parse_observation(value: Any, path: str, reasons: list[str]) -> Observation | None:
    if not isinstance(value, dict):
        reasons.append(f"{path}: must be an object")
        return None
    n_before = len(reasons)
    _check_keys(value, _OBS_KEYS, path, reasons)
    fi = value.get("frame_index")
    if not _is_int(fi) or fi < 0 or fi > MAX_INT64:
        reasons.append(f"{path}.frame_index: must be a non-negative integer, got {fi!r}")
    t = _parse_timestamp(value.get("t"), f"{path}.t", reasons) if "t" in value else None
    for key in ("x", "y"):
        if key in value and not _finite_number(value[key]):
            reasons.append(f"{path}.{key}: must be a finite number, got {value[key]!r}")
    if len(reasons) != n_before or t is None:
        return None
    return Observation(fi, t, float(value["x"]), float(value["y"]))


def _check_shared_clock(objects: list[ObjectTrajectory], reasons: list[str]) -> None:
    """One clock for all objects: a frame index has one time, and time follows frame order."""
    times: dict[int, Fraction] = {}
    for o in objects:
        for ob in o.observations:
            known = times.get(ob.frame_index)
            if known is None:
                times[ob.frame_index] = ob.t
            elif known != ob.t:
                reasons.append(
                    f"{o.object_id} frame {ob.frame_index}: t={ob.t} disagrees with t={known} from another object"
                )
    ordered = sorted(times.items())
    for (f0, t0), (f1, t1) in zip(ordered, ordered[1:]):
        if t1 <= t0:
            reasons.append(f"frames {f0} and {f1}: time does not increase with frame index across objects")


# ------------------------------------------------------------------ loading


def loads_trajectories(text: str, *, source_name: str | None = None) -> TrajectorySet:
    try:
        data = strict_json_loads(text)
    except ValueError as exc:
        raise TrajectoryError([f"not strict JSON: {exc}"], source_name) from None
    return TrajectorySet.from_json_dict(data, source_name=source_name)


def load_trajectories(path: str | Path) -> TrajectorySet:
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrajectoryError([f"not UTF-8: {exc}"], str(path)) from None
    return loads_trajectories(text, source_name=str(path))
