"""P5 (A6-lite): deterministic classical tracker for ONE clearly coloured ball.

Inputs: RGB uint8 frames and the exact A2 FrameTable timestamps (decompile.frames). The
tracker knows nothing else: no renderer settings, no ground truth, no vault, no physics
model. It imports only decompile.frames and decompile.trajectories.

Per frame, INDEPENDENTLY (detect() is a pure function of one image and the config; there
is no state, smoothing, interpolation, extrapolation or correction from other frames):
  1. chroma C = max(R,G,B) - min(R,G,B) and hue H (degrees) per pixel;
  2. mask = C >= min_chroma and circular |H - hue_center_deg| <= hue_tolerance_deg
     (the "prompt": which colour the target is; fixed in the config before tracking);
  3. 8-connected components of the mask (OpenCV); components with area >= min_area_px
     are candidates;
       0 candidates                       -> status "missing"
       >= 2 candidates                    -> status "ambiguous"
       1 candidate touching the border    -> status "touches_border" (centroid would be biased)
       1 candidate with area > max_area_px-> status "area_out_of_range"
       otherwise                          -> status "detected"
  4. sub-pixel centroid of a detection: the candidate is dilated by edge_margin_px (to
     include antialiased / blurred edge pixels) and each pixel in that region is weighted
     by max(0, C - weight_floor_chroma); centroid = weighted mean of pixel centres.
     Pixel convention (shared with decompile.render): pixel (row i, col j) has its centre
     at (j + 0.5, i + 0.5); origin top-left, +y DOWN.

A frame that is not "detected" never receives a coordinate. TrackingResult.to_trajectory_set
refuses (TrackingError, listing every failed frame) unless every frame was detected, so no
coordinate is ever invented.

P6 options (all OFF by default; with the defaults every detection, trajectory and report is
byte-identical to P5):
  - roi / exclude_boxes: pixel boxes (x0, y0, x1, y1), half-open, fixed BEFORE tracking. Mask
    pixels outside the ROI or inside an exclusion box are ignored; a detection whose dilated
    centroid region reaches a disallowed pixel is "touches_border" (its centroid would be biased).
  - hue_weighting: centroid weights also require the pixel's hue to lie within the hue window
    widened by HUE_WEIGHT_MARGIN_DEG (keeps coloured background out of the edge weights).
  - extended_diagnostics: per frame also n_mask_px (target-coloured pixels inside the allowed
    region) and largest_small_component_px (largest component below min_area_px), so "nothing
    target-coloured" can be told apart from "something target-coloured but rejected".
  - config_from_seed(rgb, box): derives the colour model from ONE frame (the first frame of the
    clip, a training frame) and a user-drawn box around the ball. Coloured balls only: a target
    without enough chroma, or not separable from its surroundings by colour, is refused.
  - track_extracted(..., frames=[...]): tracks a subset of exported frames (clip window / stride)
    with their original frame indices and exact timestamps.

P7 option (OFF by default): centre_method.
  - "colour_centroid" (default): the P5 centroid above; output byte-identical to P5/P6.
  - "edge_stadium" (recommended) / "edge_ellipse" / "edge_circle": the colour detection is only the COARSE stage (component,
    gates, statuses). The reported centre comes from decompile.ball_measure (method
    decompile.ball_measure/1): the ball OUTLINE located per ray at the 50% transition between the
    locally estimated ball and background colours, then a robust ellipse / circle fit. This removes
    the background-dependent bias of the colour centroid (a ball crossing a wall/floor boundary),
    and is insensitive to interior seam highlights and shading. Each detection then also carries a
    per-frame 1-sigma centre uncertainty (sigma_x, sigma_y; includes the method's systematic floor),
    the apparent diameter (outline minor axis) and its uncertainty, the colour centroid for
    comparison, and edge-fit diagnostics. A coarse detection whose outline cannot be measured
    reliably (too few edge points, an occluded/truncated outline, implausible axes) gets status
    "edge_fit_rejected" and NO coordinate.

Provenance: the TrajectorySet has source CLASSICAL_TRACKER; its `synthetic` flag is the
provenance of the VIDEO and must be supplied by the caller.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from decompile import ball_measure as bm
from decompile import frames as fr
from decompile import trajectories as tr

TRACKER_VERSION = "decompile.tracker/1"
SOURCE_LABEL = "CLASSICAL_TRACKER"
DETECTED, MISSING, AMBIGUOUS, TOUCHES_BORDER, AREA_OUT_OF_RANGE = (
    "detected", "missing", "ambiguous", "touches_border", "area_out_of_range")
STATUSES = (DETECTED, MISSING, AMBIGUOUS, TOUCHES_BORDER, AREA_OUT_OF_RANGE)
EDGE_REJECTED = "edge_fit_rejected"  # P7: only with an edge centre method
CENTRE_METHODS = ("colour_centroid", "edge_stadium", "edge_ellipse", "edge_circle")
PIXEL_CONVENTION = "pixel (row i, col j) has its centre at (j + 0.5, i + 0.5)"
HUE_WEIGHT_MARGIN_DEG = 15.0
# Seed derivation constants (fixed; documented in the derivation record).
SEED_CORE_SCALE = 0.6  # core sample: ellipse inscribed in the seed box, scaled by this
SEED_MIN_SIDE_PX = 6
SEED_MIN_CORE_CHROMA = 30.0  # median chroma of the core below this: not a coloured target
SEED_MIN_SEPARATION = 10.0  # chroma gap needed between the target and same-hue background
SEED_HUE_TOL_RANGE = (10.0, 45.0)
SEED_EDGE_SKIP_PX = 2  # pixels just outside the box are skipped (blurred edge), not background


class SeedError(ValueError):
    """A tracker configuration could not be derived from the seed box."""


class TrackingError(ValueError):
    """Tracking could not produce a trustworthy trajectory. `problems` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems[:10]) + (" ..." if len(self.problems) > 10 else ""))


@dataclass(frozen=True)
class TrackerConfig:
    object_id: str = "obj_01"
    hue_center_deg: float = 15.0  # orange
    hue_tolerance_deg: float = 25.0
    min_chroma: int = 60
    weight_floor_chroma: float = 20.0
    min_area_px: int = 30
    max_area_px: int = 20_000
    edge_margin_px: int = 2
    # P6 options (defaults keep P5 behaviour exactly)
    roi: tuple[int, int, int, int] | None = None
    exclude_boxes: tuple[tuple[int, int, int, int], ...] = ()
    hue_weighting: bool = False
    extended_diagnostics: bool = False
    # P7 option (default keeps P5/P6 behaviour exactly)
    centre_method: str = "colour_centroid"
    edge_params: bm.EdgeParams | None = None  # None: the method version's defaults

    def __post_init__(self) -> None:
        if self.centre_method not in CENTRE_METHODS:
            raise ValueError(f"centre_method must be one of {CENTRE_METHODS}")
        if self.edge_params is not None and not isinstance(self.edge_params, bm.EdgeParams):
            raise ValueError("edge_params must be a ball_measure.EdgeParams")
        if self.edge_params is not None and self.centre_method == "colour_centroid":
            raise ValueError("edge_params needs an edge centre_method")
        if self.roi is not None:
            object.__setattr__(self, "roi", _box(self.roi, "roi"))
        object.__setattr__(self, "exclude_boxes", tuple(_box(b, "exclude_boxes entry") for b in self.exclude_boxes))
        for name in ("hue_weighting", "extended_diagnostics"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if not isinstance(self.object_id, str) or not tr.OBJECT_ID_RE.fullmatch(self.object_id):
            raise ValueError(f"object_id must match {tr.OBJECT_ID_RE.pattern}")
        if not (math.isfinite(self.hue_center_deg) and 0 < self.hue_tolerance_deg < 180):
            raise ValueError("hue_center_deg must be finite and 0 < hue_tolerance_deg < 180")
        for name in ("min_chroma", "min_area_px", "max_area_px", "edge_margin_px"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if not 0 < self.min_area_px <= self.max_area_px:
            raise ValueError("need 0 < min_area_px <= max_area_px")
        if not (math.isfinite(self.weight_floor_chroma) and 0 <= self.weight_floor_chroma < self.min_chroma):
            raise ValueError("need 0 <= weight_floor_chroma < min_chroma")

    @property
    def restricted(self) -> bool:
        return self.roi is not None or bool(self.exclude_boxes)

    def to_json_dict(self) -> dict:
        """P5 fields always; P6 options only when set (a default config serialises exactly as in P5)."""
        d = {k: getattr(self, k) for k in ("object_id", "hue_center_deg", "hue_tolerance_deg", "min_chroma",
                                            "weight_floor_chroma", "min_area_px", "max_area_px", "edge_margin_px")}
        if self.roi is not None:
            d["roi"] = list(self.roi)
        if self.exclude_boxes:
            d["exclude_boxes"] = [list(b) for b in self.exclude_boxes]
        if self.hue_weighting:
            d["hue_weighting"] = True
        if self.extended_diagnostics:
            d["extended_diagnostics"] = True
        if self.uses_edges:
            d["centre_method"] = self.centre_method
            d["edge_method_version"] = bm.METHOD_VERSION
            d["edge_params"] = self.resolved_edge_params().to_json_dict()
        return d

    @property
    def uses_edges(self) -> bool:
        return self.centre_method != "colour_centroid"

    def resolved_edge_params(self) -> bm.EdgeParams:
        model = {"edge_circle": "circle", "edge_ellipse": "ellipse"}.get(self.centre_method, "stadium")
        base = self.edge_params if self.edge_params is not None else bm.EdgeParams(model=model)
        if base.model != model:
            raise ValueError(f"edge_params.model {base.model!r} does not match centre_method {self.centre_method!r}")
        return base

    @classmethod
    def from_json_dict(cls, d: dict) -> "TrackerConfig":
        kw = {k: d[k] for k in ("object_id", "hue_center_deg", "hue_tolerance_deg", "min_chroma",
                                "weight_floor_chroma", "min_area_px", "max_area_px", "edge_margin_px")}
        if "roi" in d:
            kw["roi"] = tuple(d["roi"])
        if "exclude_boxes" in d:
            kw["exclude_boxes"] = tuple(tuple(b) for b in d["exclude_boxes"])
        for k in ("hue_weighting", "extended_diagnostics"):
            if k in d:
                kw[k] = d[k]
        if "centre_method" in d:
            if d.get("edge_method_version") != bm.METHOD_VERSION:
                raise ValueError(f"edge method version {d.get('edge_method_version')!r} is not {bm.METHOD_VERSION}")
            kw["centre_method"] = d["centre_method"]
            ep = dict(d["edge_params"])
            ep["inner_frac"] = tuple(ep["inner_frac"])
            kw["edge_params"] = bm.EdgeParams(**ep)
        return cls(**kw)


def _box(value, name: str) -> tuple[int, int, int, int]:
    try:
        b = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be (x0, y0, x1, y1)") from None
    if len(b) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in b):
        raise ValueError(f"{name} must be four ints (x0, y0, x1, y1), got {value!r}")
    if not (0 <= b[0] < b[2] and 0 <= b[1] < b[3]):
        raise ValueError(f"{name} must satisfy 0 <= x0 < x1 and 0 <= y0 < y1, got {b}")
    return b


@dataclass(frozen=True)
class Detection:
    """What detect() found in one image (no frame identity, no time)."""
    status: str
    x: float | None
    y: float | None
    n_candidates: int
    area_px: int  # largest candidate's mask area (0 if none)
    second_area_px: int  # second-largest candidate's area (0 if none)
    bbox: tuple[int, int, int, int] | None  # x, y, w, h of the largest candidate
    fill_ratio: float | None  # area / bbox area (a disc gives about pi/4)
    weight_sum: float | None
    # extended diagnostics (None unless config.extended_diagnostics; omitted from JSON when None)
    n_mask_px: int | None = None
    largest_small_component_px: int | None = None
    # P7 edge centre method (None unless config.uses_edges; omitted from JSON when None)
    x_colour: float | None = None  # the P5 colour centroid, for comparison
    y_colour: float | None = None
    sigma_x: float | None = None
    sigma_y: float | None = None
    cov_xy: float | None = None
    diameter_px: float | None = None
    sigma_diameter_px: float | None = None
    edge: dict | None = None  # edge-fit diagnostics (n_inliers, max_gap_deg, axes, rms, rejection reason)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox) if self.bbox is not None else None
        for k in OPTIONAL_DETECTION_KEYS:
            if d[k] is None:
                del d[k]
        return d


OPTIONAL_DETECTION_KEYS = ("n_mask_px", "largest_small_component_px", "x_colour", "y_colour", "sigma_x", "sigma_y",
                           "cov_xy", "diameter_px", "sigma_diameter_px", "edge")


@dataclass(frozen=True)
class FrameDetection:
    frame_index: int
    t: Fraction
    detection: Detection

    def to_json_dict(self) -> dict:
        return {"frame_index": self.frame_index, "t": tr.timestamp_to_json(self.t), **self.detection.to_json_dict()}


# ------------------------------------------------------------------ one frame


def _chroma_hue(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = rgb.astype(np.int32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(axis=2), f.min(axis=2)
    c = mx - mn
    safe = np.where(c == 0, 1, c).astype(np.float64)
    h = np.where(mx == r, ((g - b) / safe) % 6.0,
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    return c, np.where(c == 0, 0.0, h)


def detect(rgb: np.ndarray, config: TrackerConfig) -> Detection:
    """Detect the target in ONE RGB uint8 image (H, W, 3). Pure and deterministic."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an RGB uint8 array of shape (H, W, 3)")
    height, width = rgb.shape[:2]
    chroma, hue = _chroma_hue(rgb)
    dh = np.abs((hue - config.hue_center_deg + 180.0) % 360.0 - 180.0)
    mask = ((chroma >= config.min_chroma) & (dh <= config.hue_tolerance_deg)).astype(np.uint8)
    allowed = _allowed_mask(height, width, config) if config.restricted else None
    if allowed is not None:
        mask &= allowed
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= config.min_area_px]
    comps.sort(key=lambda a: (-a[0], a[1]))
    extra = {}
    if config.extended_diagnostics:
        small = [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] < config.min_area_px]
        extra = {"n_mask_px": int(mask.sum()), "largest_small_component_px": max(small, default=0)}
    if not comps:
        return Detection(MISSING, None, None, 0, 0, 0, None, None, None, **extra)
    area, k = comps[0]
    second = comps[1][0] if len(comps) > 1 else 0
    bx, by, bw, bh = (int(stats[k, s]) for s in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    bbox, fill = (bx, by, bw, bh), area / float(bw * bh)
    if len(comps) > 1:
        return Detection(AMBIGUOUS, None, None, len(comps), area, second, bbox, fill, None, **extra)
    m = config.edge_margin_px
    if bx - m <= 0 or by - m <= 0 or bx + bw + m >= width or by + bh + m >= height:
        return Detection(TOUCHES_BORDER, None, None, 1, area, 0, bbox, fill, None, **extra)
    if area > config.max_area_px:
        return Detection(AREA_OUT_OF_RANGE, None, None, 1, area, 0, bbox, fill, None, **extra)
    region = (labels == k).astype(np.uint8)
    if m > 0:
        region = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=m)
    if allowed is not None and bool((region & (1 - allowed)).any()):
        return Detection(TOUCHES_BORDER, None, None, 1, area, 0, bbox, fill, None, **extra)  # ROI/exclusion edge
    wc = np.maximum(chroma.astype(np.float64) - config.weight_floor_chroma, 0.0)
    if config.hue_weighting:
        wc = np.where(dh <= config.hue_tolerance_deg + HUE_WEIGHT_MARGIN_DEG, wc, 0.0)
    w = np.where(region > 0, wc, 0.0)
    total = float(w.sum())
    if not total > 0:
        return Detection(MISSING, None, None, 1, area, 0, bbox, fill, None, **extra)
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    x = float((w.sum(axis=0) * cols).sum() / total)
    y = float((w.sum(axis=1) * rows).sum() / total)
    if not config.uses_edges:
        return Detection(DETECTED, x, y, 1, area, 0, bbox, fill, total, **extra)
    return _edge_refine(rgb, config, labels == k, x, y, area, bbox, fill, total, allowed, extra)


def _edge_refine(rgb, config: TrackerConfig, comp: np.ndarray, x: float, y: float, area: int, bbox, fill, total,
                 allowed, extra: dict) -> Detection:
    """P7: replace the colour centroid by the outline centre (decompile.ball_measure) of the coarse component."""
    ii, jj = np.nonzero(comp)
    extent = float(np.max(np.hypot(jj + 0.5 - x, ii + 0.5 - y))) + 1.0
    r0 = max(math.sqrt(area / math.pi), 2.0)
    m = bm.measure_ball(rgb, x, y, r0, extent=extent, allowed=allowed, params=config.resolved_edge_params())
    diag = {"n_rays": m.n_rays, "n_edges": m.n_edges, "n_inliers": m.n_inliers, "max_gap_deg": m.max_gap_deg,
            "rms_residual_px": m.rms_residual_px, "semi_major_px": m.semi_major_px, "semi_minor_px": m.semi_minor_px,
            "angle_deg": m.angle_deg, "coarse_extent_px": extent}
    if m.status != bm.MEASURED:
        diag["rejection_reason"] = m.reason
        return Detection(EDGE_REJECTED, None, None, 1, area, 0, bbox, fill, total, **extra, x_colour=x, y_colour=y,
                         edge=diag)
    return Detection(DETECTED, m.x, m.y, 1, area, 0, bbox, fill, total, **extra, x_colour=x, y_colour=y,
                     sigma_x=m.sigma_x, sigma_y=m.sigma_y, cov_xy=m.cov_xy, diameter_px=m.diameter_px,
                     sigma_diameter_px=m.sigma_diameter_px, edge=diag)


def _allowed_mask(height: int, width: int, config: TrackerConfig) -> np.ndarray:
    allowed = np.zeros((height, width), np.uint8)
    if config.roi is None:
        allowed[:] = 1
    else:
        x0, y0, x1, y1 = config.roi
        allowed[y0:y1, x0:x1] = 1
    for x0, y0, x1, y1 in config.exclude_boxes:
        allowed[y0:y1, x0:x1] = 0
    return allowed


# ------------------------------------------------------------------ seed-derived configuration


def config_from_seed(rgb: np.ndarray, seed_box, *, object_id: str = "obj_01", roi=None, exclude_boxes=(),
                     extended_diagnostics: bool = True) -> tuple[TrackerConfig, dict]:
    """Derive a colour model from ONE frame and a box drawn around the target. Deterministic.

    Returns (config, derivation record). Raises SeedError when the target is not a coloured
    object separable from its surroundings, or when the derived config does not detect it in
    the seed frame inside the box.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an RGB uint8 array of shape (H, W, 3)")
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _box(seed_box, "seed_box")
    if x1 > width or y1 > height:
        raise SeedError(f"seed box {seed_box} exceeds the {width}x{height} frame")
    if x1 - x0 < SEED_MIN_SIDE_PX or y1 - y0 < SEED_MIN_SIDE_PX:
        raise SeedError(f"seed box sides must be >= {SEED_MIN_SIDE_PX} px")
    probe = TrackerConfig(object_id=object_id, roi=roi, exclude_boxes=exclude_boxes)  # validates the boxes
    chroma, hue = _chroma_hue(rgb)
    chroma = chroma.astype(np.float64)
    rows = np.arange(height, dtype=np.float64)[:, None] + 0.5
    cols = np.arange(width, dtype=np.float64)[None, :] + 0.5
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = SEED_CORE_SCALE * (x1 - x0) / 2.0, SEED_CORE_SCALE * (y1 - y0) / 2.0
    core = ((cols - cx) / rx) ** 2 + ((rows - cy) / ry) ** 2 <= 1.0
    pad = max(6, round(0.5 * max(x1 - x0, y1 - y0)))
    outer = np.zeros((height, width), bool)
    outer[max(0, y0 - pad):min(height, y1 + pad), max(0, x0 - pad):min(width, x1 + pad)] = True
    inner = np.zeros((height, width), bool)
    e = SEED_EDGE_SKIP_PX
    inner[max(0, y0 - e):min(height, y1 + e), max(0, x0 - e):min(width, x1 + e)] = True
    ring = outer & ~inner
    if probe.restricted:
        ring &= _allowed_mask(height, width, probe).astype(bool)
    core_c, core_h = chroma[core], hue[core]
    if core_c.size == 0 or not ring.any():
        raise SeedError("seed box gives no core or background sample")
    if float(np.median(core_c)) < SEED_MIN_CORE_CHROMA:
        raise SeedError(f"the target has too little colour (median chroma {float(np.median(core_c)):.1f} < "
                        f"{SEED_MIN_CORE_CHROMA}); only coloured balls are supported")
    ang = np.deg2rad(core_h)
    hue_center = float(np.rad2deg(np.arctan2((core_c * np.sin(ang)).sum(), (core_c * np.cos(ang)).sum())) % 360.0)
    dh_core = np.abs((core_h - hue_center + 180.0) % 360.0 - 180.0)
    tol = float(np.clip(np.percentile(dh_core, 95) + 8.0, *SEED_HUE_TOL_RANGE))
    c_target = float(np.percentile(core_c, 10))
    ring_c, ring_h = chroma[ring], hue[ring]
    same_hue = np.abs((ring_h - hue_center + 180.0) % 360.0 - 180.0) <= tol
    c_bg_same = float(np.percentile(ring_c[same_hue], 99.5)) if same_hue.any() else 0.0
    if c_target <= c_bg_same + SEED_MIN_SEPARATION:
        raise SeedError(f"the target (chroma p10 {c_target:.1f}) is not separable by colour from same-hue background "
                        f"(chroma p99.5 {c_bg_same:.1f}); use an ROI/exclusion boxes or a more distinct ball")
    min_chroma = max(15, int(round((c_target + c_bg_same) / 2.0)))
    floor = float(min(np.percentile(ring_c, 99.0), min_chroma - 1))
    disc_area = math.pi / 4.0 * (x1 - x0) * (y1 - y0)
    min_area = max(8, int(0.15 * disc_area))
    max_area = max(min_area + 1, int(math.ceil(6.0 * disc_area)))
    config = TrackerConfig(object_id=object_id, hue_center_deg=round(hue_center, 6), hue_tolerance_deg=round(tol, 6),
                           min_chroma=min_chroma, weight_floor_chroma=round(floor, 6), min_area_px=min_area,
                           max_area_px=max_area, edge_margin_px=2, roi=roi, exclude_boxes=exclude_boxes,
                           hue_weighting=True, extended_diagnostics=extended_diagnostics)
    check = detect(rgb, config)
    if check.status != DETECTED or not (x0 <= check.x < x1 and y0 <= check.y < y1):
        raise SeedError(f"the derived configuration does not detect the target inside the seed box "
                        f"(status {check.status}, centroid ({check.x}, {check.y}))")
    derivation = {
        "method": "colour model from one seed frame: chroma-weighted circular mean hue of the core ellipse; hue "
                  "tolerance = p95 core hue deviation + 8 deg (clipped); min chroma = midpoint of core chroma p10 and "
                  "same-hue background chroma p99.5; weight floor = background chroma p99 (capped); areas from the "
                  "box's inscribed disc",
        "seed_box": [x0, y0, x1, y1], "core_scale": SEED_CORE_SCALE, "background_pad_px": pad,
        "edge_skip_px": SEED_EDGE_SKIP_PX, "core_pixels": int(core.sum()), "background_pixels": int(ring.sum()),
        "core_chroma_median": float(np.median(core_c)), "core_chroma_p10": c_target,
        "core_hue_p95_deviation_deg": float(np.percentile(dh_core, 95)),
        "background_same_hue_pixels": int(same_hue.sum()), "background_same_hue_chroma_p99_5": c_bg_same,
        "background_chroma_p99": float(np.percentile(ring_c, 99.0)), "disc_area_estimate_px": disc_area,
        "seed_frame_detection": check.to_json_dict(),
    }
    return config, derivation


# ------------------------------------------------------------------ many frames


@dataclass(frozen=True)
class TrackingResult:
    config: TrackerConfig
    width: int
    height: int
    detections: tuple[FrameDetection, ...]  # sorted by frame index
    source_sha256: str | None = field(default=None)

    @property
    def failures(self) -> tuple[FrameDetection, ...]:
        return tuple(d for d in self.detections if d.detection.status != DETECTED)

    def status_counts(self) -> dict[str, int]:
        statuses = STATUSES + ((EDGE_REJECTED,) if self.config.uses_edges else ())
        return {s: sum(1 for d in self.detections if d.detection.status == s) for s in statuses}

    def to_trajectory_set(self, *, synthetic: bool) -> tr.TrajectorySet:
        """CLASSICAL_TRACKER TrajectorySet; `synthetic` = provenance of the tracked video. Refuses any failed frame."""
        if not isinstance(synthetic, bool):
            raise TypeError("synthetic must be a bool (the provenance of the tracked video)")
        if self.failures:
            raise TrackingError([f"frame {d.frame_index} (t = {d.t}): {d.detection.status}" for d in self.failures])
        data = {
            "schema": tr.SCHEMA, "synthetic": synthetic, "source": SOURCE_LABEL,
            "coordinate_system": {**tr.COORDINATE_CONSTANTS, "image_width": self.width, "image_height": self.height},
            "objects": [{"object_id": self.config.object_id,
                         "observations": [_observation_json(d) for d in self.detections]}],
        }
        try:
            return tr.TrajectorySet.from_json_dict(data, source_name="classical tracker")
        except tr.TrajectoryError as exc:
            raise TrackingError(exc.reasons) from None

    def to_json_dict(self) -> dict:
        method = ("per-frame independent colour segmentation (chroma + hue window), 8-connected components, "
                  "chroma-weighted centroid of the dilated component; no temporal smoothing, interpolation "
                  "or use of other frames")
        if self.config.uses_edges:
            method = ("per-frame independent colour segmentation (chroma + hue window) and 8-connected components "
                      f"as the COARSE stage; centre from the ball outline ({bm.METHOD_VERSION}, "
                      f"{self.config.centre_method}): per-ray 50% transition between locally estimated ball and "
                      "background colours, robust outline fit, per-frame uncertainty; no temporal smoothing, "
                      "interpolation or use of other frames")
        return {
            "kind": "tracking_report", "tracker_version": TRACKER_VERSION, "source_label": SOURCE_LABEL,
            "config": self.config.to_json_dict(), "pixel_convention": PIXEL_CONVENTION,
            "method": method,
            "image": {"width": self.width, "height": self.height}, "source_sha256": self.source_sha256,
            "n_frames": len(self.detections), "status_counts": self.status_counts(),
            "frames": [d.to_json_dict() for d in self.detections],
        }


def _observation_json(d: FrameDetection) -> dict:
    o = {"frame_index": d.frame_index, "t": tr.timestamp_to_json(d.t), "x": d.detection.x, "y": d.detection.y}
    det = d.detection
    if det.sigma_x is not None:
        o["sigma_x"], o["sigma_y"] = det.sigma_x, det.sigma_y
    if det.diameter_px is not None:
        o["diameter_px"] = det.diameter_px
        if det.sigma_diameter_px is not None and det.sigma_diameter_px > 0:
            o["sigma_diameter_px"] = det.sigma_diameter_px
    return o


def track_frames(items: Iterable[tuple[fr.FrameTimestamp, np.ndarray]], config: TrackerConfig, *,
                 source_sha256: str | None = None) -> TrackingResult:
    """Detect in every (timestamp, RGB) pair, in ANY order; results are sorted by frame index."""
    out: dict[int, FrameDetection] = {}
    size: tuple[int, int] | None = None
    for ts, rgb in items:
        if not isinstance(ts, fr.FrameTimestamp) or not isinstance(ts.t, Fraction):
            raise TypeError("each item must be (FrameTimestamp, RGB array) with an exact Fraction time")
        if ts.index in out:
            raise ValueError(f"frame {ts.index} given twice")
        if size is None:
            size = (rgb.shape[1], rgb.shape[0])
        elif size != (rgb.shape[1], rgb.shape[0]):
            raise ValueError(f"frame {ts.index}: size {rgb.shape[1]}x{rgb.shape[0]} differs from {size[0]}x{size[1]}")
        out[ts.index] = FrameDetection(ts.index, ts.t, detect(rgb, config))
    if size is None:
        raise ValueError("no frames")
    ordered = tuple(out[i] for i in sorted(out))
    for a, b in zip(ordered, ordered[1:]):
        if not b.t > a.t:
            raise ValueError(f"frames {a.frame_index} and {b.frame_index}: time does not increase with frame index")
    return TrackingResult(config, size[0], size[1], ordered, source_sha256)


def track_extracted(frames_dir: str | Path, config: TrackerConfig, *, expected_source_sha256: str | None = None,
                    order: Iterable[int] | None = None, frames: Iterable[int] | None = None) -> TrackingResult:
    """Track an A2 export (frames.json + PNGs).

    order:  every exported frame index exactly once; only changes the processing order.
    frames: a SUBSET of exported frame indices to track (clip window / stride), in any order;
            the result keeps their original indices and exact timestamps. Not with `order`.
    """
    frames_dir = Path(frames_dir)
    table = fr.load_frame_table(frames_dir / fr.MANIFEST_NAME)
    if expected_source_sha256 is not None and table.source_sha256 != expected_source_sha256:
        raise ValueError("the frame export was made from a different video")
    by_index = {f.index: f for f in table.frames}
    if order is not None and frames is not None:
        raise ValueError("give either order or frames, not both")
    if frames is not None:
        indices = list(frames)
        if not indices or len(set(indices)) != len(indices) or not set(indices) <= set(by_index):
            raise ValueError("frames must be a non-empty subset of the exported frame indices, without repeats")
    else:
        indices = list(order) if order is not None else [f.index for f in table.frames]
        if sorted(indices) != sorted(by_index):
            raise ValueError("order must list every exported frame exactly once")

    def items():
        for i in indices:
            rgb = fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=i))
            if rgb.shape[:2] != (table.height, table.width):
                raise ValueError(f"frame {i}: PNG size does not match the frame table")
            yield by_index[i], rgb

    return track_frames(items(), config, source_sha256=table.source_sha256)
