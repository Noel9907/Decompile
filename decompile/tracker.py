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

from decompile import frames as fr
from decompile import trajectories as tr

TRACKER_VERSION = "decompile.tracker/1"
SOURCE_LABEL = "CLASSICAL_TRACKER"
DETECTED, MISSING, AMBIGUOUS, TOUCHES_BORDER, AREA_OUT_OF_RANGE = (
    "detected", "missing", "ambiguous", "touches_border", "area_out_of_range")
STATUSES = (DETECTED, MISSING, AMBIGUOUS, TOUCHES_BORDER, AREA_OUT_OF_RANGE)
PIXEL_CONVENTION = "pixel (row i, col j) has its centre at (j + 0.5, i + 0.5)"


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

    def __post_init__(self) -> None:
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

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox) if self.bbox is not None else None
        return d


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
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(int(stats[k, cv2.CC_STAT_AREA]), k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= config.min_area_px]
    comps.sort(key=lambda a: (-a[0], a[1]))
    if not comps:
        return Detection(MISSING, None, None, 0, 0, 0, None, None, None)
    area, k = comps[0]
    second = comps[1][0] if len(comps) > 1 else 0
    bx, by, bw, bh = (int(stats[k, s]) for s in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    bbox, fill = (bx, by, bw, bh), area / float(bw * bh)
    if len(comps) > 1:
        return Detection(AMBIGUOUS, None, None, len(comps), area, second, bbox, fill, None)
    m = config.edge_margin_px
    if bx - m <= 0 or by - m <= 0 or bx + bw + m >= width or by + bh + m >= height:
        return Detection(TOUCHES_BORDER, None, None, 1, area, 0, bbox, fill, None)
    if area > config.max_area_px:
        return Detection(AREA_OUT_OF_RANGE, None, None, 1, area, 0, bbox, fill, None)
    region = (labels == k).astype(np.uint8)
    if m > 0:
        region = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=m)
    w = np.where(region > 0, np.maximum(chroma.astype(np.float64) - config.weight_floor_chroma, 0.0), 0.0)
    total = float(w.sum())
    if not total > 0:
        return Detection(MISSING, None, None, 1, area, 0, bbox, fill, None)
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    x = float((w.sum(axis=0) * cols).sum() / total)
    y = float((w.sum(axis=1) * rows).sum() / total)
    return Detection(DETECTED, x, y, 1, area, 0, bbox, fill, total)


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
        return {s: sum(1 for d in self.detections if d.detection.status == s) for s in STATUSES}

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
                         "observations": [{"frame_index": d.frame_index, "t": tr.timestamp_to_json(d.t),
                                           "x": d.detection.x, "y": d.detection.y} for d in self.detections]}],
        }
        try:
            return tr.TrajectorySet.from_json_dict(data, source_name="classical tracker")
        except tr.TrajectoryError as exc:
            raise TrackingError(exc.reasons) from None

    def to_json_dict(self) -> dict:
        return {
            "kind": "tracking_report", "tracker_version": TRACKER_VERSION, "source_label": SOURCE_LABEL,
            "config": asdict(self.config), "pixel_convention": PIXEL_CONVENTION,
            "method": ("per-frame independent colour segmentation (chroma + hue window), 8-connected components, "
                       "chroma-weighted centroid of the dilated component; no temporal smoothing, interpolation "
                       "or use of other frames"),
            "image": {"width": self.width, "height": self.height}, "source_sha256": self.source_sha256,
            "n_frames": len(self.detections), "status_counts": self.status_counts(),
            "frames": [d.to_json_dict() for d in self.detections],
        }


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
                    order: Iterable[int] | None = None) -> TrackingResult:
    """Track an A2 export (frames.json + PNGs). `order` (frame indices) only changes the processing order."""
    frames_dir = Path(frames_dir)
    table = fr.load_frame_table(frames_dir / fr.MANIFEST_NAME)
    if expected_source_sha256 is not None and table.source_sha256 != expected_source_sha256:
        raise ValueError("the frame export was made from a different video")
    by_index = {f.index: f for f in table.frames}
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
