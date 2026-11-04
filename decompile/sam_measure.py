"""SAM-mask-guided geometric ball measurement (method decompile.sam_measure/1).

A SAM mask is NOT treated as the physical centre of the object. Its centroid depends on how the
segmentation network places its boundary (blur, shading, background similarity, mask resolution), and P7
showed that a few tenths of a pixel of systematic measurement bias can break an otherwise good physics fit.
SAM is used for what it is good at: WHICH object (identity across frames) and WHERE its support is. The
centre, apparent diameter and their uncertainties come from the ORIGINAL decoded image pixels through the
outline method of decompile.ball_measure (P7, validated on the synthetic stress suite).

Per frame and object, independently (no temporal smoothing, no interpolation, no other frame):
  1. SAM record for the frame?  no -> MISSING (no coordinate).
  2. Full-frame mask from the official decoder (decompile.sam_tracking). Empty mask -> MISSING.
     Mask touching the image border -> UNCERTAIN (outline truncated).
     SAM confidence (when requested) below min_confidence -> UNCERTAIN.
  3. Coarse stage from the mask: centroid, area -> r0 = sqrt(area / pi), extent = largest distance from the
     centroid to a mask pixel (+1 px). These only START the outline search.
  4. Support region: rays may not sample pixels of OTHER SAM objects (their masks dilated by
     other_object_margin_px), so a neighbouring object is never used as "background".
  5. decompile.ball_measure.measure_ball (rays from the coarse centre; per-ray 50% transition between the
     locally estimated object and background colours; robust ellipse/stadium/circle fit; odd-harmonic
     asymmetry check; per-frame 1-sigma sigma_x, sigma_y including the method's systematic floor; apparent
     diameter and its sigma). Rejected outline -> UNCERTAIN, with the outline method's reason.
  6. Mask/outline agreement (identity check, not a centre estimate): the fitted outline must overlap the SAM
     mask (IoU >= min_iou), its diameter must agree with the mask's equivalent diameter within
     diameter_ratio_range, and its centre must lie within max_centre_offset_frac * diameter of the mask
     centroid. Otherwise UNCERTAIN (the outline probably locked onto something else).
  7. Otherwise MEASURED: x, y, sigma_x, sigma_y, diameter_px, sigma_diameter_px.

Only MEASURED observations ever enter a trajectory; UNCERTAIN and MISSING frames are recorded with their
reasons (and the mask centroid as a DIAGNOSTIC, never as an observation) and are never interpolated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import cv2
import numpy as np

from decompile import ball_measure as bm
from decompile import sam_tracking as st
from decompile import trajectories as tr

METHOD_VERSION = "decompile.sam_measure/1"
SOURCE_LABEL = "SAM_3_1"
MEASURED, UNCERTAIN, MISSING = "measured", "uncertain", "missing"
STATUSES = (MEASURED, UNCERTAIN, MISSING)


@dataclass(frozen=True)
class SamMeasureParams:
    outline_model: str = "ellipse"  # ball_measure model: "ellipse" | "stadium" | "circle"
    min_confidence: float = 0.5  # applies only when SAM returned confidences
    min_mask_area_px: int = 12
    other_object_margin_px: int = 2
    min_iou: float = 0.6
    diameter_ratio_range: tuple[float, float] = (0.75, 1.33)
    max_centre_offset_frac: float = 0.25
    edge_params: bm.EdgeParams | None = None  # None: ball_measure defaults for outline_model

    def __post_init__(self) -> None:
        if self.outline_model not in ("ellipse", "stadium", "circle"):
            raise ValueError("outline_model must be 'ellipse', 'stadium' or 'circle'")
        if self.edge_params is not None and self.edge_params.model != self.outline_model:
            raise ValueError("edge_params.model must equal outline_model")
        lo, hi = self.diameter_ratio_range
        if not (0 < lo < 1 < hi and 0 < self.min_iou < 1 and 0 < self.max_centre_offset_frac < 1):
            raise ValueError("invalid agreement thresholds")

    def resolved_edge_params(self) -> bm.EdgeParams:
        return self.edge_params if self.edge_params is not None else bm.EdgeParams(model=self.outline_model)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["diameter_ratio_range"] = list(self.diameter_ratio_range)
        d["edge_params"] = self.resolved_edge_params().to_json_dict()
        d["edge_method_version"] = bm.METHOD_VERSION
        return d


@dataclass(frozen=True)
class FrameMeasurement:
    frame_index: int
    t: Fraction
    object_id: str  # SAM object id
    status: str
    reason: str | None
    x: float | None = None
    y: float | None = None
    sigma_x: float | None = None
    sigma_y: float | None = None
    diameter_px: float | None = None
    sigma_diameter_px: float | None = None
    # diagnostics (never used as observations)
    mask_area_px: int | None = None
    mask_centroid: tuple[float, float] | None = None
    mask_equiv_diameter_px: float | None = None
    mask_placement: str | None = None
    confidence: float | None = None
    iou_outline_mask: float | None = None
    centre_offset_from_mask_px: float | None = None
    outline: dict | None = None

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["t"] = tr.timestamp_to_json(self.t)
        d["mask_centroid"] = list(self.mask_centroid) if self.mask_centroid is not None else None
        return d


def _ellipse_raster(shape, cx, cy, a, b, angle_deg) -> np.ndarray:
    """Pixels (centres at +0.5) inside the ellipse."""
    h, w = shape
    x0, x1 = max(0, int(cx - a - 2)), min(w, int(cx + a + 3))
    y0, y1 = max(0, int(cy - a - 2)), min(h, int(cy + a + 3))
    out = np.zeros(shape, bool)
    if x1 <= x0 or y1 <= y0:
        return out
    jj, ii = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx, dy = jj - cx, ii - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    out[y0:y1, x0:x1] = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return out


def measure_frame(rgb: np.ndarray, rec: st.SamObjectFrame | None, *, frame_index: int, t: Fraction, object_id: str,
                  others: list[st.SamObjectFrame] = (), params: SamMeasureParams = SamMeasureParams()) -> FrameMeasurement:
    """Measure ONE object in ONE frame from its SAM record and the original pixels. Pure and deterministic."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an RGB uint8 array of shape (H, W, 3)")
    h, w = rgb.shape[:2]
    base = dict(frame_index=frame_index, t=t, object_id=object_id)
    if rec is None:
        return FrameMeasurement(**base, status=MISSING, reason="SAM emitted no record for this object in this frame")
    if not rec.has_mask:
        return FrameMeasurement(**base, status=MISSING, reason="SAM emitted a box but no mask")
    conf = rec.mask_confidence if rec.mask_confidence is not None else rec.box_confidence
    mask = rec.full_mask(w, h)
    area = int(mask.sum())
    placement = rec.placement(w, h)
    if area < params.min_mask_area_px:
        return FrameMeasurement(**base, status=MISSING, reason=f"SAM mask has {area} px (< {params.min_mask_area_px})",
                                mask_area_px=area, mask_placement=placement, confidence=conf)
    ii, jj = np.nonzero(mask)
    mx, my = float(jj.mean() + 0.5), float(ii.mean() + 0.5)
    d_eq = 2.0 * math.sqrt(area / math.pi)
    diag = dict(mask_area_px=area, mask_centroid=(mx, my), mask_equiv_diameter_px=d_eq, mask_placement=placement,
                confidence=conf)
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        return FrameMeasurement(**base, status=UNCERTAIN, reason="SAM mask touches the image border (outline truncated)",
                                **diag)
    if conf is not None and conf < params.min_confidence:
        return FrameMeasurement(**base, status=UNCERTAIN,
                                reason=f"SAM confidence {conf:.3f} < {params.min_confidence}", **diag)
    allowed = None
    if others:
        occ = np.zeros((h, w), np.uint8)
        for o in others:
            if o.has_mask:
                occ |= o.full_mask(w, h).astype(np.uint8)
        if occ.any():
            if params.other_object_margin_px > 0:
                occ = cv2.dilate(occ, np.ones((3, 3), np.uint8), iterations=params.other_object_margin_px)
            allowed = (1 - occ).astype(np.uint8)
    extent = float(np.max(np.hypot(jj + 0.5 - mx, ii + 0.5 - my))) + 1.0
    r0 = max(math.sqrt(area / math.pi), 2.0)
    m = bm.measure_ball(rgb, mx, my, r0, extent=extent, allowed=allowed, params=params.resolved_edge_params())
    outline = {"n_rays": m.n_rays, "n_edges": m.n_edges, "n_inliers": m.n_inliers, "max_gap_deg": m.max_gap_deg,
               "rms_residual_px": m.rms_residual_px, "semi_major_px": m.semi_major_px, "semi_minor_px": m.semi_minor_px,
               "angle_deg": m.angle_deg, "asymmetry_px": m.asymmetry_px, "coarse_extent_px": extent,
               "rejection_reason": m.reason}
    if m.status != bm.MEASURED:
        return FrameMeasurement(**base, status=UNCERTAIN, reason=f"outline rejected: {m.reason}", outline=outline, **diag)
    ell = _ellipse_raster((h, w), m.x, m.y, m.semi_major_px, m.semi_minor_px, m.angle_deg)
    union = int((ell | mask).sum())
    iou = float((ell & mask).sum()) / union if union else 0.0
    offset = math.hypot(m.x - mx, m.y - my)
    ratio = m.diameter_px / d_eq
    lo, hi = params.diameter_ratio_range
    agree = dict(iou_outline_mask=iou, centre_offset_from_mask_px=offset, outline=outline)
    if iou < params.min_iou:
        return FrameMeasurement(**base, status=UNCERTAIN, reason=f"outline/mask overlap IoU {iou:.2f} < {params.min_iou}",
                                **diag, **agree)
    if not lo <= ratio <= hi:
        return FrameMeasurement(**base, status=UNCERTAIN, reason=f"outline diameter / mask diameter = {ratio:.2f} outside "
                                f"[{lo}, {hi}]", **diag, **agree)
    if offset > params.max_centre_offset_frac * m.diameter_px:
        return FrameMeasurement(**base, status=UNCERTAIN, reason=f"outline centre is {offset:.2f} px from the mask centroid "
                                f"(> {params.max_centre_offset_frac} x diameter)", **diag, **agree)
    return FrameMeasurement(**base, status=MEASURED, reason=None, x=m.x, y=m.y, sigma_x=m.sigma_x, sigma_y=m.sigma_y,
                            diameter_px=m.diameter_px, sigma_diameter_px=m.sigma_diameter_px, **diag, **agree)


# ------------------------------------------------------------------ many frames


@dataclass(frozen=True)
class SamMeasurementResult:
    sam_object_id: str
    trajectory_object_id: str
    width: int
    height: int
    params: SamMeasureParams
    frames: tuple[FrameMeasurement, ...]  # sorted by frame index
    video_sha256: str
    sam_result_sha256: str | None = None
    provenance: dict = field(default_factory=dict)

    def status_counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.frames if f.status == s) for s in STATUSES}

    def measured(self) -> tuple[FrameMeasurement, ...]:
        return tuple(f for f in self.frames if f.status == MEASURED)

    def to_trajectory_set(self, *, synthetic: bool) -> tr.TrajectorySet:
        """SAM_3_1 TrajectorySet of the MEASURED frames only (exact source timestamps; nothing interpolated)."""
        if not isinstance(synthetic, bool):
            raise TypeError("synthetic must be a bool (the provenance of the tracked video)")
        obs = []
        for f in self.measured():
            o = {"frame_index": f.frame_index, "t": tr.timestamp_to_json(f.t), "x": f.x, "y": f.y,
                 "sigma_x": f.sigma_x, "sigma_y": f.sigma_y, "diameter_px": f.diameter_px}
            if f.sigma_diameter_px is not None and f.sigma_diameter_px > 0:
                o["sigma_diameter_px"] = f.sigma_diameter_px
            obs.append(o)
        data = {"schema": tr.SCHEMA, "synthetic": synthetic, "source": SOURCE_LABEL,
                "coordinate_system": {**tr.COORDINATE_CONSTANTS, "image_width": self.width, "image_height": self.height},
                "objects": [{"object_id": self.trajectory_object_id, "observations": obs}]}
        return tr.TrajectorySet.from_json_dict(data, source_name="SAM 3.1 + outline measurement")

    def to_json_dict(self) -> dict:
        return {"kind": "sam_measurement_report", "method_version": METHOD_VERSION, "source_label": SOURCE_LABEL,
                "sam_object_id": self.sam_object_id, "trajectory_object_id": self.trajectory_object_id,
                "image": {"width": self.width, "height": self.height}, "video_sha256": self.video_sha256,
                "sam_result_sha256": self.sam_result_sha256, "params": self.params.to_json_dict(),
                "method": "SAM 3.1 supplies identity and the support region; the centre, apparent diameter and their "
                          "1-sigma uncertainties come from the object OUTLINE in the original pixels "
                          f"({bm.METHOD_VERSION}); mask centroids are diagnostics only; per frame, no smoothing or "
                          "interpolation",
                "pixel_convention": "pixel (row i, col j) has its centre at (j + 0.5, i + 0.5)",
                "status_counts": self.status_counts(), "provenance": self.provenance,
                "frames": [f.to_json_dict() for f in self.frames]}


def select_object(result: st.SamTrackingResult, frames: list[int], rule: str | dict) -> str:
    """Choose the SAM object deterministically: {"id": "<sam id>"} or "most_frames" (ties: larger median mask
    area, then smaller id). Raises SamTrackingError when nothing qualifies."""
    ids = result.object_ids()
    if not ids:
        raise st.SamTrackingError(["SAM returned no object (empty match)"])
    if isinstance(rule, dict):
        oid = str(rule["id"])
        if oid not in ids:
            raise st.SamTrackingError([f"SAM object {oid} is not in the result (objects: {list(ids)})"])
        return oid
    if rule != "most_frames":
        raise ValueError(f"unknown object selection rule {rule!r}")
    clip = set(frames)

    def key(oid):
        fs = {i: r for i, r in result.frames_of(oid).items() if i in clip}
        areas = [(r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) for r in fs.values()]
        return (-len(fs), -(float(np.median(areas)) if areas else 0.0), len(oid), oid)

    return sorted(ids, key=key)[0]


def measure_track(result: st.SamTrackingResult, frame_source, frames: list[int], *, sam_object_id: str,
                  trajectory_object_id: str = "obj_01", params: SamMeasureParams = SamMeasureParams(),
                  sam_result_sha256: str | None = None) -> SamMeasurementResult:
    """Measure one SAM object on the given source frames.

    frame_source(i) -> (FrameTimestamp, RGB uint8) for source frame i (the A2 export of the SAME video).
    """
    if not result.completed:
        raise st.SamTrackingError([f"the SAM response is {result.outcome.get('status')!r} "
                                   f"({result.outcome.get('reason')}); only a completed response is measured"])
    if len(set(frames)) != len(frames):
        raise ValueError("frames must not repeat")
    track = result.frames_of(sam_object_id)
    out = []
    W, H = result.media["width"], result.media["height"]
    for i in sorted(frames):
        ts, rgb = frame_source(i)
        if ts.index != i:
            raise ValueError(f"frame source returned frame {ts.index} for {i}")
        if rgb.shape[:2] != (H, W):
            raise st.SamTrackingError([f"frame {i}: image {rgb.shape[1]}x{rgb.shape[0]} differs from the SAM media "
                                       f"{W}x{H}"])
        if str(i) in result.frame_times and result.frame_time(i) != ts.t:
            raise st.SamTrackingError([f"frame {i}: timestamp differs between the SAM result and the frame table"])
        others = [r for oid, r in result.at(i).items() if oid != sam_object_id]
        out.append(measure_frame(rgb, track.get(i), frame_index=i, t=ts.t, object_id=sam_object_id, others=others,
                                 params=params))
    return SamMeasurementResult(sam_object_id, trajectory_object_id, W, H, params, tuple(out),
                                result.video["sha256"], sam_result_sha256,
                                {"sam_model": result.model, "sam_prompt": result.prompt})
