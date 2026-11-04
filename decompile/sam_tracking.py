"""Typed, serialisable SAM 3.1 video tracking result keyed by SOURCE frame index (decompile.sam_tracking/1).

Built ONLY from the official meta_sam_parser result records (never from our own parsing of the special
tokens). Every mask and box is associated with its SAM object id and its source frame index, and is
validated against the exact frame table of the uploaded video (decompile.frames, A2):

  * frame indices: SAM's <Nf> is the zero-based source frame index. Records are keyed by that index, never
    by stream position; frames may arrive in any order, may skip, and absent frames stay absent ("missing").
    An index outside the frame table is an error (the result does not belong to this video).
  * timestamps: each frame's exact presentation timestamp t (Fraction seconds, from ffprobe pts * time_base)
    comes from the frame table; nominal fps is never used.
  * geometry: SAM boxes arrive inclusive (x2, y2) and the official parser converts them to half-open
    (right = x2 + 1). Every box must lie inside the DISPLAYED (rotation-applied) frame size of the frame
    table, and the documented w/h of every box token must equal that size (checked on the raw output by
    counting the documented token suffix; a mismatch, e.g. a rotation disagreement, rejects the result).
  * masks: a mask raster (H rows x W columns) is box-local. When its size equals the box size it is placed
    1:1 at the box origin ("box_local"); a full-frame raster is placed at the origin ("full_frame"); any
    other size is resampled nearest-neighbour onto the box ("box_resampled", flagged). The raster decodes
    with the official decode_mask_to_raster.
  * revisions: a later revision of the same (frame, object) mask supersedes the earlier one (counted).

JSON keeps the masks exactly as SAM emitted them (encoding marker + base85 payload + raster size), so the
file is self-contained and re-decodable with the official package.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import meta_sam_parser as msp

from decompile import frames as fr

SCHEMA = "decompile.sam_tracking/1"
PLACEMENTS = ("box_local", "full_frame", "box_resampled")


class SamTrackingError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems[:8]) + (" ..." if len(self.problems) > 8 else ""))


@dataclass(frozen=True)
class SamObjectFrame:
    frame_index: int
    object_id: str
    box: tuple[int, int, int, int]  # half-open x0, y0, x1, y1 (source pixels)
    box_confidence: float | None
    mask_encoding: str | None  # "lossless" | "one_bit" | None (no mask record)
    mask_payload: str | None
    mask_width: int | None  # raster columns
    mask_height: int | None  # raster rows
    mask_confidence: float | None
    mask_revision: int | None

    @property
    def has_mask(self) -> bool:
        return self.mask_payload is not None

    def raster(self) -> np.ndarray:
        """The decoded binary raster (rows, cols), uint8 0/1, with the official decoder."""
        if not self.has_mask:
            raise SamTrackingError([f"frame {self.frame_index} object {self.object_id}: no mask"])
        m = msp.SegmentationMask(self.mask_encoding, self.mask_payload, self.mask_width, self.mask_height)
        try:
            raw = msp.decode_mask_to_raster(m)
        except msp.InvalidSegmentationMaskError as exc:
            raise SamTrackingError([f"frame {self.frame_index} object {self.object_id}: {exc}"]) from None
        return np.frombuffer(raw, np.uint8).reshape(self.mask_height, self.mask_width)

    def placement(self, width: int, height: int) -> str:
        x0, y0, x1, y1 = self.box
        if (self.mask_width, self.mask_height) == (x1 - x0, y1 - y0):
            return "box_local"
        if (self.mask_width, self.mask_height) == (width, height):
            return "full_frame"
        return "box_resampled"

    def full_mask(self, width: int, height: int) -> np.ndarray:
        """Boolean (height, width) mask in source-frame pixels."""
        r = self.raster().astype(bool)
        out = np.zeros((height, width), bool)
        x0, y0, x1, y1 = self.box
        mode = self.placement(width, height)
        if mode == "box_local":
            out[y0:y1, x0:x1] = r
        elif mode == "full_frame":
            out[:, :] = r
        else:
            bw, bh = x1 - x0, y1 - y0
            rows = np.minimum((np.arange(bh) + 0.5) * self.mask_height / bh, self.mask_height - 1).astype(int)
            cols = np.minimum((np.arange(bw) + 0.5) * self.mask_width / bw, self.mask_width - 1).astype(int)
            out[y0:y1, x0:x1] = r[rows][:, cols]
        return out

    def to_json_dict(self) -> dict:
        return {"frame_index": self.frame_index, "object_id": self.object_id, "box": list(self.box),
                "box_confidence": self.box_confidence,
                "mask": None if not self.has_mask else {
                    "encoding": self.mask_encoding, "payload": self.mask_payload, "width": self.mask_width,
                    "height": self.mask_height, "confidence": self.mask_confidence, "revision": self.mask_revision}}

    @classmethod
    def from_json_dict(cls, d: dict) -> "SamObjectFrame":
        m = d.get("mask")
        return cls(int(d["frame_index"]), str(d["object_id"]), tuple(int(v) for v in d["box"]), d.get("box_confidence"),
                   m["encoding"] if m else None, m["payload"] if m else None, m["width"] if m else None,
                   m["height"] if m else None, m.get("confidence") if m else None, m.get("revision") if m else None)


@dataclass(frozen=True)
class SamTrackingResult:
    video: dict  # {"sha256", "filename", "size_bytes"}
    media: dict  # {"width", "height", "rotation_deg", "frame_count", "time_base"}
    prompt: dict
    model: str
    outcome: dict  # {"status": "completed" | "incomplete", "reason", "detail"}
    records: tuple[SamObjectFrame, ...]  # sorted by (frame_index, object_id)
    frame_times: dict  # str(frame_index) -> {"num", "den", "pts"} for every frame with a record
    diagnostics: tuple[dict, ...] = ()
    text_lines: tuple[str, ...] = ()
    superseded_mask_revisions: int = 0
    raw_output_sha256: str | None = None
    usage: dict | None = None
    request_sha256: str | None = None
    provenance: dict = field(default_factory=dict)

    # -- queries

    @property
    def completed(self) -> bool:
        return self.outcome.get("status") == "completed"

    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted({r.object_id for r in self.records}, key=lambda s: (len(s), s)))

    def frames_of(self, object_id: str) -> dict[int, SamObjectFrame]:
        return {r.frame_index: r for r in self.records if r.object_id == object_id}

    def at(self, frame_index: int) -> dict[str, SamObjectFrame]:
        return {r.object_id: r for r in self.records if r.frame_index == frame_index}

    def frame_time(self, frame_index: int) -> Fraction:
        d = self.frame_times[str(frame_index)]
        return Fraction(d["num"], d["den"])

    def summary(self) -> dict:
        per = {}
        for oid in self.object_ids():
            fs = self.frames_of(oid)
            confs = [r.mask_confidence for r in fs.values() if r.mask_confidence is not None]
            per[oid] = {"n_frames": len(fs), "first_frame": min(fs), "last_frame": max(fs),
                        "n_with_mask": sum(1 for r in fs.values() if r.has_mask),
                        "median_box_area_px": float(np.median([(r.box[2] - r.box[0]) * (r.box[3] - r.box[1])
                                                                for r in fs.values()])),
                        "min_confidence": min(confs) if confs else None,
                        "median_confidence": float(np.median(confs)) if confs else None}
        return {"outcome": self.outcome, "n_records": len(self.records), "n_objects": len(per),
                "n_frames_with_records": len({r.frame_index for r in self.records}),
                "n_diagnostics": len(self.diagnostics), "n_text_lines": len(self.text_lines),
                "superseded_mask_revisions": self.superseded_mask_revisions, "objects": per, "usage": self.usage}

    # -- JSON

    def to_json_dict(self) -> dict:
        return {"schema": SCHEMA, "model": self.model, "video": self.video, "media": self.media, "prompt": self.prompt,
                "outcome": self.outcome, "provenance": self.provenance, "usage": self.usage,
                "request_sha256": self.request_sha256, "raw_output_sha256": self.raw_output_sha256,
                "diagnostics": list(self.diagnostics), "text_lines": list(self.text_lines),
                "superseded_mask_revisions": self.superseded_mask_revisions,
                "frame_times": {k: self.frame_times[k] for k in sorted(self.frame_times, key=int)},
                "coordinate_convention": "boxes half-open [x0, x1) x [y0, y1) in displayed source pixels; SAM's "
                                         "inclusive x2/y2 were converted by the official parser",
                "records": [r.to_json_dict() for r in self.records]}

    @classmethod
    def from_json_dict(cls, d: dict) -> "SamTrackingResult":
        if d.get("schema") != SCHEMA:
            raise SamTrackingError([f"unsupported schema {d.get('schema')!r}"])
        return cls(video=d["video"], media=d["media"], prompt=d["prompt"], model=d["model"], outcome=d["outcome"],
                   records=tuple(SamObjectFrame.from_json_dict(r) for r in d["records"]),
                   frame_times=d["frame_times"], diagnostics=tuple(d.get("diagnostics", ())),
                   text_lines=tuple(d.get("text_lines", ())),
                   superseded_mask_revisions=d.get("superseded_mask_revisions", 0),
                   raw_output_sha256=d.get("raw_output_sha256"), usage=d.get("usage"),
                   request_sha256=d.get("request_sha256"), provenance=d.get("provenance", {}))


def _conf(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def from_parser_result(result: msp.VideoSegmentationResult, table: fr.FrameTable, *, prompt: dict, video: dict,
                       model: str = "sam-3.1", usage: dict | None = None, request: dict | None = None,
                       provenance: dict | None = None) -> SamTrackingResult:
    """Validate the official parser's result against the source frame table. Raises SamTrackingError."""
    problems: list[str] = []
    if video.get("sha256") != table.source_sha256:
        problems.append("the SAM result's video SHA-256 differs from the frame table's source")
    W, H = table.width, table.height
    by_index = {f.index: f for f in table.frames}
    boxes: dict[tuple[int, str], msp.SegmentationBoxRecord] = {}
    masks: dict[tuple[int, str], msp.SegmentationMaskRecord] = {}
    superseded = 0
    texts: list[str] = []
    for rec in result.records:
        if isinstance(rec, msp.SegmentationTextRecord):
            texts.append(rec.text)
            continue
        if rec.frame is None:
            problems.append(f"record {rec.order}: a video record without a frame index")
            continue
        key = (rec.frame.frame_index, rec.object_id)
        if key[0] not in by_index:
            problems.append(f"frame {key[0]} (object {rec.object_id}) is not in the source frame table "
                            f"(0..{len(table.frames) - 1})")
            continue
        if isinstance(rec, msp.SegmentationBoxRecord):
            if not (0 <= rec.left < rec.right <= W and 0 <= rec.top < rec.bottom <= H):
                problems.append(f"frame {key[0]} object {rec.object_id}: box {rec.left, rec.top, rec.right, rec.bottom}"
                                f" is outside the {W}x{H} displayed frame")
            boxes[key] = rec
        elif isinstance(rec, msp.SegmentationMaskRecord):
            if key in masks:
                superseded += 1
            masks[key] = rec
            if rec.mask.width > W or rec.mask.height > H:
                problems.append(f"frame {key[0]} object {rec.object_id}: mask raster {rec.mask.width}x{rec.mask.height}"
                                f" exceeds the {W}x{H} frame")
    # documented token check: every box token carries w=<W>;h=<H> of the displayed media
    raw = result.raw_output
    n_box_tokens = raw.count("<|box;")
    n_size_ok = raw.count(f";w={W};h={H}|>") + raw.count(f";w={W};h={H};c=")
    if n_box_tokens and n_size_ok < len(boxes):
        problems.append(f"SAM box tokens do not report the displayed size w={W};h={H} "
                        f"(rotation or resolution mismatch): {n_size_ok} of {n_box_tokens} box tokens match")
    if problems:
        raise SamTrackingError(problems)
    out = []
    for key in sorted(set(boxes) | set(masks), key=lambda k: (k[0], len(k[1]), k[1])):
        b, m = boxes.get(key), masks.get(key)
        if b is None:  # the grammar pairs every mask with a box; a mask alone takes its bounds from the parser
            bx = (int(m.bounds.left), int(m.bounds.top), int(m.bounds.right), int(m.bounds.bottom))
        else:
            bx = (int(b.left), int(b.top), int(b.right), int(b.bottom))
        out.append(SamObjectFrame(key[0], key[1], bx, _conf(b.confidence) if b else None,
                                  m.mask.encoding if m else None, m.mask.payload if m else None,
                                  m.mask.width if m else None, m.mask.height if m else None,
                                  _conf(m.confidence) if m else None, m.revision if m else None))
    times = {}
    for i in sorted({r.frame_index for r in out}):
        f = by_index[i]
        times[str(i)] = {"num": f.t.numerator, "den": f.t.denominator, "pts": f.pts, "timestamp_source": f.timestamp_source}
    outcome = {"status": result.outcome.status, "reason": getattr(result.outcome, "reason", None),
               "detail": getattr(result.outcome, "detail", None)}
    diags = tuple({"severity": d.severity, "code": d.code, "message": d.message, "line": d.line} for d in result.diagnostics)
    return SamTrackingResult(
        video=dict(video), media={"width": W, "height": H, "rotation_deg": table.rotation_deg,
                                  "frame_count": len(table.frames), "time_base": str(table.time_base)},
        prompt=dict(prompt), model=model, outcome=outcome, records=tuple(out), frame_times=times, diagnostics=diags,
        text_lines=tuple(texts), superseded_mask_revisions=superseded,
        raw_output_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(), usage=usage,
        request_sha256=canonical_sha256(request) if request is not None else None, provenance=dict(provenance or {}))
