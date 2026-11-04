"""TEST-ONLY builders for SAM 3.1 wire output and Responses stream events (no network).

Masks are encoded with the official package's own one_bit encoder (meta_sam_parser._mask_codec._encode_raster,
private, used here only to build fixtures); the official decoder verifies the payload canonically. Box tokens
follow the documented grammar: inclusive x2/y2, w/h = displayed media size.
"""

from __future__ import annotations

import json
import math
import socket
from fractions import Fraction

import numpy as np
from meta_sam_parser._mask_codec import _encode_raster

from decompile import frames as fr

LOOPBACK = ("127.0.0.1", "::1", "localhost")


def forbid_network(monkeypatch) -> None:
    """Refuse every connection except loopback (asyncio's Windows event loop builds a loopback socketpair)."""
    real_connect = socket.socket.connect

    def guarded(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            raise AssertionError(f"network access attempted: {address!r}")
        return real_connect(self, address, *a, **k)

    def refuse(*a, **k):
        raise AssertionError("network access attempted (create_connection)")
    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "create_connection", refuse)


LANE = {"item_id": "msg_test", "output_index": 0, "content_index": 0}


def encode_mask(raster: np.ndarray) -> str:
    r = np.asarray(raster, np.uint8)
    return _encode_raster(r.ravel().tolist(), r.shape[1], r.shape[0])


def disc_raster(box, cx, cy, radius) -> np.ndarray:
    """Binary disc (pixel centres at +0.5) restricted to the half-open box."""
    x0, y0, x1, y1 = box
    jj, ii = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    return ((jj - cx) ** 2 + (ii - cy) ** 2 <= radius * radius).astype(np.uint8)


def disc_record(oid, cx, cy, radius, size, *, conf=None, pad=2):
    """One object record (box token + mask token) for a disc; the box is the disc's tight half-open bounds."""
    W, H = size
    x0, y0 = max(0, math.floor(cx - radius) - pad), max(0, math.floor(cy - radius) - pad)
    x1, y1 = min(W, math.ceil(cx + radius) + pad), min(H, math.ceil(cy + radius) + pad)
    r = disc_raster((x0, y0, x1, y1), cx, cy, radius)
    nz_i, nz_j = np.nonzero(r)
    bx0, by0, bx1, by1 = x0 + int(nz_j.min()), y0 + int(nz_i.min()), x0 + int(nz_j.max()) + 1, y0 + int(nz_i.max()) + 1
    r = r[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
    return record(oid, (bx0, by0, bx1, by1), r, size, conf=conf)


def record(oid, box, raster, size, *, conf=None) -> str:
    W, H = size
    x0, y0, x1, y1 = box
    c = f";c={conf}" if conf is not None else ""
    rr = np.asarray(raster, np.uint8)
    return (f"{oid}<|box;x1={x0};y1={y0};x2={x1 - 1};y2={y1 - 1};w={W};h={H}{c}|>"
            f"<|mask;x=0;y=0{c};data={rr.shape[0]},{rr.shape[1]},{encode_mask(rr)}|>")


def line(frame: int, records: list[str]) -> str:
    return f"<{frame}f>" + ",".join(records) + "\n"


def events(lines: list[str], *, terminal: str = "completed", usage: dict | None = None, done: bool = True) -> list[dict]:
    text = "".join(lines)
    evs: list[dict] = [{"type": "response.created", "response": {"id": "resp_test", "status": "in_progress"}}]
    evs += [{"type": "response.output_text.delta", "delta": ln, **LANE} for ln in lines]
    if terminal == "refusal":
        return evs + [{"type": "response.refusal.done", "refusal": "cannot help", **LANE}]
    if done:
        evs.append({"type": "response.output_text.done", "text": text, **LANE})
    if terminal == "completed":
        evs.append({"type": "response.completed",
                    "response": {"id": "resp_test", "status": "completed", "usage": usage or {"video_frames_processed": 50}}})
    elif terminal == "incomplete":
        evs.append({"type": "response.incomplete",
                    "response": {"id": "resp_test", "status": "incomplete", "incomplete_details": {"reason": "max_frames"}}})
    elif terminal == "failed":
        evs.append({"type": "response.failed", "response": {"id": "resp_test", "error": {"message": "internal failure"}}})
    elif terminal == "error":
        evs.append({"type": "error", "error": {"message": "stream error"}})
    return evs


def sse_bytes(evs: list[dict], *, done_marker: bool = True) -> bytes:
    out = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs)
    if done_marker:
        out += "data: [DONE]\n\n"
    return out.encode("utf-8")


def frame_table(n: int, width: int, height: int, *, sha: str = "a" * 64, rotation: int = 0,
                time_base: Fraction = Fraction(1, 90000), pts_steps=None) -> fr.FrameTable:
    """A frame table with (optionally irregular) PTS; t = (pts - first) * time_base like decompile.frames."""
    steps = pts_steps or [3000] * n
    pts, p = [], 0
    for k in range(n):
        pts.append(p)
        p += steps[k % len(steps)]
    frames = [fr.FrameTimestamp(index=i, pts=pts[i], frame_pts=pts[i], timestamp_source=fr.TS_FRAME_PTS,
                                t=(pts[i] - pts[0]) * time_base, duration_reported=None,
                                duration_from_next_pts=None, key_frame=i == 0) for i in range(n)]
    return fr.FrameTable("synthetic.mp4", sha, time_base, pts[0], width, height, rotation, frames)
