"""P5: deterministic SYNTHETIC VIDEO renderer (generator side only).

render_video(trajectories, boundaries, out_path, settings) turns the positions of ONE
object from the independent numerical generator (decompile.synth, a TrajectorySet with
source SYNTHETIC_NUMERICAL_GENERATOR) into an H.264 MP4 that A1 (ingest) and A2 (frames)
accept. The SceneProgram simulator is never used: every ball position drawn is a
generator position, frame by frame.

Image model (pixel-area convention shared with decompile.tracker): pixel (row i, col j)
covers [j, j+1) x [i, i+1) in image coordinates, so its centre is (j + 0.5, i + 0.5).
Origin top-left, +x right, +y DOWN.

Per frame, in float RGB:
  1. flat background;
  2. contact surface: a generator plane constrains the ball CENTRE, so the visible surface
     is drawn one ball radius beyond it (the ball touches the surface at contact). Its
     coverage per pixel is the exact area fraction of the half-plane (antialiased);
  3. the ball: a disc of the given radius at the generator position, coverage from
     SUPERSAMPLE x SUPERSAMPLE samples per pixel (antialiased);
  4. a Gaussian blur (scipy.ndimage, sigma in px) of the composite;
  5. additive Gaussian noise from numpy's PCG64 seeded with (seed, frame_index), so each
     frame's noise is fixed by its index alone;
  6. rint and clip to uint8.
Frames are piped as rgb24 to the system ffmpeg (libx264, one thread, bitexact flags, all
container metadata stripped) at the frame rate given by the exact timestamps, which must
be k / fps for frames 0..N-1. A1/A2 then reproduce those timestamps from the container.

What the video and its container carry: pixels and timing only. No physical parameter,
contact time or position is written into the MP4 metadata (`-map_metadata -1`). The
render manifest (settings, seed, source trajectory hash, boundary geometry) is a
generator-side record and must not be given to inference.

Labelling: every output is SYNTHETIC VIDEO.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from decompile import trajectories as tr
from decompile.holdout import sha256_canonical

RENDER_VERSION = "decompile.render/1"
LABEL = "SYNTHETIC VIDEO"
SUPERSAMPLE = 4  # samples per pixel per axis for the disc coverage
PIXEL_CONVENTION = "pixel (row i, col j) covers [j, j+1) x [i, i+1); its centre is (j + 0.5, i + 0.5)"
FFMPEG_TIMEOUT_S = 300


class RenderError(ValueError):
    """Invalid render request or an encoding failure; no video was left behind."""


@dataclass(frozen=True)
class Boundary:
    """A generator plane, as it constrains the ball CENTRE (decompile.synth.Plane geometry)."""
    axis: str  # "x" | "y"
    position_px: float
    keep_side: str  # "less": centre stays at coordinate <= position_px; "greater": >=

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y") or self.keep_side not in ("less", "greater"):
            raise RenderError(f"invalid boundary axis/side {self.axis!r}/{self.keep_side!r}")
        if not math.isfinite(self.position_px):
            raise RenderError("boundary position must be finite")


@dataclass(frozen=True)
class RenderSettings:
    ball_radius_px: float = 12.0
    ball_rgb: tuple[int, int, int] = (225, 85, 35)  # saturated orange
    background_rgb: tuple[int, int, int] = (178, 178, 178)  # neutral grey (no chroma)
    surface_rgb: tuple[int, int, int] = (72, 72, 72)  # dark neutral grey (no chroma)
    blur_sigma_px: float = 0.8
    noise_sigma: float = 2.0  # 8-bit levels, per channel
    seed: int = 20261104
    crf: int = 12
    preset: str = "medium"
    pix_fmt: str = "yuv420p"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ball_rgb", tuple(self.ball_rgb))
        object.__setattr__(self, "background_rgb", tuple(self.background_rgb))
        object.__setattr__(self, "surface_rgb", tuple(self.surface_rgb))
        for name in ("ball_rgb", "background_rgb", "surface_rgb"):
            v = getattr(self, name)
            if len(v) != 3 or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in v):
                raise RenderError(f"{name} must be three ints in 0..255")
        if not (math.isfinite(self.ball_radius_px) and self.ball_radius_px >= 2):
            raise RenderError("ball_radius_px must be >= 2")
        if not (math.isfinite(self.blur_sigma_px) and self.blur_sigma_px >= 0):
            raise RenderError("blur_sigma_px must be >= 0")
        if not (math.isfinite(self.noise_sigma) and self.noise_sigma >= 0):
            raise RenderError("noise_sigma must be >= 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise RenderError("seed must be a non-negative int")
        if isinstance(self.crf, bool) or not isinstance(self.crf, int) or not 0 <= self.crf <= 51:
            raise RenderError("crf must be an int in 0..51")


@dataclass(frozen=True)
class RenderResult:
    video_path: Path
    video_sha256: str
    manifest: dict = field(compare=False)


# ------------------------------------------------------------------ frame drawing


def _surface_coverage(width: int, height: int, b: Boundary, radius: float) -> np.ndarray:
    """Exact per-pixel area fraction of the visible surface half-plane."""
    if b.keep_side == "less":  # centre <= P: surface occupies coordinate >= P + r
        edge = b.position_px + radius
        cov1d = lambda lo: np.clip(lo + 1.0 - edge, 0.0, 1.0)  # noqa: E731  area of [lo, lo+1) beyond edge
    else:  # centre >= P: surface occupies coordinate <= P - r
        edge = b.position_px - radius
        cov1d = lambda lo: np.clip(edge - lo, 0.0, 1.0)  # noqa: E731
    if b.axis == "y":
        return np.repeat(cov1d(np.arange(height, dtype=np.float64))[:, None], width, axis=1)
    return np.repeat(cov1d(np.arange(width, dtype=np.float64))[None, :], height, axis=0)


def _disc_coverage(width: int, height: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Supersampled coverage of the disc, evaluated only in its bounding box."""
    cov = np.zeros((height, width), dtype=np.float64)
    j0, j1 = max(0, math.floor(cx - radius) - 1), min(width, math.ceil(cx + radius) + 2)
    i0, i1 = max(0, math.floor(cy - radius) - 1), min(height, math.ceil(cy + radius) + 2)
    if j0 >= j1 or i0 >= i1:
        return cov
    offs = (np.arange(SUPERSAMPLE, dtype=np.float64) + 0.5) / SUPERSAMPLE
    xs = (np.arange(j0, j1, dtype=np.float64)[:, None] + offs[None, :]).ravel()
    ys = (np.arange(i0, i1, dtype=np.float64)[:, None] + offs[None, :]).ravel()
    inside = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= radius * radius
    block = inside.reshape(i1 - i0, SUPERSAMPLE, j1 - j0, SUPERSAMPLE).mean(axis=(1, 3))
    cov[i0:i1, j0:j1] = block
    return cov


def render_frame(width: int, height: int, cx: float, cy: float, frame_index: int,
                 boundaries: Sequence[Boundary], settings: RenderSettings) -> np.ndarray:
    """One RGB uint8 frame (H, W, 3). Deterministic given its arguments."""
    img = np.empty((height, width, 3), dtype=np.float64)
    img[:] = np.asarray(settings.background_rgb, dtype=np.float64)
    surface = np.asarray(settings.surface_rgb, dtype=np.float64)
    for b in boundaries:
        c = _surface_coverage(width, height, b, settings.ball_radius_px)[..., None]
        img = img * (1.0 - c) + surface * c
    c = _disc_coverage(width, height, cx, cy, settings.ball_radius_px)[..., None]
    img = img * (1.0 - c) + np.asarray(settings.ball_rgb, dtype=np.float64) * c
    if settings.blur_sigma_px > 0:
        img = gaussian_filter(img, sigma=(settings.blur_sigma_px, settings.blur_sigma_px, 0), mode="nearest")
    if settings.noise_sigma > 0:
        rng = np.random.Generator(np.random.PCG64([settings.seed, frame_index]))
        img = img + rng.normal(0.0, settings.noise_sigma, size=img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ validation


def _frame_rate(obj: tr.ObjectTrajectory) -> Fraction:
    obs = obj.observations
    if [o.frame_index for o in obs] != list(range(len(obs))):
        raise RenderError("the trajectory must have consecutive frames 0..N-1 (constant-rate video)")
    if obs[0].t != 0:
        raise RenderError("the first frame must be at t = 0")
    dt = obs[1].t - obs[0].t
    if any(o.t != k * dt for k, o in enumerate(obs)):
        raise RenderError("timestamps must be exactly k / fps (constant frame rate)")
    return 1 / dt


def _check_visible(obj: tr.ObjectTrajectory, width: int, height: int, boundaries: Sequence[Boundary],
                   radius: float) -> None:
    problems = []
    for o in obj.observations:
        if not (radius <= o.x <= width - radius and radius <= o.y <= height - radius):
            problems.append(f"frame {o.frame_index}: ball at ({o.x}, {o.y}) is not fully inside the image")
    for b in boundaries:
        edge = b.position_px + radius if b.keep_side == "less" else b.position_px - radius
        limit = height if b.axis == "y" else width
        if not 0 < edge < limit:
            problems.append(f"the {b.axis} surface edge at {edge} px is outside the image")
    if problems:
        raise RenderError("; ".join(problems[:5]))


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


# ------------------------------------------------------------------ rendering


def render_video(trajectories: tr.TrajectorySet, boundaries: Sequence[Boundary], out_path: str | Path,
                 settings: RenderSettings = RenderSettings()) -> RenderResult:
    """Render the single object of `trajectories` to a new MP4 at `out_path` (never overwritten)."""
    if not isinstance(trajectories, tr.TrajectorySet):
        raise RenderError(f"expected a TrajectorySet, got {type(trajectories).__name__}")
    if trajectories.source != "SYNTHETIC_NUMERICAL_GENERATOR" or not trajectories.synthetic:
        raise RenderError("only SYNTHETIC_NUMERICAL_GENERATOR trajectories (decompile.synth) are rendered")
    if len(trajectories.objects) != 1:
        raise RenderError("exactly one object is supported")
    boundaries = tuple(boundaries)
    if not all(isinstance(b, Boundary) for b in boundaries):
        raise RenderError("boundaries must be Boundary instances")
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        raise RenderError("the output must be an .mp4 file")
    if out_path.exists():
        raise FileExistsError(out_path)
    obj = trajectories.objects[0]
    width, height = trajectories.coordinate_system.image_width, trajectories.coordinate_system.image_height
    if width % 2 or height % 2:
        raise RenderError("yuv420p needs even image dimensions")
    fps = _frame_rate(obj)
    _check_visible(obj, width, height, boundaries, settings.ball_radius_px)

    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
           "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
           "-an", "-map_metadata", "-1", "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
           "-pix_fmt", settings.pix_fmt, "-threads", "1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
           "-f", "mp4"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex}.tmp.mp4")
    try:
        proc = subprocess.Popen(cmd + [str(tmp)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        try:
            for o in obj.observations:
                proc.stdin.write(render_frame(width, height, o.x, o.y, o.frame_index, boundaries, settings).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", "replace")
            code = proc.wait(timeout=FFMPEG_TIMEOUT_S)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            proc.stderr.close()
        if code != 0:
            raise RenderError(f"ffmpeg failed (exit {code}): {stderr.strip()[:500]}")
        if out_path.exists():
            raise FileExistsError(out_path)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "kind": "render_manifest", "render_version": RENDER_VERSION, "label": LABEL, "synthetic": True,
        "audience": "generator side only: never an inference input",
        "video": {"path": str(out_path), "sha256": digest, "container": "mp4", "codec": "h264",
                  "width": width, "height": height, "n_frames": len(obj.observations),
                  "frame_rate": f"{fps.numerator}/{fps.denominator}", "container_metadata": "stripped (-map_metadata -1)"},
        "source": {"trajectory_sha256": sha256_canonical(trajectories.to_json_dict()),
                   "source_label": trajectories.source, "object_id": obj.object_id,
                   "positions": "generator positions, one per frame, in frame order; timestamps k / fps"},
        "settings": {**asdict(settings), "supersample": SUPERSAMPLE, "pixel_convention": PIXEL_CONVENTION,
                     "noise": "numpy PCG64 seeded with [seed, frame_index], Gaussian, per channel, before rounding",
                     "blur": "scipy.ndimage.gaussian_filter over x and y, mode 'nearest'"},
        "boundaries": [{**asdict(b), "drawn_surface_edge_px": (b.position_px + settings.ball_radius_px
                                                               if b.keep_side == "less"
                                                               else b.position_px - settings.ball_radius_px)}
                       for b in boundaries],
        "encoder": {"command": cmd + ["<output>"], "ffmpeg": ffmpeg_version()},
    }
    return RenderResult(out_path, digest, manifest)
