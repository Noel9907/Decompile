"""Independent synthetic STRESS renderer for ball-centre measurement (GENERATOR SIDE ONLY).

Purpose: controlled scenes with EXACTLY known ball centres, used to measure the bias, RMSE and
uncertainty calibration of measurement methods (decompile.tracker, decompile.ball_measure).
This module must stay independent of every measurement, inference and physics module: it
imports only numpy (an AST test enforces this), and no measurement module may import it.

Scene model
-----------
* A sphere of radius R (metres) moves in CAMERA coordinates (X right, Y down, Z forward) under a
  constant gravity vector g_cam (any direction: a pitched camera is a rotated gravity vector)
  and bounces on a floor plane n.P = h (n = unit normal pointing from the floor towards the
  ball side) with restitution e: v <- v - (1 + e)(v.n) n. Motion is exact and piecewise
  analytic (contact times solved in closed form).
* Pinhole camera: x = cx + f X / Z, y = cy + f Y / Z. The ground-truth image centre is the
  projection of the 3D sphere centre at the frame timestamp t (mid-exposure). The sphere is
  drawn as a disc of radius f R / Z around that point; the true image of a sphere is a slightly
  larger, off-axis ellipse whose centre differs from the projected centre by O((R/Z)^2 * offset),
  below 0.01 px for every scene used here. This is documented, not hidden.
* Ball appearance: Lambert + Phong shading of the sphere normal under a fixed light, a
  great-circle SEAM band (different colour) on the body rotating with a fixed spin, and a
  specular highlight. Shading makes the ball brighter on one side: a measurement method that
  weights pixels by colour/brightness can be biased by it.
* Background: two regions split by an image line y = y0 + slope * (x - cx) (e.g. a wall above a
  floor), each a base colour with smooth seeded texture.
* Motion blur: the frame is the AVERAGE of sub-frames rendered at equally spaced instants across
  the exposure [t - E/2, t + E/2] (E = exposure_s), each with analytic antialiasing.
* Variable illumination: a per-frame global gain (seeded flicker) and a horizontal gain gradient,
  applied to the whole image (ball and background), before noise.
* Partial occlusion: optional opaque rectangles drawn in FRONT of the ball for chosen frames.
* Gaussian pixel noise (seeded per frame by [seed, frame_index]).

Pixel convention (shared with decompile.render / decompile.tracker): pixel (row i, col j) has
its centre at (j + 0.5, i + 0.5); origin top-left; +y down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

STRESS_RENDER_VERSION = "decompile.stress_render/1"


# ------------------------------------------------------------------ scene description


@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 360
    f_px: float = 560.0
    cx: float = 320.0
    cy: float = 180.0


@dataclass(frozen=True)
class Motion:
    """Sphere motion in camera coordinates (metres, seconds)."""
    p0: tuple[float, float, float]  # centre at t = 0
    v0: tuple[float, float, float]
    g: tuple[float, float, float] = (0.0, 9.81, 0.0)  # camera frame; +Y is image-down
    floor_normal: tuple[float, float, float] | None = None  # unit, pointing towards the ball side
    floor_offset: float = 0.0  # floor plane: n . P = floor_offset
    restitution: float = 0.7


@dataclass(frozen=True)
class Ball:
    radius_m: float = 0.036
    base_rgb: tuple[float, float, float] = (120.0, 22.0, 26.0)  # dark red leather
    seam_rgb: tuple[float, float, float] = (225.0, 205.0, 170.0)  # cream seam
    seam_halfwidth: float = 0.09  # half-width of the seam band on the unit sphere (0 = no seam)
    spin_axis: tuple[float, float, float] = (0.3, 0.2, 0.93)
    spin_rad_s: float = 25.0
    ambient: float = 0.45
    diffuse: float = 0.75
    specular: float = 0.0  # peak specular added (0..1 of white)
    shininess: float = 30.0
    light_dir: tuple[float, float, float] = (-0.4, -0.6, -0.7)  # direction TOWARDS the light


@dataclass(frozen=True)
class Background:
    top_rgb: tuple[float, float, float] = (192.0, 174.0, 176.0)  # pink wall (chroma ~18, like the real wall)
    bottom_rgb: tuple[float, float, float] = (168.0, 176.0, 160.0)  # grey-green floor
    boundary_y: float | None = 215.0  # image y of the boundary at x = cx; None = uniform top colour
    boundary_slope: float = 0.0
    texture_amp: float = 4.0  # smooth texture amplitude (grey levels)
    texture_seed: int = 7


@dataclass(frozen=True)
class Occluder:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 (image coordinates)
    frames: tuple[int, ...]
    rgb: tuple[float, float, float] = (60.0, 60.0, 64.0)


@dataclass(frozen=True)
class StressScene:
    name: str
    camera: Camera
    motion: Motion
    ball: Ball = Ball()
    background: Background = Background()
    fps: int = 30
    n_frames: int = 40
    exposure_s: float = 0.0  # 0 = no motion blur
    noise_sigma: float = 2.0
    gain_flicker: float = 0.0  # std of the per-frame global gain (e.g. 0.05)
    gain_gradient: float = 0.0  # gain varies linearly from 1 - g/2 (left) to 1 + g/2 (right)
    occluders: tuple[Occluder, ...] = ()
    seed: int = 20261104


@dataclass(frozen=True)
class FrameTruth:
    frame_index: int
    t: Fraction
    x: float  # exact projected centre (px)
    y: float
    radius_px: float  # f R / Z at t
    depth_m: float
    image_speed_px_s: float
    blur_px: float  # image-space path length during the exposure
    background_at_centre: str  # "top" | "bottom" | "straddle" (boundary within radius + blur)
    occluded_fraction: float  # fraction of the (sharp) disc covered by occluders


@dataclass(frozen=True)
class StressVideo:
    scene: StressScene
    frames: tuple[np.ndarray, ...]  # RGB uint8
    truth: tuple[FrameTruth, ...]
    contacts_s: tuple[float, ...]
    version: str = field(default=STRESS_RENDER_VERSION)


# ------------------------------------------------------------------ exact motion


def _unit(v) -> np.ndarray:
    a = np.asarray(v, float)
    return a / np.linalg.norm(a)


def contact_times(m: Motion, radius_m: float, t_end: float, max_events: int = 50) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Segments: list of (t_start, p_start, v_start). Contacts solved in closed form."""
    p, v, g = np.array(m.p0, float), np.array(m.v0, float), np.array(m.g, float)
    segs = [(0.0, p, v)]
    if m.floor_normal is None:
        return segs
    n = _unit(m.floor_normal)
    t0 = 0.0
    for _ in range(max_events):
        # height above contact: s(tau) = n.(p + v tau + g tau^2 / 2) - (h + R)
        c = float(n @ p) - (m.floor_offset + radius_m)
        b = float(n @ v)
        a = 0.5 * float(n @ g)
        roots = []
        if abs(a) < 1e-15:
            if b < 0:
                roots.append(-c / b)
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                roots += [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        roots = [r for r in roots if r > 1e-9 and float(n @ (v + g * r)) < 0]
        if not roots:
            break
        tau = min(roots)
        if t0 + tau > t_end:
            break
        p = p + v * tau + 0.5 * g * tau * tau
        vin = v + g * tau
        v = vin - (1.0 + m.restitution) * float(n @ vin) * n
        t0 += tau
        segs.append((t0, p, v))
    return segs


def state_at(segs, g, t: float) -> tuple[np.ndarray, np.ndarray]:
    t_s, p, v = segs[0]
    for s in segs:
        if s[0] <= t:
            t_s, p, v = s
    tau = t - t_s
    return p + v * tau + 0.5 * g * tau * tau, v + g * tau


def project(cam: Camera, P: np.ndarray, radius_m: float) -> tuple[float, float, float]:
    X, Y, Z = (float(c) for c in P)
    if Z <= radius_m:
        raise ValueError("ball behind or too close to the camera")
    return cam.cx + cam.f_px * X / Z, cam.cy + cam.f_px * Y / Z, cam.f_px * radius_m / Z


# ------------------------------------------------------------------ appearance


def _smooth_texture(h: int, w: int, amp: float, seed: int) -> np.ndarray:
    if amp <= 0:
        return np.zeros((h, w))
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    for k in range(6):  # sum of random low-frequency sinusoids (deterministic)
        fx, fy = rng.uniform(0.005, 0.05, 2)
        ph = rng.uniform(0, 2 * math.pi)
        out += np.sin(2 * math.pi * (fx * xx + fy * yy) + ph)
    return amp * out / 6.0 * 2.0


def background_image(scene: StressScene) -> np.ndarray:
    cam, bg = scene.camera, scene.background
    h, w = cam.height, cam.width
    yy, xx = np.mgrid[0:h, 0:w].astype(float) + 0.5
    img = np.empty((h, w, 3))
    if bg.boundary_y is None:
        img[:] = bg.top_rgb
    else:
        line = bg.boundary_y + bg.boundary_slope * (xx - cam.cx)
        cov_bottom = np.clip(yy - line + 0.5, 0.0, 1.0)  # antialiased boundary
        for c in range(3):
            img[..., c] = (1 - cov_bottom) * bg.top_rgb[c] + cov_bottom * bg.bottom_rgb[c]
    img += _smooth_texture(h, w, bg.texture_amp, bg.texture_seed)[..., None]
    return img


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _ball_patch(ball: Ball, xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, r: float, t: float):
    """Coverage alpha and RGB colour of the ball over pixel centres (xs, ys)."""
    dx, dy = xs - cx, ys - cy
    d = np.hypot(dx, dy)
    alpha = np.clip(r - d + 0.5, 0.0, 1.0)
    u = np.clip(dx / r, -1, 1)
    v = np.clip(dy / r, -1, 1)
    q = np.clip(1.0 - u * u - v * v, 0.0, 1.0)
    n = np.stack([u, v, -np.sqrt(q)], axis=-1)  # outward normal; camera looks along +Z
    nn = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    L = _unit(ball.light_dir)
    lam = np.clip(nn @ L, 0.0, None)
    col = np.empty(n.shape)
    base = np.asarray(ball.base_rgb, float)
    if ball.seam_halfwidth > 0:
        Rm = _rotation(np.asarray(ball.spin_axis), ball.spin_rad_s * t)
        body = nn @ Rm  # world -> body (row vectors times R == R^T applied)
        seam_normal = np.array([0.0, 0.0, 1.0])
        on_seam = np.abs(body @ seam_normal) < ball.seam_halfwidth
        col[:] = base
        col[on_seam] = ball.seam_rgb
    else:
        col[:] = base
    shade = ball.ambient + ball.diffuse * lam
    col = col * shade[..., None]
    if ball.specular > 0:
        view = np.array([0.0, 0.0, -1.0])
        refl = 2 * lam[..., None] * nn - L
        spec = np.clip(refl @ view, 0, None) ** ball.shininess
        col = col + 255.0 * ball.specular * spec[..., None]
    return alpha, col


# ------------------------------------------------------------------ rendering


def render(scene: StressScene) -> StressVideo:
    cam, ball, m = scene.camera, scene.ball, scene.motion
    g = np.array(m.g, float)
    t_end = (scene.n_frames + 1) / scene.fps
    segs = contact_times(m, ball.radius_m, t_end)
    contacts = tuple(s[0] for s in segs[1:])
    bg = background_image(scene)
    h, w = cam.height, cam.width
    xs_full = np.arange(w, dtype=float) + 0.5
    gain_x = 1.0 + scene.gain_gradient * ((xs_full / w) - 0.5)
    frames, truth = [], []
    for i in range(scene.n_frames):
        t = Fraction(i, scene.fps)
        tf = float(t)
        P, V = state_at(segs, g, tf)
        x, y, r = project(cam, P, ball.radius_m)
        # image velocity (exact derivative of the projection)
        Z = float(P[2])
        vx = cam.f_px * (V[0] * Z - P[0] * V[2]) / (Z * Z)
        vy = cam.f_px * (V[1] * Z - P[1] * V[2]) / (Z * Z)
        speed = math.hypot(vx, vy)
        blur = speed * scene.exposure_s
        n_sub = 1 if scene.exposure_s <= 0 else int(min(96, max(2, math.ceil(blur / 0.35) + 1)))
        subs = [tf] if n_sub == 1 else [tf + scene.exposure_s * (k / (n_sub - 1) - 0.5) for k in range(n_sub)]
        proj = [project(cam, state_at(segs, g, ts)[0], ball.radius_m) + (ts,) for ts in subs]
        margin = 3
        x0 = max(0, int(math.floor(min(p[0] - p[2] for p in proj))) - margin)
        x1 = min(w, int(math.ceil(max(p[0] + p[2] for p in proj))) + margin)
        y0 = max(0, int(math.floor(min(p[1] - p[2] for p in proj))) - margin)
        y1 = min(h, int(math.ceil(max(p[1] + p[2] for p in proj))) + margin)
        img = bg.copy()
        if x1 > x0 and y1 > y0:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(float) + 0.5
            win_bg = bg[y0:y1, x0:x1]
            acc = np.zeros_like(win_bg)
            for (px, py, pr, ts) in proj:
                a, col = _ball_patch(ball, xx, yy, px, py, pr, ts)
                acc += a[..., None] * col + (1 - a[..., None]) * win_bg
            img[y0:y1, x0:x1] = acc / len(proj)
        occ_frac = 0.0
        for oc in scene.occluders:
            if i in oc.frames:
                ox0, oy0, ox1, oy1 = oc.box
                yy, xx = np.mgrid[0:h, 0:w].astype(float) + 0.5
                cov = (np.clip(xx - ox0 + 0.5, 0, 1) * np.clip(ox1 - xx + 0.5, 0, 1)
                       * np.clip(yy - oy0 + 0.5, 0, 1) * np.clip(oy1 - yy + 0.5, 0, 1))
                img = img * (1 - cov[..., None]) + cov[..., None] * np.asarray(oc.rgb, float)
                disc = np.hypot(xx - x, yy - y) <= r
                if disc.any():
                    occ_frac = max(occ_frac, float((cov[disc] > 0.5).mean()))
        rng = np.random.default_rng([scene.seed, i])
        gain = 1.0 + (scene.gain_flicker * rng.standard_normal() if scene.gain_flicker > 0 else 0.0)
        img = img * (gain * gain_x)[None, :, None]
        if scene.noise_sigma > 0:
            img = img + rng.normal(0.0, scene.noise_sigma, img.shape)
        frames.append(np.clip(np.rint(img), 0, 255).astype(np.uint8))
        bgd = scene.background
        if bgd.boundary_y is None:
            label = "top"
        else:
            line = bgd.boundary_y + bgd.boundary_slope * (x - cam.cx)
            label = "straddle" if abs(y - line) <= r + blur / 2 + 1 else ("top" if y < line else "bottom")
        truth.append(FrameTruth(i, t, x, y, r, Z, speed, blur, label, occ_frac))
    return StressVideo(scene, tuple(frames), tuple(truth), contacts)


# ------------------------------------------------------------------ the stress suite


def _cam() -> Camera:
    return Camera()


def _thrown(z0: float = 1.8, vz: float = 0.0, pitch_deg: float = 0.0, floor: bool = True,
            drop: float = 0.62) -> Motion:
    """A ball thrown left-to-right that bounces once or twice on a horizontal floor.

    pitch_deg tilts the camera down: gravity and the floor normal rotate in the camera frame.
    """
    pr = math.radians(pitch_deg)
    # camera pitched down by pr: world down (0, 1, 0) seen in camera coordinates
    g_dir = np.array([0.0, math.cos(pr), -math.sin(pr)])
    g = tuple(9.81 * g_dir)
    n = tuple(-g_dir)  # horizontal floor: normal opposite to gravity
    p0 = np.array([-0.55, -0.25, z0])
    floor_point = p0 + drop * g_dir  # floor `drop` m below the start along gravity
    return Motion(p0=tuple(p0), v0=(1.0, 0.2, vz), g=g,
                  floor_normal=n if floor else None, floor_offset=float(np.dot(n, floor_point)),
                  restitution=0.65)


def stress_suite(n_frames: int = 36) -> tuple[StressScene, ...]:
    """The fixed stress scenes (defined before any method was evaluated on them)."""
    cam = _cam()
    plain_ball = Ball(seam_halfwidth=0.0, specular=0.0, ambient=1.0, diffuse=0.0)
    flat_bg = Background(boundary_y=None, texture_amp=0.0)
    return (
        StressScene("baseline_flat", cam, _thrown(), plain_ball, flat_bg, n_frames=n_frames, noise_sigma=1.0),
        StressScene("two_tone_boundary", cam, _thrown(), plain_ball, Background(boundary_y=215.0),
                    n_frames=n_frames, noise_sigma=1.0),
        StressScene("seam_highlights", cam, _thrown(), Ball(specular=0.55), flat_bg, n_frames=n_frames,
                    noise_sigma=1.0),
        StressScene("motion_blur", cam, _thrown(), plain_ball, flat_bg, n_frames=n_frames, exposure_s=1 / 40,
                    noise_sigma=1.0),
        StressScene("noise", cam, _thrown(), plain_ball, flat_bg, n_frames=n_frames, noise_sigma=6.0),
        StressScene("receding_size", cam, _thrown(z0=1.3, vz=1.4), plain_ball, flat_bg, n_frames=n_frames,
                    noise_sigma=1.0),
        StressScene("partial_occlusion", cam, _thrown(), plain_ball, flat_bg, n_frames=n_frames, noise_sigma=1.0,
                    occluders=(Occluder((300.0, 0.0, 314.0, 360.0), tuple(range(n_frames))),)),
        StressScene("variable_illumination", cam, _thrown(), Ball(seam_halfwidth=0.0), flat_bg, n_frames=n_frames,
                    noise_sigma=1.0, gain_flicker=0.06, gain_gradient=0.3),
        StressScene("perspective_pitched", cam, _thrown(z0=1.5, vz=0.9, pitch_deg=25.0, drop=0.45), plain_ball,
                    Background(boundary_y=None, texture_amp=3.0), n_frames=n_frames, noise_sigma=1.0),
        StressScene("realistic_combined", cam, _thrown(z0=1.4, vz=1.0, pitch_deg=12.0), Ball(specular=0.5),
                    Background(boundary_y=215.0), n_frames=n_frames, exposure_s=1 / 40, noise_sigma=2.5,
                    gain_flicker=0.03, gain_gradient=0.15),
    )
