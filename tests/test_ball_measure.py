"""P7: outline-based ball-centre measurement (decompile.ball_measure, tracker centre_method) on the
independent synthetic stress renderer (decompile.stress_render; exact centres known)."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from decompile import ball_measure as bm
from decompile import stress_render as sr
from decompile import tracker as tk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import measurement_stress as ms  # noqa: E402


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ({n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})


def test_stress_renderer_is_independent_of_every_measurement_and_physics_module():
    mods = _imports(ROOT / "decompile" / "stress_render.py")
    assert {m for m in mods if m.startswith("decompile")} == set()
    assert mods <= {"__future__", "math", "dataclasses", "fractions", "numpy"}
    for name in ("tracker", "ball_measure", "kinematic_fit", "projective_fit", "projective_program", "predict",
                 "evaluate", "holdout", "scene_program"):
        assert not {m for m in _imports(ROOT / "decompile" / f"{name}.py") if "stress_render" in m}, name
        tree = ast.parse((ROOT / "decompile" / f"{name}.py").read_text(encoding="utf-8"))
        assert "stress_render" not in {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}


def test_ball_measure_sees_one_image_only():
    mods = _imports(ROOT / "decompile" / "ball_measure.py")
    assert {m for m in mods if m.startswith("decompile")} == set()
    names = {n.id for n in ast.walk(ast.parse((ROOT / "decompile" / "ball_measure.py").read_text(encoding="utf-8")))
             if isinstance(n, ast.Name)}
    for forbidden in ("truth", "vault", "schedule", "frames", "Fraction"):
        assert forbidden not in names


def _scene(name: str, n: int = 24) -> sr.StressVideo:
    return sr.render([s for s in sr.stress_suite(n) if s.name == name][0])


def _cfg(video: sr.StressVideo, method: str) -> tk.TrackerConfig:
    base = ms.seed_config(video)
    kw = {k: getattr(base, k) for k in ("object_id", "hue_center_deg", "hue_tolerance_deg", "min_chroma",
                                        "weight_floor_chroma", "min_area_px", "max_area_px", "edge_margin_px")}
    return tk.TrackerConfig(**kw, hue_weighting=True, centre_method=method)


def _errors(video, method, frames=None):
    cfg = _cfg(video, method)
    out = []
    for i, (img, t) in enumerate(zip(video.frames, video.truth)):
        if frames is not None and i not in frames:
            continue
        d = tk.detect(img, cfg)
        out.append((t, d))
    return out


@pytest.fixture(scope="module")
def seam():
    return _scene("seam_highlights")


def test_outline_centre_removes_the_shading_and_highlight_bias_of_the_colour_centroid(seam):
    col = _errors(seam, "colour_centroid")
    edge = _errors(seam, "edge_ellipse")
    ce = [math.hypot(d.x - t.x, d.y - t.y) for t, d in col]
    ee = [math.hypot(d.x - t.x, d.y - t.y) for t, d in edge if d.status == tk.DETECTED]
    assert len(ee) == len(edge)
    assert np.mean(ce) > 1.0, "the stress scene must expose the colour-centroid bias"
    assert np.sqrt(np.mean(np.square(ee))) < 0.15 and max(ee) < 0.4


def test_background_boundary_does_not_bias_the_outline_centre():
    v = _scene("two_tone_boundary", 36)
    straddle = [i for i, t in enumerate(v.truth) if t.background_at_centre == "straddle"]
    assert len(straddle) >= 3
    for t, d in _errors(v, "edge_ellipse", set(straddle)):
        assert d.status == tk.DETECTED and math.hypot(d.x - t.x, d.y - t.y) < 0.06


def test_occluded_outlines_are_rejected_not_silently_biased():
    v = _scene("partial_occlusion", 36)
    res = _errors(v, "edge_ellipse")
    rejected = [t for t, d in res if d.status == tk.EDGE_REJECTED]
    assert rejected and all(t.occluded_fraction > 0 for t in rejected)
    for t, d in res:
        if d.status == tk.DETECTED:
            assert math.hypot(d.x - t.x, d.y - t.y) < 0.1
        if d.status == tk.EDGE_REJECTED:
            assert d.x is None and d.edge["rejection_reason"] and d.x_colour is not None


def test_per_frame_uncertainty_is_not_overconfident_on_the_realistic_scene():
    v = _scene("realistic_combined", 36)
    res = [(t, d) for t, d in _errors(v, "edge_ellipse") if d.status == tk.DETECTED]
    assert len(res) >= 28
    z = np.array([((d.x - t.x) / d.sigma_x, (d.y - t.y) / d.sigma_y) for t, d in res])
    assert (np.abs(z) <= 2).all(axis=1).mean() >= 0.9
    assert np.sqrt(np.mean(z ** 2)) <= 1.2  # calibrated or conservative
    assert np.median([d.sigma_x for _, d in res]) < 1.5  # but not uselessly large


def test_diameter_tracks_the_true_apparent_size():
    v = _scene("receding_size", 30)
    res = [(t, d) for t, d in _errors(v, "edge_ellipse") if d.status == tk.DETECTED]
    rel = [(d.diameter_px - 2 * t.radius_px) / (2 * t.radius_px) for t, d in res]
    assert abs(np.mean(rel)) < 0.01 and max(abs(r) for r in rel) < 0.03


def test_measure_ball_validates_inputs_and_is_deterministic(seam):
    img = seam.frames[3]
    t = seam.truth[3]
    a = bm.measure_ball(img, t.x + 1.0, t.y - 1.0, t.radius_px)
    b = bm.measure_ball(img.copy(), t.x + 1.0, t.y - 1.0, t.radius_px)
    assert a == b and a.status == bm.MEASURED and math.hypot(a.x - t.x, a.y - t.y) < 0.3
    with pytest.raises(ValueError):
        bm.measure_ball(img.astype(float), t.x, t.y, t.radius_px)
    with pytest.raises(ValueError):
        bm.measure_ball(img, t.x, t.y, 0.5)
    with pytest.raises(ValueError):
        bm.EdgeParams(model="square")


def test_edge_config_round_trips_and_the_default_config_is_unchanged():
    assert "centre_method" not in tk.TrackerConfig().to_json_dict()
    cfg = tk.TrackerConfig(centre_method="edge_ellipse")
    d = cfg.to_json_dict()
    assert d["centre_method"] == "edge_ellipse" and d["edge_method_version"] == bm.METHOD_VERSION
    assert tk.TrackerConfig.from_json_dict(d) == tk.TrackerConfig(centre_method="edge_ellipse",
                                                                  edge_params=bm.EdgeParams(model="ellipse"))
    with pytest.raises(ValueError):
        tk.TrackerConfig(centre_method="sam")
    with pytest.raises(ValueError):
        tk.TrackerConfig(edge_params=bm.EdgeParams())


def test_edge_rejection_is_counted_only_for_edge_configs(seam):
    from decompile import frames as fr
    from fractions import Fraction
    items = [(fr.FrameTimestamp(i, i, i, "frame_pts", Fraction(i, 30), 1, 1, False), seam.frames[i]) for i in range(3)]
    col = tk.track_frames(items, _cfg(seam, "colour_centroid"))
    edge = tk.track_frames(items, _cfg(seam, "edge_ellipse"))
    assert tk.EDGE_REJECTED not in col.status_counts() and tk.EDGE_REJECTED in edge.status_counts()
    traj = edge.to_trajectory_set(synthetic=True)
    o = traj.objects[0].observations[0]
    assert o.sigma_x > 0 and o.diameter_px > 0
    assert "sigma_x" not in col.to_trajectory_set(synthetic=True).objects[0].observations[0].to_json_dict()
