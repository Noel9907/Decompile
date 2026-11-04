"""Tests for decompile.tracker (P5 classical tracker). OFFLINE.

Ground-truth positions appear ONLY here, on test-side, to measure tracking accuracy. Test-only
scenes are used, never the pre-registered experiment scene.
"""

from __future__ import annotations

import ast
import json
import math
import random
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import frames as fr
from decompile import render as rd
from decompile import synth
from decompile import tracker as tk
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
# test-only throw: moves left, rises, falls, bounces once
TRACK_SCENE = synth.SceneSpec("obj_01", 30, 45, 500.0, 100.0, -80.0, -100.0, 0.0, 900.0,
                              planes=(synth.Plane("y", 400.0, "less", 0.7),))
# accuracy acceptance thresholds (px), fixed before measuring
MAX_RMSE_PX = 0.1
MAX_ABS_PX = 0.25


@pytest.fixture(scope="module")
def scene():
    return synth.generate(TRACK_SCENE)


@pytest.fixture(scope="module")
def export(tmp_path_factory, scene):
    d = tmp_path_factory.mktemp("track")
    r = rd.render_video(scene.trajectories, [rd.Boundary(p.axis, p.position_px, p.keep_side) for p in TRACK_SCENE.planes],
                        d / "v.mp4")
    table = fr.extract_frames(r.video_path, d / "frames")
    return d / "frames", table, r


@pytest.fixture(scope="module")
def result(export):
    frames_dir, table, r = export
    return tk.track_extracted(frames_dir, tk.TrackerConfig(), expected_source_sha256=r.video_sha256)


def _rms(v):
    return math.sqrt(sum(a * a for a in v) / len(v))


def test_tracking_accuracy_against_test_side_ground_truth(result, scene):
    assert result.status_counts()[tk.DETECTED] == TRACK_SCENE.n_frames
    traj = result.to_trajectory_set(synthetic=True)
    gt = scene.trajectories.objects[0].observations
    ex = [o.x - g.x for o, g in zip(traj.objects[0].observations, gt)]
    ey = [o.y - g.y for o, g in zip(traj.objects[0].observations, gt)]
    stats = {"x_rmse": _rms(ex), "x_max": max(map(abs, ex)), "y_rmse": _rms(ey), "y_max": max(map(abs, ey))}
    print("tracking accuracy (px):", stats)
    assert stats["x_rmse"] < MAX_RMSE_PX and stats["y_rmse"] < MAX_RMSE_PX
    assert stats["x_max"] < MAX_ABS_PX and stats["y_max"] < MAX_ABS_PX


def test_processing_order_does_not_change_any_detection(export, result):
    frames_dir, table, r = export
    idx = [f.index for f in table.frames]
    shuffled = idx[:]
    random.Random(1234).shuffle(shuffled)
    for order in (idx[::-1], shuffled):
        again = tk.track_extracted(frames_dir, tk.TrackerConfig(), order=order)
        assert again.detections == result.detections


def test_detection_of_a_frame_does_not_depend_on_other_frames(export, result):
    frames_dir, table, _ = export
    for k in (0, 17, len(table.frames) - 1):
        alone = tk.detect(fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=k)), tk.TrackerConfig())
        assert alone == result.detections[k].detection
    first_half = tk.track_frames(((table.frames[i], fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=i)))
                                  for i in range(20)), tk.TrackerConfig())
    assert first_half.detections == result.detections[:20]  # withheld frames change nothing earlier


def test_classical_tracker_provenance_and_exact_time(result, export):
    _, table, r = export
    for synthetic in (True, False):
        traj = result.to_trajectory_set(synthetic=synthetic)
        assert traj.source == "CLASSICAL_TRACKER" and traj.synthetic is synthetic
    traj = result.to_trajectory_set(synthetic=True)
    obs = traj.objects[0].observations
    assert [o.frame_index for o in obs] == [f.index for f in table.frames]
    assert [o.t for o in obs] == [f.t for f in table.frames] and all(isinstance(o.t, Fraction) for o in obs)
    assert result.source_sha256 == r.video_sha256
    with pytest.raises(TypeError):
        result.to_trajectory_set(synthetic=None)


def test_trajectory_serialization_round_trip(result, tmp_path):
    traj = result.to_trajectory_set(synthetic=True)
    path = tmp_path / "tracked.json"
    path.write_text(json.dumps(traj.to_json_dict()), encoding="utf-8")
    assert tr.load_trajectories(path) == traj
    report = result.to_json_dict()
    assert report["tracker_version"] == tk.TRACKER_VERSION and report["config"]["min_chroma"] == 60
    assert len(report["frames"]) == len(traj.objects[0].observations)
    assert set(report["frames"][0]) >= {"frame_index", "t", "status", "x", "y", "area_px", "fill_ratio", "n_candidates"}


S0 = replace(rd.RenderSettings(), blur_sigma_px=0.8, noise_sigma=2.0)


def _image(*balls, w=320, h=240, settings=S0):
    img = rd.render_frame(w, h, -100.0, -100.0, 0, [], settings)  # background only
    for (x, y, r) in balls:
        s = replace(settings, ball_radius_px=r, noise_sigma=0.0)
        ball = rd.render_frame(w, h, x, y, 0, [], s).astype(int)
        bg = rd.render_frame(w, h, -100.0, -100.0, 0, [], s).astype(int)
        img = np.clip(img.astype(int) + (ball - bg), 0, 255).astype(np.uint8)
    return img


def test_missing_ambiguous_border_and_oversized_detections_get_no_coordinate():
    cfg = tk.TrackerConfig()
    cases = {
        tk.MISSING: _image(),
        tk.AMBIGUOUS: _image((80.0, 80.0, 10.0), (200.0, 150.0, 10.0)),
        tk.TOUCHES_BORDER: _image((3.0, 120.0, 10.0)),
        tk.AREA_OUT_OF_RANGE: _image((160.0, 120.0, 90.0)),
    }
    for status, img in cases.items():
        d = tk.detect(img, cfg)
        assert d.status == status, (status, d)
        assert d.x is None and d.y is None
    ok = tk.detect(_image((150.3, 101.7, 10.0)), cfg)
    assert ok.status == tk.DETECTED and abs(ok.x - 150.3) < 0.1 and abs(ok.y - 101.7) < 0.1


def test_failed_frames_are_refused_not_invented(export):
    frames_dir, table, _ = export
    items = [(table.frames[i], fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=i))) for i in range(6)]
    items[3] = (table.frames[3], _image(w=640, h=480))  # the ball vanishes in frame 3
    res = tk.track_frames(items, tk.TrackerConfig())
    assert [d.frame_index for d in res.failures] == [3]
    with pytest.raises(tk.TrackingError, match="frame 3"):
        res.to_trajectory_set(synthetic=True)


def test_input_validation(export):
    frames_dir, table, _ = export
    img = fr.read_png_rgb(frames_dir / fr.PNG_PATTERN.format(index=0))
    with pytest.raises(ValueError, match="twice"):
        tk.track_frames([(table.frames[0], img), (table.frames[0], img)], tk.TrackerConfig())
    with pytest.raises(ValueError):
        tk.detect(img.astype(np.float32), tk.TrackerConfig())
    with pytest.raises(ValueError, match="different video"):
        tk.track_extracted(frames_dir, tk.TrackerConfig(), expected_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="exactly once"):
        tk.track_extracted(frames_dir, tk.TrackerConfig(), order=[0, 1])
    for bad in ({"min_area_px": 0}, {"weight_floor_chroma": 80.0}, {"object_id": "Ball"}, {"hue_tolerance_deg": 200.0}):
        with pytest.raises(ValueError):
            tk.TrackerConfig(**bad)


def test_tracker_is_independent_of_generator_renderer_physics_and_vault():
    tree = ast.parse((ROOT / "decompile" / "tracker.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
               {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                for a in n.names} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in imported if m.startswith("decompile")} <= \
        {"decompile", "decompile.frames", "decompile.trajectories", "decompile.ball_measure"}  # P7: outline centre
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("synth", "render", "scene_program", "kinematic_fit", "holdout", "evaluate", "vault",
                      "ground_truth", "GroundTruth", "simulate"):
        assert forbidden not in names


# ------------------------------------------------------------------ P6 options (default-off)


def test_default_config_serialises_exactly_as_in_p5():
    assert tk.TrackerConfig().to_json_dict() == {
        "object_id": "obj_01", "hue_center_deg": 15.0, "hue_tolerance_deg": 25.0, "min_chroma": 60,
        "weight_floor_chroma": 20.0, "min_area_px": 30, "max_area_px": 20_000, "edge_margin_px": 2}
    d = tk.detect(_image((150.3, 101.7, 10.0)), tk.TrackerConfig()).to_json_dict()
    assert "n_mask_px" not in d and "largest_small_component_px" not in d


def test_roi_and_exclusion_boxes():
    img = _image((80.0, 80.0, 10.0), (220.0, 150.0, 10.0))  # two balls: ambiguous by default
    assert tk.detect(img, tk.TrackerConfig()).status == tk.AMBIGUOUS
    excl = tk.detect(img, tk.TrackerConfig(exclude_boxes=((190, 120, 250, 180),)))
    roi = tk.detect(img, tk.TrackerConfig(roi=(0, 0, 160, 240)))
    for d in (excl, roi):
        assert d.status == tk.DETECTED and abs(d.x - 80.0) < 0.15 and abs(d.y - 80.0) < 0.15
    # a ball cut by the ROI edge would have a biased centroid: refused, no coordinate
    cut = tk.detect(_image((158.0, 120.0, 10.0)), tk.TrackerConfig(roi=(0, 0, 160, 240)))
    assert cut.status == tk.TOUCHES_BORDER and cut.x is None
    for bad in ({"roi": (10, 10, 5, 20)}, {"roi": (0, 0, 1)}, {"exclude_boxes": ((0, 0, 1.5, 2),)},
                {"hue_weighting": 1}):
        with pytest.raises(ValueError):
            tk.TrackerConfig(**bad)


def test_extended_diagnostics_separate_absence_from_rejection():
    cfg = tk.TrackerConfig(extended_diagnostics=True, min_area_px=200)
    empty = tk.detect(_image(), cfg)
    small = tk.detect(_image((150.0, 100.0, 5.0)), cfg)  # target-coloured but below min_area_px
    assert empty.status == small.status == tk.MISSING
    assert empty.n_mask_px == 0 and small.n_mask_px > 0 and small.largest_small_component_px > 0


def test_seeded_config_tracks_a_non_orange_ball():
    s = replace(S0, ball_rgb=(40, 200, 70))  # green
    img = _image((120.4, 90.6, 12.0), settings=s)
    cfg, derivation = tk.config_from_seed(img, (106, 76, 136, 106))
    assert 100 < cfg.hue_center_deg < 150 and cfg.hue_weighting and derivation["seed_box"] == [106, 76, 136, 106]
    d = tk.detect(_image((200.2, 150.8, 12.0), settings=s), cfg)
    assert d.status == tk.DETECTED and abs(d.x - 200.2) < 0.15 and abs(d.y - 150.8) < 0.15
    assert tk.detect(_image((200.2, 150.8, 12.0)), cfg).status == tk.MISSING  # an orange ball is not the target
    with pytest.raises(tk.SeedError):
        tk.config_from_seed(img, (10, 10, 40, 40))  # box on plain background


def test_frame_subsets(export, result):
    frames_dir, table, _ = export
    sub = tk.track_extracted(frames_dir, tk.TrackerConfig(), frames=[40, 0, 20])
    assert [d.frame_index for d in sub.detections] == [0, 20, 40]
    assert [d.detection for d in sub.detections] == [result.detections[i].detection for i in (0, 20, 40)]
    for bad in ([], [0, 0], [999]):
        with pytest.raises(ValueError):
            tk.track_extracted(frames_dir, tk.TrackerConfig(), frames=bad)
    with pytest.raises(ValueError, match="either"):
        tk.track_extracted(frames_dir, tk.TrackerConfig(), frames=[0], order=[0])
