"""Tests for decompile.render (P5 SYNTHETIC VIDEO renderer). OFFLINE.

Only test-only scenes are rendered (never the pre-registered experiment scene), into pytest
temporary directories.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from decompile import frames as fr
from decompile import ingest
from decompile import render as rd
from decompile import synth
from decompile import trajectories as tr

ROOT = Path(__file__).resolve().parent.parent
ALT = synth.SceneSpec("obj_01", 30, 40, 120.0, 80.0, 70.0, 0.0, 0.0, 1100.0, planes=(synth.Plane("y", 420.0, "less", 0.6),))


def boundaries(spec):
    return [rd.Boundary(p.axis, p.position_px, p.keep_side) for p in spec.planes]


@pytest.fixture(scope="module")
def scene():
    return synth.generate(ALT)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, scene):
    d = tmp_path_factory.mktemp("render")
    return rd.render_video(scene.trajectories, boundaries(ALT), d / "v.mp4")


def test_rendering_is_deterministic(tmp_path, scene, rendered):
    again = rd.render_video(scene.trajectories, boundaries(ALT), tmp_path / "again.mp4")
    assert again.video_sha256 == rendered.video_sha256
    assert (tmp_path / "again.mp4").read_bytes() == rendered.video_path.read_bytes()
    other = rd.render_video(scene.trajectories, boundaries(ALT), tmp_path / "seed.mp4", replace(rd.RenderSettings(), seed=7))
    assert other.video_sha256 != rendered.video_sha256


def test_valid_decodable_mp4_with_correct_size_and_count(rendered):
    info = ingest.probe_video(rendered.video_path)
    assert ingest.validate(info) == []
    assert (info.width, info.height) == (640, 480)
    assert info.decoded_frame_count == ALT.n_frames and info.frame_count_status == "consistent"
    assert info.nominal_fps == "30/1" and info.codec == "h264" and not info.has_audio


def test_frame_extraction_preserves_generator_timestamps_and_order(tmp_path, scene, rendered):
    table = fr.extract_frames(rendered.video_path, tmp_path / "frames")
    obs = scene.trajectories.objects[0].observations
    assert [f.index for f in table.frames] == [o.frame_index for o in obs]
    assert [f.t for f in table.frames] == [o.t for o in obs]  # exact Fractions
    assert all(f.t == Fraction(k, ALT.fps) for k, f in enumerate(table.frames))
    assert table.synthesized_timestamp_count == 0 and (table.width, table.height) == (640, 480)


def test_container_carries_no_scene_metadata(rendered):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json",
                          str(rendered.video_path)], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    assert set(data["format"].get("tags", {})) <= {"major_brand", "minor_version", "compatible_brands"}
    assert set(data["streams"][0].get("tags", {})) <= {"language", "handler_name", "vendor_id", "encoder"}


def test_manifest_records_settings_and_provenance_without_ground_truth(rendered, scene):
    m = rendered.manifest
    assert m["label"] == "SYNTHETIC VIDEO" and m["synthetic"] is True
    assert m["video"]["sha256"] == rendered.video_sha256
    assert m["video"]["width"] == 640 and m["video"]["n_frames"] == ALT.n_frames and m["video"]["frame_rate"] == "30/1"
    assert m["settings"]["seed"] == rd.RenderSettings().seed and m["settings"]["blur_sigma_px"] > 0
    from decompile.holdout import sha256_canonical
    assert m["source"]["trajectory_sha256"] == sha256_canonical(scene.trajectories.to_json_dict())
    text = json.dumps(m)
    for forbidden in ("contacts", "restitution", "ay", "vy0", "t_s", "ground_truth"):
        assert f'"{forbidden}"' not in text


def test_frame_content_ball_surface_antialiasing_and_noise():
    s = replace(rd.RenderSettings(), blur_sigma_px=0.0, noise_sigma=0.0)
    b = [rd.Boundary("y", 300.0, "less")]
    img = rd.render_frame(200, 360, 100.0, 150.0, 0, b, s)
    assert tuple(img[150, 100]) == s.ball_rgb  # ball centre
    assert tuple(img[10, 10]) == s.background_rgb
    assert tuple(img[340, 10]) == s.surface_rgb  # surface drawn below position + radius = 312
    assert tuple(img[311, 10]) == s.background_rgb and tuple(img[312, 10]) == s.surface_rgb
    off = rd.render_frame(200, 360, 100.37, 150.21, 0, b, s)[130:171, 80:121, 0].astype(int)
    lo, hi = sorted((s.ball_rgb[0], s.background_rgb[0]))
    partial = int(((off > lo) & (off < hi)).sum())
    assert partial >= 20, "antialiased edge pixels expected around the disc"
    noisy = rd.RenderSettings()
    a = rd.render_frame(200, 360, 100.0, 150.0, 3, b, noisy)
    assert np.array_equal(a, rd.render_frame(200, 360, 100.0, 150.0, 3, b, noisy))
    assert not np.array_equal(a, rd.render_frame(200, 360, 100.0, 150.0, 4, b, noisy))  # noise keyed by frame index


def _with(data_edit):
    data = synth.generate(ALT).trajectories.to_json_dict()
    data_edit(data)
    return tr.TrajectorySet.from_json_dict(data)


def test_refusals(tmp_path, scene, rendered):
    with pytest.raises(FileExistsError):
        rd.render_video(scene.trajectories, boundaries(ALT), rendered.video_path)
    fixture = _with(lambda d: d.update(source="SYNTHETIC_ANALYTIC_FIXTURE"))
    with pytest.raises(rd.RenderError, match="SYNTHETIC_NUMERICAL_GENERATOR"):
        rd.render_video(fixture, [], tmp_path / "a.mp4")
    tracked = _with(lambda d: d.update(source="CLASSICAL_TRACKER"))
    with pytest.raises(rd.RenderError):
        rd.render_video(tracked, [], tmp_path / "b.mp4")
    gap = _with(lambda d: d["objects"][0].update(observations=d["objects"][0]["observations"][::2]))
    with pytest.raises(rd.RenderError, match="consecutive"):
        rd.render_video(gap, [], tmp_path / "c.mp4")
    outside = _with(lambda d: d["objects"][0]["observations"][5].update(x=3.0))
    with pytest.raises(rd.RenderError, match="not fully inside"):
        rd.render_video(outside, [], tmp_path / "d.mp4")
    with pytest.raises(rd.RenderError, match=".mp4"):
        rd.render_video(scene.trajectories, [], tmp_path / "e.mkv")
    assert not any(p.suffix == ".mp4" for p in tmp_path.iterdir())  # nothing left behind


def test_renderer_never_uses_the_scene_program_simulator():
    tree = ast.parse((ROOT / "decompile" / "render.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    for forbidden in ("decompile.scene_program", "decompile.kinematic_fit", "decompile.predict", "decompile.tracker",
                      "decompile.evaluate"):
        assert forbidden not in imported
