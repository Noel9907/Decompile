"""Tests for A2 (frame extraction with original timestamps).

Two kinds of tests, labelled in every docstring:
  REAL FFMPEG   - run the real ffmpeg/ffprobe binaries on SYNTHETIC TEST FIXTURES
                  generated at test time in pytest temp dirs (never committed).
  TEST DOUBLE   - feed hand-written ffprobe-style records or in-memory byte streams
                  to pure functions. No FFmpeg involved.

[ACn] tags refer to the 18 approved A2 acceptance criteria.
"""

import gc
import hashlib
import io
import json
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

import decompile.frames as frames_mod
from decompile.frames import (
    MARKER_NAME,
    MARKER_SCHEMA,
    TS_BEST_EFFORT_DIFFERS,
    TS_BEST_EFFORT_GUESS,
    TS_FRAME_PTS,
    FrameReader,
    FrameReaderClosedError,
    FrameStreamError,
    FrameStreamOverflowError,
    SourceMismatchError,
    TimestampError,
    VideoRejectedError,
    _first_frame_size,
    build_timestamps,
    check_frame_counts,
    classify_timestamp,
    extract_frames,
    iter_frame_bytes,
    iter_frames,
    load_frame_table,
    main,
    open_frames,
    read_frame_table,
    read_png_rgb,
    write_png_rgb,
)

ROOT = Path(__file__).resolve().parent.parent

# Irregular (variable-frame-rate) schedule for the index-coded fixture.
VFR_N = 60
VFR_TIMESCALE = 90000


def vfr_expected_pts(n: int) -> int:
    return round((n / 30 + n * n / 6000) * VFR_TIMESCALE)


# --------------------------------------------------------------------------------------
# Fixture helpers (all output is SYNTHETIC)
# --------------------------------------------------------------------------------------

def ffmpeg(*args: str, stdin: bytes | None = None) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], input=stdin, check=True)


def encode_rgb_frames(path: Path, frames: list[np.ndarray], codec_args: list[str], vf: str | None = None, extra: list[str] = ()) -> Path:
    h, w = frames[0].shape[:2]
    args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30", "-i", "-"]
    if vf:
        args += ["-vf", vf]
    ffmpeg(*args, *codec_args, *extra, str(path), stdin=b"".join(f.tobytes() for f in frames))
    return path


def index_coded_frame(n: int) -> np.ndarray:
    """128x64 frame whose 8 top blocks encode n in binary (white = 1). Survives lossy coding."""
    img = np.full((64, 128, 3), 40, np.uint8)
    for bit in range(8):
        if (n >> bit) & 1:
            img[8:24, bit * 16: bit * 16 + 16] = 255
    return img


def decode_index(img: np.ndarray) -> int:
    return sum(1 << bit for bit in range(8) if img[16, bit * 16 + 8, 0] > 128)


def marker_frame() -> np.ndarray:
    """64 wide x 48 high, white 8x8 marker in the stored top-left corner."""
    img = np.zeros((48, 64, 3), np.uint8)
    img[:8, :8] = 255
    return img


def colour_frame() -> np.ndarray:
    img = np.zeros((64, 64, 3), np.uint8)
    img[:32, :32] = (255, 0, 0)
    img[:32, 32:] = (0, 255, 0)
    img[32:, :32] = (0, 0, 255)
    img[32:, 32:] = (200, 30, 90)
    return img


LOSSLESS_RGB = ["-c:v", "libx264rgb", "-qp", "0"]


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_a2")
    m = {}
    m["cfr"] = d / "synthetic_cfr.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(m["cfr"]))

    m["vfr"] = encode_rgb_frames(
        d / "synthetic_vfr_coded.mp4",
        [index_coded_frame(n) for n in range(VFR_N)],
        ["-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "3",
         "-video_track_timescale", str(VFR_TIMESCALE)],
        vf=f"settb=1/{VFR_TIMESCALE},setpts=(N/30+N*N/6000)/TB",
    )

    m["offset"] = d / "synthetic_offset.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-output_ts_offset", "1", str(m["offset"]))

    m["rot0"] = encode_rgb_frames(d / "synthetic_rot0.mp4", [marker_frame()] * 30, LOSSLESS_RGB)
    for r in (90, 180, 270):
        m[f"rot{r}"] = d / f"synthetic_rot{r}.mp4"
        ffmpeg("-display_rotation", str(r), "-i", str(m["rot0"]), "-c", "copy", str(m[f"rot{r}"]))

    m["colour"] = encode_rgb_frames(d / "synthetic_colour.mp4", [colour_frame()] * 30, LOSSLESS_RGB)

    # 8 s (not 3 s) so trace-level stderr (~90 KB) clearly exceeds a 64 KB pipe buffer.
    m["big"] = d / "synthetic_1080p.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=8", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(m["big"]))

    m["faststart"] = d / "synthetic_faststart.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(m["faststart"]))
    data = m["faststart"].read_bytes()
    m["truncated"] = d / "synthetic_truncated_partial.mp4"
    m["truncated"].write_bytes(data[: int(len(data) * 0.6)])

    m["audio_only"] = d / "synthetic_audio_only.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(m["audio_only"]))
    return m


def ffprobe_frames(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pts,best_effort_timestamp,duration,pict_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["frames"]


def decode_without_autorotate(path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


# ======================================================================================
# Timestamps
# ======================================================================================

def test_cfr_timestamps_exact(media):
    """REAL FFMPEG [AC1]"""
    table = read_frame_table(media["cfr"])
    assert len(table.frames) == 60
    assert table.time_base == Fraction(1, 15360)
    assert [f.t for f in table.frames] == [Fraction(n, 30) for n in range(60)]
    assert all(f.timestamp_source == TS_FRAME_PTS for f in table.frames)
    assert table.synthesized_timestamp_count == 0
    pts = [f.pts for f in table.frames]
    assert all(b > a for a, b in zip(pts, pts[1:]))


def test_vfr_bframes_order_and_timestamps(media):
    """REAL FFMPEG [AC2] irregular timestamps + B-frames; pixel-coded indices prove ordering."""
    probe = ffprobe_frames(media["vfr"])
    assert "B" in {f["pict_type"] for f in probe}, "fixture must contain B-frames to be meaningful"

    table = read_frame_table(media["vfr"])
    assert table.time_base == Fraction(1, VFR_TIMESCALE)
    decoded_indices = []
    for timestamp, image in iter_frames(media["vfr"], table):
        decoded_indices.append(decode_index(image))
        assert abs(timestamp.pts - vfr_expected_pts(timestamp.index)) <= 1
    assert decoded_indices == list(range(VFR_N))
    pts = [f.pts for f in table.frames]
    assert all(b > a for a, b in zip(pts, pts[1:]))


def test_nonzero_start_time(media):
    """REAL FFMPEG [AC3]"""
    table = read_frame_table(media["offset"])
    assert table.first_pts == 15360  # 1.0 s in time_base 1/15360, preserved raw
    assert table.frames[0].pts == 15360
    assert table.frames[0].t == 0
    assert table.frames[1].t == Fraction(1, 30)


def test_double_timestamp_classification():
    """TEST DOUBLE [AC4] hand-written ffprobe records."""
    assert classify_timestamp({"pts": 10, "best_effort_timestamp": 10}) == (10, 10, TS_FRAME_PTS)
    assert classify_timestamp({"best_effort_timestamp": 10}) == (10, None, TS_BEST_EFFORT_GUESS)
    assert classify_timestamp({"pts": 9, "best_effort_timestamp": 10}) == (10, 9, TS_BEST_EFFORT_DIFFERS)
    with pytest.raises(TimestampError):
        classify_timestamp({"pts": 10})  # no best_effort_timestamp: must not be invented
    with pytest.raises(TimestampError):
        classify_timestamp({"pts": 10, "best_effort_timestamp": "N/A"})


def test_double_non_increasing_timestamps_rejected():
    """TEST DOUBLE [AC4]"""
    tb = Fraction(1, 1000)
    with pytest.raises(TimestampError):
        build_timestamps([{"best_effort_timestamp": 0}, {"best_effort_timestamp": 0}], tb)
    with pytest.raises(TimestampError):
        build_timestamps([{"best_effort_timestamp": 5}, {"best_effort_timestamp": 3}], tb)
    with pytest.raises(TimestampError):
        build_timestamps([], tb)


def test_double_synthesized_count():
    """TEST DOUBLE [AC4]"""
    records = [
        {"pts": 0, "best_effort_timestamp": 0},
        {"best_effort_timestamp": 40},
        {"pts": 70, "best_effort_timestamp": 80},
    ]
    frames = build_timestamps(records, Fraction(1, 1000))
    assert [f.timestamp_source for f in frames] == [TS_FRAME_PTS, TS_BEST_EFFORT_GUESS, TS_BEST_EFFORT_DIFFERS]
    assert [f.pts for f in frames] == [0, 40, 80]


# ======================================================================================
# Durations
# ======================================================================================

def test_duration_reported_matches_ffprobe(media):
    """REAL FFMPEG [AC5] reported durations copied exactly; absent stays None."""
    cfr_probe = ffprobe_frames(media["cfr"])
    cfr = read_frame_table(media["cfr"])
    assert [f.duration_reported for f in cfr.frames] == [p.get("duration") for p in cfr_probe]
    assert all(f.duration_reported == 512 for f in cfr.frames)

    vfr_probe = ffprobe_frames(media["vfr"])
    assert all("duration" not in p for p in vfr_probe), "fixture expected to lack per-frame durations"
    vfr = read_frame_table(media["vfr"])
    assert all(f.duration_reported is None for f in vfr.frames)


def test_double_missing_duration_stays_none():
    """TEST DOUBLE [AC5]"""
    frames = build_timestamps([{"best_effort_timestamp": 0}, {"best_effort_timestamp": 7, "duration": 3}], Fraction(1, 10))
    assert frames[0].duration_reported is None
    assert frames[1].duration_reported == 3


def test_duration_from_next_pts(media):
    """REAL FFMPEG [AC6] derived separately from adjacent pts; never from nominal fps."""
    table = read_frame_table(media["vfr"])
    for a, b in zip(table.frames, table.frames[1:]):
        assert a.duration_from_next_pts == b.pts - a.pts
        expected_gap = vfr_expected_pts(b.index) - vfr_expected_pts(a.index)
        assert abs(a.duration_from_next_pts - expected_gap) <= 2
    assert table.frames[-1].duration_from_next_pts is None
    gaps = [f.duration_from_next_pts for f in table.frames[:-1]]
    assert len(set(gaps)) > 10, "irregular schedule must produce many distinct gaps"


# ======================================================================================
# Streaming reliability
# ======================================================================================

def _assert_exited(process: subprocess.Popen) -> None:
    process.wait(timeout=5)
    assert process.poll() is not None


def test_cleanup_on_break_in_context_manager(media):
    """REAL FFMPEG [AC7]"""
    with open_frames(media["big"]) as reader:
        for _ in reader:
            break
    _assert_exited(reader.process)


def test_cleanup_on_exception(media):
    """REAL FFMPEG [AC7]"""
    reader = None
    with pytest.raises(RuntimeError):
        with open_frames(media["big"]) as reader:
            for _ in reader:
                raise RuntimeError("consumer failure")
    _assert_exited(reader.process)
    assert reader.terminated_by_reader


def test_cleanup_on_garbage_collection(media):
    """REAL FFMPEG [AC7] no context manager, reader dropped mid-stream."""
    reader = open_frames(media["big"])
    next(reader)
    process = reader.process
    del reader
    gc.collect()
    _assert_exited(process)


def test_cleanup_when_iter_frames_generator_closed(media):
    """REAL FFMPEG [AC7] the iter_frames generator, closed after one frame."""
    gen = iter_frames(media["big"])
    next(gen)
    process = gen.gi_frame.f_locals["reader"].process
    gen.close()
    _assert_exited(process)


def test_no_deadlock_with_large_stderr(media):
    """REAL FFMPEG [AC8] 1080p frames on stdout while FFmpeg writes trace-level stderr."""
    table = read_frame_table(media["big"])
    result = {}

    def consume():
        with open_frames(media["big"], table, loglevel="trace") as reader:
            result["count"] = sum(1 for _ in reader)
        result["stderr_len"] = len(reader.stderr_text)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive(), "frame reading deadlocked"
    assert result["count"] == len(table.frames) == 240
    assert result["stderr_len"] > 65536, "stderr must exceed a typical pipe buffer for this test to mean anything"


def test_double_partial_frame():
    """TEST DOUBLE [AC9] in-memory byte stream with 5.5 frames."""
    frame_bytes = 12
    stream = io.BytesIO(b"x" * (frame_bytes * 5 + 6))
    chunks = iter_frame_bytes(stream, frame_bytes, expected_frames=6)
    for _ in range(5):
        assert len(next(chunks)) == frame_bytes
    with pytest.raises(FrameStreamError, match="partial frame at index 5: got 6 of 12 bytes"):
        next(chunks)


def test_double_unexpected_eof():
    """TEST DOUBLE [AC9] fewer frames than the table lists."""
    stream = io.BytesIO(b"x" * 24)
    with pytest.raises(FrameStreamError, match="unexpected end of stream: got 2 of 3 frames"):
        list(iter_frame_bytes(stream, 12, expected_frames=3))


def test_double_too_many_frames():
    """TEST DOUBLE [AC9] more frames than the table lists."""
    stream = io.BytesIO(b"x" * 36)
    with pytest.raises(FrameStreamOverflowError, match="more than the expected 2 frames"):
        list(iter_frame_bytes(stream, 12, expected_frames=2))


def test_decode_error_reports_ffmpeg_stderr(media):
    """REAL FFMPEG [AC10] decoder stage run directly on a truncated mp4."""
    intact = read_frame_table(media["faststart"])
    reader = FrameReader(media["truncated"], intact.width, intact.height, intact.frames)
    with pytest.raises(FrameStreamError) as excinfo:
        for _ in reader:
            pass
    message = str(excinfo.value)
    assert "ffmpeg decode failed" in message
    assert reader.stderr_text.strip() and reader.stderr_text.strip() in message
    _assert_exited(reader.process)


def test_table_from_other_video_rejected_before_ffmpeg(media, monkeypatch):
    """REAL FFMPEG (table build) [AC11] mismatch detected before any FrameReader starts."""
    table = read_frame_table(media["cfr"])

    def must_not_start(*args, **kwargs):
        raise AssertionError("FrameReader must not be constructed for a mismatched table")

    monkeypatch.setattr(frames_mod, "FrameReader", must_not_start)
    with pytest.raises(SourceMismatchError):
        open_frames(media["offset"], table)


def test_double_frame_count_check():
    """TEST DOUBLE [AC12]"""
    check_frame_counts(a=5, b=5, c=5)
    with pytest.raises(TimestampError):
        check_frame_counts(a=5, b=4)
    with pytest.raises(TimestampError):
        check_frame_counts(a=None, b=5)


# ======================================================================================
# Rotation and colour
# ======================================================================================

# Expected corner of the stored top-left marker after FFmpeg auto-rotation.
# Derived from FFmpeg's documented convention (-display_rotation is counter-clockwise).
EXPECTED_MARKER_CORNER = {0: "TL", 90: "BL", 180: "BR", 270: "TR"}


def lit_corners(img: np.ndarray) -> list[str]:
    h, w = img.shape[:2]
    corners = {"TL": img[2, 2], "TR": img[2, w - 3], "BL": img[h - 3, 2], "BR": img[h - 3, w - 3]}
    return [name for name, px in corners.items() if px[0] > 128]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_orientation(media, rotation):
    """REAL FFMPEG [AC13] reported rotation, output size, pixel orientation."""
    path = media[f"rot{rotation}"]
    table = read_frame_table(path)
    assert table.rotation_deg == rotation
    expected_size = (48, 64) if rotation in (90, 270) else (64, 48)
    assert (table.width, table.height) == expected_size
    assert _first_frame_size(path) == expected_size

    unrotated = decode_without_autorotate(path, 64, 48)
    k = rotation // 90
    for i, (_, image) in enumerate(iter_frames(path, table)):
        assert np.array_equal(image, np.rot90(unrotated[i], k=k))
        assert lit_corners(image) == [EXPECTED_MARKER_CORNER[rotation]]


def test_exact_rgb_colours_survive_extraction_and_png(media, tmp_path):
    """REAL FFMPEG [AC14] lossless RGB video -> exact values -> PNG -> bit-identical."""
    expected = colour_frame()
    table = read_frame_table(media["colour"])
    out_dir = tmp_path / "frames"
    extract_frames(media["colour"], out_dir)
    for timestamp, image in iter_frames(media["colour"], table):
        assert np.array_equal(image, expected), "decoded colours are not exact"
        assert tuple(image[0, 0]) == (255, 0, 0)
        assert tuple(image[0, 63]) == (0, 255, 0)
        assert tuple(image[63, 0]) == (0, 0, 255)
        assert tuple(image[63, 63]) == (200, 30, 90)
        png = read_png_rgb(out_dir / f"frame_{timestamp.index:06d}.png")
        assert np.array_equal(png, image)


def test_png_helpers_convert_channel_order(tmp_path):
    """UNIT, OpenCV only (no FFmpeg, no test doubles) [AC15]"""
    rgb = colour_frame()
    path = tmp_path / "colour.png"
    write_png_rgb(path, rgb)
    assert np.array_equal(read_png_rgb(path), rgb)
    plain_opencv = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert tuple(plain_opencv[0, 0]) == (0, 0, 255), "plain OpenCV read is BGR: red appears as (0,0,255)"
    assert np.array_equal(plain_opencv[..., ::-1], rgb)


# ======================================================================================
# Export and integration
# ======================================================================================

def test_extract_frames_export(media, tmp_path):
    """REAL FFMPEG [AC16]"""
    out_dir = tmp_path / "export"
    table = extract_frames(media["vfr"], out_dir)
    pngs = sorted(out_dir.glob("frame_*.png"))
    assert len(pngs) == len(table.frames) == VFR_N
    assert not list(out_dir.glob("*.tmp"))
    for timestamp, image in iter_frames(media["vfr"], table):
        assert np.array_equal(read_png_rgb(out_dir / f"frame_{timestamp.index:06d}.png"), image)
    reloaded = load_frame_table(out_dir / "frames.json")
    assert reloaded == table
    assert all(f.duration_reported is None for f in reloaded.frames)  # None survives JSON
    assert not (out_dir / MARKER_NAME).exists(), "marker must be removed after a completed export"
    before = snapshot(out_dir)
    with pytest.raises(FileExistsError, match="completed export"):
        extract_frames(media["vfr"], out_dir)
    assert snapshot(out_dir) == before, "a completed export must not be modified"


# ======================================================================================
# A2 cleanup: export failure cleanup, retry safety, closed reader, Popen failure
# ======================================================================================

def snapshot(directory: Path) -> dict[str, str]:
    """Every entry under directory -> sha256 of its bytes (or 'DIR'). Detects any change."""
    result = {}
    for entry in sorted(directory.rglob("*")):
        rel = entry.relative_to(directory).as_posix()
        result[rel] = "DIR" if entry.is_dir() else hashlib.sha256(entry.read_bytes()).hexdigest()
    return result


def fail_png_writes_after(monkeypatch, n_ok: int, partial: bool = True) -> list[Path]:
    """TEST DOUBLE: replace write_png_rgb so the (n_ok+1)-th write leaves a partial file and fails."""
    real_write = frames_mod.write_png_rgb
    calls: list[Path] = []

    def flaky_write(path, rgb):
        calls.append(Path(path))
        if len(calls) > n_ok:
            if partial:
                Path(path).write_bytes(b"partial png")
            raise RuntimeError("simulated disk failure")
        real_write(path, rgb)

    monkeypatch.setattr(frames_mod, "write_png_rgb", flaky_write)
    return calls


def write_marker(directory: Path, source_sha256: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MARKER_NAME).write_text(json.dumps({"schema": MARKER_SCHEMA, "source_sha256": source_sha256}))


def assert_complete_export(out_dir: Path, video: Path) -> None:
    table = load_frame_table(out_dir / "frames.json")
    pngs = sorted(out_dir.glob("frame_*.png"))
    assert len(pngs) == len(table.frames)
    assert sorted(p.name for p in out_dir.iterdir()) == sorted([p.name for p in pngs] + ["frames.json"])
    for timestamp, image in iter_frames(video, table):
        assert np.array_equal(read_png_rgb(out_dir / f"frame_{timestamp.index:06d}.png"), image)


# --- Fix 1: cleanup of directories created by a failed extraction ----------------------

def test_failed_extraction_removes_created_nested_dirs(media, tmp_path):
    """REAL FFMPEG [cleanup 1] rejected video: no directory is left behind."""
    out_dir = tmp_path / "a" / "b" / "out"
    with pytest.raises(VideoRejectedError):
        extract_frames(media["audio_only"], out_dir)
    assert not (tmp_path / "a").exists()
    assert tmp_path.exists()


def test_failure_mid_write_removes_created_dirs(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLE writer [cleanup 1] partial PNGs, marker and created dirs removed."""
    fail_png_writes_after(monkeypatch, n_ok=3)
    out_dir = tmp_path / "new_parent" / "out"
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        extract_frames(media["cfr"], out_dir)
    assert not (tmp_path / "new_parent").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_failure_in_preexisting_empty_dir_keeps_dir(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLE writer [cleanup 1] pre-existing dir survives and is empty again."""
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    fail_png_writes_after(monkeypatch, n_ok=3)
    with pytest.raises(RuntimeError):
        extract_frames(media["cfr"], out_dir)
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []


def test_original_exception_preserved_when_cleanup_fails(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLEs [cleanup 1] cleanup errors become notes on the original exception."""
    fail_png_writes_after(monkeypatch, n_ok=2)

    def broken_remove(path):
        raise PermissionError(f"simulated: cannot delete {path.name}")

    monkeypatch.setattr(frames_mod, "_remove_file", broken_remove)
    with pytest.raises(RuntimeError, match="simulated disk failure") as excinfo:
        extract_frames(media["cfr"], tmp_path / "out")
    notes = getattr(excinfo.value, "__notes__", [])
    assert notes and all("cleanup after failed export" in n for n in notes)
    assert any("simulated: cannot delete" in n for n in notes)


def test_marker_written_atomically(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLE writer [cleanup 2] during export the marker is complete and valid, no tmp left."""
    out_dir = tmp_path / "out"
    seen = {}
    real_write = frames_mod.write_png_rgb

    def inspecting_write(path, rgb):
        if not seen:
            seen["marker"] = json.loads((out_dir / MARKER_NAME).read_text())
            seen["names"] = sorted(p.name for p in out_dir.iterdir())
        real_write(path, rgb)

    monkeypatch.setattr(frames_mod, "write_png_rgb", inspecting_write)
    table = extract_frames(media["cfr"], out_dir)
    assert seen["marker"] == {"schema": MARKER_SCHEMA, "source_sha256": table.source_sha256}
    assert seen["names"] == [MARKER_NAME], "marker tmp file must already be renamed away"


def test_failed_marker_rename_leaves_nothing(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLE os.replace [cleanup 2] marker is absent or complete, never partial."""
    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(frames_mod.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        extract_frames(media["cfr"], tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


# --- Fix 2: retry of an incomplete export ----------------------------------------------

def test_retry_after_real_failure_succeeds(media, tmp_path, monkeypatch):
    """REAL FFMPEG + TEST DOUBLE writer [cleanup 2] failed export stays retryable, retry completes it."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_marker(out_dir, read_frame_table(media["cfr"]).source_sha256)
    fail_png_writes_after(monkeypatch, n_ok=5)
    with pytest.raises(RuntimeError):
        extract_frames(media["cfr"], out_dir)
    assert sorted(p.name for p in out_dir.iterdir()) == [MARKER_NAME], "pre-existing marker kept, attempt files removed"

    monkeypatch.undo()
    extract_frames(media["cfr"], out_dir)
    assert_complete_export(out_dir, media["cfr"])


def test_retry_replaces_stale_files(media, tmp_path):
    """REAL FFMPEG [cleanup 2] stale PNGs and a stale manifest tmp from an earlier attempt are replaced."""
    out_dir = tmp_path / "out"
    write_marker(out_dir, read_frame_table(media["cfr"]).source_sha256)
    (out_dir / "frame_000000.png").write_bytes(b"stale")
    (out_dir / "frame_000999.png").write_bytes(b"stale, index beyond this video")
    (out_dir / "frames.json.tmp").write_text("{stale")
    extract_frames(media["cfr"], out_dir)
    assert_complete_export(out_dir, media["cfr"])
    assert not (out_dir / "frame_000999.png").exists()


REFUSAL_REASON = {
    "marker_other_video": "different video",
    "malformed_marker": "unreadable or malformed",
    "marker_wrong_schema": "unreadable or malformed",
    "unrelated_file": "unexpected entry",
    "subdirectory": "unexpected entry",
    "user_pngs_without_marker": "not an incomplete export",
    "marker_and_manifest": "completed export",
    "bad_png_name": "unexpected entry",
}


@pytest.mark.parametrize("case", list(REFUSAL_REASON))
def test_retry_refused_and_dir_untouched(media, tmp_path, case):
    """REAL FFMPEG [cleanup 2] anything not an incomplete export of this video is refused untouched,
    for the specific documented reason."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sha = read_frame_table(media["cfr"]).source_sha256
    if case == "marker_other_video":
        write_marker(out_dir, "0" * 64)
        (out_dir / "frame_000000.png").write_bytes(b"from another video")
    elif case == "malformed_marker":
        (out_dir / MARKER_NAME).write_text("{not json")
    elif case == "marker_wrong_schema":
        (out_dir / MARKER_NAME).write_text(json.dumps({"schema": "something/else", "source_sha256": sha}))
    elif case == "unrelated_file":
        write_marker(out_dir, sha)
        (out_dir / "notes.txt").write_text("user data")
    elif case == "subdirectory":
        write_marker(out_dir, sha)
        (out_dir / "frame_000001.png").mkdir()
    elif case == "user_pngs_without_marker":
        (out_dir / "frame_000000.png").write_bytes(b"user's own image")
    elif case == "marker_and_manifest":
        write_marker(out_dir, sha)
        (out_dir / "frames.json").write_text("{}")
    elif case == "bad_png_name":
        write_marker(out_dir, sha)
        (out_dir / "frame_1.png").write_bytes(b"not our naming scheme")
    before = snapshot(out_dir)
    with pytest.raises(FileExistsError, match=REFUSAL_REASON[case]):
        extract_frames(media["cfr"], out_dir)
    assert snapshot(out_dir) == before


def test_output_path_is_a_file_refused(media, tmp_path):
    """REAL FFMPEG [cleanup 2]"""
    target = tmp_path / "out"
    target.write_text("user file")
    with pytest.raises(FileExistsError, match="not a plain directory"):
        extract_frames(media["cfr"], target)
    assert target.read_text() == "user file"


# --- Fix 3: next() after close() --------------------------------------------------------

def test_next_after_close_mid_stream(media):
    """REAL FFMPEG [cleanup 3]"""
    reader = open_frames(media["cfr"])
    next(reader)
    reader.close()
    with pytest.raises(FrameReaderClosedError, match="closed after 1 of 60 frames"):
        next(reader)
    assert issubclass(FrameReaderClosedError, FrameStreamError)


def test_next_after_close_before_first_frame(media):
    """REAL FFMPEG [cleanup 3]"""
    reader = open_frames(media["cfr"])
    reader.close()
    with pytest.raises(FrameReaderClosedError):
        next(reader)


def test_next_after_exhaustion_keeps_stopiteration(media):
    """REAL FFMPEG [cleanup 3] successful completion: StopIteration forever, also after close()."""
    reader = open_frames(media["cfr"])
    assert sum(1 for _ in reader) == 60
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(reader)
    reader.close()
    with pytest.raises(StopIteration):
        next(reader)


def test_next_after_stream_error_raises_closed_error(media):
    """REAL FFMPEG [cleanup 3] a reader shut down by a stream error is 'closed', not 'exhausted'."""
    intact = read_frame_table(media["faststart"])
    reader = FrameReader(media["truncated"], intact.width, intact.height, intact.frames)
    with pytest.raises(FrameStreamError):
        for _ in reader:
            pass
    with pytest.raises(FrameReaderClosedError):
        next(reader)


# --- Fix 4: temp file closed immediately when FFmpeg cannot start ----------------------

def test_stderr_tempfile_closed_when_popen_fails(media, monkeypatch):
    """TEST DOUBLES (Popen, TemporaryFile wrapper) [cleanup 4] closed immediately, not by GC."""
    table = read_frame_table(media["cfr"])
    created_files = []
    real_tempfile = frames_mod.tempfile.TemporaryFile

    def recording_tempfile(*args, **kwargs):
        f = real_tempfile(*args, **kwargs)
        created_files.append(f)  # strong reference: refcount finalisation cannot close it
        return f

    def failing_popen(*args, **kwargs):
        raise FileNotFoundError("simulated: ffmpeg executable not found")

    monkeypatch.setattr(frames_mod.tempfile, "TemporaryFile", recording_tempfile)
    monkeypatch.setattr(frames_mod.subprocess, "Popen", failing_popen)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(FileNotFoundError, match="simulated: ffmpeg executable not found"):
            FrameReader(media["cfr"], table.width, table.height, table.frames)
        assert len(created_files) == 1
        assert created_files[0].closed, "temp file must be closed before the exception propagates"
    finally:
        if gc_was_enabled:
            gc.enable()


def test_rejected_video_raises(media):
    """REAL FFMPEG [AC17]"""
    with pytest.raises(VideoRejectedError) as excinfo:
        read_frame_table(media["audio_only"])
    assert "no video stream" in excinfo.value.reasons


def test_cli_exit_codes(media, tmp_path, capsys):
    """REAL FFMPEG [AC18] in-process."""
    assert main([str(media["cfr"]), str(tmp_path / "ok")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["frame_count"] == 60 and summary["synthesized_timestamp_count"] == 0
    assert main([str(media["audio_only"]), str(tmp_path / "bad")]) == 1
    assert "VideoRejectedError" in json.loads(capsys.readouterr().out)["error"]


def test_cli_subprocess(media, tmp_path):
    """REAL FFMPEG [AC18] as a real subprocess."""
    ok = subprocess.run([sys.executable, "-m", "decompile.frames", str(media["cfr"]), str(tmp_path / "ok")],
                        cwd=ROOT, capture_output=True, text=True)
    bad = subprocess.run([sys.executable, "-m", "decompile.frames", str(media["audio_only"]), str(tmp_path / "bad")],
                         cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert bad.returncode == 1, bad.stderr
