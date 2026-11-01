"""Tests for A1 (video ingest and validation).

All media here are SYNTHETIC TEST FIXTURES generated at test time from ffmpeg's lavfi
test sources inside pytest temp directories. Nothing is committed.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from decompile.ingest import main, probe_video, reconcile_frame_count, validate

ROOT = Path(__file__).resolve().parent.parent


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_testsrc(path: Path, size="320x240", rate=30, duration=2.0, codec_args=("-c:v", "libx264", "-pix_fmt", "yuv420p"), extra=()):
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={duration}", *codec_args, *extra, str(path))
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_media")
    m = {}
    m["valid_mp4"] = make_testsrc(d / "synthetic_valid.mp4")
    m["valid_webm"] = make_testsrc(d / "synthetic_valid.webm", codec_args=("-c:v", "libvpx-vp9"))
    m["short"] = make_testsrc(d / "synthetic_short.mp4", duration=0.2)
    m["tiny"] = make_testsrc(d / "synthetic_tiny.mp4", size="32x32")
    m["mkv"] = make_testsrc(d / "synthetic_valid_codec.mkv")

    m["text"] = d / "synthetic_text.mp4"
    m["text"].write_text("this is not a video\n" * 50)

    m["truncated_1kb"] = d / "synthetic_truncated_1kb.mp4"
    m["truncated_1kb"].write_bytes(m["valid_mp4"].read_bytes()[:1024])

    # moov atom at the front (faststart) so ffprobe can still read metadata, but the
    # media data is cut off: metadata says 60 frames, the decoder cannot deliver them.
    faststart = make_testsrc(d / "synthetic_faststart.mp4", extra=("-movflags", "+faststart"))
    data = faststart.read_bytes()
    m["truncated_partial"] = d / "synthetic_truncated_partial.mp4"
    m["truncated_partial"].write_bytes(data[: int(len(data) * 0.6)])

    m["audio_only"] = d / "synthetic_audio_only.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(m["audio_only"]))
    return m


# --- acceptance criterion 1: a valid video is accepted with exact metadata -------------

def test_valid_mp4_metadata_exact(media):
    path = media["valid_mp4"]
    info = probe_video(path)
    assert validate(info) == []
    assert info.codec == "h264"
    assert (info.width, info.height) == (320, 240)
    assert info.nominal_fps == "30/1"
    assert info.avg_fps == "30/1"
    assert info.duration_s == pytest.approx(2.0, abs=1e-3)
    assert info.decoded_frame_count == 60
    assert info.reported_frame_count == 60
    assert info.frame_count == 60
    assert info.frame_count_status == "consistent"
    assert info.decode_errors == []
    assert info.has_audio is False
    assert info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info.size_bytes == path.stat().st_size


# --- requirement: missing frame-count metadata handled explicitly ----------------------

def test_webm_missing_frame_count_uses_decoded(media):
    info = probe_video(media["valid_webm"])
    assert info.reported_frame_count is None  # webm carries no nb_frames
    assert info.decoded_frame_count == 60
    assert info.frame_count == 60
    assert info.frame_count_status == "reported_missing"
    assert any("does not report a frame count" in w for w in info.warnings)
    assert validate(info) == []


# --- requirement: real decodability, not just readable metadata ------------------------

def test_partially_truncated_mp4_fails_decode_check(media):
    info = probe_video(media["truncated_partial"])
    assert info.probe_error is None, "ffprobe should still read metadata from the faststart header"
    assert info.reported_frame_count == 60
    assert info.decode_errors, "ffmpeg must report the corruption"
    assert info.decoded_frame_count < 60
    assert info.frame_count == info.decoded_frame_count  # metadata count is never trusted
    assert info.frame_count_status == "mismatch"
    reasons = validate(info)
    assert any("decode errors" in r for r in reasons), reasons


# --- acceptance criterion 2: each invalid input rejected with a specific reason --------

@pytest.mark.parametrize(
    "key, expected",
    [
        ("text", "unreadable by ffprobe"),
        ("truncated_1kb", "unreadable by ffprobe"),
        ("audio_only", "no video stream"),
        ("short", "too short"),
        ("tiny", "resolution 32x32"),
        ("mkv", "unsupported container"),
    ],
)
def test_invalid_inputs_rejected(media, key, expected):
    reasons = validate(probe_video(media[key]))
    assert any(expected in r for r in reasons), reasons


# --- acceptance criterion 3: missing path ----------------------------------------------

def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "does_not_exist.mp4")


# --- frame-count reconciliation logic ---------------------------------------------------

@pytest.mark.parametrize(
    "reported, decoded, status",
    [(60, 60, "consistent"), (None, 60, "reported_missing"), (60, 59, "mismatch"), (60, None, "undecoded")],
)
def test_reconcile_frame_count(reported, decoded, status):
    assert reconcile_frame_count(reported, decoded) == status


# --- acceptance criterion 4: CLI exit codes (in-process and as a real subprocess) -------

def test_cli_main_exit_codes(media, capsys):
    assert main([str(media["valid_mp4"])]) == 0
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert main([str(media["text"])]) == 1
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_cli_subprocess(media):
    ok = subprocess.run([sys.executable, "-m", "decompile.ingest", str(media["valid_mp4"])], cwd=ROOT, capture_output=True, text=True)
    bad = subprocess.run([sys.executable, "-m", "decompile.ingest", str(media["audio_only"])], cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert bad.returncode == 1, bad.stderr
    assert json.loads(ok.stdout)["info"]["decoded_frame_count"] == 60
