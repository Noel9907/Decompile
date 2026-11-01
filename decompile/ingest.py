"""A1: video ingest and validation.

probe_video() inspects a file with ffprobe, hashes it, and fully decodes its first
video stream with ffmpeg (frames are discarded to a null sink) to prove the file is
actually decodable and to count frames. The decoded count is the authoritative frame
count; the container-reported count is kept only for comparison, because it is often
missing (e.g. webm) or wrong.

validate() turns a VideoInfo into a list of rejection reasons (empty = accepted).

CLI:  python -m decompile.ingest <video>   -> JSON on stdout, exit 0 accepted / 1 rejected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
# ffprobe format_name values that may back those extensions.
ALLOWED_FORMAT_TOKENS = {"mp4", "mov", "webm"}
MIN_DURATION_S = 0.5
MIN_FRAMES = 10
MIN_SIDE_PX = 64
MAX_SIDE_PX = 4096
DECODE_TIMEOUT_S = 300


@dataclass
class VideoInfo:
    path: str
    size_bytes: int
    sha256: str
    extension: str
    container: str | None = None  # ffprobe format_name, e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    nominal_fps: str | None = None  # r_frame_rate as an exact fraction string, e.g. "30/1"
    avg_fps: str | None = None  # avg_frame_rate as an exact fraction string
    has_audio: bool = False
    reported_frame_count: int | None = None  # container metadata (nb_frames); may be absent/wrong
    decoded_frame_count: int | None = None  # frames actually decoded by ffmpeg
    duration_frame_estimate: int | None = None  # round(duration * avg_fps), informational only
    frame_count_status: str = "undecoded"  # consistent | reported_missing | mismatch | undecoded
    probe_error: str | None = None
    decode_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int | None:
        """Authoritative frame count: only ever the decoded count."""
        return self.decoded_frame_count


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _fraction_or_none(value: str | None) -> Fraction | None:
    try:
        frac = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return frac if frac > 0 else None


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_ffprobe(path: Path) -> tuple[dict | None, str | None]:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or f"ffprobe exited with code {proc.returncode}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"ffprobe returned invalid JSON: {exc}"


def decode_video_stream(path: Path) -> tuple[int | None, list[str]]:
    """Decode every frame of the first video stream to a null sink.

    Returns (decoded_frame_count, error_lines). -fps_mode passthrough guarantees frames
    are neither duplicated nor dropped, so the count is the true decoded frame count.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "error",
        "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough",
        "-f", "null", "-",
        "-progress", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, [f"decode timed out after {DECODE_TIMEOUT_S}s"]

    errors = [line for line in proc.stderr.splitlines() if line.strip()]
    if proc.returncode != 0 and not errors:
        errors.append(f"ffmpeg exited with code {proc.returncode}")

    frames = None
    for line in proc.stdout.splitlines():
        if line.startswith("frame="):
            frames = int(line.split("=", 1)[1])  # last progress block wins
    return frames, errors


def reconcile_frame_count(reported: int | None, decoded: int | None) -> str:
    """Classify agreement between container-reported and decoded frame counts."""
    if decoded is None:
        return "undecoded"
    if reported is None:
        return "reported_missing"
    return "consistent" if reported == decoded else "mismatch"


def probe_video(path: str | Path) -> VideoInfo:
    """Inspect, hash and decode a video file. Never raises for bad media, only for a missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such video file: {path}")

    info = VideoInfo(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        extension=path.suffix.lower(),
    )

    probe, error = _run_ffprobe(path)
    if error:
        info.probe_error = error
        return info

    fmt = probe.get("format", {})
    info.container = fmt.get("format_name")
    streams = probe.get("streams", [])
    info.has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return info

    info.codec = video.get("codec_name")
    info.width = video.get("width")
    info.height = video.get("height")
    info.nominal_fps = video.get("r_frame_rate")
    info.avg_fps = video.get("avg_frame_rate")
    # Stream duration is preferred; webm and some other containers only carry format duration.
    info.duration_s = _float_or_none(video.get("duration")) or _float_or_none(fmt.get("duration"))
    nb = video.get("nb_frames")
    info.reported_frame_count = int(nb) if nb is not None and str(nb).isdigit() and int(nb) > 0 else None

    avg = _fraction_or_none(info.avg_fps)
    if info.duration_s is not None and avg is not None:
        info.duration_frame_estimate = round(info.duration_s * float(avg))

    info.decoded_frame_count, info.decode_errors = decode_video_stream(path)
    info.frame_count_status = reconcile_frame_count(info.reported_frame_count, info.decoded_frame_count)

    if info.frame_count_status == "reported_missing":
        info.warnings.append("container does not report a frame count; using decoded count")
    elif info.frame_count_status == "mismatch":
        info.warnings.append(
            f"container reports {info.reported_frame_count} frames but {info.decoded_frame_count} "
            "were decoded; using decoded count"
        )
    if (
        info.duration_frame_estimate is not None
        and info.decoded_frame_count is not None
        and abs(info.duration_frame_estimate - info.decoded_frame_count) > 1
    ):
        info.warnings.append(
            f"duration x avg_fps suggests {info.duration_frame_estimate} frames but "
            f"{info.decoded_frame_count} were decoded (truncation, variable frame rate or "
            "bad metadata; timestamps are checked in A3)"
        )
    return info


def validate(info: VideoInfo) -> list[str]:
    """Return rejection reasons. An empty list means the video is accepted."""
    if info.probe_error:
        return [f"unreadable by ffprobe: {info.probe_error}"]
    if info.codec is None:
        return ["no video stream"]

    reasons = []
    container_tokens = set((info.container or "").split(","))
    if info.extension not in ALLOWED_EXTENSIONS or not container_tokens & ALLOWED_FORMAT_TOKENS:
        reasons.append(
            f"unsupported container: extension {info.extension!r}, format {info.container!r} "
            f"(allowed: {sorted(ALLOWED_EXTENSIONS)})"
        )
    if info.decode_errors:
        reasons.append(f"decode errors: {'; '.join(info.decode_errors[:3])}")
    if not info.decoded_frame_count:
        reasons.append("no frames could be decoded")
    elif info.decoded_frame_count < MIN_FRAMES:
        reasons.append(f"too few frames: {info.decoded_frame_count} decoded (minimum {MIN_FRAMES})")
    if info.duration_s is None:
        reasons.append("duration unknown")
    elif info.duration_s < MIN_DURATION_S:
        reasons.append(f"too short: {info.duration_s:.3f}s (minimum {MIN_DURATION_S}s)")
    if info.width and info.height:
        longest = max(info.width, info.height)
        if longest < MIN_SIDE_PX or longest > MAX_SIDE_PX:
            reasons.append(
                f"resolution {info.width}x{info.height} outside allowed longest side "
                f"{MIN_SIDE_PX}-{MAX_SIDE_PX}px"
            )
    else:
        reasons.append("resolution unknown")
    return reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe and validate a video for Decompile.")
    parser.add_argument("video")
    args = parser.parse_args(argv)
    try:
        info = probe_video(args.video)
    except FileNotFoundError as exc:
        print(json.dumps({"accepted": False, "reasons": [str(exc)]}, indent=2))
        return 1
    reasons = validate(info)
    report = {"accepted": not reasons, "reasons": reasons, "info": asdict(info)}
    print(json.dumps(report, indent=2))
    return 0 if not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
