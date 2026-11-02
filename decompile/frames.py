"""A2: frame extraction with original timestamps.

Timestamp semantics (important):
    `FrameTimestamp.pts` is FFmpeg's `best_effort_timestamp`: the DECODED PRESENTATION
    timestamp, i.e. the container timestamp after demuxing, mp4 edit-list handling and
    decoder reordering. It is NOT necessarily the camera's capture time. It is stored as
    a raw integer in the stream time base, exactly as FFmpeg reports it. Timestamps are
    never invented from frame indices or a nominal frame rate: if FFmpeg reports no
    timestamp for a frame, extraction fails.

Pipeline:
    1. A1 (`probe_video` + `validate`) must accept the file.
    2. ffprobe lists every decoded frame's timestamp, in decoder output order.
    3. ffmpeg decodes the same stream to raw RGB on stdout (passthrough, no frame
       duplication or dropping); pixel frames are paired with timestamps by order.
    4. Frame counts from A1, ffprobe and the pixel stream must all agree.

Rotation: FFmpeg auto-rotates according to the display-matrix metadata. The ffprobe
`rotation` value r corresponds to numpy `np.rot90(stored_frame, k=(r % 360) // 90)`
(counter-clockwise), verified by tests on this FFmpeg build.

Colour: all arrays are RGB uint8 (H, W, 3). OpenCV is BGR, so PNG I/O goes only through
`write_png_rgb` / `read_png_rgb`, which convert explicitly.

CLI:  python -m decompile.frames <video> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, Iterator

import cv2
import numpy as np

from decompile.ingest import VideoInfo, probe_video, sha256_file, validate

TS_FRAME_PTS = "frame_pts"  # frame carries its own pts, equal to best_effort_timestamp
TS_BEST_EFFORT_GUESS = "best_effort_guess"  # no frame pts; FFmpeg synthesized the timestamp
TS_BEST_EFFORT_DIFFERS = "best_effort_differs"  # both present but different

MANIFEST_NAME = "frames.json"
MANIFEST_TMP_NAME = MANIFEST_NAME + ".tmp"
MANIFEST_SCHEMA = "decompile.frames/1"
MARKER_NAME = ".export_in_progress.json"
MARKER_SCHEMA = "decompile.frames.export_in_progress/1"
PNG_PATTERN = "frame_{index:06d}.png"
PNG_NAME_RE = re.compile(r"frame_\d{6}\.png")
TERMINATE_TIMEOUT_S = 5
NATURAL_EXIT_TIMEOUT_S = 30
TIMESTAMP_SEMANTICS = (
    "pts is FFmpeg best_effort_timestamp: the decoded presentation timestamp in stream "
    "time_base ticks (after demuxing, edit lists and decoder reordering). It is not "
    "necessarily the camera capture time. t = (pts - first_pts) * time_base, exact."
)


class VideoRejectedError(Exception):
    def __init__(self, path: str, reasons: list[str]):
        self.reasons = reasons
        super().__init__(f"{path} rejected by A1 validation: {'; '.join(reasons)}")


class TimestampError(Exception):
    """Missing, non-increasing or inconsistent frame timestamps / counts."""


class FrameStreamError(Exception):
    """The raw pixel stream from FFmpeg was incomplete, oversized or reported errors."""


class FrameStreamOverflowError(FrameStreamError):
    """FFmpeg produced more frames than the timestamp table lists (it may still be running)."""


class FrameReaderClosedError(FrameStreamError):
    """next() was called on a FrameReader that was closed before its stream completed.

    Raised after an explicit close() mid-stream (or before the first frame), and after the
    reader shut down because of a stream error. A reader whose stream completed
    successfully keeps raising StopIteration instead, even after close().
    """


class SourceMismatchError(Exception):
    """A FrameTable or VideoInfo was used with a different video file."""


@dataclass(frozen=True)
class FrameTimestamp:
    index: int
    pts: int
    frame_pts: int | None
    timestamp_source: str
    t: Fraction
    duration_reported: int | None
    duration_from_next_pts: int | None
    key_frame: bool


@dataclass
class FrameTable:
    source_path: str
    source_sha256: str
    time_base: Fraction
    first_pts: int
    width: int  # output width, after rotation
    height: int  # output height, after rotation
    rotation_deg: int
    frames: list[FrameTimestamp] = field(default_factory=list)

    @property
    def synthesized_timestamp_count(self) -> int:
        return sum(1 for f in self.frames if f.timestamp_source != TS_FRAME_PTS)

    def times_s(self) -> np.ndarray:
        return np.array([float(f.t) for f in self.frames], dtype=np.float64)

    def to_json_dict(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA,
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "time_base": str(self.time_base),
            "first_pts": self.first_pts,
            "width": self.width,
            "height": self.height,
            "rotation_deg": self.rotation_deg,
            "frame_count": len(self.frames),
            "synthesized_timestamp_count": self.synthesized_timestamp_count,
            "png_pattern": PNG_PATTERN,
            "frames": [
                {
                    "index": f.index,
                    "pts": f.pts,
                    "frame_pts": f.frame_pts,
                    "timestamp_source": f.timestamp_source,
                    "t": str(f.t),
                    "duration_reported": f.duration_reported,
                    "duration_from_next_pts": f.duration_from_next_pts,
                    "key_frame": f.key_frame,
                }
                for f in self.frames
            ],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "FrameTable":
        if data.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema: {data.get('schema')!r}")
        frames = [
            FrameTimestamp(
                index=f["index"],
                pts=f["pts"],
                frame_pts=f["frame_pts"],
                timestamp_source=f["timestamp_source"],
                t=Fraction(f["t"]),
                duration_reported=f["duration_reported"],
                duration_from_next_pts=f["duration_from_next_pts"],
                key_frame=f["key_frame"],
            )
            for f in data["frames"]
        ]
        return cls(
            source_path=data["source_path"],
            source_sha256=data["source_sha256"],
            time_base=Fraction(data["time_base"]),
            first_pts=data["first_pts"],
            width=data["width"],
            height=data["height"],
            rotation_deg=data["rotation_deg"],
            frames=frames,
        )


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------

def _int_or_none(value) -> int | None:
    """ffprobe omits absent fields in JSON; also guard against 'N/A' strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None


def classify_timestamp(record: dict) -> tuple[int, int | None, str]:
    """Return (pts, frame_pts, timestamp_source) for one ffprobe frame record."""
    best_effort = _int_or_none(record.get("best_effort_timestamp"))
    if best_effort is None:
        raise TimestampError(f"frame has no best_effort_timestamp: {record}")
    frame_pts = _int_or_none(record.get("pts"))
    if frame_pts is None:
        source = TS_BEST_EFFORT_GUESS
    elif frame_pts == best_effort:
        source = TS_FRAME_PTS
    else:
        source = TS_BEST_EFFORT_DIFFERS
    return best_effort, frame_pts, source


def build_timestamps(records: list[dict], time_base: Fraction) -> list[FrameTimestamp]:
    """Convert ffprobe frame records (in decoder output order) to FrameTimestamps."""
    if not records:
        raise TimestampError("no frames reported by ffprobe")

    classified = [classify_timestamp(r) for r in records]
    pts_values = [pts for pts, _, _ in classified]
    for i in range(1, len(pts_values)):
        if pts_values[i] <= pts_values[i - 1]:
            raise TimestampError(
                f"timestamps not strictly increasing at frame {i}: "
                f"{pts_values[i - 1]} -> {pts_values[i]}"
            )

    first_pts = pts_values[0]
    frames = []
    for i, (record, (pts, frame_pts, source)) in enumerate(zip(records, classified)):
        is_last = i == len(records) - 1
        frames.append(
            FrameTimestamp(
                index=i,
                pts=pts,
                frame_pts=frame_pts,
                timestamp_source=source,
                t=(pts - first_pts) * time_base,
                duration_reported=_int_or_none(record.get("duration")),
                duration_from_next_pts=None if is_last else pts_values[i + 1] - pts,
                key_frame=bool(record.get("key_frame", 0)),
            )
        )
    return frames


def check_frame_counts(**counts: int | None) -> None:
    """All named frame counts must be present and equal."""
    if any(v is None for v in counts.values()) or len(set(counts.values())) != 1:
        raise TimestampError(f"frame counts disagree: {counts}")


# --------------------------------------------------------------------------------------
# Rotation and stream metadata
# --------------------------------------------------------------------------------------

def normalize_rotation(rotation: float | int | None) -> int:
    if rotation is None:
        return 0
    if float(rotation) != int(rotation) or int(rotation) % 90 != 0:
        raise ValueError(f"unsupported rotation {rotation}; only multiples of 90 degrees")
    return int(rotation) % 360


def rotated_size(width: int, height: int, rotation_deg: int) -> tuple[int, int]:
    return (height, width) if rotation_deg in (90, 270) else (width, height)


def _run_ffprobe_json(args: list[str]) -> dict:
    proc = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise TimestampError(f"ffprobe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _probe_stream(path: Path) -> dict:
    data = _run_ffprobe_json([
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,time_base:stream_side_data=rotation",
        str(path),
    ])
    return data["streams"][0]


def _probe_frame_records(path: Path) -> list[dict]:
    data = _run_ffprobe_json([
        "-select_streams", "v:0",
        "-show_entries", "frame=pts,best_effort_timestamp,duration,key_frame",
        str(path),
    ])
    return data.get("frames", [])


def _stream_rotation(stream: dict) -> int:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            return normalize_rotation(side_data["rotation"])
    return 0


def _first_frame_size(path: Path) -> tuple[int, int]:
    """Decode the first (auto-rotated) frame as PNG, which is self-describing, and read its size."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"],
        capture_output=True,
    )
    image = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_UNCHANGED)
    if proc.returncode != 0 or image is None:
        raise FrameStreamError(f"could not decode first frame: {proc.stderr.decode(errors='replace')}")
    return image.shape[1], image.shape[0]


def read_frame_table(path: str | Path, info: VideoInfo | None = None) -> FrameTable:
    path = Path(path)
    if info is None:
        info = probe_video(path)
    elif info.sha256 != sha256_file(path):
        raise SourceMismatchError(f"VideoInfo sha256 does not match {path}")

    reasons = validate(info)
    if reasons:
        raise VideoRejectedError(str(path), reasons)

    stream = _probe_stream(path)
    time_base = Fraction(stream["time_base"])
    rotation = _stream_rotation(stream)
    width, height = rotated_size(stream["width"], stream["height"], rotation)

    actual_size = _first_frame_size(path)
    if actual_size != (width, height):
        raise FrameStreamError(
            f"computed output size {width}x{height} (rotation {rotation}) does not match "
            f"decoded first frame {actual_size[0]}x{actual_size[1]}"
        )

    records = _probe_frame_records(path)
    check_frame_counts(a1_decoded=info.decoded_frame_count, ffprobe_frames=len(records))
    frames = build_timestamps(records, time_base)

    return FrameTable(
        source_path=str(path.resolve()),
        source_sha256=info.sha256,
        time_base=time_base,
        first_pts=frames[0].pts,
        width=width,
        height=height,
        rotation_deg=rotation,
        frames=frames,
    )


# --------------------------------------------------------------------------------------
# Raw frame streaming
# --------------------------------------------------------------------------------------

def read_exact(stream: BinaryIO, n: int) -> bytes:
    """Read up to n bytes, looping over short reads. Returns fewer only at EOF."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def iter_frame_bytes(stream: BinaryIO, frame_bytes: int, expected_frames: int) -> Iterator[bytes]:
    """Yield exactly `expected_frames` chunks of `frame_bytes`, or raise FrameStreamError."""
    for index in range(expected_frames):
        data = read_exact(stream, frame_bytes)
        if not data:
            raise FrameStreamError(f"unexpected end of stream: got {index} of {expected_frames} frames")
        if len(data) < frame_bytes:
            raise FrameStreamError(
                f"partial frame at index {index}: got {len(data)} of {frame_bytes} bytes"
            )
        yield data
    if stream.read(1):
        raise FrameStreamOverflowError(f"stream has more than the expected {expected_frames} frames")


class FrameReader:
    """Decodes a video to RGB frames through an FFmpeg subprocess.

    Use as a context manager (or call close()) so FFmpeg is always cleaned up. FFmpeg's
    stderr goes to a temporary file, not a pipe, so it can never block the stdout pipe.

    Iterator semantics:
      - StopIteration: the stream completed successfully (repeats on every later next(),
        including after close()).
      - FrameReaderClosedError: the reader was closed before the stream completed.
    """

    def __init__(self, path: str | Path, width: int, height: int, timestamps: list[FrameTimestamp], loglevel: str = "error"):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.timestamps = timestamps
        self.loglevel = loglevel
        self.stderr_text = ""
        self.terminated_by_reader = False  # True if close() had to stop a still-running FFmpeg
        self._closed = True  # until FFmpeg has started, there is nothing for close()/__del__ to do
        self._exhausted = False  # True only after a successful, complete stream
        self._stderr_file = tempfile.TemporaryFile()
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-v", loglevel,
            "-i", str(self.path),
            # passthrough: no frame duplication/dropping. enc_time_base demux: keep the input
            # time base so VFR timestamps are not rescaled into false muxer dts errors.
            "-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base", "demux",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._stderr_file, stdin=subprocess.DEVNULL)
        except BaseException:
            # FFmpeg never started: release the temp file now, not at garbage collection.
            self._stderr_file.close()
            raise
        self._closed = False
        self._chunks = iter_frame_bytes(self.process.stdout, width * height * 3, len(timestamps))
        self._index = 0

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> "FrameReader":
        return self

    def __next__(self) -> tuple[FrameTimestamp, np.ndarray]:
        if self._exhausted:
            raise StopIteration
        if self._closed:
            raise FrameReaderClosedError(
                f"FrameReader for {self.path.name} was closed after {self._index} of "
                f"{len(self.timestamps)} frames, before the stream completed"
            )
        try:
            data = next(self._chunks)
        except StopIteration:
            self._finish()  # raises if FFmpeg reported errors
            self._exhausted = True
            raise
        except FrameStreamError as exc:
            self._finish(stream_error=exc)
            raise  # _finish raises a more specific error if FFmpeg reported one
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        timestamp = self.timestamps[self._index]
        self._index += 1
        return timestamp, image

    def _finish(self, stream_error: FrameStreamError | None = None) -> None:
        """Wait for FFmpeg, then raise if it failed or reported errors."""
        if isinstance(stream_error, FrameStreamOverflowError):
            # FFmpeg may still be writing; stop it. Its exit status is irrelevant here.
            self.close()
            return
        # stdout reached EOF, so FFmpeg is exiting: let it finish and flush stderr naturally.
        try:
            self.process.wait(timeout=NATURAL_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass
        self.close()
        ffmpeg_failed = self.process.returncode != 0 and not self.terminated_by_reader
        # At loglevel "error", any stderr output is an error. At chattier levels only the exit code counts.
        reported_errors = self.loglevel == "error" and self.stderr_text.strip()
        if ffmpeg_failed or reported_errors:
            raise FrameStreamError(
                f"ffmpeg decode failed (exit {self.process.returncode}) after "
                f"{self._index} frames: {self.stderr_text.strip()}"
            ) from stream_error

    def close(self) -> None:
        """Stop FFmpeg (terminate, then kill) and collect its stderr. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.terminated_by_reader = True
            self.process.terminate()
            try:
                self.process.wait(timeout=TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._stderr_file.seek(0)
        self.stderr_text = self._stderr_file.read().decode("utf-8", errors="replace")
        self._stderr_file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def open_frames(path: str | Path, table: FrameTable | None = None, loglevel: str = "error") -> FrameReader:
    """Open a FrameReader. A supplied table must come from the same file (checked by SHA-256)."""
    path = Path(path)
    if table is None:
        table = read_frame_table(path)
    elif table.source_sha256 != sha256_file(path):
        raise SourceMismatchError(
            f"FrameTable was built from a different file (sha256 {table.source_sha256[:12]}...) than {path}"
        )
    return FrameReader(path, table.width, table.height, table.frames, loglevel=loglevel)


def iter_frames(path: str | Path, table: FrameTable | None = None) -> Iterator[tuple[FrameTimestamp, np.ndarray]]:
    with open_frames(path, table) as reader:
        yield from reader


# --------------------------------------------------------------------------------------
# PNG I/O (explicit RGB <-> BGR conversion; byte-based so non-ASCII Windows paths work)
# --------------------------------------------------------------------------------------

def write_png_rgb(path: str | Path, rgb: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"PNG encoding failed for {path}")
    Path(path).write_bytes(encoded.tobytes())


def read_png_rgb(path: str | Path) -> np.ndarray:
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not decode PNG {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------

# Export directory states, decided before anything is created or deleted:
#   absent     -> created by this export
#   empty      -> used as is
#   incomplete -> retried: no frames.json; a valid in-progress marker whose source_sha256
#                 equals this video's; every other entry is a regular file named
#                 frame_NNNNNN.png or frames.json.tmp
#   anything else (including a completed export) -> FileExistsError, nothing touched
DIR_ABSENT = "absent"
DIR_EMPTY = "empty"
DIR_INCOMPLETE = "incomplete"


def _read_marker(marker_path: Path) -> dict | None:
    try:
        data = json.loads(marker_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != MARKER_SCHEMA:
        return None
    return data


def _inspect_output_dir(out_dir: Path, source_sha256: str) -> tuple[str, list[Path]]:
    """Classify out_dir. Returns (state, stale_files_to_replace). Raises FileExistsError to refuse."""
    if not out_dir.exists() and not out_dir.is_symlink():
        return DIR_ABSENT, []
    if not out_dir.is_dir() or out_dir.is_symlink():
        raise FileExistsError(f"output path exists and is not a plain directory: {out_dir}")

    entries = list(out_dir.iterdir())
    if not entries:
        return DIR_EMPTY, []

    names = {entry.name for entry in entries}
    if MANIFEST_NAME in names:
        raise FileExistsError(f"output directory holds a completed export (refusing to overwrite): {out_dir}")
    if MARKER_NAME not in names:
        raise FileExistsError(f"output directory is not empty and is not an incomplete export: {out_dir}")

    marker = _read_marker(out_dir / MARKER_NAME)
    if marker is None:
        raise FileExistsError(f"in-progress marker is unreadable or malformed: {out_dir / MARKER_NAME}")
    if marker.get("source_sha256") != source_sha256:
        raise FileExistsError(f"output directory holds an incomplete export of a different video: {out_dir}")

    stale = []
    for entry in entries:
        if entry.name == MARKER_NAME:
            continue
        is_plain_file = entry.is_file() and not entry.is_symlink()
        is_export_file = PNG_NAME_RE.fullmatch(entry.name) or entry.name == MANIFEST_TMP_NAME
        if not (is_plain_file and is_export_file):
            raise FileExistsError(f"unexpected entry in incomplete export (refusing to touch it): {entry}")
        stale.append(entry)
    return DIR_INCOMPLETE, stale


def _missing_dirs(out_dir: Path) -> list[Path]:
    """Directories from out_dir upward that do not exist yet, deepest first."""
    missing = []
    current = out_dir.absolute()
    while not current.exists():
        missing.append(current)
        current = current.parent
    return missing


def _write_text_atomic(path: Path, text: str, written: list[Path]) -> None:
    """Write via a sibling .tmp file and os.replace, so `path` is either absent or complete."""
    tmp_path = path.with_name(path.name + ".tmp")
    written.append(tmp_path)
    tmp_path.write_text(text)
    written.append(path)
    os.replace(tmp_path, path)


def _remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _cleanup_failed_export(written: list[Path], created_dirs: list[Path]) -> list[str]:
    """Remove files written by this attempt and directories it created. Returns cleanup errors."""
    errors = []
    for file_path in reversed(written):
        try:
            _remove_file(file_path)
        except OSError as exc:
            errors.append(f"could not remove {file_path}: {exc}")
    for directory in created_dirs:  # deepest first
        try:
            directory.rmdir()  # only succeeds if empty; never removes user data
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"could not remove created directory {directory}: {exc}")
    return errors


def extract_frames(path: str | Path, out_dir: str | Path) -> FrameTable:
    """Export every frame as PNG plus frames.json. frames.json present = complete export.

    Safety rules:
      - A completed export (frames.json present) is never modified.
      - A non-empty directory is only written to if it is an incomplete export of this
        same video (see _inspect_output_dir); its stale frame files are replaced.
      - On failure, files written by this attempt and directories it created are removed;
        pre-existing directories and the pre-existing marker are kept, so an incomplete
        export stays retryable. Cleanup problems are attached as notes to the original
        exception, which is always re-raised unchanged.
    """
    out_dir = Path(out_dir)
    table = read_frame_table(path)  # validates the video before touching the filesystem
    state, stale_files = _inspect_output_dir(out_dir, table.source_sha256)

    written: list[Path] = []
    created_dirs: list[Path] = []
    try:
        if state == DIR_ABSENT:
            created_dirs = _missing_dirs(out_dir)
            out_dir.mkdir(parents=True)
        if state == DIR_INCOMPLETE:
            for stale in stale_files:
                _remove_file(stale)
        else:
            marker = {"schema": MARKER_SCHEMA, "source_sha256": table.source_sha256}
            _write_text_atomic(out_dir / MARKER_NAME, json.dumps(marker, indent=2), written)

        for timestamp, image in iter_frames(path, table):
            png_path = out_dir / PNG_PATTERN.format(index=timestamp.index)
            written.append(png_path)
            write_png_rgb(png_path, image)

        _write_text_atomic(out_dir / MANIFEST_NAME, json.dumps(table.to_json_dict(), indent=2), written)
    except BaseException as exc:
        for problem in _cleanup_failed_export(written, created_dirs):
            exc.add_note(f"cleanup after failed export: {problem}")
        raise

    # The export is complete from here on; a leftover marker next to frames.json is harmless.
    try:
        _remove_file(out_dir / MARKER_NAME)
    except OSError:
        pass
    return table


def load_frame_table(manifest_path: str | Path) -> FrameTable:
    return FrameTable.from_json_dict(json.loads(Path(manifest_path).read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frames with original timestamps.")
    parser.add_argument("video")
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    try:
        table = extract_frames(args.video, args.out_dir)
    except (FileNotFoundError, FileExistsError, VideoRejectedError, TimestampError,
            FrameStreamError, SourceMismatchError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    times = table.times_s()
    summary = {
        "ok": True,
        "out_dir": str(Path(args.out_dir).resolve()),
        "frame_count": len(table.frames),
        "time_base": str(table.time_base),
        "first_pts": table.first_pts,
        "last_t_s": float(times[-1]),
        "width": table.width,
        "height": table.height,
        "rotation_deg": table.rotation_deg,
        "synthesized_timestamp_count": table.synthesized_timestamp_count,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
