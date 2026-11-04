"""Secret redaction, filename sanitisation and safe path resolution."""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"
SECRET_ENV_VARS = ("MODEL_API_KEY",)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-~+/=]{8,}")
_KEYLIKE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")
RUN_ID_RE = re.compile(r"^web_\d{8}T\d{6}Z_[0-9a-f]{8}$")
RECON_ID_RE = re.compile(r"^r\d{1,4}$")
CF_ID_RE = re.compile(r"^cf_[0-9a-f]{16}$")


def _secrets() -> list[str]:
    return [v for v in (os.environ.get(n) for n in SECRET_ENV_VARS) if v and len(v) >= 4]


def redact(obj: Any) -> Any:
    """Remove configured secret values (and bearer tokens) from any string inside a JSON-like object."""
    if isinstance(obj, str):
        s = obj
        for secret in _secrets():
            s = s.replace(secret, REDACTED)
        s = _BEARER.sub(lambda m: m.group(1) + REDACTED, s)
        return _KEYLIKE.sub(REDACTED, s)
    if isinstance(obj, dict):
        return {redact(k): redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def sanitize_filename(name: str | None, *, default: str = "video.mp4", max_len: int = 120) -> str:
    """A display-only filename: no directories, no control or shell-special characters."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ ()\-]+", "_", name).strip(" .")
    if not name:
        return default
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: max_len - len(ext) - 1] + "." + ext) if dot and len(ext) <= 8 else name[:max_len]
    return name


def safe_child(base: Path, rel: str) -> Path:
    """Resolve ``rel`` inside ``base``; raises ValueError for absolute paths, '..' or escapes."""
    if not rel or rel.startswith(("/", "\\")) or ":" in rel or "\x00" in rel:
        raise ValueError("invalid path")
    parts = Path(rel.replace("\\", "/")).parts
    if any(p in ("..", ".", "") for p in parts):
        raise ValueError("invalid path")
    base_r = base.resolve()
    p = (base_r / Path(*parts)).resolve()
    if base_r != p and base_r not in p.parents:
        raise ValueError("invalid path")
    return p
