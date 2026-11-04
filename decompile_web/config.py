"""Server-side configuration (environment variables only; nothing here is ever sent to the browser except the
explicitly public fields of ``Settings.public``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DECOMPILE_WEB_DATA", ROOT / "runs" / "web")))
    # A live (billed) SAM request is impossible unless this is explicitly enabled on the server AND the user
    # confirms the exact plan in the browser.
    live_sam_enabled: bool = field(default_factory=lambda: _env_bool("DECOMPILE_LIVE_SAM", False))
    max_upload_bytes: int = field(default_factory=lambda: int(_env_float("DECOMPILE_MAX_UPLOAD_MB", 200) * 1024 * 1024))
    max_sam_cost_usd: float = field(default_factory=lambda: _env_float("DECOMPILE_MAX_SAM_COST_USD", 1.0))
    max_import_bytes: int = 256 * 1024 * 1024
    frontend_dist: Path = field(default_factory=lambda: ROOT / "frontend" / "dist")
    env_file: Path = field(default_factory=lambda: ROOT / ".env")
    workers: int = 2

    def load_env_file(self) -> None:
        """Load .env (MODEL_API_KEY) into the server process environment; never overrides real env vars."""
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover
            return
        if self.env_file.is_file():
            load_dotenv(self.env_file, override=False)

    @staticmethod
    def api_key_configured() -> bool:
        return bool(os.environ.get("MODEL_API_KEY"))

    def public(self) -> dict:
        return {"live_sam_enabled": self.live_sam_enabled, "api_key_configured": self.api_key_configured(),
                "max_upload_bytes": self.max_upload_bytes, "max_sam_cost_usd": self.max_sam_cost_usd}
