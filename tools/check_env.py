"""Environment check for Decompile.

Prints PASS/FAIL per requirement and exits non-zero if any hard requirement fails.
Makes no network calls. Never prints the API key, only whether it is present.
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 11)

# (import name, distribution name as pinned in requirements.txt)
PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("openai", "openai"),
    ("httpx2", "httpx2"),
    ("meta_sam_parser", "meta-sam-parser"),
    ("dotenv", "python-dotenv"),
    ("pytest", "pytest"),
]

failures = 0


def report(ok: bool, label: str, detail: str = "", hard: bool = True) -> None:
    global failures
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    if not ok and hard:
        failures += 1
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    report(v[:2] >= MIN_PYTHON, "python >= 3.11", f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def check_tool(name: str) -> None:
    path = shutil.which(name)
    if not path:
        report(False, name, "not found on PATH")
        return
    out = subprocess.run([name, "-version"], capture_output=True, text=True)
    first_line = out.stdout.splitlines()[0] if out.stdout else "(no version output)"
    report(out.returncode == 0, name, first_line)


def check_packages() -> None:
    from importlib.metadata import PackageNotFoundError, version

    pinned = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            dist, ver = line.split("==")
            pinned[dist.strip()] = ver.strip()

    for module, dist in PACKAGES:
        try:
            importlib.import_module(module)
            installed = version(dist)
        except (ImportError, PackageNotFoundError) as exc:
            report(False, dist, f"import failed: {exc}")
            continue
        expected = pinned.get(dist)
        report(installed == expected, dist, f"installed {installed}, pinned {expected}")


def check_env_file() -> None:
    # Soft check: a missing key is reported but does not fail the environment check.
    env_path = ROOT / ".env"
    if not env_path.exists():
        report(False, ".env MODEL_API_KEY", ".env missing (copy .env.example)", hard=False)
        return
    from dotenv import dotenv_values

    key = dotenv_values(env_path).get("MODEL_API_KEY") or ""
    report(bool(key.strip()), ".env MODEL_API_KEY", "present" if key.strip() else "empty", hard=False)


def main() -> int:
    check_python()
    check_tool("ffmpeg")
    check_tool("ffprobe")
    check_packages()
    check_env_file()
    print(f"\n{'ALL REQUIRED CHECKS PASSED' if failures == 0 else f'{failures} REQUIRED CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
