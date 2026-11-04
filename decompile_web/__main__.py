"""python -m decompile_web [--host 127.0.0.1] [--port 8000]"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m decompile_web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    # one process: run state is guarded by in-process locks (files on disk remain the source of truth)
    uvicorn.run(create_app(), host=a.host, port=a.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
