"""Durable run state: one directory per run with ``run.json`` (atomic rewrite) and ``events.jsonl`` (append-only).

The files on disk are the source of truth; in-memory locks only serialise writers inside this process. A browser
refresh, or a server restart, reloads everything from disk. Runs with a job still marked "running" when the server
starts are marked "interrupted" (see ``RunStore.recover``).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .security import RUN_ID_RE, redact

RUN_FILE = "run.json"
EVENTS_FILE = "events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_id() -> str:
    return f"web_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(redact(obj), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    for attempt in range(40):  # Windows: replace fails while another process (e.g. a scanner) holds the file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.025)


class RunNotFound(KeyError):
    pass


class RunStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    # -- paths

    def run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise RunNotFound(run_id)
        return self.data_dir / run_id

    def lock(self, run_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(run_id, threading.RLock())

    # -- run.json

    def create(self, run: dict) -> dict:
        d = self.run_dir(run["id"])
        d.mkdir(parents=True, exist_ok=False)
        write_json_atomic(d / RUN_FILE, run)
        return run

    def exists(self, run_id: str) -> bool:
        try:
            return (self.run_dir(run_id) / RUN_FILE).is_file()
        except RunNotFound:
            return False

    def load(self, run_id: str) -> dict:
        p = self.run_dir(run_id) / RUN_FILE
        with self.lock(run_id):  # readers and writers of one run never overlap inside this process
            if not p.is_file():
                raise RunNotFound(run_id)
            return json.loads(p.read_text(encoding="utf-8"))

    def save(self, run: dict) -> dict:
        run["updated_utc"] = utc_now()
        with self.lock(run["id"]):
            write_json_atomic(self.run_dir(run["id"]) / RUN_FILE, run)
        return run

    @contextmanager
    def edit(self, run_id: str) -> Iterator[dict]:
        """Read-modify-write under the run lock."""
        with self.lock(run_id):
            run = self.load(run_id)
            yield run
            self.save(run)

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.data_dir.iterdir(), reverse=True):
            if p.is_dir() and RUN_ID_RE.fullmatch(p.name) and (p / RUN_FILE).is_file():
                try:
                    out.append(self.load(p.name))
                except (OSError, ValueError, RunNotFound):
                    continue
        return out

    # -- events

    def append_event(self, run_id: str, event: dict) -> dict:
        with self.lock(run_id):
            p = self.run_dir(run_id) / EVENTS_FILE
            seq = 0
            if p.is_file():
                with open(p, "rb") as f:
                    seq = sum(1 for _ in f)
            ev = redact({"seq": seq + 1, "t": utc_now(), **event})
            with open(p, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(ev, sort_keys=True, allow_nan=False) + "\n")
            return ev

    def read_events(self, run_id: str, after: int = 0) -> list[dict]:
        p = self.run_dir(run_id) / EVENTS_FILE
        with self.lock(run_id):
            if not p.is_file():
                return []
            with open(p, encoding="utf-8") as f:
                lines = f.readlines()
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue  # a line being written concurrently; it will be read next time
            if ev.get("seq", 0) > after:
                out.append(ev)
        return out

    # -- crash recovery

    def recover(self) -> list[str]:
        """Mark jobs left 'running' by a previous server process as interrupted. Returns the affected run ids."""
        affected = []
        for run in self.list():
            job = run.get("job")
            if job and job.get("status") == "running":
                with self.edit(run["id"]) as r:
                    _mark_interrupted(r)
                self.append_event(run["id"], {"type": "job", "kind": job.get("kind"), "status": "interrupted",
                                              "message": "The server stopped while this job was running."})
                affected.append(run["id"])
        return affected


def _mark_interrupted(run: dict) -> None:
    job = run["job"]
    job.update({"status": "interrupted", "finished_utc": utc_now()})
    for st in run.get("stages", {}).values():
        if st.get("status") in ("running", "pending"):
            st["status"] = "interrupted"
    if job.get("kind") == "sam_tracking":
        run["sam"]["status"] = "interrupted"
        run["sam"].setdefault("warnings", []).append(
            "The server stopped during the SAM request. If the request had already been sent it may have been "
            "processed and billed; it was NOT retried. Check the API evidence before confirming another request.")
    elif job.get("kind") == "reconstruction":
        for rec in run.get("reconstructions", []):
            if rec.get("status") == "running":
                rec["status"] = "interrupted"
                rec["error"] = "The server stopped while this reconstruction was running."
