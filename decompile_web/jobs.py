"""Background job runner. Job state is persisted in run.json (the source of truth); the executor only runs code."""

from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from .pipeline import PipelineError
from .security import redact
from .store import RunStore, utc_now


class JobConflict(Exception):
    pass


class JobRunner:
    def __init__(self, store: RunStore, workers: int = 2) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decompile-job")
        self.futures: dict[str, Future] = {}

    def submit(self, run_id: str, kind: str, fn: Callable[[], None], *, before: Callable[[dict], None] | None = None) -> dict:
        """Mark the job running (refusing if another job of this run is running), then execute ``fn`` in the pool.
        ``before(run)`` may raise to refuse the job; it runs under the run lock, atomically with the mark."""
        with self.store.edit(run_id) as run:
            job = run.get("job")
            if job and job.get("status") == "running":
                raise JobConflict(f"a {job.get('kind')} job is already running for this run")
            if before is not None:
                before(run)
            run["job"] = {"kind": kind, "status": "running", "started_utc": utc_now(), "finished_utc": None,
                          "error": None}
        self.store.append_event(run_id, {"type": "job", "kind": kind, "status": "running"})
        fut = self.pool.submit(self._run, run_id, kind, fn)
        self.futures[run_id] = fut
        return self.store.load(run_id)["job"]

    def _run(self, run_id: str, kind: str, fn: Callable[[], None]) -> None:
        status, error, details = "completed", None, []
        try:
            fn()
        except PipelineError as exc:
            status, error, details = "failed", str(exc), exc.details
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", redact(f"{type(exc).__name__}: {exc}")
            self.store.append_event(run_id, {"type": "log", "level": "debug",
                                             "message": redact(traceback.format_exc(limit=6))})
        with self.store.edit(run_id) as run:
            run["job"].update({"status": status, "finished_utc": utc_now(), "error": error, "details": details})
        self.store.append_event(run_id, {"type": "job", "kind": kind, "status": status, "error": error,
                                         "details": details})

    def wait(self, run_id: str, timeout: float | None = None) -> None:
        fut = self.futures.get(run_id)
        if fut is not None:
            fut.result(timeout=timeout)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
