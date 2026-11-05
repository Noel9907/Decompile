#!/usr/bin/env python3
"""Run every MP4 in a folder through a running Decompile web server, exactly as the browser would.

  python tools/web_batch.py clips/ --server http://127.0.0.1:8010 [--confirm-live-sam]

Per clip: upload -> validation -> SAM 3.1 plan (prompt = file name, e.g. "tennis ball.mp4" -> "tennis ball";
"refuse_shoe.mp4" -> "shoe") -> LIVE billed SAM call ONLY with --confirm-live-sam (the cost is printed first) ->
automatic bounce window (same rule as the web UI) -> decompile (projective, then image-space if refused) ->
held-out prediction of the frames after the window -> Muse Spark review. Prints a scoreboard. Failures are kept
and reported as "refused (reason)"; nothing is retried.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx


def prompt_for(path: Path) -> str:
    stem = re.sub(r"^(refuse|reject)[_\-\s]*", "", path.stem, flags=re.I)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\d+", "", stem).strip()
    return stem or "ball"


def suggest_window(tracks: dict, max_win: int = 300):
    """Python port of the web UI's suggestWindow (frontend/src/pages/ProcessTab.tsx)."""
    objs = tracks["objects"]
    if not objs:
        return None
    oid = max(objs, key=lambda k: len(objs[k]))
    pts = [(r["f"], (r["box"][1] + r["box"][3]) / 2) for r in objs[oid]]
    if len(pts) < 8:
        return None
    n = tracks["frame_count"]
    bounces = [i for i in range(2, len(pts) - 2)
               if pts[i][1] >= pts[i - 1][1] and pts[i][1] >= pts[i + 1][1]
               and pts[i][1] - pts[i - 2][1] > 2 and pts[i][1] - pts[i + 2][1] > 2]
    if not bounces:
        return None
    best = None
    for k, bi in enumerate(bounces):
        bf = pts[bi][0]
        s0 = pts[bounces[k - 1]][0] + 4 if k > 0 else pts[0][0]
        e0 = pts[bounces[k + 1]][0] - 3 if k < len(bounces) - 1 else pts[-1][0] + 1
        score = min(bf - s0, e0 - bf)
        if best is None or score > best[3]:
            best = (s0, e0, bf, score)
    start, end, bounce = max(0, best[0]), min(n, best[1]), best[2]
    if end - start > max_win:
        end = start + max_win
    return {"sam_object": oid, "start": start, "end": end, "bounce": bounce}


def wait(c: httpx.Client, rid: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{rid}").json()
        if r.get("job") and r["job"]["status"] != "running":
            return r
        time.sleep(1)
    raise TimeoutError(rid)


def run_clip(c: httpx.Client, path: Path, live: bool) -> dict:
    out = {"clip": path.name, "status": "refused"}
    r = c.post("/api/runs", files={"file": (path.name, path.read_bytes(), "video/mp4")}, data={"provenance": "real"})
    if r.status_code != 201:
        return {**out, "reason": f"upload: {r.json().get('detail')}"}
    rid = r.json()["id"]
    out["run_id"] = rid
    run = wait(c, rid)
    if run["state"] == "rejected":
        return {**out, "reason": "; ".join(run["rejection_reasons"])}
    prompt = prompt_for(path)
    plan = c.post(f"/api/runs/{rid}/plan", json={"prompt": {"kind": "concept", "text": prompt}}).json()
    print(f"  {path.name}: prompt '{prompt}', {plan['expected_processed_frames']} frames, est. ${plan['cost']['expected_cost_usd']:.4f}")
    if not live:
        return {**out, "status": "planned", "reason": "live SAM not confirmed (--confirm-live-sam)"}
    r = c.post(f"/api/runs/{rid}/sam/confirm", json={"confirm": True, "plan_sha256": plan["sha256"],
                                                      "acknowledged_cost_usd": plan["cost"]["expected_cost_usd"]})
    if r.status_code != 202:
        return {**out, "reason": f"SAM not started: {r.json().get('detail')}"}
    run = wait(c, rid)
    if run["sam"]["status"] != "completed":
        return {**out, "reason": f"SAM {run['sam']['status']}: {run['job'].get('error')}"}
    out["sam_cost_usd"] = run["sam"]["attempts"][-1].get("actual_cost_usd")
    tracks = c.get(f"/api/runs/{rid}/sam/tracks").json()
    win = suggest_window(tracks)
    if win is None:
        return {**out, "reason": "no bounce found in the SAM track"}
    out["window"] = [win["start"], win["end"] - 1]
    rec = None
    for model in ("projective", "image_space"):
        r = c.post(f"/api/runs/{rid}/reconstructions", json={"sam_object": win["sam_object"], "start_frame": win["start"],
                                                             "end_frame": win["end"], "physics_model": model})
        if r.status_code != 202:
            out["reason"] = f"decompile not started: {r.json().get('detail')}"
            continue
        recon_id = r.json()["recon_id"]
        run = wait(c, rid)
        rec = next(x for x in run["reconstructions"] if x["id"] == recon_id)
        if rec["outcome"] == "scene_program":
            out["model"] = model
            break
        out["reason"] = rec.get("error") or f"{model}: no executable scene program"
        rec = None
    if rec is None:
        return out
    out["status"] = "decompiled"
    p = c.post(f"/api/runs/{rid}/reconstructions/{rec['id']}/predict")
    out["prediction"] = p.json().get("summary") if p.status_code == 200 else {"error": p.json().get("detail")}
    e = c.post(f"/api/runs/{rid}/reconstructions/{rec['id']}/explain")
    out["review"] = e.json().get("headline") if e.status_code == 200 else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8010")
    ap.add_argument("--confirm-live-sam", action="store_true", help="make the billed SAM 3.1 call for each clip")
    a = ap.parse_args(argv)
    clips = sorted(p for p in a.folder.iterdir() if p.suffix.lower() == ".mp4")
    if not clips:
        print("no .mp4 files in", a.folder)
        return 2
    c = httpx.Client(base_url=a.server, timeout=900)
    results = []
    for p in clips:
        print(f"> {p.name}")
        try:
            res = run_clip(c, p, a.confirm_live_sam)
        except Exception as exc:  # noqa: BLE001 - keep going, report it
            res = {"clip": p.name, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        results.append(res)
        print(f"  -> {res['status']}: {res.get('reason') or res.get('prediction') or ''}")
    print("\nSCOREBOARD")
    for r in c.get("/api/scoreboard").json():
        pr = r.get("prediction") or {}
        print(f"{r['filename'][:34]:34} {r['status']:10} e={r.get('restitution') or 0:.3f} "
              f"pre-bounce err={pr.get('pre_bounce_error_px')} next-bounce err ms={pr.get('next_bounce_error_ms')} "
              f"{(r.get('reason') or '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
