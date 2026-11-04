# Decompile web application

*Turn reality into executable physics.* A browser workspace around the existing Decompile pipeline:

```
upload MP4 → validate + exact frame table → SAM 3.1 plan + cost → (explicit confirmation) → SAM tracking
           → select object + window → outline measurement → trajectory → physics fit → scene program
           → deterministic simulation → counterfactuals (restitution, gravity) → artifacts
```

The website contains **no physics, geometry or SAM logic of its own**. The backend (`decompile_web/`) calls
`tools/sam_demo.py` (`make_plan`, `track`, `reconstruct`, `counterfactual`) and `decompile.*` (ingest, frames,
sam_api, sam_tracking, scene_program, projective_program). The frontend (`frontend/`) only displays what the
backend returns.

## Run locally

Prerequisites: the project's Python venv (`.venv`, see `requirements.txt`), FFmpeg on PATH, Node 18+.

```powershell
# one-time
.venv\Scripts\python -m pip install -r requirements-web.txt
cd frontend; npm install; npm run build; cd ..

# start (serves the built UI and the API on http://127.0.0.1:8000)
.venv\Scripts\python -m decompile_web
```

Live SAM requests are **disabled by default**. To allow them (the server reads `MODEL_API_KEY` from `.env`):

```powershell
$env:DECOMPILE_LIVE_SAM = "1"; .venv\Scripts\python -m decompile_web
```

Frontend development with hot reload (the Vite dev server proxies `/api` to port 8000):

```powershell
.venv\Scripts\python -m decompile_web          # terminal 1
cd frontend; npm run dev                       # terminal 2 → http://localhost:5173
```

### Configuration (environment variables, server side only)

| Variable | Default | Meaning |
|---|---|---|
| `DECOMPILE_LIVE_SAM` | `0` | `1` allows confirmed, billed SAM 3.1 requests |
| `MODEL_API_KEY` | from `.env` | Meta Model API key; never sent to the browser, logged or written to artifacts |
| `DECOMPILE_MAX_SAM_COST_USD` | `1.0` | refuse any SAM plan whose estimated cost is higher |
| `DECOMPILE_MAX_UPLOAD_MB` | `200` | upload size limit |
| `DECOMPILE_WEB_DATA` | `runs/web` | where runs are stored (git-ignored like all of `runs/`) |

Run as a single process (`python -m decompile_web` does this): run state is guarded by in-process locks.

## Demo without a new paid request

A SAM result produced earlier by `tools/sam_demo.py track` for the **same video bytes** can be attached with
*Import sam_tracking.json* on the Processing tab. It is verified against the uploaded video's SHA-256 and frame
table, labelled "imported", and no request or charge is made. Example (the P6c clip):

1. Upload `WhatsApp Video 2026-11-04 at 4.37.47 PM.mp4`.
2. Import `runs/sam_demo_p6c_qualitative/track/sam_tracking.json` (read only; it is copied into the run).
3. Reconstruct frames 28–54 (end exclusive), physics model *Projective*.
4. Open Reconstruction → Physics → Simulation & counterfactuals → Artifacts.

## What each tab shows

| Tab | Content (all from engine outputs) |
|---|---|
| Processing | video preview and metadata, suitability checks, live stage list (SSE), SAM prompt (concept or box), cost plan, confirmation dialog, SAM usage, object selection, reconstruction window and model |
| Reconstruction | original video; exact decoded frames with SAM masks/boxes, fitted outline, measured path and scene program; per-frame status strip (measured / uncertain / missing with reasons); x(t), y(t), D(t) with ±1σ/±3σ bars; contact event with bracketing frames; full measurement table |
| Physics | result category (fitted reconstruction vs held-out prediction vs counterfactual), restitution, gravity, contact time, initial state, fit RMS, adequacy gate, model verdicts, residuals, assumptions, limitations; or the reasons no program was produced |
| Simulation | observed frame vs executed simulation side by side, restitution / gravity / horizon controls re-running the simulator on the server, fitted-vs-counterfactual table, hypothetical labelling, counterfactual video rendering |
| Artifacts | run summary, engine report.md, every file that exists (download / view) |

## Safety and cost controls

* No SAM request on upload or planning. A request needs: `DECOMPILE_LIVE_SAM=1`, an API key on the server, a
  plan, and a confirmation that echoes the plan's SHA-256 and estimated cost (checked server side), within
  `DECOMPILE_MAX_SAM_COST_USD`.
* The existing client never retries a billed request after it was sent; the web layer never retries either and
  records whether the request was sent and a billing note on failure. Actual usage
  (`usage.video_frames_processed`) and the cost derived from it are recorded.
* Secrets: every persisted JSON and event passes through redaction (the configured key value and bearer tokens);
  the SAM client scrubs its own evidence files. Tests assert the key appears in no file or response.
* Uploads: `.mp4` with an `ftyp` header only, streamed with a size limit, stored under a server-generated run id;
  the original filename is display-only and sanitised. FFmpeg validation rejects corrupt or unsupported media.
* Downloads are served only for files in the artifact listing of that run (no path parameters reach the
  filesystem otherwise); ids are validated by pattern. No subprocess is built from user input.
* Restart safety: run state lives in `run.json` + `events.jsonl`. Jobs left running by a stopped server are
  marked *interrupted* on start (with a billing warning for SAM).

## Known limitations

* One body, one contact per window (the implemented physics). Multi-object / multi-contact is not supported.
* The web app produces **fitted reconstructions** (consistency checks) and **hypothetical counterfactuals**. Held-out
  prediction (pre-registration, freeze, reveal) remains CLI-only: `tools/real_video_demo.py`.
* `decompile.projective_program.simulate` does not model resting contact: after a counterfactual body comes to rest,
  its samples pass through the floor. The web layer marks them (`valid_until_s`) and does not draw them; the engine
  itself is unchanged.
* The browser's video clock gives an approximate frame number ("≈"); the frame inspector uses exact decoded frames.
* HEVC playback depends on the browser; analysis is unaffected.

## Tests

```powershell
.venv\Scripts\python -m pytest tests/test_web_app.py      # backend, offline (mocked SAM transport, network forbidden)
cd frontend; npm test                                      # UI: upload validation, confirmation gate, counterfactuals, ...
```

API reference: `web_API.md`, or the live OpenAPI page at `/api/docs`.
