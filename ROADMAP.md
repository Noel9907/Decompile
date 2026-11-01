# DECOMPILE — Roadmap

Workflow: **PLAN → IMPLEMENT ONE SMALL TASK → TEST → VERIFY → APPROVE → NEXT TASK.**

Status legend:
- ⬜ not started
- 🔨 implemented: code exists, not yet tested
- 🧪 tested: automated or scripted checks pass
- ✅ verified: acceptance criteria passed on real inputs **and** approved by the project owner
- ⛔ blocked

Priority: **P0** submission fails without it · **P1** makes it strong · **P2** nice to have.

Deadline: submission due **Nov 4, 21:00 ET**. Internal: feature freeze Nov 4 12:00 ET, submit by 18:00 ET.

## Labelling rules (binding)

- **SYNTHETIC** data (videos we generate) is always labelled as such in file names (`synthetic_*`), manifests (`"synthetic": true`) and any UI or plot. Synthetic results are **unit-test evidence only**, never presented as real-world results.
- The **classical tracker** (Track A) is a real, separate perception backend. It is never called "SAM" and never presented as a Meta integration.
- **No mocked Meta responses.** Track B code is only marked 🧪/✅ after real API calls whose raw responses are saved to disk. Interfaces may be defined ahead of time, but an interface without a real backend is reported as "not integrated".
- **Synthetic-generator independence:** the generator integrates physics by brute-force fine time-stepping (`decompile/synth/`). The inference simulator (`decompile/physics/`) is event-driven and analytic, so fitting is never tested against an identical copy of the generator's own model.

## Shared interfaces (Track A defines them, Track B plugs in)

Each interface is created in the task that first needs it, not ahead of time.

| Interface | Track A implementation | Track B implementation |
|---|---|---|
| `Tracker.track(video, prompts) → TrackSet` (per-frame timestamp, object ID, box, centroid, mask area) | `ClassicalTracker` (threshold + contours) | `Sam31Tracker` |
| Scene reader (object names, size and restitution priors) | user-supplied JSON (labelled as manual input) | Muse Spark |
| Residual critic (propose model changes) | none; fixed candidate list scored by numbers | Muse Spark |
| Appearance asset (sprite) | crop from an observed frame | Muse Image |

---

## Phase 0: Foundations

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M0.1 | Repo setup, ROADMAP, environment check | none | P0 | ✅ |

## TRACK A: API-independent development

### A-I. Video input layer

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A1 | Video ingest and validation: ffprobe metadata, SHA-256, full decode check, explicit frame-count reconciliation, accept/reject rules, CLI | M0.1 | P0 | ✅ |
| A2 | Frame extraction: ffmpeg decodes **every** frame (passthrough, no duplication or drop), count matches ffprobe | A1 | P0 | ⬜ |
| A3 | Frame timestamps and metadata: per-frame presentation timestamps, variable-frame-rate detection, slow-motion capture/playback mismatch flag | A2 | P0 | ⬜ |

### A-II. Synthetic ground truth (labelled SYNTHETIC)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A4 | Ground-truth physics generator: ball under gravity, bouncing on a floor with restitution e, optional air drag; brute-force fine stepping; analytic sanity tests | M0.1 | P0 | ⬜ |
| A5 | Synthetic video renderer: GT trajectory → mp4 at a set fps and px/m scale, optional blur and noise; writes a `ground_truth.json` manifest | A4, A1 | P0 | ⬜ |

### A-III. Perception (classical backend)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A6 | `Tracker` interface + `TrackSet` data structure + `ClassicalTracker` on synthetic videos; centroid error vs GT | A3, A5 | P0 | ⬜ |

### A-IV. Experimental protocol

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A7 | Run directory + split by time → `observed.mp4` and `vault/withheld.mp4`; frame-conservation test | A3 | P0 | ⬜ |
| A8 | Leak guards: inference code cannot reach the vault (static test + runtime check) | A7 | P0 | ⬜ |

### A-V. Physics inference

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A9 | Scene IR (Body, Gravity, PlaneContact, LinearDrag) + event-driven deterministic simulator, analytic tests | M0.1 | P0 | ⬜ |
| A10 | Initial estimates: bounce detection, per-flight parabola fits → g (px/s²), e, floor height | A6, A9 | P0 | ⬜ |
| A11 | Global fit through the simulator (multiple shooting, robust loss); parameter-recovery tests across noise levels | A10 | P0 | ⬜ |
| A12 | Calibration px→m from known object size; recovered g in m/s² vs 9.81 | A11 | P0 | ⬜ |

### A-VI. Prediction and evaluation

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A13 | Forward prediction + commit step: `prediction.json` and its SHA-256 written before reveal | A11, A8 | P0 | ⬜ |
| A14 | Evaluator (sole vault reader): hash check, error vs horizon, bounce-timing error, apex error | A13 | P0 | ⬜ |
| A15 | Baselines: constant velocity, constant acceleration (last N), polynomial | A14 | P0 | ⬜ |
| A16 | First full synthetic experiment recorded here (labelled SYNTHETIC) | A15 | P0 | ⬜ |

### A-VII. Reconstruction and program output

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A17 | Compiler: IR → standalone readable Python scene program; output matches the simulator to within 1e-9 | A9 | P0 | ⬜ |
| A18 | Reconstruction render: background plate (observed frames only) + sprite compositor | A7, A6 | P1 | ⬜ |
| A19 | Reveal video: prediction vs reality overlay + error trail | A14, A18 (or plot-only fallback) | P0 | ⬜ |

### A-VIII. Generalisation and science

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A20 | Model selection: nested holdout inside the observed portion + BIC (drag vs no-drag) | A11 | P0 | ⬜ |
| A21 | Parameter uncertainty → prediction bands | A11 | P1 | ⬜ |
| A22 | Pendulum: generator, IR, simulator, fit (pivot, L, damping) | A9, A20 | P0 | ⬜ |
| A23 | Learning curve over split ratios {50, 60, 70, 80}% (set in advance) | A15 | P0 | ⬜ |
| A24 | Real-footage pipeline on dev set (classical tracker) | A16, D2 | P0 | ⬜ |
| A25 | Transfer test: ball video A → ball video B | A24 | P1 | ⬜ |
| A26 | Frozen test-set run, once, results recorded unedited | all P0 above, D3 | P0 | ⬜ |
| A27 | Counterfactual parameter edits ("moon gravity"), re-simulate | A17 | P1 | ⬜ |

### A-IX. Product

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| A28 | FastAPI backend serving run directories | A7 | P0 | ⬜ |
| A29a | UI: Observe / Discover | A28 | P0 | ⬜ |
| A29b | UI: Predict / Reveal | A29a | P0 | ⬜ |
| A29c | UI: Challenge (counterfactuals) | A29b, A27 | P1 | ⬜ |
| A30 | Demo script, recording, README, write-up, submission | A26 | P0 | ⬜ |

## TRACK B: Meta API integrations (⛔ blocked: no API key yet)

Every B task saves raw API responses to disk as evidence. No mocks count as integration.

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| B1 | Auth: minimal text call to `muse-spark-1.3`; record region/account errors verbatim | API key | P0 | ⛔ |
| B2 | SAM 3.1 raw call on one short mp4 (Files API upload, forward propagation), raw stream saved | B1 | P0 | ⛔ |
| B3 | SAM parsing (`meta_sam_parser`) → `TrackSet`; frames returned vs sent | B2, A6 | P0 | ⛔ |
| B4 | `Sam31Tracker` vs `ClassicalTracker` on synthetic (vs GT) and real footage | B3 | P0 | ⛔ |
| B5 | Muse Spark video call (Files API) | B1 | P0 | ⛔ |
| B6 | Muse Spark structured output on video: object noun phrases, identity, size prior | B5 | P0 | ⛔ |
| B7 | Spark → SAM chain (Spark's noun phrases drive SAM) | B6, B3 | P0 | ⛔ |
| B8 | Spark residual critique inside the nested-holdout repair loop | B6, A20 | P0 | ⛔ |
| B9 | Spark-as-oracle baseline (direct coordinate prediction) | B5, A14 | P0 | ⛔ |
| B10 | Ablation: full vs no-Spark vs Spark-only | B8, B9 | P0 | ⛔ |
| B11 | Plain-English edit → parameter change via Spark | B6, A27 | P1 | ⛔ |
| B12 | Muse Image smoke test, then sprite swap ("orange") | B1, A18 | P2 | ⛔ |

**Fallback if the key arrives late or the API is unusable:** self-host SAM 3.1 weights (`facebookresearch/sam3`) on Modal credits. Until then the classical tracker carries the pipeline, clearly labelled.

## Track D: data collection (project owner)

| ID | Task | Needed by | Status |
|---|---|---|---|
| D1 | One quick test clip of a bouncing ball | A24 / B2 | ⬜ |
| D2 | Dev set: 2 ball + 1 pendulum, with diameter and length measured | Nov 2 AM ET | ⬜ |
| D3 | Test set (sealed until A26): 2–3 ball + 2 pendulum | Nov 3 | ⬜ |

---

## Verified results log

Only results that were actually run go here, with date, command and outcome.

### M0.1: Repo setup and environment check (2026-11-01) ✅
- Command: `.venv\Scripts\python tools\check_env.py` → all required checks PASS, exit 0.
- Python 3.13.7 (venv), ffmpeg 7.1.1 and ffprobe 7.1.1 (gyan.dev essentials build), git 2.44.0.
- All 8 pinned packages import at their pinned versions; a fresh venv installs from `requirements.txt` and passes.
- Negative test: with ffmpeg removed from PATH the check reports 2 FAIL lines and exits 1.
- `.env` absent (no API key yet), reported as WARN by design.
- Note: the global pip config lists an unreachable extra index (`pypi.ngc.nvidia.com`). Installs succeed but print retry warnings.
- Committed by owner as `ba75492` and pushed to `github.com/Noel9907/Decompile`.

### A1: Video ingest and validation (2026-11-01) ✅
- Approved by owner 2026-11-01 based on the test results and acceptance evidence below.
- Files: `decompile/ingest.py`, `tests/test_ingest.py`, `decompile/__init__.py`.
- Command: `.venv\Scripts\python -m pytest tests/test_ingest.py -v` → **16 passed** (2.15 s).
- All test media are SYNTHETIC fixtures (ffmpeg `testsrc` / `sine`), generated in pytest temp dirs.
- Decodability: the first video stream is fully decoded to a null sink (`-fps_mode passthrough`). The decoded count is the only authoritative frame count; the container's `nb_frames` is kept for comparison only.
- Observed: a faststart mp4 truncated to 60% of its bytes still passes ffprobe (reports 60 frames), but only **17** decode, with `partial file` / `Invalid NAL unit size` errors. It is rejected, status `mismatch`.
- Observed: webm (VP9) reports no `nb_frames` → status `reported_missing`, decoded 60, accepted with a warning.
- Rejected with specific reasons: text renamed .mp4, 1 KB truncated mp4, audio-only mp4, 0.2 s clip, 32×32 video, h264-in-mkv (unsupported container).
- Known limitation: decoding is O(video length). Fine for the short clips we target; there is a 300 s timeout.
- Not yet tested on real phone footage (no D1 clip yet).
