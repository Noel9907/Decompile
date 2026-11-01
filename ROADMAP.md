# DECOMPILE — Roadmap

Workflow: **PLAN → IMPLEMENT ONE SMALL TASK → TEST → VERIFY → APPROVE → NEXT TASK.**

Status legend:
- ⬜ not started
- 🔨 implemented: code exists, not yet tested
- 🧪 tested: automated or scripted checks pass
- ✅ verified: acceptance criteria passed on real inputs **and** approved by the project owner

Priority: **P0** submission fails without it · **P1** makes it strong · **P2** nice to have.

Deadline: submission due **Nov 4, 21:00 ET**. Internal: feature freeze Nov 4 12:00 ET, submit by 18:00 ET.

---

## Phase 0: Foundations

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M0.1 | Repo setup, ROADMAP, environment check (Python, ffmpeg/ffprobe, pinned deps) | none | P0 | 🧪 |
| M0.2 | Run-directory convention (`runs/<id>/manifest.json`), file hashing helper, test | M0.1 | P0 | ⬜ |

## Phase 1: Perception and experimental protocol (M1)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M1.1 | Meta API auth: minimal text call to `muse-spark-1.3` | M0.1 | P0 | ⬜ |
| M1.2 | SAM 3.1 raw call on one short mp4, raw stream saved | M1.1 | P0 | ⬜ |
| M1.3 | SAM parsing → per-frame table; frames returned vs frames sent | M1.2 | P0 | ⬜ |
| M1.4 | Overlay video, trajectory plot, centroid jitter | M1.3 | P0 | ⬜ |
| M1.5 | Muse Spark video call (Files API upload) | M1.1 | P0 | ⬜ |
| M1.6 | Muse Spark structured output on video | M1.5 | P0 | ⬜ |
| M1.7 | Muse Image smoke test | M1.1 | P2 | ⬜ |
| M1.8 | Ingest: per-frame timestamps, variable-frame-rate/slow-motion flags, SHA-256 | M0.2 | P0 | ⬜ |
| M1.9 | Split by time → `observed.mp4` + `vault/withheld.mp4`; frame conservation test | M1.8 | P0 | ⬜ |
| M1.10 | Leak guards (vault isolation, SAM forward-only on observed clip) | M1.9, M1.3 | P0 | ⬜ |
| M1.11 | Spark → SAM chain; M1 findings and go/no-go | M1.6, M1.10 | P0 | ⬜ |

## Phase 2: Physics core (M2). No API dependency

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M2.1 | Scene IR dataclasses (Body, Gravity, PlaneContact, Pendulum, LinearDrag) | M0.1 | P0 | ⬜ |
| M2.2 | Event-driven simulator: ballistic + plane bounce, analytic impacts | M2.1 | P0 | ⬜ |
| M2.3 | Pendulum + damping simulator, analytic tests | M2.1 | P0 | ⬜ |
| M2.4 | Synthetic trajectory generator (unit-test data only) | M2.2 | P0 | ⬜ |
| M2.5 | Event-segmented initial estimates (bounces, g, e) | M2.4 | P0 | ⬜ |
| M2.6 | Global multiple-shooting fit, robust loss, parameter-recovery tests | M2.5 | P0 | ⬜ |
| M2.7 | Model selection: nested holdout + BIC | M2.6, M2.3 | P0 | ⬜ |
| M2.8 | Compiler: IR → standalone readable program, equivalence test | M2.2 | P0 | ⬜ |
| M2.9 | Parameter uncertainty → confidence intervals, prediction bands | M2.6 | P1 | ⬜ |

## Phase 3: First real predictive experiment (M3)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M3.1 | Measurement: centroid, radius, timestamps, gaps | M1.4 | P0 | ⬜ |
| M3.2 | Calibration px→m; plane-of-motion check | M3.1 | P0 | ⬜ |
| M3.3 | Floor geometry from bounce minima | M3.1 | P0 | ⬜ |
| M3.4 | End-to-end fit on a real observed clip → IR + program | M2.7, M2.8, M3.2, M3.3 | P0 | ⬜ |
| M3.5 | Commit step: `prediction.json` + SHA-256 before reveal | M3.4, M1.10 | P0 | ⬜ |
| M3.6 | Evaluator (only vault reader): hash check, ground truth, metrics vs horizon | M3.5 | P0 | ⬜ |
| M3.7 | Baselines: constant velocity, constant acceleration, polynomial | M3.6 | P0 | ⬜ |
| M3.8 | First experiment recorded here with plots | M3.7 | P0 | ⬜ |

## Phase 4: Muse Spark as the scientist (M4)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M4.1 | Scene reading → SAM prompts + labelled priors | M1.11, M3.4 | P0 | ⬜ |
| M4.2 | Residual critique → structured model change | M2.7 | P0 | ⬜ |
| M4.3 | Repair loop inside the nested holdout, accept/reject log | M4.2 | P0 | ⬜ |
| M4.4 | Spark-as-oracle baseline | M3.6 | P0 | ⬜ |
| M4.5 | Ablation: full vs no-Spark vs Spark-only | M4.3, M4.4 | P0 | ⬜ |
| M4.6 | Spark-written commented program, numerical equivalence test | M2.8 | P1 | ⬜ |

## Phase 5: Rendering and reveal (M5)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M5.1 | Background plate from observed frames only | M1.9, M1.3 | P1 | ⬜ |
| M5.2 | Sprite + deterministic compositor → predicted video | M5.1, M3.5 | P1 | ⬜ |
| M5.3 | Reveal video: side-by-side + overlay + error trail | M5.2 (or fallback), M3.6 | P0 | ⬜ |

## Phase 6: Generalisation (M6)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M6.1 | Real pendulum: pivot, length, damping | M3.4, M2.3 | P0 | ⬜ |
| M6.2 | Automatic model-structure selection | M6.1, M4.3 | P1 | ⬜ |
| M6.3 | Learning curve over split ratios {50, 60, 70, 80}% (set in advance) | M3.7 | P0 | ⬜ |
| M6.4 | Transfer test: ball video A → ball video B | M3.7 | P1 | ⬜ |
| M6.5 | Frozen test-set run, once, results recorded unedited | all above | P0 | ⬜ |

## Phase 7: Counterfactuals (M7)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M7.1 | Plain-English edit → parameter change → re-simulate | M3.4 | P1 | ⬜ |
| M7.2 | Muse Image sprite swap | M5.2, M1.7 | P2 | ⬜ |

## Phase 8: Product and submission (M8)

| ID | Task | Depends | Pri | Status |
|---|---|---|---|---|
| M8.1 | FastAPI backend serving run directories | M0.2 | P0 | ⬜ |
| M8.2a | UI: Observe / Discover | M8.1 | P0 | ⬜ |
| M8.2b | UI: Predict / Reveal | M8.2a | P0 | ⬜ |
| M8.2c | UI: Challenge (counterfactuals) | M8.2b, M7.1 | P1 | ⬜ |
| M8.3 | Demo script and recording | M6.5 | P0 | ⬜ |
| M8.4 | README, technical write-up, submission | M6.5 | P0 | ⬜ |

## Phase 9: Stretch

Two-body collision with mass-ratio inference; uncertainty calibration plot.

## Track D: data collection (project owner)

| ID | Task | Needed by | Status |
|---|---|---|---|
| D1 | One quick test clip of a bouncing ball | M1.2 | ⬜ |
| D2 | Dev set: 2 ball + 1 pendulum, with diameter and length measured | Nov 2 AM ET | ⬜ |
| D3 | Test set (sealed until M6.5): 2–3 ball + 2 pendulum | Nov 3 | ⬜ |

---

## Verified results log

Only results that were actually run go here, with date, command and outcome.

### M0.1: Repo setup and environment check (2026-11-01)
- Command: `.venv\Scripts\python tools\check_env.py`
- Python 3.13.7 (venv), ffmpeg 7.1.1 and ffprobe 7.1.1 (gyan.dev essentials build), git 2.44.0.
- All 8 pinned packages import at their pinned versions.
- Note: the global pip config lists an unreachable extra index (`pypi.ngc.nvidia.com`). Installs still succeed but print retry warnings.
- Status: 🧪 tested, waiting for owner verification.
