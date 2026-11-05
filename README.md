# Decompile — turn reality into executable physics

**Decompile is a decompiler for real-world video.** Give it a phone clip of a bouncing ball. It returns the physics
that produced the motion as **source code that runs**. It **predicts what happens next**, and it lets you rewrite
reality: *"what if this happened on the Moon?"*

```
video ──▶ Meta SAM 3.1 ──▶ sub-pixel geometry ──▶ physics fit ──▶ decompiled program ──▶ Muse Spark "what if" ──▶ simulator
          (segment + track)  (centre, size, σ)     (3-D, perspective)  (runnable Python)      (plain English → parameters)
```

From a 2.5 s phone video (real footage, live SAM 3.1 call through the web app, $0.015):

| Recovered from pixels | Value |
|---|---|
| Bounciness (coefficient of restitution) | **0.681**: the ball keeps 68 % of its speed at the bounce |
| Bounce time | 1.436 s (inside measured frames 39–44) |
| Gravity | 142 ball-diameters/s², which implies a **6.9 cm** ball if g = 9.81 m/s² |
| Fit to the 22 frames it saw | **0.73 px RMS** (0.82 σ), adequacy gate passed |
| **Prediction of frames it never saw** | next frames within **6–8 px**; next bounce predicted at 1.899 s, **33 ms (one frame)** from where the real ball lands |

## Meta models, each doing a distinct job

| Model | Job in Decompile |
|---|---|
| **SAM 3.1** (Meta Model API) | Sees: segments and tracks the object in every frame of the uploaded video (identity + support region) |
| **Muse Spark** (Meta Model API) | Reasons: (1) reviews each fit like a physicist (explains the numbers, sanity-checks the implied size, critiques weak evidence); (2) turns plain-English what-ifs ("on the Moon", "no floor", "throw it twice as hard") into program edits through strict structured output, and refuses what the program cannot represent (wind, spin) |
| Decompile (this repo) | Computes: sub-pixel measurement, perspective-aware physics fit, executable program, deterministic simulation, held-out prediction |

Not used (and not claimed): Muse Image, Muse Wonder, open-weight models.

The decompiled program is real Python against the Decompile engine. It is generated from the fitted scene
program and tested to reproduce the engine's simulation **bit for bit**:

```python
program = pp.ProjectiveProgram(
    t0=Fraction(77951, 90000),          # s, first measured frame
    p0=(1.9626, 0.9962, 13.0951),       # start position (ball diameters)
    v0=(1.0607, -28.2631, -16.7539),    # start velocity (D/s)
    g=(2.386, 121.233, 74.477),         # gravity, |g| = 142.3 D/s^2
    restitution=0.6811,                 # measured: bounce in frames 39-44
    ...
)
sim = pp.simulate(program, times)
```

## How it works

1. **Perceive — Meta SAM 3.1.** The video bytes are uploaded to the Meta Model API. SAM 3.1 segments and tracks the
   object in every frame, decoded with the official `meta_sam_parser` and checked against the exact frame table.
2. **Measure — Decompile.** SAM gives identity and support. The centre, apparent diameter and a per-frame 1σ come from
   a sub-pixel outline fit in the original pixels. Frames that cannot be measured are flagged, **never interpolated**.
3. **Decompile — physics.** A perspective-aware fit uses the apparent diameter as a depth cue. It recovers 3-D gravity,
   the contact and the contact law, gated by model comparison and an adequacy test. Output: an executable scene
   program with exact (rational) timestamps.
4. **Rewrite — Meta Muse Spark.** Ask in plain English. Muse Spark maps the question onto the parameters the program
   actually has (restitution, gravity, horizon) with strict structured output, or **refuses honestly** ("strong wind
   needs drag, which this program cannot represent"). The deterministic simulator then computes what would have
   happened, and the UI shows the edit as a code diff.

## Run it

```powershell
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-web.txt
cd frontend; npm install; npm run build; cd ..
$env:DECOMPILE_LIVE_SAM = "1"           # allow (confirmed, cost-capped) SAM 3.1 calls; key in .env as MODEL_API_KEY
.venv\Scripts\python -m decompile_web   # http://127.0.0.1:8000
```

Details: [`README_WEB.md`](README_WEB.md) (web app, safety, cost controls) · [`web_API.md`](web_API.md) (HTTP API) ·
[`ROADMAP.md`](ROADMAP.md) (research log).

## Why trust the numbers

* **Every number traces to pixels and frame timestamps.** Exact PTS from FFprobe, per-frame σ, rejected frames shown
  with reasons.
* **Fitted vs predicted are never conflated.** The web app reports a *fitted reconstruction* (a consistency check). A
  separate CLI protocol (`tools/real_video_demo.py`) pre-registers a clip, freezes a prediction from the first 70% of
  frames, and only then reveals the withheld 30%.
* **Hypotheticals are labelled.** Counterfactuals are orange, dashed and marked hypothetical everywhere, including in
  rendered videos and exported JSON.
* **Paid calls are explicit.** SAM requests show the exact cost (with the 50-frame billing minimum) and need
  confirmation. They are never retried after being sent, and actual billed usage is recorded. The API key never
  leaves the server.
* **Tests.** 1,100+ Python tests (offline, network blocked), including bit-exact regression guards, plus web backend
  and UI tests.

## Why only a bouncing ball?

On purpose, for now. A bouncing ball has one measurable truth, so Decompile can *prove* the loop works: it fits one
bounce, then predicts the next one before seeing it. The same five stages (segment, measure, fit, program,
simulate) carry over to richer bodies. The next targets are pendulums, sliding and rolling objects, multi-body
collisions and a moving camera.

## Scope and limits (today)

One rigid body, one bounce in the fitted window, a static camera. The focal length comes from a field-of-view prior,
not a calibration, so the metric scale is inferred. The projective simulator does not model resting contact; the UI
hides samples after rest. The held-out comparison uses Decompile's own measurements of the later frames, not
independent ground truth.
