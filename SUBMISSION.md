# Decompile: submission kit

## One-liner
Decompile turns a phone video into the physics program behind it. It predicts what happens next, and lets you ask "what if?"

## Short description (~70 words)
Decompile is a decompiler for reality. Meta SAM 3.1 tracks an object through a phone video. Decompile measures it
with sub-pixel precision, fits perspective-aware physics and emits it as runnable code. Fitted on one bounce, it
predicted the next bounce to within one frame. Muse Spark reviews every fit like a physicist and turns questions like
"what if this happened on the Moon?" into program edits that a deterministic simulator plays out.

## Full description
**Problem.** AI can describe a video, but not the laws behind it. Robots, simulators and scientists need the
executable model: gravity, the bounce, the contact law, with uncertainty, so they can predict and ask
counterfactual questions. Calibrating a simulator normally takes a lab, motion capture and days. Decompile does it
from a phone video for $0.015.

**How it works.**
1. **Meta SAM 3.1** segments and tracks the object in every frame (Meta Model API).
2. Decompile fits the ball's outline in the original pixels. That gives a sub-pixel centre, an apparent diameter
   (a depth cue) and a per-frame uncertainty. Bad frames are flagged, never invented.
3. A perspective-aware physics fit recovers 3-D gravity, the bounce and the coefficient of restitution, gated by
   model comparison and an adequacy test.
4. The result is **decompiled into runnable Python** that reproduces the motion exactly on the Decompile engine.
5. **It predicts.** Fitted only on frames 26–54, the program predicted the frames after them to within 6–8 px, and
   placed the next bounce **33 ms (one frame)** from where the real ball lands.
6. **Meta Muse Spark** reviews the fit like a physicist: it explains the numbers, sanity-checks the implied 6.9 cm
   ball, and lists the weakest evidence. It also turns plain-English what-ifs (Moon, Jupiter, no floor, throw twice
   as hard, superball) into program edits through strict structured output, and refuses what the program can't
   represent (wind). A deterministic simulator plays out each alternate reality, shown next to the real footage
   with a code diff.

**Real result** (2.5 s phone video, live): bounciness 0.681, bounce at 1.436 s, a 6.9 cm ball implied, fit
0.73 px RMS, and the next bounce predicted within one frame.

**Trust.** Every number traces to pixels and exact frame timestamps. Fitted results, held-out predictions and
hypothetical edits are always labelled separately. Paid calls show their cost and need confirmation. 1,100+
offline tests.

**Built with:** Meta SAM 3.1 and Muse Spark (Meta Model API), Python (NumPy, SciPy, OpenCV, FFmpeg), FastAPI and
React/TypeScript.

**Scope today:** one rigid body and a static camera, chosen deliberately because a ball's motion can be checked.
Next: pendulums, rolling, multi-body collisions and a moving camera.

## Demo video script (2:30), led by the payoff

Setup:
- Start the server with `$env:DECOMPILE_LIVE_SAM="1"; .venv\Scripts\python -m decompile_web`.
- Open http://127.0.0.1:8000 in a 1440-wide window, light mode.
- The `bounce_wall.mp4` project is fully live-computed. Its what-if questions and the Muse Spark review are
  cached, so they answer instantly.

| Time | Screen | Say |
|---|---|---|
| 0:00 | Result page, scroll to **Did it predict the future?**, playing | "I filmed a ball bouncing. Decompile only saw the first bounce. This is its prediction of the rest of the video. It called the next bounce to within one frame." |
| 0:15 | Scroll up: code panel | "Because Decompile doesn't describe video. It turns it into a program you can run. This is that ball, decompiled." |
| 0:30 | Home → upload a fresh clip → "ball" → **Track with SAM 3.1** → cost dialog → confirm | "Here's how. Meta SAM 3.1 tracks the object. It's a paid call, so you see the exact cost (1.5 cents) and approve it." |
| 0:55 | Step 2 height plot, **Decompile** | "It finds the bounce by itself. One click." (Cut to the result.) |
| 1:05 | Metrics + **Muse Spark's review** | "Bounciness 68%, a 6.9 cm ball, fit to under a pixel. Muse Spark reviews it like a physicist, including what it doesn't trust." |
| 1:25 | **What if…** → "What if this happened on the Moon?" | "Now rewrite reality. Muse Spark edits the program; the simulator plays it out next to the real footage." |
| 1:45 | "What if there was no floor?", then "Throw it twice as hard"; point at the `~` line in the code diff | "Every edit is a line of code." |
| 2:00 | "Add a strong wind" | "And when the physics isn't in the program, it says so. No hallucinated physics." |
| 2:10 | Evidence tab (frame strip), then back to Result | "Every number traces to pixels. Today it's one ball, chosen because it can be checked. Next: pendulums, collisions, robots. Decompile: from pixels to programs." |

## Pre-submit checklist
- [ ] Record the video (≤ 3 min), upload it (YouTube unlisted / Loom)
- [ ] Push the repo; `README.md` is the landing page
- [ ] Indie track · Push the Frontier · models: SAM 3.1, Muse Spark
- [ ] Submit before **Nov 4, 9:00 PM ET = Nov 5, 07:30 IST** (aim for 06:30)
