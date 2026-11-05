# Council transcript — Decompile, 2026-11-05 04:20 IST

## Original question
The founder: "Let's have a real council meeting… the point is to really stand out and win… only a bouncing ball seems too narrow, what else? We need to be 1st in the world."

## Framed question
QUESTION: What should "Decompile" do in the next ~2 hours of engineering (before a 45-minute demo-video recording and a hard 07:30 IST submission deadline; it is now 04:10 IST) to maximise the chance of placing FIRST in the Meta Global AI Developer Hackathon? The founder says a bouncing ball is too narrow: "we need to be 1st in the world... only a bouncing ball seems too narrow, what else?" Answer that question concretely.

EVENT: Meta Global AI Developer Hackathon, $1M prize pool, up to 20,000 accepted entrants, remote. Indie track judged on "creativity, technical ambition, and whether it works". Theme "Push the Frontier": "give a machine a new capability ... unlock a better way to understand and control a physical system". Judges mostly watch a 2-3 min demo video and skim the repo; live finals for top entries on Nov 9. Meta models available: SAM 3.1 (video segmentation/tracking), Muse Spark (reasoning LLM, strict JSON output), Muse Image, Muse Wonder, open-weight models. $140 of API credit left (SAM costs ~$0.015 per short clip).

WHAT EXISTS AND WORKS (live-verified tonight, 24 web tests + 1100+ engine tests):
- Web app (React + FastAPI, light mode): upload MP4 -> cost-confirmed SAM 3.1 tracking -> auto-suggested window -> "Decompile".
- Engine: sub-pixel ellipse outline measurement of a ROUND object from SAM masks (per-frame sigma; bad frames flagged), perspective-aware 3-D physics fit using apparent size as depth (gravity vector, bounce time, restitution, contact law), model comparison + adequacy gates, executable scene program, deterministic simulator.
- Result on a real 2.5 s phone video of a ball: bounciness 0.681, 6.9 cm ball implied, fit 0.73 px; fitted on frames 26-54 it PREDICTED unseen frames 55-56 within 6-8 px and the next bounce within 33 ms (one frame) of reality.
- Fitted physics shown as runnable Python (reproduces engine exactly).
- Muse Spark: (a) reviews each fit like a physicist (explains, sanity-checks size, critiques evidence, confidence); (b) translates plain-English what-ifs (Moon, Jupiter, no floor, throw 2x harder, superball, heavier = no change) into program edits; refuses wind/spin honestly. Simulation of the alternate reality plays next to real footage with a code diff.
- Engine also contains (unused by the web app): a 2-D kinematic analysis with BIC model selection over constant-velocity, constant-acceleration, linear-drag, and one-contact/impulse models, working on any tracked point (e.g. SAM mask centroids) -- so motion of non-ball objects (walking person = ~constant velocity, sliding/skiing = constant acceleration/drag, thrown object in flight = constant acceleration) could be classified and quantified honestly, but WITHOUT a 3-D program.
- The founder can film new phone clips in minutes (a pendulum on a string, a rolling ball, a toss, a person walking, a toy car).

LIMITS: physics program only for one round body, one bounce, static camera. No pendulum/rolling/collision programs. Muse Spark vision (image input) is unverified. Muse Image unused.

WHAT'S AT STAKE: a first-place finish among up to 20,000 entrants vs. a respectable top-10% that wins nothing. A demo that breaks or overclaims would hurt credibility ("whether it works").

ASK: Critique hard, then give a concrete plan for the next 2 hours: what single capability or story change would most plausibly make this #1, how to broaden beyond a bouncing ball honestly (which object/motion, which model, which clip to film), what to cut, and the exact first thing to do.


## Advisor responses (anonymized as presented to reviewers)
Response A:
Stop framing this as "bouncing ball vs. broader." The upside you're undervaluing: you already built a falsifiable loop — fit → predict unseen frames → next bounce within one frame (33 ms). Nobody else in 20,000 entries will have a prediction that gets checked against reality. Make that the headline: "Decompile turns video into a physics program that predicts the future — and you can edit the program to change reality."
Next 2 hours:
1. One second object, not more — 45 minutes max: pendulum or tossed ball in flight. Use the existing 2-D BIC kinematic engine on SAM centroids. Film a toss and a person walking. Show the engine selecting the model automatically: ball = constant acceleration (gravity), walker = constant velocity. Add a slide-in-hallway clip for drag. Label it honestly: "2-D classification, no 3-D program." Wire it into the UI as a "motion type" card. That claim — "any tracked object" — is true for exactly the motions you tested.
2. Held-out prediction scoreboard across 4–5 clips (bounce, toss, walk, whatever you film): report prediction error in px and frames. A multi-clip table is the proof of generality judges can read in 5 seconds.
3. Muse Spark what-if on the emotional beat: "What if this were on Jupiter?" → simulated ball beside real footage + code diff.
4. Frontier pitch line: a camera becomes a sensor that returns editable, testable physical laws; adjacent uses (sports analytics, robotics testing, physics education) in one voiceover sentence.
Cut: Muse Image, unverified vision, pendulum/rolling programs, UI polish beyond necessity.
First step: film the toss + walk clips now (10 min), run through tracking, see if BIC picks the right model. If yes, wire the card. If no, drop it and spend the time on the demo script. Record by 06:15; 45 minutes buffer.

Response B:
Critique: "Broaden beyond a bouncing ball" is the wrong 2-hour move. 3h20m to deadline, a 45-min recording, and a pipeline that's honest because it's narrow. Adding a pendulum or rolling program now = new physics + new gates + new tests = the most likely way to ship a broken demo, under a criterion that literally says "whether it works." Overclaiming costs more than narrowness. The real weakness is story, not scope: judges remember "it predicted the next bounce within one frame on an unseen part of a real phone video."
Plan with clock: 1) (0:00–0:10) Film 3 more bounce clips — different ball, different surface, one handheld "tilted camera" you expect to fail — run all three through the app. Zero new engineering; generalization + honest failure handling. 2) (0:10–1:00) Wire the unused kinematic BIC analysis into the web app as a "Not a bouncing ball?" path; film one toss and one person walking; app says "constant acceleration, 9.6 m/s² implied" or "constant velocity, no forces" with BIC evidence, no 3-D program. Kill if stalled past 45 min. 3) (1:00–1:30) Muse Spark explains that path's result and refuses what it can't know (no image input — unverified). 4) (1:30–2:00) Rehearse, fix only breaks, freeze.
Cut: Muse Image, pendulum/rolling programs, new what-if types.
Story: real clip → fit predicts unseen frames → Moon/Jupiter what-ifs next to real footage → toss + walk analyzed honestly → "this is where it says no." Frame it as a system that knows what it can and can't explain. Reality check: 1st of 20,000 is a long shot either way — spend effort on flawlessness, not breadth.

Response C:
The fatal flaw: you're asking "what else?" when the real risk is breaking what already works. At 04:10 with a 07:30 deadline, any new physics program — pendulum, rolling, collision — is a new fit, new adequacy gates, new failure modes, built unverified at 5am, for a demo whose criterion is "whether it works." A half-working second domain is worse than one flawless one. Judges who see a demo glitch will remember it; they won't remember you tried to broaden.
"1st in the world" is a framing trap. Among 20,000 entrants, breadth of object types won't win; memorable story + verified credibility will. Your strongest asset is the held-out prediction: fitting 26–54 and predicting the next bounce within one frame is falsifiable science. Almost no hackathon demo does that. Lead with it: "we predicted the future of real footage, then let you rewrite physics in English." A bouncing ball is not too narrow if it ends with a ball bouncing on the Moon next to the real clip.
Cut: the 2-D BIC analysis as a headline (no 3-D program — a classification label, not control); Muse Image; any unverified Spark vision claim; any "works on anything" language — overclaiming is your biggest exposure.
Honest broadening, 45 minutes max: film 2 more ball clips (a toss in flight; a different ball, tennis vs superball) through the existing pipeline. Showing the fit recovers different restitution values is a real generalization claim with no new code. Optional: one short non-ball segment (walker → constant velocity via BIC), labelled kinematics-only, only if < 20 minutes and zero risk to the main path.
First step: freeze code. Film and run the two extra clips; record failures honestly as stated limitations. Rehearse the script before 06:00 so recording ends by 07:00; 30-minute upload margin minimum.

Response D:
Outsider view: a judge scrolling hundreds of videos will not grasp "sub-pixel ellipse outline, BIC model selection, perspective-aware 3-D fit." They'll grasp one sentence: "Film anything. It writes the physics as code, predicts the future, and lets you rewrite reality." Your pitch is buried under jargon. The ball isn't too narrow — your explanation of it is too narrow and too technical.
Critique: the strongest moment is the prediction (fit 26–54, called the next bounce within one frame) — the proof "it works" — but it sits beside a mass of engine detail. A newcomer can't tell why a bouncing ball matters. Say what it's for: any camera becomes a physics instrument — no sensors, no manual tuning.
Don't build pendulum/rolling/collision programs in 2 hours; that's how demos break at 07:00, and overclaiming loses "whether it works."
Broaden honestly with what exists: the unused 2-D kinematic BIC analysis on a second clip (person walking or a thrown object): "the system decides what kind of motion this is — constant speed, accelerating, drag, bounce — and says so honestly." Two or three clips in the same tool, one non-ball labelled "classified and quantified, no 3-D program yet," is breadth without new physics.
Cut: Muse Image, wind/spin attempts, any feature you can't live-verify.
Plan: 1) 20 min: film a thrown object or walking person, run SAM + kinematic analysis, check sanity; if not sane, drop. 2) Wire it into the web app as a minimal "Motion type" result. 3) Rewrite the video script for a stranger: prediction → what it's for → what-if (Moon) → broader clip → code. 4) Record by 06:15, submit by 07:00. First: film the second clip right now and run it.

Response E:
You're asking the wrong question. "What else besides a bouncing ball?" is a breadth question; judges at 20,000 entries don't reward breadth, they reward one thing they can see and trust. Breadth you can't back up reads as overclaiming, and "whether it works" is a stated criterion.
What you actually have is rare: a closed loop — film → fit → predict an unseen future → change physics → re-simulate. The prediction is the asset (fit 26–54, next bounce within one frame; falsifiable; almost no hackathon demo has it). The story: "A camera becomes a physics lab, and the proof is that it predicts the future it never saw." The ball isn't too narrow; it's the instrument that proves the claim.
Critique: your risk is not narrowness, it's a polished "ball app" that looks like a school assignment. Pendulum/rolling programs in 2 hours = new physics, new gates, new bugs = likeliest way to break the demo. Unverified Spark vision and Muse Image are traps.
Plan: 1) Broaden via the existing kinematic BIC module, not new programs: film a toss (constant acceleration) and a walking person/toy car (constant velocity); show the system choosing the right model and saying "no bounce model, here's why" — honest model selection across objects; ~30 min behind a clearly labelled "motion classifier" panel. 2) Keep the ball as hero + 3-D program; lead the video with the held-out prediction overlay. 3) Add one deliberate failure: a clip the system rejects (non-round object or wind) with the adequacy gate refusing — credibility beats any feature. 4) Cut Muse Image, vision; freeze at 06:00. 5) Record 06:15 latest; submit by 07:00. First: film toss + walk now, run through the BIC module in a script; if it misclassifies, drop breadth and spend the time on the demo script.


## Peer reviews
ANONYMIZATION MAP: A = The Expansionist, B = The Executor, C = The Contrarian, D = The Outsider, E = The First Principles Thinker.

FACILITATOR FACT (given to reviewers): a 5-minute test of the existing 2-D kinematic BIC module on raw SAM 3.1 mask centroids from the real ball clip gave noisy/inconclusive selection: free flight (26-42) -> "linear_drag_impulse", adequacy not_assessable; full window (26-55) -> "impulse_uniform_acceleration", adequacy "inadequate", all verdicts inconclusive; post-bounce (44-56) -> "impulse_uniform_acceleration".

Reviewer 1: Strongest B (clock plan, 45-min kill switch, honest "says no", admits long shot); E close. Blind spot A: assumes BIC classifies cleanly and proposes a multi-clip held-out table the non-bounce clips can't support. All missed: nobody ran the BIC test first; BIC needs sub-pixel points or a noise estimate (engineering, not wiring); keep BIC "inconclusive" as an honest-refusal beat; segmentation failure risk on new clips; Nov 9 finals prep. Recommends B with breadth demoted to a labelled "refuses honestly" segment.

Reviewer 2: Strongest E (names the asset, ball as hero, labelled BIC panel, deliberate rejected clip, kill switch, freeze times); B close. Blind spot A (assumes outcomes, scope creep, deadline margin). All missed: verify BIC on real centroids first; feed BIC the sub-pixel ellipse centres instead of mask centroids; fallback breadth story if BIC unreliable (refusal demo or second ball clip with different restitution); submission artifacts and finals reliability.

Reviewer 3 (first-place lens): Strongest E (A close). Blind spot B and C: optimise for safety, concede the win, ignore that "Push the Frontier" rewards ambition. All missed: BIC unreliable on raw centroids -> smooth/use ellipse points or drop it; cheap honest breadth = more round-object clips through the verified pipeline + multi-clip held-out prediction table; API failure during recording / pre-recorded fallback; README and submission text.

Reviewer 4 (judge perception lens): Strongest B (E close; D only one reasoning about judge perception). Blind spot A. All missed: BIC unreliable -> honest broadening should be a second ball-physics result (toss through the 3-D fit, different ball with different restitution), not a classifier panel; nobody planned the first 15 seconds of the video (prediction-vs-reality overlay as the hook); no backup take; submit early and check the form; "within one frame" is a single-clip, single-sample claim -> run the held-out test on 3 clips and report the spread so it isn't cherry-picked.

Reviewer 5 (execution/fatigue lens): Strongest B (E second). Blind spot A. All missed: every plan makes BIC the broadening move though it's noisy -> wiring it would show "inconclusive" on camera; decide the fallback now (proven ellipse pipeline only, or BIC "inconclusive" as an honest-refusal beat); no safety net (pre-recorded fallback, tagged commit, run instructions for Nov 9); founder fatigue at 4 a.m. (one tired person films, debugs, records, writes) -> stop new work before 06:00; check submission format before 07:00.


## Chairman's verdict
1. WHERE THE COUNCIL AGREES — The asset is the held-out prediction (fit 26-54, next bounce within one frame). Do not build pendulum/rolling/collision programs tonight; cut Muse Image and unverified Spark vision. Overclaiming is the biggest risk ("whether it works"). The ball is not too narrow; the explanation of it is.
2. WHERE THE COUNCIL CLASHES — Breadth via the 2-D BIC classifier (A, D, E) vs safety (B, C). The facilitator's test settles it: BIC on raw SAM centroids was inconclusive or wrong even on the ball clip; wiring it now means showing "inconclusive" on camera. Safety vs ambition: pure safety concedes first place, and one-clip evidence looks cherry-picked. Ruling: be ambitious in the CLAIM and EVIDENCE, not the SCOPE — broaden by proving generality inside the pipeline that works.
3. BLIND SPOTS — Nobody tested BIC before recommending it (it failed). "Within one frame" is n=1; it needs a spread across clips. Nobody designed the first 15 seconds. No safety net (backup take, tagged commit, API-failure fallback, Nov 9 run instructions). Founder fatigue: stop new work well before 06:00.
4. RECOMMENDATION — New story: "Film any round object, on any surface — Decompile writes its physics as a program that predicts frames it never saw, verified across N real clips, and refuses what it can't explain."
04:20-04:35 founder films 5 clips: tennis ball, superball, basketball on carpet, a high toss with one bounce, and one clip that SHOULD be rejected (non-round object or handheld camera).
04:35-05:15 engineer runs all through the existing pipeline with held-out final frames (kill switch: drop any clip that fails segmentation; never debug). Output: scoreboard — clip, restitution, prediction error px, bounce-time error ms; failures listed as "refused (reason)". If fewer than 2 new clips pass, the story becomes "the gates said no, honestly".
05:15-05:45 engineer: scoreboard view + ghost overlay drawing the predicted path before the real frames play (kill switch 05:35: fall back to a screen-recorded side-by-side); README + limits. Founder: stranger-proof script — 0-15 s ghost predicts the bounce, real ball lands on it; scoreboard; Moon/Jupiter what-if with code diff; refusal clip; close "a camera becomes a physics instrument".
05:45 code freeze, tag commit. 05:50-06:30 record two takes, pre-record SAM/Spark calls as fallback. 06:30-07:00 edit + submit (check form fields early). 07:00-07:30 buffer only.
BIC: cut from the UI; one roadmap sentence at most, unless clip runs finish early and BIC on ellipse centres cleanly classifies a toss by 05:00.
5. THE ONE THING TO DO FIRST — The founder films the five clips now. Their results decide the story.
