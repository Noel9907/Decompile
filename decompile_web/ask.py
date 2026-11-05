"""Plain-English what-if questions, answered by Muse Spark + the deterministic simulator.

Muse Spark does ONE bounded job: translate the user's question into the parameters the scene program actually
supports (restitution, gravity scale, simulation horizon), or say the question is unsupported. It never produces
trajectories. The returned parameters are validated here, then decompile's simulator runs the counterfactual.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from decompile import meta_api

from .security import redact

PURPOSE = "decompile-web-whatif"
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["supported", "restitution", "gravity_scale", "velocity_scale", "remove_floor", "extend_s", "headline",
                 "reasoning", "assumptions", "unsupported_reason"],
    "properties": {
        "supported": {"type": "boolean"},
        "restitution": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "null"}]},
        "gravity_scale": {"anyOf": [{"type": "number", "minimum": 0.1, "maximum": 3}, {"type": "null"}]},
        "velocity_scale": {"anyOf": [{"type": "number", "minimum": 0.2, "maximum": 3}, {"type": "null"}]},
        "remove_floor": {"type": "boolean"},
        "extend_s": {"type": "number", "minimum": 0, "maximum": 3},
        "headline": {"type": "string", "maxLength": 140},
        "reasoning": {"type": "string", "maxLength": 700},
        "assumptions": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 200}},
        "unsupported_reason": {"anyOf": [{"type": "string", "maxLength": 300}, {"type": "null"}]},
    },
}

DEVELOPER = """You translate a user's "what if" question about a filmed scene into the parameters of an already
fitted, executable physics program. You do not simulate anything yourself; a deterministic simulator runs the
program with the parameters you choose.

The program models ONE body moving under constant gravity and bouncing on a floor. The ONLY parameters you may
change are:
  - restitution: the coefficient of restitution of every bounce, 0..1 (0 = no bounce, 1 = perfectly elastic).
    null = keep the value measured from the video.
  - gravity_scale: multiplies the measured gravity vector, 0.1..3. 1 = the gravity in the video (Earth).
    Reference surface gravities relative to Earth: Moon 0.165, Mars 0.38, Mercury 0.38, Venus 0.90, Jupiter 2.53,
    Saturn 1.07, Neptune 1.14. null = keep the measured gravity.
  - velocity_scale: multiplies the initial velocity at the start of the clip, 0.2..3 ("throw it twice as hard" = 2,
    "gently" = 0.5). null = keep the measured velocity.
  - remove_floor: true removes the floor entirely (the body keeps falling); false keeps it.
  - extend_s: seconds to keep simulating beyond the end of the video, 0..3. Use 1.5 unless the question implies
    otherwise; use more when slower motion (low gravity) needs time to play out.
Typical restitution values: superball 0.85-0.9, basketball ~0.75-0.85, tennis ball ~0.7-0.75, cricket/hard ball
~0.5-0.6, lump of clay or wet sponge 0-0.15, perfectly elastic 1.

Mass does not change the motion in this program (no air drag): if asked to make the object heavier or lighter,
answer supported=true with every parameter unchanged (nulls, remove_floor false) and explain why in reasoning.
If the question asks for anything the program cannot represent (air drag, wind, spin, different shapes, other
objects, collisions between objects, moving or tilted floors, friction changes, changing the camera, replacing
the object's appearance) or needs values outside the ranges, set supported=false, all numeric parameters null,
remove_floor false,
extend_s 0, and explain briefly in unsupported_reason. Never pretend an unsupported effect is simulated.
If supported, unsupported_reason is null. headline: one short sentence describing the change (e.g. "Moon gravity:
0.165 x"). reasoning: 1-3 sentences on how you chose the numbers. assumptions: what you assumed (max 4)."""


class AskError(Exception):
    pass


def ask_key(rid: str, question: str) -> str:
    norm = " ".join(question.lower().split())
    return hashlib.sha256(f"{rid}\n{norm}".encode()).hexdigest()[:16]


def interpret(question: str, context: dict, client_factory: Callable[[Path], meta_api.MuseSparkClient],
              evidence_dir: Path) -> dict:
    question = " ".join(question.split())
    if not 2 <= len(question) <= 300:
        raise AskError("Ask a question of 2-300 characters.")
    user = ("Scene (measured from the video):\n" + json.dumps(context, indent=2, sort_keys=True) +
            f"\n\nQuestion: {question}")
    try:
        client = client_factory(evidence_dir)
        res = client.chat([{"role": "developer", "content": DEVELOPER}, {"role": "user", "content": user}],
                          purpose=PURPOSE, max_completion_tokens=1200, reasoning_effort="low", json_schema=SCHEMA,
                          schema_name="whatif")
    except meta_api.MetaAPIError as exc:
        raise AskError(redact(f"Muse Spark request failed ({type(exc).__name__}): {exc}")) from None
    p = res.parsed
    if not isinstance(p, dict):
        raise AskError("Muse Spark returned no structured answer.")
    r, g, ext, vs = p["restitution"], p["gravity_scale"], float(p["extend_s"]), p.get("velocity_scale")
    if (r is not None and not 0 <= r <= 1 or g is not None and not 0.1 <= g <= 3 or not 0 <= ext <= 3
            or vs is not None and not 0.2 <= vs <= 3):
        raise AskError("Muse Spark proposed parameters outside the supported ranges; nothing was simulated.")
    floor_off = bool(p.get("remove_floor"))
    if not p["supported"]:
        r = g = vs = None
        floor_off = False
    return {"question": question, "supported": bool(p["supported"]), "restitution": r, "gravity_scale": g,
            "velocity_scale": vs, "remove_floor": floor_off,
            "extend_s": ext, "headline": p["headline"], "reasoning": p["reasoning"],
            "assumptions": list(p["assumptions"]), "unsupported_reason": p["unsupported_reason"],
            "model": res.model, "response_id": res.response_id,
            "usage": {"prompt_tokens": getattr(res.usage, "prompt_tokens", None),
                      "completion_tokens": getattr(res.usage, "completion_tokens", None)},
            "latency_s": res.latency_s, "authority": "Muse Spark chose the parameters; the deterministic simulator "
                                                       "computed the motion."}


# ------------------------------------------------------------------ "explain and critique this fit"

EXPLAIN_PURPOSE = "decompile-web-explain"
EXPLAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headline", "explanation", "material_guess", "plausibility", "critique", "confidence"],
    "properties": {
        "headline": {"type": "string", "maxLength": 120},
        "explanation": {"type": "string", "maxLength": 700},
        "material_guess": {"type": "string", "maxLength": 160},
        "plausibility": {"type": "string", "maxLength": 500},
        "critique": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 240}},
        "confidence": {"enum": ["high", "medium", "low"]},
    },
}

EXPLAIN_DEVELOPER = """You are a physicist reviewing a physics model that was fitted to a short real video by a
measurement pipeline. Use ONLY the numbers provided; never invent measurements. Write for a smart non-expert.
- headline: one short sentence with the key finding in everyday words.
- explanation: 2-3 sentences that translate the numbers (e.g. restitution e means the ball keeps e of its speed and
  1 - e^2 of its kinetic energy is lost at the bounce; relate the implied size to familiar objects).
- material_guess: what kind of ball/material these numbers are consistent with, hedged appropriately.
- plausibility: sanity-check the numbers against everyday physics (is the implied size plausible for the object
  named? is the restitution plausible?). Say clearly if something looks off.
- critique: up to 4 concrete weaknesses of THIS fit (flagged frames, gaps around the bounce, camera prior instead of
  calibration, fitted-not-predicted, few frames). Be specific, cite the numbers.
- confidence: high / medium / low that the recovered physics is right, given the evidence."""


def explain(context: dict, client_factory, evidence_dir: Path) -> dict:
    user = "Fitted model and measurement summary:\n" + json.dumps(context, indent=2, sort_keys=True)
    try:
        client = client_factory(evidence_dir)
        res = client.chat([{"role": "developer", "content": EXPLAIN_DEVELOPER}, {"role": "user", "content": user}],
                          purpose=EXPLAIN_PURPOSE, max_completion_tokens=1500, reasoning_effort="low",
                          json_schema=EXPLAIN_SCHEMA, schema_name="fit_review")
    except meta_api.MetaAPIError as exc:
        raise AskError(redact(f"Muse Spark request failed ({type(exc).__name__}): {exc}")) from None
    p = res.parsed
    if not isinstance(p, dict):
        raise AskError("Muse Spark returned no structured answer.")
    return {**p, "model": res.model, "response_id": res.response_id, "latency_s": res.latency_s,
            "input": context, "authority": "Muse Spark's commentary on numbers measured by Decompile; it did not "
                                          "change any measurement or parameter."}
