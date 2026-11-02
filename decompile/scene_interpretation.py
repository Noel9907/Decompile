"""B1b-1: scene interpretation schema `decompile.scene_interpretation/1` and its validator.

A scene interpretation is a qualitative reading of a trajectory set. It keeps two kinds
of content apart:

- observed_facts: statements about what the observations show. Every fact cites the
  observations it rests on as (object_id, frame_index) pairs.
- hypotheses (physical models, events, interactions): candidate explanations only.
  Each carries status "candidate", a support level (weak/moderate/strong; there is no
  "proven"), and points to facts, never directly to observations.

Architectural boundary: an interpretation never contains numerical physical parameters
(gravity, restitution, drag, speeds in physical units...). The schema has no field for
them, and free text is scanned for estimates (detect_parameter_leaks). Those numbers are
recovered by the classical fitting pipeline, not by the language model.

The interpretation contains no timestamps either. An event is placed in a conservative
temporal bracket between two observed frame indices; up to MAX_EVENT_INTERIOR_FRAMES
observed frames may lie inside the bracket. The bracket is NOT an event timestamp: it
only says the event is hypothesised to occur somewhere strictly inside it. Its exact
bounds are derived here from the trajectory set's Fraction timestamps (EventWindow), so
a model cannot invent a time and no precision is lost to floating point.

The interpretation makes no object-identity guess (no category such as "ball"): the
input carries trajectories only, with no visual information to support one.

The JSON Schema (build_interpretation_schema) is generated per trajectory set: object_id
is an enum of the input's object ids. It stays inside Meta's documented strict subset
and the subset checked by meta_api.check_strict_schema. It deliberately uses only
keywords already exercised in a live B1a call (type, properties, required,
additionalProperties, items, enum, minItems, maxItems): a single-value enum stands in
for const, and definitions are inlined instead of $ref. Ranges, string lengths, patterns
and uniqueness are checked by validate_interpretation instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from decompile.meta_api import check_strict_schema, validate_instance
from decompile.trajectories import TrajectorySet

SCHEMA_VERSION = "decompile.scene_interpretation/1"
SCHEMA_NAME = "scene_interpretation"

MOTION_CLASSES = (
    "stationary",
    "constant_velocity_like",
    "ballistic_like",
    "bouncing_like",
    "oscillating_like",
    "irregular",
    "insufficient_data",
)
SUPPORT_LEVELS = ("weak", "moderate", "strong")  # deliberately no "proven"/"certain"
# Mapped onto the A9 scene IR (Gravity, PlaneContact, LinearDrag) where one exists.
PHYSICAL_MODELS = (
    "uniform_gravity",
    "plane_contact",
    "linear_drag",
    "quadratic_drag",
    "constant_velocity",
    "object_collision",
    "other",
)
PLANE_HINTS = ("floor", "ceiling", "left_wall", "right_wall", "not_applicable", "unknown")
EVENT_TYPES = (
    "possible_bounce",
    "possible_contact_start",
    "possible_contact_end",
    "possible_collision",
    "possible_release",
    "possible_stop",
    "possible_occlusion",
    "other",
)
INTERACTION_TYPES = ("collision", "contact", "support", "attachment", "independent", "unknown")
CANDIDATE = "candidate"

ID_PATTERNS = {
    "fact": re.compile(r"F[1-9][0-9]{0,3}"),
    "hypothesis": re.compile(r"H[1-9][0-9]{0,3}"),
    "event": re.compile(r"E[1-9][0-9]{0,3}"),
    "interaction": re.compile(r"I[1-9][0-9]{0,3}"),
    "ambiguity": re.compile(r"A[1-9][0-9]{0,3}"),
}

MAX_STRING_CHARS = 600
MAX_ARRAY_ITEMS = 50
MAX_EVIDENCE_REFS = 20
# An event window is a conservative bracket, not a timestamp: up to this many observed
# frames may lie strictly inside it (see EventWindow).
MAX_EVENT_INTERIOR_FRAMES = 2


class SceneInterpretationError(ValueError):
    """An interpretation failed schema or semantic validation."""

    def __init__(self, errors: list[str], warnings: list[str]) -> None:
        self.errors = list(errors)
        self.warnings = list(warnings)
        super().__init__("invalid scene interpretation: " + "; ".join(self.errors))


# ------------------------------------------------------------------ schema


def _str() -> dict:
    return {"type": "string"}


def _enum(values: tuple[str, ...] | list[str]) -> dict:
    return {"type": "string", "enum": list(values)}


def _array(items: dict, min_items: int = 0) -> dict:
    schema: dict[str, Any] = {"type": "array", "items": items}
    if min_items:
        schema["minItems"] = min_items
    return schema


def _object(props: dict[str, dict]) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def build_interpretation_schema(trajectories: TrajectorySet) -> dict:
    """Strict JSON Schema for an interpretation of `trajectories`."""
    object_id = _enum(list(trajectories.object_ids))
    ids = _array(_str())
    obs_ref = _object({"object_id": object_id, "frame_index": {"type": "integer"}})
    status = _enum([CANDIDATE])
    support = _enum(SUPPORT_LEVELS)

    schema = _object(
        {
            "schema_version": _enum([SCHEMA_VERSION]),
            "observed_facts": _array(
                _object(
                    {
                        "fact_id": _str(),
                        "object_ids": _array(object_id, 1),
                        "statement": _str(),
                        "evidence": _array(obs_ref, 1),
                    }
                )
            ),
            "objects": _array(
                _object(
                    {
                        "object_id": object_id,
                        "motion_class": _enum(MOTION_CLASSES),
                        "motion_description": _str(),
                        "supporting_fact_ids": ids,
                    }
                ),
                1,
            ),
            "physical_model_hypotheses": _array(
                _object(
                    {
                        "hypothesis_id": _str(),
                        "status": status,
                        "model": _enum(PHYSICAL_MODELS),
                        "plane_hint": _enum(PLANE_HINTS),
                        "applies_to": _array(object_id, 1),
                        "support": support,
                        "supporting_fact_ids": ids,
                        "contradicting_fact_ids": ids,
                        "rationale": _str(),
                        "alternatives_considered": _array(_str()),
                    }
                )
            ),
            "event_hypotheses": _array(
                _object(
                    {
                        "event_id": _str(),
                        "status": status,
                        "type": _enum(EVENT_TYPES),
                        "object_ids": _array(object_id, 1),
                        "after_frame_index": {"type": "integer"},
                        "before_frame_index": {"type": "integer"},
                        "support": support,
                        "supporting_fact_ids": ids,
                        "rationale": _str(),
                    }
                )
            ),
            "interaction_hypotheses": _array(
                _object(
                    {
                        "interaction_id": _str(),
                        "status": status,
                        "object_ids": _array(object_id, 2),
                        "type": _enum(INTERACTION_TYPES),
                        "support": support,
                        "supporting_fact_ids": ids,
                        "rationale": _str(),
                    }
                )
            ),
            "ambiguities": _array(
                _object(
                    {
                        "ambiguity_id": _str(),
                        "about_ids": _array(_str(), 1),
                        "description": _str(),
                        "alternatives": _array(_str(), 1),
                        "what_would_resolve": _str(),
                    }
                )
            ),
            "data_quality_notes": _array(_str()),
        }
    )
    check_strict_schema(schema)  # never hand out a schema the client would refuse
    return schema


# ------------------------------------------------------------------ data structures


@dataclass(frozen=True)
class ObsRef:
    object_id: str
    frame_index: int


@dataclass(frozen=True)
class ObservedFact:
    fact_id: str
    object_ids: tuple[str, ...]
    statement: str
    evidence: tuple[ObsRef, ...]


@dataclass(frozen=True)
class ObjectInterpretation:
    object_id: str
    motion_class: str
    motion_description: str
    supporting_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelHypothesis:
    hypothesis_id: str
    status: str
    model: str
    plane_hint: str
    applies_to: tuple[str, ...]
    support: str
    supporting_fact_ids: tuple[str, ...]
    contradicting_fact_ids: tuple[str, ...]
    rationale: str
    alternatives_considered: tuple[str, ...]


@dataclass(frozen=True)
class EventHypothesis:
    event_id: str
    status: str
    type: str
    object_ids: tuple[str, ...]
    after_frame_index: int
    before_frame_index: int
    support: str
    supporting_fact_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class InteractionHypothesis:
    interaction_id: str
    status: str
    object_ids: tuple[str, ...]
    type: str
    support: str
    supporting_fact_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Ambiguity:
    ambiguity_id: str
    about_ids: tuple[str, ...]
    description: str
    alternatives: tuple[str, ...]
    what_would_resolve: str


@dataclass(frozen=True)
class EventWindow:
    """Conservative temporal bracket of an event, derived from the trajectory set (not the model).

    This is a bracket, not an event timestamp: the event is hypothesised to occur
    somewhere strictly inside t_after < t < t_before, and nothing more precise is
    claimed. Both bounds are observed frames, with exact Fraction times taken from the
    input (never floats). interior_frame_indices lists the observed frames inside the
    bracket, at most MAX_EVENT_INTERIOR_FRAMES. From positions alone a bounce can usually
    only be bracketed around the extreme sample (frames k-1..k+1), not placed in one gap:
    in the synthetic bounce fixture the bracket is frame 18 to frame 20, with frame 19 as
    an interior observation. Locating the instant inside the bracket is left to numerical
    fitting.
    """

    event_id: str
    after_frame_index: int
    before_frame_index: int
    t_after: Fraction
    t_before: Fraction
    interior_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class SceneInterpretation:
    observed_facts: tuple[ObservedFact, ...]
    objects: tuple[ObjectInterpretation, ...]
    physical_model_hypotheses: tuple[ModelHypothesis, ...]
    event_hypotheses: tuple[EventHypothesis, ...]
    interaction_hypotheses: tuple[InteractionHypothesis, ...]
    ambiguities: tuple[Ambiguity, ...]
    data_quality_notes: tuple[str, ...]
    event_windows: tuple[EventWindow, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    event_windows: tuple[EventWindow, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


# ------------------------------------------------------------------ parameter leaks

_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:\s*[eE][-+]?\d+)?"
_PARAM_WORDS = (
    r"gravity|gravitational|acceleration|accel|restitution|coefficient|cor|drag|friction|damping|"
    r"elasticity|bounciness|mass|speed|velocity|vx|vy|ratio|factor|fraction|energy\s+loss"
)
_STANDARD_G = r"(?<![\d.,])9[.,]8\d*(?![\d])"
_GRAVITY_CONTEXT = r"(?:\bgravit\w*|\bearth(?:'s)?\b|\bfree[- ]?fall\w*|(?<![\w.])g(?![\w.]))"
_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "parameter assigned a number",
        re.compile(
            rf"\b(?:{_PARAM_WORDS})\b[^.;:\n]{{0,30}}?"
            rf"(?:=|≈|~|≅|\bof\b|\bis\b|\bequals?\b|\babout\b|\baround\b|\broughly\b|\bapprox\w*)\s*{_NUM}",
            re.IGNORECASE,
        ),
    ),
    (
        "number with a rate unit",
        re.compile(
            rf"{_NUM}\s*(?:m|cm|mm|px|pixels?|meters?|metres?)\s*(?:/|per)\s*(?:s|sec|seconds?)\b"
            r"(?:\s*(?:\^?\s*2|²|squared))?",
            re.IGNORECASE,
        ),
    ),
    ("g assigned a number", re.compile(rf"(?<![\w.])g\s*(?:=|≈|~|≅)\s*{_NUM}")),
    ("unit of acceleration", re.compile(r"\b(?:m|px|cm)\s*/\s*s\s*(?:\^?\s*2|²)", re.IGNORECASE)),
    # 9.8x only counts as a gravity estimate in a gravity context (gravity, gravitational,
    # Earth, free fall, or a standalone "g") within the same clause; a bare observed
    # coordinate such as "y is 9.81 px at frame 4" is not flagged. Explicit estimates
    # ("gravity is 9.81", "g ≈ 9.81", "9.81 m/s^2") are also caught by the patterns above.
    (
        "standard gravity value",
        re.compile(
            rf"{_GRAVITY_CONTEXT}[^.;\n]{{0,40}}?{_STANDARD_G}|{_STANDARD_G}[^.;\n]{{0,40}}?{_GRAVITY_CONTEXT}",
            re.IGNORECASE,
        ),
    ),
    (
        "percentage of a physical quantity",
        re.compile(
            rf"{_NUM}\s*%[^.;\n]{{0,40}}\b(?:speed|velocity|energy|height|bounce|rebound|restitution)\b"
            rf"|\b(?:speed|velocity|energy|height|bounce|rebound|restitution)\b[^.;\n]{{0,40}}?{_NUM}\s*%",
            re.IGNORECASE,
        ),
    ),
)

_CERTAINTY_RE = re.compile(
    r"\b(?:proven|proves?|proof|confirm(?:s|ed)?|definite(?:ly)?|certain(?:ly)?|undoubtedly|"
    r"conclusive(?:ly)?|guarantee[ds]?|unquestionabl[ey]|without\s+doubt)\b",
    re.IGNORECASE,
)


def detect_parameter_leaks(text: str) -> list[str]:
    """Return descriptions of numerical physical-parameter estimates found in `text`.

    Heuristic. Observed coordinates and frame numbers ("y rises from 450.0 to 269.6
    between frames 19 and 36") are allowed; parameter-like quantities are not.
    """
    found = []
    for label, pattern in _LEAK_PATTERNS:
        m = pattern.search(text)
        if m:
            found.append(f"{label}: {m.group(0).strip()!r}")
    return found


# ------------------------------------------------------------------ validation


def _text_fields(data: dict) -> list[tuple[str, str]]:
    """(path, text) for every free-text field of a schema-valid interpretation."""
    out: list[tuple[str, str]] = []
    for i, f in enumerate(data["observed_facts"]):
        out.append((f"observed_facts[{i}].statement", f["statement"]))
    for i, o in enumerate(data["objects"]):
        out.append((f"objects[{i}].motion_description", o["motion_description"]))
    for i, h in enumerate(data["physical_model_hypotheses"]):
        out.append((f"physical_model_hypotheses[{i}].rationale", h["rationale"]))
        for j, a in enumerate(h["alternatives_considered"]):
            out.append((f"physical_model_hypotheses[{i}].alternatives_considered[{j}]", a))
    for kind in ("event_hypotheses", "interaction_hypotheses"):
        for i, e in enumerate(data[kind]):
            out.append((f"{kind}[{i}].rationale", e["rationale"]))
    for i, a in enumerate(data["ambiguities"]):
        out.append((f"ambiguities[{i}].description", a["description"]))
        out.append((f"ambiguities[{i}].what_would_resolve", a["what_would_resolve"]))
        for j, alt in enumerate(a["alternatives"]):
            out.append((f"ambiguities[{i}].alternatives[{j}]", alt))
    for i, n in enumerate(data["data_quality_notes"]):
        out.append((f"data_quality_notes[{i}]", n))
    return out


def _hypothesis_text_fields(data: dict) -> list[tuple[str, str]]:
    """Text where certainty language would overstate a candidate explanation."""
    out = []
    for kind in ("physical_model_hypotheses", "event_hypotheses", "interaction_hypotheses"):
        for i, h in enumerate(data[kind]):
            out.append((f"{kind}[{i}].rationale", h["rationale"]))
    for i, o in enumerate(data["objects"]):
        out.append((f"objects[{i}].motion_description", o["motion_description"]))
    return out


def _duplicates(values: list) -> list:
    seen, dups = set(), []
    for v in values:
        if v in seen and v not in dups:
            dups.append(v)
        seen.add(v)
    return dups


def validate_interpretation(data: Any, trajectories: TrajectorySet) -> ValidationReport:
    """Check `data` against the schema and every semantic rule. Never raises on bad data."""
    schema = build_interpretation_schema(trajectories)
    violations = validate_instance(data, schema)
    if violations:
        # Semantic checks assume the schema shape; stop here.
        return ValidationReport(tuple(f"schema: {v}" for v in violations), (), ())

    errors: list[str] = []
    warnings: list[str] = []
    input_ids = list(trajectories.object_ids)
    frames = {o.object_id: set(o.frame_indices) for o in trajectories.objects}

    facts = data["observed_facts"]
    objects = data["objects"]
    models = data["physical_model_hypotheses"]
    events = data["event_hypotheses"]
    interactions = data["interaction_hypotheses"]
    ambiguities = data["ambiguities"]

    # -- 1. ID formats and global uniqueness ------------------------------------------
    declared: list[tuple[str, str, str]] = []  # (kind, path, id)
    for kind, items, key, section in (
        ("fact", facts, "fact_id", "observed_facts"),
        ("hypothesis", models, "hypothesis_id", "physical_model_hypotheses"),
        ("event", events, "event_id", "event_hypotheses"),
        ("interaction", interactions, "interaction_id", "interaction_hypotheses"),
        ("ambiguity", ambiguities, "ambiguity_id", "ambiguities"),
    ):
        for i, item in enumerate(items):
            ident = item[key]
            path = f"{section}[{i}].{key}"
            if not ID_PATTERNS[kind].fullmatch(ident):
                errors.append(f"{path}: {ident!r} does not match {ID_PATTERNS[kind].pattern}")
            declared.append((kind, path, ident))
    for dup in _duplicates([ident for _, _, ident in declared]):
        errors.append(f"id {dup!r} is declared more than once")
    fact_ids = {ident for kind, _, ident in declared if kind == "fact"}
    referable = {ident for kind, _, ident in declared if kind != "ambiguity"} | set(input_ids)

    # -- 2. Size limits -----------------------------------------------------------------
    for section in ("observed_facts", "objects", "physical_model_hypotheses", "event_hypotheses",
                    "interaction_hypotheses", "ambiguities", "data_quality_notes"):
        if len(data[section]) > MAX_ARRAY_ITEMS:
            errors.append(f"{section}: {len(data[section])} items (max {MAX_ARRAY_ITEMS})")
    for path, text in _text_fields(data):
        if len(text) > MAX_STRING_CHARS:
            errors.append(f"{path}: {len(text)} characters (max {MAX_STRING_CHARS})")
        if not text.strip() and not path.startswith("data_quality_notes"):
            errors.append(f"{path}: must not be empty")

    # -- 3. Observed facts: evidence must exist and match the fact's objects ----------
    fact_objects: dict[str, set[str]] = {}
    for i, f in enumerate(facts):
        path = f"observed_facts[{i}]"
        fact_objects[f["fact_id"]] = set(f["object_ids"])
        for dup in _duplicates(f["object_ids"]):
            errors.append(f"{path}.object_ids: duplicate {dup!r}")
        refs = [(r["object_id"], r["frame_index"]) for r in f["evidence"]]
        if len(refs) > MAX_EVIDENCE_REFS:
            errors.append(f"{path}.evidence: {len(refs)} references (max {MAX_EVIDENCE_REFS})")
        for dup in _duplicates(refs):
            errors.append(f"{path}.evidence: duplicate reference {dup}")
        for obj, frame in refs:
            if frame not in frames.get(obj, set()):
                errors.append(f"{path}.evidence: {obj!r} has no observation at frame {frame}")
        cited = {obj for obj, _ in refs}
        if cited != set(f["object_ids"]):
            errors.append(
                f"{path}: object_ids {sorted(set(f['object_ids']))} do not match the objects "
                f"cited in evidence {sorted(cited)}"
            )

    # -- 4. Object coverage: every input object exactly once --------------------------
    covered = [o["object_id"] for o in objects]
    for dup in _duplicates(covered):
        errors.append(f"objects: {dup!r} is interpreted more than once")
    missing = [oid for oid in input_ids if oid not in covered]
    if missing:
        errors.append(f"objects: no interpretation for input objects {missing}")

    def check_fact_refs(path: str, ids: list[str], subjects: set[str]) -> None:
        for dup in _duplicates(ids):
            errors.append(f"{path}: duplicate fact reference {dup!r}")
        for ref in ids:
            if ref not in fact_ids:
                errors.append(f"{path}: {ref!r} is not a declared observed fact")
            elif not fact_objects[ref] & subjects:
                warnings.append(f"{path}: fact {ref!r} is about none of {sorted(subjects)}")

    for i, o in enumerate(objects):
        check_fact_refs(f"objects[{i}].supporting_fact_ids", o["supporting_fact_ids"], {o["object_id"]})

    # -- 5. Physical model hypotheses ---------------------------------------------------
    for i, h in enumerate(models):
        path = f"physical_model_hypotheses[{i}]"
        subjects = set(h["applies_to"])
        for dup in _duplicates(h["applies_to"]):
            errors.append(f"{path}.applies_to: duplicate {dup!r}")
        check_fact_refs(f"{path}.supporting_fact_ids", h["supporting_fact_ids"], subjects)
        check_fact_refs(f"{path}.contradicting_fact_ids", h["contradicting_fact_ids"], subjects)
        both = set(h["supporting_fact_ids"]) & set(h["contradicting_fact_ids"])
        if both:
            errors.append(f"{path}: facts {sorted(both)} are cited as both supporting and contradicting")
        if h["model"] == "plane_contact" and h["plane_hint"] == "not_applicable":
            errors.append(f"{path}: plane_contact needs a plane_hint other than 'not_applicable'")
        if h["model"] != "plane_contact" and h["plane_hint"] != "not_applicable":
            errors.append(f"{path}: plane_hint must be 'not_applicable' for model {h['model']!r}")
        if h["model"] == "object_collision" and len(subjects) < 2:
            errors.append(f"{path}: object_collision must apply to at least two distinct objects")
        if not h["supporting_fact_ids"]:
            warnings.append(f"{path}: no supporting facts")
        if h["support"] == "strong" and h["contradicting_fact_ids"]:
            warnings.append(f"{path}: support 'strong' despite contradicting facts")
        if not h["alternatives_considered"]:
            warnings.append(f"{path}: no alternatives considered")

    # -- 6. Events: conservative brackets between observed frames, exact bounds derived
    windows: list[EventWindow] = []
    for i, e in enumerate(events):
        path = f"event_hypotheses[{i}]"
        involved = e["object_ids"]
        for dup in _duplicates(involved):
            errors.append(f"{path}.object_ids: duplicate {dup!r}")
        check_fact_refs(f"{path}.supporting_fact_ids", e["supporting_fact_ids"], set(involved))
        if not e["supporting_fact_ids"]:
            warnings.append(f"{path}: no supporting facts")
        if e["type"] == "possible_collision" and len(set(involved)) < 2:
            errors.append(f"{path}: possible_collision needs at least two distinct objects "
                          "(contact with a surface is a bounce or contact event)")
        a, b = e["after_frame_index"], e["before_frame_index"]
        window_ok = True
        if a >= b:
            errors.append(f"{path}: after_frame_index {a} must be < before_frame_index {b}")
            window_ok = False
        for obj in set(involved):
            observed = frames[obj]
            for frame in (a, b):
                if frame not in observed:
                    errors.append(f"{path}: {obj!r} has no observation at frame {frame}")
                    window_ok = False
            between = sorted(f for f in observed if a < f < b)
            if len(between) > MAX_EVENT_INTERIOR_FRAMES:
                errors.append(
                    f"{path}: window ({a}, {b}) is too wide for {obj!r}: {len(between)} observed frames "
                    f"inside it (max {MAX_EVENT_INTERIOR_FRAMES})"
                )
                window_ok = False
        if window_ok:
            interior = sorted({f for obj in set(involved) for f in frames[obj] if a < f < b})
            windows.append(
                EventWindow(
                    e["event_id"], a, b, trajectories.frame_time(a), trajectories.frame_time(b), tuple(interior)
                )
            )

    # -- 7. Interactions -----------------------------------------------------------------
    for i, it in enumerate(interactions):
        path = f"interaction_hypotheses[{i}]"
        distinct = set(it["object_ids"])
        if len(distinct) < 2:
            errors.append(f"{path}: an interaction needs at least two distinct objects")
        for dup in _duplicates(it["object_ids"]):
            errors.append(f"{path}.object_ids: duplicate {dup!r}")
        check_fact_refs(f"{path}.supporting_fact_ids", it["supporting_fact_ids"], distinct)
        if not it["supporting_fact_ids"]:
            warnings.append(f"{path}: no supporting facts")

    # -- 8. Ambiguities --------------------------------------------------------------------
    for i, amb in enumerate(ambiguities):
        path = f"ambiguities[{i}].about_ids"
        for dup in _duplicates(amb["about_ids"]):
            errors.append(f"{path}: duplicate {dup!r}")
        for ref in amb["about_ids"]:
            if ref not in referable:
                errors.append(f"{path}: {ref!r} is not a declared fact, hypothesis, event, interaction or object")
    if not ambiguities:
        warnings.append("ambiguities: none recorded; alternative explanations were not documented")

    # -- 9. No numerical physical-parameter estimates anywhere ---------------------
    for path, text in _text_fields(data):
        for leak in detect_parameter_leaks(text):
            errors.append(f"{path}: numerical physical-parameter estimate ({leak})")

    # -- 10. Certainty language in candidate explanations (warning) ----------------
    for path, text in _hypothesis_text_fields(data):
        m = _CERTAINTY_RE.search(text)
        if m:
            warnings.append(f"{path}: certainty language {m.group(0)!r} in a candidate explanation")

    return ValidationReport(tuple(errors), tuple(warnings), tuple(windows) if not errors else ())


def parse_interpretation(data: Any, trajectories: TrajectorySet) -> SceneInterpretation:
    """Validate and convert to data structures. Raises SceneInterpretationError on any error."""
    report = validate_interpretation(data, trajectories)
    if not report.ok:
        raise SceneInterpretationError(list(report.errors), list(report.warnings))
    t = tuple
    return SceneInterpretation(
        observed_facts=t(
            ObservedFact(f["fact_id"], t(f["object_ids"]), f["statement"],
                         t(ObsRef(r["object_id"], r["frame_index"]) for r in f["evidence"]))
            for f in data["observed_facts"]
        ),
        objects=t(
            ObjectInterpretation(o["object_id"], o["motion_class"], o["motion_description"],
                                 t(o["supporting_fact_ids"]))
            for o in data["objects"]
        ),
        physical_model_hypotheses=t(
            ModelHypothesis(h["hypothesis_id"], h["status"], h["model"], h["plane_hint"], t(h["applies_to"]),
                            h["support"], t(h["supporting_fact_ids"]), t(h["contradicting_fact_ids"]),
                            h["rationale"], t(h["alternatives_considered"]))
            for h in data["physical_model_hypotheses"]
        ),
        event_hypotheses=t(
            EventHypothesis(e["event_id"], e["status"], e["type"], t(e["object_ids"]), e["after_frame_index"],
                            e["before_frame_index"], e["support"], t(e["supporting_fact_ids"]), e["rationale"])
            for e in data["event_hypotheses"]
        ),
        interaction_hypotheses=t(
            InteractionHypothesis(i["interaction_id"], i["status"], t(i["object_ids"]), i["type"], i["support"],
                                  t(i["supporting_fact_ids"]), i["rationale"])
            for i in data["interaction_hypotheses"]
        ),
        ambiguities=t(
            Ambiguity(a["ambiguity_id"], t(a["about_ids"]), a["description"], t(a["alternatives"]),
                      a["what_would_resolve"])
            for a in data["ambiguities"]
        ),
        data_quality_notes=t(data["data_quality_notes"]),
        event_windows=report.event_windows,
        warnings=report.warnings,
    )
