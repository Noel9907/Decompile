"""B1c-2: deterministic physics review of a scene interpretation against kinematic fits.

review_interpretation(interp, kin) compares a SceneInterpretation (B1b; candidate
explanations, typically written by Muse Spark) with a KinematicsReport (B1c-1; numerical
fits of the same trajectory). The kinematic report is the authority for numerical
evidence; the interpretation is a set of hypotheses to be checked, never evidence.

The reviewer only reads its inputs: it never mutates them, never refits, never calls an
API. It returns explicit findings. A kinematic match is reported as "kinematic
consistency" only: a uniform acceleration is not proof of gravity, an impulse is not
proof of contact with a particular surface, and no surface identity (floor, ceiling,
wall) is ever confirmed from a trajectory.

Finding structure (Finding): code, severity, subject_id (fact/hypothesis/event/ambiguity
or object id, when available), object_id, verdict, message, evidence (numbers).

Severity rules (conservative; a weak or inconclusive numerical result never becomes an error):
    error    the kinematic analysis decisively contradicts a claim: the fitter's verdict for
             the mapped claim is "contradicted", or a contact-type event window excludes the
             whole contact-time interval of a supported impulse, or the interpretation's
             frame times disagree with the kinematic report (inputs do not match).
    warning  the claim is "not_supported" (positively disfavoured, not decisively refuted),
             whatever its support label; an impulse has no matching event; an ambiguity keeps
             open a hypothesis or event that the analysis does not support; object coverage
             differs; the conservative direction-wording heuristic finds a conflict.
    info     consistent, inconclusive or not_evaluated results, support-label notes and
             kinematic facts (e.g. the smooth explanation of an abrupt change is contradicted).

Finding verdicts: supported, contradicted, not_supported, inconclusive, not_evaluated
(kinematic vocabulary), consistent (an event window agrees with the fit), flagged
(wording heuristic). Support labels (weak/moderate/strong) never change a verdict: a weak
label does not rescue a rejected claim; a strong label on a rejected claim is noted.

Codes (stable):
    OBJECT_NOT_IN_KINEMATICS, OBJECT_NOT_INTERPRETED, INPUT_MISMATCH
    KINEMATICS_INADEQUATE
    MOTION_CLASS_CHECK, MOTION_CLASS_NOT_EVALUATED
    HYPOTHESIS_CHECK, HYPOTHESIS_NOT_EVALUATED, SUPPORT_LABEL_NOTE
    SURFACE_IDENTITY_NOT_EVALUATED, SURFACE_DIRECTION_COMPATIBLE, SURFACE_DIRECTION_INCOMPATIBLE
    SMOOTH_EXPLANATION_CONTRADICTED, SMOOTH_ALTERNATIVE_CONTRADICTED
    EVENT_CONTACT_WINDOW, EVENT_BRACKETS_TURNING_POINT, EVENT_NOT_EVALUATED, IMPULSE_WITHOUT_EVENT
    AMBIGUITY_ON_DECIDED_CLAIM, AMBIGUITY_REFERENCE_UNKNOWN
    DIRECTION_WORDING_CONFLICT, EXTREMUM_WORDING_CONFLICT

Image coordinates: +y is DOWN the image. Increasing y is downward image motion; the
smallest y is the highest point in the image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from decompile import kinematic_fit as kf
from decompile.scene_interpretation import PHYSICAL_MODELS, SceneInterpretation

METHOD_VERSION = "decompile.physics_review/1"

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITIES = (ERROR, WARNING, INFO)
CONSISTENT, FLAGGED = "consistent", "flagged"

# Interpretation physical models -> kinematic verdict names. Unlisted models are not evaluated.
MODEL_TO_KINEMATIC = {
    "uniform_gravity": "uniform_acceleration",
    "plane_contact": "impulse",
    "linear_drag": "linear_drag",
    "quadratic_drag": "quadratic_drag",
    "constant_velocity": "constant_velocity",
}
# Models with no velocity discontinuity: listed as an alternative to a supported impulse, each is
# contradicted as a sole explanation. Matched by exact enum token only, never by wording.
SMOOTH_MODELS = frozenset({"uniform_gravity", "constant_velocity", "linear_drag", "quadratic_drag"})
# Event types expected to coincide with a velocity discontinuity of each involved object.
IMPULSE_EVENT_TYPES = frozenset({"possible_bounce", "possible_collision"})
# Expected reversal for a plane hint, in image directions (+y down). Direction compatibility only.
PLANE_REVERSAL = {
    "floor": ("y", "down_to_up"),
    "ceiling": ("y", "up_to_down"),
    "left_wall": ("x", "left_to_right"),
    "right_wall": ("x", "right_to_left"),
}

LIMITATIONS = (
    "Only the listed kinematic checks are performed; passing them does not establish that the "
    "interpretation is physically true.",
    "A uniform acceleration is checked, not its cause; gravity is never confirmed by a trajectory alone.",
    "An impulse supports a contact candidate; surface identity (floor, ceiling, wall) is never confirmed.",
    "Free text is not interpreted. Ambiguity alternatives are linked only through about_ids and exact "
    "model tokens; the wording heuristic is conservative, English-only and produces warnings only.",
    "Interactions between objects (collisions, support, attachment) are not evaluated.",
    "Every limitation of the kinematic report (decompile.kinematic_fit) applies.",
)


class PhysicsReviewError(ValueError):
    """Invalid input to review_interpretation."""


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    subject_id: str | None
    object_id: str | None
    verdict: str
    message: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PhysicsReview:
    method_version: str
    kinematics_method_version: str
    findings: tuple[Finding, ...]
    counts: dict[str, int]
    assessment: str  # "contradictions_found" | "concerns_found" | "no_contradictions_found"
    assessment_note: str
    limitations: tuple[str, ...]

    def by_code(self, code: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.code == code)

    def for_subject(self, subject_id: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.subject_id == subject_id)


# ------------------------------------------------------------------ helpers


def _claim_severity(kin_verdict: str) -> str:
    if kin_verdict == kf.CONTRADICTED:
        return ERROR
    if kin_verdict == kf.NOT_SUPPORTED:
        return WARNING
    return INFO


def _label_note(kin_verdict: str, label: str | None) -> str:
    if label is None:
        return ""
    if kin_verdict in (kf.CONTRADICTED, kf.NOT_SUPPORTED) and label == "strong":
        return " The support label 'strong' overstates the evidence."
    if kin_verdict in (kf.CONTRADICTED, kf.NOT_SUPPORTED):
        return f" The support label '{label}' does not change this."
    if kin_verdict == kf.SUPPORTED and label == "weak":
        return " The support label 'weak' understates the numerical evidence."
    if kin_verdict == kf.INCONCLUSIVE and label == "strong":
        return " The support label 'strong' exceeds what the numerical analysis can establish."
    return ""


def _frame_times(o: kf.ObjectKinematics) -> dict[int, Fraction]:
    times: dict[int, Fraction] = {}
    if o.finite_differences:
        t = o.t_start_s
        times[o.finite_differences[0].frame_a] = t
        for fd in o.finite_differences:
            t = t + fd.dt_s
            times[fd.frame_b] = t
    return times


def _y_steps(o: kf.ObjectKinematics) -> list[tuple[int, int, int]]:
    """(frame_a, frame_b, sign of vy) per observed interval; +1 is moving DOWN the image."""
    return [(fd.frame_a, fd.frame_b, (fd.vy_px_s > 0) - (fd.vy_px_s < 0)) for fd in o.finite_differences]


def _image_extrema(o: kf.ObjectKinematics) -> dict[int, tuple[str, str]]:
    """frame -> ("image_high" | "image_low", basis). image_high = locally smallest y."""
    out: dict[int, tuple[str, str]] = {}
    steps = _y_steps(o)
    for (a0, b0, s0), (a1, b1, s1) in zip(steps, steps[1:]):
        if s0 < 0 < s1:
            out[b0] = ("image_high", "sampled local minimum of y")
        elif s0 > 0 > s1:
            out[b0] = ("image_low", "sampled local maximum of y")
    for tp in o.turning_points:
        if tp.axis == "y":
            kind = "image_high" if tp.kind == "y_minimum" else "image_low"
            for f in tp.between_frames:
                out.setdefault(f, (kind, f"adjacent to the fitted {tp.kind} at t = {tp.t_s:.6g} s"))
    return out


# ------------------------------------------------------------------ wording heuristic

_UP = r"upwards?|rising|rises|rise|ascending|ascends|ascend"
_DOWN = r"downwards?|falling|falls|fall|descending|descends|descend|dropping|drops"
_MOVE = r"(?:moves?|moving|goes|going|travels?|travell?ing|heads?|heading)\s+(up|down)\b"
_DIR_RE = re.compile(rf"\b(?:({_UP})|({_DOWN})|{_MOVE})\b", re.IGNORECASE)
_NUMERIC_SUBJECT = re.compile(r"\b[xy](?:[- ](?:value|coordinate|position))?\s+(?:\w+ly\s+|first\s+|then\s+|again\s+)?$",
                              re.IGNORECASE)
_NUMERIC_CHANGE = re.compile(r"\s+(?:to|from|by)\s+[-+]?\d", re.IGNORECASE)
_LOW_WORDS, _HIGH_WORDS = ("low", "lowest", "bottom"), ("high", "highest", "top", "peak", "apex")
_EXT = "|".join(_LOW_WORDS + _HIGH_WORDS)
_EXT_RES = (
    re.compile(rf"\b(?:the\s+)?({_EXT})(?:\s+point)?\s+(?:at|near|around|in)\s+frame\s+(\d+)", re.IGNORECASE),
    re.compile(rf"\bframe\s+(\d+)\s+is\s+(?:the\s+|a\s+)?(?:local\s+)?({_EXT})\b", re.IGNORECASE),
)


def _direction_findings(subject: str, oid: str, text: str, frames: set[int], o: kf.ObjectKinematics) -> list[Finding]:
    out: list[Finding] = []
    if len(frames) < 2:
        return out
    lo, hi = min(frames), max(frames)
    span = [s for a, b, s in _y_steps(o) if a >= lo and b <= hi]
    if not span:
        return out
    for m in _DIR_RE.finditer(text):
        before = text[max(0, m.start() - 30):m.start()]
        if _NUMERIC_SUBJECT.search(before) or _NUMERIC_CHANGE.match(text, m.end()):
            continue  # a statement about a number ("y rises", "rising to 38.89 px"), not image motion
        claims_up = bool(m.group(1)) or (m.group(3) or "").lower() == "up"
        if claims_up and not any(s < 0 for s in span):
            out.append(Finding(
                "DIRECTION_WORDING_CONFLICT", WARNING, subject, oid, FLAGGED,
                f"{subject}: {m.group(0)!r} describes upward image motion, but over frames {lo}-{hi} of {oid} "
                "y increases on every observed interval, i.e. the object moves DOWN the image (+y is down).",
                dict(phrase=m.group(0), frames=(lo, hi), intervals_moving_down=sum(s > 0 for s in span),
                     intervals_moving_up=0, convention="+y is down the image"),
            ))
        elif not claims_up and not any(s > 0 for s in span):
            out.append(Finding(
                "DIRECTION_WORDING_CONFLICT", WARNING, subject, oid, FLAGGED,
                f"{subject}: {m.group(0)!r} describes downward image motion, but over frames {lo}-{hi} of {oid} "
                "y decreases on every observed interval, i.e. the object moves UP the image (+y is down).",
                dict(phrase=m.group(0), frames=(lo, hi), intervals_moving_down=0,
                     intervals_moving_up=sum(s < 0 for s in span), convention="+y is down the image"),
            ))
    extrema = _image_extrema(o)
    for i, rx in enumerate(_EXT_RES):
        for m in rx.finditer(text):
            word, frame = (m.group(1), int(m.group(2))) if i == 0 else (m.group(2), int(m.group(1)))
            if frame not in extrema:
                continue  # cannot judge: not an image extremum in the data
            kind, basis = extrema[frame]
            says_low = word.lower() in _LOW_WORDS
            if says_low and kind == "image_high":
                msg = (f"{subject}: {m.group(0)!r}, but frame {frame} is the HIGHEST point in the image nearby "
                       f"({basis}; the smallest y is the highest image point since +y is down). 'Low' conflicts "
                       "with the image direction unless it refers to the numeric y value.")
            elif not says_low and kind == "image_low":
                msg = (f"{subject}: {m.group(0)!r}, but frame {frame} is the LOWEST point in the image nearby "
                       f"({basis}; the largest y is the lowest image point since +y is down).")
            else:
                continue
            out.append(Finding("EXTREMUM_WORDING_CONFLICT", WARNING, subject, oid, FLAGGED, msg,
                               dict(phrase=m.group(0), frame=frame, image_extremum=kind, basis=basis,
                                    convention="+y is down the image")))
    return out


# ------------------------------------------------------------------ main entry point


def review_interpretation(interp: SceneInterpretation, kin: kf.KinematicsReport) -> PhysicsReview:
    """Compare `interp` with `kin` and return explicit findings. Reads both; mutates neither."""
    if not isinstance(interp, SceneInterpretation):
        raise PhysicsReviewError(f"expected a SceneInterpretation, got {type(interp).__name__}")
    if not isinstance(kin, kf.KinematicsReport):
        raise PhysicsReviewError(f"expected a KinematicsReport, got {type(kin).__name__}")

    F: list[Finding] = []
    kobj = {o.object_id: o for o in kin.objects}
    interp_ids = [o.object_id for o in interp.objects]
    facts = {f.fact_id: f for f in interp.observed_facts}
    windows = {w.event_id: w for w in interp.event_windows}

    def known(oid: str, subject: str) -> kf.ObjectKinematics | None:
        if oid in kobj:
            return kobj[oid]
        F.append(Finding("OBJECT_NOT_IN_KINEMATICS", WARNING, subject, oid, kf.NOT_EVALUATED,
                         f"{subject} refers to {oid!r}, which is not in the kinematic report; not evaluated.",
                         dict(kinematic_objects=list(kobj))))
        return None

    # ---- object coverage and kinematic adequacy
    for oid in kobj:
        if oid not in interp_ids:
            F.append(Finding("OBJECT_NOT_INTERPRETED", WARNING, oid, oid, kf.NOT_EVALUATED,
                             f"{oid!r} is in the kinematic report but the interpretation does not describe it.",
                             dict(interpreted_objects=interp_ids)))
    for oid in interp_ids:
        o = kobj.get(oid)
        if o is None:
            known(oid, oid)
        elif o.adequacy == "inadequate":
            F.append(Finding("KINEMATICS_INADEQUATE", INFO, oid, oid, kf.INCONCLUSIVE,
                             f"the kinematic models are inadequate for {oid}, so every kinematic verdict is "
                             "inconclusive and no claim about it can be confirmed or rejected.",
                             dict(reasons=list(o.adequacy_reasons))))

    # ---- kinematic facts independent of the interpretation: smooth explanations of an abrupt change
    for oid in interp_ids:
        o = kobj.get(oid)
        if o is None or o.contact is None or o.verdicts["impulse"].verdict != kf.SUPPORTED:
            continue
        ev = o.verdicts["impulse"].evidence
        for a in o.contact.axes:
            if not a.significant:
                continue
            what = f"reversal {a.reversal}" if a.reversal else "abrupt velocity change"
            F.append(Finding(
                "SMOOTH_EXPLANATION_CONTRADICTED", INFO, oid, oid, kf.CONTRADICTED,
                f"{oid}: the {what} on axis {a.axis} near t = {o.contact.t_contact_s:.6g} s needs a velocity "
                f"discontinuity of {a.delta_v_px_s:.6g} px/s (z = {a.delta_v_z:.3g}); the best smooth model "
                f"({ev['no_contact_model']}) is worse by Delta BIC = {ev['delta_bic']:.4g}. A smooth change inside "
                f"the {o.contact.gap_frames} gap would need {a.smooth_equivalent_accel_px_s2:.6g} px/s^2"
                + (f" ({a.ratio_to_uniform_accel:.3g} x the uniform acceleration)" if a.ratio_to_uniform_accel else "")
                + ". This contradicts an explanation without an impulse; it does not by itself identify a surface "
                "or prove contact with a particular object.",
                dict(axis=a.axis, reversal=a.reversal, t_contact_s=o.contact.t_contact_s,
                     gap_frames=o.contact.gap_frames, delta_v_px_s=a.delta_v_px_s, delta_v_z=a.delta_v_z,
                     smooth_equivalent_accel_px_s2=a.smooth_equivalent_accel_px_s2,
                     ratio_to_uniform_accel=a.ratio_to_uniform_accel, delta_bic=ev["delta_bic"],
                     no_contact_model=ev["no_contact_model"]),
            ))

    # ---- motion classes
    for oi in interp.objects:
        o = kobj.get(oi.object_id)
        if o is None:
            continue
        v = o.verdicts
        cls = oi.motion_class
        checks: list[tuple[str, str, str]] = []  # (kinematic verdict name, verdict used, expectation)
        if cls == "stationary":
            checks = [("stationary", v["stationary"].verdict, "at rest")]
        elif cls == "constant_velocity_like":
            checks = [("constant_velocity", v["constant_velocity"].verdict, "constant velocity")]
        elif cls == "ballistic_like":
            imp = v["impulse"].verdict
            checks = [("uniform_acceleration", v["uniform_acceleration"].verdict, "uniform acceleration"),
                      ("impulse", {kf.SUPPORTED: kf.NOT_SUPPORTED, kf.NOT_SUPPORTED: kf.SUPPORTED}.get(imp, imp),
                       "no velocity discontinuity")]
        elif cls == "bouncing_like":
            checks = [("impulse", v["impulse"].verdict, "a velocity discontinuity")]
        else:
            F.append(Finding("MOTION_CLASS_NOT_EVALUATED", INFO, oi.object_id, oi.object_id, kf.NOT_EVALUATED,
                             f"motion_class {cls!r} has no kinematic counterpart in this review.",
                             dict(motion_class=cls)))
            continue
        for name, verdict, expectation in checks:
            F.append(Finding("MOTION_CLASS_CHECK", _claim_severity(verdict), oi.object_id, oi.object_id, verdict,
                             f"{oi.object_id} motion_class {cls!r} implies {expectation}; kinematic verdict "
                             f"{name} = {v[name].verdict}: {v[name].reason}",
                             dict(motion_class=cls, kinematic_verdict=name, kinematic_result=v[name].verdict,
                                  **_numbers(v[name].evidence))))

    # ---- physical model hypotheses
    hyp_outcome: dict[str, list[str]] = {}
    for h in interp.physical_model_hypotheses:
        name = MODEL_TO_KINEMATIC.get(h.model)
        for oid in h.applies_to:
            o = known(oid, h.hypothesis_id)
            if o is None:
                continue
            if name is None or o.verdicts[name].verdict == kf.NOT_EVALUATED:
                why = ("is not mapped to a kinematic check" if name is None
                       else f"maps to {name}, which the kinematic report does not evaluate")
                F.append(Finding("HYPOTHESIS_NOT_EVALUATED", INFO, h.hypothesis_id, oid, kf.NOT_EVALUATED,
                                 f"{h.hypothesis_id}: model {h.model!r} {why}.",
                                 dict(model=h.model, known_model=h.model in PHYSICAL_MODELS)))
                continue
            kv = o.verdicts[name]
            hyp_outcome.setdefault(h.hypothesis_id, []).append(kv.verdict)
            F.append(Finding("HYPOTHESIS_CHECK", _claim_severity(kv.verdict), h.hypothesis_id, oid, kv.verdict,
                             _hypothesis_message(h.hypothesis_id, h.model, name, kv, o),
                             dict(model=h.model, kinematic_verdict=name, support_label=h.support,
                                  provisional=kv.provisional, **_numbers(kv.evidence))))
            note = _label_note(kv.verdict, h.support)
            if note:
                F.append(Finding("SUPPORT_LABEL_NOTE", INFO, h.hypothesis_id, oid, kv.verdict, f"{h.hypothesis_id}:{note}",
                                 dict(support_label=h.support, kinematic_result=kv.verdict)))
            if h.model == "plane_contact":
                F.extend(_plane_findings(h, o))
            if kv.verdict == kf.SUPPORTED and name == "impulse":
                for alt in h.alternatives_considered:
                    token = alt.strip()
                    if token in SMOOTH_MODELS:
                        F.append(Finding(
                            "SMOOTH_ALTERNATIVE_CONTRADICTED", INFO, h.hypothesis_id, oid, kf.CONTRADICTED,
                            f"{h.hypothesis_id} lists the model {token!r} as an alternative. That model has no velocity "
                            "discontinuity, so as a sole explanation of the abrupt change it is contradicted (see "
                            "SMOOTH_EXPLANATION_CONTRADICTED).",
                            dict(alternative=token, delta_bic=o.verdicts["impulse"].evidence["delta_bic"])))

    # ---- events
    event_outcome: dict[str, list[str]] = {}
    explained: set[str] = set()
    for e in interp.event_hypotheses:
        w = windows.get(e.event_id)
        for oid in e.object_ids:
            o = known(oid, e.event_id)
            if o is None or w is None:
                continue
            times = _frame_times(o)
            if times.get(w.after_frame_index) != w.t_after or times.get(w.before_frame_index) != w.t_before:
                F.append(Finding("INPUT_MISMATCH", ERROR, e.event_id, oid, kf.NOT_EVALUATED,
                                 f"{e.event_id}: the window frame times do not match the kinematic report for {oid}; "
                                 "the interpretation and the kinematics do not describe the same trajectory.",
                                 dict(window=(str(w.t_after), str(w.t_before)),
                                      kinematic=(str(times.get(w.after_frame_index)),
                                                 str(times.get(w.before_frame_index))))))
                continue
            ta, tb = float(w.t_after), float(w.t_before)
            window_ev = dict(window_frames=(w.after_frame_index, w.before_frame_index),
                             window_s=(str(w.t_after), str(w.t_before)), event_type=e.type)
            contact_in = False
            if e.type in IMPULSE_EVENT_TYPES:
                imp = o.verdicts["impulse"]
                if imp.verdict == kf.SUPPORTED:
                    lo, hi = o.contact.profile_interval_s
                    ev = dict(window_ev, t_contact_s=o.contact.t_contact_s, contact_interval_s=(lo, hi),
                              contact_gap_frames=o.contact.gap_frames)
                    if ta < lo and hi < tb:
                        contact_in = True
                        explained.add(oid)
                        res, sev = CONSISTENT, INFO
                        msg = (f"{e.event_id} ({e.type}) window ({w.t_after}, {w.t_before}) s, frames "
                               f"{w.after_frame_index}-{w.before_frame_index}, contains the fitted contact-time interval "
                               f"[{lo:.7g}, {hi:.7g}] s (contact in the gap {o.contact.gap_frames}). Consistent with "
                               "the supported impulse; this is kinematic consistency, not proof of a particular contact.")
                    elif hi < ta or lo > tb:
                        res, sev = kf.CONTRADICTED, ERROR
                        msg = (f"{e.event_id} ({e.type}) window ({w.t_after}, {w.t_before}) s excludes the fitted "
                               f"contact-time interval [{lo:.7g}, {hi:.7g}] s of the supported impulse.")
                    else:
                        res, sev = kf.INCONCLUSIVE, INFO
                        msg = (f"{e.event_id} ({e.type}) window ({w.t_after}, {w.t_before}) s overlaps but does not "
                               f"contain the fitted contact-time interval [{lo:.7g}, {hi:.7g}] s.")
                    msg += _label_note(res if res != CONSISTENT else kf.SUPPORTED, e.support)
                    F.append(Finding("EVENT_CONTACT_WINDOW", sev, e.event_id, oid, res, msg, ev))
                    event_outcome.setdefault(e.event_id, []).append(res)
                else:
                    sev = _claim_severity(imp.verdict)
                    F.append(Finding("EVENT_CONTACT_WINDOW", sev, e.event_id, oid, imp.verdict,
                                     f"{e.event_id} ({e.type}) expects a velocity discontinuity of {oid}; kinematic "
                                     f"verdict impulse = {imp.verdict}: {imp.reason}" + _label_note(imp.verdict, e.support),
                                     dict(window_ev, **_numbers(imp.evidence))))
                    event_outcome.setdefault(e.event_id, []).append(imp.verdict)
            bracketed = [tp for tp in o.turning_points if ta < tp.t_s < tb]
            for tp in bracketed:
                extra = ""
                if e.type in IMPULSE_EVENT_TYPES and not contact_in:
                    extra = " An apex is not a bounce or collision."
                F.append(Finding(
                    "EVENT_BRACKETS_TURNING_POINT", INFO, e.event_id, oid, kf.NOT_EVALUATED,
                    f"{e.event_id} ({e.type}) window ({w.t_after}, {w.t_before}) s brackets the fitted {tp.kind} of "
                    f"{oid} at t = {tp.t_s:.6g} s ({tp.image_description}). A turning point such as an apex is a "
                    "smooth kinematic feature, not a physical event or collision; no velocity discontinuity is "
                    "needed there" + (" and none further was detected" if o.additional_discontinuity is None else "")
                    + "." + extra,
                    dict(window_ev, turning_point=tp.kind, t_turning_s=tp.t_s, approx_se_s=tp.approx_se_s,
                         value_px=tp.value_px)))
            if e.type not in IMPULSE_EVENT_TYPES and not bracketed:
                note = ""
                if o.contact is not None and o.verdicts["impulse"].verdict == kf.SUPPORTED:
                    lo, hi = o.contact.profile_interval_s
                    if ta < lo and hi < tb:
                        note = " Its window contains the fitted contact-time interval."
                F.append(Finding("EVENT_NOT_EVALUATED", INFO, e.event_id, oid, kf.NOT_EVALUATED,
                                 f"{e.event_id}: event type {e.type!r} has no kinematic counterpart in this review."
                                 + note, window_ev))

    for oid in interp_ids:
        o = kobj.get(oid)
        if o is not None and o.verdicts["impulse"].verdict == kf.SUPPORTED and oid not in explained:
            F.append(Finding(
                "IMPULSE_WITHOUT_EVENT", WARNING, oid, oid, kf.SUPPORTED,
                f"{oid}: a velocity discontinuity is supported near t = {o.contact.t_contact_s:.6g} s (gap "
                f"{o.contact.gap_frames}), but no {sorted(IMPULSE_EVENT_TYPES)} event hypothesis has a window "
                "containing its contact-time interval.",
                dict(t_contact_s=o.contact.t_contact_s, contact_interval_s=o.contact.profile_interval_s,
                     gap_frames=o.contact.gap_frames)))

    # ---- ambiguities (linked only through about_ids)
    impulse_ids = ({h.hypothesis_id for h in interp.physical_model_hypotheses if h.model == "plane_contact"}
                   | {e.event_id for e in interp.event_hypotheses if e.type in IMPULSE_EVENT_TYPES})
    declared = (set(facts) | {h.hypothesis_id for h in interp.physical_model_hypotheses}
                | {e.event_id for e in interp.event_hypotheses}
                | {i.interaction_id for i in interp.interaction_hypotheses} | set(interp_ids))
    for a in interp.ambiguities:
        for ref in a.about_ids:
            if ref not in declared:
                F.append(Finding("AMBIGUITY_REFERENCE_UNKNOWN", INFO, a.ambiguity_id, None, kf.NOT_EVALUATED,
                                 f"{a.ambiguity_id} refers to {ref!r}, which is not declared in the interpretation.",
                                 dict(reference=ref)))
        decided = {}
        for ref in a.about_ids:
            outcomes = hyp_outcome.get(ref, []) + event_outcome.get(ref, [])
            outcomes = [x for x in outcomes if x in (kf.SUPPORTED, CONSISTENT, kf.CONTRADICTED, kf.NOT_SUPPORTED)]
            if outcomes:
                decided[ref] = outcomes
        if not decided:
            continue
        rejected = {r: v for r, v in decided.items() if any(x in (kf.CONTRADICTED, kf.NOT_SUPPORTED) for x in v)}
        if rejected:
            sev = WARNING
            msg = (f"{a.ambiguity_id} keeps open {sorted(rejected)}, which the kinematic analysis does not support "
                   f"({rejected}). Alternatives relying on them are not supported by the data. ")
        else:
            sev = INFO
            impulse_refs = sorted(r for r in decided if r in impulse_ids)
            msg = (f"{a.ambiguity_id} is about {sorted(decided)}, which the kinematic analysis supports "
                   f"({decided}). Any listed alternative that removes the supported effect is not supported by the "
                   "data" + (f"; for the impulse ({impulse_refs}) an explanation without a velocity discontinuity is "
                             "contradicted" if impulse_refs else "") + ". ")
        msg += "Alternatives are quoted for review, not interpreted: " + "; ".join(repr(x) for x in a.alternatives)
        F.append(Finding("AMBIGUITY_ON_DECIDED_CLAIM", sev, a.ambiguity_id, None,
                         kf.NOT_SUPPORTED if rejected else kf.SUPPORTED, msg,
                         dict(about_ids=list(a.about_ids), decided=decided, alternatives=list(a.alternatives))))

    # ---- direction wording (conservative heuristic, warnings only)
    for f in interp.observed_facts:
        if len(set(f.object_ids)) == 1 and f.object_ids[0] in kobj:
            F.extend(_direction_findings(f.fact_id, f.object_ids[0], f.statement,
                                         {r.frame_index for r in f.evidence}, kobj[f.object_ids[0]]))
    for e in interp.event_hypotheses:
        if len(set(e.object_ids)) == 1 and e.object_ids[0] in kobj:
            F.extend(_direction_findings(e.event_id, e.object_ids[0], e.rationale,
                                         set(range(e.after_frame_index, e.before_frame_index + 1)), kobj[e.object_ids[0]]))
    for h in interp.physical_model_hypotheses:
        if len(set(h.applies_to)) == 1 and h.applies_to[0] in kobj:
            oid = h.applies_to[0]
            frames = {r.frame_index for fid in h.supporting_fact_ids if fid in facts
                      for r in facts[fid].evidence if r.object_id == oid}
            F.extend(_direction_findings(h.hypothesis_id, oid, h.rationale, frames, kobj[oid]))
    for oi in interp.objects:
        if oi.object_id in kobj:
            o = kobj[oi.object_id]
            F.extend(_direction_findings(oi.object_id, oi.object_id, oi.motion_description, set(_frame_times(o)), o))

    counts = {s: sum(1 for f in F if f.severity == s) for s in SEVERITIES}
    if counts[ERROR]:
        assessment = "contradictions_found"
    elif counts[WARNING]:
        assessment = "concerns_found"
    else:
        assessment = "no_contradictions_found"
    note = (f"{counts[ERROR]} error(s), {counts[WARNING]} warning(s), {counts[INFO]} info finding(s). "
            "Absence of errors means only that the listed kinematic checks did not contradict the interpretation; "
            "it does not establish that the interpretation is physically true.")
    return PhysicsReview(METHOD_VERSION, kin.method_version, tuple(F), counts, assessment, note, LIMITATIONS)


def _numbers(evidence: dict[str, Any]) -> dict[str, Any]:
    keep = ("delta_bic", "k_per_s", "k_upper_95_per_s", "k_upper_times_span", "t_contact_s", "gap_frames",
            "profile_interval_s", "a_px_s2", "a_se_px_s2", "magnitude_px_s2", "direction_deg_image",
            "delta_v_px_s", "delta_v_z", "reversal", "base_model", "best_alternative")
    return {k: evidence[k] for k in keep if k in evidence}


def _hypothesis_message(hid: str, model: str, name: str, kv: kf.Verdict, o: kf.ObjectKinematics) -> str:
    base = f"{hid} ({model}) vs kinematic {name} = {kv.verdict}: {kv.reason}."
    if name == "uniform_acceleration" and kv.verdict == kf.SUPPORTED:
        ev = kv.evidence
        return (f"{hid} ({model}): kinematically consistent with a uniform acceleration of "
                f"{ev['magnitude_px_s2']:.6g} px/s^2 at {ev['direction_deg_image']:.2f} deg in image coordinates "
                "(90 deg = down the image). The trajectory alone does not establish that the cause is gravity. "
                f"({kv.reason})")
    if name == "impulse" and kv.verdict == kf.SUPPORTED:
        return (f"{hid} ({model}): an impulse (abrupt velocity change) is supported near t = "
                f"{o.contact.t_contact_s:.6g} s in the gap {o.contact.gap_frames}. This supports a contact candidate; "
                f"it does not identify the surface or prove contact with a particular object. ({kv.reason})")
    if name == "linear_drag" and kv.verdict == kf.NOT_SUPPORTED:
        return (f"{hid} ({model}) is not supported: {kv.reason}. The drag-free model explains the data, and "
                f"the one-sided 95% upper bound is k = {kv.evidence['k_upper_95_per_s']:.3g} 1/s.")
    return base


def _plane_findings(h, o: kf.ObjectKinematics) -> list[Finding]:
    out: list[Finding] = []
    hint = h.plane_hint
    if hint not in PLANE_REVERSAL:
        return out
    out.append(Finding(
        "SURFACE_IDENTITY_NOT_EVALUATED", INFO, h.hypothesis_id, o.object_id, kf.NOT_EVALUATED,
        f"{h.hypothesis_id}: plane_hint {hint!r} is not confirmed. A centroid trajectory carries no surface geometry; "
        "the reversal direction alone cannot identify a floor, ceiling or wall.",
        dict(plane_hint=hint, kinematic_surface_identity=o.verdicts["surface_identity"].verdict)))
    if o.contact is None or o.verdicts["impulse"].verdict != kf.SUPPORTED:
        return out
    axis, expected = PLANE_REVERSAL[hint]
    got = next(a.reversal for a in o.contact.axes if a.axis == axis)
    ev = dict(plane_hint=hint, axis=axis, expected_reversal=expected, observed_reversal=got,
              assumption="the hinted surface's side is taken in image coordinates; camera orientation is unknown")
    if got == expected:
        out.append(Finding("SURFACE_DIRECTION_COMPATIBLE", INFO, h.hypothesis_id, o.object_id, CONSISTENT,
                           f"{h.hypothesis_id}: the observed reversal on {axis} ({got}) is compatible in direction with "
                           f"a {hint} in image coordinates. Direction compatibility is not surface identity.", ev))
    else:
        out.append(Finding("SURFACE_DIRECTION_INCOMPATIBLE", WARNING, h.hypothesis_id, o.object_id, kf.NOT_SUPPORTED,
                           f"{h.hypothesis_id}: a {hint} would produce a {expected} reversal on {axis}, but the fitted "
                           f"impulse shows {got or 'no reversal'} on that axis.", ev))
    return out
