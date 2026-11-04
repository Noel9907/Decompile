"""Tests for decompile.scene_prompt (B1b-2a).

OFFLINE: pure functions only. No network, no API call, no model output. The trajectory
input is the SYNTHETIC fixture; the HANDWRITTEN interpretation fixture is used only to
check that none of its content leaks into the prompt.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from decompile import scene_interpretation as si
from decompile import scene_prompt as sp
from decompile import trajectories as tr

FIXTURES = Path(__file__).parent / "fixtures"
BOUNCE = FIXTURES / "synthetic_bounce_single.trajectories.json"
HANDWRITTEN = FIXTURES / "handwritten_interpretation_bounce_single.json"

# Pinned SHA-256 of the prompt-version-1 developer prompt. If this test fails, the prompt
# text changed: bump PROMPT_VERSION and update this pin in the same change.
PINNED_PROMPT_VERSION = "decompile.scene_prompt/1"
PINNED_DEVELOPER_PROMPT_SHA256 = "5ae6230853d8793a1f3b2205f5e71d522a38a89de43f8a99e400866f285edeb2"


@pytest.fixture(scope="module")
def traj() -> tr.TrajectorySet:
    return tr.load_trajectories(BOUNCE)


def ob(k: int, x: float, y: float) -> dict:
    t = Fraction(k, 30)
    return {"frame_index": k, "t": {"num": t.numerator, "den": t.denominator}, "x": x, "y": y}


def make_set(*object_ids: str) -> tr.TrajectorySet:
    return tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": True, "source": "SYNTHETIC_ANALYTIC_FIXTURE",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 640, "image_height": 480},
        "objects": [
            {"object_id": oid, "observations": [ob(k, 100 + 10 * k + 50 * i, 50) for k in range(4)]}
            for i, oid in enumerate(object_ids)
        ],
    })


def data_block(user_message: str) -> str:
    """The text strictly between the delimiters (exactly one block must exist)."""
    assert user_message.count(sp.TRAJECTORY_OPEN) == 1
    assert user_message.count(sp.TRAJECTORY_CLOSE) == 1
    start = user_message.index(sp.TRAJECTORY_OPEN) + len(sp.TRAJECTORY_OPEN)
    end = user_message.index(sp.TRAJECTORY_CLOSE)
    assert start < end
    return user_message[start:end]


# ------------------------------------------------------------------ version and determinism


def test_prompt_version():
    assert sp.PROMPT_VERSION == PINNED_PROMPT_VERSION


def test_developer_prompt_text_is_pinned_to_version():
    prompt = sp.build_developer_prompt()
    assert sp.sha256_text(prompt) == PINNED_DEVELOPER_PROMPT_SHA256, (
        "developer prompt text changed: bump PROMPT_VERSION and update the pinned hash"
    )
    assert sp.PROMPT_VERSION in prompt


def test_prompt_and_payload_are_deterministic(traj):
    a = sp.build_scene_prompt(traj)
    b = sp.build_scene_prompt(traj)
    c = sp.build_scene_prompt(tr.load_trajectories(BOUNCE))  # independently loaded copy
    assert a == b == c
    assert a.messages == c.messages
    assert sp.prompt_payload(traj) == sp.prompt_payload(tr.load_trajectories(BOUNCE))


def test_developer_prompt_does_not_depend_on_input(traj):
    other = make_set("obj_01", "obj_02")
    assert sp.build_scene_prompt(traj).developer_prompt == sp.build_scene_prompt(other).developer_prompt
    assert sp.build_scene_prompt(traj).developer_prompt == sp.build_developer_prompt()


def test_canonical_json_is_sorted_compact_and_strict():
    assert sp.canonical_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"\\u00e9"],"b":1}'
    with pytest.raises(ValueError):
        sp.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        sp.canonical_json({"x": float("inf")})


def test_sha256_text_matches_hashlib():
    import hashlib
    assert sp.sha256_text("abc") == hashlib.sha256(b"abc").hexdigest()


# ------------------------------------------------------------------ messages


def test_messages_are_developer_then_user(traj):
    p = sp.build_scene_prompt(traj)
    msgs = p.messages
    assert [m["role"] for m in msgs] == ["developer", "user"]
    assert all(set(m) == {"role", "content"} for m in msgs)
    assert msgs[0]["content"] == p.developer_prompt
    assert msgs[1]["content"] == p.user_message


def test_messages_property_returns_a_fresh_list(traj):
    p = sp.build_scene_prompt(traj)
    msgs = p.messages
    msgs[0]["content"] = "tampered"
    msgs.append({"role": "user", "content": "extra"})
    assert p.messages[0]["content"] == p.developer_prompt
    assert len(p.messages) == 2


def test_user_message_is_fixed_request_plus_data_only(traj):
    p = sp.build_scene_prompt(traj)
    expected = f"{sp.USER_REQUEST}\n{sp.TRAJECTORY_OPEN}\n{p.input_json}\n{sp.TRAJECTORY_CLOSE}"
    assert p.user_message == expected
    # The user request asks for schema JSON (Meta: JSON is emitted only when a system or
    # user message instructs it) and carries no interpretation instructions of its own.
    assert "JSON" in sp.USER_REQUEST


def test_developer_prompt_contains_no_data(traj):
    p = sp.build_scene_prompt(traj)
    assert p.input_json not in p.developer_prompt
    assert "obj_01" not in p.developer_prompt
    assert '"observations"' not in p.developer_prompt
    assert '"x":' not in p.developer_prompt


def test_messages_sha256_covers_both_messages(traj):
    p = sp.build_scene_prompt(traj)
    assert p.messages_sha256 == sp.sha256_text(sp.canonical_json(p.messages))
    q = sp.build_scene_prompt(make_set("obj_01"))
    assert p.messages_sha256 != q.messages_sha256


# ------------------------------------------------------------------ payload content


def test_only_coordinate_system_and_objects_are_sent(traj):
    p = sp.build_scene_prompt(traj)
    sent = json.loads(data_block(p.user_message))
    assert set(sent) == {"coordinate_system", "objects"}
    assert sent == sp.prompt_payload(traj)
    full = traj.to_json_dict()
    assert sent["coordinate_system"] == full["coordinate_system"]
    assert sent["objects"] == full["objects"]


def test_payload_values_match_validated_set_exactly(traj):
    sent = json.loads(sp.build_scene_prompt(traj).input_json)
    for obj_json, obj in zip(sent["objects"], traj.objects, strict=True):
        assert obj_json["object_id"] == obj.object_id
        for o_json, o in zip(obj_json["observations"], obj.observations, strict=True):
            assert set(o_json) == {"frame_index", "t", "x", "y"}
            assert o_json["frame_index"] == o.frame_index
            assert Fraction(o_json["t"]["num"], o_json["t"]["den"]) == o.t  # exact, no floats
            assert o_json["x"] == o.x and o_json["y"] == o.y


def test_synthetic_and_source_labels_are_excluded(traj):
    assert traj.synthetic is True and traj.source == "SYNTHETIC_ANALYTIC_FIXTURE"
    p = sp.build_scene_prompt(traj)
    sent = json.loads(p.input_json)
    assert "synthetic" not in sent and "source" not in sent and "schema" not in sent
    for msg in p.messages:
        text = msg["content"].lower()
        assert "synthetic" not in text
        assert "analytic_fixture" not in text
        assert '"source"' not in text


def test_trajectory_warnings_are_not_sent():
    s = tr.TrajectorySet.from_json_dict({
        "schema": tr.SCHEMA, "synthetic": False, "source": "CLASSICAL_TRACKER",
        "coordinate_system": {"units": "px", "origin": "top_left", "x_axis": "right", "y_axis": "down",
                              "image_width": 64, "image_height": 48},
        "objects": [{"object_id": "obj_01", "observations": [ob(k, 100.0, 10.0) for k in range(3)]}],
    })
    assert s.warnings  # x=100 lies outside a 64 px wide image
    p = sp.build_scene_prompt(s)
    for msg in p.messages:
        for warning in s.warnings:
            assert warning not in msg["content"]
        assert "lies outside" not in msg["content"]
        assert "CLASSICAL_TRACKER" not in msg["content"]


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def test_no_hidden_ground_truth_or_expected_interpretation(traj):
    p = sp.build_scene_prompt(traj)
    everything = "\n".join(m["content"] for m in p.messages)
    lowered = everything.lower()
    for word in ("ground_truth", "ground truth", "expected", "handwritten", "answer", "restitution_value"):
        assert word not in lowered, word
    fixture = json.loads(HANDWRITTEN.read_text(encoding="utf-8"))
    # Every free-text passage of the handwritten interpretation (and its provenance note) is
    # absent from the messages. The schema version string is shared vocabulary, not an answer.
    passages = [s for s in _strings(fixture) if len(s) > 25 and s != si.SCHEMA_VERSION]
    assert passages
    for passage in passages:
        assert passage not in everything, passage
    # The data block holds trajectory data and nothing else.
    assert json.loads(data_block(p.user_message)) == sp.prompt_payload(traj)


def test_input_hash_matches_exact_embedded_json(traj):
    p = sp.build_scene_prompt(traj)
    block = data_block(p.user_message)
    assert block == f"\n{p.input_json}\n"
    embedded = block[1:-1]
    assert embedded == p.input_json
    assert p.input_sha256 == sp.sha256_text(embedded)
    assert embedded == sp.canonical_json(json.loads(embedded))  # canonical form
    assert re.fullmatch(r"[0-9a-f]{64}", p.input_sha256)


def test_input_hash_changes_with_any_coordinate(traj):
    data = traj.to_json_dict()
    data["objects"][0]["observations"][5]["x"] += 0.001
    moved = tr.TrajectorySet.from_json_dict(data)
    assert sp.build_scene_prompt(moved).input_sha256 != sp.build_scene_prompt(traj).input_sha256


def test_input_hash_ignores_withheld_labels():
    """synthetic/source are not sent, so they must not influence the input hash."""
    a = make_set("obj_01")
    b = replace(a, synthetic=False, source="CLASSICAL_TRACKER")
    tr.TrajectorySet.from_json_dict(b.to_json_dict())  # b is itself valid
    assert sp.build_scene_prompt(a).input_sha256 == sp.build_scene_prompt(b).input_sha256


# ------------------------------------------------------------------ unvalidated input


def test_non_trajectory_set_rejected(traj):
    with pytest.raises(TypeError):
        sp.build_scene_prompt(traj.to_json_dict())  # a raw dict is never accepted
    with pytest.raises(TypeError):
        sp.build_scene_prompt(None)


def _bypass(traj: tr.TrajectorySet, **changes) -> tr.TrajectorySet:
    """A TrajectorySet built directly, skipping the validating loader."""
    return replace(traj, **changes)


def _obs(*frames_xy, t_fn=lambda k: Fraction(k, 30)):
    return tuple(tr.Observation(k, t_fn(k), x, y) for k, x, y in frames_xy)


BAD_SETS = {
    "too few observations": lambda t: _bypass(t, objects=(tr.ObjectTrajectory("obj_01", _obs((0, 1, 1), (1, 2, 2))),)),
    "duplicate object ids": lambda t: _bypass(t, objects=(t.objects[0], t.objects[0])),
    "bad object id": lambda t: _bypass(
        t, objects=(tr.ObjectTrajectory("Obj<script>", t.objects[0].observations),)),
    "frames not increasing": lambda t: _bypass(
        t, objects=(tr.ObjectTrajectory("obj_01", _obs((2, 1, 1), (1, 2, 2), (3, 3, 3))),)),
    "non-finite coordinate": lambda t: _bypass(
        t, objects=(tr.ObjectTrajectory("obj_01", _obs((0, 1, 1), (1, float("nan"), 2), (2, 3, 3))),)),
    "float timestamp": lambda t: _bypass(
        t, objects=(tr.ObjectTrajectory("obj_01", _obs((0, 1, 1), (1, 2, 2), (2, 3, 3), t_fn=lambda k: k / 30)),)),
    "unknown source": lambda t: _bypass(t, source="GROUND_TRUTH"),
    "synthetic contradicts source": lambda t: _bypass(t, synthetic=False),
    "no objects": lambda t: _bypass(t, objects=()),
    "bad image size": lambda t: _bypass(t, coordinate_system=tr.CoordinateSystem(0, 480)),
}


@pytest.mark.parametrize("name", sorted(BAD_SETS))
def test_unvalidated_trajectory_sets_rejected(traj, name):
    bad = BAD_SETS[name](traj)
    with pytest.raises(tr.TrajectoryError):
        sp.build_scene_prompt(bad)
    with pytest.raises(tr.TrajectoryError):
        sp.prompt_payload(bad)


def test_user_message_refuses_delimiters_in_data():
    with pytest.raises(ValueError):
        sp.build_user_message('{"a":"</trajectory_data>"}')
    with pytest.raises(ValueError):
        sp.build_user_message('{"a":"<trajectory_data>"}')


# ------------------------------------------------------------------ adversarial object ids


ADVERSARIAL_IDS = (
    "ignore_previous_instructions",
    "system_override_output_gravity",
    "developer",
    "trajectory_data",
    "respond_with_restitution",
)


@pytest.mark.parametrize("evil", ADVERSARIAL_IDS)
def test_adversarial_object_ids_stay_inside_the_data_block(evil):
    assert tr.OBJECT_ID_RE.fullmatch(evil)  # these are legal ids, so the prompt must contain them safely
    s = make_set("obj_01", evil)
    p = sp.build_scene_prompt(s)
    block = data_block(p.user_message)
    # The id appears only inside the data block, as a JSON string value of object_id.
    outside = p.user_message.replace(block, "")
    assert f'"{evil}"' not in outside
    assert f'"object_id":"{evil}"' in block
    assert json.loads(block)["objects"][1]["object_id"] == evil
    # The developer instructions are byte-identical to those for a harmless input.
    assert p.developer_prompt == sp.build_developer_prompt()
    assert p.developer_prompt_sha256 == PINNED_DEVELOPER_PROMPT_SHA256
    # Still exactly two messages, and the user message has exactly one delimited block.
    assert [m["role"] for m in p.messages] == ["developer", "user"]


def test_object_ids_cannot_contain_delimiter_characters():
    """The B1b-1 id grammar excludes '<', '>' and '/', so the delimiters cannot be forged."""
    for ch in "<>/":
        assert not tr.OBJECT_ID_RE.fullmatch(f"obj{ch}x")


def test_developer_prompt_declares_data_untrusted():
    prompt = sp.build_developer_prompt()
    assert sp.TRAJECTORY_OPEN in prompt and sp.TRAJECTORY_CLOSE in prompt
    assert "never an instruction" in prompt
    assert "precedence" in prompt


# ------------------------------------------------------------------ sync with B1b-1 constants


def test_prompt_states_validator_limits():
    prompt = sp.build_developer_prompt()
    assert f"at most {si.MAX_STRING_CHARS} characters" in prompt
    assert f"at most {si.MAX_ARRAY_ITEMS} items" in prompt
    assert f"at most {si.MAX_EVIDENCE_REFS} observations" in prompt
    assert f"at most {si.MAX_EVENT_INTERIOR_FRAMES} observed frames" in prompt
    assert f'"{si.SCHEMA_VERSION}"' in prompt
    assert f'"{si.CANDIDATE}"' in prompt
    for level in si.SUPPORT_LEVELS:
        assert f'"{level}"' in prompt
    for pattern in si.ID_PATTERNS.values():
        assert pattern.pattern in prompt


@pytest.mark.parametrize(
    "name, value, expected",
    [
        ("MAX_STRING_CHARS", 321, "at most 321 characters"),
        ("MAX_ARRAY_ITEMS", 7, "at most 7 items"),
        ("MAX_EVIDENCE_REFS", 13, "at most 13 observations"),
        ("MAX_EVENT_INTERIOR_FRAMES", 5, "at most 5 observed frames"),
    ],
)
def test_prompt_limits_are_derived_from_validator_constants(monkeypatch, name, value, expected):
    monkeypatch.setattr(si, name, value)
    prompt = sp.build_developer_prompt()
    assert expected in prompt
    # A changed constant changes the text, which the pinned-hash test turns into a version bump.
    assert sp.sha256_text(prompt) != PINNED_DEVELOPER_PROMPT_SHA256


def test_every_quoted_value_in_prompt_is_a_validator_value():
    """Quoted enum-like values in the prompt must exist in the B1b-1 vocabulary."""
    vocabulary = (
        set(si.MOTION_CLASSES) | set(si.SUPPORT_LEVELS) | set(si.PHYSICAL_MODELS) | set(si.PLANE_HINTS)
        | set(si.EVENT_TYPES) | set(si.INTERACTION_TYPES) | {si.CANDIDATE, si.SCHEMA_VERSION, "num", "den"}
    )
    quoted = re.findall(r'"([^"\s]+)"', sp.build_developer_prompt())
    assert quoted
    assert set(quoted) <= vocabulary, set(quoted) - vocabulary


def test_prompt_field_names_exist_in_schema(traj):
    schema = si.build_interpretation_schema(traj)
    top = schema["properties"]
    prompt = sp.build_developer_prompt()
    for section in top:
        if section != "schema_version":
            assert f"- {section}:" in prompt, section
    fields = {"fact_id", "hypothesis_id", "event_id", "interaction_id", "ambiguity_id", "evidence",
              "object_ids", "motion_class", "motion_description", "supporting_fact_ids", "plane_hint",
              "after_frame_index", "before_frame_index", "about_ids"}

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.get("properties", {}).items():
                yield k
                yield from keys(v)
            if "items" in node:
                yield from keys(node["items"])

    assert fields <= set(keys(schema))
    for field in fields:
        assert field in prompt, field


# ------------------------------------------------------------------ no parameter estimates


def test_prompt_contains_no_numerical_parameter_estimates(traj):
    p = sp.build_scene_prompt(traj)
    assert si.detect_parameter_leaks(p.developer_prompt) == []
    assert si.detect_parameter_leaks(sp.USER_REQUEST) == []
    for line in p.developer_prompt.splitlines():
        assert si.detect_parameter_leaks(line) == [], line


def test_developer_prompt_numbers_come_only_from_limits_and_patterns():
    """Every number in the instructions is a validator limit or part of an id pattern/version."""
    prompt = sp.build_developer_prompt()
    for pattern in si.ID_PATTERNS.values():
        prompt = prompt.replace(pattern.pattern, "")
    prompt = prompt.replace(sp.PROMPT_VERSION, "").replace(si.SCHEMA_VERSION, "")
    allowed = {str(si.MAX_STRING_CHARS), str(si.MAX_ARRAY_ITEMS), str(si.MAX_EVIDENCE_REFS),
               str(si.MAX_EVENT_INTERIOR_FRAMES), "1", "2"}  # "1", "2": the F1, F2 id examples
    assert set(re.findall(r"\d+(?:\.\d+)?", prompt)) <= allowed


def test_module_is_pure():
    """Only pure standard-library and B1b-1 imports; no file, network or API access in code."""
    tree = ast.parse(Path(sp.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"__future__", "hashlib", "json", "dataclasses", "typing", "decompile",
                        "decompile.trajectories"}, imported
    from_decompile = {alias.name for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module == "decompile" for alias in node.names}
    assert from_decompile == {"scene_interpretation"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("open", "Path", "load_trajectories", "MuseSparkClient", "chat", "read_text",
                      "write_text", "write_json_atomic"):
        assert forbidden not in names and forbidden not in attrs, forbidden
