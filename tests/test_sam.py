"""SAM 3.1 integration, OFFLINE: prompts, request contract, SSE + official parser, client transport/retry policy,
typed tracking result (frame index -> exact PTS, geometry validation), and mask-guided geometric measurement."""

from __future__ import annotations

import dataclasses
import json
import math
import socket
from fractions import Fraction

import httpx2
import numpy as np
import pytest

import sam_fake as fk
from decompile import ball_measure as bm
from decompile import kinematic_fit as kf
from decompile import sam_api as sa
from decompile import sam_measure as sm
from decompile import sam_tracking as st
from decompile import stress_render as sr
from decompile import trajectories as tr

KEY = "sk-test-SECRET-key-123"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    fk.forbid_network(monkeypatch)


# ------------------------------------------------------------------ prompts and request contract


def test_concept_prompt_rules():
    assert sa.ConceptPrompt("  red ball ").text == "red ball"
    for bad in ("", "red ball, blue ball", "ball; cup", "ball + cup", "<0f>1<|box|>", "a b c d e f g"):
        with pytest.raises(sa.PromptError):
            sa.ConceptPrompt(bad)


def test_object_prompt_uses_official_builder_inclusive_coordinates():
    p = sa.ObjectPrompt((640, 360), ((1, 30, (10, 20, 40, 60), ()), (1, 120, None, ((20, 40, "positive"), (30, 50, "negative")))),
                        "forward")
    text = p.input_text()
    assert text == ("<30f>1<|box;x1=10;y1=20;x2=39;y2=59;w=640;h=360|>"
                    "<120f>1<|point;x=20;y=40;w=640;h=360|>-<|point;x=30;y=50;w=640;h=360|>")
    req = sa.build_request(p, "file-1")
    assert req["metadata"] == {"include_confidence": "true", "propagation_direction": "forward"}
    with pytest.raises(sa.PromptError):
        sa.ObjectPrompt((640, 360), ((1, 0, (10, 20, 700, 60), ()),))  # outside the media
    with pytest.raises(sa.PromptError):
        sa.ObjectPrompt((640, 360), ((1, 0, (10, 20, 40, 60), ()),), "backward")  # needs a later frame
    assert sa.prompt_from_json(p.to_json_dict()) == p


def test_request_body_matches_documented_contract():
    req = sa.build_request(sa.ConceptPrompt("tennis ball"), "file-842549258569145", include_confidence=False)
    assert req == {"model": "sam-3.1", "stream": True, "metadata": {"include_confidence": "false"},
                   "input": [{"type": "message", "role": "user", "content": [
                       {"type": "input_text", "text": "tennis ball"},
                       {"type": "input_video", "file_id": "file-842549258569145"}]}]}
    assert all(isinstance(v, str) for v in req["metadata"].values())  # documented: string values


def test_cost_estimate_uses_documented_minimum():
    assert sa.cost_estimate(30)["expected_billed_frames"] == 50
    assert sa.cost_estimate(30)["expected_cost_usd"] == pytest.approx(0.01)
    assert sa.cost_estimate(1000)["expected_cost_usd"] == pytest.approx(0.20)


# ------------------------------------------------------------------ SSE + official parser


def test_sse_parsing_joins_data_lines_skips_comments_and_stops_at_done():
    lines = [": keep-alive", "event: x", 'data: {"type":', 'data: "a"}', "", 'data: {"type": "b"}', "", "data: [DONE]", "",
             'data: {"type": "after"}', ""]
    assert [e["type"] for e in sa.iter_sse_events(lines)] == ["a", "b"]


SIZE = (64, 48)


def _two_frame_lines():
    return [fk.line(7, [fk.disc_record(2, 30.2, 20.7, 5.0, SIZE, conf=0.91), fk.disc_record(0, 10.5, 30.5, 3.0, SIZE, conf=0.4)]),
            fk.line(3, [fk.disc_record(2, 28.0, 19.0, 5.0, SIZE, conf=0.95)])]


def test_parser_keys_by_frame_index_not_stream_order_and_reads_confidence():
    log = sa.StreamLog()
    seen = []
    res = sa.parse_video_events(fk.events(_two_frame_lines()), log=log, on_snapshot=lambda s: seen.append(len(s.records)))
    assert res.outcome.status == "completed"
    boxes = [r for r in res.records if r.kind == "box"]
    assert [(b.frame.frame_index, b.object_id) for b in boxes] == [(7, "2"), (7, "0"), (3, "2")]
    assert boxes[0].confidence == pytest.approx(0.91)
    assert seen == sorted(seen) and seen[-1] == len(res.records)  # cumulative snapshots, never accumulated
    assert log.usage == {"video_frames_processed": 50} and log.response_id == "resp_test"


def test_parser_diagnostics_for_malformed_lines_and_empty_match():
    res = sa.parse_video_events(fk.events(["<4f>garbage\n"] + _two_frame_lines()))
    assert [d.code for d in res.diagnostics] == ["malformed_record"]
    empty = sa.parse_video_events(fk.events([]))
    assert empty.outcome.status == "completed" and empty.records == ()


@pytest.mark.parametrize("terminal,reason", [("failed", "response_failed"), ("refusal", "response_refusal"),
                                             ("error", "response_error")])
def test_parser_terminal_failures_raise(terminal, reason):
    with pytest.raises(sa.SamStreamError) as ei:
        sa.parse_video_events(fk.events(_two_frame_lines(), terminal=terminal))
    assert ei.value.reason == reason


def test_incomplete_and_truncated_streams_are_kept_but_flagged():
    inc = sa.parse_video_events(fk.events(_two_frame_lines(), terminal="incomplete"))
    assert inc.outcome.status == "incomplete" and inc.outcome.reason == "response"
    eof = sa.parse_video_events(fk.events(_two_frame_lines(), terminal="none", done=False))
    assert eof.outcome.status == "incomplete" and eof.outcome.reason == "eof"


def test_completed_without_finalised_lane_is_a_lane_error():
    with pytest.raises(sa.SamStreamError) as ei:
        sa.parse_video_events(fk.events(_two_frame_lines(), done=False))
    assert ei.value.reason == "lane"


# ------------------------------------------------------------------ client (mock transport)


class Recorder:
    def __init__(self, handlers):
        self.handlers = list(handlers)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        h = self.handlers.pop(0)
        return h(request) if callable(h) else h


def client(handlers, tmp_path, **kw):
    rec = Recorder(handlers)
    c = sa.SamClient(KEY, transport=httpx2.MockTransport(rec), evidence_dir=tmp_path / "ev", sleep=lambda s: None, **kw)
    return c, rec


def _no_secret(tmp_path):
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text(encoding="utf-8", errors="replace"), p


def test_missing_key_and_repr(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(sa.MissingAPIKeyError):
        sa.SamClient()
    assert KEY not in repr(sa.SamClient(KEY))


def test_upload_sends_the_file_bytes_as_multipart_user_data(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 4)
    ok = httpx2.Response(200, json={"id": "file-1", "object": "file", "bytes": video.stat().st_size,
                                    "filename": "clip.mp4", "purpose": "user_data", "status": "uploaded"})
    c, rec = client([httpx2.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}}), ok],
                    tmp_path)
    up = c.upload_video(video)
    assert up.file_id == "file-1" and up.bytes == video.stat().st_size and len(rec.requests) == 2
    r = rec.requests[-1]
    assert r.method == "POST" and r.url.path == "/v1/files" and r.headers["authorization"] == f"Bearer {KEY}"
    body = r.read()
    assert b'name="purpose"' in body and b"user_data" in body and video.read_bytes() in body
    _no_secret(tmp_path)


def test_reuse_or_upload_reuses_a_recorded_file_id(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 100)
    record = tmp_path / "upload_record.json"
    up_ok = httpx2.Response(200, json={"id": "file-9", "bytes": 100, "purpose": "user_data"})
    c, rec = client([up_ok, httpx2.Response(200, json={"id": "file-9", "bytes": 100})], tmp_path)
    assert c.reuse_or_upload(video, record).reused is False
    again = c.reuse_or_upload(video, record)
    assert again.reused and again.file_id == "file-9" and rec.requests[-1].method == "GET"


def test_segment_video_streams_and_parses(tmp_path):
    sse = fk.sse_bytes(fk.events(_two_frame_lines(), usage={"video_frames_processed": 50}))
    c, rec = client([httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse)], tmp_path)
    call = c.segment_video("file-1", sa.ConceptPrompt("ball"), raw_sse_path=tmp_path / "raw.sse")
    assert call.result.outcome.status == "completed" and call.stream.usage == {"video_frames_processed": 50}
    sent = json.loads(rec.requests[0].read())
    assert sent["model"] == "sam-3.1" and sent["stream"] is True and rec.requests[0].headers["accept"] == "text/event-stream"
    assert (tmp_path / "raw.sse").read_text(encoding="utf-8").count("response.output_text.delta") >= 2
    _no_secret(tmp_path)


def test_server_error_is_not_retried_and_key_is_scrubbed(tmp_path):
    c, rec = client([httpx2.Response(500, json={"error": {"message": f"boom {KEY}"}})], tmp_path)
    with pytest.raises(sa.SamHTTPError) as ei:
        c.segment_video("file-1", sa.ConceptPrompt("ball"))
    assert ei.value.status == 500 and len(rec.requests) == 1 and KEY not in str(ei.value)
    _no_secret(tmp_path)


def test_bad_request_reports_api_message(tmp_path):
    err = {"error": {"code": "invalid_request", "message": "invalid prompt: object prompts cannot be combined with a phrase"}}
    c, _ = client([httpx2.Response(400, json=err)], tmp_path)
    with pytest.raises(sa.SamHTTPError, match="cannot be combined") as ei:
        c.segment_video("file-1", sa.ConceptPrompt("ball"))
    assert ei.value.code == "invalid_request"


def test_connect_error_is_retried_but_read_error_is_not(tmp_path):
    def connect_fail(req):
        raise httpx2.ConnectError("refused", request=req)
    sse = fk.sse_bytes(fk.events(_two_frame_lines()))
    c, rec = client([connect_fail, httpx2.Response(200, content=sse)], tmp_path)
    assert c.segment_video("file-1", sa.ConceptPrompt("ball")).result.outcome.status == "completed"
    assert len(rec.requests) == 2

    def read_fail(req):
        raise httpx2.ReadTimeout("silence", request=req)
    c2, rec2 = client([read_fail, httpx2.Response(200, content=sse)], tmp_path / "b")
    with pytest.raises(sa.SamTransportError) as ei:
        c2.segment_video("file-1", sa.ConceptPrompt("ball"))
    assert ei.value.sent is None and len(rec2.requests) == 1


def test_stream_failure_events_raise_from_client(tmp_path):
    sse = fk.sse_bytes(fk.events(_two_frame_lines(), terminal="failed"))
    c, _ = client([httpx2.Response(200, content=sse)], tmp_path)
    with pytest.raises(sa.SamStreamError) as ei:
        c.segment_video("file-1", sa.ConceptPrompt("ball"))
    assert ei.value.reason == "response_failed" and ei.value.evidence_path is not None


# ------------------------------------------------------------------ typed tracking result


def _table(n=10, size=SIZE, **kw):
    return fk.frame_table(n, *size, pts_steps=[3000, 3011], **kw)


def _tracking(lines, table, **kw):
    res = sa.parse_video_events(fk.events(lines, **kw))
    return st.from_parser_result(res, table, prompt={"kind": "concept", "text": "ball"},
                                 video={"sha256": table.source_sha256, "filename": "synthetic.mp4", "size_bytes": 1})


def test_tracking_maps_frame_index_to_exact_pts_and_round_trips():
    table = _table()
    t = _tracking(_two_frame_lines(), table)
    assert t.completed and t.object_ids() == ("0", "2")
    assert sorted(t.frames_of("2")) == [3, 7] and list(t.frames_of("0")) == [7]
    assert t.frame_time(7) == table.frames[7].t != Fraction(7, 30)  # real PTS, not nominal fps
    assert t.frame_times["7"]["pts"] == table.frames[7].pts
    r = t.frames_of("2")[7]
    assert r.placement(*SIZE) == "box_local" and r.box_confidence == pytest.approx(0.91)
    m = r.full_mask(*SIZE)
    assert m.shape == (SIZE[1], SIZE[0]) and m.sum() > 0
    ys, xs = np.nonzero(m)
    assert r.box == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)  # half-open box = mask bounds
    again = st.SamTrackingResult.from_json_dict(json.loads(json.dumps(t.to_json_dict())))
    assert again == t


def test_tracking_rejects_frames_outside_the_table_size_mismatch_and_wrong_video():
    with pytest.raises(st.SamTrackingError, match="not in the source frame table"):
        _tracking([fk.line(12, [fk.disc_record(1, 30, 20, 5, SIZE)])], _table(n=10))
    rotated = _table(size=(48, 64))  # same pixels count, rotated display: SAM reports w=64;h=48
    with pytest.raises(st.SamTrackingError):
        _tracking([fk.line(1, [fk.disc_record(1, 20, 20, 5, SIZE)])], rotated)
    res = sa.parse_video_events(fk.events(_two_frame_lines()))
    with pytest.raises(st.SamTrackingError, match="SHA-256"):
        st.from_parser_result(res, _table(), prompt={}, video={"sha256": "b" * 64})


def test_tracking_keeps_latest_mask_revision_and_incomplete_outcome():
    lines = [fk.line(2, [fk.disc_record(1, 30, 20, 5, SIZE)]), fk.line(2, [fk.disc_record(1, 31, 20, 5, SIZE)])]
    t = _tracking(lines, _table(), terminal="incomplete")
    assert t.superseded_mask_revisions == 1 and t.frames_of("1")[2].mask_revision == 2
    assert not t.completed and t.outcome["reason"] == "response"


# ------------------------------------------------------------------ mask-guided measurement


@pytest.fixture(scope="module")
def scene():
    base = [s for s in sr.stress_suite(24) if s.name == "perspective_pitched"][0]
    return sr.render(dataclasses.replace(base, noise_sigma=2.0))


def _scene_tracking(v, *, bias=(1.6, -1.2), grow=1.5, skip=(), conf=0.9, extra=None):
    """SAM-like masks that are deliberately BIASED (shifted and dilated) relative to the true ball outline."""
    cam = v.scene.camera
    size = (cam.width, cam.height)
    lines = []
    for tt in reversed(v.truth):  # stream order != chronological order
        if tt.frame_index in skip:
            continue
        recs = [fk.disc_record(1, tt.x + bias[0], tt.y + bias[1], tt.radius_px + grow, size, conf=conf)]
        if extra is not None:
            recs += extra(tt, size)
        lines.append(fk.line(tt.frame_index, recs))
    table = fk.frame_table(len(v.frames), *size, time_base=Fraction(1, 30), pts_steps=[1])
    return _tracking(lines, table), table


def _source(v, table):
    return lambda i: (table.frames[i], v.frames[i])


def test_measurement_corrects_biased_masks_with_original_pixels(scene):
    trk, table = _scene_tracking(scene)
    res = sm.measure_track(trk, _source(scene, table), list(range(len(scene.frames))), sam_object_id="1")
    meas = res.measured()
    assert len(meas) >= 0.8 * len(scene.frames)
    truth = {t.frame_index: t for t in scene.truth}
    err = [math.hypot(f.x - truth[f.frame_index].x, f.y - truth[f.frame_index].y) for f in meas]
    mask_err = [math.hypot(f.mask_centroid[0] - truth[f.frame_index].x, f.mask_centroid[1] - truth[f.frame_index].y)
                for f in meas]
    assert np.median(mask_err) > 1.5  # the SAM-like mask centroid is biased by construction
    assert np.median(err) < 0.4 and max(err) < 1.0  # the outline measurement is not
    for f in meas:  # uncertainty comes from the outline method and includes its systematic floor
        assert f.sigma_x >= bm.EdgeParams().sys_floor_px and f.sigma_diameter_px > 0
        assert abs(f.diameter_px - 2 * truth[f.frame_index].radius_px) < 0.1 * f.diameter_px


def test_missing_uncertain_and_low_confidence_frames_get_no_coordinate(scene):
    trk, table = _scene_tracking(scene, skip={5, 6})
    res = sm.measure_track(trk, _source(scene, table), list(range(10)), sam_object_id="1")
    by = {f.frame_index: f for f in res.frames}
    assert by[5].status == by[6].status == sm.MISSING and by[5].x is None
    low, _ = _scene_tracking(scene, conf=0.3)
    r2 = sm.measure_track(low, _source(scene, table), [0, 1], sam_object_id="1")
    assert {f.status for f in r2.frames} == {sm.UNCERTAIN} and all(f.x is None for f in r2.frames)
    traj = res.to_trajectory_set(synthetic=True)
    assert [o.frame_index for o in traj.objects[0].observations] == [f.frame_index for f in res.measured()]
    assert traj.source == "SAM_3_1"
    o = traj.objects[0].observations[0]
    assert o.t == table.frames[o.frame_index].t and o.sigma_x is not None and o.diameter_px is not None


def test_a_mask_on_the_wrong_object_is_not_measured(scene):
    trk, table = _scene_tracking(scene, bias=(60.0, 40.0), grow=0.0)  # mask on plain background, far from the ball
    res = sm.measure_track(trk, _source(scene, table), list(range(6)), sam_object_id="1")
    assert all(f.status != sm.MEASURED for f in res.frames)


def test_mask_touching_the_border_is_uncertain():
    rgb = np.full((48, 64, 3), 120, np.uint8)
    table = _table()
    t = _tracking([fk.line(0, [fk.record(1, (0, 10, 8, 20), np.ones((10, 8), np.uint8), SIZE)])], table)
    f = sm.measure_frame(rgb, t.frames_of("1")[0], frame_index=0, t=Fraction(0), object_id="1")
    assert f.status == sm.UNCERTAIN and "border" in f.reason


def test_other_objects_are_excluded_from_the_outline_background(scene):
    def neighbour(tt, size):
        return [fk.disc_record(2, tt.x + 2.6 * tt.radius_px, tt.y, tt.radius_px, size)]
    trk, table = _scene_tracking(scene, extra=neighbour)
    res = sm.measure_track(trk, _source(scene, table), list(range(8)), sam_object_id="1")
    # rays into the neighbour's (dilated) mask are dropped; the measurement remains outline-based or is refused
    for f in res.frames:
        assert f.status in (sm.MEASURED, sm.UNCERTAIN)
        if f.status == sm.MEASURED:
            assert f.outline["n_edges"] < f.outline["n_rays"]


def test_object_selection_rules(scene):
    def neighbour(tt, size):
        return [fk.disc_record(2, 40.0, 40.0, 4.0, size)] if tt.frame_index < 3 else []
    trk, _ = _scene_tracking(scene, extra=neighbour)
    frames = list(range(len(scene.frames)))
    assert sm.select_object(trk, frames, "most_frames") == "1"
    assert sm.select_object(trk, frames, {"id": 2}) == "2"
    with pytest.raises(st.SamTrackingError):
        sm.select_object(trk, frames, {"id": 7})
    empty = _tracking([], _table())
    with pytest.raises(st.SamTrackingError, match="empty match"):
        sm.select_object(empty, [0], "most_frames")


def test_incomplete_responses_are_not_measured(scene):
    trk, table = _scene_tracking(scene)
    inc = dataclasses.replace(trk, outcome={"status": "incomplete", "reason": "eof", "detail": None})
    with pytest.raises(st.SamTrackingError, match="incomplete"):
        sm.measure_track(inc, _source(scene, table), [0, 1], sam_object_id="1")


def test_sam_trajectory_feeds_the_kinematic_fit_with_per_frame_sigma(scene):
    trk, table = _scene_tracking(scene)
    res = sm.measure_track(trk, _source(scene, table), list(range(len(scene.frames))), sam_object_id="1")
    traj = res.to_trajectory_set(synthetic=True)
    again = tr.TrajectorySet.from_json_dict(json.loads(json.dumps(traj.to_json_dict())))
    assert again == traj
    rep = kf.analyze_kinematics(traj, weights="per_observation")
    assert rep.objects[0].n_observations == len(res.measured())
