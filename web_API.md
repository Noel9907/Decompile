# Decompile web API (decompile_web/1)

JSON over HTTP, all under `/api`. Interactive schema: `/api/docs`. Errors: `{"detail": str, "stage"?: str,
"details"?: [str]}` with 400 (invalid / refused), 404, 409 (state conflict, job running), 413, 415.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | version, `live_sam_enabled`, `api_key_configured` (boolean only), limits, pricing, stage catalogue, limitations |
| GET | `/api/runs` | run list |
| POST | `/api/runs` | multipart `file` (MP4) + `provenance` (`real`/`synthetic`) → 201 run; starts the ingest job |
| GET | `/api/runs/{id}` | run state (`run.json`): state, video, suitability, stages, plan, sam, reconstructions, job |
| GET | `/api/runs/{id}/events?after=N[&once=1]` | Server-Sent Events, one JSON event per message, `id` = sequence; honours `Last-Event-ID`; `once=1` returns the backlog and closes |
| GET | `/api/runs/{id}/video` | original upload (Range supported) |
| GET | `/api/runs/{id}/frame-table` | exact PTS per frame (`t` exact rational strings, `t_s` floats) |
| GET | `/api/runs/{id}/preview/{index}` | decoded source frame PNG (prompt canvas) |
| POST | `/api/runs/{id}/plan` | `{"prompt": {"kind": "concept", "text"}}` or `{"prompt": {"kind": "objects", "frame", "box": [x0,y0,x1,y1], "points": [{"x","y","negative"?}]}}` → plan with cost; **no request** |
| POST | `/api/runs/{id}/sam/confirm` | `{"confirm": true, "plan_sha256", "acknowledged_cost_usd"}` → 202; the only route that can make a billed SAM request |
| POST | `/api/runs/{id}/sam/import` | multipart `file` = `sam_tracking.json` for the same video; verified; no request |
| GET | `/api/runs/{id}/sam/tracks` | per-object frames, boxes, confidences, summary (no new request) |
| POST | `/api/runs/{id}/reconstructions` | `{"sam_object": "most_frames"|id, "start_frame", "end_frame"|null, "physics_model": "projective"|"image_space", "outline_model", "hfov_deg", "min_confidence"}` → 202 `{recon_id}` |
| GET | `/api/runs/{id}/reconstructions/{rid}` | reconstruction record + `analysis` (per-frame table, engine report, program, fitted simulation) |
| GET | `/api/runs/{id}/reconstructions/{rid}/perception` | SAM boxes, confidences, mask contours per window frame |
| GET | `/api/runs/{id}/reconstructions/{rid}/frames/{index}` | exact decoded frame PNG |
| POST | `/api/runs/{id}/reconstructions/{rid}/counterfactuals` | `{"restitution": 0..1|null, "gravity_scale": 0.1..3|null, "extend_s": 0..3}` → deterministic counterfactual (cached by parameters) |
| GET | `/api/runs/{id}/reconstructions/{rid}/counterfactuals/{cf}` | stored counterfactual |
| POST | `/api/runs/{id}/reconstructions/{rid}/counterfactuals/{cf}/video` | render the hypothetical overlay MP4 → `{path}` |
| GET | `/api/runs/{id}/report[?recon_id=]` | machine-readable run summary |
| GET | `/api/runs/{id}/artifacts` | files that exist: `{path, category, label, size_bytes, scope, media_type}` |
| GET | `/api/runs/{id}/files/{path}[?download=1]` | a listed artifact only |

## Run states

`ingesting → uploaded | rejected → planned → tracking → tracked | tracking_failed → reconstructing → reconstructed`
(reconstruction failures return to `tracked` with the reconstruction marked `failed`).

## Events

`{"seq", "t", "type", ...}` with `type` in `upload`, `job` (`kind`, `status`, `error`), `stage` (`stage`, `status`
= started|progress|done|failed|skipped, `detail`, `recon_id`), `log`, `plan`, `confirmation`, `sam_import`,
`artifact`. Stages: tracking `ingest, timestamps, sam_upload, sam_segmentation, sam_validation`; reconstruction
`frames, measurement, trajectory, physics, scene_program, simulation, rendering, views`.

## Run directory (`runs/web/<run id>/`)

```
run.json, events.jsonl
input/video.mp4, probe.json, frame_table.json, preview/
sam/plan.json, sam/attempt_N/{sam_tracking.json, raw_stream.sse, raw_output.txt, upload_record.json, api_evidence/}
sam/imported_N/sam_tracking.json
recon/rN/   (exactly what tools/sam_demo.py reconstruct writes) + web/{analysis, perception, run_report}.json,
            web/counterfactuals/cf_*.json|mp4
```
