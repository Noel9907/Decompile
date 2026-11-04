"""B1b-3 LIVE: one real Muse Spark scene-interpretation call on the SYNTHETIC bounce fixture.

Pipeline (all existing code, nothing re-implemented here):
    tests/fixtures/synthetic_bounce_single.trajectories.json   (SYNTHETIC input)
      -> decompile.scene_interpreter.interpret_scene()
           build_scene_prompt -> build_interpretation_schema -> MuseSparkClient.chat()
           -> validate_interpretation -> parse_interpretation -> validation sidecar

Exactly one outbound request:
- The SDK's own retries are already disabled by B1a (max_retries=0).
- B1a's retry loop reads meta_api.MAX_ATTEMPTS at call time. This runner sets it to 1 in
  its own process before the client is used (force_single_attempt), so a 429/500/503 or
  a connect failure is reported, never retried. B1a source is not modified, and the B1a
  evidence record shows retry_policy.max_attempts == 1.
- After the call the runner re-reads the evidence record and reports how many attempts
  it contains.
- The default transport is used (no injected http_client), so the evidence record
  truthfully says transport "network".

Safety:
- Without --live this is a dry run: it prints the configuration and hashes and sends nothing.
- With --live it refuses to run if runs/scene_interpretation/ already holds a sidecar,
  unless --allow-additional-run is given, so a second experiment is never made by accident.
- The API key is read from the environment or .env (python-dotenv, as tools/meta_smoke.py)
  and never printed.

Usage:
    .venv\\Scripts\\python tools\\scene_interpret_live.py            # dry run, no request
    .venv\\Scripts\\python tools\\scene_interpret_live.py --live     # the one real call
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from decompile import meta_api  # noqa: E402
from decompile import scene_interpretation as si  # noqa: E402
from decompile import scene_interpreter as sint  # noqa: E402
from decompile.scene_prompt import build_scene_prompt, canonical_json, sha256_text  # noqa: E402
from decompile.trajectories import load_trajectories  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_bounce_single.trajectories.json"
EVIDENCE_DIR = ROOT / "runs" / "api_evidence"
SIDECAR_DIR = ROOT / "runs" / "scene_interpretation"
MODEL = "muse-spark-1.3"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 8192
EXPECTED_ENDPOINT = "https://api.meta.ai/v1/chat/completions"


def force_single_attempt() -> None:
    """Cap B1a's retry loop at one attempt for this process."""
    meta_api.MAX_ATTEMPTS = 1
    if meta_api.MAX_ATTEMPTS != 1:  # pragma: no cover - defensive
        raise RuntimeError("could not force a single attempt")


def existing_sidecars() -> list[Path]:
    return sorted(SIDECAR_DIR.glob("*.json")) if SIDECAR_DIR.exists() else []


def evidence_attempts(path: Path | None) -> int | None:
    if path is None or not Path(path).is_file():
        return None
    record = meta_api.strict_json_loads(Path(path).read_text(encoding="utf-8"))
    return len(record.get("attempts", []))


def describe_result(result: sint.SceneInterpretationResult) -> None:
    interp = result.interpretation
    u = result.usage
    print("[PASS] structured output conforms to the schema and passed semantic validation")
    print(f"       response_provenance={result.response_provenance}")
    print(f"       requested_model={result.requested_model} returned_model={result.model!r}")
    print(f"       response_id={result.response_id} request_id={result.request_id} attempts={result.attempts}")
    print(f"       latency_s={result.latency_s}")
    print(f"       tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} "
          f"reasoning={u.reasoning_tokens} cached={u.cached_tokens} total={u.total_tokens}")
    print("  objects:")
    for o in interp.objects:
        print(f"    {o.object_id}: motion_class={o.motion_class} facts={list(o.supporting_fact_ids)}")
        print(f"      {o.motion_description}")
    print(f"  observed_facts: {len(interp.observed_facts)}")
    print("  physical_model_hypotheses:")
    for h in interp.physical_model_hypotheses:
        print(f"    {h.hypothesis_id}: {h.model} plane_hint={h.plane_hint} applies_to={list(h.applies_to)} "
              f"support={h.support} supporting={list(h.supporting_fact_ids)} "
              f"contradicting={list(h.contradicting_fact_ids)}")
    print("  event_hypotheses:")
    for e in interp.event_hypotheses:
        print(f"    {e.event_id}: {e.type} objects={list(e.object_ids)} bracket=({e.after_frame_index}, "
              f"{e.before_frame_index}) support={e.support}")
    print("  event_windows (derived from the input timestamps, exact):")
    for w in interp.event_windows:
        print(f"    {w.event_id}: frames {w.after_frame_index}..{w.before_frame_index} "
              f"t in ({w.t_after}, {w.t_before}) s, interior frames {list(w.interior_frame_indices)}")
    print(f"  interaction_hypotheses: {len(interp.interaction_hypotheses)}")
    print(f"  ambiguities: {len(interp.ambiguities)}")
    print(f"  data_quality_notes: {len(interp.data_quality_notes)}")
    print(f"  validation warnings: {len(result.warnings)}")
    for w in result.warnings:
        print(f"    - {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="send the one real request (default: dry run)")
    parser.add_argument("--allow-additional-run", action="store_true",
                        help="permit a live run although a sidecar already exists")
    args = parser.parse_args(argv)

    traj = load_trajectories(FIXTURE)
    prompt = build_scene_prompt(traj)
    schema_sha256 = sha256_text(canonical_json(si.build_interpretation_schema(traj)))
    print("B1b-3 scene interpretation " + ("LIVE (one real Meta API request)" if args.live else "DRY RUN (no request)"))
    print(f"  input fixture: {FIXTURE.relative_to(ROOT)} (SYNTHETIC; provenance withheld from the model)")
    print(f"  object_ids={list(traj.object_ids)}")
    print(f"  prompt_version={prompt.prompt_version}")
    print(f"  developer_prompt_sha256={prompt.developer_prompt_sha256}")
    print(f"  input_sha256={prompt.input_sha256}")
    print(f"  messages_sha256={prompt.messages_sha256}")
    print(f"  schema_sha256={schema_sha256}")
    print(f"  model={MODEL} reasoning_effort={REASONING_EFFORT} max_completion_tokens={MAX_COMPLETION_TOKENS}")
    print(f"  endpoint={EXPECTED_ENDPOINT}")

    if not args.live:
        print("DRY RUN: no client created, no request sent.")
        return 0

    existing = existing_sidecars()
    if existing and not args.allow_additional_run:
        print(f"[REFUSED] {len(existing)} sidecar(s) already in {SIDECAR_DIR}; "
              "the single B1b-3 experiment has already been made. No request sent.")
        return 2

    force_single_attempt()
    print(f"  meta_api.MAX_ATTEMPTS={meta_api.MAX_ATTEMPTS} (SDK max_retries=0)")
    load_dotenv(ROOT / ".env", override=False)
    try:
        client = meta_api.MuseSparkClient(evidence_dir=EVIDENCE_DIR, model=MODEL)
    except meta_api.MissingAPIKeyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    endpoint = f"{client.base_url.rstrip('/')}/chat/completions"
    if endpoint != EXPECTED_ENDPOINT:
        print(f"[REFUSED] client endpoint {endpoint!r} != {EXPECTED_ENDPOINT!r}. No request sent.")
        return 1
    print(f"  client={client!r}")

    evidence_path = None
    try:
        result = sint.interpret_scene(
            traj, client, sidecar_dir=SIDECAR_DIR,
            max_completion_tokens=MAX_COMPLETION_TOKENS, reasoning_effort=REASONING_EFFORT,
        )
    except sint.InterpretationRejectedError as exc:
        evidence_path = exc.evidence_path
        print(f"[FAIL] semantic validation rejected the response ({len(exc.errors)} error(s), "
              f"{len(exc.warnings)} warning(s))")
        for e in exc.errors:
            print(f"    error: {e}")
        for w in exc.warnings:
            print(f"    warning: {w}")
        print(f"       evidence={exc.evidence_path}\n       sidecar={exc.sidecar_path}")
        code = 1
    except sint.InterpretationAPIError as exc:
        evidence_path = exc.evidence_path
        print(f"[FAIL] {type(exc.api_error).__name__}: {exc.api_error}")
        print(f"       evidence={exc.evidence_path}\n       sidecar={exc.sidecar_path}")
        code = 1
    except sint.SidecarWriteError as exc:
        evidence_path = exc.evidence_path
        print(f"[FAIL] sidecar not written (status {exc.status}): {exc}")
        print(f"       evidence={exc.evidence_path}")
        code = 1
    else:
        evidence_path = result.evidence_path
        describe_result(result)
        print(f"  evidence={result.evidence_path}")
        print(f"  sidecar={result.sidecar_path}")
        code = 0
    print(f"  attempts recorded in B1a evidence: {evidence_attempts(evidence_path)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
