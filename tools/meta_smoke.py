"""B1a LIVE smoke test against the real Meta Model API (network, costs a few tokens).

Makes exactly two calls through decompile.meta_api.MuseSparkClient:
  1. plain text      -> expects a non-empty reply with finish_reason "stop"
  2. structured JSON -> response_format json_schema strict=true; the reply must parse
                        and conform to SMOKE_SCHEMA (checked locally again here)

Every call writes a raw evidence record under runs/api_evidence/ (gitignored).
The API key is read from the environment, or from .env via python-dotenv (already a
pinned dependency). It is never printed.

Usage:  .venv\\Scripts\\python tools\\meta_smoke.py [--max-completion-tokens 512] [--reasoning-effort low]
Exit code 0 only if both calls succeed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from decompile import meta_api  # noqa: E402

EVIDENCE_DIR = ROOT / "runs" / "api_evidence"

SMOKE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "sum": {"type": "integer"},
        "primary_colors_of_light": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["status", "sum", "primary_colors_of_light"],
    "additionalProperties": False,
}


def describe(label: str, result: meta_api.ChatResult) -> None:
    u = result.usage
    print(f"[PASS] {label}")
    print(f"       model={result.model} finish_reason={result.finish_reason} attempts={result.attempts}")
    print(f"       response_id={result.response_id} request_id={result.request_id} latency_s={result.latency_s}")
    print(
        f"       tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} "
        f"reasoning={u.reasoning_tokens} cached={u.cached_tokens} total={u.total_tokens}"
    )
    print(f"       content={result.content!r}")
    print(f"       evidence={result.evidence_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default=meta_api.DEFAULT_REASONING_EFFORT)
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env", override=False)
    print("LIVE Meta Model API smoke test (real network calls)")
    try:
        client = meta_api.MuseSparkClient(evidence_dir=EVIDENCE_DIR)
    except meta_api.MissingAPIKeyError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print(f"client={client!r} max_completion_tokens={args.max_completion_tokens} reasoning_effort={args.reasoning_effort}")

    failures = 0
    common = {"max_completion_tokens": args.max_completion_tokens, "reasoning_effort": args.reasoning_effort}

    try:
        result = client.chat(
            [{"role": "user", "content": "Reply with exactly this text and nothing else: DECOMPILE_API_OK"}],
            purpose="b1a-smoke-plain",
            **common,
        )
        describe("plain-text call", result)
    except meta_api.MetaAPIError as exc:
        failures += 1
        print(f"[FAIL] plain-text call: {type(exc).__name__}: {exc}\n       evidence={exc.evidence_path}")

    try:
        result = client.chat(
            [
                {
                    "role": "user",
                    "content": "Set status to ok, sum to 17 + 25, and list the three primary colors of light.",
                }
            ],
            purpose="b1a-smoke-structured",
            json_schema=SMOKE_SCHEMA,
            schema_name="decompile_smoke",
            **common,
        )
        violations = meta_api.validate_instance(result.parsed, SMOKE_SCHEMA)
        if violations:  # the client already raises on this; re-checked here explicitly
            raise meta_api.StructuredOutputError("; ".join(violations), evidence_path=result.evidence_path)
        describe("structured json_schema call", result)
        print(f"       parsed={result.parsed} schema_violations={violations}")
        print(f"       note: sum == 42 is {result.parsed['sum'] == 42} (content check, informational)")
    except meta_api.MetaAPIError as exc:
        failures += 1
        print(f"[FAIL] structured call: {type(exc).__name__}: {exc}\n       evidence={exc.evidence_path}")

    print("\nALL LIVE SMOKE CHECKS PASSED" if failures == 0 else f"\n{failures} LIVE SMOKE CHECK(S) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
