"""Prepare, execute and inspect a reviewed M5a cycle; no service activation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thermo_lab.hashing import canonical_json
from thermo_lab.research_cycle import worker
from thermo_lab.research_cycle.contracts import Request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare", help="prepare a request for owner review")
    prepare.add_argument("--repo", required=True, type=Path)
    prepare.add_argument("--job-id", required=True)
    prepare.add_argument("--question", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--workers", default=2, type=int)
    prepare.add_argument("--budget-seconds", default=21600, type=int)
    prepare.add_argument("--max-attempts", default=2, type=int)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--repo", required=True, type=Path)
        command.add_argument("--request", required=True, type=Path)
        if name == "run":
            command.add_argument("--state-root", required=True, type=Path)
    for name in ("inspect", "verify", "mirror"):
        command = commands.add_parser(name)
        command.add_argument("--state-root", required=True, type=Path)
        command.add_argument("--job-id", required=True)
        if name == "mirror":
            command.add_argument("--directory", required=True, type=Path)
            command.add_argument("--connected-uri")
            command.add_argument("--verified-by")
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
            request = worker.prepare_request(
                args.repo,
                args.job_id,
                args.question,
                workers=args.workers,
                budget_seconds=args.budget_seconds,
                max_attempts=args.max_attempts,
            )
            worker.write_json(args.output, request.to_dict(), immutable=True)
            result = {"request": str(args.output), "request_digest": request.digest}
        elif args.operation == "validate":
            request = Request.from_dict(worker.read_json(args.request))
            worker.validate_checkout(args.repo, request)
            if (
                worker._scientific_identity(args.repo.resolve())
                != request.scientific_request_digest
            ):
                raise ValueError("scientific request differs from reviewed source")
            result = {"valid": True, "request_digest": request.digest}
        elif args.operation == "run":
            result = worker.run(args.request, args.repo, args.state_root)
        else:
            result = worker.inspect(args.state_root, args.job_id)
            job = args.state_root / args.job_id
            if args.operation == "verify":
                manifest = worker.verify_evidence(job)
                result = {"valid": True, "evidence_digest": worker.canonical_sha256(manifest)}
            elif args.operation == "mirror":
                result = worker.verify_mirror(
                    job,
                    args.directory,
                    connected_uri=args.connected_uri,
                    verified_by=args.verified_by,
                )
        print(canonical_json(result))
        return 1 if result.get("phase") in ("failed", "blocked") else 0
    except (ValueError, OSError) as exc:
        print(f"research-cycle: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
