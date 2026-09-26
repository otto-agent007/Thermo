"""Tiny disposable subprocess contract fixture; never scientific evidence."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from thermo_lab.hashing import canonical_json, canonical_sha256


def study_request():
    return {"fixture": "NOT SCIENTIFIC EVIDENCE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir()
    body = {
        "request": study_request(),
        "targets": [{"reading": r, "seed": s} for r in ("A", "B") for s in range(5)],
        "chains": [
            {"reading": r, "seed": s, "cap": c, "method": m}
            for r in ("A", "B")
            for s in range(5)
            for c in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0, 10.0)
            for m in ("constructive", "variational")
        ],
        "fits": {},
        "integrity": {"passed": True, "failures": []},
    }
    body["request_digest"] = canonical_sha256(body["request"])
    body["result_digest"] = canonical_sha256(
        {k: body[k] for k in ("targets", "fits", "chains", "integrity")}
    )
    archive = gzip.compress(canonical_json(body).encode())
    generation = {
        "status": "available",
        "request_digest": body["request_digest"],
        "result_digest": body["result_digest"],
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "runtime": {"fixture": True},
        "generation_seconds": 0.01,
    }
    provenance = {"schema_version": 1, "generation": generation, "replay": {"full": True}}
    if args.replay_from:
        archive = (args.replay_from / "study.json.gz").read_bytes()
        body = json.loads(gzip.decompress(archive))
        generation = json.loads((args.replay_from / "generation-provenance.json").read_text())
        provenance["generation"] = generation
    (args.output_dir / "generation-provenance.json").write_text(canonical_json(generation))
    (args.output_dir / "study.json.gz").write_bytes(archive)
    (args.output_dir / "provenance.json").write_text(canonical_json(provenance))
    (args.output_dir / "summary.md").write_text("Fixture only, zero scientific meaning.\n")
    completion = {
        "status": "meta_ebm_cap_baseline_complete",
        "targets": 10,
        "caps": 9,
        "methods": 2,
        "chains": 180,
        "samples": 0,
        "integrity": True,
        "replayed": True,
        "request_digest": body["request_digest"],
        "result_digest": body["result_digest"],
        "provenance_digest": canonical_sha256(provenance),
    }
    (args.output_dir / "completion.json").write_text(canonical_json(completion))


if __name__ == "__main__":
    main()
