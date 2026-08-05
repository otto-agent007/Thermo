"""Command-line entry point for reproducible smoke experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _repository_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
            return candidate
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermo-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run exact Torx and local THRML checks")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--samples", type=int, default=2_500)
    smoke.add_argument(
        "--allow-accelerator",
        action="store_true",
        help="allow JAX to select a non-CPU device (CPU is the reproducible default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_accelerator:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # Import only after selecting the default JAX platform.
    from thermo_lab.backends import ThrmlLocalBackend, TorxStateVectorBackend
    from thermo_lab.experiments import ising_chain_spec, torx_smoke_spec

    repository_root = _repository_root(Path.cwd())
    torx_record = TorxStateVectorBackend(repository_root).run(torx_smoke_spec(args.seed))
    thrml_record = ThrmlLocalBackend(repository_root).run(
        ising_chain_spec(seed=args.seed + 1, n_samples=args.samples)
    )

    output_dir: Path = args.output_dir
    torx_path = output_dir / "torx-statevector.json"
    thrml_path = output_dir / "thrml-ising-chain.json"
    torx_record.write_json(torx_path)
    thrml_record.write_json(thrml_path)
    summary = {
        "status": "passed",
        "records": [str(torx_path), str(thrml_path)],
        "torx_max_abs_error": torx_record.metrics["max_abs_error_vs_analytic"].value,
        "thrml_max_marginal_error": thrml_record.metrics["max_marginal_error"].value,
        "thrml_total_variation": thrml_record.metrics["empirical_total_variation"].value,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
