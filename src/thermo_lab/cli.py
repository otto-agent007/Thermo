"""Command-line entry point for reproducible smoke experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _nonnegative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed


def _seed_list(value: str) -> tuple[int, ...]:
    if not value:
        raise argparse.ArgumentTypeError("--seeds requires a comma-separated list")
    seeds = tuple(_nonnegative_seed(item.strip()) for item in value.split(","))
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


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
    run = subparsers.add_parser("run", help="execute a checked TOML experiment")
    run.add_argument("config_path", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    seed_group = run.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=_nonnegative_seed)
    seed_group.add_argument("--seeds", type=_seed_list)
    run.add_argument(
        "--allow-accelerator",
        action="store_true",
        help="allow JAX to select a non-CPU device (CPU is the reproducible default)",
    )
    run.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_accelerator:
        os.environ["JAX_PLATFORMS"] = "cpu"

    if args.command == "run":
        from thermo_lab.runner import run_experiment

        selected_seeds = args.seeds
        if args.seed is not None:
            selected_seeds = (args.seed,)
        aggregate = run_experiment(
            args.config_path,
            args.output_dir,
            seeds=selected_seeds,
            overwrite=args.overwrite,
        )
        summary = {
            "aggregate": str(args.output_dir / "aggregate.json"),
            "completed_runs": aggregate.completed_runs,
            "failed_runs": aggregate.failed_runs,
            "report": str(args.output_dir / "report.md"),
            "seeds": aggregate.seeds,
            "status": aggregate.completion_state.value,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if aggregate.completion_state.value == "complete" else 1

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
