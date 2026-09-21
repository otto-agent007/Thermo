"""Canonical repository gate commands; output locations are operational inputs."""

from pathlib import Path

PACKAGING_CODE = """from pathlib import Path
import tarfile, zipfile
configs = sorted(str(p) for p in Path('configs/experiments').glob('*.toml'))
wheel = next(Path('dist').glob('*.whl'))
sdist = next(Path('dist').glob('*.tar.gz'))
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
    assert all(any(n.endswith(c) for n in names) for c in configs)
with tarfile.open(sdist) as z:
    names = z.getnames()
    assert all(any(n.endswith(c) for n in names) for c in configs)
"""


def gate_arguments(name, repository_root):
    """Arguments after uv, with placeholders only for result/source locations."""
    simple = {
        "sync-frozen": ["sync", "--frozen"],
        "lock-check": ["lock", "--check", "--offline"],
        "ruff-format": ["run", "ruff", "format", "--check", "."],
        "ruff-check": ["run", "ruff", "check", "."],
        "build": ["build"],
        "packaging": ["run", "python", "-c", PACKAGING_CODE],
        "tests-unit-upstream": ["run", "pytest", "tests/unit", "tests/upstream_regressions", "-q"],
    }
    if name in simple:
        return simple[name]
    if name in ("tests-integration-a", "tests-integration-b"):
        root = Path(repository_root)
        paths = sorted((root / "tests/integration").glob("test_*.py"))
        index = 0 if name.endswith("-a") else 1
        return ["run", "pytest", *[str(p.relative_to(root)) for p in paths[index::2]], "-q"]
    modules = {
        "quality-budget-full-preflight": "quality_budget_full_preflight",
        "quality-budget-training-runner": "quality_budget_runner_preflight",
        "quality-budget-training-preflight": "quality_budget_training_preflight",
        "quality-budget-preflight": "quality_budget_preflight",
        "conservation-diagnostic": "conservation_audit",
        "local-conservation-tradeoff": "conservation_tradeoff_audit",
        "context-weighted-conservation": "context_conservation_audit",
        "asymmetry-preservation": "asymmetry_preservation_audit",
    }
    if name in modules:
        return ["run", "python", "-m", f"thermo_lab.{modules[name]}", "--output-dir", "<output>"]
    commands = {
        "frozen-pair-finite-sweeps": "audit-finite-sweeps",
        "bounded-finite-sweep-refinement": "refine-finite-sweeps",
        "matched-training-budget": "compare-training-laws",
    }
    if name in commands:
        return [
            "run",
            "thermo-lab",
            commands[name],
            "<source0>",
            "<source1>",
            "<source2>",
            "--output-dir",
            "<output>",
        ]
    if name in ("smoke", "finite-sweep-gradient-contract"):
        command = "smoke" if name == "smoke" else "check-finite-sweep-gradients"
        return ["run", "thermo-lab", command, "--output-dir", "<output>"]
    experiments = {
        "torx-run": ("torx-two-gate", "0,1,2"),
        "thrml-run": ("thrml-ising-chain", "7,8,9,10"),
        "weighted-graph-walk": ("torx-weighted-graph-walk", None),
        "independent-pasym-swap": ("thrml-independent-pasym-swap", "0,1,2"),
        "target-context-pasym-swap": ("thrml-target-context-pasym-swap", "0,1,2"),
        "model-context-pasym-swap": ("thrml-model-context-pasym-swap", "0,1,2"),
        "trajectory-reinforce-estimator": ("numpy-trajectory-reinforce-pasym-swap", "0,1,2"),
        "trajectory-reinforce-one-step": (
            "numpy-trajectory-reinforce-pasym-swap-one-step",
            "0,1,2",
        ),
        "composed-finite-gibbs": ("numpy-composed-pasym-swap-finite-gibbs", "0,1,2"),
        "composed-trajectory-refinement-one-step": (
            "numpy-composed-pasym-swap-trajectory-refinement-one-step",
            "0,1,2",
        ),
    }
    config, seeds = experiments[name]
    return [
        "run",
        "thermo-lab",
        "run",
        f"configs/experiments/{config}.toml",
        *(["--seeds", seeds] if seeds else []),
        "--output-dir",
        "<output>",
    ]


def validate_gate_command(name, command, repository_root):
    args = list(command)
    if args[:2] == ["taskset", "-c"]:
        if len(args) < 4 or not all(c.isdigit() or c in ",-" for c in args[2]):
            raise ValueError("invalid CPU affinity prefix")
        args = args[3:]
    if not args or args.pop(0) != "uv":
        raise ValueError("gate command must use the frozen uv environment")
    expected = gate_arguments(name, repository_root)
    if len(args) != len(expected):
        raise ValueError("gate command differs from required plan")
    for actual, required in zip(args, expected, strict=True):
        if required == "<output>":
            if not actual or actual.startswith("-"):
                raise ValueError("gate output path required")
        elif required.startswith("<source"):
            seed = int(required[-2])
            if Path(actual).name != f"seed-{seed:010d}.json":
                raise ValueError("gate source seed order differs")
        elif actual != required:
            raise ValueError("gate command differs from required plan")
