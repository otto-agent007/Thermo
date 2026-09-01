import subprocess
import sys
from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord
from thermo_lab.cli import main

ROOT = Path(__file__).parents[2]
TORX_CONFIG = ROOT / "configs/experiments/torx-two-gate.toml"


def test_run_cli_uses_config_seed_by_default(tmp_path: Path) -> None:
    result = main(["run", str(TORX_CONFIG), "--output-dir", str(tmp_path)])

    assert result == 0
    assert (tmp_path / "runs/seed-0000000000.json").exists()


def test_run_cli_accepts_ordered_seed_list(tmp_path: Path) -> None:
    result = main(
        [
            "run",
            str(TORX_CONFIG),
            "--seeds",
            "7,2,5",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    aggregate = AggregateRecord.model_validate_json(
        (tmp_path / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate.seeds == (7, 2, 5)
    assert aggregate.run_record_paths == (
        "runs/seed-0000000007.json",
        "runs/seed-0000000002.json",
        "runs/seed-0000000005.json",
    )


def test_seed_and_seeds_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                str(TORX_CONFIG),
                "--seed",
                "1",
                "--seeds",
                "2,3",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize("value", ["1,1", "", "1,-2", "1,nope"])
def test_seed_list_rejects_duplicates_and_invalid_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                str(TORX_CONFIG),
                "--seeds",
                value,
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert error.value.code == 2


def test_existing_smoke_command_regression(tmp_path: Path) -> None:
    result = main(["smoke", "--output-dir", str(tmp_path), "--samples", "1500"])

    assert result == 0
    assert (tmp_path / "torx-statevector.json").exists()
    assert (tmp_path / "thrml-ising-chain.json").exists()


def test_smoke_rejects_negative_seed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["smoke", "--output-dir", str(tmp_path), "--seed", "-1"])

    assert error.value.code == 2


def test_cli_and_runner_import_without_loading_jax() -> None:
    """The CLI must be able to select the CPU platform before JAX is first imported."""

    script = (
        "import sys, thermo_lab.cli, thermo_lab.runner, thermo_lab.provenance; "
        "sys.exit(1 if 'jax' in sys.modules else 0)"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False)

    assert completed.returncode == 0
