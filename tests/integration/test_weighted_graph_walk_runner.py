from pathlib import Path

import pytest

from thermo_lab.aggregate import CompletionState
from thermo_lab.runner import run_experiment

ROOT = Path(__file__).parents[2]
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"


def test_runner_dispatches_weighted_graph_backend(tmp_path: Path) -> None:
    aggregate = run_experiment(GRAPH_CONFIG, tmp_path)

    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.seeds == (0,)
    assert aggregate.run_record_paths == ("runs/seed-0000000000.json",)


@pytest.mark.parametrize("seeds", [(1,), (0, 1)])
def test_runner_rejects_nonzero_graph_seed_before_touching_outputs(
    tmp_path: Path, seeds: tuple[int, ...]
) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    snapshot = tmp_path / "config.snapshot.toml"
    snapshot.write_text("pre-existing output", encoding="utf-8")

    with pytest.raises(ValueError, match="seed zero"):
        run_experiment(GRAPH_CONFIG, tmp_path, seeds=seeds, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert snapshot.read_text(encoding="utf-8") == "pre-existing output"
    assert not (tmp_path / "aggregate.json").exists()


def test_runner_persists_graph_acceptance_failure(tmp_path: Path) -> None:
    config = tmp_path / "graph-failure.toml"
    config.write_text(
        GRAPH_CONFIG.read_text(encoding="utf-8").replace(
            "finest_final_half_l1_tolerance = 0.003",
            "finest_final_half_l1_tolerance = 1e-12",
        ),
        encoding="utf-8",
    )

    aggregate = run_experiment(config, tmp_path / "output")

    assert aggregate.completion_state is CompletionState.FAILED
    assert aggregate.completed_runs == 0
    assert aggregate.run_record_paths == ()
    assert len(aggregate.failures) == 1
    failure = aggregate.failures[0]
    assert "N=128" in failure.message
    assert "canonical" in failure.message
    assert "value=" in failure.message
    assert "bound=" in failure.message
