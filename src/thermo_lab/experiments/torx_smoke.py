"""Exact two-gate Torx smoke specification."""

from thermo_lab.records import ExperimentSpec
from thermo_lab.schemas import JAX_NUMERIC_DTYPE


def torx_smoke_spec(seed: int = 0) -> ExperimentSpec:
    """PNOT(0), then PCNOT(0,1), each applied with probability one-half."""

    return ExperimentSpec(
        experiment_id="torx.two_gate_statevector.v1",
        seed=seed,
        model_config={
            "gates": [
                {"type": "pnot", "sites": [0], "theta": 0.0},
                {"type": "pcnot", "sites": [0, 1], "theta": 0.0},
            ],
            "initial_distribution": [1.0, 0.0, 0.0, 0.0],
            "numeric_dtype": JAX_NUMERIC_DTYPE,
        },
        run_config={
            "expected_distribution": [0.5, 0.0, 0.25, 0.25],
            "absolute_tolerance": 1e-6,
        },
        sample_definition=(
            "Exact final probability mass over basis states [00, 01, 10, 11]; "
            "not a Monte Carlo sample."
        ),
    )
