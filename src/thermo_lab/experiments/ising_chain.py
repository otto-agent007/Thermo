"""Small Ising-chain exact-versus-THRML validation."""

from thermo_lab.exact import IsingModel
from thermo_lab.records import ExperimentSpec
from thermo_lab.schemas import JAX_KEY_POLICY


def ising_chain_spec(seed: int = 7, n_samples: int = 2_500) -> ExperimentSpec:
    model = IsingModel(
        biases=(0.20, -0.15, 0.05, 0.10, -0.05),
        edges=((0, 1), (1, 2), (2, 3), (3, 4)),
        weights=(0.35, -0.25, 0.30, 0.20),
        beta=0.8,
    )
    return ExperimentSpec(
        experiment_id="thrml.ising_chain_exact_validation.v1",
        seed=seed,
        model_config=model.as_config(),
        run_config={
            "block_partition": [[0, 2, 4], [1, 3]],
            "n_warmup": 200,
            "n_samples": n_samples,
            "steps_per_sample": 2,
            "max_marginal_error_tolerance": 0.10,
            "total_variation_tolerance": 0.15,
            "key_policy": JAX_KEY_POLICY,
        },
        sample_definition=(
            "One recorded full five-spin state after two complete ordered block-Gibbs "
            "sweeps; recorded-state count is not an effective-independent-sample count."
        ),
    )
