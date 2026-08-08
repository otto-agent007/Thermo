import importlib.metadata

import jax
import pytest

from thermo_lab.backends import ThrmlLocalBackend, TorxStateVectorBackend
from thermo_lab.evidence import EvidenceClass
from thermo_lab.experiments import ising_chain_spec, torx_smoke_spec


def test_pinned_packages_import_in_one_process() -> None:
    import thrml
    import torx

    assert thrml.__version__ == "0.1.4"
    assert torx.__version__ == "0.0.1"
    assert importlib.metadata.version("extro-torx") == "0.0.1"
    assert jax.default_backend() == "cpu"


def test_torx_statevector_smoke_matches_analytic_distribution() -> None:
    record = TorxStateVectorBackend().run(torx_smoke_spec())

    assert record.evidence_class is EvidenceClass.EXACT_REFERENCE
    assert record.metrics["probability_sum"].value == pytest.approx(1.0)
    assert record.metrics["max_abs_error_vs_analytic"].value <= 1e-6


def test_thrml_chain_smoke_meets_predeclared_statistical_tolerances() -> None:
    backend = ThrmlLocalBackend()
    execution = backend.execute(ising_chain_spec(seed=8, n_samples=1_500))
    record = execution.record

    assert record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert record.metrics["max_marginal_error"].value <= 0.10
    assert record.metrics["empirical_total_variation"].value <= 0.15
    assert record.metrics["minimum_spin_ess"].value <= 1_500
    assert record.metrics["median_spin_ess"].value <= 1_500
    assert record.metrics["complete_gibbs_sweeps_per_recorded_state"].value == 2
    assert execution.diagnostic_series["spin_states"].shape == (1_500, 5)
    assert "spin_states" not in record.model_dump_json()
