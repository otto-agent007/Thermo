import numpy as np
import pytest

from thermo_lab.exact import IsingModel, enumerate_ising


def test_one_spin_matches_analytic_mean() -> None:
    model = IsingModel(biases=(0.7,), edges=(), weights=(), beta=1.2)
    result = enumerate_ising(model)

    assert result.probabilities.sum() == pytest.approx(1.0)
    assert result.mean_spins[0] == pytest.approx(np.tanh(1.2 * 0.7))
    assert result.expected_energy == pytest.approx(-1.2 * 0.7 * np.tanh(1.2 * 0.7))


def test_two_ferromagnetic_spins_prefer_alignment() -> None:
    model = IsingModel(biases=(0.0, 0.0), edges=((0, 1),), weights=(1.0,), beta=1.0)
    result = enumerate_ising(model)
    aligned = result.probabilities[[0, 3]].sum()
    anti_aligned = result.probabilities[[1, 2]].sum()

    assert aligned > anti_aligned
    assert result.mean_spins.tolist() == pytest.approx([0.0, 0.0])


def test_exact_enumerator_has_a_hard_size_limit() -> None:
    model = IsingModel(biases=(0.0,) * 5, edges=(), weights=(), beta=1.0)
    with pytest.raises(ValueError, match="limited to 4 nodes"):
        enumerate_ising(model, max_nodes=4)
