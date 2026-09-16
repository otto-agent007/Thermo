"""Independent conservation checks catch hidden-bit, re-entry and flux mistakes."""

import itertools

import numpy as np
import pytest


def _joint(visible):
    # Hidden bit is independent, with unequal probabilities to detect wrong marginalization.
    return np.concatenate((0.3 * visible, 0.7 * visible), axis=-1)


def _exhaustive(visible, groups, sites, dimension):
    states = tuple(itertools.product((0, 1), repeat=dimension))
    mass = {state: float(state == (1,) + (0,) * (dimension - 1)) for state in states}
    history = []
    for group, (left, right) in zip(groups, sites, strict=True):
        next_mass = dict.fromkeys(states, 0.0)
        for state, weight in mass.items():
            parent = 2 * state[left] + state[right]
            for outcome in range(4):
                child = list(state)
                child[left], child[right] = outcome // 2, outcome % 2
                if sum(child) == 1:
                    next_mass[tuple(child)] += weight * visible[group, parent, outcome]
        mass = next_mass
        history.append(sum(mass.values()))
    return history


def test_exact_first_exit_matches_exhaustive_killed_full_state_chain():
    from thermo_lab.conservation_diagnostic import exact_survival

    visible = np.array(
        [
            [
                [0.6, 0.1, 0.2, 0.1],
                [0.1, 0.5, 0.3, 0.1],
                [0.2, 0.2, 0.5, 0.1],
                [0.1, 0.2, 0.3, 0.4],
            ],
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.3, 0.2, 0.4, 0.1],
                [0.1, 0.6, 0.2, 0.1],
                [0.2, 0.1, 0.2, 0.5],
            ],
        ]
    )
    groups, sites = (0, 1, 0, 1), ((1, 2), (0, 2), (2, 1), (1, 0))
    rows = exact_survival(_joint(visible), groups, sites, site_count=3)
    np.testing.assert_allclose(
        [row["survival_probability"] for row in rows],
        _exhaustive(visible, groups, sites, 3),
        rtol=0,
        atol=2e-16,
    )
    assert rows[0]["first_exit_by_parent"] == pytest.approx([0.4, 0, 0, 0])
    assert sum(sum(row["first_exit_by_parent"]) for row in rows) + rows[-1][
        "survival_probability"
    ] == pytest.approx(1)


def test_exit_then_return_never_counts_as_uninterrupted_survival():
    from thermo_lab.conservation_diagnostic import exact_survival, sample_trace

    # First force 10 -> 00; then 00 -> 10. Terminal conservation hides a certain exit.
    visible = np.zeros((2, 4, 4))
    visible[0, :, 0] = 1
    visible[1, :, 2] = 1
    tables = _joint(visible)
    groups, sites = (0, 1), ((0, 1), (0, 1))
    exact = exact_survival(tables, groups, sites, site_count=3)
    sampled = sample_trace(tables, groups, sites, site_count=3, batch_size=8, seed=41)
    assert [r["survival_probability"] for r in exact] == [0, 0]
    assert exact[0]["first_exit_by_parent"] == [0, 0, 1, 0]
    assert exact[0]["first_exit_creation_probability"] == 0
    assert exact[0]["first_exit_destruction_probability"] == 1
    assert sampled[0]["first_exit_by_parent"] == [0, 0, 8, 0]
    assert sampled[0]["first_exit_creation_count"] == 0
    assert sampled[0]["first_exit_destruction_count"] == 8
    assert sampled[0]["particle_histogram"] == [8, 0, 0, 0]
    assert sampled[1]["particle_histogram"] == [0, 8, 0, 0]
    assert sampled[1]["ever_exited_count"] == 8
    assert sampled[1]["return_count"] == 8
    assert sampled[1]["valid_after_exit_count"] == 8
    assert sampled[1]["occupancy_counts"] == [8, 0, 0]


def test_identity_kernel_preserves_particle_and_flux_in_every_step():
    from thermo_lab.conservation_diagnostic import exact_survival, sample_trace

    tables = _joint(np.eye(4)[None])
    groups, sites = (0, 0), ((0, 1), (1, 2))
    exact = exact_survival(tables, groups, sites, site_count=3)
    sampled = sample_trace(tables, groups, sites, site_count=3, batch_size=8, seed=7)
    for row in exact:
        assert row["survival_probability"] == 1
        assert row["first_exit_by_parent"] == [0, 0, 0, 0]
    for row in sampled:
        assert row["ever_exited_count"] == row["return_count"] == 0
        assert row["particle_histogram"] == [0, 8, 0, 0]
        assert sum(map(sum, row["local_transition_counts"])) == 8


@pytest.mark.parametrize("horizon", [4, "equilibrium"])
def test_instrumentation_preserves_existing_joint_endpoint_draws(horizon):
    from thermo_lab.composed_trajectory_refinement import sample_equilibrium_terminal_occupancy
    from thermo_lab.conservation_diagnostic import endpoint_tables, sample_trace
    from thermo_lab.finite_sweep_sampling import sample_finite_sweep_terminal_occupancy

    parameters = ((0.1, -0.2, 0.3, 0.2, -0.1, 0.5, -0.3, 0.1, 0.2),)
    groups, sites = (0, 0, 0), ((0, 1), (1, 2), (2, 0))
    settings = dict(site_count=3, batch_size=256, seed=17)
    rows = sample_trace(endpoint_tables(parameters, horizon), groups, sites, **settings)
    if horizon == 4:
        expected = sample_finite_sweep_terminal_occupancy(
            parameters, groups, sites, beta=1.0, horizon=4, **settings
        )
    else:
        expected = sample_equilibrium_terminal_occupancy(
            parameters, groups, sites, beta=1.0, **settings
        )
    assert tuple(rows[-1]["occupancy_counts"]) == expected.occupancy_counts
    previous_particles = 256
    previous_leakage = 0
    counts = np.array([0, 1, 1, 2])
    for row in rows:
        transitions = np.array(row["local_transition_counts"])
        flux = int(np.sum(transitions * (counts[None, :] - counts[:, None])))
        particles = sum(i * n for i, n in enumerate(row["particle_histogram"]))
        assert particles - previous_particles == flux
        leakage = 256 - row["particle_histogram"][1]
        assert leakage - previous_leakage == row["exit_count"] - row["return_count"]
        previous_particles, previous_leakage = particles, leakage


@pytest.mark.parametrize("corruption", ["negative", "nan", "mass", "shape"])
def test_invalid_probability_tables_rejected(corruption):
    from thermo_lab.conservation_diagnostic import exact_survival

    tables = _joint(np.eye(4)[None])
    if corruption == "negative":
        tables[0, 0, 0] = -0.1
    elif corruption == "nan":
        tables[0, 0, 0] = np.nan
    elif corruption == "mass":
        tables[0, 0, 0] += 0.1
    else:
        tables = tables[:, :, :4]
    with pytest.raises(ValueError):
        exact_survival(tables, (0,), ((0, 1),), site_count=3)
