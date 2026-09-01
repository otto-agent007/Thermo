from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import (
    COLOR_ORDER,
    COORDINATE_PAIR_CLASSES,
    WORD_ORDER,
    build_paper_fixture,
    build_pasym_swap_conditional,
    hop_probability,
    paper_logit,
)


def test_pasym_swap_table_is_input_major_and_oriented() -> None:
    observed = np.asarray(build_pasym_swap_conditional(p_ij=0.03, p_ji=0.07))
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.93, 0.07, 0.0],
            [0.0, 0.03, 0.97, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("p_ij, p_ji", [(0.0, 0.07), (0.03, 1.0)])
def test_pasym_swap_table_rejects_non_strict_hop_probabilities(p_ij: float, p_ji: float) -> None:
    with pytest.raises(ValueError, match="strictly between zero and one"):
        build_pasym_swap_conditional(p_ij=p_ij, p_ji=p_ji)


def test_paper_fixture_has_complete_colored_torus_schedule() -> None:
    fixture = build_paper_fixture()

    assert fixture.side == 5
    assert fixture.color_order == ("H1", "H2", "H3", "V1", "V2", "V3")
    assert fixture.color_order == COLOR_ORDER
    assert sum(len(edges) for edges in fixture.color_classes.values()) == 50
    assert len(fixture.occurrences) == 500
    assert {item.macrostep for item in fixture.occurrences} == set(range(10))
    assert {item.layer for item in fixture.occurrences} == set(range(60))
    assert all(
        len({vertex for edge in edges for vertex in edge}) == 2 * len(edges)
        for edges in fixture.color_classes.values()
    )
    assert COORDINATE_PAIR_CLASSES["H1"] == ((0, 1), (2, 3))
    assert COORDINATE_PAIR_CLASSES["H3"] == ((4, 0),)


def test_paper_fixture_probabilities_obey_rate_identity() -> None:
    fixture = build_paper_fixture()

    for target in fixture.targets:
        assert target.p_ij + target.p_ji == pytest.approx(0.1, abs=1e-15)
        assert target.target_hash == canonical_sha256(
            {"word_order": WORD_ORDER, "conditional": target.conditional}
        )
        assert target.conditional[0] == (1.0, 0.0, 0.0, 0.0)
        assert target.conditional[3] == (0.0, 0.0, 0.0, 1.0)


def test_paper_target_conditionals_are_exactly_input_stochastic() -> None:
    fixture = build_paper_fixture()

    for target in fixture.targets:
        assert all(sum(row) == 1.0 for row in target.conditional)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (
            (0, 0),
            (1, 0),
            (
                1.2953502868090965,
                -0.9438077588037361,
                0.009628878136877513,
                0.0903711218631225,
            ),
        ),
        (
            (4, 0),
            (0, 0),
            (
                -2.508875778371518,
                1.2953502868090965,
                0.09782089948214531,
                0.002179100517854692,
            ),
        ),
        (
            (2, 4),
            (2, 0),
            (
                -2.508875778371518,
                0.7499999999999996,
                0.09629907447759076,
                0.0037009255224092464,
            ),
        ),
    ],
)
def test_paper_logit_and_directed_probabilities_match_numeric_fixture(
    source: tuple[int, int], target: tuple[int, int], expected: tuple[float, float, float, float]
) -> None:
    source_logit, target_logit, forward_probability, reverse_probability = expected

    assert paper_logit(*source) == pytest.approx(source_logit, abs=1e-15)
    assert paper_logit(*target) == pytest.approx(target_logit, abs=1e-15)
    assert hop_probability(source, target) == pytest.approx(forward_probability, abs=1e-15)
    assert hop_probability(target, source) == pytest.approx(reverse_probability, abs=1e-15)


def test_target_identities_and_sorted_targets_ignore_edge_enumeration_order() -> None:
    forward = build_paper_fixture()
    reversed_edges = build_paper_fixture(reverse_edge_enumeration=True)

    assert forward.targets == reversed_edges.targets
    assert tuple(target.target_hash for target in forward.targets) == tuple(
        target.target_hash for target in reversed_edges.targets
    )
    assert tuple(target.target_hash for target in forward.targets) == tuple(
        sorted(target.target_hash for target in forward.targets)
    )
    assert tuple(item.target_hash for item in forward.occurrences) == tuple(
        item.target_hash for item in reversed_edges.occurrences
    )
    assert {item.target_hash for item in forward.occurrences} == {
        target.target_hash for target in forward.targets
    }


def test_paper_fixture_records_and_collections_are_immutable() -> None:
    fixture = build_paper_fixture()

    with pytest.raises(FrozenInstanceError):
        fixture.side = 7  # type: ignore[misc]
    with pytest.raises(TypeError):
        fixture.color_classes["H1"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        COORDINATE_PAIR_CLASSES["H1"] = ()  # type: ignore[index]
