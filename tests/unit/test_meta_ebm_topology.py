"""M5c lattice rule, degree repair, placement verification and packing checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_topology as m5c

M5B_COMPLETION = (
    Path(__file__).resolve().parents[2]
    / "docs/experiment-reports/2026-09-28-meta-ebm-thermalization/completion.json"
)


@pytest.fixture(scope="module")
def primary():
    source = m5c.load_source()
    refs = m5c.references(source, [0])
    structures = m5a.structures(m5a.make_target(0, "B"))
    vectors = [np.asarray(p, dtype=np.float64) for p in refs[0]["parameters"]]
    return refs, structures, vectors


@pytest.fixture(scope="module")
def small_placement(primary):
    _, structures, vectors = primary
    p, s = vectors[7], structures[7]
    layout, solver = m5c.place_kernel(p, s)
    return p, s, layout, solver


def test_published_offsets_are_sixteen_bipartite_and_rotation_closed():
    assert len(m5c.NEIGHBORS) == 16 == m5c.DEGREE_CAP
    assert all((dx + dy) % 2 == 1 for dx, dy in m5c.OFFSETS)
    assert all(m5c.rotate(o, 1) in m5c.NEIGHBORS for o in m5c.OFFSETS)
    assert all((-o[0], -o[1]) in m5c.NEIGHBORS for o in m5c.OFFSETS)


def test_request_pins_the_m5b_source_and_implementations():
    completion = json.loads(M5B_COMPLETION.read_text())
    assert m5c.SOURCE["archive_sha256"] == completion["archive_sha256"]
    assert m5c.SOURCE["request_digest"] == completion["request_digest"]
    assert m5c.SOURCE["result_digest"] == completion["result_digest"]
    request = m5c.study_request()
    assert set(request["implementation_sha256"]) == {*m5c.PINNED, "meta_ebm_topology.py"}
    assert request["seeds"] == [0, 1, 2, 3, 4]
    assert request["refit"]["options"]["maxiter"] == 20_000
    assert request["placement"]["time_limit_seconds"] == 600.0
    assert m5c.study_request("benchmark")["seeds"] == [0]
    with pytest.raises(ValueError):
        m5c.study_request("other")


def test_j_mask_repairs_degree_without_touching_beta(primary):
    _, structures, vectors = primary
    masked = [
        site
        for site, (p, s) in enumerate(zip(vectors, structures, strict=True))
        if m5c.j_mask(p, s)
    ]
    assert masked == [1, 10]
    for site in masked:
        p, s = vectors[site], structures[site]
        mask = m5c.j_mask(p, s)
        J = m5a.unpack(p, s)[0]
        assert len(mask) == m5c.output_degree(p, s) - 16
        kept = [abs(J[i]) for i in np.flatnonzero(J) if i not in mask]
        assert max(abs(J[i]) for i in mask) <= min(kept)
        q = p.copy()
        q[mask] = 0.0
        assert m5c.output_degree(q, s) == 16
        np.testing.assert_array_equal(m5a.unpack(q, s)[4], m5a.unpack(p, s)[4])


def test_refit_checks_reject_mask_and_cap_violations(primary):
    _, structures, vectors = primary
    p, s = vectors[1], structures[1]
    q = p.copy()
    q[m5c.j_mask(p, s)] = 0.0
    assert m5c.refit_checks(p, q, s)["output_degree"] == 16
    broken = q.copy()
    broken[m5c.j_mask(p, s)[0]] = 0.1
    with pytest.raises(ValueError, match="mask or cap"):
        m5c.refit_checks(p, broken, s)
    broken = q.copy()
    broken[-1] = 1.5
    with pytest.raises(ValueError, match="mask or cap"):
        m5c.refit_checks(p, broken, s)


def test_placement_is_exact_and_meets_copy_bounds(small_placement):
    p, s, layout, solver = small_placement
    assert solver["optimal"]
    counts = m5c.verify_layout(layout, p, s)
    assert counts["conditional_error"] < 1e-12
    assert counts["copies"] >= counts["copy_parity_bound"]
    assert counts["j_copies"] == np.count_nonzero(m5a.unpack(p, s)[0])
    assert counts["physical_pbits"] == counts["free_pbits"] + counts["copies"]
    assert m5c.place_kernel(p, s)[0] == layout


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda layout: layout[:-1], "missing adjacent input copy"),
        (lambda layout: [*layout, list(layout[-1])], "share a site"),
        (lambda layout: [[e[0] + 1, *e[1:]] if e[2] == "h" else e for e in layout], "lattice"),
    ],
)
def test_verify_layout_rejects_broken_layouts(small_placement, tamper, message):
    p, s, layout, _ = small_placement
    with pytest.raises(ValueError, match=message):
        m5c.verify_layout(tamper([list(e) for e in layout]), p, s)


def test_verify_layout_rejects_misassigned_input(small_placement):
    p, s, layout, _ = small_placement
    swapped = [list(e) for e in layout]
    copies = [e for e in swapped if e[2] == "copy"]
    other = next(e for e in copies if e[3] != copies[0][3])
    copies[0][3], other[3] = other[3], copies[0][3]
    with pytest.raises(ValueError):
        m5c.verify_layout(swapped, p, s)


def test_packing_is_disjoint_inside_and_deterministic(small_placement):
    p, s, layout, _ = small_placement
    record = m5c.patch_record([layout] * 3, [p] * 3, [s] * 3)
    assert record["sites"] == 3 * len(layout)
    assert record["conditional_error"] < 1e-12
    assert record == m5c.patch_record([layout] * 3, [p] * 3, [s] * 3)
    placed = [m5c.transform(layout, spot) for spot in record["placements"]]
    sites = [(e[0], e[1]) for kernel in placed for e in kernel]
    assert len(set(sites)) == len(sites)
    assert all(0 <= x < record["side"] and 0 <= y < record["side"] for x, y in sites)
    with pytest.raises(ValueError, match="outside"):
        m5c.verify_layout(placed[0], p, s, side=1)
