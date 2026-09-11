import pytest


@pytest.fixture(scope="module")
def audit():
    from thermo_lab.finite_sampler_audit import build_sampler_audit

    return build_sampler_audit()


def test_sampler_audit_round_trips_all_fixed_diagnostics(audit):
    from thermo_lab.finite_sampler_audit import FiniteSamplerAudit

    assert len(audit.cells) == 18
    assert audit.request.batch_size == 32768
    assert FiniteSamplerAudit.model_validate_json(audit.model_dump_json()) == audit
    assert all(cell.reference_evidence_class == "exact_reference" for cell in audit.cells)
    assert all(cell.sample_evidence_class == "software_simulation" for cell in audit.cells)


@pytest.mark.parametrize("mutation", ("mean", "second", "sample", "seed", "order", "request"))
def test_rehashed_sampler_evidence_reconstructs(audit, mutation):
    from thermo_lab.finite_sampler_audit import FiniteSamplerAudit, sampler_result_digest
    from thermo_lab.hashing import canonical_sha256

    payload = audit.model_dump(mode="json")
    if mutation == "mean":
        payload["cells"][0]["reference_mean"][0] += 0.01
    elif mutation == "second":
        payload["cells"][0]["reference_second_moment"][0] += 0.01
    elif mutation == "sample":
        payload["cells"][0]["sample"]["component_sum"][0][0] += 0.01
    elif mutation == "seed":
        payload["cells"][0]["sampling_seed"] += 1
    elif mutation == "order":
        payload["cells"].reverse()
    else:
        payload["request"]["batch_size"] = 1024
        payload["request_hash"] = canonical_sha256(payload["request"])
    payload["result_digest"] = sampler_result_digest(payload["request_hash"], payload["cells"])
    with pytest.raises(ValueError):
        FiniteSamplerAudit.model_validate(payload)
