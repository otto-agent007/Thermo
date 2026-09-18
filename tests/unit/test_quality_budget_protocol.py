"""Detect budget omissions, seed reuse and mutable scientific contracts."""

import numpy as np
import pytest
from pydantic import ValidationError

from thermo_lab.quality_budget_protocol import (
    QualityBudgetProtocol,
    cost_ledger,
    evaluation_manifest,
    fit_manifest,
    role_schedule,
)


def test_contract_roundtrips_and_cannot_change_threshold_or_randomness():
    request = QualityBudgetProtocol()
    assert QualityBudgetProtocol.model_validate_json(request.model_dump_json()) == request
    for key, value in [("survival_min", 0.90), ("batch_size", 12), ("namespace", 0x4D3442)]:
        with pytest.raises(ValueError):
            QualityBudgetProtocol(**{key: value})
    with pytest.raises(ValidationError):
        request.batch_size = 8


def test_roles_use_all_71_children_in_fixed_order_and_never_reuse_evaluation():
    all_roles = []
    for seed in (0, 1, 2):
        roles = role_schedule(seed)
        expected = [
            int(c.generate_state(1, dtype=np.uint64)[0])
            for c in np.random.SeedSequence([0x4D3447, seed]).spawn(71)
        ]
        flat = [
            s[k]
            for f in roles["fits"]
            for s in f["steps"]
            for k in ("occupancy_seed", "gradient_seed")
        ]
        assert flat == expected[:70]
        assert roles["evaluation_seed"] == expected[70]
        assert [f["horizon"] for f in roles["fits"]] == [1, 2, 4, 8, 16, 30, "equilibrium"]
        all_roles.extend(flat + [roles["evaluation_seed"]])
    assert len(set(all_roles)) == 213


@pytest.mark.parametrize("seed", [True, -1, 3, 0.0])
def test_invalid_seed_is_integrity_error(seed):
    with pytest.raises(ValueError):
        role_schedule(seed)


def test_manifests_do_not_cross_evaluate_finite_fits_or_omit_oracles():
    fits, cells = fit_manifest(), evaluation_manifest()
    assert len(fits) == 21 and len(cells) == 60
    assert len({c["cell_id"] for c in cells}) == 60
    finite = [c for c in cells if c["member"] == "finite"]
    assert len(finite) == 18
    fit_by_id = {f["fit_id"]: f for f in fits}
    for cell in finite:
        assert fit_by_id[cell["fit_id"]]["horizon"] == cell["horizon"]
    assert sum(c["horizon"] == "equilibrium" for c in cells) == 6
    assert all(c["fit_id"] is None for c in cells if c["member"] == "frozen")


def test_costs_count_references_entire_training_grid_and_oracle_work():
    ledger = cost_ledger()
    finite, eq = ledger["training"]["finite"], ledger["training"]["equilibrium"]
    assert finite["endpoint_draws"] == 4423680000
    assert eq["endpoint_draws"] == 737280000
    assert eq["modeled_sweeps"] is None
    assert ledger["training"]["endpoint_draws"] == 5160960000
    assert ledger["evaluation"]["endpoint_draws"] == 983040000
    assert ledger["evaluation"]["equilibrium"]["endpoint_draws"] == 98304000
    assert finite["modeled_sweeps"] == 44974080000
    assert finite["pbit_updates"] == 134922240000
    assert finite["reset_bit_draws"] == 13271040000
    # 18 fits x 5 updates x 2 genuinely propagated training batches.
    assert finite["complete_trajectories"] == 5898240
    assert finite["reference_endpoint_draws"] == 1474560000
    one = ledger["inference_by_horizon"][0]
    assert one["per_trajectory"]["endpoint_draws"] == 500
    assert one["per_trajectory"]["modeled_sweeps"] == 500
    assert one["per_trajectory"]["reset_bit_draws"] == 1500
    assert ledger["inference_by_horizon"][-1]["per_batch"]["modeled_sweeps"] == 491520000
