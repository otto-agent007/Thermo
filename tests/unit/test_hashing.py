import math

import pytest

from thermo_lab.hashing import canonical_json, canonical_sha256


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    left = {"beta": 1.0, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "beta": 1.0}

    assert canonical_json(left) == canonical_json(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_sha256(left).startswith("sha256:")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        canonical_json({"value": value})
