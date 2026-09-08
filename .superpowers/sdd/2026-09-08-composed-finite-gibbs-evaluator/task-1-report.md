# Task 1 report: exact target checkpoints

## Implementation details

- Added `ExactTargetCheckpoint` and `derive_exact_target_checkpoints` in `src/thermo_lab/composed_pasym_swap_artifacts.py`.
- The implementation validates the checked 5x5, 500-occurrence, 37-target paper fixture and exact canonical target/order identity.
- It accepts only canonical checkpoint boundaries `(0, 50, ..., 500)`.
- It propagates a single particle through each checked target conditional using simultaneous endpoint values captured before each gate, validating finite nonnegative probabilities and unit mass after every occurrence.
- Added the two requested unit tests covering all checkpoints, mass conservation/nonnegativity, and rejection of noncanonical boundaries.

## TDD evidence

RED command:

```text
uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py -q
```

Result: collection failed as expected with `ModuleNotFoundError: No module named 'thermo_lab.composed_pasym_swap_artifacts'`.

GREEN command:

```text
uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py tests/unit/test_pasym_swap_context.py -q
```

Result: `22 passed in 0.43s`.

Additional verification:

```text
uv run ruff check src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py
uv run ruff format --check src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py
```

Result: all checks passed; both files already formatted.

## Files changed

- `src/thermo_lab/composed_pasym_swap_artifacts.py`
- `tests/unit/test_composed_pasym_swap_artifacts.py`
- This report.

## Self-review

- Confirmed the initial checkpoint is exactly the one-particle `(0, 0)` state.
- Confirmed endpoint updates use pre-operation endpoint masses and preserve the other 23 sites.
- Confirmed canonical fixture and checkpoint validation occurs before propagation.
- Confirmed focused tests and existing context tests pass, and Ruff reports no issues.

## Concerns

None. The fixture identity check intentionally rejects altered schedules or target conditionals, as required for exact checkpoints.

## Fix Round 1

Addressed the persistence-contract review finding by converting `ExactTargetCheckpoint` to a strict frozen Pydantic model with forbidden extras, strict integer/float validation, finite 25-site occupancy validation, explicit `exact_reference` digest binding, and `exact_reference` evidence classification. Added `validate_exact_target_checkpoints`, which regenerates the checked fixture/request and rejects persisted or rehashed tampering. Added focused tests for malformed construction, extra fields, digest validation, tampering, and successful deep reload/regeneration.

Verification:

```text
uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py tests/unit/test_pasym_swap_context.py -q
```

Result: `25 passed`.

```text
uv run ruff check src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py
uv run ruff format --check src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py
```

Result: all checks passed; both files formatted.
