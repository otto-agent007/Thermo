# Verification record

Evidence generated from source commit `1f4522952e23d14843a068a7bd6b355965d0f173`
with Python 3.11, CPU and the pinned `uv.lock` dependencies. The request and
result digests are in `archive-metadata.json`; runtime provenance is separate.

## Passed

- `uv run pytest -q tests/unit/test_return_fixture_objectives.py` — 10 passed.
- `uv run pytest -q tests/unit/test_return_fixture_study.py` — 9 passed.
- An independent review found missing row diagnostics; tests for all visited rows and the expanded report were run RED→GREEN. The final `uv run pytest -q tests/unit/test_return_fixture_objectives.py tests/unit/test_return_fixture_study.py` run passed all 20 tests in 64.27 seconds.
- `uv lock --check --offline` — dependency lock verified.
- `uv run ruff format --check .` — repository formatting verified.
- `uv run ruff check .` — no lint errors.
- `uv build` — wheel and sdist built successfully.
- `uv run python -m thermo_lab.return_fixture_study --output-dir results/return-fixture-study` — produced six arms and a final completion record after complete persisted replay.
- Strict validation of `study.json` reloaded from disk passed. Decompressed reviewed archive bytes, size, archive SHA-256, request digest and result digest match `archive-metadata.json`; complete replay regenerated the committed report byte for byte.
- A second reviewed run at fresh `results/return-fixture-study-reviewed-verification` produced the same request and result digests, six arms and zero samples.
- `git diff --check` — no whitespace errors. Existing two-operation and M4G archives were not modified.

## Broader repository tests

`uv run pytest` collected 2,113 tests and began running the older integration
suite. It was stopped during `tests/integration/test_composed_pasym_swap_runner.py`
because that integration group was taking substantially longer than the new
fixture gate. `uv run pytest -q tests/unit` was also stopped after its first
long tests. Neither command is reported as passing. CI must finish the
repository-wide gates before this branch is treated as merge ready.

This verification establishes exact-reference consistency for the new
fixture, not a full repository release gate or scientific support beyond the
frozen three-operation circuit.
