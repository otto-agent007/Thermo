# Verification

- Protocol frozen at commit `07a642a` before any fit; the runner
  `src/thermo_lab/raised_cap_screen.py` is unchanged since that commit, and
  the request binds its SHA-256.
- `uv run python -m thermo_lab.raised_cap_screen --output-dir results/raised-cap-path-kl-screen`
  on 2026-09-25: generation 687 s, complete persisted replay 668 s, replay
  identical in canonical JSON. `completion.json` records `replayed: true`.
- Preflight passed before fitting: M4C K4 survival replay within 1.4e-17 of
  the archive; pooled path KL equal to the enumerated return-fixture
  trajectory KL at both points; 27 gradient checks, worst scaled error
  2.35e-8; all 512 cap-6 corners finite and normalized.
- `uv run pytest tests/unit/test_raised_cap_screen.py`: 8 passed, including a
  brute-force path-KL enumeration and a bitwise match of the study's capped
  law to `finite_sweep_joint_law` inside [-2, 2].
- `uv run ruff format --check .`, `uv run ruff check .` and
  `uv lock --check --offline` pass.

## Post hoc diagnostic (not part of the frozen report)

Computed afterwards from the persisted fitted parameters, so it is
descriptive only. The same K4-fitted kernels evaluated at K16 and at
equilibrium keep median occupied-edge leakage near 6% (cap 4: 6.3%; cap 6:
6.0%) while hop MAE rises from about 0.020 at K4 to 0.045–0.047. The higher
survival at longer horizons in the summary therefore comes with worse hop
fidelity; no tested horizon meets the full contract with these parameters.
