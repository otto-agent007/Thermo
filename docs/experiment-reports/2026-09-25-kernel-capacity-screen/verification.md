# Verification

- Protocol frozen at commit `89b7c2b` and runner committed at `8beac2d`,
  both before any fit; `kernel_capacity_screen.py` and `raised_cap_screen.py`
  are unchanged since, and the request binds their SHA-256.
- `uv run python -m thermo_lab.kernel_capacity_screen --output-dir results/kernel-capacity-screen`
  on 2026-09-25 with 8 worker processes: generation 805 s, complete persisted
  replay 796 s, replay identical in canonical JSON. `completion.json` records
  `replayed: true`.
- Preflight passed before fitting: nesting against the M3 law within
  1.9e-14; warm-start objectives within relative 6.5e-10 of M4H; stationarity
  and row sums within 3.3e-16; 36 gradient checks, worst scaled error 1.3e-8;
  M4H cap-2 and cap-4 base metrics replayed.
- `uv run pytest tests/unit/test_kernel_capacity_screen.py tests/unit/test_raised_cap_screen.py`
  passes, including the archived-evidence replay without refitting.

## Reading note

In the summary's "Groups above base objective" column, `29` is a group
index, not a count. Only group 29 (second hidden spin, cap 2) ended above its
M4H base objective, by 5.9e-14 (relative 4.2e-13). Its warm start was
admissible; the gap is floating-point difference between the two evaluators,
within the protocol's 1e-9 warm-start tolerance, not a worse fit.
