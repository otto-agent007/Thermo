# THRML on every compiled M5a kernel's inner sweep (E0 stage C): protocol (draft, proposal P-0003)

Drafted on 2026-10-09 for queue row `e0-remaining-kernels`. Not frozen until
the owner approves it; no runner exists yet. Probe:
[`docs/research/2026-10-09-e0-remaining-kernels-probe.md`](../research/2026-10-09-e0-remaining-kernels-probe.md)
(exploration, exact side for all 60 kernels plus THRML timing on five shapes).
Modelled on [E0 stage B](thrml-m5a-kernel-inner-sweep.md), whose contract it
replicates except where this document says otherwise and why.

**Cost flag: the production run is estimated at 7.5 to 8.5 CPU-hours, at or
over the loop's 8 CPU-hour line. The owner decides.** Two cheaper options are
listed under Budgets.

## Question

Does THRML 0.1.4 match the archived exact inner-K law for every compiled M5a
kernel, as stage B showed for one? Stage B ran the M5c primary arm (reading B,
variational, cap 1), seed 0, site 1. This study runs the other 59 kernels of
that arm (seeds 0 to 4, sites 0 to 11, less seed 0 site 1) at K in {1, 2, 4},
clamped to every blanket input, and reports per kernel the cells passing on
output rate and on the (hidden, output) joint, and the control rejections.

## Fixed choices and what they bound

- **Arm and kernels.** M5c primary arm, reading B (owner standing decision),
  the 59 archived parameter vectors read from the M5b archive through
  `meta_ebm_topology.load_source` and `references`, structures from
  `meta_ebm_cap_baseline.structures(make_target(seed, "B"))`. *Bounds:* the
  claim is about these kernels only, not other caps, methods, the refit or
  pruned M5c methods, or reading A.
- **K in {1, 2, 4}, y0 in {-1, +1}, hidden spins start at -1, hidden block
  then output block.** Stage B's values. *Bounds:* on fast kernels the K = 4
  law is within the tolerance of K = 5 and of the stationary limit (probe:
  16 kernels with lambda_max 0.09 to 0.46). Those cells still test the right
  law but cannot tell K from K + 1. K = 1 keeps every control at 5.8x its
  tolerance or more on every kernel, so the per-sweep conventions are tested
  decisively everywhere at K = 1.
- **N = 65,536 chains per input.** Stage B's value. *Bounds:* output
  tolerances of 0.006 to 0.010 and joint TV tolerances near 0.02 (stage B
  0.019 to 0.022; 0.024 on the largest kernel in the probe); smaller
  mismatches are not detectable. N also sets the runtime.
- **Tolerance rule.** Stage B's: per cell, the 0.999 quantile of the
  worst-over-inputs deviation an exact multinomial(N, law) sample shows, from
  4,000 (output) and 1,000 (joint) exact-side draws. *Bounds:* per-test
  false-failure rate 0.001; the family rule below handles 708 tests.
- **Control eligibility (changed from stage B).** Stage B required every
  control to separate before sampling and stopped otherwise. Applied to 59
  kernels, that rule stops the study before its first THRML call (probe: 72
  of 1,416 checks do not separate, all on fast kernels, some by
  construction). Here a control is **powered** in a cell when its exact
  separation exceeds 2x the cell's output tolerance; unpowered controls are
  reported with their separation and do not gate. *Bounds:* the number of
  controls that can be rejected (probe: 1,310 of 1,416 powered).
- **THRML model.** Stage B's `IsingEBM` encoding, beta = 1, float32,
  `SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)`, 16 inputs
  per batch (32-input kernels run as two batches). *Bounds:* float32 floor
  near 1e-7, far below tolerances.
- **Execution.** CPU only, one kernel per worker process, three workers,
  `JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and XLA
  intra-op threading limited to one thread per worker. *Bounds:* wall time
  only, not results.

## Pre-registered expectations (from the probe)

- All 354 cells pass on output rate and joint, up to about one false failure
  from multiplicity (expected 0.7 under the null).
- Every powered control is rejected (1,310 expected powered; the exact count
  is fixed by the exact side before sampling and recorded).
- Nearest K is K in every cell where off-by-one is powered (323 expected
  powered-or-marginal at the 1x level; 311 at the 2x level).
- If a kernel fails, the first suspects are the largest blankets (11 inputs,
  7 kernels) and the 512-state joints (seed 2 site 5, seed 3 site 2), which
  stage B did not exercise.

## Exact anchor and evidence classes

Exact references are `exact_reference`: float64 NumPy, the archived M5b law
through the hash-bound `meta_ebm_thermalization_core.powered_rates`, and stage
B's study-local joint enumeration (imported, not copied). Before sampling,
the joint must reproduce the archived output law to 1e-10 at every input of
every kernel (probe: 3.9e-15). THRML cells are `software_simulation`: THRML
0.1.4 on CPU in float32. Nothing here is hardware evidence; inner sweeps are
not device operations; timings are descriptive CPU wall time.

## Arms and controls

One arm: THRML block Gibbs as above, 59 kernels x 6 cells = 354 cells,
31,232 inputs per (y0, K) pair, 187,392 (input, cell) pairs.

Controls are stage B's four wrong exact references, per cell: `off_by_one`
(p0 T^(K+1); K - 1 reported as a diagnostic), `output_first` (block order),
`negated_inputs` (clamp sign) and `marginal_limit` (K -> infinity). 1,416
checks. Each is labelled powered or unpowered on the exact side before any
THRML call and the labels are part of the hashed exact record.

## Primary decision rule (thresholds fixed now)

Per cell, the output test passes if the worst-over-inputs |P_hat - p| is at
most the cell's output tolerance; the joint test passes if the
worst-over-inputs joint TV is at most the cell's joint tolerance. Both as in
stage B.

1. **Family verdict (primary).** THRML **matches the inner-K law on all 59
   kernels** if (a) at most 3 of the 708 tests fail (P(more than 3 false
   failures) about 0.006 under the null at 0.001 per test), (b) no failing
   test exceeds 1.5x its tolerance, and (c) every powered control is
   rejected. It **does not match** if any of (a) to (c) fails; the report
   names the kernels and cells. No other outcome.
2. **Per kernel (reported).** Output cells passed of 6, joint cells passed
   of 6, powered controls rejected of powered, cells nearest K of cells where
   off-by-one is powered. A kernel is listed as **clean** when all are full.
3. **Nearest K** is reported only where off-by-one is powered. Elsewhere the
   cell says "K and K+1 indistinguishable at N".

## What one recorded sample means

One independent chain's (hidden, output) state after exactly K inner sweeps
at one fixed blanket input of one kernel. N chains per input; chains are
independent, so N is the effective sample count.

## Seeds

JAX root 20261021, fresh (stage B used 20261004; probes used 9901). Per
kernel the key is `fold_in(root, 12 * seed + site)`, then the cell index,
then the batch start, then one key per chain split into an (unused) init key
and a sampling key, as stage B. Tolerance draws use NumPy seed 20261022 plus
a per-kernel and per-cell offset (`20261022 + 100 * (12 * seed + site) +
cell`, and `+ 50,000` for the joint), fresh. Unit tests run on probe seeds
only.

## Budgets and CPU estimate

From the probe, single-core rates (2.1x to 2.2x the multithreaded ones):

- THRML: about 22,500 CPU-seconds (6.3 CPU-hours) for 59 kernels x 6 cells.
- Joint tolerances: about 2,000 s; output tolerances and exact laws: 400 s.
- Full replay redraws the tolerances: about 2,400 s more.
- **Total about 7.5 to 8.5 CPU-hours; about 2.7 to 3 hours wall on three
  workers.**

The runner calibrates on the first three kernels (smallest, middle, largest)
and stops if it projects over 10 CPU-hours.

Owner options if the cost is too high (not the default):
- Drop the K = 4 cells on the 20 kernels where off-by-one is not powered at
  K = 4 (they test the right law but nothing about the sweep count): saves
  about 0.5 CPU-hours.
- Drop K = 4 everywhere: about 3.5 CPU-hours saved, K = 1 and 2 keep full
  convention power; this departs from the row's target family.

## Autosave and resume

The run exceeds 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).
The work unit is one kernel: its exact record (laws not stored, tolerances,
control powers and separations) and its six THRML cells, written atomically
under `units/seed-S-site-NN.json.gz`. A unit is reused only if the request
digest and runner source hash match. Resume:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m thermo_lab.thrml_m5a_all_kernels \
  --output-dir results/thrml-m5a-all-kernels --workers 3 --resume
```

A unit test interrupts a two-kernel run at reduced N and checks the resumed
record is identical to an uninterrupted one.

## Persistence and archive size

Dense joint histograms for 59 kernels are about 34 million bins (about 26 MB
gzipped), over the 4 MB budget. So:

- `study.json.gz` (committed): request and digest, per-kernel structure and
  parameter SHA-256, tolerances, control separations and power labels, per
  (input, cell) output counts (187,392 ints) and joint TV (187,392 floats,
  rounded to 1e-9), evaluation, result digest. Expected about 1 to 1.5 MB.
- `histograms.npz` (not committed, kept under `results/`): dense joint
  counts; its SHA-256 is in the archive and in the result digest.

Replay authenticates the archive by digest and the M5b archive by its
loader, recomputes all exact laws and control separations (numeric, 1e-12),
redraws output tolerances (within 2 counts / N as stage B) and, in a full
replay, joint tolerances, and recomputes the evaluation from the archived
counts and TVs. When the sidecar is present it also recomputes every joint
TV from the dense counts (within 1e-9). CI replays with `--light` (output
tolerances only, about the cost of the exact side); the full replay is a
local gate. A prefix check reruns the first batch of three kernels (smallest,
256-state, 512-state) at K = 1 and requires identical counts.

## Gate

`completion.json`, written last by the runner, must show:
`status=thrml_m5a_all_kernels_complete`, `kernels=59`, `cells=354`,
`inputs=31232`, `chains_per_input=65536`, `exact_agreement_passed=true`,
`powered_controls` (count), `prefix_check_passed=true`, `replayed=true`,
`autosave="per kernel; --resume"`, plus `output_tests_failed`,
`joint_tests_failed`, `powered_controls_rejected`, `clean_kernels` and
`family_verdict`. Verdicts and counts are scientific results, not integrity
checks.

## Stop rules

- The exact joint disagrees with the archived law above 1e-10 on any input,
  or any K = 1 control is unpowered: stop before sampling and report.
- Calibration projects over 10 CPU-hours: stop and report before the full
  run.
- Two consecutive failed runs with the same error: stop and report; no third
  variant.
- A replay mismatch is an integrity failure; no report is written as
  evidence.
- A failed cell is not a stop: the study completes and reports it.

## What a negative result means

A failing cell means THRML's execution of that kernel, or Thermo's reading
of it (encoding, block order, clamping), differs from the M5b law by more than
sampling noise at N = 65,536. The kernel's shape (blanket size, hidden count,
lambda) tells which convention to suspect; stage B's single site would then
not stand for the arm, and M5b/M5c finite-K statements for the failing sites
would lose their THRML backing. An unpowered control is not a negative
result: it says the kernel mixes faster than N can resolve. None of these is
an integrity failure.

## Not claimed

Hardware behaviour, device operations, latency or energy. Other M5 arms or
methods (refit, pruned), reading A, GPU float32 (A2), the placed M5c patch
(B4), or any K outside {1, 2, 4}.
