# Z1 acceptance kit: protocol (draft, proposal P-0015)

Drafted on 2026-11-15 for queue row `z1-acceptance-kit`. Not frozen until
the owner approves it; no runner exists yet. Probe:
[`docs/research/2026-11-15-z1-acceptance-kit-probe.md`](../research/2026-11-15-z1-acceptance-kit-probe.md)
(exploration, one sampler seed per size).

## The number this study produces

For each admitted defect on the nine P-0001 `greedy-long` graphs (64, 256
and 1,024 spins, zero field, beta = 4), the study reports **the reads, and
the writes, an acceptance test needs to detect that defect at 5 percent false
alarm and 5 percent miss, with reads one sweep apart.** It also reports the
smallest defect of each kind detectable within 10,000 reads. Both are read
budgets that a test plan for a sampled array is designed against. Each count
is validated by running the test itself on the ideal and the defective
sampler. Means and shifts are `exact_reference`. The read correlation and
the operating characteristic are `software_simulation`. It is a metrology
row: there is no chip, no claim about any chip and no digital baseline to
price.

## Question

Which exact-answer tests on Z1-native graphs can accept or reject a sampled
array, and how many reads does each need to detect a stated defect at stated
error rates?

- **User and decision.** A person qualifying a sampler, such as a chip
  bring-up engineer or a later loop row, needs to decide how many reads to
  spend before accepting an array as sampling the programmed law.
- **Mechanism.** A defect that keeps the model a zero-field planar Ising law
  shifts exact Kac-Ward moments. The most powerful test of a simple pair of
  hypotheses is the log-likelihood ratio (LLR), and its read count follows
  from exact means and variances.
- **Falsifier.** The sampled operating characteristic misses 5 percent false
  alarm or 5 percent miss at the predicted count, by more than the declared
  tolerance.

## Hardware constraints (rule 8)

Applied:

- the two-colour schedule (colour A, then colour B, every sweep);
- 16-offset placement on a planar subgraph, which uses a subset of the
  lattice's edges and is therefore placeable;
- coupling quantization as a defect under the constraint layer's codebook;
- zero field;
- no replica exchange.

Relaxed or assumed, and why:

1. The ideal model has float couplings. It is the reference the defects are
   measured against.
2. Static beta jitter is not applied. The probe shows its stationary law is
   not an Ising law, so it has no exact reference here (see the arms).
3. Starts are full-state writes of a Néel ground state. The archived random
   start leaves the ideal sampler unattributable at 256 and 1,024 spins
   (probe finding 2).
4. Planarity restricts the chip's degree-16 rule. It is required because the
   exact reference exists only on planar graphs.

## Resource accounting (rule 9)

Every test reports:

- elapsed sweeps, including burn-in;
- full-array reads and node reads (N per full read, 1 per single-site read);
- writes: one coupling-programming write per model and one state
  initialization write per chain.

Projected joules are not reported. This is not a task row, so the standing
task-row pricing rules do not apply, and no conventional baseline is priced.
The exact reference itself is the "digital" side: one Kac-Ward evaluation
takes about 14 CPU-seconds at 1,024 spins.

## Model, representation, update and estimator

- **Intended model.** Zero-field Ising on each target's graph, with the
  archived couplings and beta = 4.
- **Representation.** The same couplings after the defect. Rounding uses
  `thermo_lab.constraints.quantize.quantize_array` with the `full` cap (1.0),
  delegating to M5b `round_parameters` at 4, 6 and 8 bits. A missing
  coupling sets one J to 0. A stuck p-bit pins one site.
- **Actual update.** Parallel heat bath on colour A, then on colour B, with
  P(s_i = +1) = sigmoid(2 beta h_i), the archived sampler's convention.
- **Estimator.** The test statistic averaged over n reads, compared with a
  threshold fixed from the ideal model's exact moments.

## Fixed choices and what they bound

- **Graphs: the nine `greedy-long` targets of P-0001** (seeds 4510 to 4512;
  L = 8, 16, 32), rebuilt from seed. The paired grids are not targets.
  *Bounds:* missing-edge detectability, through local degree.
- **beta = 4**, the only archived temperature. Ordinary Gibbs qualified
  there at 64 spins in the archive. *Bounds:* detection power. No 8-bit
  defect is detectable within 10,000 reads at any size (probe finding 3).
  Not varied.
- **Start: flip-symmetric Néel.** Even-numbered chains start in one ground
  state and odd-numbered chains in the other. There are 256 burn-in sweeps
  before the first read. *Bounds:* mixing error (probe bias ≤ 1e-6 per spin,
  autocorrelation ≤ 1.29 sweeps). Flip symmetry keeps the stuck-spin test
  exact.
- **Read spacing: one sweep.** Read counts are multiplied by the measured
  integrated autocorrelation of each statistic on the ideal model.
- **Error rates: 5 percent and 5 percent, one-sided** (the defect hypothesis
  is simple). Read limit 10,000.
- **Statistics:**
  - energy per spin with the nominal couplings;
  - each edge product, both at a known location and scanned over all edges
    with Bonferroni-corrected false alarm;
  - the defect's LLR, which for a missing coupling is its own edge product;
  - for a stuck p-bit, one site's value across independent chains, at a
    known site and scanned over all sites.
- **Codebook.** The `full` cap 1 on weights {1..5}/5. *Bounds:* bit
  monotonicity. 5 and 9 bits are exact on this alphabet, so they are
  secondary rows with zero defect.

## Pre-registered expectations (from the probe)

1. All 18 P-0001 references reproduce bitwise.
2. LLR read counts at independent reads match the probe's table to the
   printed digits. 4-bit: 34 to 46 reads at 1,024 spins. 6-bit: 1,889 to
   3,021 reads at 1,024 spins. 8-bit: above 10,000 at every size.
3. The smallest rounding defect detectable within 10,000 reads is 4 bits at
   64 spins and 6 bits at 1,024 spins. At 256 spins it is 6 bits on two of
   three targets.
4. Of the edges at 64 spins, 40 to 50 percent can go missing detectably
   within 10,000 reads at a known location. The share is no higher at 1,024
   spins.
5. Stuck p-bit: 5 or 6 reads at a known site, and 12, 14 and 16 reads
   scanning all sites, with zero miss.
6. Every sampled test passes its operating-characteristic check, and the
   ideal sampler's energy bias at 10,000 reads is within 3 standard errors
   of zero.

## Exact anchor and evidence classes

The anchor is the Kac-Ward determinant on the explicit embedding, imported
unchanged from `planar_16_offset_ferro`. A zero coupling is handled by
dropping the edge (the graph stays planar). Its own correlation is
d ln Z / dK_e at 0, by central difference.

For all edges of a target, the study computes these from one inverse per
target with a rank-2 determinant update. On 20 declared edges per target the
update is checked against direct evaluation, to 1e-9.

Per-read variances are second derivatives of ln Z along the defect
direction. The finite-difference check against brute force on 4 x 4 patches
must pass to a relative 1e-6.

Before any sampling, two checks run on an enumerated 12-spin patch:

- the stuck-spin conditional law and odd identity, to 1e-10;
- the beta-variation non-Ising residual. Its fit residual must exceed 1e-4
  at ±10 percent, or beta variation is reconsidered and the run stops for
  the owner.

Means, shifts, per-read variances and independent-read counts are
`exact_reference`. Counts under the normal approximation are labelled as
such. Autocorrelations, sampled means and operating characteristics are
`software_simulation` (NumPy float64, CPU). Nothing runs on hardware.

## Arms and controls

- **Control:** the ideal model, with sampled false alarm measured at every
  test's read count and threshold.
- **Rounding:** 4, 6 and 8 bits. Exact secondary rows at 5, 7, 9, 10 and 12
  bits, not sampled.
- **Missing coupling:** every edge of every target, exact. Sampled on three
  declared edges per target, chosen among edges detectable within 10,000
  reads at a known location: the cheapest, the median and the dearest. If
  fewer than three exist, all of them are sampled.
- **Stuck p-bit:** every site, exact. Sampled with site 0 and the centre
  site (L/2, L/2) pinned at +1, on the scan-over-all-sites test.
- **Excluded:** static per-site beta variation of 10 and 30 percent. There
  is no exact reference at 64 spins or more (probe finding 7). It is
  reported as excluded, with the patch evidence, and no simulated substitute
  is labelled exact.

## Primary decision rule (fixed in advance)

The predicted count n* for a (target, defect, statistic) triple is the
independent-read count multiplied by the ideal model's measured integrated
autocorrelation of that statistic, rounded up. Each triple with n* ≤ 10,000
is run as 400 replicate tests on fresh chains, with the threshold set from
exact ideal moments. A triple **passes** if the ideal sampler rejects in at
most 31 of 400 replicates and the defective sampler rejects in at least 369
of 400. That is about 2.3 binomial standard deviations each way; the
one-sided risk of a wrong failure is about 1 percent per check.

Before any test runs, an **attribution check** is applied per target: the
ideal sampler's energy-per-spin mean over 400 chains × 10,000 reads must lie
within 3 standard errors of the exact value. A target that fails is
unattributable. Its tests are reported but do not count.

Device error and mixing error are reported apart for every target:

- device error is the exact shift;
- mixing error is the ideal sampled mean minus the exact mean, with its
  standard error.

Row verdict, one of three sentences:

- **The kit holds** if every target passes the attribution check and every
  triple with n* ≤ 10,000 passes.
- **The kit holds except for named triples** if every target passes the
  attribution check and some triples fail. Each failure is named with its
  false-alarm and detection counts.
- **The kit is not attributable at size L** if any target of size L fails
  the attribution check.

The headline per defect is n* for the best passing statistic, with its reads
and writes. Defects with n* above 10,000 are reported as "not detectable
within 10,000 reads" from the exact count and are not sampled.

## What one recorded sample means

One read is the full array after one complete two-colour sweep, with
nothing averaged inside the read. One replicate test uses n consecutive
reads from one chain, after 256 burn-in sweeps. The stuck-spin test is the
exception: it uses one read from each of n independent chains. Replicate
tests, not reads, are the replication unit for operating characteristics.

## Seeds

Graph and coupling seeds 4510, 4511 and 4512 are fixed by the row, as
reproduced from P-0001. They are target identities, not sampler seeds. The
sampler root is 20261116 (fresh; the probe used 20261115 plus the size),
folded with target, model and replicate.

## Budgets and CPU estimate

The estimate is scaled from probe timings (about 45 µs per chain-sweep at
1,024 spins; Kac-Ward 3.5 s for ln Z and 14 s with correlations).

Exact part, about 25 CPU-minutes:

- references, 3 min;
- rounding with variances, about 15 min;
- every missing edge by rank-2 update, plus direct checks, about 8 min.

Sampled part, at most about 90 CPU-minutes:

- per 1,024-spin target, about 20 min: ideal chains for 10,256 sweeps × 400,
  defect chains for at most 10,256 sweeps × 400 each, and stuck tests;
- 256-spin targets about a quarter of that, 64-spin targets about 1/16.

Replay: about 10 min (exact recomputation and statistics; no sweeps).

**Total about 2 CPU-hours**, about 45 min wall with three workers. Under the
8 CPU-hour flag. Peak memory about 0.3 GiB per worker for the 1,024-spin
inverse.

## Autosave and resume

The run is over 30 minutes, so it follows the autosave contract in
`docs/experiment-runner.md`. The work unit is one (target, model) group:
its exact rows, or its sampled replicate statistics and rejection counts.
Each unit is written atomically under `units/` with a digest in a unit
index. `--resume` re-authenticates the request and sources, verifies every
completed unit and runs only the missing ones.

Interruption test: kill after at least one sampled unit, resume, and
require bitwise equality with an uninterrupted run of the same small
request.

Resume command:
`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit --resume`.

## Persistence and archive size

The archive holds:

- the request, graph edge lists and references;
- exact tables for every defect and statistic, with every missing edge's
  c0, c1 and its known and scanned counts;
- per sampled unit: the 400 replicate statistic values (float32) and
  rejection counts, the ideal prefix means at each declared n, and
  autocorrelation estimates;
- summaries and decisions.

No sweep traces. Estimated at under 1 MB gzipped. Above 4 MB, the run
completes but the executor does not commit the archive and asks the owner.

## Gate

`uv run python -m thermo_lab.z1_acceptance_kit --output-dir results/z1-acceptance-kit`.
`completion.json` is written last, after a full replay, and must show:

- `status=z1_acceptance_kit_complete`;
- `targets=9` and `references_reproduced=18`;
- `rounding_rows=27`;
- `missing_edges_exact` equal to the summed edge count of the nine targets;
- `stuck_sites_exact` equal to the summed site count (4,032);
- `sampled_triples` equal to the declared count;
- `attribution_checks_passed`, `reference_checks_passed`,
  `patch_checks_passed` and `replayed`, all `true`.

CPU only, single-thread BLAS. CI replays the archive only.

## Stop rules

- A reference, rank-2, finite-difference or patch check fails: stop before
  sampling and report. No tolerance is loosened.
- A unit fails to verify on resume: rerun only that unit. If it fails
  twice, stop and report.
- Wall time above 2x the estimate: checkpoint and report.
- Archive above 4 MB: do not commit; ask the owner.
- No defect, edge, statistic, threshold or seed is added or removed after
  seeing results.

## Advance, stop or unresolved

- **Advance:** "the kit holds". The read budgets become a reference for
  scoring a sampled array on these graphs at beta = 4. A later row could
  ask the same at a temperature with more information per read; this row
  does not.
- **Stop:** the kit is not attributable at a size. That size's budgets
  are withdrawn, and the start design goes back to the owner.
- **Unresolved:** some triples fail. The normal approximation or the
  autocorrelation correction is wrong for them, and their read counts need
  empirical calibration before use.

## What a negative result means

A failed triple means the exact-moment read count does not deliver the
stated error rates for that statistic on this sampler. That is a statement
about the test design at beta = 4 on these planar graphs, not about any
defect's physics or any chip. An unattributable size means the ideal
two-colour sampler, even from a ground-state start, does not hold the exact
law to the needed precision there. No test at that size can then be scored.
None of this covers frustrated targets, the non-planar degree-16 lattice,
static beta variation, or hardware.

## Evidence obligations

From the dated index `loop/evidence-obligations.json` (checked
2026-11-11), reconciled against the current main snapshot:

- EO-001, EO-002, EO-004 (rounding-rule literature and transfer): do not
  apply. This row uses nearest rounding as a defect, not as a lever.
- EO-003 (adsorption unpowered read allocation lesson): adjacent, still
  open, not closed here. This protocol powers its own read counts, but the
  owed lesson concerns the adsorption probe and needs a documentation
  author. Closure evidence: missing.
- EO-005 (amendment 1 evaluation): open and unaffected.

The row's source card `extropic-thermalizers` exists. It covers the
Thermalizers paper, not acceptance testing, and supports no claim here
beyond the kernel convention.
