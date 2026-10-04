# THRML's sweep is Thermo's sweep: 64 of 64 finite-K cells match p0 T^K

THRML 0.1.4's block-Gibbs sweep on the five-spin chain reproduces the exact
32-state law p0 T^K at every tested budget K in {0, 1, 2, 3, 4, 8, 16, 30},
for both block orders, three initial distributions and two clamped arms.
**All 64 cells pass** their predeclared tolerance (0.999 multinomial
quantile at N = 400,000 chains per cell). The negative-control gate passed on
the exact side before sampling, and the samples reject **41 of 48** wrong
references; the seven that are not rejected have exact separations of
0.0003 to 0.0027 against tolerances near 0.005, so they could not have been
rejected by any sampler. The result is positive: no convention mismatch.

Exact references are `exact_reference` (float64 NumPy, 2^5 enumeration,
kernel powers). THRML cells are `software_simulation` (THRML 0.1.4, CPU,
float32). Nothing here is hardware evidence; sweep counts are not device
operations. The [frozen protocol](../../experiments/thrml-finite-sweep-contract.md)
lists every fixed choice. One run took 76 s on one CPU core, including
compilation. Replaying the archive reproduces the request digest, result
digest, archive SHA-256 and every count in `completion.json`; only
`provenance_digest` differs, because a replay records its own provenance.

## What each convention check found

- **Energy sign and temperature.** The `flipped_j` and `inverse_beta`
  references are rejected in all 12 and 12 of their cells, by TV 0.34 to 0.46
  and 0.10 to 0.12 against tolerances of 0.005.
- **Spin encoding and the single-site conditional.** The exact kernel's unit
  test pins P(s_i=+1 | rest) = sigmoid(2 beta h_i); the sampled K = 1 cells
  from the all-minus point mass match that kernel at TV 0.0024 to 0.0035, and
  rule out the K = 0 and K = 2 laws by 0.93 and 0.13 to 0.19.
- **Initialization.** At K = 0 the uniform and all-minus laws match their
  declarations. `hinton_init` matches the product law with
  P(S_i = 1) = sigmoid(beta b_i) at TV 0.0041 and 0.0039 (tolerance 0.0051)
  and sits 0.045 from the sigmoid(2 beta b_i) alternative. THRML's
  initializer uses beta b, not 2 beta b, exactly as its 0.1.4 source says.
  That initial law is not the Gibbs conditional with no neighbours; it is a
  heuristic, and Thermo should model it as such wherever `hinton_init` is
  used.
- **Block order.** Each order matches its own kernel. The other order's
  kernel is rejected at K = 1 from every init (TV 0.009 to 0.20 against
  0.005) and at K = 2 from the all-minus init (0.013).
- **Clamping.** With spin 0 pinned to +1 or to -1 through `clamped_blocks`,
  all 16 cells match the kernel restricted to the consistent states; no
  sampled mass lands outside them, and the K = 30 cells match the conditional
  Boltzmann law at TV 0.0016 and 0.0027.
- **What counts as a sweep.** `SamplingSchedule(n_warmup=K, n_samples=1,
  steps_per_sample=1)` returns the state after exactly K full block sweeps.
  In all **18 cells where p0 T^(K-1), p0 T^K and p0 T^(K+1) are mutually
  separated by more than the tolerance**, the histogram is closest to
  p0 T^K. The remaining 38 sweep cells are undecided because the chain has
  already mixed, and their nearest-law column is noise.

## What this does and does not settle

The contract holds for one five-spin chain at beta 0.8 with two-colour
blocks. Chromatic schedules with more colours, larger graphs, non-spin nodes
and float32 effects at scale are not tested here; the 1e-7 float32 floor is
three orders of magnitude below these tolerances. The backlog's stage B (one
M5a site through THRML at K in {1, 2, 4}) stays open and needs the owner's
reading choice. The existing 200-warm-up THRML gate remains the equilibrium
check; this study is the finite-K check that it was missing.

## Files

- `study.json.gz` (46 KiB, SHA-256
  `a05e7ee695e6fffd3f0a0e1693ed02ebe342a0882e0063cf507f418f0310422b`):
  request, exact references, controls, 64 histograms, evaluation and the
  result digest.
- `summary.md`: the per-cell and per-control tables rendered from the
  archive.
- `completion.json`: 64 cells, 64 passed, 41 of 48 controls rejected,
  18 of 18 decisive off-by-one cells closest to K, replayed.
- `provenance.json`, `run.log`: runtime (JAX 0.10.2 CPU, x64 off, THRML 0.1.4
  pinned by hash, Python 3.11.15) and per-cell compile/execute seconds.
