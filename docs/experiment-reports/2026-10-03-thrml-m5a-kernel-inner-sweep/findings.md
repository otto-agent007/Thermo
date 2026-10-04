# THRML executes an M5a kernel's inner sweep exactly: 6 of 6 cells match the archived finite-K law

THRML 0.1.4, running one compiled M5a site kernel (M5c primary arm: reading B,
variational, cap 1, seed 0, site 1; 10 clamped blanket inputs, 7 hidden spins,
one output) for exactly K in {1, 2, 4} block-Gibbs sweeps from each incoming
output value, reproduces the archived M5b inner-K law at all 1,024 blanket
inputs. **All 6 cells pass** both predeclared statistics (0.999 multinomial
quantile at N = 65,536 chains per input): the output rate against the
hash-bound `powered_rates` law (worst deviation 0.0051 to 0.0067 against
tolerances 0.0079 to 0.0091) and the 256-state (hidden, output) joint against
the study-local enumeration (worst TV 0.0173 to 0.0183 against 0.0194 to
0.0220). The negative-control gate passed on the exact side before sampling
and the samples reject **24 of 24** wrong references. Every cell is nearest
to p0 T^K, and at K = 4 the K+1 law sits 0.055 to 0.069 away against a 0.009
tolerance, so the sweep count is decisive at every K. The result is positive:
no convention mismatch between THRML's block Gibbs and the M5 finite-K
matrices.

Exact references are `exact_reference` (float64 NumPy; the archived law via
the hash-bound `meta_ebm_thermalization_core.powered_rates`, the kernel vector
via the hash-bound `meta_ebm_topology.load_source`, and a study-local 256-state
enumeration that reproduces the archived output law to 2e-15). THRML cells are
`software_simulation` (THRML 0.1.4, CPU, float32). Nothing here is hardware
evidence; inner sweeps are not device operations. The
[frozen protocol](../../experiments/thrml-m5a-kernel-inner-sweep.md) lists
every fixed choice. One run took 634 s on one CPU core: 118 s exact side
(tolerance draws dominate), 29/54/102 s per cell at K = 1/2/4 (compilation
under 1 s each), 113 s in-run replay. Replaying the archive reproduces the
request digest, result digest, archive SHA-256 and every count in
`completion.json`; only `provenance_digest` differs, because a replay records
its own provenance.

## What each convention check found

- **Energy, encoding and clamping together.** The M5a energy
  `-(J.x+h) y - sum_a (A_a.x+b_a) w_a - sum_a beta_a w_a y` entered THRML as
  an `IsingEBM` at beta = 1 with the inputs as a clamped block. The
  `negated_inputs` reference (every clamp flipped) is rejected in all 6 cells
  by 0.32 to 0.53; a sign error in J, A or the biases would have shown the
  same way. THRML's `clamped_blocks` holds the 10 inputs fixed for all 2^10
  assignments.
- **Block order.** Hidden block then output block matches M5b's
  `inner_kernel`; the output-first kernel is rejected by 0.29 to 0.93.
- **What counts as a sweep.** `n_warmup=K, n_samples=1, steps_per_sample=1`
  returns the state after exactly K sweeps: the K+1 law is rejected in all 6
  cells (0.055 to 0.24) and the K-1 law in all 4 cells where it exists
  (0.082 to 0.93).
- **Finite K is not the limit.** The exact marginal q(x) is rejected in all
  6 cells by 0.22 to 0.65. At this site lambda_max = 0.78, so K = 4 is far
  from equilibrium, and THRML tracks that transient.
- **Hidden start.** With hidden spins at -1 the first hidden block forgets
  them exactly (the exact kernel rows agree to 0.0), so the law depends on
  (x, y0) only, as M5b assumes.

## What this does and does not settle

E0 certified THRML's sweep on a toy chain; stage B certifies it on a kernel
Thermo actually compiled, including clamping of a 10-input blanket and a
two-block hidden/output schedule. M5b's finite-K matrices can now be read as
statements about what THRML does. One site, one seed, one arm: the other 59
compiled kernels share the construction but were not run. Float32 on the
THRML side is three orders of magnitude below the tolerances here and is not
the float32 question that a GPU arm (A2) must answer. Nothing here supports
a hardware claim or an execution-cost number.

## Files

- `study.json.gz` (1.19 MiB, SHA-256
  `f0043e0d85f08e8f024dab8d86985dd01972c6e798c65db4cdf058c5345b9e27`):
  request, bounded exact references (tolerances, control separations,
  structure, parameter SHA-256), 6 x 1,024 x 256 histograms, evaluation and
  the result digest. The per-input laws are recomputed at replay.
- `summary.md`: the per-cell and per-control tables rendered from the
  archive.
- `completion.json`: 6 cells, 6 output and 6 joint passes, 6 nearest to K,
  24 of 24 controls rejected, replayed, no autosave layer.
- `provenance.json`, `run.log`: runtime (JAX 0.10.2 CPU, x64 off, Python
  3.11.15, commit faecfbb) and per-cell compile/execute seconds.
