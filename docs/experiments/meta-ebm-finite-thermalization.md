# M5b: inner thermalization and precision sensitivity

*Status: frozen protocol approved September 28, 2026 UTC (September 27 owner
discussion). The [full 80-reference/720-cell study](../experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md)
completed with persisted replay on September 28.*

## Question and scope

How does the M5a cap–accuracy tradeoff change when each conditional update has
only K alternating hidden/output updates? Separately, how sensitive is the
exactly marginalized compiled chain to rounding its parameters?

Use the [recorded M5a baseline](../experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md),
with its source-bound evaluator and frozen target definitions. All results
are float64 CPU `exact_reference` calculations of explicitly declared models,
with zero generated samples. No production fitting, topology restriction,
training-objective change, physical-device calibration or hardware advantage
claim belongs in M5b. The small optimizer continuation in
[the exploratory note](../research/2026-09-28-m5b-freeze-probe.md) does not
replace any M5a endpoint.

Pin the compressed M5a archive SHA-256
`e9b59a929c613fde7260580d672a0c5e307b2afc6980b236ea4e9fcf8b79346a`, request digest
`sha256:85ca494706e6f7e2a2634d431e249bace08ea60a098b93a067164cf0a09e4c6d` and result
digest `sha256:474fd591f31dda2ff43acdd69f3e07b84e19adb0ba51af488ab6fd6293b4e036`.
Use stored variational endpoints; reconstruct constructive parameters only
from the pinned M5a recipe and replay the corresponding archived metrics.
Never edit `meta_ebm_cap_baseline.py` or `hashing.py` to implement M5b.

## Fixed grid

| Choice | Frozen choice |
|---|---|
| Targets | Both M5a readings, all five seeds; B primary, A a stronger-interaction stress case |
| Methods | Constructive and variational, with the original M5a parameters |
| Caps | 0.3, 1, 3, 10: low-cap case, primary-reading accuracy transition, intermediate case and largest archived cap |
| Inner K | 1, 2, 4, 8, 16, 32; plus the exact-marginal limit as a shared reference |
| Precision b | 4, 8, 12 bits under the mathematical codebook below, using exact marginalization |
| Outer sweep | One update of sites 0 through 11 in M5a order |
| Outer start/horizon | Uniform over the 4,096 visible states; t = 0 through 30 |
| Inference | Descriptive paired comparisons, no significance or optimality claim |

There are 80 shared baseline chains, 480 finite-K chains and 240 rounded
exact-marginal chains: 720 new scientific cells. Do not cross K with precision
in this release or silently reduce the grid after seeing outcomes. Additional
caps, K values, codebooks or combinations require a dated amendment.

## Inner update, initialization and exact reduction

For one site, clamp its blanket inputs x for the entire inner computation.
Let y be its output and w its hidden spins. Retain the M5a energy

`E(x,w,y) = -(J·x+h)y - sum_a (A_a·x+b_a)w_a - sum_a beta_a w_a y`.

At the start of each conditional update, y is the **current visible value**
of that site. The hidden spins have no retained state: initialize them to -1
by convention and immediately overwrite them in the first hidden block.
One complete inner sweep consists of:

1. Redraw every hidden spin independently conditional on the fixed x and
   current y: `Pr(w_a=+1) = sigmoid(2(A_a·x+b_a+beta_a*y))`.
2. Redraw y conditional on that newly drawn w and x:
   `Pr(y=+1) = sigmoid(2(J·x+h+sum_a beta_a*w_a))`.

Repeat these two blocks K times, write the final y to the visible state, and
discard w. These are model transition probabilities; the exact evaluator
enumerates them and draws no random numbers. K=0 is excluded: it would leave
every visible state unchanged and have no unique outer stationary law.
Repeated hidden-only redraws while holding y fixed are not an inner sweep.

There are no hidden–hidden edges in M5a. Thus the first hidden block forgets
the incoming hidden state exactly. Under this particular schedule, carrying
old hidden values would give the same result as resetting them. This
simplification need not hold for output-first schedules, partial hidden
updates or future kernel families.

For each fixed x, enumerate at most 2^n_h hidden configurations to obtain
`a(x) = Pr(y_new=+1 | y_old=-1,x)` and
`b(x) = Pr(y_new=-1 | y_old=+1,x)`. The output transition is

```text
T(x) = [[1-a, a],
        [b, 1-b]]
```

Its stationary probability is `q(x) = a/(a+b)`, exactly the M5a marginalized
conditional. For every K >= 1, `T(x)^K` still preserves q and satisfies local
detailed balance. The contraction factor is `lambda(x) = 1-a-b`; for this
alternating Gibbs construction it lies in [0,1), up to roundoff. In particular,

`Pr(y_K=+1 | y_0,x) = q(x) + lambda(x)^K * (1[y_0=+1] - q(x))`.

This is the new site transition. It depends on the incoming y. The M5a
`sweep` helper merges the two incoming y states before updating; it cannot
be reused unchanged for finite K. A study-local sweep must propagate their
two transition rows separately.

Local detailed balance is **not** lost at finite K. A full outer systematic
sweep need not itself be reversible, even at the exact-marginal limit.
Separately fitted local conditionals need not be compatible with one joint
distribution, so changing their transition rates can change the outer
stationary law. If the local conditionals are compatible with a common joint
law, finite-K local updates preserve that law. Include that case as a control.

## The cap–inner-mixing tradeoff

Larger caps permit stronger hidden/output couplings, which can move lambda
toward 1 and increase the K needed to approach q. This is a hypothesis to
measure, not an assumption that lambda is monotone in cap or identical across
fitted methods.

For every reading, seed, method, cap and site, report lambda's minimum,
median, 95th percentile and maximum over the uniformly enumerated blanket
inputs. Pin quantiles to NumPy's `method="linear"`, record the maximizing
input (lowest input code breaks ties), and report min(a+b) and the largest K
required to reduce local worst-start TV error below 1e-3 and 1e-6. These are
contraction diagnostics, not effective independent sample counts. Summaries
must be replayable from parameters; do not persist every transition matrix.

Form a and b as separate positive sums; do not obtain a tiny transition by
subtracting a near-one probability from one. Carry `a+b` and use
`log1p(-(a+b))` / `expm1` for powers and escape probabilities when lambda
rounds to one. If the prescribed float64 implementation cannot resolve a
positive transition or a unique stationary law reliably, stop the affected
calculation with a numerical-integrity error and retain its checkpoint. Do
not publish a spurious stationary bias or silently add regularization.
An amended numerical method must be reviewed before completing that grid.

## Separate precision sensitivity

For b in {4,8,12}, set `L = 2^(b-1)-1`, `Delta = cap/L`, and map every
parameter to `Delta * clip(rint(parameter/Delta), -L, L)`. `rint` uses nearest
rounding with ties to even. This is a symmetric zero-containing codebook
with 2^b-1 levels and one unused b-bit code. Apply it to all visible fields,
input couplings, hidden fields, hidden/input couplings and hidden/output
couplings. Keep cap and target fixed; do not refit or rescale the target.

Evaluate the rounded kernels by exact hidden marginalization and the same
outer chain calculations as M5a. Retain the unrounded exact-marginal chain
as the shared reference. Label every comparison **precision sensitivity**;
these bit widths and rounding rules are not a claim about Z1 encoding.

## Required measurements

For every finite-K cell record:

- worst local row TV versus its unrounded marginalized kernel and versus the
  exact target conditional, with per-site and global maxima;
- outer one-sweep maximum row TV versus both the M5a compiled sweep and the
  exact-target sweep;
- stationary TV bias against the target, stationary TV shift from the M5a
  compiled chain, stationarity residual, and mean/max site expectation error;
- visible-law distance against the ideal chain and against M5a for outer
  t=0..30, always from the common uniform start;
- TV distance of that visible law to the true target and to its own
  stationary law at every t=0..30, separating total finite-budget error
  from convergence to the compiled chain's stationary law;
- the lambda summaries and declared inner work: `K * sum_n(n_h(n)+1)` spin
  redraws per outer sweep (72K for these targets). This counts hypothetical
  Gibbs work, not executed samples, device cycles, energy or latency.

For every precision cell record parameter rounding error, local conditional
error and the same outer-chain metrics against the target and unrounded M5a.
For each parameter coefficient define rounding error as rounded minus
original value. Report the maximum and mean absolute error within each site;
for the whole cell report the maximum and mean over all coefficients, with
each coefficient weighted equally. Report both original parameter units and
errors divided by cap. Keep every seed and summarize each of these metrics
across seeds by min/median/max, comparing matching reading, method, cap,
precision and site where applicable. The stored parameters and codebook
must suffice to reconstruct all coefficient errors.
Keep initialization/mixing error distinct from stationary bias. M5a's
`settle` is the first t close to delta_30; it is not a mixing-time or
independent-sample guarantee and must not be relabeled as one.

Report every seed, with descriptive min/median/max across the five seeds.
Preserve negative and non-monotone outcomes. Do not select a winning cap or K
afterward without displaying its accuracy and algorithmic work together.

## Integrity and replay

1. Authenticate the archive, original implementation hashes and new request.
   Replay the 80 selected baseline chains once from their M5a definitions.
2. Verify local row normalization/nonnegativity, the q invariant,
   `q(-)*a = q(+)*b`, and agreement with M5a hidden marginalization. Require
   absolute error <= 1e-12. Also require agreement of the log probabilities
   of both output states and of the two positive detailed-balance flows
   within 1e-8, computing them with stable positive sums/log-sums so tiny
   probabilities do not pass only by an absolute-tolerance test.
3. Independently enumerate the joint hidden/output transition on a small
   fixture and compare with `T^K` at K=1,2,4. Verify hidden reset independence,
   output initialization dependence, and rejection of K=0 as a study cell.
4. Check a compatible joint-model fixture: finite K changes transient behavior
   but preserves its stationary target. Catch a sweep that incorrectly drops
   dependence on the current visible output.
5. **Large-K and exact-limit checks:** the K=infinity implementation must
   reproduce M5a's conditional probabilities, full sweeps and chain metrics.
   Separately choose a per-chain K_star from the worst local contraction so
   every local worst-start error is <= 1e-14; evaluate this power analytically.
   Use `K_star = max_x,site max(1, ceil(log(1e-14)/log(lambda(x))))`, with
   K_star=1 for a zero lambda and stable `log1p(-(a+b))` in the denominator.
   Verify the achieved residual after choosing the integer; the formula uses
   the conservative bound `max_y TV <= lambda^K`.
   Verify that finite K_star approaches the M5a sweep and stationary metrics.
   Use max row TV <= 2e-12 and chain-metric tolerance
   `atol=1e-9, rtol=1e-8`; check the stationary-law residual <= 1e-10.
   Do not assume K=32 is close to equilibrium. Store K_star and the measured
   residuals. If rates cannot be resolved or the stationary solve fails these
   checks, retain the failure and require a numerical-method amendment.
   The local invariant is cheap; full global replay still has a compute cost.
6. Check rounding ties, sign symmetry, zero, cap endpoints and cap compliance.
   An unrounded control must exactly preserve the selected parameter vectors.
7. Recompute persisted metrics from the stored request/parameters before
   publishing `completion.json`. No new fits may occur during replay.

The implementation must make tolerances explicit in its hashed request and
use independent small-model checks before the expensive full grid. A small
fixture and selected persisted replay belong in existing CI jobs; the full
grid remains a local gate. No new preflight subsystem is required.

## Runtime and recovery contract

Measure one representative finite-K and one precision cell before launching
the full grid. Record elapsed time and peak memory, then estimate generation
plus persisted replay from actual timings. Use at most four dense workers on
the previously profiled 8-GiB host, reducing concurrency if available memory
is lower. The full run is expected to exceed 30 minutes.

The study-local runner must atomically autosave each completed cell and
replay unit, bind reuse to the complete request, source and archive hashes,
and support `--resume` in the same output directory. Persist status, flushed
logs, elapsed time, caught errors, resource limits and available OOM counters.
Do not edit or switch the executing checkout during a run. Demonstrate that
interruption preserves completed work and that changed requests are rejected.

Store a bounded gzip record, summary and provenance. Completion requires all
720 new cells, the 80 source baselines, integrity checks and persisted replay.
Do not launch the production grid until its actual command, resume command,
runtime estimate and interruption test are documented in the release gate.

The implemented entry point is `python -m thermo_lab.meta_ebm_thermalization`.
The [release gate](../release-gates.md#m5b-inner-thermalization-and-precision-sensitivity)
documents run, benchmark and resume commands. The
[runtime calibration](../research/2026-09-28-m5b-runtime.md) records a real
SIGTERM/restart exercise and estimates generation plus replay. Calibration
is a bounded subset and cannot satisfy the full-study completion gate.

## After M5b

M5c introduces published connection offsets in an explicitly synthetic lattice
region, with placement overhead and unresolved constraints reported. A later
training/objective stage requires a measured binding constraint and explicit
owner approval; M5b does not reopen the closed M4 conservation line.
