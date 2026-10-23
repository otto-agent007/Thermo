# Native adsorption eligibility: hold this operating point

2026-10-20. **Exploration, not a recorded study.** The owner selected the
adsorption recommendation after the [three-family comparison](2026-10-20-next-task-recommendation.md).
This probe uses the research loop's preliminary-probe allowance. It changes
no frozen study and tests no premise in the closed denoising line.

**Decision: do not promote this allocation to a recorded study.** The exact
8-bit model already misses the loading tolerance at the middle chemical
potential. The sampled allocation is also unresolved by its uncertainty
rules, including in the ideal control. The selected queue row is proposed
as `held`; no additional seeds, sweeps, cap tuning or replacement task are
queued. The result rejects this operating point, not the entire adsorption
family or thermodynamic computing.

## What was fixed and run

The [fixed choices](native_adsorption_probe/fixed-choices.md) specify a
3×3×16 open lattice gas, beta=epsilon=1 and literal positive lateral wall
potential phi=3. It is an uncalibrated, repulsive-wall model-class screen;
128 of 144 sites are adjacent to a lateral wall. It is not a reproduction of
measured adsorption. See the [model source card](../knowledge/sources/schneider-local-adsorption.md).

The spin representation has 144 variables, 327 pairwise bonds and maximum
degree six. Mapping `(x,y,z)` to `(x+4*z,y+z)` gives permitted lattice edges
and no collisions, copies or global latent connections. Its 63×18 bounding
rectangle contains 1134 slots. Actual chip/core shape remains unpublished.

Three chemical potentials were determined from full-precision exact target
loadings 0.2, 0.5, 0.8, then frozen across bits, jitter and starts. The family
field cap is **1.9904094766263793**; coupling cap is **0.25**. Rounding reuses
the constraint layer's codebook. These are fixed family-wide split caps,
not per-point rescaling. Placement, gains, local conditional law and pricing
also reuse or are checked against the layer.

The probe ran ideal, 8-bit/±10% static gain and 6-bit/±10% static gain arms;
three independent seed/gain groups; both empty and full starts; all three
chemical potentials. Each of 54 trajectories used 4096 sweeps, including
1024 burn-in, then 64 reads separated by 48 sweeps. The successful process
used **10.20 CPU seconds** (9.58 seconds in its timed run section), well below
the 30 CPU-minute probe cap. No physical hardware was accessed.

## Exact representation result

These are `exact_reference` equilibrium expectations, with no sampling or
jitter. Full-precision loadings equal their targets to numerical precision.

| Chemical potential | Nominal loading | 6-bit loading | 8-bit loading | 8-bit error |
| --- | ---: | ---: | ---: | ---: |
| -0.164405292069 | 0.200000 | 0.202725 | 0.200734 | +0.000734 |
| 0.484719844298 | 0.500000 | 0.505640 | 0.511078 | **+0.011078** |
| 0.980818953253 | 0.800000 | 0.807690 | 0.795776 | -0.004224 |

The 8-bit middle-point loading error is **0.0110776976334**, exceeding the
fixed 0.01 tolerance. More draws from that rounded equilibrium cannot remove
this bias. Maximum absolute 8-bit field rounding error is only 0.0076413;
small parameter errors need not imply sufficiently small observable errors.

The exact 8-bit working capacity is 0.5950418804 versus 0.6000000000; its
0.00495812 error is within 0.01. Maximum pair-occupation error is 0.0115918,
within 0.02. Loading is the representation failure.

The diagnostic 6-bit exact loading errors happen to be smaller at the middle
point and remain below 0.01 at all three points. These separately scaled,
rounded grids are not nested; observable error need not fall monotonically
with bits at individual fields. This does not license substituting the
diagnostic arm for the preselected operating case or choosing new caps after
seeing the result.

## Finite-budget software result

The operating arm's pooled loading errors and pointwise 95% Student-t
intervals over three independent seed groups are:

| Nominal loading | Mean error | 95% interval of error |
| ---: | ---: | ---: |
| 0.2 | +0.001208 | [-0.001174, +0.003590] |
| 0.5 | +0.015263 | [+0.004787, +0.025740] |
| 0.8 | -0.006000 | [-0.016985, +0.004984] |

Working-capacity error is -0.00720848, interval [-0.0205192, +0.00610226].
The middle point estimate fails loading tolerance; neither its interval nor
the capacity interval establishes the required accuracy.

The gate also examines each start separately, preventing opposing transient
biases from canceling. Below is the largest absolute interval endpoint over
all points and pooled/start-specific estimates, not a simultaneous confidence
bound. The working-capacity column uses paired low/high seed groups.

| Arm | Loading bound (limit 0.01) | Pair bound (limit 0.02) | Capacity bound (limit 0.01) | Paired start agreement |
| --- | ---: | ---: | ---: | --- |
| Ideal | 0.039261 | 0.044479 | 0.035851 | Unresolved |
| 8-bit, ±10% gains | 0.033883 | 0.039079 | 0.036089 | Unresolved |
| 6-bit, ±10% gains | 0.028209 | 0.035544 | 0.040847 | Unresolved |

All three fail the fixed eligibility checks. This is **not evidence that
jitter alone causes the sampled failure**: the ideal control is unresolved
too. Three gain maps do not characterize a device population. Nonuniform
site gains need not share a symmetric Gibbs stationary law. The zero-jitter
exact rounded model is a diagnostic reference, not an assumed equilibrium
for the jittered process.

Eight consecutive batch means and lag-one diagnostics per trajectory are
preserved. Operating-arm loading lag-one correlations range from -0.228 to
+0.203; short noisy diagnostics do not certify mixing or independent reads.

## Reads, writes and conventional baseline

Each trajectory performs **9216 node reads and 288 charged full-node writes**,
with 589824 algorithmic updates and zero clamps. Writes include one complete
parameter program plus one charged initialization per active site. The six
trajectories used for one point therefore cost 55296 reads and 1728 writes;
the full three-point operating evaluation costs 165888 reads, 5184 writes
and 10616832 updates. A retained state is not an independent useful sample.

| Accounting scenario | Joules per trajectory | Joules for the 18-trajectory operating evaluation |
| --- | ---: | ---: |
| Active sites only | 6.40121e-8 | 1.15222e-6 |
| Program, initialize and update all 1134 bounding slots; read 144 model sites | 3.96890e-7 | 7.14403e-6 |

These are `calibrated_projection` scenarios using the sealed profile's
7.09e-15 J/update, 1.692e-12 J/read and 1.536e-10 J/full-node write. Active-only
energy comprises 4.18185e-9 J updates, 1.55935e-8 J reads and 4.42368e-8 J
writes. There are no clamp changes, so the standing read/write clamp band
collapses. The footprint scenario is not an upper bound on an actual system.
Host reduction, interface/idle/board power, actual core transfer amplification
and I/O latency are excluded. No energy-at-qualifying-accuracy point exists
in this probe, so these costs cannot establish an advantage.

The exact digital comparator uses a **512-state factorized forward moment
calculation** to return loading and aggregate occupied-pair fraction. It
does not construct a dense 512² transfer matrix or compute unneeded edge
marginals. Median execution times across five repeats, including parameter
setup and observable computation, were **11.36, 10.92 and 11.00 ms** per point.
This is software wall timing, not digital energy or hardware speedup. The
implementation is a credible exact baseline for eligibility; it is not a
claim of the fastest possible transfer implementation.

For the queue's MAC sensitivity, the implementation's local-field matrix
product alone has `16*512*9 = 73728` scalar dot-product MAC terms per point.
Its **partial**, calibrated projection at 0.1, 1 and 10 pJ/MAC is respectively
7.3728e-9, 7.3728e-8 and 7.3728e-7 J per point. This excludes local-pair
construction, log-add-exp transfers, exponentials, reductions and memory;
it is not a complete digital denominator. Signs and repeated fields may
allow further digital simplification. Do not assign those omitted operations
invented MAC equivalents. No system energy was measured, so the proposed
3× total-energy target and residual host allowance are not established.

Optimized GCMC, valid cluster methods and GPU batching remain strong later
comparators if a new workload is explicitly selected; this failed eligibility
screen does not justify implementing them. See the [baseline source card](../knowledge/sources/native-adsorption-baselines.md).

## Integrity, persistence and reproduction

- [Evidence JSON](native_adsorption_probe/probe.json), 270196 bytes; SHA-256
  `c4db957d7f672eb4e22400eaf610a7db549125871fe193e5da1b051f2a12a7ef`.
- [Probe script](native_adsorption_probe/adsorption_probe.py) and
  [exact helper](native_adsorption_probe/adsorption_reference.py) are exploratory
  artifacts under `docs/research/`, not production runner code.
- Fixed choices and the explicit start-agreement rule were committed locally before
  sampling at `c435c28`. The successful run used `93ccd36`; full source hashes,
  canonical requested/derived input hashes, UTC time and versions are in JSON.
  The [source-history bundle](native_adsorption_probe/source-history.bundle)
  preserves those exact local commits and their pre-run ordering. Publishing
  through the GitHub connector creates a separate delivery commit; it does
  not rewrite the scientific provenance. The bundle requires main commit
  `5a1cd2f` and can be verified with `git bundle verify`.
- One earlier identical allocation reached JSON output and failed on a NumPy
  boolean. No result was saved or inspected. The only subsequent change cast
  interval width to a native float and added a serialization regression check;
  the same scientific inputs, seeds, schedule and thresholds were repeated.
  This is an integrity retry, not a new trial or a selected stochastic result.
- The successful run reports an otherwise clean tracked tree and one unrelated
  pre-existing untracked temporary archive under the old sampling-time report.
  That file was not used, modified or included in this PR. This was exploratory
  execution, not a claimed clean-tree production run.
- Transfer calculations passed 24 small enumeration cases (maximum error
  3.55e-15); gas/spin conversion and aggregate-only references also agree.
  The heterogeneous-gain four-spin one-sweep kernel matches the constraint
  layer exactly; scalar conditional error is at most 5.55e-16.
- An independent audit recomputed all nine exact models, checked the
  aggregate references (maximum discrepancy 9.99e-16), reconstructed all
  uncertainty intervals and gates from saved chain means, and verified
  source/input hashes, rounded arrays, gains, batch summaries and pricing.
  It ran no new trajectories and agreed with holding this allocation.

Preflight, without a stochastic probe rerun:

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check docs/research/native_adsorption_probe
uv run ruff check docs/research/native_adsorption_probe
THERMO_REPO="$PWD" uv run python docs/research/native_adsorption_probe/adsorption_probe.py --self-test
```

The recorded command was the same script with
`--run --output docs/research/native_adsorption_probe/probe.json`; it refuses
an existing output path. Reviewing this decision requires checking the saved
summaries and hashes, not drawing new chains. No historical study was rerun,
no release gate was altered, and no production completion record is created.

## Research decision

Keep the selected allocation held. Do not commission the conditional
two-hour study from the recommendation. It would not repair the exact 8-bit
representation failure, and the observed uncertainty cannot be cured by
post-hoc budget expansion under this probe's rules.

Adsorption still offers a direct local representation, unlike the closed
tied-copy denoising construction. To revisit it requires a new explicit
decision on a scientifically calibrated native model with defensible
precision requirements and digital headroom, followed by its own bounded
question. Changing this pore's caps, temperature, geometry or read schedule
is not an automatic follow-up. The other ranked families remain unselected.
The November 11 charter conclusion remains: no tested task wins on both
quality and cost.

## Follow-up diagnosis: why the probe missed

At the owner's request, a second implementation independently reconstructed
the open box directly in occupation variables and used a dense 512×512
transfer matrix with particle-number moments. This is a verification method,
not a replacement for the efficient digital baseline. It imports neither
the probe nor its reference helper. All nine fixed models agree with the
saved loading values within 1.22e-15, pair statistics within 9.99e-16, and
log partition functions within 5.68e-14. Geometry, boundary coordination,
spin/gas energy identities and transfer factors pass. No new trajectories,
parameter scan, cap change or tolerance change was performed.

The [audit code](native_adsorption_probe/rounding_audit.py) and
[audit record](native_adsorption_probe/rounding-audit.json) preserve the
calculation and original evidence hash.

**The deterministic cause is coherent field rounding.** The 8-bit field
step is 1.9904094766263793/127 = 0.015672515564. Couplings are represented
exactly. At middle loading, all 128 wall-site fields round upward:

| Sites | Nominal field | 8-bit field | Change |
| ---: | ---: | ---: | ---: |
| 8 | -0.50764008 | -0.50152050 | +0.00611958 |
| 64 | -0.25764008 | -0.25076025 | +0.00687983 |
| 56 | -0.00764008 | 0 | +0.00764008 |

The 16 central-site fields round downward, but those sites are already
nearly occupied. At fixed exact couplings, a field change `delta_h` is
equivalent to a local chemical-potential change `2*delta_h`. The wall errors
therefore reinforce a preference for occupation rather than canceling.

The exact nominal midpoint variance of total occupancy is 112.341229;
`d(loading)/d(mu) = Var(N)/144 = 0.7801474`. More directly, linear response
to the **actual nonuniform rounding pattern**, computed as
`Cov(N, sum_i 2*delta_h_i*n_i)/144`, predicts a loading shift of
**0.01104259**, versus the actual **0.01107770**: 99.68% of the shift.
The wall part of that first-order response is +0.01110419; the central-site
part is -0.00006160. This supports rounding amplified by the model's response,
with no implementation discrepancy found. The 6-bit codebook has a different
alignment and a smaller midpoint shift; that does not establish that fewer
bits are generally preferable.

**The sampling failure is insufficient statistical precision, not a
demonstrated mixing failure.** Every ideal-control point estimate, pooled
and averaged separately by start, lies inside its observable tolerance.
Its largest pooled errors are 0.005169 loading, 0.006508 pair occupation
and 0.006087 working capacity. The confidence intervals fail qualification.
At the ideal midpoint, the three seed-group loading estimates are
0.4940321, 0.5110677 and 0.5040148. Their standard error is 0.0049419;
the two-degree-of-freedom t multiplier of 4.30265 gives a half-width of
0.0212634. Start agreement likewise remains uncertified; a persistent
initialization effect was not demonstrated.

The exact variance exposes a weakness in the original allocation: even
assuming independent retained states, 64 reads give a per-trajectory
loading standard error of about **0.00920**. Using the true iid variance
for all 384 midpoint reads, the three-group t-scaled half-width is about
**0.0162** before accounting for mean error. This is a design approximation,
not a new confidence interval or a guarantee that every realization fails.
It nevertheless shows that the allocation was poorly powered to certify
0.01 accuracy, even in an ideal control. This limitation should have been
made explicit when recommending the read/replication budget.

For the 8-bit/jitter midpoint, the sampled mean is 0.5152633. Subtracting
the exact rounded target 0.5110777 leaves **0.0041856**, with exploratory
95% interval **[-0.0062906, +0.0146619]**. That residual is compatible with
zero; it cannot be attributed specifically to jitter or poor mixing.

The exact miss is **1.10777 percentage points against a 1-point tolerance**,
only 0.10777 percentage points beyond the screen's chosen threshold. That
threshold is not an experimentally validated material requirement. Keeping
this frozen allocation held follows the declared rule, but neither this
small precision miss nor the inconclusive sampled estimates rules out native
adsorption, local Gibbs updates or 8-bit hardware as a family.
