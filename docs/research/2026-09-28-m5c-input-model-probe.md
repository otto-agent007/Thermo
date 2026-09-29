# M5c input model and degree probe

September 28, 2026 (owner local date). **Exploration, not a recorded study or
release gate.** This note checks graph feasibility before choosing a placement
heuristic. It does not alter the M5a or M5b archive, fit, or frozen protocol.
Updated September 29 with the approved chain/pruning comparison, reading-B
mixing controls, and the full outer-chain check.

Reproduce from a checkout (CPU, NumPy float64):

```bash
uv run python docs/research/m5c_degree_probe.py
OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_chain_probe.py
OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_outer_probe.py --workers 3
```

The first script prints the degree/field counts and the 11 over-degree sites.
The second prints JSON with every site's equilibrium errors, changed-edge
indices, finite-K errors, arm summaries, and per-seed work counts. Its stderr
marks completion of equilibrium calculations before finite-K calculations for
each arm. Both authenticate the compressed archive by SHA-256. These scripts
are reproducible **exploration**, with `exact_reference` numerical semantics;
they create no new study archive, runner, or gate.
The chain probe now includes B variational caps 3 and 10 alongside the original
three cohorts. The outer probe uses those same five arms and prints JSON lines:
its declared scope, one result per completed arm/seed, then min/median/max
summaries. Every seed and all t=0..30 target distances remain in its output.

## Dated topology source

Extropic's *Thermalizing Stochastic Programs*, arXiv:2608.01615v2 (revised
August 13, 2026), [§II.2.1](https://arxiv.org/html/2608.01615v2), specifies
four connection rules on an integer-coordinate grid:

| Rule `(a,b)` | Four edges from `(x,y)` |
| --- | --- |
| `(1,0)` | `(x+a,y+b)`, `(x-b,y+a)`, `(x-a,y-b)`, `(x+b,y-a)` |
| `(2,1)` | Same four rotations |
| `(2,3)` | Same four rotations |
| `(4,1)` | Same four rotations |

These give 16 neighbors at an interior node, fewer at a boundary. Every
offset changes checkerboard color, so the graph is bipartite. The paper's
[§II.2](https://arxiv.org/html/2608.01615v2) clamps kernel inputs while
thermalizing hidden and output spins, and [Appendix J.2](https://arxiv.org/html/2608.01615v2)
identifies a parity obstruction when one input must couple to both an output
and a hidden spin. The publication supplies the local rule, not a complete
physical-chip graph, boundary/core map, or placement API. Any constructed
patch in M5c must be labeled a **synthetic lattice**.

The paper denotes coupling and field limits separately by `J_max` and
`h_max`; it does not establish that they are equal for a device. The
[M5a baseline](../experiments/meta-ebm-cap-baseline.md) caps every stored
field and coupling coefficient by its tested cap. Folding several coefficients
into one *effective* field changes what that cap means and needs an explicit
field-range assumption.

## Archived-kernel degree count

The [M5b record](../experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz)
(SHA-256 `4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`)
contains the unchanged M5a parameter vectors. For each arm below, this
read-only calculation used all five seeds and all 12 sites: 60 kernels per
arm. An edge is present if its stored float64 coefficient is exactly nonzero;
no fitted coefficient is thresholded. `J` connects blanket input to output,
`A` connects blanket input to hidden spin, and `beta` connects hidden spin to
output, using `meta_ebm_cap_baseline.unpack` and the recorded target
structures. These are graph counts, not achieved placements.

| Arm | Output degree, maximum | Outputs above 16 | Hidden degree, maximum | Inputs needing both checkerboard colors | Shared-visible degree, maximum | Folded fields above tested cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reading B, variational, cap 1 | 18 | 11/60 | 12 | 494/494 input occurrences | 71 | 358/360 |
| Reading A, variational, cap 3 | 18 | 11/60 | 12 | 494/494 | 71 | 353/360 |
| Reading A, constructive, cap 10 | 18 | 11/60 | 3 | 422/494 | 34 | 315/360 |

The blanket has 5–11 inputs and there are 1–8 hidden spins per site in
these targets. Both readings reuse the seed's target graph, which explains
their equal structural degrees. The output maximum is **18** in the actual
archive, even though the separate maxima `|B|=11` and `n_h=8` suggest a
loose bound of 19. For a live shared visible spin, the table counts distinct
neighboring visible and hidden nodes across all site kernels, counting a
visible-to-visible edge once even if both directed site fits use it. This
includes a visible spin's beta links to its **own** kernel's hidden spins.

The folded-field column tests the additional assumption that each effective
field must obey the same numeric cap as the individual stored coefficients.
For each blanket configuration, the largest possible magnitude is
`|h| + sum |J|` at the output and `|b_a| + sum |A_a|` at hidden spin `a`.
The table counts violations among all 360 output/hidden fields in each arm.
The largest effective-field/cap ratios are 4.66, 4.62, and 2.00 respectively.
This is **not** a claim about Z1's actual `h_max`.

## Input choices before placement

- **Input-dependent host fields:** The free graph is a star, with output
  degree at most eight, but the effective-field ranges above often exceed
  the M5a per-coefficient cap. It also requires host calculation and field
  writes for each conditional update. A sourced `h_max` and an explicit I/O
  model are prerequisites for calling this feasible.
- **Clamped input p-bits (approved default):** This matches the paper's
  conditional-kernel semantics. A blanket input used by both `J` and `A`
  needs copies on opposite checkerboard colors; clamping both to the same
  value preserves the ideal conditional but adds nodes and clamp traffic.
  In 11 of 60 site kernels per arm, the output has degree 17 or 18; two
  coupled output nodes have enough aggregate degree arithmetically, but a
  short physical placement is unproven. The exploration below fixes chain
  strength, initialization, update order, and designated-node readout. A chain
  coupling consumes the declared coupling cap; an uncapped chain would change
  the comparison.
  Splitting the output changes the Gibbs schedule, so M5b's hidden-first,
  output-second K cannot be reused as though it counted the same updates.
- **Live shared visible spins:** The measured degree reaches 71 for the
  variational arms and 34 for the constructive arm. This also changes the
  per-kernel clamped-input semantics, so it is not the default M5c model.

With fixed clamped inputs and at most one extra free output node, a kernel
has at most ten free spins (`n_h + 2`) and its conditional can be enumerated
over at most 1,024 free states. The 12-spin outer law remains 4,096 states.
Longer chains or extra free routing nodes would need a fresh enumeration
bound. Degree and parity are only necessary checks: the exact published
offsets and a finite synthetic region may still block a placement.

## Fixed choices for the chain/pruning exploration

Only the 11 over-degree kernels per arm change. The other **49 retain their
M5b parameters and hidden-first/output-second schedule**, and are reported
alongside the changed sites. There is no refitting, quantization, or sampling.

- **Split:** `y1` retains `h`, every nonzero `J`, and the largest-magnitude
  `beta` edges that fit in its remaining `15 - count_nonzero(J)` slots.
  Ties use ascending hidden index. `y2` has zero field and receives the
  remaining beta edges: two edges at degree 17, three at degree 18.
  The six degree-18 sites need three moved edges because the chain itself
  consumes one of `y1`'s 16 slots.
- **Strength:** One ferromagnetic `y1-y2` coupling `c = cap`, hence 1, 3,
  or 10 in the respective arms. Every other coefficient stays unchanged.
- **Readout/reset:** Read `y1`; no vote or tie rule. Initialize both output
  nodes from the incoming logical `y`, and all hidden spins to -1, as in M5b.
  At `K=0`, the returned logical spin is exactly the incoming spin.
- **Colors/schedule:** Color 0 contains `y2` and hidden spins retained on
  `y1`; color 1 contains `y1` and hidden spins moved to `y2`. Each sweep
  redraws color 0, then color 1, with independent Gibbs draws within each
  color. The first redraw overwrites initial `y2`; the initial moved-hidden
  values do affect it. Input copies take the opposite color to their free
  neighbors, with one copy per used input/color. All copies are clamped to
  the same blanket value throughout that conditional update.
- **Pruning:** Independently remove `degree - 16` (one or two) nonzero
  output couplings with smallest magnitude across **both J and beta**.
  Ties use J before beta, then coefficient index. Fields and A are untouched;
  all hidden spins remain and use the M5b update schedule.
- **Order and horizon:** Enumerate equilibrium (`K=∞`) first, then
  `K ∈ {0,1,2,4,8,16,32}`. No accuracy pass threshold or winner-selection
  rule is introduced. This isolates the cap's equilibrium effect before
  interpreting finite thermalization.

## Equilibrium first: finite chain strength changes the conditional

For a fixed input, summing the hidden spins assigned to `y2` produces a
half-log-odds contribution `t`. The original kernel adds `t` to the readout
field; the chain instead adds
`atanh(tanh(c) * tanh(t))`. Thus a finite chain changes the conditional even
after unlimited sweeps. For this fixed split and positive strength, increasing
`c` monotonically reduces the pointwise error versus the original kernel.
Using `c = cap` is therefore the best equilibrium preservation available
within this split's allowed strength range, though it can slow mixing.

Errors below are absolute differences in `P(y1=+1 | x)`, equivalently binary
conditional TV. Maxima cover **all blanket configurations of all 11 changed
sites**. The mean first averages uniformly over each site's input rows, then
uniformly over the 11 sites. These are local errors, not outer stationary bias.

| Arm | Original vs target, max | Chain vs original, max (mean) | Prune vs original, max (mean) | Pruning has smaller per-site max |
| --- | ---: | ---: | ---: | ---: |
| B variational cap 1 | 1.2250e-5 | 0.166278 (0.020115) | 0.076341 (0.012781) | 9/11 |
| A variational cap 3 | 3.4589e-4 | 0.043057 (0.002565) | 0.184567 (0.018678) | 2/11 |
| A constructive cap 10 | 6.1084e-5 | 1.0480e-7 (5.1348e-9) | 0.214644 (0.022284) | 0/11 |

At cap 1, unlimited sweeps still leave a worst conditional shift of **0.1663**.
Pruning reduces that arm's worst shift to **0.07634**, but also substantially
changes its original conditional. Neither error can be repaired by increasing K.
At cap 3, this chain preserves equilibrium better than pruning in 9/11 sites
by the per-site maximum metric. At cap 10, equilibrium preservation is close
to the unchained kernel, but this says nothing about its finite-sweep cost.
There is no universal pruning-or-chain winner across caps or error summaries:
by per-site *mean* error pruning wins 6/11, 0/11, and 0/11 respectively.

The maximum equilibrium chain-disagreement probabilities `P(y1 != y2 | x)`
are **0.36901**, **0.15829**, and **4.1738e-7**. Their uniform-site/input means
are 0.13191, 0.014623, and 2.7761e-8. The JSON also reports chain/prune error
against the true target, avoiding conflation with preservation of M5b's fit.

For the **49 unchanged kernels**, chain and pruning shifts are identically
zero. Their original maximum conditional errors against the target are
1.4361e-5 (B cap 1), **0.118532** (A cap 3), and 6.1020e-5 (constructive
cap 10). The A cap-3 fit error in this unchanged group already exceeds the
chain's worst equilibrium error in the changed group; keep both visible.

## Finite sweeps and mixing

At complete-sweep boundaries the color-1 state `(y1, moved hidden)` is Markov
after summing out color 0. It has at most 16 states in these repairs. The
script powers that exact transition; it does not sample or approximate the
1024-state free-spin law. Equilibrium is computed separately by summing
hidden spins to leave four `(y1,y2)` configurations.

These maxima cover the 11 changed sites, all inputs, and both incoming logical
spins under the declared -1 hidden reset. Each column uses its own schedule;
equal K does **not** mean equal work.

| Arm | K | Original vs original equilibrium | Chain vs original equilibrium | Prune vs original equilibrium | Chain readout vs its own equilibrium |
| --- | ---: | ---: | ---: | ---: | ---: |
| B variational cap 1 | 1 | 0.804872 | 0.910097 | 0.810264 | 0.896033 |
| B variational cap 1 | 32 | 0.043818 | 0.168743 | 0.076353 | 0.061154 |
| A variational cap 3 | 1 | 0.996274 | 0.998959 | 0.996321 | 0.998930 |
| A variational cap 3 | 32 | 0.977277 | 0.987408 | 0.979307 | 0.987423 |
| A constructive cap 10 | 1 | 0.999908 | 0.999998 | 0.999906 | 0.999998 |
| A constructive cap 10 | 32 | 0.999462 | 0.999995 | 0.999460 | 0.999995 |

The reading-A cap-10 chain is nearly correct at equilibrium but retains a
worst-case readout error near one after 32 sweeps. **Reading A's unchained
baseline is already slow**, so these rows alone do not isolate the added chain
penalty. The following reading-B variational controls do. Their equilibrium
and K=32 columns all measure error against the original equilibrium conditional,
maximized over the same 11 changed sites and all inputs (plus both incoming
logical spins at finite K):

| Reading B cap | Chain at equilibrium | Prune at equilibrium | Original at K=32 | Chain at K=32 | Prune at K=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.166278 | 0.076341 | 0.043818 | 0.168743 | 0.076353 |
| 3 | 0.00124835 | 0.0468893 | 0.00173794 | 0.580343 | 0.0468893 |
| 10 | 1.03696e-9 | 0.0463721 | 0.00155408 | 0.999136 | 0.0463721 |

At caps 3 and 10 the original B kernels mix well on this local diagnostic;
chaining creates the slowdown. The output nodes flip sequentially under this
schedule, crossing a temporarily broken-chain state. For an isolated zero-field
pair, the flip probability per sweep is `sech(c)^2 / 2`, asymptotically
`2 exp(-2c)`; inputs and hidden fields modify this barrier in the actual kernels.
The probe tests `c=cap`, not every weaker strength or alternative split. Nor is
pruning error mathematically cap-independent: it depends on the fitted vectors
and removed edges, even though the two higher-cap B rows are similar.

The JSON includes every listed K and also worst-start TV of the entire
**reduced color-1 state** over all its starts (not full free-state TV). For the
original three cohorts, those K=32 maxima remain 0.061154, 0.987646, and 0.999995.

All 49 unchanged sites' finite results are also in the JSON. Every original
site's finite-K maximum errors, including the 11 before repair, are checked
against M5b's archived local metrics for all six positive K values. These local
worst cases do not determine the outer-chain ranking; that check follows below.

## Work per 12-site outer sweep

Every seed has 60 hidden spins summed across its kernels. The unchained and
pruned schedules each cost **72K free-spin redraws** per outer sweep. The chain
schedule costs **74K** for seeds 0–3 and **75K** for seed 4, including `y2` at
only the repaired sites. The `K=∞` comparison is an equilibrium oracle with no
finite redraw count.

Clamped inputs do not undergo Gibbs redraws. Count their assignments separately
instead of treating them as free work. The following clamp-write counts sum
one assignment per input copy per conditional update, in **seed order 0–4**:

| Arm | Original with parity copies | Chain with parity copies | Prune with parity copies |
| --- | --- | --- | --- |
| B variational cap 1 | 184, 200, 204, 196, 204 | 184, 200, 204, 196, 204 | 183, 197, 201, 194, 199 |
| A variational cap 3 | 184, 200, 204, 196, 204 | 184, 200, 204, 196, 204 | 182, 197, 201, 194, 201 |
| A constructive cap 10 | 174, 182, 192, 178, 190 | 172, 175, 188, 173, 182 | 172, 179, 189, 175, 186 |
| B variational cap 3 | 184, 200, 204, 196, 204 | 184, 200, 204, 196, 204 | 182, 197, 201, 193, 200 |
| B variational cap 10 | 184, 199, 204, 196, 204 | 184, 199, 204, 196, 204 | 182, 196, 201, 193, 200 |

Reset writes are another 72 per outer sweep for original/prune, or 74/75 for
chain, assigning hidden spins and output nodes before thermalization. Logical
readout is 12 bits per outer sweep in every arm. For a simple unweighted count
of spin updates including inputs and resets, use `redraws + clamp writes +
reset writes`; e.g. B cap-1 chain, seed 0: **74K + 258**, plus 12 readout bits.
The JSON keeps these terms separate so no device timing equivalence is implied.
These schedules describe one fresh initialization per conditional; caching
values or eliding overwritten resets would be a different accounting choice.

The copy counts satisfy parity only. Physical placement can demand additional
copies, routing nodes, or coefficient traffic. Coefficients are held resident
in this abstract comparison: no flashing/reconfiguration cost is estimated.
These are algorithmic counts and lower bounds on copy requirements, not a
calibrated execution model or evidence of chip feasibility.

## Full 4096-state outer-chain check

Before computing results, the bounded grid was fixed to the **five arms above,
all five seeds, original/chain/prune, and K=4, K=32, or exact equilibrium**:
225 comparisons. K=4 is a short budget where M5b's primary B arm already worked
well; K=32 is M5b's finite-grid endpoint. This is a declared exploratory subset,
not a replacement for M5b's frozen grid. K=0 is excluded here because the identity
outer transition has no unique stationary law.

`m5c_outer_probe.py` uses M5b's unchanged `core.sweep`, `stationary_law`, and
`trajectory` helpers. Each site's two positive escape probabilities are expanded
over the current visible state's blanket code. Sites update in order 0..11;
the outer start is uniform over all 4096 states, and the horizon is 30 sweeps.
Only the over-degree sites receive new rates. For the other 49, chain/prune
reuse the original rate arrays. All 75 original comparisons are recomputed and
checked against M5b's archived stationary bias and complete target-TV trajectory.

The chain's two escape probabilities are separately summed, including rare
transitions below machine epsilon; neither is formed as `1 - near_one`.
Accumulated row-mass roundoff is checked before normalizing the two readout
outcomes. There is no clipping, damping, or regularization of the stationary
solve. With fresh auxiliary resets, the finite-K chained readout need not preserve
its Boltzmann marginal. Thus its **finite-K outer stationary bias** and the
**equilibrium-oracle outer bias** are distinct quantities, as are stationary bias
and transient distance to the target at t=30.

The table reports five-seed medians as **stationary TV bias / TV to target at
t=30**. Medians are computed separately for each metric; paired comparisons
below use matching seeds. The script also prints min/median/max for every row,
the individual seeds, stationarity residuals, and TV to the method's own
stationary law at t=30.

| Arm | K | Original: bias / TV30 | Chain: bias / TV30 | Prune: bias / TV30 |
| --- | ---: | ---: | ---: | ---: |
| B variational cap 1 | 4 | 1.25208e-6 / 1.25200e-6 | 0.0503927 / 0.0503927 | 0.0173731 / 0.0173731 |
| B variational cap 1 | 32 | 1.26406e-6 / 1.26406e-6 | 0.0200013 / 0.0200013 | 0.0181058 / 0.0181058 |
| B variational cap 1 | ∞ | 1.26406e-6 / 1.26406e-6 | 0.0199645 / 0.0199645 | 0.0181058 / 0.0181058 |
| B variational cap 3 | 4 | 3.49465e-7 / 3.49465e-7 | 0.0466927 / 0.0590152 | 0.0138609 / 0.0138609 |
| B variational cap 3 | 32 | 3.52024e-7 / 3.52024e-7 | 0.00529787 / 0.00529630 | 0.0138823 / 0.0138823 |
| B variational cap 3 | ∞ | 3.52024e-7 / 3.52024e-7 | 0.000511508 / 0.000511508 | 0.0138823 / 0.0138823 |
| B variational cap 10 | 4 | 2.40249e-9 / 2.40280e-9 | 0.0470928 / 0.337397 | 0.0137304 / 0.0137304 |
| B variational cap 10 | 32 | 2.40541e-9 / 2.40544e-9 | 0.00595937 / 0.337394 | 0.0137463 / 0.0137463 |
| B variational cap 10 | ∞ | 2.40541e-9 / 2.40544e-9 | 2.45970e-9 / 2.45972e-9 | 0.0137463 / 0.0137463 |
| A variational cap 3 | 4 | 1.21052e-5 / 0.201894 | 0.446184 / 0.411042 | 0.0348460 / 0.192770 |
| A variational cap 3 | 32 | 1.34509e-5 / 0.0875622 | 0.100274 / 0.161222 | 0.0281745 / 0.0866665 |
| A variational cap 3 | ∞ | 1.45417e-5 / 0.0770114 | 0.00559896 / 0.0828500 | 0.0244021 / 0.0779825 |
| A constructive cap 10 | 4 | 1.23214e-5 / 0.301504 | 0.243542 / 0.495972 | 0.0627063 / 0.327981 |
| A constructive cap 10 | 32 | 1.23214e-5 / 0.201761 | 0.0347268 / 0.493214 | 0.0551984 / 0.202474 |
| A constructive cap 10 | ∞ | 1.23214e-5 / 0.0770910 | 1.23244e-5 / 0.0770910 | 0.0509148 / 0.0923994 |

At K=4, original/prune cost **288 redraws per outer sweep**, while chain costs
**296 or 300**. At K=32, these become **2304** versus **2368 or 2400**.
Multiply by 30 for the finite-horizon redraw budget. Clamp/reset writes and
readouts above are additional per-sweep terms; the JSON records them for every
seed. Equilibrium is an oracle, with no finite redraw budget assigned. These
are comparisons at specified K with explicit work, not exactly matched-work
claims.

The repaired reading-B K=32 seed ranges put the median results in context:

| Cap | Chain bias range | Chain TV30 range | Prune bias range | Prune TV30 range |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.007523–0.045732 | 0.007523–0.045732 | 0.005558–0.049109 | 0.005558–0.049109 |
| 3 | 0.002121–0.010228 | 0.002110–0.010245 | 0.003193–0.030633 | 0.003193–0.030633 |
| 10 | 0.002837–0.011899 | 0.082664–0.493340 | 0.003159–0.030340 | 0.003159–0.030340 |

Three conclusions survive the full-chain check:

1. **Pruning loses measurable task quality at unchanged redraw count.** In the
   primary B/cap-1 arm, K=4 target TV rises from 1.252e-6 to 0.01737 median;
   at K=32 it is 0.01811. Its loss is much smaller than the local 0.07634 worst
   case, but it is not removed by evaluating typical outer trajectories.
   The higher-cap B pruning medians remain around 0.014. No acceptance threshold
   was declared, so these are costs to judge, not failed accuracy gates.
2. **Cap-10 chaining has an outer transient penalty.** In B at K=32, median
   stationary bias is 0.00596 but TV30 is 0.33739, despite equilibrium-oracle
   error near the original fit. It loses to pruning on TV30 in all five seeds.
3. **Cap-3 chaining is not ruled out by its local worst case.** In B at K=32,
   it beats pruning on TV30 in four of five seeds, with median 0.00530 versus
   0.01388. At K=4 it loses in all five seeds. B/cap-1 chaining wins three of
   five seeds at K=32 even though its median is worse than pruning's. Reading A
   likewise shows why bias and transient error must be separate: pruning can
   slightly lower TV30 while increasing stationary bias substantially.

## Verification and next step

The probe checks the reduced transition against independent full-state Gibbs
enumeration on signed, unequal-coupling controls, including empty retained or
moved groups and zero chain strength. Maximum control discrepancy was
3.56e-15. It checks K=0 identity, transition normalization, stationary
invariance, and agreement between independently computed stationary readout
marginals for every chained input. The expanded five-arm check now also bounds
accumulated mass error through K=32; its largest local residual was 9.84e-13,
below the 2e-12 tolerance. A below-machine-epsilon escape control agrees with
independent full enumeration to 7.11e-15 in log probability. All previously
reported local metrics remain unchanged at displayed precision (maximum
absolute difference 1.39e-13).
A strong-chain *control only* recovers the original marginal; no reported arm
uses an above-cap chain. Pruning asserts degree at most 16. Unchanged vectors
are compared bitwise and original finite errors replay against the archive.

All 225 outer comparisons completed. The largest discrepancy from the 75
archived original baselines was 5.85e-15; the largest outer stationary residual
was 4.28e-15 and row-sum error 7.78e-16. All five archive-bound implementation
hashes were checked unchanged. The nine B/cap-1/seed-0 comparisons were evaluated
once for runtime calibration and again in the full run, with bitwise-identical
metric/work records. The complete exploration took **685.6 seconds (11.4 min)**
on this session using three CPU workers and one BLAS thread each. This is
evaluator wall time, not modeled device latency. `--first-only` reproduces that
one-seed subset; it is explicitly labeled and does not substitute for the full
five-arm output.

The input model remains **clamped copies**. The measured degree limit and the
full-chain pruning loss now justify proposing a **bounded degree-constrained
refit**: start with just the 11 B/variational/cap-1 sites, freeze the pruning
mask so output degree stays at most 16, keep the cap, hidden count, and M5a
uniform-input KL objective, and retain the other 49 fits exactly. Compare the
refit against original/prune/chain using the same outer metrics and budgets.
Unchanged spin count does not guarantee unchanged mixing after refitting, so
finite-K checks remain necessary. This is a recommendation requiring the M5
protocol's owner approval for training changes; **no refit has been run**.

Degree and parity still do not prove physical placement. A named synthetic
patch, actual placement, and any extra copy/routing costs remain open before
freezing an M5c placement protocol. No hardware latency, energy, or
production-valid embedding follows from this probe.
