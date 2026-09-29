# M5c input model and degree probe

September 28, 2026 (owner local date). **Exploration, not a recorded study or
release gate.** This note checks graph feasibility before choosing a placement
heuristic. It does not alter the M5a or M5b archive, fit, or frozen protocol.
Updated September 29 (UTC) with the owner's approved chain/pruning comparison.

Reproduce from a checkout (CPU, NumPy float64):

```bash
uv run python docs/research/m5c_degree_probe.py
OPENBLAS_NUM_THREADS=1 uv run python docs/research/m5c_chain_probe.py
```

The first script prints the degree/field counts and the 11 over-degree sites.
The second prints JSON with every site's equilibrium errors, changed-edge
indices, finite-K errors, arm summaries, and per-seed work counts. Its stderr
marks completion of equilibrium calculations before finite-K calculations for
each arm. Both authenticate the compressed archive by SHA-256. These scripts
are reproducible **exploration**, with `exact_reference` numerical semantics;
they create no new study archive, runner, or gate.

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

The cap-10 chain is nearly correct at equilibrium but retains a worst-case
readout error near one after 32 sweeps. This demonstrates the accuracy/mixing
tradeoff; it does not establish a useful mixing horizon. The JSON includes
every listed K and also worst-start TV of the entire **reduced color-1 state**
over all its starts (not full free-state TV). At K=32 those reduced-state
maxima are 0.061154, 0.987646, and 0.999995.

All 49 unchanged sites' finite results are also in the JSON. Every original
site's finite-K maximum errors, including the 11 before repair, are checked
against M5b's archived local metrics for all six positive K values. No outer
4096-state transition or trajectory is recomputed in this exploration; none
of these local worst-case quantities substitutes for M5b's outer metrics.

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

## Verification and next step

The probe checks the reduced transition against independent full-state Gibbs
enumeration on signed, unequal-coupling controls, including empty retained or
moved groups and zero chain strength. Maximum control discrepancy was
3.56e-15. It checks K=0 identity, transition normalization, stationary
invariance, and agreement between independently computed stationary readout
marginals for every chained input; the largest such residual was 2.68e-14.
A strong-chain *control only* recovers the original marginal; no reported arm
uses an above-cap chain. Pruning asserts degree at most 16. Unchanged vectors
are compared bitwise and original finite errors replay against the archive.

The input model is settled: **clamped copies**. The chain is not automatically
the degree repair to freeze: cap 1 has a substantial equilibrium floor under
this split, and the high-cap chain mixes slowly. Carry the pruning comparison
into the M5c design. Before claiming placement feasibility, declare a bounded
synthetic patch using the offsets above and resolve actual placement, extra
copy/routing costs, and a task-relevant accuracy/work criterion. Then freeze
one M5c protocol. No hardware latency, energy, or production-valid embedding
follows from this probe.
