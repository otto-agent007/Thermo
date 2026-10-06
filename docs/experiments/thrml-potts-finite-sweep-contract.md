# THRML categorical finite-sweep contract on a three-state Potts patch (Potts stage A / A3)

**Status: draft, 2026-10-06, awaiting the owner's go-ahead to freeze.** No
study cell has run. The exploratory probe
(`docs/research/potts_contract_probe.py`) sampled THRML on the same graph with
different parameters (seed 0) to check that the contract is testable; its
output is exploration, not evidence. The frozen study draws fresh parameters
from seed 20261006. Before freezing, only exact-side quantities were computed
for those parameters (control separations and tolerances below). Change any
value after freezing only through a new protocol version.

## Question and scope

After exactly K ordered block-Gibbs sweeps of THRML 0.1.4's
`CategoricalGibbsConditional` from a declared initial distribution p0, does the
sampled state distribution equal p0 T^K, where T is Thermo's exact one-sweep
kernel of a small Potts-type model?

E0 checked THRML's spin path. Its research note lists non-spin nodes and
schedules with more than two colours as untested, and names categorical
conditionals as dependency check A3. Every later result on the Potts charter
track (`PROJECT_CHARTER.md`, track A) will rest on this contract. Stage B, a
native Potts inference or optimization study, needs its own protocol and is
out of scope here.

Evidence classes. Exact references are `exact_reference`: float64 NumPy,
3^6 = 729-state enumeration, kernel powers. THRML cells are
`software_simulation`: THRML 0.1.4 on CPU in float32. Nothing here is
hardware evidence, and sweep counts are not device operations.

## Fixed choices

Every value below goes into the hashed request.

- **Graph.** Six sites on a 2 x 3 patch, rows (0, 1, 2) and (3, 4, 5), with
  nine edges: (0,1) (1,2) (3,4) (4,5) (0,3) (1,4) (2,5) (0,4) (1,5). The two
  parallel diagonals make the chromatic number 3.
- **Schedule.** Colour blocks A = {0, 5}, B = {1, 3}, C = {2, 4}, each an
  independent set, so a block update is a product of single-site conditionals.
  Orders `forward` (A, B, C) and `reversed` (C, B, A), each compared with its
  own kernel.
- **Model.** q = 3 labels {0, 1, 2}. Energy
  `E(c) = -beta * (sum_i h_i[c_i] + sum_e W_e[c_a, c_b])` with beta = 0.8, so
  P(c_i = k | rest) is softmax over k of beta times the local field. Fields h
  (6 x 3) and pair tables W (9 x 3 x 3) are drawn from N(0, 0.8^2) with NumPy
  seed 20261006. W is deliberately not symmetric in (c_a, c_b), so the
  orientation of each table is tested. The generic Potts coupling
  J delta(c_a, c_b) is a special case and is not separately run.
- **THRML encoding.** `CategoricalNode` (uint8 labels). Fields as one
  `CategoricalEBMFactor` over all six nodes with weights beta * h; pairs as one
  factor over (head, tail) node groups with weights beta * W[e, c_head,
  c_tail]. THRML's energy is `-sum W[...]` and its conditional samples
  softmax(theta) with no factor of 2, unlike the spin sampler's
  sigmoid(2 gamma); the `doubled_beta` control below checks this. Numeric
  dtype float32 on the THRML side, float64 on the exact side.
- **Factor construction.** Two arms, each against the same kernel:
  `generic` (`CategoricalEBMFactor`) and `square`
  (`SquareCategoricalEBMFactor`, which merges interaction groups and is the
  path performance code will use).
- **Initial distributions.** All labels 0 (point mass, the most sensitive to
  an off-by-one) and uniform over the 729 states, drawn per chain from its
  init key.
- **Clamped arm.** Site 1 (degree 4) clamped to label 2 through
  `clamped_blocks`, free blocks {0, 5}, {3}, {2, 4}, all-zero init on the free
  sites. The reference is the same kernel restricted to the 243 consistent
  states; its stationary law is the conditional Boltzmann law. Clamping to a
  non-zero label makes a clamp read as 0 detectable.
- **Sweep counts.** K in {0, 1, 2, 3, 4, 8, 16}, run as
  `SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)`.
- **Chains.** N = 400,000 independent chains per cell. One recorded sample
  is one chain's full six-site label vector after K sweeps. Each chain has its
  own JAX key, split into an init key and a sampling key, folded in from one
  root seed (20261006) and the cell index.
- **Statistic.** Total variation between the empirical histogram (729 bins,
  or 243 clamped) and p0 T^K, with the largest per-site marginal error
  reported beside it.
- **Tolerance.** Per cell, the 0.999 quantile of the TV that a
  multinomial(N, p0 T^K) sample itself shows, from 4,000 exact-side draws with
  a fixed NumPy seed.
- **Off-by-one diagnostic.** For every K >= 1 cell, the TV to p0 T^(K-1) and
  p0 T^(K+1), with the nearest law reported. A comparison is called decisive
  only when both neighbours sit more than the cell tolerance from p0 T^K on the
  exact side.

**Cells:** 2 constructions x 2 orders x 2 inits x 7 budgets free, plus
2 constructions x 7 budgets clamped, 70 cells in total.

**What could cap the metric.** With these parameters the chain is close to
stationary by K = 8, so K >= 8 cells lose the power to separate K from K +- 1.
That is a property of the model, not of THRML; the K <= 3 cells and the
all-zero init carry the finite-K information. A histogram with 729 bins raises
the TV tolerance (about 0.011 to 0.013 at N = 400,000) compared with E0's 32
bins, but every gated control below separates by at least 3x that.
Float32 on the THRML side puts a floor near 1e-7 in the conditionals, far
below the tolerances. Three labels and six sites do not exercise uint8
overflow, more than three colours, or mixed spin-categorical factors.

## Convention checks and their negative controls

A control is a wrong exact reference that the samples must reject at the cell
tolerance.

| Convention | Named control | Exact separation (TV), all-zero init, K = 1 / 2 |
| --- | --- | --- |
| Energy sign | `negated_energy`: beta replaced by -beta | 0.987 / 0.961 |
| Softmax scale | `doubled_beta`: the spin-style factor 2 | 0.338 / 0.363 |
| Pair-table orientation | `transposed_w`: W_e[c_b, c_a] | 0.668 / 0.621 |
| Label encoding | `label_shift`: every table read with labels shifted c -> c+1 mod 3 | 0.813 / 0.813 |
| Block order | `reversed_order`: the other order's kernel | 0.348 / 0.100 |
| What counts as a sweep | `off_by_one`: p0 T^(K+1) | 0.229 / 0.043 |
| Clamping | `clamp_as_zero`: site 1 read as label 0 (clamped arm) | 0.382 / 0.482 |

Tolerances at those cells are 0.011 / 0.012 (clamped arm 0.008 / 0.009). From the uniform init every
control also separates at K = 1 and 2 (smallest: `off_by_one` at K = 2,
0.038 against 0.012).

**Gate before sampling.** Every control at K in {1, 2}, all-zero init,
forward order, generic construction, and `clamp_as_zero` at K in {1, 2} in
the clamped arm, must separate from the right reference by more than the cell
tolerance on the exact side. If any does not, the runner stops before any
THRML call. Controls from the uniform init are reported but not gated.

## Outputs and completion

`results/<dir>/`: `study.json.gz` (request, exact references, controls,
histograms, evaluation), `summary.md`, `findings.md` (written after the run),
`provenance.json`, `run.log`, and `completion.json` written last. Replay
recomputes the exact side from the archived request, rejects drift above
1e-12 absolute in any archived exact value, redraws each tolerance from the
archived law with 1e-3 relative slack, re-evaluates the archived histograms
and checks the result digest. It compares numerically, so it passes on any
CPU's BLAS, as the E0 replays do.

`completion.json` must show `status=thrml_potts_contract_complete`,
`cells=70`, `chains_per_cell=400000`, `controls_gate_passed=true`,
`replayed=true`, and the counts of cells passed, controls rejected and
off-by-one-decisive cells. A failed cell is a scientific result about THRML's
categorical kernel or Thermo's reading of it, not an integrity failure.

**Expected cost.** The probe ran five 200,000-chain cells in about 6 s on CPU,
and the exact side takes about 1 s, so the full study should run in a few
minutes including tolerance draws. No autosave or resume layer; restart in a
fresh directory. CI replays the committed archive through a unit test and
does not resample.

## Not claimed

Sampler speed, any Z1 or TSU hardware behaviour, agreement on models larger
than the one tested, more than three colours, q > 3, or mixed
spin-categorical factors.
