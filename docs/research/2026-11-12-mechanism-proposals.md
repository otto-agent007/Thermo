# Three mechanism proposals: reads, global latents and gate margins

2026-11-12. **Proposals, not evidence.** Written in an owner session with
Claude and added to the queue at the owner's request the same day (rows
`margin-budget-copies`, `aggregator-read-gain`, `directed-relay-latent`).
The only new numbers are the exploratory calculation in proposal 1
([`aggregator_read_gain_probe.py`](aggregator_read_gain_probe.py)) and
arithmetic on archived results. Read against main at the merge of PR #154,
charter amendments 1 and 2, the [agenda](2026-10-20-lab-research-agenda.md)
and the [four-week evaluation](2026-11-11-four-week-evaluation.md).

## Where these sit

The four-week evaluation recommends parking the loop until a task with
demonstrated digital cost headroom and a resident model is named. **None of
these proposals names that task.** They are mechanism and architecture
questions of the kind amendment 2 allows ("a synthetic mechanism study can be
useful if it tests a bounded prediction with a design consequence"), each
aimed at a bottleneck the lab has already measured.

| Proposal | Measured bottleneck | Source |
| --- | --- | --- |
| 1. Read-optimal estimation | I/O is 98 to 99.996 percent of modelled energy; a read costs about 239 updates | constraint-layer priced examples; [hardware model](../z1-hardware-model.md) |
| 2. Directed relay trees | A global latent as tied copies misses its gate on finite tie strength | P-0008, D-0005 |
| 3. Margin-budget theorem | Leakage-product floor near e^(−4c) under a coupling cap | [cap leakage analysis](2026-09-23-cap-leakage-analysis.md), M4H, M4I |

Two things apply to all three.

- **Novelty is unchecked.** No literature search was done. A source card is
  owed before any protocol: dithered one-bit quantization and stochastic
  computing (1); asynchronous or stale Gibbs bias bounds (2); noisy-gate
  reliability bounds in the von Neumann and Evans-Schulman line (3).
- **Proposals 1 and 2 depend on one-way couplings.** The hardware model says
  "Directional endpoint couplings are supported" and that symmetry is required
  only to claim a Boltzmann equilibrium. No cost, precision or placement rule
  for a one-way coupling is published. Both proposals treat it as a declared,
  relaxed assumption (charter amendment 1, rule 8) and stop if it is wrong.

Order: 3 first (minutes of exact compute, extends an existing script), then
1, then 2, which builds on 1.

## Proposal 1. Read-optimal estimation with directed aggregator p-bits

**Number to design against:** node reads per estimate of a scalar observable
at a stated tolerance, and the fan-in gain over reading sites directly.

### Thesis

Under the sealed profile an update costs 7.09 fJ, a node read 1.692 pJ and a
full node write 153.6 pJ. Thinning is nearly free, so the scarce resource for
a resident model is information per read, not effective samples per sweep. A
p-bit with one-way couplings from k model sites is a dithered one-bit
quantizer of their block sum. Reading it, instead of the k sites, should
deliver more information per read about the mean and the fluctuations of that
block.

### Mechanism

- Aggregator: a ~ Bernoulli(σ(2(w·Σ sᵢ + b))) over k ≤ 16 sites, with the
  couplings one-way from model to aggregator so the model's stationary law is
  unchanged.
- Thresholds: a fixed bank of aggregators at different b, or one aggregator
  whose b the host adapts. One write is about 91 reads, so occasional
  adaptation is affordable.
- Estimator: maximum likelihood for the block-sum distribution from the
  threshold counts. The mean gives loading or magnetization; the variance
  gives susceptibility or compressibility.
- Trees: a second-level aggregator over 16 first-level aggregators covers 256
  sites per read.

### Exploratory calculation

Exact Fisher information for independent sites, relative to one site read
(exploration, not evidence):

| Site mean | w | Centred | Off by 1 site | Off by 2 | Off by 4 |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.25 | 5.95 | 5.60 | 4.68 | 2.22 |
| 0.5 | 0.5 | 8.62 | 7.97 | 6.26 | 2.19 |
| 0.5 | 1 | 9.71 | 8.91 | 6.82 | 2.08 |
| 0.2 | 0.5 | 7.94 | 6.54 | 3.99 | 0.37 |
| 0.2 | 1 | 9.53 | 7.88 | 4.52 | 0.07 |

The centred limit is (2/π)·k, about 10.2 at k = 16. A two-level tree at site
mean 0.5 gives 74 (w = 0.5, 0.5) to 97 (w = 2, 2) of a possible 256. A
coupling of 0.5 already captures most of the gain, so coarse coefficients
should be enough.

### Prediction and falsifier

On a resident model, an aggregator read at k = 16 needs at least 3 times
fewer reads than sparse, thinned site reads to reach the same tolerance, at 4
to 6 bits and 10 percent static beta jitter. The falsifier is a gain under 3
times in the exact calculation with within-block correlations included.

### Failure modes

- **Off-centre thresholds.** The gain falls to about 2 at four sites off and
  to near zero at low loading with a sharp w. The adaptive or banked threshold
  is part of the design, and its writes are priced.
- **Correlated sites.** Positive within-block correlation widens the block
  sum and lowers the gain. The exact check includes it; the table does not.
- **No one-way couplings.** With symmetric couplings the aggregator
  back-acts: marginalizing it adds an effective multi-body term among the
  sensed sites. That is a different model, and the proposal stops.
- **Scope.** This helps only when the model stays resident. Any task with
  per-query clamps is still write-dominated.

### Cheapest decisive check

Exact, no sampling, under 30 CPU-minutes: the frozen 3×3×16 adsorption pore
from `native-adsorption-eligibility`, with aggregators added to the transfer
matrix. Report reads to reach loading error 0.01 at each frozen chemical
potential.

- **Baseline to beat:** sparse site reads spaced beyond the correlation
  length and thinned in time, not full-array reads.
- **Second case, only on a pass:** susceptibility on the planar ferromagnet
  with the Kac-Ward reference.
- **What it cannot establish:** an advantage claim (the pore's digital
  reference takes about 11 ms per point), and it does not unhold
  `native-adsorption-eligibility`.

## Proposal 2. Directed relay trees for global latents

**Number to design against:** the sweeps and extra sites a global binary
latent needs to reach a stated posterior error, compared with the tied-copy
construction and with the hub degree P-0008 named.

### Thesis

P-0008's tied-copy latent represents the posterior at the tie limit (8-bit
gap 0.0003) but misses Bayes + 0.01 at finite tie strength, and stronger ties
made the sampled gap worse (0.0197 at tie 1.75, about 0.039 at tie 3).
Symmetric copies trade copy fidelity against domain-wall mixing, and both are
exponential in the tie. A one-way copy has no back-action, so it has no
domain walls: errors accumulate with depth, not with mixing time.

### Mechanism

- **Fan-in:** aggregators (proposal 1) in their near-linear regime deliver a
  noisy estimate of each latent unit's logit.
- **Fan-out:** a one-way copy tree broadcasts the latent. A child with r
  agreeing parents at coupling w disagrees with probability about e^(−2wr).
  Depth is log₁₆ n.
- **Law:** the chain is non-reversible. Analyse it as blocked Gibbs in which
  the pixel block sees a latent that is d sweeps stale and the latent sees a
  noisy logit. The expected bias has the form (staleness × per-sweep drift +
  relay noise) × an influence factor, the shape of asynchronous-Gibbs bounds.
  That form is a hypothesis to check against enumeration, not a result.
- **Noise correction:** Gaussian-like logit noise of variance σ² flattens the
  latent's conditional roughly as σ(ℓ/√(1 + πσ²/8)), which a gain rescale
  partly undoes if the cap permits.

### Prediction and falsifier

On P-0008's `t4_h05_q035` archive, a directed construction reaches Bayes +
0.01 at K ≤ 1024 where the tied-copy construction's best priced gap was
0.0166. The falsifier is an exact stationary gap above 0.01 on the reduced
instance at 8 bits; that would say lossy fan-in, not mixing, is the binding
limit.

### Failure modes

- **No detailed balance.** The stationary law has no closed form and must be
  enumerated or bounded. Nothing may be called a Boltzmann equilibrium.
- **Lossy fan-in.** Binary relays cannot sum linearly, and a four-template
  posterior needs accurate logits. This is the most likely stop.
- **Footprint.** Trees add sites and writes at programming time; both are
  counted.
- **One-way couplings**, as above.

### Cheapest decisive check

Under 30 CPU-minutes, exact enumeration of the directed chain's stationary
law on a reduced instance (one latent, a small pixel patch), NumPy only, no
THRML and no training. Metrics: stationary posterior error, sweeps to reach
it from a priced start, extra sites.

The denoising line is closed. This uses the P-0008 archive as a fixture for
an architecture mechanism and does not reopen that line.

## Proposal 3. A margin-budget theorem for bounded-coupling gates

**Number to design against:** the product of cap and input fan-in, c·r, that
a compiled gate needs for a T-operation program to survive at a stated rate.

### Thesis

The frontier L₀₀·L₀₁ ≈ e^(−4c) in the five-spin PAsymSwap family was found
by local search and matched to optimizer resolution at caps 2 to 6. It looks
like a Lipschitz law of the kernel, not an optimizer artifact.

### Proof sketch

In [`cap_probe.py`](cap_probe.py)'s convention each output's log-odds is
2·(hᵢ + Σₖ xₖ J(k→i) + h·jhᵢ). At a fixed hidden state, flipping input bit k
changes that log-odds by exactly 4·J(k→i), so by at most 4c. Inputs 00 and 01
differ in one bit, yet 00 must give output 00 and 01 must give 01 or 10, so
at least one output must flip. Its odds must swing by about 1/(L₀₀·L₀₁),
hence L₀₀·L₀₁ ≳ e^(−4c).

**The gap:** the hidden spin's distribution also shifts with the input,
through the outputs. The sketch is at fixed hidden state. Closing it is the
first piece of work.

### Generalization to test

- Inputs at Hamming distance r, or r clamped copies of one input bit: floor
  e^(−4cr).
- n-bit coefficients at step δ: c = δ·(2^(n−1) − 1), which ties the floor to
  precision.
- Hidden units that see the inputs can amplify margin by majority; the rate
  is a large-deviation exponent to derive.
- Finite K: use the exact K-step conditional M3 validated.

### Prediction and falsifier

With r input copies the best product is e^(−4cr) within optimizer resolution
at caps 1 to 4. The falsifier is a deviation of more than 10 times in either
direction.

**Consequence if it holds.** The cap analysis puts the product needed for 95
percent survival over 500 operations near 10⁻⁷, which is c·r ≈ 4: four
copies at cap 1, two at cap 2. It is consistent with M4I reaching 89.61
percent at cap 4 with r = 1, at the edge.

### Failure modes

- **Not tight outside this family.** M4I's output-output coupling already
  moves the frontier, so the scope must be stated per kernel family.
- **Copies cost writes.** Each clamped copy is a 153.6 pJ write; M5c
  placements already use 272 to 306.
- **Closed line.** The lessons register says not to propose more
  conservation pilots, and amendment 2 parks the Thermalizers work. This is a
  zero-sample theory check that explains a closed result. It does not reopen
  the conservation line.

### Cheapest decisive check

Extend `cap_probe.py` with r ∈ {1, 2, 4} input copies and, separately, one
input-hidden coupling; tabulate the best product against e^(−4cr) and the
exact 500-operation survival against M4I. Minutes on two cores, exact only.

## What none of these establishes

- No physical-hardware claim; every energy figure is the sealed calibrated
  projection.
- No task advantage. A pass on all three leaves the un-park condition (a
  task with digital headroom) open.
- Proposals 1 and 2 price a feature whose Z1 cost is unpublished.
