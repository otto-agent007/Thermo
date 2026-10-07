# Associative memory stage A: can pairwise binary hidden units stand in for a categorical one?

**Status: frozen 2026-10-06 on the owner's go-ahead, before any study cell
ran.** The owner chose a cap-agnostic range curve over a nominal coupling cap. Two exploratory probes shaped it:
[`associative_memory_probe.py`](../research/associative_memory_probe.py) and
[`am_emulation_probe.py`](../research/am_emulation_probe.py) (pattern seeds
9000+ and 9700+, which this study does not reuse). Their output is exploration,
not evidence. A literature scan found no prior measurement of this question;
the sources it relies on are carded in the knowledge base.

## Question

A dense associative memory whose hidden layer is one categorical unit, with one
label per stored pattern, recalls far better than Hebbian Hopfield. Its visible
marginal is the log-sum-exp energy of exponential-capacity memories
([Demircigil et al.](../knowledge/sources/demircigil-huge-capacity.md),
[Krotov and Hopfield 2021](../knowledge/sources/krotov-hopfield-large-memory.md)).
Binary sampling hardware has no categorical unit. On pairwise binary units, how
much of that recall can each emulation recover, at equilibrium and within a
fixed number of Gibbs sweeps, and at what cost in hidden units, couplings,
coupling range and sequential steps per sweep?

## Task

- N = 24 visible ±1 spins storing P random ±1 patterns, P ∈ {8, 32, 128}.
- A cue clamps the first c bits of one stored pattern, c ∈ {12, 8}. The
  sampler fills in the other N − c bits. With an 8-bit cue capacity matters
  even for the categorical memory: in the probe, recall was 0.86 at P = 128.
- **Recall** is the fraction of missing bits equal to the stored pattern. Exact
  equilibrium recall is computed by enumerating all 2^(N−c) completions, at
  most 65,536, with each hidden layer summed out exactly (below).

## Arms (hidden-layer realizations)

a_μ = β ξ^μ · σ / √N is pattern μ's field, and a_max = β √N its largest possible
value. Penalties and biases are set as fractions of a_max so the grids mean the
same at every β.

| Arm | Hidden units | Energy (log weight) | Exact marginal | Gibbs blocks per sweep |
| --- | --- | --- | --- | --- |
| `categorical` (reference) | 1 categorical, P labels | a_label | log Σ_μ e^(a_μ) | 2: visible, hidden |
| `onehot` | P binary {0,1} | Σ_μ h_μ a_μ − λ (Σ_μ h_μ − 1)² (pairwise inhibition 2λ plus bias) | elementary symmetric polynomials over active count | P + 1: visible, then each hidden unit alone |
| `domainwall` | P − 1 spins in a chain, ends fixed ([Chancellor](../knowledge/sources/chancellor-domain-wall.md)) | Σ_μ a_μ (z_(μ−1) − z_μ)/2 + J Σ_k z_k z_(k+1) | transfer recursion along the chain | 3: visible, odd chain, even chain |
| `bias` | P binary {0,1}, no inhibition ([Bonnaire et al.](../knowledge/sources/bonnaire-latent-features.md)) | Σ_μ h_μ (a_μ − θ) | Σ_μ log(1 + e^(a_μ − θ)) | 2: visible, hidden |
| `hopfield` (baseline) | none | (β / 2N) Σ_μ (ξ^μ · σ)² | closed form | N − c: one visible spin at a time |

Grids: β ∈ {4, 8, 16}. For `onehot`, λ / a_max ∈ {0.1, 0.4, 1.6}. For
`domainwall`, J / a_max ∈ {0.1, 0.4, 1.6}. For `bias`, θ / a_max ∈ {0, 0.2, 0.4,
0.6, 0.8}. As λ or J → ∞ the `onehot` and `domainwall` marginals become the
categorical one exactly; `bias` approaches it only when other patterns' fields
stay below θ. Domain-wall labels follow pattern index, so the order is fixed
and arbitrary.

All sampling runs in THRML 0.1.4: spin factors for the binary arms, and the
categorical node with mixed spin–categorical factors for the reference.

## Fixed choices and what could cap the result

| Choice | Value | Could it bound the result? |
| --- | --- | --- |
| Size | N = 24 | Bounded by exact enumeration over the missing bits. Small N favours every arm's capacity; trends in P matter more than absolute values. |
| Patterns | i.i.d. uniform ±1 | Correlated patterns are harder and are not tested. |
| Sweep budgets | K ∈ {4, 16, 64, 256} | An arm that has not mixed by 256 sweeps is reported as not reached, not as failing at equilibrium. The probe showed `onehot` at large λ and `domainwall` stuck far below their equilibrium. |
| Start state | missing bits uniform, `onehot` and `bias` hidden all off, `domainwall` chain uniform, categorical label uniform | `onehot` is sensitive to the start: whichever unit switches on first can lock in. Starting with every unit off is the natural hardware reset. Other starts are not tested. |
| Penalty grids | up to 1.6 a_max | The probe matched categorical equilibrium recall at about 1.6 a_max. |
| Coupling range | not capped; a range-sensitivity curve instead | Hardware bounds the coupling range, and a large λ or J compresses the Hebbian couplings ([Doucet et al.](../knowledge/sources/doucet-qubo-encoding.md)). A single nominal cap would only re-read the β and penalty grid at an arbitrary number (the M4 lesson), so the study reports recall as a function of the allowed range instead. |

## Development and held-out split

Parameters are chosen per arm, per (P, c) cell and per budget K on
**development** pattern sets (seeds 7000–7003), using sampled recall (and exact
recall for the equilibrium column). The chosen configurations are then run once
on **held-out** pattern sets (seeds 7100–7111). Every comparison below uses
held-out data only.

Each pattern set uses 3 target patterns and 256 independent chains per target.
One recorded sample is one chain's missing bits after exactly K sweeps. Pattern
sets, not chains, are the replication unit for uncertainty: paired bootstrap
over the 12 held-out sets.

## Comparisons

- **Equilibrium:** exact held-out recall of each arm's best equilibrium
  configuration against `categorical` and `hopfield`, per (P, c).
- **Finite budget (primary):** held-out sampled recall per arm at each K,
  against `categorical` at the same K, as a paired difference with a 95%
  bootstrap interval. An emulation **matches** at (P, c, K) if the interval
  lies within ±0.02 of zero. It **falls short** if the interval lies below
  −0.02.
- **Cost:** per arm, hidden units, couplings, coupling dynamic range (largest
  over smallest nonzero |coupling|) of the chosen configuration, Gibbs blocks
  per sweep and total single-unit updates per sweep. Sweep counts are not
  device operations.
- **Range-sensitivity curve:** a configuration's dynamic range D is the
  largest magnitude among all its two-body couplings and single-unit fields,
  divided by its smallest nonzero two-body coupling magnitude. D is taken in
  the ±1 spin form THRML samples, with h = (s + 1)/2 for {0, 1} hidden units,
  and averaged over the pattern sets of a cell. The categorical reference has
  D = 1 because its multi-state unit is native. For R ∈ {2, 4, 8, 16, 32, 64,
  128, ∞}, the **equilibrium curve** chooses, per arm and cell, the
  configuration with the best development exact recall among those with
  D ≤ R, and reports its held-out exact recall. The **finite-budget curve**
  does the same with development sampled recall at each K and is labelled
  development-only (descriptive), because held-out sampling covers only the
  configurations selected without a range limit. When a real hardware range
  is published, the answer is read off these curves.
- **Diagnostics:** for `onehot`, the fraction of samples with exactly one
  hidden unit on, and label switches per chain (the number of sweeps whose
  label differs from the previous sweep's, where the label is the active
  unit's index or "invalid"); for `domainwall`, the number of domain walls.

## Integrity

- Each exact hidden-layer sum (elementary symmetric polynomials, chain
  transfer, closed forms) matches brute-force enumeration on small instances
  in unit tests.
- Preflight: for every arm on a tiny instance (N = 6, P = 3), one THRML sweep
  reproduces the exact one-sweep law within a 0.999 multinomial tolerance, as
  in Potts stage A. This covers the mixed spin–categorical factor and the
  domain-wall fields. Production does not start unless all arms pass.
- Budgets are prefixes of one run per configuration, checked by re-running a
  shorter budget. The archive keeps per-chain recall counts and diagnostics,
  no trajectories. Replay recomputes exact references, selections and
  comparisons, and compares numerically.

## Outputs and completion

One runner (`thermo_lab.am_binary_emulation`), one replay test and one
report. Completion requires `status=am_binary_emulation_complete`, the
preflight passed, all development and held-out cells present, prefix checks
passed and `replayed=true`. Whether an emulation matches is a scientific
result, not an integrity failure.

**Expected cost.** Development sampling covers 39 configurations × 6 cells ×
4 pattern sets, and held-out sampling the selected configurations × 12 sets,
likely over 30 minutes. The runner therefore saves each work unit (one arm,
configuration, cell and pattern set) atomically and resumes, per the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract). It
verifies the request digest and runner source hash before reusing a unit, and an
interruption test covers resume. Runtime is calibrated on probe-only seeds
before the full run.

## Not claimed

Any hardware speed, energy or latency; behaviour beyond N = 24 or for
correlated patterns; the native multi-state p-dit hardware the reference stands
in for; optimality of the tested grids. Sampling is `software_simulation`;
enumeration is `exact_reference`.
