# A Framework for Stochastic Differentiable Programming

```toml
id = "extropic-torx-paper"
title = "A Framework for Stochastic Differentiable Programming"
kind = "paper"
status = "asserted"
last_checked = 2026-10-01

[source]
url = "https://arxiv.org/abs/2608.01612v2"
version = "v2"
released = 2026-08-13
authors = "Verdon, Tyrpak, Lockwood, Morton, Neagoe, Sugolov, MacCormack and Amico (Extropic)"
note = "v1 posted 2026-08-03. Full v2 text read 2026-10-01. The paper names no torx version or code license; it points to torx-docs.vercel.app."

[code]
card = "extropic-torx"
note = "The paper introduces torx, which Thermo pins."

[[claims]]
id = "C1"
status = "asserted"
text = "Parametrized Stochastic Circuits (PSCs) are ordered layers of local stochastic gates with disjoint supports, acting on binary (pbit), categorical (pdit) and continuous (pmode) wires. Every PSC unrolls to a directed factor graph of kernels. Gate types match the native primitives of probabilistic hardware, so the cost of lowering stays visible."

[[claims]]
id = "C2"
status = "pinned_dependency"
text = "torx's StateVectorSimulator propagates a circuit's full 2^n probability vector, exact up to floating-point error. Thermo's torx_statevector and weighted-graph-walk backends use it in the pinned 0.0.2 (0.0.1 until 2026-10-07)."

[[claims]]
id = "C10"
status = "asserted"
text = "torx also samples circuits with SampleSimulator and HybridSampleSimulator (mixed discrete and continuous wires), and a hardware backend swaps the pseudorandom source for X0 p-bits without changing the circuit. Neither name, nor any X0 or XTR-0 backend, appears in torx 0.0.1 or 0.0.2; their sampled simulator is BranchingSimulator."

[[claims]]
id = "C3"
status = "asserted"
text = "Three gradient estimators run as circuits: REINFORCE for any smoothly parametrized gate; a parameter-shift rule for sigmoid-mixture gates G = (1 - sigma(theta)) I + sigma(theta) B, using the exact identity dG/dtheta = G(theta) - G(theta-minus); and an energy-based estimator for Boltzmann-conditional gates."

[[claims]]
id = "C4"
status = "asserted"
text = "Lie-Trotter splitting of k local generators over N steps has global error O(t^2 k^2 / N); Strang splitting reaches O(t^3 k^3 / N^2) at 2k - 1 gates per step. Euler kernels I + tau Q_j add an O(k t^2 / N) error, so Lie-Trotter suffices when only Euler gates are available. Grouping commuting generators by graph coloring replaces k with the number of colors."

[[claims]]
id = "C5"
status = "asserted"
text = "Diffusion on a 5-node weighted graph (Q = -L) is a Trotterized circuit of one PSWAP gate per edge: PSWAP(w tau) = I + tau Q_ij for the Euler step and PSWAP((1 - e^(-2 w tau)) / 2) = e^(tau Q_ij) for the exact one."

[[claims]]
id = "C6"
status = "asserted"
text = "On an 8-site Ising ring at beta = 1.5, composing one two-site PIsing (Glauber) gate per edge does not sample the Boltzmann distribution: TV 0.402, against 0.052 for chromatic Gibbs over 6,000 chains and 240 sweeps. A single per-site gate holding the exact Gibbs conditional matches the exact distribution."

[[claims]]
id = "C7"
status = "asserted"
text = "On 28 x 28 binary MNIST with 10 solver steps, adding K = 50 PCNOT gates to a tau-leaping reverse step lowers bit-error rate from 0.114 to 0.113 and closes 41% of the FID gap between corrupted and clean distributions."

[[claims]]
id = "C8"
status = "asserted"
text = "A stochastic graph network (one PIsing gate per edge, parameters tied under graph automorphisms, N = 5 sweeps, S = 1,024 samples per step, Metropolis search) recovers the optimal cut of 12 on a random 3-regular 8-node graph in the majority of runs."

[[claims]]
id = "C9"
status = "asserted"
text = "With Bernoulli randomness from one calibrated X0 p-bit (via XTR-0) and 20,000 samples on the same 8-site ring, importance sampling reaches TV 0.0622 against 0.0525 for software randomness, and Metropolis-Hastings 0.1126 against 0.0887. The authors call these consistent with the software baselines."
```

## What it is

The paper behind [Torx](extropic-torx.md). It defines PSCs, an elementary gate
library (PNOT, PSWAP and PCNOT for p-bits, Gaussian and Ornstein–Uhlenbeck
gates for pmodes, controlled shifts and mixtures of Gaussians for hybrids, and
the generator-derived PIsing gate), product-formula approximations and gradient
estimators. It then runs six toy demonstrations, sized "to showcase the
capabilities that a PSC can express rather than to show an efficiency or
scalability advantage". The companion [Thermalizers](extropic-thermalizers.md)
paper compiles PSCs onto Z1.

## Relevance to Thermo

- **Already in use.** Thermo pins torx 0.0.2, and its `torx_statevector`
  backend is the StateVectorSimulator of C2. The weighted-graph-walk baseline
  reimplements C5's fixture and gate order from the v1 text
  (`configs/experiments/torx-weighted-graph-walk.toml`, `docs/studies.md`). No
  Thermo report compares its numbers with the paper's figure, so C5 stays
  `asserted`.
- **Backlog E0.** C6 is the paper's own demonstration that "what counts as a
  sweep" changes the stationary law: tiling two-site Glauber gates is not a
  Gibbs sweep. E0's conventions table should cite it. The paper's PIsing gate
  also uses {0, 1} wires with spins 2s − 1, the spin-encoding row in that
  table.
- **Gradients.** M3 checked exact finite-sweep gradients for a discrete
  circuit by direct chain rule, autodiff and finite differences. C3's
  estimators are what a sampled torx circuit would be checked against.

## Cautions

- C9 rests on one run per method, with no uncertainty or repeated runs
  reported. Hardware randomness gave the higher TV in both methods, and the
  per-p-bit calibration error is about 5% in absolute probability. Treat
  "consistent" as the authors' reading.
- The quoted p-bit energy ("hundreds of attojoules to single femtojoules per
  sample") and rates come from cited device papers, not from this experiment.
- C9 is Extropic's measurement on its own test chip. Thermo has no access to
  X0, so nothing here becomes `physical_hardware` evidence in a Thermo record.
- The paper doesn't say which torx version it describes, and its API names
  don't all match the releases (C10). Checked against the 0.0.1 and 0.0.2
  wheels on 2026-10-01: both have StateVectorSimulator, AffineGaussianSimulator
  and BranchingSimulator, and neither has a hardware backend. Check behavior
  against the pinned 0.0.2 before relying on any API detail.
