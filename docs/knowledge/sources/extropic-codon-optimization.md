# Energy-efficient Codon Optimization on Thermodynamic Hardware

```toml
id = "extropic-codon-optimization"
title = "Energy-efficient Codon Optimization on Thermodynamic Hardware"
kind = "paper"
status = "asserted"
last_checked = 2026-11-11

[source]
url = "https://github.com/extropic-ai/codon_opt"
version = "47275f95137f4fe64576a6fff4d424a4279dd72f"
released = 2026-05-30
authors = "Andraž Jelinčič and Ross C. Walker (Extropic)"
note = "Manuscript read from paper/main.tex at this commit. The README says it is being released on arXiv and was submitted to a journal; no arXiv version existed at the last check."

[code]
url = "https://github.com/extropic-ai/codon_opt"
license = "none"
note = "No license file at the commit read, although the paper says the code is open source. Read, do not copy. Depends on THRML from GitHub."

[[claims]]
id = "C1"
status = "asserted"
text = "mRNA codon optimization of the SARS-CoV-2 spike protein (L = 1,273 positions, E. coli codon usage) is posed as an annealed Potts model on a 2-colourable chain, with the global GC-content term replaced by an adaptive per-chain linear bias updated on the host between sampling bursts."

[[claims]]
id = "C2"
status = "asserted"
text = "Domain-wall encoding compiles the Potts model to 3,147 Ising spins (2.47 per position, at most 5 per chain), maximum degree 12, 4-colourable. Annealing beta (2 to 200) and the chain penalty P on independent log schedules lets early defects shortcut barriers; the paper states that mixing at a single domain wall does not depend on P."

[[claims]]
id = "C3"
status = "asserted"
text = "Under the standard (Fox) weights, mean score over 512 chains (lower is better): Potts sampler 234.6 +/- 0.2 at 10 sweeps, Ising (domain-wall) sampler 243.0 +/- 0.8 at 2 x 10^4 sweeps, tuned genetic algorithm 234.8 +/- 0.4. The paper calls these comparable (within about 4 percent)."

[[claims]]
id = "C4"
status = "asserted"
text = "Projected TSU energy per single-chain run is 82 nJ for the Ising chip (E = K x N x 1.3 fJ, with K = 2 x 10^4 and N = 3,147) and 0.84 nJ for a projected Potts (p-dit) chip that does not yet exist. The 1.3 fJ per spin per Gibbs step comes from the DTM paper's Appendix E model, said to agree with prototype measurements within an order of magnitude."

[[claims]]
id = "C5"
status = "asserted"
text = "On an RTX 3090 the GPU energy is bracketed between FLOPs x peak efficiency (Potts 13 uJ, Ising 20 mJ, GA 59 mJ) and wall time x 350 W TDP (Potts 245 mJ, Ising about 84 J, GA 21 J). The paper reports TSU savings of 10^5 to 10^9 times against a GPU running the GA; the abstract says about 10^6 times."
```

## Claim

A real pharmaceutical optimization problem maps onto a p-bit sampler with
local connectivity. Three methods reach about the same score. The Ising chip
is projected to use five to nine orders of magnitude less energy than a GPU,
and even the most generous GPU bound (idealized Potts) is 160 times the
projected chip energy.

## Method

The energy has three terms: codon usage, a repeat penalty between adjacent
positions, and a global GC-content target. The GC term would need all-to-all
couplings, so a per-chain coefficient λ folds it into the unary biases and is
updated on the host at every annealing step. Domain-wall encoding
([Chancellor](chancellor-domain-wall.md)) represents a K-state codon choice as
K − 1 spins in a thermometer pattern. Codons are sorted by GC count so that one
spin flip changes GC content by at most one. The default Ising run in
`ising/ising_main.py` is 10 annealing steps of 2,000 sweeps on 512 chains.
Every annealing step reflashes the weights for the new β, P and λ. All
sampling is THRML on a GPU, simulating a TSU.

## Relevance to Thermo

- **Domain-wall mixing.** Associative memory stage A found that domain-wall
  chains reproduce the categorical equilibrium but do not mix: 0.58 to 0.76
  held-out recall at 256 sweeps, with 2.6 to 41 walls left per chain
  ([findings](../../experiment-reports/2026-10-07-am-binary-emulation/findings.md)).
  This paper reports a domain-wall Ising run that reaches within 4 percent of
  the Potts score. The settings differ in ways that may explain both results.
  Here the chains are short (at most 5 spins, against P − 1 = 7 to 127 in
  stage A), the target is a low-energy state reached by annealing rather than
  an equilibrium at fixed cold β, and P is ramped from low to high on purpose
  so defects can form and heal. The paper's P-independence claim covers a
  single wall moving; stage A's failure was multi-wall configurations at a
  fixed penalty. That is Thermo's reading, not a tested result. Checking it
  would be a probe, and running one is the owner's call.
- **I/O is not priced.** C4 counts Gibbs updates only, at 1.3 fJ each. Under
  Thermo's sealed Z1 profile (`docs/z1-hardware-model.md`: 7.09 fJ per update,
  1.692 pJ per read, 153.6 pJ per full SRAM write), the same default run would
  be about 0.45 uJ of updates plus 10 full reflashes of 3,147 nodes (about
  4.8 uJ) plus one readout (about 5 nJ): roughly 5.3 uJ, about 65 times the
  paper's 82 nJ. That is this card's arithmetic (analysis, not evidence), and
  it assumes every reflash writes every node. On that accounting, I/O
  dominates (D-0002), and the margin against the idealized GPU Potts bound
  (13 uJ) shrinks from 160 times to about 2.5 times.
- **The baseline comparison is the kind amendment 1 asks for.** It matches
  quality first and then compares cost. The strongest digital baseline is the
  Potts sampler on a GPU, which needs 10 sweeps; the GA needs about 4,600 times
  more FLOPs. Pricing the GPU at wall time × TDP for a single 10^6-FLOP chain,
  where kernel launch overhead dominates, is not a strong baseline.

## Cautions

- Every energy figure is a projection, or a GPU bracket. No TSU ran this
  problem. The Potts chip is hypothetical.
- The headline ratio depends on which baseline is used: 160 times against the
  idealized GPU Potts sampler, up to about 10^9 times against measured GPU
  Ising.
- The repository has no license, so its code can be read but not copied into
  Thermo.
- Not peer reviewed at the last check, and not on arXiv.
