# Supercritical Fluid Adsorption to Weakly Attractive Solids: Universal Scaling Laws

```toml
id = "native-adsorption-baselines"
title = "Supercritical Fluid Adsorption to Weakly Attractive Solids: Universal Scaling Laws"
kind = "paper"
status = "asserted"
last_checked = 2026-10-20

[source]
url = "https://doi.org/10.1021/acs.jpcc.8b04390"
version = "Journal of Physical Chemistry C 122 (2018), 15558–15566"
authors = "Miroslaw S. Gruszkiewicz, Gernot Rother, Lukas Vlcek and Victoria H. Distefano"
note = "Application claim read from the ORNL publication abstract linked below."

[[claims]]
id = "C1"
status = "asserted"
text = "Lattice-gas GCMC in a modeled silica-aerogel pore framework produced excess adsorption isotherms agreeing with experimental measurements and predicted adsorbate microstructures."
```

Source locator: [ORNL publication abstract](https://www.ornl.gov/publications/supercritical-fluid-adsorption-to-weakly-attractive-solids-universal-scaling-laws/).
The digital implementation claim is recorded separately in the
[Block–Virnau–Preis source card](block-multigpu-ising.md).

## What transfers

The application source supports a practical purpose for adsorption calculations.
It does not validate our narrow pore, the Schneider wall convention, or a
Thermo placement of the aerogel model.

The linked digital source establishes implementation techniques, not performance of
our inhomogeneous three-dimensional pore. Its historical GPU speed ratios and
hardware power must not be extrapolated to this screen.

## Our baseline requirements and hypotheses

- A nine-site transverse slice has 512 occupation states. Exact finite-pore
  transfer computation is a candidate for the cheapest qualifying baseline;
  use factorized operations and include setup and observable evaluation.
  This is our structural derivation, not a result asserted by either source.
- Compare optimized compiled grand-canonical occupancy updates, with lookup
  tables or vectorization where valid. Multispin coding may be worthwhile;
  applicability and measured benefit require checking for the actual fields.
- Evaluate valid cluster methods if competitive; require batched GPU comparison
  before a later scale claim. A slow Python Gibbs loop is insufficient evidence
  of digital headroom.
- Match loading, working-capacity and pair-statistic accuracy. Count programming,
  reads, reductions, random generation, memory and equilibration. Do not assign
  invented MAC costs to bit operations or claim energy advantage from an unknown
  digital-energy denominator.

Lower total energy at matched accuracy remains our hypothesis. These sources
establish neither Thermo accuracy nor a hardware or economic advantage.
