# A Blueprint for Equilibrium-Based Differentiable Continuous-Variable Thermodynamic Computing

```toml
id = "extropic-cv-blueprint"
title = "A Blueprint for Equilibrium-Based Differentiable Continuous-Variable Thermodynamic Computing"
kind = "paper"
status = "asserted"
last_checked = 2026-11-11

[source]
url = "https://arxiv.org/abs/2607.16183v2"
version = "v2"
released = 2026-08-14
authors = "Lockwood, Bejanin, Bus, Chamberland, Huembeli, Schäfer and Verdon (Extropic; Huembeli also Noumenal Labs, Verdon also Waterloo)"
license = "CC BY-NC-ND 4.0"
note = "v1 posted 2026-07-17. 42 pages."

[code]
note = "None linked from arXiv as of the last check."

[[claims]]
id = "C1"
status = "asserted"
text = "Proposes an energy-based thermodynamic computing stack in which the stochastic process is Langevin dynamics with tunable energy potentials. Outputs are equilibrium (Gibbs) expectations estimated by averaging samples, so the Monte Carlo error falls as N^(-1/2)."

[[claims]]
id = "C2"
status = "asserted"
text = "Gradients of expected outputs are estimated from covariances with the energy derivative, from equilibrium samples alone. The paper says this resembles equilibrium propagation but is distinct. Training uses contrastive-divergence-style rules."

[[claims]]
id = "C3"
status = "asserted"
text = "Runtime and energy are analyzed theoretically and numerically, against the Landauer scale k_B T ln 2 at cryogenic temperatures. The projections exclude cryogenic cooling, control electronics, calibration and readout."

[[claims]]
id = "C4"
status = "asserted"
text = "A superconducting double-well thermodynamic neuron was measured below 1.2 K. Its escape energy at low temperature was E_esc/h = 4.02 GHz, almost 3 times the expected value. It rose at about 12 GHz/K against the expected 20.8 GHz/K and levelled off above about 165 mK, possibly from quasiparticle loss. Readout is binary and dispersive, averaged over 1,000 repeats."
```

## Claim

A continuous-variable, superconducting line of thermodynamic computing,
trained with equilibrium covariances, is sketched end to end. One single-neuron
device has been characterized, and it departs from the model by a factor of
about 3.

## Relevance to Thermo

- It is Extropic research, but a separate line from the discrete p-bit Z1
  platform that Thermo's constraint layer models. The paper does not mention
  Z1. Don't borrow its energy scale for Z1 projections.
- The N^(-1/2) averaging cost (C1) is the estimator cost that limited the
  adsorption probe's read allocation. Any covariance-gradient training on
  hardware pays it per gradient.
- Its continuous Langevin model is the substrate of backlog items E1 and E2
  ([Whitelam](whitelam-generative.md)), not of the THRML studies.

## Cautions

- The energy figures are projections that exclude cooling and control, which
  dominate at millikelvin.
- The device measurement disagrees with the model (C4), and the authors call
  the paper a first step.
- The PDF header date (2026-08-24) does not match arXiv's v2 date.
- Not peer reviewed at the last check.
