# On the equivalence of Hopfield Networks and Boltzmann Machines

```toml
id = "barra-hopfield-boltzmann"
title = "On the equivalence of Hopfield Networks and Boltzmann Machines"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/1105.2790v3"
version = "v3"
released = 2012-01-10
authors = "Adriano Barra, Alberto Bernacchia, Enrica Santucci and Pierluigi Contucci"

[[claims]]
id = "C1"
status = "asserted"
text = "A hybrid Boltzmann machine with N binary visible units and P Gaussian (analog) hidden units, with weights given by the stored patterns, is statistically equivalent to a Hopfield network with Hebbian couplings once the hidden units are marginalized: the N visible units are the neurons and the P hidden units the patterns. The equivalence is in distribution, not pointwise."

[[claims]]
id = "C2"
status = "asserted"
text = "Simulating the Hopfield network's N neurons and N(N-1)/2 synapses can be replaced by the hybrid machine's N + P units and N P synapses."
```

## Claim

A restricted Boltzmann machine with binary visible units and Gaussian hidden
units, with the patterns as weights, has the same visible-unit thermodynamics
as a Hopfield network. The hidden layer trades N²/2 couplings for NP.

## Method

The proof marginalizes the analog hidden units with a Gaussian integral and
compares the resulting visible distribution with the Hopfield Boltzmann
distribution.

## Relevance to Thermo

- It explains the probe's "binary hidden" result. With Gaussian hidden units a
  Hebbian bipartite memory is a Hopfield network. With binary hidden units the
  marginal becomes Σ_μ log cosh(β ξ^μ · σ / √N), which for large β tends to
  Σ_μ |ξ^μ · σ|. Every pattern then pushes each bit equally, and recall gets
  worse than Hopfield's. That reading is Thermo's, not the paper's.
- It frames the coupling count Thermo reports: NP visible–hidden couplings for
  bipartite constructions against N(N−1)/2 for Hopfield.

## Cautions

- The equivalence needs Gaussian hidden units. Binary hidden units, as on
  p-bit hardware, do not give a Hopfield network.
- Results are thermodynamic. They do not measure mixing time or the cost of
  sampling either form.
