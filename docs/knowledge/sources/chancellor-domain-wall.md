# Domain wall encoding of discrete variables for quantum annealing and QAOA

```toml
id = "chancellor-domain-wall"
title = "Domain wall encoding of discrete variables for quantum annealing and QAOA"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/1903.05068v4"
version = "v4"
released = 2019-07-20
authors = "Nicholas Chancellor"

[[claims]]
id = "C1"
status = "asserted"
text = "A discrete variable with N + 1 values can be encoded as the position of a single domain wall in a ferromagnetic chain of N spins with fixed end conditions, using only two-body Ising terms; arbitrary two-variable interactions are also two-body."

[[claims]]
id = "C2"
status = "asserted"
text = "A single spin flip next to the domain wall moves it by one position at no energy cost, while a flip elsewhere creates two domain walls and raises the energy by 4 lambda."

[[claims]]
id = "C3"
status = "asserted"
text = "Compared with one-hot encoding, domain-wall encoding uses fewer spins and, for many realistic problem structures, a less connected interaction graph, giving better embedding efficiency on annealer hardware graphs."
```

## Claim

A categorical variable can be stored as the position of a domain wall in a
short ferromagnetic spin chain. That takes one fewer spin than one-hot, and
every term is two-body.

## Method

Fix the chain's end spins with strong fields. The single domain wall in the
ground manifold can then sit between any of N + 1 pairs of neighbouring spins,
which encodes the N + 1 values (section II). Interactions between variables are
built from two-body products of neighbouring spins. Published in Quantum Science
and Technology 4, 045004 (2019).

## Relevance to Thermo

- It gives a pairwise alternative to the one-hot inhibition penalty for
  emulating a categorical hidden unit on binary sampling hardware. Moving
  between adjacent labels costs nothing, but reaching a distant label means
  walking the wall along the chain. The order in which memories are assigned to
  chain positions can therefore matter. That is Thermo's inference, not the
  paper's.

## Cautions

- The paper addresses quantum annealing and QAOA for optimization. It makes no
  claim about Gibbs sampling, mixing time or equilibrium accuracy.
