# Thermodynamic significance of QUBO encoding on quantum annealers

```toml
id = "doucet-qubo-encoding"
title = "Thermodynamic significance of QUBO encoding on quantum annealers"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/2601.04402v1"
version = "v1"
released = 2026-01-07
authors = "Emery Doucet, Zakaria Mzaouali, Reece Robertson, Bartłomiej Gardas, Sebastian Deffner and Krzysztof Domino"

[[claims]]
id = "C1"
status = "asserted"
text = "For one-hot and precedence penalties in job-shop scheduling QUBOs, weak penalties generate low-energy infeasible manifolds, while overly strong penalties suppress the effective problem energy scale and increase irreversibility, reducing thermodynamic efficiency (reverse-annealing experiments on D-Wave hardware)."
```

## Claim

Penalty strength in a QUBO encoding is a real trade-off on hardware. Too weak
leaves infeasible states at low energy. Too strong compresses the energy scale
of the problem itself.

## Method

The authors encode job-shop scheduling with one-hot and precedence penalties,
then run reverse annealing on D-Wave hardware to measure energy changes and infer
thermodynamic bounds. Published in New Journal of Physics 28, 054512 (2026).

## Relevance to Thermo

- On hardware with a bounded coupling range, a large one-hot λ forces the
  Hebbian couplings to be scaled down, which lowers their effective β. A
  Thermo study of one-hot emulation must state λ relative to any coupling cap
  and check whether that cap limits recall. It is the same structural trap as
  the M4 conservation line's ±2 cap.

## Cautions

- The setting is quantum annealing for optimization, not equilibrium sampling
  on p-bits.
