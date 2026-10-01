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
note = "v1 posted 2026-08-03. This card is written from the v2 abstract; the full text has not been read."

[code]
card = "extropic-torx"
note = "The paper introduces torx, which Thermo pins."

[[claims]]
id = "C1"
status = "asserted"
text = "Parametrized Stochastic Circuits (PSCs) are a gate-based intermediate representation in which typed local stochastic kernels with tunable parameters compose over explicit binary, categorical and continuous wires. torx constructs, executes and differentiates them in JAX."

[[claims]]
id = "C2"
status = "asserted"
text = "PSC data types and kernels are chosen to match the native operations of emerging probabilistic hardware, so that the energy advantage at that level is not lost to decomposition, communication or control overhead."

[[claims]]
id = "C3"
status = "asserted"
text = "The framework is demonstrated on random walks on graphs, discrete diffusion, stochastic graph networks, jump diffusion and Ising sampling."

[[claims]]
id = "C4"
status = "asserted"
text = "On the X0 subthreshold CMOS test chip, hosted by the XTR-0 desktop platform, p-bits supply physical randomness for Metropolis-Hastings and importance-sampling estimators, which give estimates consistent with a software pseudorandom baseline."
```

## Relevance to Thermo

- It is the paper behind [Torx](extropic-torx.md), which Thermo pins at 0.0.1.
  Thermo's two-gate and weighted-graph-walk experiments run on that library,
  and the paper's graph random walks are the same family as the latter.
- C4 is the only physical-hardware result in this knowledge base. It uses the
  chip as a randomness source and reports estimator agreement, not speed or
  energy.

## Cautions

- Read the full text before citing anything beyond the abstract. In
  particular, record which torx version the paper describes before comparing
  it with Thermo's pinned 0.0.1.
- C4 is Extropic's measurement on its own test chip. Thermo has no access to
  X0, so nothing here becomes `physical_hardware` evidence in a Thermo record.
