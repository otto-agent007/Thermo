# An efficient probabilistic hardware architecture for diffusion-like models

```toml
id = "extropic-dtm-hardware"
title = "An efficient probabilistic hardware architecture for diffusion-like models"
kind = "paper"
status = "asserted"
last_checked = 2026-09-29

[source]
url = "https://arxiv.org/abs/2510.23972v2"
version = "v2"
released = 2025-12-10
note = "v1 posted 2025-10-28. Published in npj Unconventional Computing 3, Article 30 (2026)."
authors = "Jelinčič, Lockwood, Garlapati, Schillinger, Chuang, Verdon and McCourt (Extropic)"
published = "https://doi.org/10.1038/s44335-026-00075-3"

[[claims]]
id = "C1"
status = "asserted"
text = "An all-transistor probabilistic computer can run denoising models directly in hardware."

[[claims]]
id = "C2"
status = "asserted"
text = "A system-level analysis projects performance parity with GPUs on a simple image benchmark at roughly 10,000 times less energy. This is a projection from a hardware model, not a measurement."
```

## Claim

An all-transistor probabilistic computer can run denoising models directly in
hardware. A system-level analysis projects performance parity with GPUs on a
simple image benchmark at roughly 10,000 times less energy.

## Relevance to Thermo

- It is Extropic's discrete, hardware-native counterpart to
  [generative thermodynamic computing](whitelam-generative.md): denoising by
  chained discrete EBMs rather than continuous Langevin dynamics.
- Its energy figure comes from a hardware model. Thermo would label anything
  derived from it `calibrated_projection`, with the full contract in the
  evidence policy.

## Cautions

- The 10,000× figure is a projection, not a measurement of a shipped chip.
- Record the exact version and cost-model assumptions before comparing any
  Thermo count against it.
