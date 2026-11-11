# CN101 - A Digital Thermodynamic Computer for Generative AI

```toml
id = "normal-cn101"
title = "CN101 - A Digital Thermodynamic Computer for Generative AI"
kind = "paper"
status = "asserted"
last_checked = 2026-11-11

[source]
url = "https://arxiv.org/abs/2608.00754v1"
version = "v1"
released = 2026-08-01
authors = "Holdijk, Melanson, Mensch, Birchall, Cheung, Lehrter, Aifer, Duffield, Ernst, Salegame, Martinez, Crooks, Cheng, Belateche, Bright, Coles and Sbahi (Normal Computing, with Oxford, Amsterdam, National Taiwan University and Academia Sinica)"
license = "CC BY-NC-ND 4.0"
note = "Blog post: https://www.normalcomputing.com/blog/cn101---a-digital-thermodynamic-computer-for-generative-ai (2026-08-10)."

[code]
note = "None linked from arXiv as of the last check."

[[claims]]
id = "C1"
status = "asserted"
text = "CN101 is a fabricated digital CMOS chip that runs discrete accumulator dynamics with stochastic computing on Bernoulli bitstreams. Per chip: 4 tiles, each a 64 x 64 stochastic matrix multiplier with a 64-neuron bank (16,384 MAC cells, 256 accumulators of 32 bits), a 65-bit LFSR per neuron and a clock of about 500 MHz. Weights are deterministic 8-bit values, not a second stream. The process node is not stated."

[[claims]]
id = "C2"
status = "asserted"
text = "A conditional VAE on natively 8 x 8 handwritten digits runs on one chip: scale-invariant RMSE 0.083 +/- 0.028 against the floating-point reference at 500,000 cycles per tile, with a bias floor of about 0.09 from T = 10^6 set by 8-bit weight quantization and independent of T. This is a measurement on the chip."

[[claims]]
id = "C3"
status = "asserted"
text = "A convolutional flow on CIFAR-10 runs across six chips with pixel RMSE about 0.06 against the floating-point reference. Concurrent execution needs about 10M cycles per pixel against about 620M sequentially (62 times fewer). The model is overfit to ten images, 60 of its 100 layers hit the cycle cap, and the group norms and nonlinearities run on the host."

[[claims]]
id = "C4"
status = "asserted"
text = "Alanine dipeptide conformer free energies on six chips agree with a CPU estimate within 0.8 kJ/mol."

[[claims]]
id = "C5"
status = "asserted"
text = "The paper reports no power, energy, energy-per-sample or GPU speed comparison. Its baseline is the same 8-bit model in floating point on the host CPU, and it defers energy and latency targets to later chips."
```

## Claim

The first fabricated thermodynamic chip from Normal Computing runs small
generative models and a free-energy calculation with digital stochastic
computing and reproduces their floating-point outputs within stated errors.
It is a substrate prototype, not an efficiency result.

## Method

Thermodynamic linear algebra and Ornstein–Uhlenbeck-style dynamics are
discretized into accumulator updates driven by stochastic bitstreams; the
precision cost of a stream scales as ε⁻². Inside a chip, tiles exchange
bitstreams on-chip, and the VAE cascade runs in one dispatch. Across chips the
host reads each output accumulator every window and applies bias,
normalization and activation.

## Relevance to Thermo

- It is the only fabricated-hardware generative sampling result in the card
  set. Its precision limit has the same shape as Thermo's: an 8-bit weight
  register sets a bias floor that more cycles cannot remove (C2), as in M5b
  and the bit-requirement studies.
- Host round trips are on the critical path for every multi-chip and
  normalized model (C3), and they are not priced. That is the I/O cost Thermo
  counts first (D-0002).
- It is a different substrate from Extropic's p-bits: a digital Gaussian/OU
  sampler, not a Boltzmann-machine Gibbs sampler. A comparison with a Thermo
  count needs the substrate stated.

## Cautions

- There are no energy numbers to cite, and no process node. Do not quote it
  for efficiency.
- The CIFAR-10 result reproduces ten training images, not generalization.
- Not peer reviewed at the last check. The appendices (B to F) were checked
  only through a summary.
