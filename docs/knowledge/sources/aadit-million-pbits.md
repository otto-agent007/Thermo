# Programmable Probabilistic Computer with 1,000,000 p-bits

```toml
id = "aadit-million-pbits"
title = "Programmable Probabilistic Computer with 1,000,000 p-bits"
kind = "paper"
status = "asserted"
last_checked = 2026-11-11

[source]
url = "https://arxiv.org/abs/2606.25313v1"
version = "v1"
released = 2026-06-24
authors = "Aadit, Zhang, Chowdhury, Callahan-Coray, Lee, Bunaiyan, Seshan, Thomas, Twigg, Seawright, Brewer, Srimani and Camsari (UC Santa Barbara, KFUPM, Carnegie Mellon, Siemens)"
license = "CC BY 4.0"

[code]
note = "The paper has a data and code availability section; not checked."

[[claims]]
id = "C1"
status = "asserted"
text = "Networked FPGAs act as one p-bit Ising machine: DSIM-1 is 6 FPGAs in a chain with independent clocks, and DSIM-2 is an 18-FPGA Siemens Veloce proFPGA system (AMD VP1902) with a shared clock. Only 1-bit boundary states cross devices, couplings stay on-chip, and updates are chromatic (graph-coloured) Gibbs with annealing."

[[claims]]
id = "C2"
status = "asserted"
text = "A single timing ratio, eta = f_comm / f_p-bit, sets the trade-off between parallelism and quality. A conservative bound is f_p-bit <= f_comm / (2 N_color C_max). On a 37^3 3D spin glass, quality saturates near eta = 300 (predicted about 305), and above the threshold the distributed machine is indistinguishable from a monolithic one."

[[claims]]
id = "C3"
status = "asserted"
text = "Measured throughput: 2.53 x 10^12 flips/s on DSIM-1 (50,653 p-bits, overclocked at 50 MHz) and 10^12 flips/s on DSIM-2 (10^6 p-bits), 3 x 10^12 overclocked. Overclocking, which lets boundary states go stale, reaches a residual-energy target 62 to 410 times faster on DSIM-1 but only 2.2 to 2.8 times faster on DSIM-2."

[[claims]]
id = "C4"
status = "asserted"
text = "G81 Max-Cut (20,000 spins) reaches the certified optimum of 14,060 on DSIM-1. A 250,011-p-bit 3SAT instance satisfies 55,416 of 55,558 clauses (99.74 percent) after 10^9 sweeps on DSIM-2. An RTX 6000 Ada GPU is a monolithic quality reference, with no wall-clock comparison."

[[claims]]
id = "C5"
status = "asserted"
text = "DSIM-1 draws 150 to 180 W and DSIM-2 1.4 to 1.6 kW, with the method not stated. A modelled 7 nm ASIC partition (ASAP7 place and route) is 0.66 mm^2 at 248 mW, closing at 100 MHz. No measured energy per flip is given."
```

## Claim

Partitioning a p-bit graph across devices costs nothing in quality as long as
boundary communication is fast enough relative to p-bit updates. One ratio
predicts the threshold, and above it a million-p-bit machine behaves like a
monolithic one.

## Relevance to Thermo

- C2 is a communication-versus-update budget like the one Thermo prices
  through the constraint layer. It concerns inter-device boundary traffic,
  while Thermo's dominant cost is host writes and clamp changes, but both say
  that I/O sets the useful update rate. A multi-chip Z1 projection would need
  this ratio.
- It is an FPGA-emulated p-bit machine, a digital pseudo-random substrate, not
  analog p-bits. Treat its flips per second as a conventional-hardware
  reference for chromatic Gibbs throughput, not as TSU evidence.
- Large frustrated 3D spin glasses and Max-Cut are optimization targets.
  Planar annealing reached the same conclusion for frustrated grids at β 4.

## Cautions

- The power figures are board draws with no stated method, and the ASIC
  numbers are modelled.
- The DSIM-2 clock is reported inconsistently: the Fig. 5 caption says 11 and
  33 MHz, while the text says 1, 3 and 33 MHz.
- Some planted-instance and N_color details were checked only through a
  summary.
- Not peer reviewed at the last check.
