# Generative thermodynamic computing

```toml
id = "whitelam-generative"
title = "Generative thermodynamic computing"
kind = "paper"
status = "asserted"
last_checked = 2026-09-29

[source]
url = "https://arxiv.org/abs/2506.15121v3"
version = "v3"
released = 2025-10-30
note = "v1 posted 2025-06-18."
authors = "Stephen Whitelam"

[code]
note = "None linked from arXiv as of the last check."

[[claims]]
id = "C1"
status = "asserted"
text = "A physical system governed by Langevin dynamics can generate structured data from noise by its own time evolution, without a neural network doing the denoising."

[[claims]]
id = "C2"
status = "asserted"
text = "Training the system to reproduce time-reversed noising trajectories makes it generate data with minimal heat emission. Shown in digital simulation."
```

## Claim

A physical system governed by Langevin dynamics can generate structured data
from noise by its own time evolution, without a neural network doing the
denoising. Training the system to reproduce the reverse of noising
trajectories also makes it generate data with minimal heat emission. The paper
demonstrates this in digital simulation.

## Method

Data are noised by a forward stochastic process. The thermodynamic computer's
couplings are trained to maximize the probability that its own Langevin
dynamics produce the time-reversed noising trajectories. Generation then runs
the physical dynamics from noise, with no injected noise schedule or active
denoising control.

## Relevance to Thermo

- It is the continuous analogue of Extropic's denoising hardware proposal
  ([card](extropic-dtm-hardware.md)), so the two can be compared on method.
- Its heat claim can be tested on a toy target where the data distribution is
  known exactly. See backlog item E2.

## Cautions

- Read v3, not v1, before designing a study. Record which version a protocol
  follows, as M5a did for the Thermalizers paper.
- Whether the couplings are fixed or time-dependent during generation affects
  both the heat accounting and the cost model. Check the paper's definition
  before choosing a parameterization.
- "Minimal heat" is a statement about the model's stochastic thermodynamics,
  not a measured device energy.
