# Extropic August 2026 release intake

Status date: 2026-09-01 UTC

Extropic's release exposes a layered programming direction rather than only a
THRML update:

```text
Torx → direct software/API execution
  └→ Thermalizers lowering → THRML kernels ← native THRML
                                 └→ local software/API/future Z1
```

## Available components

| Component | Pinned status | Thermo decision |
|---|---|---|
| THRML | `thrml==0.1.4` (tag `v0.1.4`, unchanged) | Hardware-near EBM and Gibbs-sampling dependency |
| Torx | `extro-torx==0.0.1` (tag `v0.0.1`, unchanged) | High-level stochastic-circuit dependency |
| Thermalizers | Paper published; source/repository unavailable | Define experiments and boundaries; no placeholder compiler |
| `extro-sim` | `extro-sim==0.5.0`, uploaded 2026-08-04 | Apache-2.0 authenticated remote execution client; not used by local checked runs |
| Z1 | Systems announced; external access planned | Model topology/cost assumptions; no physical backend yet |

Sources:

- [THRML repository](https://github.com/extropic-ai/thrml)
- [Torx repository](https://github.com/extropic-ai/torx)
- [Torx paper](https://arxiv.org/abs/2608.01612)
- [Thermalizers paper](https://arxiv.org/abs/2608.01615)
- [Z1 announcement](https://extropic.ai/writing/from-one-to-one-billion/)

## Reproducible package identity

| Distribution | Import | Version | Release commit | Wheel SHA-256 |
|---|---|---:|---|---|
| `thrml` | `thrml` | 0.1.4 | `9c4e6fbb800f5e5c627122e668ff1b158ef3782b` | `6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858` |
| `extro-torx` | `torx` | 0.0.1 | `769d2f90abdfda14798fceb521143f4b99d370da` | `e51d6efe0a8bc62fb4b2b417d5e4ac8190e3fb22c9d14d9342c207afdc64a23c` |

THRML requires Python 3.10+, while Torx requires Python 3.11+. Python 3.11 is
the shared project baseline and the only minor version present in both upstream
test matrices.

Both packages use broad lower bounds. A fresh solve can move JAX, jaxlib,
Equinox, NumPy, and SciPy considerably, so the committed `uv.lock`—not only the
two direct pins—is the reproducible environment.

## THRML 0.1.4 contracts

The repository preserves four release-relevant public behaviors:

1. sampler/free-block count mismatch fails during construction;
2. mixed spin/categorical moment accumulation preserves node-state values;
3. a fully visible positive KL phase works without a free positive state;
4. categorical state counts that exceed their integer dtype fail loudly.

These protect against delayed index errors, mixed-dtype corruption, an
all-visible training failure, and silent categorical wraparound.

## Torx 0.0.1 caution

The published wheel is the initial tagged commit. Torx `main` had already moved
ahead by 2026-08-05 with fixes involving repeated-circuit addition, hybrid
dimension inference, and Glauber sigmoid stability. Thermo intentionally uses
the published 0.0.1 API and avoids those affected paths. It will not mix main
source behavior into records labeled as the 0.0.1 release.

## August paper and client update

Both paper sources were revised on 2026-08-13: [Torx
arXiv:2608.01612v2](https://arxiv.org/abs/2608.01612v2) and [Thermalizers
arXiv:2608.01615v2](https://arxiv.org/abs/2608.01615v2). The current GitHub
tags remain Torx `v0.0.1` and THRML `v0.1.4`.

[`extro-sim==0.5.0`](https://pypi.org/project/extro-sim/0.5.0/) was uploaded on 2026-08-04 under Apache-2.0. It is an
authenticated remote compute client (authenticated remote execution), not
Thermalizers, not a compiler, and not hardware. It supplies no basis for a
hosted-execution result in Thermo's checked local experiments.

Thermalizers remains unpublished: its source and repository are unavailable.
Thermo's five-spin PAsymSwap reconstruction is an internal, declared research
method and makes no official Thermalizers-compatibility claim.

## Architectural consequence

Torx is the logical stochastic-program layer; THRML is the hardware-near EBM
layer. The most important research problem is now faithful composition:
locally accurate thermodynamic kernels can accumulate trajectory drift,
conservation leakage, and task error. Future benchmarks therefore report:

- conditional kernel error;
- trajectory drift and invariant violations;
- final readout/task error;
- finite-thermalization residual;
- topology and I/O costs;
- evidence class for every claim.
