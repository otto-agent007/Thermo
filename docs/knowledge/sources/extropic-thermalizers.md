# Thermalizing Stochastic Programs

```toml
id = "extropic-thermalizers"
title = "Thermalizing Stochastic Programs"
kind = "paper"
status = "reproduced"
last_checked = 2026-09-29
reports = [
  "../../experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md",
  "../../experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md",
]

[source]
url = "https://arxiv.org/abs/2608.01615v2"
version = "v2"
released = 2026-08-13
note = "v1 posted 2026-08-03."
authors = "Amico, Jelinčič, Nancarrow, Tyrpak, Roberts, Morton, Sakthivadivel, Gopal and Verdon (Extropic)"

[code]
note = "Thermalizers implementation not yet public (charter, track C)."

[[claims]]
id = "C1"
status = "reproduced"
text = "Stochastic programs can be compiled into thermodynamic kernels for TSU-style hardware. Thermo reconstructs the compilation methods on the PAsymSwap program in the biased-random-walk sequence (Phase 2, M1 to M4)."
reports = [
  "../../experiment-reports/2026-09-10-frozen-pair-finite-sweep-audit.md",
  "../../experiment-reports/2026-09-11-finite-sweep-gradient-contract.md",
  "../../experiment-reports/2026-09-11-bounded-finite-sweep-refinement.md",
]

[[claims]]
id = "C2"
status = "reproduced"
text = "The d = 12 three-body meta-EBM demonstration: its method and qualitative claims, under four recorded open choices. The instance behind the figure is unpublished."
reports = [
  "../../research/2026-09-26-meta-ebm-target-reading.md",
  "../../experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md",
  "../../experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md",
]

[[claims]]
id = "C3"
status = "reproduced"
text = "The plotted 'parameter-free bound' is a guaranteed bound. Negative result: it uses an exact-diagonalization mixing estimate where the Dobrushin coefficient is needed, so it is not guaranteed on matching instances (research note, issue 4)."
reports = ["../../research/2026-09-26-meta-ebm-target-reading.md"]
```

## Claim

The paper proposes compiling stochastic programs into thermodynamic kernels for
TSU-style hardware, and analyzes the error from finite thermalization and
parameter caps. The paper includes a d = 12 three-body meta-EBM demonstration.

## Where Thermo has checked it

- The biased-random-walk sequence (Phase 2, M1–M4) reconstructs its
  compilation methods on the PAsymSwap program.
- M5a and M5b check the meta-EBM demonstration. Four open choices and one
  formula that cannot do what the text says are recorded in the
  [September 26 research note](../../research/2026-09-26-meta-ebm-target-reading.md).
  Results are in the [M5a report](../../experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md)
  and the [M5b findings](../../experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md),
  all `exact_reference`.

## Cautions

- The instance behind the paper's meta-EBM figure is unpublished, so only the
  method and qualitative claims can be reproduced.
- The plotted "parameter-free bound" uses an exact-diagonalization mixing
  estimate where the Dobrushin coefficient is needed. It is not a guaranteed
  bound on matching instances (research note, issue 4).
