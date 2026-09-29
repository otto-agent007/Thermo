# M5c input model and degree probe

September 28, 2026 (owner local date). **Exploration, not a recorded study or
release gate.** This note checks graph feasibility before choosing a placement
heuristic. It does not alter the M5a or M5b archive, fit, or frozen protocol.

## Dated topology source

Extropic's *Thermalizing Stochastic Programs*, arXiv:2608.01615v2 (revised
August 13, 2026), [§II.2.1](https://arxiv.org/html/2608.01615v2), specifies
four connection rules on an integer-coordinate grid:

| Rule `(a,b)` | Four edges from `(x,y)` |
| --- | --- |
| `(1,0)` | `(x+a,y+b)`, `(x-b,y+a)`, `(x-a,y-b)`, `(x+b,y-a)` |
| `(2,1)` | Same four rotations |
| `(2,3)` | Same four rotations |
| `(4,1)` | Same four rotations |

These give 16 neighbors at an interior node, fewer at a boundary. Every
offset changes checkerboard color, so the graph is bipartite. The paper's
[§II.2](https://arxiv.org/html/2608.01615v2) clamps kernel inputs while
thermalizing hidden and output spins, and [Appendix J.2](https://arxiv.org/html/2608.01615v2)
identifies a parity obstruction when one input must couple to both an output
and a hidden spin. The publication supplies the local rule, not a complete
physical-chip graph, boundary/core map, or placement API. Any constructed
patch in M5c must be labeled a **synthetic lattice**.

The paper denotes coupling and field limits separately by `J_max` and
`h_max`; it does not establish that they are equal for a device. The
[M5a baseline](../experiments/meta-ebm-cap-baseline.md) caps every stored
field and coupling coefficient by its tested cap. Folding several coefficients
into one *effective* field changes what that cap means and needs an explicit
field-range assumption.

## Archived-kernel degree count

The [M5b record](../experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz)
(SHA-256 `4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`)
contains the unchanged M5a parameter vectors. For each arm below, this
read-only calculation used all five seeds and all 12 sites: 60 kernels per
arm. An edge is present if its stored float64 coefficient is exactly nonzero;
no fitted coefficient is thresholded. `J` connects blanket input to output,
`A` connects blanket input to hidden spin, and `beta` connects hidden spin to
output, using `meta_ebm_cap_baseline.unpack` and the recorded target
structures. These are graph counts, not achieved placements.

| Arm | Output degree, maximum | Outputs above 16 | Hidden degree, maximum | Inputs needing both checkerboard colors | Shared-visible degree, maximum | Folded fields above tested cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reading B, variational, cap 1 | 18 | 11/60 | 12 | 494/494 input occurrences | 71 | 358/360 |
| Reading A, variational, cap 3 | 18 | 11/60 | 12 | 494/494 | 71 | 353/360 |
| Reading A, constructive, cap 10 | 18 | 11/60 | 3 | 422/494 | 34 | 315/360 |

The blanket has 5–11 inputs and there are 1–8 hidden spins per site in
these targets. Both readings reuse the seed's target graph, which explains
their equal structural degrees. The output maximum is **18** in the actual
archive, even though the separate maxima `|B|=11` and `n_h=8` suggest a
loose bound of 19. For a live shared visible spin, the table counts distinct
neighboring visible and hidden nodes across all site kernels, counting a
visible-to-visible edge once even if both directed site fits use it.

The folded-field column tests the additional assumption that each effective
field must obey the same numeric cap as the individual stored coefficients.
For each blanket configuration, the largest possible magnitude is
`|h| + sum |J|` at the output and `|b_a| + sum |A_a|` at hidden spin `a`.
The table counts violations among all 360 output/hidden fields in each arm.
The largest effective-field/cap ratios are 4.66, 4.62, and 2.00 respectively.
This is **not** a claim about Z1's actual `h_max`.

## Input choices before placement

- **Input-dependent host fields:** The free graph is a star, with output
  degree at most eight, but the effective-field ranges above often exceed
  the M5a per-coefficient cap. It also requires host calculation and field
  writes for each conditional update. A sourced `h_max` and an explicit I/O
  model are prerequisites for calling this feasible.
- **Clamped input p-bits (primary candidate):** This matches the paper's
  conditional-kernel semantics. A blanket input used by both `J` and `A`
  needs copies on opposite checkerboard colors; clamping both to the same
  value preserves the ideal conditional but adds nodes and clamp traffic.
  In 11 of 60 site kernels per arm, the output has degree 17 or 18; two
  coupled output nodes have enough aggregate degree arithmetically, but a
  short physical placement is unproven. Chain strength, initialization and
  readout (including ties if using a two-node vote) must be fixed before
  measuring accuracy or mixing. A chain coupling also consumes the declared
  coupling cap; an uncapped chain would change the comparison.
- **Live shared visible spins:** The measured degree reaches 71 for the
  variational arms and 34 for the constructive arm. This also changes the
  per-kernel clamped-input semantics, so it is not the default M5c model.

With fixed clamped inputs and at most one extra free output node, a kernel
has at most ten free spins (`n_h + 2`) and its conditional can be enumerated
over at most 1,024 free states. The 12-spin outer law remains 4,096 states.
Longer chains or extra free routing nodes would need a fresh enumeration
bound. Degree and parity are only necessary checks: the exact published
offsets and a finite synthetic region may still block a placement.

**Next decision:** Specify the clamped-copy and output readout rules on a
named synthetic patch, and measure topology residual, chain breaks, mixing,
copy/clamp/read traffic, and exact conditional error on the three arms above.
Keep this as a bounded exploration under `docs/research/`; freeze one M5c
protocol only after its input model and cap assumptions are reviewed. No
hardware latency, energy, or production-valid embedding follows from this
probe.
