# Planar 16-offset ferro: what the recorded run changes (2026-10-08)

Short note for the research loop. The recorded study is
[`2026-10-08-planar-16-offset-ferro`](../experiment-reports/2026-10-08-planar-16-offset-ferro/findings.md)
(proposal P-0001, row `planar-16-offset-ferro`, now `done`). Sweeps are
`software_simulation` and references are `exact_reference`. There is no
hardware claim.

- **Verdict: #92 holds on the 16-offset subgraph.** It holds on a maximal
  planar subgraph of the Z1 16-offset rule, for ferro targets at beta 4.
  Tempering is required past 64 spins. The nine-replica ladder keeps #92's
  budgets (256 sweeps at L = 16, 1024 at L = 32) on every target.
- **The probe overcalled one effect.** The one-seed probe predicted that the
  thin five-replica ladder would slip one budget step at L = 32. On three
  fresh seeds it did not:
  - `tempering5-k1` is `holds`, and `tempering5-k4` is `mixed`.
  - The mechanism the probe found is real. Cold-end acceptance falls more
    than tenfold on the long-edge graph.
  - The effect is smaller than one fourfold budget step: the arm qualifies at
    1024 sweeps with no margin.
  - Future probes should report the qualifying margin, not only the budget,
    before naming a slip.
- **Advice unchanged.** Keep the nine-replica ladder (or wider) above
  64 spins. #92 and the October 8 ladder probe already say this.
- **Open, owner's call.** This answers the planar question only. The
  non-planar degree-16 lattice has no exact reference in reach. A study of it
  would need a different anchor, such as a reference-free convergence
  criterion or a matched exact sub-problem. That is a new row, not a
  follow-up THERMES designs on its own.
