# Planar 16-offset ferro: summary

**Verdict: #92 holds on the 16-offset subgraph.** This is the pre-registered
row verdict. Clauses (a) and (b) both pass.

- **Graph.** `greedy-long` is a maximal straight-line planar subgraph of the
  Z1 16-offset rule. At L = 32, about 42 percent of its edges are long. It is
  paired with open grids under the same coupling seeds (4510 to 4512).
- **Ordinary arms.** At 256 and 1024 spins, the long chain and five cold
  chains qualify on no `greedy-long` target within 4096 sweeps.
- **Tempering.** Every tempering arm qualifies on 3/3 targets. The
  nine-replica ladder qualifies at 256 sweeps (L = 16) and 1024 sweeps
  (L = 32), on every graph and seed.
- **Expectation 3 was wrong.** The probe predicted a thin five-replica slip
  at L = 32; it did not occur. `tempering5-k1` is `holds` at 1024 everywhere
  and `tempering5-k4` is `mixed`. The thin ladder's cold-pair acceptance
  still drops more than tenfold on `greedy-long` (0.003 against 0.035 to
  0.049 at 4096 sweeps). It qualifies at 1024 with no margin (seed-mean MAE
  up to 0.0497).
- **Integrity.** Completion shows `planar_16_offset_ferro_complete`,
  18 targets, 540 cells, 108 decisions, 18 references recomputed. The
  reference, kernel and graph-rebuild checks all pass. Replay took 183 s.
- **Run.** 62 minutes wall on a local CPU with two workers, one attempt, no
  resume. The runner was at commit 3963ecd with a clean tree.

Evidence classes:

- Sweeps: `software_simulation` (CPU, JAX float32).
- References: `exact_reference` (Kac-Ward, float64).
- Z1 figures: `calibrated_projection` (host latency excluded).

There is no hardware claim. The graph is a synthetic patch of the published
rule, not the chip map.

Details: [findings.md](findings.md). Figure: `error-versus-sweeps.png`.
