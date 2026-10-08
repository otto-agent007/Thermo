# Exchange cost projection: exploratory protocol

Frozen on 2026-10-07 before production sampling. The recorded
symmetry-plus-tempering studies recommend tempering with the analytic flip
estimator because it qualifies on more targets at smaller sweep budgets. Those
budgets count redraws. On the Z1 cost model a replica exchange is not a redraw:
it is host I/O priced per p-bit read and per full SRAM write, and the Appendix-B
constants put one node write at about 21,665 node updates. This protocol asks
whether the recommendation survives that pricing, and where the trade-off sits
when exchanges are attempted less often.

## Fixed choices and what they bound

Listed first, as `CLAUDE.md` asks. Each one bounds the metric in a known way.

- **Cost model.** The sealed `Z1HardwareProfile` (`src/thermo_lab/hardware/z1.py`):
  7.09 fJ per Gibbs node update, 1.692 pJ per node read, 153.6 pJ per full
  node SRAM write, 50 MHz maximum complete-sweep rate. Host energy and latency
  are excluded by construction, so sweep time is a lower bound that cannot see
  the latency an exchange adds. Energy is the dimension the model can price.
- **Exchange conventions.** Two are priced side by side for every cell:
  - `published`: an attempt reads every p-bit of both replicas in a pair (the
    host needs both energies); an accepted exchange writes both replicas'
    states back. Nothing outside the published profile.
  - `beta_knob`: the same reads, but an accepted exchange swaps the pair's
    inverse temperatures through a hypothetical per-replica control at zero
    node-write cost. The profile has no such control; this bounds how much
    one could help. An on-chip energy register is not modelled.
  Under both, a 50%-accepted exchange step on 16 spins costs on the order of
  10^3 (`beta_knob`) to 10^3.5 (`published`) sweeps of sampling. Exchanging
  every sweep is therefore dominated by I/O at any budget in this study; the
  exchange interval is the only knob that can move the ratio, and the largest
  interval (256) cannot reach parity. This is known before sampling.
- **Sweep convention.** One systematic sequential Gibbs sweep over `n` sites is
  one elapsed complete sweep with `n` node updates. The Appendix-B sweep is
  two-colour; the targets are cubic graphs that are not bipartite, so this is
  an update-count equivalence, not a schedule equivalence.
- **Placement.** One logical p-bit per physical p-bit, no embedding overhead,
  `5n` physical p-bits for five replicas. Each replica of a tempering or
  independent arm runs in parallel, so its elapsed sweeps are `T`; the long
  arm runs `5T` on `n` p-bits.
- **Targets, ladder, estimator, qualification.** Inherited unchanged from the
  archived study (`docs/experiments/symmetry-tempering.md`): six zero-field
  weighted cubic graphs, ladder [0.25, 0.5, 1, 2, 4], cold β = 4, quarter
  burn-in, analytic global-flip estimator, both mean joint TV and edge MAE
  ≤ 0.05 sustained at all larger tested budgets.

## Stage A: price the archive

Authenticate `docs/experiment-reports/2026-10-02-symmetry-tempering/evidence.tar.gz`
by its manifest, then price all 120 cells and 24 decisions from the archived
exchange flags under both conventions. No sample is drawn. The archived
qualification decisions must be reproduced exactly. Report, per target and
convention, the cheapest qualifying symmetry-aware ordinary baseline, the
tempering-flip energy ratio against it, and the sweep-time ratio against the
fastest qualifying ordinary baseline. Censor failed thresholds at the largest
tested budget.

## Stage B: fresh-seed exchange-interval sweep

Seven arms: `long-flip`, `independent-flip`, and `tempering-flip-k{1,4,16,64,256}`,
where `k` is the number of sweeps between exchange attempts. Pairs (0,1)/(2,3)
and (1,2)/(3,4) still alternate, by exchange index. The study-local sampler
must reproduce the archived sampler bit for bit at `k = 1` (unit test), and
every sweep still splits an exchange key so the sampling stream does not depend
on `k`. Fresh JAX root 20261008, 16 trials, budgets T = 16, 64, 256, 1024, 4096,
one compile per arm/size at the horizon, with a prefix check at T = 16.
Initialization is shared across arms; the original method index 0/1/2 is
folded into the sampling key as before. The `k = 1` arm is a fresh-seed
replication of the archived tempering-flip arm.

Per cell: the inherited accuracy metrics, per-trial pass fraction, and per-trial
Z1 operation counts (node updates, reads, writes, exchange attempts, host round
trips) with projected energy under both conventions and sweep time at the
assumed maximum clock. Per (target, arm): the qualification decision with its
energy and sweep time. Per target: the energy and sweep-time ratios of each
qualifying tempering arm against the ordinary baselines, and the set of
qualifying arms not dominated on (sweep time, published energy).

## Integrity, scope and artifacts

Persist the cold-replica traces of tempering arms, all replicas of ordinary
arms, and every exchange flag, packed. Replay authenticates sources, request,
traces and the stage-A archive, then recomputes references, estimates, work
counts, pricing, decisions, comparisons and the frontier, using the inherited
fixed atol = 2e-12. Write completion last.

Traces are `software_simulation`; references are `exact_reference`; every
energy and time figure is a `calibrated_projection` with the profile's listed
exclusions and carries no hardware measurement. Expected execution is minutes
on CPU; restart an interrupted run in a fresh directory.
