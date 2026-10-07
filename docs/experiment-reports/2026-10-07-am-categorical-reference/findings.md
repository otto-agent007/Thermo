# Label-first updating does not rescue the sampled categorical memory

Stage A's sampled categorical reference recalled far less within a sweep budget
than at equilibrium. Its findings blamed the update order: visible bits first,
from a random label. **That explanation was wrong.** Updating the label first,
from exactly the same start states, helps only at the shortest budgets:

| Budget K | Label-first minus visible-first, held-out recall, across the six cells |
| --- | --- |
| 4 | +0.010 to +0.132 |
| 16 | +0.002 to +0.036 |
| 64 | −0.007 to +0.014 |
| 256 | +0.003 to +0.043 (above +0.01 only at P = 8 with an 8-bit cue) |

At 256 sweeps the label-first reference still recalls 0.64 to 0.94, against an
equilibrium of 0.88 to 1.00.

The cause is a temperature trade-off that does not depend on update order.
Only β = 16 reaches the categorical memory's equilibrium recall, and at β = 16
block Gibbs from a uniform start barely moves. At P = 32 with an 8-bit cue,
development recall after 256 sweeps is 0.67, against an equilibrium of 0.97.
At β = 4 the chains mix fully (0.75 against 0.75), but the equilibrium itself
is low. Development selection therefore picks β = 4 or 8 at every budget beyond
the shortest. At P = 128 with an 8-bit cue, only 35% of chains sit on the
target's label after 256 sweeps.

The binary bias design does not have this problem. It keeps β = 16 (sometimes
8) at most budgets and still mixes. Against the label-first reference it is
ahead by 0.05 to 0.15 at 256 sweeps in all six cells. It exceeds the reference
in 16 of 24 cell–budget pairs, matches it in 2 (both at K = 4) and is
inconclusive in 6; it never falls short. Stage A's finite-budget ranking
therefore stands: within a sweep budget, binary hidden units with a negative
bias recall better than THRML's sampled categorical unit, whichever order that
unit is updated in.

The [frozen amendment](../../experiments/am-categorical-reference.md) changed
only the reference's block order. It kept stage A's cells, β grid, budgets,
seeds, chains, unit keys, and per-chain start states value for value. The
binary arms were read from the stage A archive, authenticated by its SHA-256.
Sampling is `software_simulation` (THRML 0.1.4, CPU, float32). Exact recall is
`exact_reference` (float64 enumeration). Nothing here is hardware evidence.

## Integrity

- 72 development and 144 held-out units ran, and every exact reference
  recomputed.
- The label-first preflight matched the exact one-sweep law (TV 0.006 at one and
  two sweeps). The K = 64 prefix check passed.
- The reference's equilibrium recall equals stage A's to 1e-12, since it does
  not depend on order.
- The run took 212 s from commit `7921247` on a clean tree, and replayed
  before `completion.json` was written.

## Verdicts against the label-first reference (stage A's verdicts in brackets)

| Arm | Exceeds | Matches | Inconclusive | Falls short |
| --- | --- | --- | --- | --- |
| Bias | 16 (22) | 2 (0) | 6 (2) | 0 (0) |
| One-hot | 9 (12) | 1 (0) | 13 (12) | 1 (0) |
| Domain-wall | 0 (0) | 0 (1) | 2 (1) | 22 (22) |
| Hopfield | 0 (2) | 0 (0) | 11 (10) | 13 (12) |

Each cell–budget table, with paired 95% intervals, is in `summary.md`.

## What this changes

- **Corrected:** stage A's findings said the slow reference came from its
  update order. A dated note there now points here.
- **Strengthened:** the bias design's finite-budget lead is not an artefact of
  a mis-ordered reference.
- **Open, and not this study's to settle:** whether a categorical unit started
  from the cue would close the gap, for example with its label set to the
  pattern that best matches the cue. That start uses information the binary
  arms do not get. It would be a third reference variant, so it waits for the
  owner (see the CLAUDE.md rule on consecutive failures of the same kind).
- **Hypothesis, untested:** the bias arm may mix because its hidden units start
  off and switch on independently. Several candidate patterns can then be
  active at once, instead of one label being committed and having to cross a
  barrier at β = 16.

## Files

- `study.json.gz` (24 KiB, SHA-256 in `completion.json`): request, preflight,
  development and held-out units, with exact recall, per-target recall counts
  and on-target-label counts by budget, plus the evaluation.
- `summary.md`, `completion.json`, `provenance.json`, `run.log`.
