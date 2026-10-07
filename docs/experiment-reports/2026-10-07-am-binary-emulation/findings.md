# Binary hidden units with a negative bias stand in for a categorical memory unit

On pairwise binary units, the simplest construction recovers almost all of a
dense associative memory's recall: P binary hidden units with a negative bias of
0.6 to 0.8 a_max and **no inhibition at all**. At equilibrium its held-out recall
is within 0.000 to 0.009 of the categorical memory in all six task cells, at a
coupling dynamic range of 14 to 26, with two Gibbs blocks per sweep. It is also
the strongest binary design within sweep budgets in most cells: at 256 sweeps
it recalls 0.97 to 0.985 of the missing bits with 12-bit cues.

One-hot inhibition and domain-wall chains reproduce the categorical equilibrium
exactly once their penalty is large enough. In practice both fall short:

- **One-hot** needs a coupling range above 128 (1,210 at P = 128) and P + 1
  sequential blocks per sweep. The settings it chose within a budget barely
  operate as one-hot: 1–3% of samples at P = 128 have exactly one unit on.
- **Domain-wall** needs a range of only about 16, but it does not mix: held-out
  recall at 256 sweeps is 0.58 to 0.76, and its chains keep 2.6 to 41 domain walls.

**The primary comparison against the sampled categorical reference is
confounded, and its verdicts should not be read as written.** The reference
updates its visible bits first, from a uniformly random label. The visible bits
then copy the wrong pattern and lock in with it, so the reference's own sampled
recall at 256 sweeps (0.64 to 0.93) sits well below its equilibrium (0.88 to
1.00). The protocol fixed that update order and start without flagging the
effect. Against such a reference, "exceeds" says little.

The [frozen protocol](../../experiments/am-binary-emulation.md) fixed five arms,
their grids, six (P, cue) cells, budgets K ∈ {4, 16, 64, 256}, development
pattern sets for choosing parameters and 12 held-out sets for comparing them.
All 936 development, 624 held-out sampled and 2,808 held-out exact units
completed and replayed. The preflight showed that one THRML sweep matches the
exact one-sweep law for every arm, and the prefix checks passed. Sampling is
`software_simulation` (THRML 0.1.4, CPU, float32). References are
`exact_reference` (float64 enumeration of the missing bits, with each hidden
layer summed exactly). Nothing here is hardware evidence.

## Equilibrium (held-out exact recall of the development-best configuration)

| Cell (P, cue bits) | Categorical | One-hot | Domain-wall | Bias | Hopfield |
| --- | --- | --- | --- | --- | --- |
| 8, 12 | 0.999 | 0.999 | 0.999 | 0.999 | 0.896 |
| 8, 8 | 0.998 | 0.998 | 0.998 | 0.998 | 0.754 |
| 32, 12 | 0.992 | 0.992 | 0.992 | 0.991 | 0.648 |
| 32, 8 | 0.975 | 0.975 | 0.975 | 0.966 | 0.583 |
| 128, 12 | 0.993 | 0.993 | 0.993 | 0.992 | 0.615 |
| 128, 8 | 0.877 | 0.877 | 0.877 | 0.874 | 0.596 |

The best equilibrium configurations used β = 16. For one-hot and domain-wall
that meant penalty 1.6 a_max (domain-wall chose 0.4 at P = 32); for bias, θ =
0.6 to 0.8 a_max. Hopfield's capacity limit shows clearly from P = 32 on.

## Within a sweep budget (held-out sampled recall)

| Cell | Categorical (confounded) | One-hot | Domain-wall | Bias | Hopfield |
| --- | --- | --- | --- | --- | --- |
| 8, 12 · K=16 | 0.863 | 0.927 | 0.712 | **0.972** | 0.867 |
| 8, 12 · K=256 | 0.931 | 0.949 | 0.757 | **0.985** | 0.885 |
| 8, 8 · K=256 | 0.791 | 0.840 | 0.681 | **0.943** | 0.767 |
| 32, 12 · K=16 | 0.829 | 0.873 | 0.603 | **0.950** | 0.667 |
| 32, 12 · K=256 | 0.923 | 0.890 | 0.618 | **0.982** | 0.667 |
| 32, 8 · K=16 | 0.704 | **0.764** | 0.573 | 0.738 | 0.592 |
| 32, 8 · K=256 | 0.749 | 0.799 | 0.594 | **0.899** | 0.607 |
| 128, 12 · K=16 | 0.762 | 0.826 | 0.566 | **0.842** | 0.606 |
| 128, 12 · K=256 | 0.884 | 0.850 | 0.588 | **0.972** | 0.609 |
| 128, 8 · K=16 | 0.594 | **0.705** | 0.550 | 0.618 | 0.573 |
| 128, 8 · K=256 | 0.637 | **0.723** | 0.579 | 0.711 | 0.576 |

Every budget and its bootstrap intervals are in `summary.md`. Among the binary
designs, bias is best in 8 of the 11 rows shown. One-hot leads in the other
three, all with 8-bit cues (P = 32 at K = 16, and P = 128 at K = 16 and 256),
by 0.01 to 0.09.

## Cost

| Design | Hidden units | Couplings | Dynamic range (chosen config) | Blocks per sweep |
| --- | --- | --- | --- | --- |
| Categorical | 1 multi-state | N · P (native) | 1 | 2 |
| One-hot | P | N · P + P(P − 1)/2 | 58 (P = 8) to 1,210 (P = 128) | P + 1 |
| Domain-wall | P − 1 | N(P − 1) + chain | 2 to 10 | 3 |
| Bias | P | N · P | 14 to 26 | 2 |
| Hopfield | 0 | N(N − 1)/2 | 4 to 17 | N − cue |

Dynamic range is the largest |coupling or field| over the smallest nonzero
two-body coupling, in the ±1 spin form THRML samples. Sweeps and blocks are
algorithmic counts, not device operations.

## Range sensitivity

Held-out exact equilibrium recall of the best development configuration whose
dynamic range is at most R:

- **Domain-wall** reaches the categorical value from R = 16 in every cell.
- **Bias** reaches within 0.01 of it from R = 16 (P = 8, 32) or R = 32 (P = 128).
- **One-hot** stays at 0.71 to 0.96 up to R = 128 at P = 8 and 32. At P = 128
  no one-hot configuration has a range of 128 or less. It reaches the
  categorical value only above R = 128.
- **Hopfield** is flat at its capacity-limited value once its own range (4 to
  17) is allowed.

On hardware with a coupling range of a few tens, the bias design is the only
binary emulation that is both accurate and fast. The finite-budget curves, from
development data only, are in the archive.

## Why the categorical reference is slow (post-hoc exploration, not evidence)

On a probe pattern set (P = 8, 12-bit cue, β = 16), the study's update order
puts 54% of chains on the target label after one sweep and 87% after 256, for
recall 0.94. Updating the label first instead puts 91% on the target after one
sweep and gives recall 0.96 immediately. The slow reference therefore comes
from the protocol's update order combined with its random-label start, not from
categorical memories as such. The binary arms start with every hidden unit off,
which avoids the lock-in. A fair finite-budget reference needs a
label-first order or a cue-informed start. That is a small follow-up, not
something to infer from this run.

## What this does and does not settle

- **Settled for this setting:** pairwise binary hardware can hold a dense
  associative memory without one-hot machinery. A Bonnaire-style negative bias
  on ordinary binary hidden units is enough at N = 24 for P up to 128 (5.3 N),
  at modest coupling range and two blocks per sweep. Inhibition-based one-hot is
  expensive in range and in sequential steps. Domain-wall's exactness does not
  translate into mixing under Gibbs sampling from a random chain.
- **Not settled:** whether any binary design matches a native categorical unit
  within a sweep budget, because the reference is confounded. Behaviour beyond
  N = 24, correlated patterns, quantized couplings, real hardware ranges, and
  any speed or energy claim are also open.

## Disclosures

- The study ran from commit `8fdf416` on a clean tree; the runner and tests
  were committed before it started. It took 8,093 s, including a 44-minute
  single-process full replay of every exact reference. Parallelizing that
  replay would help next time.
- The update-order confound was found after the run, from the reference's own
  sampled recall. The trapping check above ran afterwards on probe-only
  patterns and changed nothing in the archive.
- The CI replay recomputes exact references for the 12-bit-cue cells and checks
  all evaluation numerically. The full replay, `--full`, is a local gate.

## Files

- `study.json.gz` (0.21 MiB, SHA-256
  `c83ee9dcd33c0a698ed03f60913f869ffa9088355902c7d1f07bde1fbd31b26f`): request, preflight, every development and held-out
  unit's exact recall, dynamic range, per-target recall counts by budget and
  diagnostics, the evaluation, and the result digest.
- `summary.md`: per-cell budget tables with bootstrap intervals and verdicts,
  the cost table and the range curves.
- `preflight.json`, `completion.json`, `provenance.json`, `run.log`.
