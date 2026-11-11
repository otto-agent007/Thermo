# Domain-wall mixing under an annealed penalty: probe

2026-11-11. **Exploration, not a recorded study and not evidence.** The owner
asked for this probe after the
[codon-optimization card](../knowledge/sources/extropic-codon-optimization.md)
(backlog open question 7). Sampled recalls are `software_simulation` (THRML
0.1.4, CPU, float32); equilibrium recalls are `exact_reference`. No hardware
claim.

**Finding: chain length decides it.** Ramping the chain penalty, and with it
β, during the run fixes domain-wall mixing on short chains but not on long
ones. On P = 6 and 8 (5 and 7 chain spins), the ramp leaves about one wall per
chain (91 to 100 percent of chains valid at 256 sweeps). It lifts held-out
recall over stage A's fixed penalty by 0.14 to 0.16 at 256 sweeps, which
reaches the sampled categorical reference. On P = 32 and 128 (31 and 127
spins), chains keep 2 to 8 walls at 256 sweeps and the gain is 0.01 to 0.07.
Either way the bias design stays ahead in 30 of 32 cell-budgets, so stage A's
recommendation for associative memory stands. The codon paper's working
domain-wall runs fit this picture: its chains have at most 5 spins.

## Fixed choices

All were fixed in [`probe.py`](am_domainwall_anneal_probe/probe.py) and
committed as `aeca1d1` before the full run. A two-unit timing smoke test ran
before that commit; it showed one unit's numbers and changed nothing.

| Choice | Value | Can it bound the metric? |
| --- | --- | --- |
| Model, patterns, cues, start, chains, sets, budgets | Stage A's, imported unedited from `thermo_lab.am_binary_emulation` (SHA-256 checked against the archive) | Same caveats as stage A: N = 24 random patterns |
| Cells | Stage A's P ∈ {8, 32, 128} × cue ∈ {12, 8}, plus P = 6 (five chain spins, the codon paper's maximum) | Yes: chain length is the variable under test |
| Ramp configurations | β_final ∈ {8, 16} × {penalty only; penalty and β}; penalty fraction log-spaced 0.1 → 1.6 of a_max; β from β_final / 8 | Could. A slower ramp, more stages or a hotter start are untested |
| Stages | S = min(K, 8) stages of K / S sweeps; each budget runs its own schedule | Could. Eight stages may be too coarse for long chains |
| Selection and comparison | Development-best ramp per cell and budget; paired by held-out set against stage A's development-best fixed domain-wall, bias and categorical arms; bootstrap, margin 0.02 | P = 6 baselines come from stage A's own pipeline run here, not from the archive |

## Controls

All three passed before any result was read ([`probe.json`](am_domainwall_anneal_probe/probe.json),
SHA-256 `7eb6e5901cac24b1e45416f3efe8d927a587ff7f492c1719ece1da740a802ea3`).
The run took 796 s on 4 workers.

1. Stage A's `run_unit` reproduces the archived held unit
   `held/domainwall/b8/f0.1/P8/c12/s7100` exactly.
2. The staged runner with a constant schedule reproduces stage A's `sample`
   bitwise at K = 256.
3. The recomputed ramp-endpoint exact recalls equal the archived held-out exact
   recalls (largest difference 4.8e-14).

## Result

Held-out recall at 256 sweeps. In the verdict columns, ↑ exceeds, = matches,
? inconclusive and ↓ falls short, each for ramp minus the named arm. Ramp
walls are per chain; valid is the fraction of chains with exactly one wall.

| Cell | Chain spins | Ramp | Fixed DW | Ramp vs fixed DW | Ramp walls (fixed DW) | Valid | Bias | vs bias | Categorical | vs categorical |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| P6/c12 | 5 | 0.966 | 0.824 | ↑ +0.142 | 1.05 (2.34) | 0.97 | 0.995 | ? −0.029 | 0.953 | ? +0.013 |
| P6/c8 | 5 | 0.895 | 0.740 | ↑ +0.154 | 1.00 (2.79) | 1.00 | 0.949 | ↓ −0.054 | 0.821 | ↑ +0.074 |
| P8/c12 | 7 | 0.913 | 0.757 | ↑ +0.156 | 1.16 (3.40) | 0.92 | 0.985 | ↓ −0.072 | 0.931 | ? −0.018 |
| P8/c8 | 7 | 0.840 | 0.681 | ↑ +0.158 | 1.18 (3.88) | 0.91 | 0.943 | ↓ −0.103 | 0.791 | ↑ +0.048 |
| P32/c12 | 31 | 0.685 | 0.618 | ↑ +0.066 | 2.16 (10.93) | 0.46 | 0.982 | ↓ −0.297 | 0.923 | ↓ −0.238 |
| P32/c8 | 31 | 0.639 | 0.594 | ↑ +0.045 | 2.07 (2.61) | 0.50 | 0.899 | ↓ −0.260 | 0.749 | ↓ −0.110 |
| P128/c12 | 127 | 0.633 | 0.588 | ↑ +0.045 | 7.79 (9.11) | 0.00 | 0.972 | ↓ −0.339 | 0.884 | ↓ −0.251 |
| P128/c8 | 127 | 0.594 | 0.579 | ? +0.014 | 7.77 (40.83) | 0.00 | 0.711 | ↓ −0.118 | 0.637 | ↓ −0.044 |

Across all 32 cell-budgets (K = 4, 16, 64 and 256):

- **Against the fixed penalty.** From K = 16 on, the ramp exceeds it in all 12
  short-chain cell-budgets, and in 6 of 12 long-chain ones (the other 6 are
  inconclusive).
- **Against the bias design.** The ramp falls short in 15 of 16 short-chain
  and 15 of 16 long-chain cell-budgets. The two exceptions are inconclusive:
  P6/c12 at K = 256 and P128/c8 at K = 4.
- **Against the sampled categorical reference.** On short chains, the ramp is
  matched, exceeds or is inconclusive in 13 of 16, and falls short in 3
  (P8/c12 at K = 4 and 16, P8/c8 at K = 16). On long chains it falls short
  in 13 of 16.
- Development selection chose the penalty-and-β schedule in 30 of 32
  cell-budgets.
- Not the cause: the endpoint's equilibrium. For long chains selection
  usually chose β_final = 8, whose own exact equilibrium is lower (0.955 at
  P32/c12, 0.804 at P128/c8). Recall at 256 sweeps (0.59 to 0.69) is far
  below those values, so mixing is the limit.

## Mechanism

A domain wall moves freely, but two walls annihilate only when they meet.
Starting from a random chain, a long chain begins with many walls that must
random-walk together. A low early penalty makes extra walls cheap to create
and destroy, so short chains settle to one wall before the penalty rises. On
31 or 127 spins, eight stages of at most 32 sweeps are too few for the walls
to find each other. That is consistent with the wall counts above; it is this
note's reading, not a separately tested mechanism.

## Resource accounting

The comparison is at equal sweeps. Each schedule stage changes the weights:
- **Penalty-only ramp:** changes the chain couplings and the end fields.
- **Penalty-and-β ramp** (chosen almost everywhere): changes every coupling
  and field.

So a K ≥ 8 run rewrites the weights 7 more times than a fixed-penalty run.
Under the sealed Z1 profile (`docs/z1-hardware-model.md`), one full node write
costs as much as 21,665 Gibbs updates. For P8/c12 with the penalty-and-β ramp,
that is 7 × 31 node writes (about 33 nJ). Against those, 256 sweeps of 19
updates come to about 0.03 nJ, so the writes cost about 1,000 times the
sampling. That is this note's arithmetic (analysis). On projected energy, the
ramp is strictly worse than a fixed-penalty or bias run, which load once.

## Reading

1. **Open question 7, answered for this model.** Domain-wall encoding mixes
   under an annealed penalty when chains are short (5 to 7 spins), and does
   not when they are long (31 to 127). Stage A's negative was a long-chain
   result plus a fixed schedule. Its "does not mix" stands for P ≥ 32 and
   should be scoped to fixed penalties for P ≤ 8.
2. **The codon paper is consistent with this.** Its chains have at most 5
   spins and its penalty and β are annealed, the regime where this probe
   finds mixing.
3. **No change to the associative-memory recommendation.** The bias design
   stays ahead, needs no reflash, and its I/O advantage grows under the
   sealed profile.
4. **Where domain-wall is worth using:** categorical variables with few
   labels (codons, small-q Potts), where it costs P − 1 spins and two colours,
   and when the budget allows reflashes. Not for a memory with tens of labels
   or more.

## Not done

- No slower ramp, more stages, label-ordered or cue-informed start, or
  replica exchange for long chains.
- No Potts-target check.
- No protocol: this is a probe, and a recorded study would need the owner.

No source file or archive was edited.
