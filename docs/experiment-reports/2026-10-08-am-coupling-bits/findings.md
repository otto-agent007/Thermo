# Associative-memory coupling bits: program fields relative to J; ±10% beta jitter is negligible

How many codebook bits does stage A's `bias` binary memory need to keep its
exact equilibrium recall within 0.01 of the unquantized design? The answer
depends on where the codebook's cap is placed:

| Cap | Spec number b* (max over the six cells) | Pre-registered expectation | Outcome |
| --- | --- | --- | --- |
| `coupling` (step = J) | **6 bits** (5 at P = 8 and at P = 32 with an 8-bit cue) | 6, and 5 at P = 8 | Met |
| `split` (separate coupling and field caps) | **10 bits** (driven by P = 128, 8-bit cue; 4 to 7 elsewhere) | at most 7 | **Missed** |
| `full` (one cap = largest field) | **12 bits** (10 at P = 32, 12 at P = 128, 5 at P = 8) | 8, and 10 to 12 at P = 128 with an 8-bit cue | **Missed** at P = 32 and P = 128 with a 12-bit cue |

b* is the smallest tested width at which, at it and every wider tested width,
the lower end of the paired 95% interval of (quantized − unquantized) exact
equilibrium recall stays at or above −0.01 (16 held-out pattern sets, three
targets each). No cap needs more than 12 bits.

The `split` miss is an interval-width result, not a mean shift. At P = 128
with an 8-bit cue, `split` at 8 bits has mean difference +0.0013 but interval
[−0.0152, +0.0184]. Sixteen sets cannot confirm 8 bits there, so the rule
returns 10. The `full` miss is real loss. At P = 128, `full` at 8 bits loses
0.10 (both cues), and at 10 bits the 12-bit cue still reaches −0.0195 at the
lower end. A shared full-scale cap spends its levels on the largest visible
field (up to 28 J at P = 128), so J gets few steps. The protocol's negative
reading applies: program this design's fields in units of J, not relative to
the largest field. Recall is not monotone in b under `full` and `split`
(figure). At P = 32 with an 8-bit cue, `full` reaches +0.011 at 7 bits and
falls to −0.011 at 8.

Under `coupling`, recall is flat from 6 to 12 bits at a small fixed offset
(−0.0035 to +0.0014). From 6 bits upward the field cap L·J no longer clips,
so the only remaining error is rounding the hidden field −θNJ to a multiple
of J (14.4 J to 14 J at θ = 0.6; 19.2 J to 19 J at θ = 0.8). That offset is
inside the tolerance. 5 bits (L = 15) clips the larger fields at P ≥ 32
and loses up to 0.17.

## Beta jitter

Static ±10% per-site gain jitter is **negligible in all six cells**. All 240
paired sampled intervals (every arm, budget and cell) lie within ±0.01. The
mean shifts range from −0.0022 to +0.0012, inside the pre-registered 0.005.
The ±30% stress level on the unquantized design stays within −0.0022 to
+0.0019 (expected within 0.015). The exact chain law at P = 8 with a 12-bit
cue agrees. The mean effect of ±10% is at most 0.001 at every budget and at
stationarity, and the worst single unit is −0.0049. At ±30% the worst unit is
−0.0081. Gain uniformity is not a hardware requirement for this design at
these levels. Per-sweep beta noise and coupling-value noise were not tested.

## Within a sweep budget (sampled, ±0.02 margin)

At K = 256:

- `coupling` at 6 and 8 bits **holds** in all six cells.
- `split` at 8 bits holds in 5 of 6 cells. At P = 128 with an 8-bit cue it is
  inconclusive (+0.0063 [−0.0086, +0.0214]); the upper end exceeds the
  margin.
- `full` at 8 bits **falls short** at P = 128 with a 12-bit cue (−0.146),
  P = 128 with an 8-bit cue (−0.060) and P = 32 with a 12-bit cue (−0.084), as
  pre-registered (0.06 to 0.15). It is inconclusive at P = 32 with an 8-bit
  cue.
- 4 bits falls short under every cap at P ≥ 32, except `split` at P = 32 with
  an 8-bit cue (inconclusive, −0.022 [−0.040, −0.005]).
- `full` at 4 bits recalls 0.53 to 0.57 at P ≥ 32, the disconnected-memory
  floor (pre-registered).

At short budgets, quantized arms often recall more than the control. Examples
are `coupling` at 6 bits, +0.05 at K = 16 at P = 8 with an 8-bit cue, and
`coupling` at 4 bits, +0.10 at K = 4 at P = 8 with a 12-bit cue. That puts 38
verdicts in inconclusive with the whole interval above +0.02. Rounded fields
change how fast the chain mixes. These are not recall gains at equilibrium.

Verdict counts over 480 arm × cell × budget comparisons: 218 hold, 96 fall
short, 166 inconclusive.

## Integrity and protocol notes

- Preflight passed: gain-aware THRML matched the exact one- and two-sweep
  laws (TV 0.0088 and 0.0076 against tolerances 0.0114 and 0.0108). The
  wrong-gain control was rejected (TV 0.033 and 0.051). Stock equality passed
  bitwise.
- The unquantized exact equilibrium matches stage A's function to
  1.6e-13. Every unjittered chain's stationary law matches its exact
  equilibrium to 3.4e-13. All 512 chain laws converged.
- Prefix check and full replay passed. The stage A archive was authenticated
  by SHA-256 (`c83ee9dc…`).
- **Run count.** The protocol's arm table gives 32 sampled runs per set and
  cell: unquantized at j = 0, 0.1, 0.3 with 1, 2 and 2 gain draws, plus 27
  quantized. Its text says 31, and its gate says `sampled_runs=2976`. Every
  arm in the table was run, so `sampled_runs=3072`. No decision rule depends
  on the count. `completion.json` states the discrepancy.
- **Stationary chain law.** Plain power iteration cannot reach an L1 change
  of 1e-13 on these chains, because some targets have a second eigenvalue
  within 1e-5 of 1. The runner squares the transition matrix to take doubling
  strides and stops at a stride of at least 2^24 sweeps with an L1 change
  below 1e-13. It checks the result against the exact equilibrium of every
  unjittered model, as above. The protocol names this quantity; only the
  method differs.
- Run: two attempts, both in `attempts.jsonl`. The first was stopped during
  calibration, before any unit was saved, so that provenance would record a
  clean tree. The second ran with `--resume` in the same directory. The final
  attempt took 2,842 s wall on 3 workers, each pinned to one core. Sampling
  used 6,503 CPU-seconds. Calibration projected 1.83 CPU-hours against a
  4-hour stop.

## Evidence classes and limits

- Exact equilibrium recall and the P = 8, 12-bit-cue chain laws are float64
  enumerations: `exact_reference`.
- All finite-budget results are THRML 0.1.4 on CPU in float32:
  `software_simulation`.
- The codebook is a mathematical rounding rule, and the gains are a static
  noise model. Neither is a Z1 encoding, a DAC or a measured device property.
- Nothing here is hardware evidence.
- The configurations (β = 16; θ = 0.6 or 0.8) are stage A's unquantized
  choices and were not re-selected per bit width.
- N = 24 throughout. Spec numbers are in units of the field-to-coupling
  ratio, which grows like √P, so they do not carry over to other sizes
  without that conversion.

[Summary tables](summary.md), [figure](recall-versus-bits.png),
[protocol](../../experiments/am-coupling-bits.md).
