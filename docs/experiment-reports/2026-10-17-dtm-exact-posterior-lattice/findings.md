# DTM exact-posterior lattice: findings

**Negative: stop (b) fires on `t4_h05_q035`, and the frozen rule 7 names
the mechanism finite-w representation.** No priced start on the operating
arm (8 bits `split`, 10 percent jitter) reaches Bayes + 0.01 within K = 1024.
The oracle pair agrees at K = 4096 on a common gap of 0.0117, above the 0.01
tolerance. There is no operating point, so a reader has none to use. The
best priced cell (random start, K = 1024, pooled gap 0.0166) does not reach.
`smooth8_h05_q035` did not run, so the cost stop (c) is undecided.

Protocol: [docs/experiments/dtm-exact-posterior-lattice.md](../../experiments/dtm-exact-posterior-lattice.md),
approved at d4f8b2a (proposal P-0008, PR #126). Runner:
`thermo_lab.dtm_exact_posterior_lattice`, run from b8aba65 with a clean tree.

## What ran

| Stage | Content | Result |
| --- | --- | --- |
| Preflight | layouts, tie limit, sampler against exact kernels, pricing, oracle, oracle start | all passed (sampler max \|z\| 3.41 over pooled bins; tie limit 1.8e-15; oracle 7.0e-15) |
| Calibration | full-length cells, both families | 8.22 CPU-hours projected (2.82 for `t4`), under the 10-hour stop |
| E-dev (exact) | 2 layouts x 6 tie values, 300 development images | every w admitted on both layouts; index 4 carried |
| D (sampled) | 6 development cells | w\* = 1.75 (development gap 0.0197, the smallest of 0.0197 to 0.0392) |
| E-test (exact) | 8 rows at w\*, 1,000 test images | stop (a) does not fire |
| S (sampled) | 81 cells, float K = 4096 reference, oracle pair | stop (b) fires |
| `smooth8` | not run | the `t4` stop closes it (stop rules) |

Development gaps here are sampled (stage D, one seed, K = 1024, random
start). They are not the exact tie-limit gaps.

## Exact stages: the placed model can hold the posterior

At w\* on the 1,000 test images (`exact_reference`). Gaps are against the
Bayes MPM error of 0.2234; the noisy input is at 0.3496.

| Model | MPM gap | 64-draw gap |
| --- | --- | --- |
| placed, float | 0.0001 | 0.0029 |
| placed, 8 bits | 0.0003 | 0.0031 |
| placed, 6 bits | 0.0025 | 0.0054 |
| placed, 4 bits | 0.0752 | 0.0768 |
| unplaced, 8 bits | 0.0006 | 0.0030 |
| true posterior, 64-draw floor | — | 0.0028 |

At 8 bits the tie-limit model is within 0.0003 of the true posterior's
64-draw floor. Precision does not bind at 8 or 6 bits; at 4 bits it does.
Stop (a) is judged on the 8-bit MPM gap, 0.0003, and does not fire.

## Sampled stage: no start reaches

Pooled 64-draw gap (mean of three per-seed gaps, 1,500 images), with the
95 percent stream bootstrap (`software_simulation`, NumPy two-colour Gibbs
on CPU):

| Arm | Start | K = 64 | K = 256 | K = 1024 |
| --- | --- | --- | --- | --- |
| 8 bits, jitter 0.1 (operating) | random | 0.0352 | 0.0256 | 0.0166 (0.013 to 0.020) |
| | reset every 5 | 0.0470 | 0.0359 | 0.0197 (0.016 to 0.024) |
| | continuation | 0.0525 | 0.0409 | 0.0209 (0.017 to 0.025) |
| 8 bits, jitter 0 | random | 0.0355 | 0.0260 | 0.0155 (0.012 to 0.019) |
| | reset every 5 | 0.0463 | 0.0359 | 0.0197 |
| | continuation | 0.0545 | 0.0409 | 0.0201 |
| 6 bits, jitter 0 | random | 0.0384 | 0.0292 | 0.0214 |
| | reset every 5 | 0.0502 | 0.0416 | 0.0251 |
| | continuation | 0.0572 | 0.0449 | 0.0262 |

Reference cells at K = 4096 (unpriced):

| Cell | Pooled gap | 95 percent interval | Per seed |
| --- | --- | --- | --- |
| float, random start | 0.0124 | 0.0096 to 0.0156 | 0.0108, 0.0123, 0.0141 |
| oracle pair, oracle start | 0.0112 | 0.0083 to 0.0142 | 0.0085, 0.0120, 0.0131 |
| oracle pair, random start | 0.0121 | 0.0091 to 0.0153 | 0.0099, 0.0123, 0.0142 |

No cell meets rule 3 (pooled gap at most 0.01 and every seed at most
0.015). The carried-state check passes for continuation and reset at
K = 1024 only. At K = 64 and 256 they are more than 0.01 worse than a
random start. Copy agreement at the last sweep is 0.84 to 0.89 in every
cell. This diagnostic does not gate.

## Rule 7: the mechanism

1. **Precision or jitter: no.** No K at which the jitter-0 arm reaches
   while the operating arm does not; neither reaches anywhere. The float
   K = 4096 reference (0.0124) does not reach either, so the "float reaches
   where the oracle pair's random start does not" clause does not apply.
2. **Oracle pair: agrees.** g_o = 0.0112, g_r = 0.0121, d = -0.0009
   (paired interval -0.0029 to 0.0010), so |d| ≤ 0.01.
3. **g\* = 0.0117 > 0.01, so the mechanism is finite-w representation.** Two
   starts that differ in the latent agree at K = 4096 on an equilibrium that
   misses. At w = 1.75, the finite-tie model's own equilibrium sits about
   0.009 above its tie limit (0.0031). Tied copies at a tie strength the
   codebook can express do not represent the posterior within tolerance.

Caveats on the size of this result:

- The margin is small: 0.0017 above the tolerance. Both pair cells'
  pooled intervals include 0.01, and seed 0 of each is below it. The rule
  gates on the point estimate, as frozen. The power analysis puts a true
  gap of +0.012 at a 0.17 chance of reaching, so a cell this close to the
  line is a near-miss, not a wide one.
- The float K = 4096 reference has the same gap (0.0124), so the miss
  does not come from 8-bit rounding or jitter.
- w\* was chosen by sampled development gaps at K = 1024. A larger tie value
  would move the finite-w equilibrium towards the tie limit, but it mixes
  worse: the development gaps rise from 0.0197 at w = 1.75 to 0.039 at
  w = 3.0. The study does not test w outside the frozen grid.

## Pre-registered expectations against outcome

| # | Expectation | Outcome |
| --- | --- | --- |
| 1 | `t4` passes (a); every w admitted; index 4 carried | held |
| 2 | w\* in {1.75, 2, 2.25}; operating arm between +0.004 and +0.015 | w\* = 1.75 held; operating arm at K = 1024 is +0.017 to +0.021, above the range |
| 3 | if (b) fires, the oracle pair disagrees ("mixing beyond 4096") | **missed**: the pair agrees; mechanism is finite-w representation |
| 4 | continuation no better than random in any seed | held: continuation is worse at every K |
| 5 | jitter changes nothing | held: 0.0155 against 0.0166 at random K = 1024 |
| 6 | `smooth8` does not run | held, through the `t4` stop rather than its own stage E stop |

On expectation 3, the probes ran the oracle start only at K = 1024. By
K = 4096 it has forgotten its start. The oracle cell's 0.0112 is far from
its conditioned target of -0.057 and next to the random-start cell, so the
latent does move.

## Resource accounting (standing rules)

Per sample at K = 1024, index 4, 189 free nodes. Reads and writes are the
headline:

| Start | Reads | Writes | Free-node writes | Updates | J per 64-draw estimate, write constant | J, clamp at read |
| --- | --- | --- | --- | --- | --- | --- |
| continuation | 64 | 64 | 0 | 193,536 | 7.24e-7 | 1.02e-7 |
| reset every 5 | 64 | 101.8 | 37.8 | 193,536 | 1.10e-6 | 4.73e-7 |
| random | 64 | 253 | 189 | 193,536 | 2.58e-6 | 1.96e-6 |

Joules are `calibrated_projection` on the sealed Z1 profile. Host transfer
and control are excluded. The archived `t4` rung `cnn2_w32_k5` (1.74e6
MACs, `calibrated_projection`) costs 1.74e-7, 1.74e-6 or 1.74e-5 J per image
at 0.1, 1 or 10 pJ per MAC. Activations, memory traffic and the bias map are
excluded. This is context, not a verdict. `t4` tests representability and
mixing, and stop (c) is defined on `smooth8`, which did not run.

CPU time (software on this box): 4.56 CPU-hours over 99 units. The `t4`
calibration projected 2.82 (1.6x under) and the protocol estimated 2.6.
The oracle-pair and float cells cost 836 to 882 CPU-s each, against a
protocol figure of about 480.

## What this closes

The frozen protocol closes the denoising line at L ≤ 16 for this
construction. The construction is hand-built bias-design template units,
spread as tied copies on the index-4 pixel basis, with relay paths on the
16-offset lattice, an 8-bit `split` codebook and NumPy two-colour Gibbs (not
the lowered THRML program). No other layout or tie strength outside
{1.5, ..., 3.0} was tested. The real binarized set opens only on a positive,
so it stays closed.

Not claimed: no Z1 measurement, latency or device noise; no result from the
lowered THRML program; no lower bound on digital cost; no `smooth8` result;
no cost verdict (c).

## Integrity

`completion.json` was written last by the runner after a full replay
(178 s). The replay re-authenticated sources and recomputed Bayes, both
exact stages, the floors and the conditioned targets at 1e-12. It also
recomputed admission, w\*, reaching, the oracle pair, the mechanism,
bootstraps, prices and verdicts from the archived counts, re-verified
placement, and reran one stage S and one development cell for identical
counts. Gate fields:

- `status=dtm_exact_posterior_lattice_complete`;
- `families_run=1`, `exact_rows=20`;
- `sampled_cells=96`: 6 development cells and 90 stage S and reference
  cells;
- `preflight_passed`, `placement_passed`, `references_recomputed`,
  `prefix_check_passed` and `replayed` all `true`.

The archive `evidence.tar.gz` is 473,408 bytes; its SHA-256 is in
`manifest.json`. It holds `study.json.gz`, the request, the provenance and
the hashed sources, without the per-unit files.

One runner deviation from a literal reading, recorded for review:

- Preflight 3 tests each enumerated state with an expected count of at
  least 5 at |z| < 5. States below that are pooled into one bin.
- Unpooled, the maximum |z| was 12.9. It came from states with expected
  counts far below 1, where the normal z is undefined in practice.
- The engine under test is the production `gibbs` function itself.
