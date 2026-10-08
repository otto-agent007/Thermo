# Associative-memory recall against coupling bits and beta jitter (probe, exploration)

Exploration on 2026-10-08 for queue row `am-coupling-bits` (proposal P-0002).
Script: [`am_coupling_bits_probe.py`](am_coupling_bits_probe.py); raw numbers:
[`2026-10-08-am-coupling-bits-probe.json`](2026-10-08-am-coupling-bits-probe.json).
**Not recorded evidence**: probe-only pattern seeds 9800 to 9803, four pattern
sets per cell, a NumPy sampler instead of THRML, no archive, no replay, no
gate. It exists to find the fixed choice that bounds the metric and to size
the protocol, not to be cited as a result.

## Setup

Stage A's `bias` design ([findings](../experiment-reports/2026-10-07-am-binary-emulation/findings.md)):
N = 24 visible spins, P in {8, 32, 128} binary hidden units with bias
-theta a_max and no inhibition, cues of 12 or 8 bits, beta in {4, 8, 16},
theta in {0, 0.2, 0.4, 0.6, 0.8}. Exact recall enumerates every completion of
the missing bits with the hidden layer summed out, and matches
`am_binary_emulation.exact_recall` to 1e-12 (checked before the run).

In the +-1 spin form THRML samples, the design has only three kinds of number:

- every visible-hidden coupling is +-J, with J = beta / (2 sqrt N);
- every hidden field is b_h = -theta N J (14.4 J at theta 0.6, 19.2 J at 0.8);
- visible field i is b_v,i = J sum_mu xi_mu,i, an integer multiple of J,
  largest 6 to 8 J at P = 8 and 22 to 28 J at P = 128.

So the coupling *magnitudes* are all equal. Its "dynamic range" of 14 to 26
is the fields' size in units of J, not a spread among couplings.

Quantization uses the M5b codebook rule
(`meta_ebm_thermalization_core.round_parameters`): L = 2^(b-1) - 1 levels a
side, step = cap / L, nearest rounding, clip at +-L steps. The rule leaves the
cap open, and stage A has no cap. Three caps were probed:

- `full`: one codebook, cap = the instance's largest |parameter| (a field).
  This is the literal reading of "the M5b codebook" applied to stage A.
- `split`: couplings and fields get separate full-scale codebooks. Couplings
  are then exact, because they share one magnitude.
- `coupling`: step = J, so couplings are exact and fields are rounded
  to multiples of J and clipped at L J.

Beta jitter: a static per-site gain g_i ~ U[1 - j, 1 + j], drawn once per
chain set, so each site samples P(s_i = +1) = sigmoid(2 g_i gamma_i). The
effective couplings become asymmetric, so the chain has no Boltzmann law.
At P = 8 with a 12-bit cue its finite-K and stationary laws were computed
exactly by transition matrices. Everywhere else, 256 chains per target ran in
NumPy with common random numbers. The sampler matched the exact transition
matrices within 0.003 at K = 1, 4 and 16 under 30 percent jitter.

## Finding 1: the codebook cap, not the bit count, sets the answer

Smallest bit width whose exact equilibrium recall stays within 0.01 of the
unquantized design at that and every larger tested width (configuration fixed
at the unquantized best, mean of 4 probe sets x 3 targets):

| Cell (P, cue) | full | split | coupling |
| --- | ---: | ---: | ---: |
| 8, 12 | 5 | 4 | 5 |
| 8, 8 | 5 | 4 | 5 |
| 32, 12 | 8 | 6 | 5 |
| 32, 8 | 8 | 4 | 6 |
| 128, 12 | 8 | 6 | 6 |
| 128, 8 | 12 | 7 | 6 |

Under the `full` cap, 4 bits gives step = max|field| / 7, which is larger
than 2J once the range exceeds 14. Every coupling then rounds to zero, the
hidden layer disconnects, and recall falls to 0.52 to 0.70. That is 0.70
at P = 8 from the rounded visible fields alone, and near chance at P = 128.
At 6 bits, `full` still loses 0.06 to 0.36 at P >= 32 (0.003 at P = 8). It reaches tolerance at 8 bits
(12 at P = 128 with 8-bit cues: 0.800 at 8 bits, 0.888 at 10, against 0.915).
Recall is also not monotone in bits under `full` (P = 32, 12-bit cue:
0.878, 0.855, 0.722, 0.998 at 5, 6, 7, 8 bits). Where J lands on the grid
matters more than the grid's fineness.

With couplings exact, the requirement is a field requirement. `coupling`
needs L >= max|field| / J: 15 (5 bits) covers P = 8, and 31 (6 bits) covers
every cell (fields up to 28 J at P = 128, hidden bias 19.2 J at theta 0.8).
At 6 bits its recall is within 0.002 of unquantized in all six cells; at 5
bits it loses 0.14 at P = 128 (clipping). `split` is not monotone below 7 bits
(P = 32 with 8-bit cues: 0.986, 0.984, 0.987, 0.984 at 4 to 7 bits), because
rounding the hidden bias can help or hurt.

Re-selecting beta and theta per bit width (descriptive, on the same probe
sets) recovers part of the low-bit loss under `split` and `full`. For
example, `split` at 4 bits for P = 128 with a 12-bit cue goes from 0.637 to
0.902. It does not change the 6-bit `coupling` answer.

## Finding 2: plus or minus 10 percent beta jitter does not move recall

Paired change in sampled recall against the unjittered chain with the same
random numbers (16 units per cell = 4 sets x 4 gain draws). Worst cell mean,
then worst single unit, over all six cells:

| Jitter | K = 4 | K = 16 | K = 64 | K = 256 |
| --- | --- | --- | --- | --- |
| +-5% | -0.0006 / -0.002 | -0.0002 / -0.002 | -0.0003 / -0.002 | -0.0005 / -0.002 |
| +-10% | -0.0004 / -0.004 | -0.0003 / -0.002 | -0.0002 / -0.004 | -0.0004 / -0.003 |
| +-20% | -0.0008 / -0.009 | -0.0006 / -0.006 | -0.0006 / -0.008 | -0.0007 / -0.007 |
| +-30% | -0.0014 / -0.010 | -0.0014 / -0.011 | -0.0014 / -0.008 | -0.0012 / -0.006 |

The exact stationary recall at P = 8 with a 12-bit cue moves by at most
0.0007 at +-10% and 0.003 at +-30%. At beta = 16 the local fields are large,
so the conditionals are near 0 or 1 and a 10 percent gain change barely moves
them. Adding +-10% jitter to a quantized chain moves its mean recall by at
most 0.005 in any cell, budget or scheme.

## Finding 3: within a sweep budget the picture is the same, with more spread

Sampled recall after 256 sweeps, change against the unquantized chain (cell
means, then the worst single pattern set):

- `split` at 8 bits: -0.013 to +0.005 (worst set -0.045, P = 128, 8-bit cue).
- `split` at 6 bits: -0.044 to -0.002 (worst set -0.084).
- `full` at 8 bits: -0.148 to -0.065 at P = 128 and at P = 32 with a 12-bit
  cue; within 0.006 elsewhere. Quantization slows mixing as well as moving
  the law.
- `full` at 4 bits: -0.42 to -0.16 at P >= 32. `split` at 4 bits: -0.35 to
  -0.15 at P >= 32, except -0.012 at P = 32 with an 8-bit cue.
- `coupling` was not sampled in the probe.

## What this changes

1. **The fixed choice that bounds the metric is the codebook cap.** "4-, 6-
   and 8-bit couplings from the M5b codebook", read literally (one shared
   full-scale cap), measures where J lands on a grid sized by the largest
   field, and its 4-bit point is a disconnected network. The answer to "how
   many coupling bits" in this design is "sign only". The bits that matter
   are the fields' bits in units of J. A protocol must name the cap, and it
   should report the cap as an arm, not hide it.
2. **Jitter is not a constraint at +-10%.** A study can confirm it on fresh
   held-out sets cheaply, but the expected result is a null within 0.005.
3. **Pre-registered expectation for the spec number:** 6 bits for the fields
   with couplings at step J (L = 31 >= the largest field over J), 8 to 12
   bits under a shared full-scale cap, and at most 7 bits under a split cap.
4. Static gain jitter is one noise model. Per-sweep (dynamic) beta noise and
   noise on the couplings themselves are different questions and were not
   probed.

## Cost and fixed choices of the probe

About 13 minutes of wall time on 4 worker processes (346 s exact, 421 s for
1,128 chain units), about 51 CPU-minutes. A first launch of the same script
was cut off by a tool timeout after 7 minutes and discarded. The probe used
4 probe pattern sets per cell, the unquantized best configuration per cell
from the probe's own exact sweep (it matches stage A's choices: beta 16,
theta 0.6 at P = 8 and theta 0.8 at P >= 32, except theta 0.6 at P = 32 with
an 8-bit cue), hidden start all off, missing bits uniform, visible block
first, and NumPy rather than THRML sampling.
