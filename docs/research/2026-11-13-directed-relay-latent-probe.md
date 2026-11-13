# Directed relay latent: probe (P-0013)

2026-11-13. **Exploration, not a recorded study and not evidence.** Probe
for queue row `directed-relay-latent`, proposal 2 of the
[mechanism proposals](2026-11-12-mechanism-proposals.md). Every number is
the exact law of a small Markov chain, computed on a CPU by evolving its full
probability vector over every state and every observation. No sampling, no
training, no THRML, no hardware. Script, logs and output:
[`directed_relay_latent_probe/`](directed_relay_latent_probe/).

**Finding: no, under the Z1's two-colour schedule.** With the update order
the hardware uses (two colour phases per sweep), the directed construction's
stationary latent error is 0.065 to 0.079, even with ideal noiseless relays.
That holds for every feasible colouring of the relays (24 of 32). The
binding limit is **staleness**, not lossy fan-in and not mixing. A latent
carried by one-way relays reaches the pixels through at least four hops,
so the pixels see a latent conditioned on old pixel values. The chain is then
a stale Gibbs sampler with a biased stationary law. Relay quality cannot fix
that.

Two further results bound the reading:

- **With a sequential update order** (each site updated in turn, fan-in,
  latent, copies, then pixels; a relaxed constraint that the Z1 does not
  offer), the construction does reach 0.01: thermometer fan-in at gain
  w_a = 4 with one copy level at w_c = 3 gives 0.0044 (float) and 0.0052
  (8 bits). It reaches in 28 to 29 sweeps from a random start and 29 from a
  wrong-latent start, using 8 extra sites (6 aggregators, 2 copies) and 9
  programming writes. Noisy "near-linear" fan-in does not reach at any
  tested point (0.075 to 0.143).
- **On this reduced instance the tied-copy baseline already reaches**:
  two symmetric copies at tie 2 give 0.0012 stationary error (float; 0.0011
  at 8 bits) with one extra site. They reach in 105 to 114 sweeps from a
  random start and 160 from a wrong-latent start. The reduced instance does
  not reproduce P-0008's finite-tie miss, so the row's comparison runs the
  other way here.

The pre-registered prediction ("the directed construction reaches 0.01
where tied copies at finite tie do not") fails on both halves. The
registered falsifier fires: the 8-bit two-colour error is 0.0589.
But its attribution is wrong. The ideal-relay arm shows the cause is
staleness, not lossy fan-in.

## The reduced instance

P-0008's `t4_h05_q035` family (J = 0.35, h0 = 0.5, q = 0.35, the four 8 × 8
templates) cut to a 2 × 3 window and two templates. The rule, fixed in the
script before it ran: the window and pair with the most differing pixels,
first in raster order. That is rows 3 to 4, columns 0 to 2, vertical bar
against horizontal bar. All six pixels differ (D = −0.5 each, m = 0), and
the latent bias is 0. The reduced model is its own open-boundary grid Ising.
Its exact posterior P(s | y) is enumerated for all 64 observations and
weighted by p(y). The latent is ambiguous: 94 percent of the p(y) mass has a
posterior between 0.1 and 0.9.

The model is symmetric under a global flip, so the chains run on the 32
observations with y₀ = −1 at double weight. Control C0 reruns the headline
staleness cell on all 64 and matches to 1e-16.

Metric: E = Σ_y p(y)·|P_stat(s = +1 | y) − P(s = +1 | y)| (primary), with the
maximum over y reported beside it. Reach is the sweeps from a uniform random
start until E stays at or below 0.01.

## Intended model, representation, update process, estimator

- **Intended model:** the exact posterior p(x, s | y) of the reduced
  instance.
- **Representation:** six pixel p-bits, one latent p-bit and relays. Fan-in
  is by aggregator p-bits that read one pixel colour class each (blocks of 3,
  the reduced stand-in for the 16-site fan-in cap). Fan-out is by one-way copy
  p-bits: r = 1 means two leaf copies read the latent; r = 2 means two
  level-1 copies read it and each leaf reads both. Each pixel reads its
  class's leaf with coupling D_i, one way.
- **Update process:** `chromatic` is the Z1 schedule, two phases per sweep.
  A site that reads a same-colour site sees its value from before the
  phase. `sequential` is a diagnostic, not available on the Z1. With ideal
  relays it is exact systematic-scan Gibbs (control C1).
- **Estimator:** the latent site's exact marginal. For tied copies, the
  mean copy marginal.

The chain has no detailed balance. Nothing here is called a Boltzmann
equilibrium; the stationary law is computed, not assumed. The tied baseline
is reversible, and its static law is enumerated as a check (control C2).

## Fixed choices and whether each bounds the metric

| Choice | Value | Bounds the metric? |
| --- | --- | --- |
| Update schedule | Z1 two colours; sequential as a relaxed diagnostic | **Yes, it decides the answer.** Two colours: 0.065 or more at ideal relays. Sequential: exact at ideal relays (1.1e-9). |
| Relay colouring | all 32 assignments of the five r = 1 relays to two colours; 8 have a same-colour loop and are skipped | **No.** Every feasible colouring gives one of two values, 0.0651 or 0.0708 (r = 2: 0.0793). |
| Fan-in rule | `linear` (threshold 0, w_a = 0.5, m in {1, 3}, gain-rescaled latent weight) or `thermo` (a thermometer bank, one aggregator per gap between attainable block sums) | **Yes.** Linear: 0.075 to 0.143 even with ideal fan-out and no staleness. Thermometer at w_a = 1, 2, 4: 0.053, 0.021, 0.0033. |
| Fan-out | r in {1, 2}, w_c in {1, 2, 3} | **Yes.** r = 1: 0.040, 0.0081, 0.0012. r = 2 adds a level and is worse at equal w_c (0.058, 0.013, 0.0020). |
| Bits | float, 8 under `split` | Small next to the schedule; see the results table. |
| Instance | 2 × 3 window, two templates, all pixels differing | **Could.** A strongly coupled, ambiguous latent maximizes the staleness bias. A latent with a weaker per-pixel coupling would show less. Not tested. |
| Convergence | stop at L1 change below 1e-9 or 400 sweeps | No. Every sequential, combined and tied-static number converged. The 0.0708 colourings and both r = 2 staleness cells stopped at the cap, and their last change was not logged. But 16 different colourings agree on 0.070842 to 1e-9, and the two r = 2 cells agree to 3e-10. The 0.0651 cells converged in 343 to 380 sweeps. Tied chains at w_t ≥ 2 did not converge in 400 sweeps, so their stationary errors are taken from the static law. |

## Results

Stationary latent error E (p(y)-weighted), reach from a random start, and
footprint. Extra sites exclude the six pixels and the one latent. Writes are
one per programmed non-pixel site.

| Construction | Schedule | Bits | E | Max over y | Reach (sweeps) | Extra sites | Writes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| directed, ideal relays, r = 1 | sequential (C1) | float | 1.1e-9 | 3.8e-9 | 25 | 8 | 9 |
| directed, ideal relays, r = 1, best colouring | two-colour | float | 0.0651 | 0.109 | never | 8 | 9 |
| directed, ideal relays, r = 1, other colourings | two-colour | float | 0.0651 or 0.0708 | up to 0.121 | never | 8 | 9 |
| directed, ideal relays, r = 2 | two-colour | float | 0.0793 | 0.132 | never | 10 | 11 |
| thermo w_a = 4, r = 1, w_c = 3 | two-colour | float | 0.0598 | 0.101 | never | 8 | 9 |
| thermo w_a = 4, r = 1, w_c = 3 | two-colour | 8 | 0.0589 | 0.099 | never | 8 | 9 |
| thermo w_a = 4, r = 1, w_c = 3 | sequential | float | 0.0044 | 0.0067 | 28 | 8 | 9 |
| thermo w_a = 4, r = 1, w_c = 3 | sequential | 8 | 0.0052 | 0.0079 | 29 (wrong start 29) | 8 | 9 |
| thermo w_a = 4, r = 1, w_c = 2 | sequential | 8 | 0.0087 | 0.0140 | 35 | 8 | 9 |
| thermo w_a = 4, r = 2, w_c = 3 | sequential | 8 | 0.0060 | 0.0094 | 31 | 10 | 11 |
| thermo w_a = 4, r = 2, w_c = 2 | sequential | 8 | 0.0141 | 0.0230 | never | 10 | 11 |
| tied direct, w_t = 2 | two-colour | float | 0.0012 (static 0.00123) | 0.0020 | 114 | 1 | 2 |
| tied direct, w_t = 2 | two-colour | 8 | 0.0011 | 0.0017 | 105 (wrong start 160) | 1 | 2 |
| tied direct, w_t = 1 | two-colour | float | 0.0087 | 0.014 | 55 | 1 | 2 |
| tied path (2 relays), w_t = 2 | static only | float / 8 | 0.0038 / 0.0014 | | not run | 3 | 4 |

Relay losses without staleness (sequential, float; the other relay ideal):

| Fan-in, ideal fan-out | E | Fan-out, ideal fan-in | E |
| --- | --- | --- | --- |
| linear m = 1, w_a = 0.5, λ = 1 or 2 | 0.143 | r = 1, w_c = 1 | 0.040 |
| linear m = 3, w_a = 0.5, λ = 1 | 0.086 | r = 1, w_c = 2 | 0.0081 |
| linear m = 3, w_a = 0.5, λ = 2 | 0.075 | r = 1, w_c = 3 | 0.0012 |
| thermo w_a = 1 | 0.053 | r = 2, w_c = 1 | 0.058 |
| thermo w_a = 2 | 0.021 | r = 2, w_c = 2 | 0.013 |
| thermo w_a = 4 | 0.0033 | r = 2, w_c = 3 | 0.0020 |

The tied static law (`exact_reference`) falls with tie strength in float,
from 0.0221 at w_t = 0.5 to 2e-5 at w_t = 4. At 8 bits it bottoms out near
0.0012 and rises again at w_t = 6 (0.0088), because a large tie coarsens the
coupling codebook for J, g and D. The tied chains mix slowly at strong ties.
At w_t = 3 and 4 the 400-sweep law is still 0.027 and 0.15 from exact, the
pattern P-0008 saw at full size.

## Reading

**Why staleness binds.** Exact block Gibbs needs the latent drawn from the
current pixels, then each pixel class drawn from that latent. In a
two-colour schedule each phase advances one hop. The loop pixel → aggregator
→ latent → leaf → pixel has four hops, so at least one pixel class always
reads a latent drawn from pixel values a sweep or more old. The resulting
law is the stationary law of a stale sampler. It is not the posterior, and
here it misses by 0.065 to 0.079. The note's hypothesis, a bias of
(staleness × per-sweep drift + relay noise) × influence, has the right
shape in one respect: the bias does not vanish as relay noise goes to zero.
The probe did not fit the form. The proposal's claim that one-way copies have
"no domain walls, errors accumulate with depth, not with mixing time" holds
only under sequential order.

**What the sequential results say.** Without staleness, the binary-fan-in
worry is real but solvable. A noisy near-linear aggregator is a poor logit
estimator, but a thermometer bank recovers the block sum exactly in the
noiseless limit, and at w_a = 4 it is within 0.0033. Its footprint is one
aggregator per gap between block sums. For the full 64-pixel problem that is
about one aggregator per differing pixel, not one per block. The sequential
order itself is not a Z1 feature.

**Design consequence.** On the supported machine, a one-way relay tree
does not carry a global latent within 0.01 at any relay quality. The
probe's best guess for a future design is a schedule that freezes
the pixels while the relay loop settles; that would cost extra phases or
clamp writes, which the sealed profile prices at a full write each. That is
a new row for the owner, not part of this one.

## Controls and integrity

- C1: ideal relays under sequential order give 1.07e-9 for both r = 1 and
  r = 2. This is exact Gibbs at the convergence tolerance.
- C2: the tied model's chromatic chain matches its static Boltzmann law
  (0.0012327 after 400 sweeps against 0.0012250 static; the chain's last
  L1 change was 1.1e-6).
- C0: the half-observation set matches all 64 observations to 1e-16.
- The script was committed before it ran (`4cf71ec`). Three edits followed,
  each committed before its run. `20ce17d` skips colourings with a
  same-colour loop, which the engine cannot evaluate; the first attempt
  crashed on one. `815f4cc` and `c3feb05` trim stages C and D to the CPU
  ceiling and reuse cells already logged by earlier runs (`run1-screen.log`,
  `run2-partial.log`) instead of recomputing them.

CPU: about 27 CPU-minutes over five invocations, against a 30-minute
ceiling. The first attempt hit the 420-second tool timeout before writing
anything, and a 12-second timing check followed. A second attempt crashed on
a same-colour loop after 35 seconds. Run 1 was stopped after the colouring
screen (9.5 minutes), run 2 after one cell (1.7 minutes), and run 3 finished
(8.0 minutes, `run3.log`). `probe.json` is 49 KB uncompressed (7 KB
gzipped).

## What this does not establish

- Nothing about the Z1 itself: one-way couplings are a relaxed constraint
  (rule 8) with no published cost, precision or placement rule. The probe
  assumed the symmetric codebook applies to them.
- Nothing at full size: a 64-pixel instance has a weaker per-pixel latent
  coupling, and its staleness bias was not computed.
- No hardware or energy claim, no task advantage, and no reopening of the
  denoising line.

Not done: the source card on asynchronous or stale Gibbs bias bounds is not
written. It was owed before a protocol, and none follows. The existing
[Aadit million-p-bit card](../knowledge/sources/aadit-million-pbits.md)
records staleness from overclocking as a speed lever, not a bias bound.
No novelty is claimed.

Dependency on `aggregator-read-gain` (P-0012, draft PR #157, not merged):
its declared arm did not stop on back-action or on a gain under 3 (5.2 at
k = 16), so the probe ran. That probe's post-hoc check found the archived
placement limits fan-in to 5 sites with gains of 1.1 to 2.7. This probe's
blocks of 3 sit inside that limit.

Evidence obligations (index of 2026-11-11): EO-001, EO-002 and EO-004 concern
rounding rules and do not apply. EO-003 (adsorption read allocation) does not
apply; this probe reuses no read allocation. EO-005 (the 2026-11-11
evaluation) is open and unaffected.
