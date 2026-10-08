# Associative-memory coupling bits: exploratory protocol (draft, proposal P-0002)

Drafted on 2026-10-08 for queue row `am-coupling-bits`. Not frozen until the
owner approves it; no runner exists yet. Probe:
[`docs/research/2026-10-08-am-coupling-bits-probe.md`](../research/2026-10-08-am-coupling-bits-probe.md)
(exploration, four probe pattern sets per cell, NumPy sampler).

## Question

How does associative-memory recall fall with coupling precision and per-site
beta noise? The design is stage A's `bias` binary memory
([findings](../experiment-reports/2026-10-07-am-binary-emulation/findings.md)).
The metric is recall against bit width and beta jitter, measured against
stage A's exact references. The study produces a spec number: the bits needed
to keep recall within tolerance.

In the +-1 spin form this design has every visible-hidden coupling equal to
+-J, with J = beta / (2 sqrt N). Its hidden fields are -theta N J, and its
visible fields are integer multiples of J, up to 28 J at P = 128. Its "dynamic
range" of 14 to 26 is therefore field size in units of J. The probe found
that the answer depends almost entirely on how a codebook's cap places J,
so the cap is an arm, not a hidden constant.

## Fixed choices and what they bound

- **Design and configurations.** The `bias` arm only. Per cell, beta and
  theta are stage A's archived equilibrium choice for that cell (beta 16;
  theta 0.6 at P = 8 and at P = 32 with an 8-bit cue, theta 0.8 otherwise),
  read from stage A's archive and not re-selected. *Bounds:* recall is
  measured at a configuration chosen for unquantized hardware. Re-selecting
  beta and theta per bit width recovers part of the loss at low bits under
  the `full` and `split` caps (probe: 0.64 to 0.90 at 4 bits, `split`, P = 128,
  12-bit cue). Re-selection is not an arm: it would need its own development
  split, and it leaves the `coupling` answer unchanged.
- **Task.** N = 24, P in {8, 32, 128}, cue c in {12, 8}, i.i.d. uniform +-1
  patterns, three target patterns per set. These are the six stage A cells,
  with stage A's generator `default_rng([seed, P])`. *Bounds:* small N.
  The spec number is stated in units of the design's field-to-coupling
  ratio, which grows like sqrt(P) at fixed N, so it does not carry over to
  other sizes without that conversion.
- **Codebook rule.** The M5b rule
  (`meta_ebm_thermalization_core.round_parameters`): L = 2^(b-1) - 1 levels a
  side, step = cap / L, q(x) = step * clip(rint(x / step), -L, L), ties to
  even. It is applied to every visible-hidden coupling, hidden field and
  visible field in spin form. The M5b function accepts only b in {4, 8, 12}
  and refuses values outside the cap. The study therefore uses a study-local
  copy with clipping and other widths, tested bitwise against
  `round_parameters` at 4, 8 and 12 bits where no clipping occurs.
- **Codebook cap (the choice that bounds the metric), three arms:**
  - `full`: one cap equal to the instance's largest |parameter|, which is
    always a field. This is the literal reading of the row. At 4 bits the
    step exceeds 2J once the range passes 14, every coupling rounds to zero
    and the memory disconnects.
  - `split`: separate full-scale caps for couplings and fields. Couplings are
    exact, because all have magnitude J.
  - `coupling`: step = J (cap = L J). Couplings are exact, and fields are
    rounded to multiples of J and clipped at L J.
- **Bit widths.** Exact equilibrium: b in {3, 4, 5, 6, 7, 8, 10, 12}. The
  probe showed thresholds between the row's 4, 6 and 8, and recall that is not
  monotone in b under `full` and `split`. Sampled: b in {4, 6, 8}, the row's
  widths. *Bounds:* a spec number above 12 bits reads as "> 12".
- **Beta jitter.** A static per-site gain g_i ~ U[1 - j, 1 + j], one draw
  per simulated device and held for all chains and sweeps. Site i samples
  P(s_i = +1) = sigmoid(2 g_i gamma_i). This applies to visible and hidden
  sites alike; clamped cue sites are not sampled. j in {0, 0.1}, the row's
  value, plus j = 0.3 as a stress level on the unquantized design only.
  *Bounds:* per-sweep (dynamic) beta noise and noise on the coupling values
  are not this model and are not tested. Gains make the effective couplings
  asymmetric, so the jittered chain has no Boltzmann law.
- **Sampler and schedule.** THRML 0.1.4 two-block Gibbs on CPU in float32, as
  stage A: the missing visible block first, then the hidden block. Hidden
  start all off, missing bits uniform. Gains enter through a study-local
  subclass of `SpinGibbsConditional` that multiplies its `gamma` by a
  per-node gain array. This is THRML's documented extension point; no
  THRML or stage A file is edited.
- **Budgets.** K in {4, 16, 64, 256}, prefixes of one 256-sweep run, as stage
  A. 256 chains per target, three targets per set.

## Pre-registered expectations (from the probe)

1. **Spec number, exact equilibrium, within 0.01 of unquantized:** under
   `coupling`, 6 bits in every cell and 5 bits at P = 8. In general,
   L >= max|field| / J. Under `split`, at most 7 bits. Under `full`, 8 bits,
   and 10 to 12 bits at P = 128 with an 8-bit cue.
2. **Jitter null:** +-10% gain jitter changes recall by less than 0.005 at
   every budget in every cell, with or without quantization. +-30% stays
   within 0.015.
3. **Within a budget:** at K = 256, `split` at 8 bits stays within 0.02 of
   unquantized. `full` at 8 bits falls short by 0.06 to 0.15 at P = 128 and
   at P = 32 with a 12-bit cue, because quantization also slows mixing.
4. **4 bits under `full`** gives recall near 0.5 at P >= 32, the
   disconnected-memory floor. This is pre-registered and is not a failure.

## Exact anchor and evidence classes

- Exact equilibrium recall of each quantized, unjittered model enumerates
  every completion of the missing bits with the hidden layer summed out, as
  stage A's `exact_recall`. Its unquantized value must match stage A's function
  to 1e-12 on every held-out unit. `exact_reference`, float64.
- For jittered chains, the exact reference is the chain's own law, built
  from its two-block transition matrices at P = 8 with a 12-bit cue
  (4,096 x 256 states): recall after K sweeps, and at stationarity by power
  iteration to an L1 change of 1e-13. This is the only cell where the
  matrices fit. Elsewhere, jitter is measured as a paired sampled difference.
  `exact_reference`.
- All THRML sampling is `software_simulation`. Bit widths are a mathematical
  codebook and gains a static noise model. Neither is a Z1 encoding or a
  measured device property, and nothing here is hardware evidence.

## Arms and controls

Per cell, pattern set and simulated device (gain draw), sampled runs:

| Arm | Bits | Jitter | Gain draws |
| --- | --- | --- | --- |
| unquantized (control) | none | 0, 0.1, 0.3 | 1, 2, 2 |
| `full`, `split`, `coupling` | 4, 6, 8 | 0, 0.1 | 1, 2 |

That makes 4 + 9 x 3 = 31 sampled runs per set and cell. Every run of a set
and cell uses the same JAX keys (common random numbers), so the paired
differences against the unquantized, unjittered control remove most of the
sampling noise. The exact equilibrium part covers 3 caps x 8 widths plus the
unquantized model per set, cell and target.

Integrity controls:

- **Preflight.** On a tiny instance (N = 6, P = 3, cue 2, beta 2), one and two
  THRML sweeps of the gain-aware conditional, at j = 0.3 and with 4-bit
  `split` parameters, must reproduce the exact one- and two-sweep laws within
  a 0.999 multinomial tolerance (stage A's preflight method, 100,000 chains).
  A wrong-gain control (gains omitted) must be rejected. Production does not
  start unless both hold.
- At j = 0 and no quantization, the gain-aware conditional's states must equal
  the stock `SpinGibbsConditional` bitwise for one stage A held-out-style run
  on a probe seed.
- The NumPy probe sampler is not used in the study.

## Primary decision rule (thresholds fixed now)

Replication unit: the pattern set (and the gain draw within it for jittered
arms). Paired 95% bootstrap over sets, 2,000 draws, as stage A.

1. **Spec number (primary).** For each cap and cell, b*(cap, cell) is the
   smallest tested width b such that, at b and every larger tested width, the
   lower bound of the 95% interval of mean (quantized - unquantized) exact
   equilibrium recall is at least -0.01. The headline per cap is the maximum
   of b* over the six cells. If no width up to 12 qualifies, b* is "> 12". A
   cap whose b* is "> 12" is a result, not a failure.
2. **Finite budget (secondary).** For each sampled arm, cell and K, the paired
   difference against the control at the same K. It **holds** if the 95%
   interval lies within +-0.02 of zero (stage A's margin) and **falls short**
   if it lies below -0.02. Anything else is inconclusive.
3. **Jitter.** +-10% jitter is **negligible** in a cell if, at every K and for
   the unquantized arm and every quantized arm, the paired interval lies
   within +-0.01. It **matters** if any interval lies below -0.01. At P = 8
   with a 12-bit cue the exact chain-law difference is also reported. It does
   not gate.

## What one recorded sample means

One chain's missing visible bits after exactly K sweeps. Recall is the
fraction of those bits equal to the target pattern. The archive stores
per-target counts of correct bits at each K, not states or trajectories.

## Seeds

Held-out pattern sets use seeds 7400 to 7415 (16 sets), with stage A's
generator. These seeds are not used by stage A (7000 to 7003, 7100 to 7111),
stage A2 or any probe (9000+, 9700+, 9800+). There is no development split,
because nothing is selected. The JAX root is 20261013, fresh. Gain draws
come from `default_rng([seed, P, cue, 31, draw])`. Every run of a set and
cell shares one key, derived from the root, set, P and cue. Unit tests run
on probe seeds only.

## Budgets and CPU estimate

- Exact equilibrium: 16 sets x 3 targets x 6 cells x 25 models. From the
  probe's 23 CPU-minutes for 375 models per target on 4 sets, expect about
  10 CPU-minutes. The transition-matrix references for P = 8 with a 12-bit
  cue add about 2 CPU-minutes.
- Sampled: 31 runs x 16 sets x 6 cells = 2,976 THRML runs of 3 x 256 chains
  for 256 sweeps. Stage A's `bias` held-out units took 0.2 to 0.4 s at P = 8
  and 2.3 to 3.1 s at P = 128, so about 1.3 s on average and roughly
  1.1 CPU-hours. Gain-aware conditionals may add up to about 20 percent.
- Preflight, prefix check and full replay: about 0.3 CPU-hours.
- **Total about 1.5 to 2 CPU-hours.** That is about 40 to 60 minutes of wall
  time on 3 workers (`docs/environments.md` limits for this box), well under
  the 8 CPU-hour flag. The runner calibrates on probe seeds before the full
  run and reports the measured rate.

## Autosave and resume

The run exceeds 30 minutes, so the runner follows the
[autosave contract](../experiment-runner.md#autosave-and-resume-contract).
Each work unit (cell, set and its 31 sampled runs, or one cell and set of
exact references) is written atomically under `units/`. A unit is reused only
if the request digest and runner source hash match. The resume command is the
same command with `--resume` and the same output directory. A unit test
interrupts a small run and checks that the resumed record is identical.

## Persistence and archive size

`study.json.gz` holds the request, the preflight, per-unit exact recalls
(16 x 3 x 6 x 25 floats), per-unit sampled correct-bit counts (2,976 runs x
3 targets x 4 budgets ints), the gain draws' seeds (not the gains), the
evaluation and result digest. Expected size under 0.3 MB gzipped. Replay
recomputes every exact reference and the transition-matrix laws, checks the
archived values numerically (tolerance 1e-12 relative, never by rehashing
floats), and recomputes the bootstrap, spec numbers and verdicts from the
archived counts. It does not resample. A prefix check reruns one unit per
cell at K = 64 and requires identical counts. CI replays the archive with the
12-bit-cue exact references only (`--light`); the full replay is a local
gate.

## Gate

`completion.json`, written last by the runner, must show:
`status=am_coupling_bits_complete`, `cells=6`, `held_sets=16`,
`exact_units=288`, `sampled_runs=2976`, `preflight_passed=true`,
`stock_equality_passed=true`, `prefix_check_passed=true`,
`stage_a_archive_verified=true` (the archived configurations are read from an
archive authenticated by SHA-256), `replayed=true`, plus `spec_bits` per cap
and `verdict_counts`. Spec numbers and verdicts are scientific results, not
integrity checks.

## Stop rules

- Preflight or the stock-equality check fails: stop before production and
  report. Do not tune tolerances.
- The calibration rate projects over 4 CPU-hours: stop and report before the
  full run.
- Two consecutive failed runs with the same error: stop and report; no third
  variant.
- A replay mismatch is an integrity failure: no report is written as
  evidence.

## What a negative result means

If `full` needs more than 12 bits, the result means a shared full-scale
codebook is the wrong way to program this design: its fields must be encoded
relative to J, not relative to the largest field. It does not mean the
memory needs high precision. If `coupling` needs more than 6 bits at
P = 128, the probe's rule L >= max|field| / J is wrong and the spec number is
the measured one. If +-10% jitter matters anywhere, gain uniformity becomes a
hardware requirement for this design, quantified by the measured loss. None
of these outcomes is an integrity failure.

## Not claimed

Any Z1 encoding, DAC behaviour or measured device noise. Speed, energy or
latency. Behaviour beyond N = 24, for correlated patterns, for other designs
(one-hot, domain-wall), for dynamic beta noise or coupling-value noise, or
for configurations re-selected per bit width.
