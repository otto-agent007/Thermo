# Aggregator read gain: probe (P-0012)

2026-11-13. **Exploration, not a recorded study and not evidence.** Probe for
queue row `aggregator-read-gain`. Every number is an `exact_reference`
equilibrium quantity of the frozen adsorption pore, or of a one-way listener
attached to it, computed by transfer matrix on a CPU. No sampling, no seeds,
no training, no hardware. Read counts are asymptotic (Fisher) counts, not
sampled runs.

**Finding: no on any placement the archive supports; yes only under the
row's relaxed constraints, where the result is a hardware requirement, not
a saving.** The probe answers the row.

- **Constrained (archived map and caps): the gain is under 3.** On the
  pore's archived map an aggregator in a free slot reaches at most 5 pore
  sites, and the gain is 1.1 to 2.7 at every point (post-hoc). On the pore's
  coefficient caps it misses at low aggregator gain (2.93).
- **Relaxed (the row's design): the gain is 5.2.** With one-way couplings,
  k = 16 sites per aggregator along each z-channel, coupling w = 0.5, its
  own coefficient caps and a threshold the host adapts to a 0.5 firing rate,
  one aggregator read carries 5.16 to 5.98 times the loading information of
  a round-robin site read. That holds at all three frozen chemical
  potentials, at float, 4 and 6 bits, and at 0.9, 1 and 1.1 static gain.
  Reads to loading error 0.01 fall from 1,038 / 2,083 / 1,454 to
  188 / 393 / 250 (4 bits, loadings 0.2 / 0.5 / 0.8). The falsifier does not
  fire at w = 0.5 or 1. It does fire at w = 0.25 with 10 percent low gain
  (2.94 at loading 0.2).
- **Requirement read off the relaxed arm:** one-way fan-in of at least 16
  model sites per node, w of at least 0.5, and a bias range of about 8 in the
  pore's field units (its cap is 1.99). Reads are the smaller term on the
  pore: at the eligibility probe's allocation, writes are about 44 nJ and
  reads about 16 nJ per trajectory (D-0010's arithmetic, sealed profile), so
  even the relaxed gain trims a minority of the projected joule.

**The relaxed constraints decide the result.** Each of the following removes
the gain or stops the design:

- Archived placement: an aggregator in a free slot of the pore's map sees at
  most 5 pore sites, not 16. At k ≤ 5 the gain is 1.1 to 2.7, under 3 at every
  point (post-hoc check).
- Archived coefficient caps: the centring bias needs |b| up to 7.95 at
  w = 0.5, four times the pore's field cap of 1.99. On the pore's caps the gain
  is 2.93 to 4.34, under 3 at low gain, and worst-block inversion under jitter
  breaks down.
- Threshold rule: a fixed bias gives about 1 away from the middle point. A
  three-member bank read round-robin gives 2.2 to 2.6.
- Symmetric couplings: one aggregator's back-action moves loading by 0.011
  (w = 0.25) to 0.139 (w = 1), above the 0.01 tolerance. That case is a stop, as
  the row requires.

Within-block correlations cost 15 to 39 percent of the independent-site
gain on z-channels across all three w (22 to 39 percent at w = 0.5 and 1,
float, gain 1). They do not decide the verdict.

## Design question, mechanism, falsifier

- **Decision informed:** whether one-way fan-in nodes are worth asking of the
  hardware as a read-reduction feature for resident-model estimates.
- **Mechanism:** an aggregator p-bit is a dithered one-bit quantizer of its
  block's spin sum. Near the block median, one bit carries up to (2/π)·k
  times one site's information.
- **Falsifier (row):** a gain under 3 at any frozen chemical potential with
  correlations included.
- **Intended model:** the pore's Boltzmann law at each frozen chemical
  potential. **Representation:** the archived 144-spin map, plus aggregators
  as one-way listener nodes outside that law. **Update process:** chromatic
  Gibbs on the pore, assumed equilibrated. Each aggregator is drawn from its
  conditional given the current sites. Reads are thinned in time until they
  are independent; that thinning is assumed and not simulated, and its update
  cost is not counted. **Estimator:** maximum likelihood for the chemical
  potential from the read counts, mapped to loading. The aggregator arm and
  the primary baseline both use the model. A model-free sample-mean baseline
  is reported as well.
- **Strongest practical digital baseline:** the 512-state exact transfer
  computes loading exactly in about 11 ms per point (archived probe). This is
  not an advantage claim. The comparison is read paths inside one sampler.

## Fixed choices and whether each can bound the metric

| Choice | Value | Can it bound the metric? |
| --- | --- | --- |
| Model | frozen pore: 3×3×16, beta = epsilon = 1, wall phi = 3, archived spin map, three frozen chemical potentials (loadings 0.2, 0.5, 0.8); float | Fixed by the row. Its correlations are short: centre-channel correlation length 0.6 to 0.9 sites along z. Var(N) is 2.1 to 3.7 times the independent-site value; with short z-correlations, most of that must sit within slices (inferred, not measured). |
| Block geometry | primary: 9 z-channels of 16 sites; secondary: 9 compact slabs | **Yes, moderately.** Compact slabs widen the block sum more (Var 1.6 to 2.3 times independent, against 1.0 to 1.7). Slab gains are 4.23 to 4.89 at w = 0.5. |
| k | 16 | **Yes, under placement.** On the archived map, k ≤ 5 (see Placement). |
| w | 0.25, 0.5, 1 | **Yes.** w = 0.25 misses 3 at low gain. w = 0.5 already reaches most of the gain. |
| Threshold rule | primary: host-adapted to firing rate 0.5; also oracle optimum, fixed at the middle point, bank of 3 (round-robin or select) | **Yes, decisively.** See Results. |
| Coefficient caps | primary: own caps (coupling cap w, bias cap = largest half-rate \|b\|); secondary: pore's archived split caps (0.25, 1.9904) | **Yes.** The bias range must reach about w·k·\|2θ_block − 1\|: 3.97, 7.95 and 15.94 at w = 0.25, 0.5, 1. |
| Bits | float, 4, 6 on the constraint layer's codebook | No, under own caps: on z-channels 4 bits changes gains by at most 0.17. |
| Jitter | static aggregator gain 0.9, 1.1; host assumes 1 | **Only at w = 0.25** (gain). The declared worst-block bias check fails at w = 1 (0.0116). The pooled estimator's bias is at most 0.0009 (post-hoc). |
| Read model | one node read per independent, time-thinned instant | Favours the baseline, which is the row's condition. Thinning costs updates, which are not counted. |
| Tolerance | one asymptotic standard error 0.01 | No. The read gain is a ratio of reads, so this convention cancels. |
| Baseline | round-robin model-based site reads (`B_rr`); also best single site, random site, stratified mean | `B_rr` is within 1.3 percent of the model-free stratified mean. A random site read is 1.1 to 1.5 times weaker. |

## What ran

[`probe.py`](aggregator_read_gain_probe/probe.py), committed as `41fbd34`
before it ran, with every fixed choice in its docstring. The placement count
was printed once in a pre-commit smoke check. No gain was computed before the
commit. Output: [`probe.json.gz`](aggregator_read_gain_probe/probe.json.gz)
(gzipped after review; the uncompressed JSON is 524,692 bytes, SHA-256
`c797ba07653528cd0501941c4f248854d9089a4017e68a68a3fc944b53eb7678`).
386 CPU-seconds in one process, peak RSS 256 MiB.

1. **Authentication.** The archived transfer helper and the archived probe
   JSON are loaded through `thermo_lab.adsorption_precision_requirement`'s
   SHA-256-checked loaders. The frozen chemical potentials and caps come from
   that JSON. Nothing hash-bound was edited.
2. **Checks.** A study-local count-distribution transfer (joint law of
   occupied counts in one or two site masks) matches independent enumeration
   on 24 small cases, with maximum error 1.8e-15. On the full pore,
   finite-difference dL/dmu equals Var(N)/144 to 1e-8 at all three points.
   The midpoint Var(N) = 112.34123 equals the archived 112.341229.
3. **Grid.** 2 geometries × 3 w × (float own, 4 and 6 bits on each cap rule) ×
   3 gains × 3 chemical potentials × 9 blocks, all six threshold rules. Also:
   the independent-site counterfactual, the worst-block jitter inversion bias,
   the symmetric back-action shift and the placement count.

A post-hoc script, [`posthoc.py`](aggregator_read_gain_probe/posthoc.py),
was written after `probe.json` existed. It took 198 CPU-seconds. Output:
[`posthoc.json`](aggregator_read_gain_probe/posthoc.json) (SHA-256
`17d6990c51ba07f945022ac9574139c5707d6a9d48103cf646f2a4f4ae6493c1`). It
computes the pooled estimator's jitter bias, the gain if the tolerance is read
as RMSE, and the gain of placement-limited aggregators. It changes no declared
verdict. Total CPU, including checks, was about 10 minutes.

## Results

Primary arm: z-channels, own caps, host-adapted half-rate threshold,
round-robin over the 9 aggregators. Gain is aggregator reads against
round-robin site reads at the same error.

| w | float | 4 bits | 6 bits | min over 0.9 / 1.1 gain | Verdict (≥ 3 at all three points) |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 3.31 / 4.11 / 3.91 | 3.30 / 4.05 / 3.96 | 3.29 / 4.11 / 3.92 | **2.94** (6 bits, gain 0.9, loading 0.2) | fails |
| 0.5 | 5.52 / 5.39 / 5.75 | 5.52 / 5.29 / 5.81 | 5.49 / 5.39 / 5.77 | 5.16 | **holds** |
| 1 | 6.77 / 5.92 / 6.64 | 6.78 / 5.75 / 6.64 | 6.73 / 5.93 / 6.65 | 5.71 | holds on gain; declared bias check fails |

Each cell is loadings 0.2 / 0.5 / 0.8 at gain 1.

**Reads to error 0.01, w = 0.5, 4 bits:** 188 / 393 / 250 aggregator reads.
The baselines need 1,038 / 2,083 / 1,454 round-robin site reads,
719 / 1,584 / 908 reads of the best single site, and 1,600 / 2,500 / 1,600
random site reads.

**Threshold rules** (columns, w = 0.5, float, gain 1, loadings 0.2 / 0.5 / 0.8):

| Rule | Gain | Writes per chemical potential |
| --- | --- | --- |
| Oracle optimum (upper bound) | 5.98 / 5.42 / 5.95 | not implementable |
| Host-adapted half rate (primary) | 5.52 / 5.39 / 5.75 | 9 programming + 9 × bits adaptation (45 at 4 bits, 63 at 6) |
| Bank of 3, select best member | 5.52 / 5.39 / 5.75 | 27 programming, once; no adaptation; covers only the three frozen points |
| Bank of 3, round-robin | 2.25 / 2.42 / 2.26 | 27 programming |
| Fixed at middle point | 0.96 / 5.39 / 1.01 | 9 programming |

At w = 1 the fixed rule falls to 0.87 and 0.92 at the outer points. The bank
results are the probe's exploratory numbers made exact: the threshold rule,
not the coupling, sets whether the design works.

**Writes against reads.** The baseline needs no extra writes. At 4 bits the
adapted rule spends 45 writes per chemical potential to save 850 / 1,690 /
1,204 reads per estimate. At the sealed profile's write-to-read ratio (153.6
pJ / 1.692 pJ = 90.8, `calibrated_projection`), 45 writes equal about 4,100
reads. The saving repays them only after about 3 to 5 estimates at one
chemical potential. The select-from-bank rule moves the writes to 27 at
programming time. Both add 9 or 27 nodes and 144 or 432 one-way couplings,
none of which has a published cost.

**Within-block correlations.** The same blocks with independent sites at the
exact marginals give 7.05 / 8.35 / 7.87 at w = 0.5 and 9.07 / 9.69 / 9.55 at
w = 1, which is close to the note's independent ceiling of 9.7. With
correlations included the gain is 22 to 39 percent lower.

**Jitter.** At w = 0.5 the declared worst-block loading bias, with the host
assuming gain 1, is 0.0054 (float), 0.0085 (4 bits) and 0.0055 (6 bits), all
under 0.01. At w = 1 it is 0.0091 to 0.0116, so the declared check fails.
Pooled over the nine blocks, the round-robin estimator's bias is at most
0.0005 at w = 0.5 and 1, and at most 0.0009 at w = 0.25 (post-hoc). Read as
RMSE, that leaves every gain within 1 percent of the unbiased value. Under the
pore's caps, clipped biases saturate some blocks, and the worst-block
inversion bias is 0.247 or has no root.

**Pore caps** (w clips to 0.25; b clips at 1.99): 3.24 / 4.11 / 3.95 at gain
1, and 2.93 at gain 0.9. On the hardware-shaped caps the design misses at low
gain.

**Placement** (post-hoc). On the archived map (x + 4z, y + z) with the 16 Z1
offsets, a free slot sees at most 5 pore sites. 30 distinct 5-site sets exist.
The half-rate aggregator over such a set reaches 1.09 / 2.06 / 1.80 (w = 0.5)
and 2.03 / 2.72 / 2.58 (w = 1) of a round-robin site read. The best set
against the best site gives 0.81 to 2.34. k = 16 needs a placement in which
every one of an aggregator's offsets lands on a model site, or a relay tree.
The probe models neither.

**Symmetric couplings** (stop case). One aggregator at its half-rate bias,
coupled symmetrically, moves loading by up to 0.011, 0.054 and 0.139 at
w = 0.25, 0.5 and 1 on z-channels, and similarly on slabs. Every value is
above tolerance at the row's couplings, so the one-way assumption is
load-bearing.

## Constraints applied and relaxed (rule 8)

- **Relaxed:** one-way couplings (supported per the hardware model; no
  cost, precision or placement rule published); aggregator placement (k = 16
  is not placeable on the archived map); aggregator coefficient caps (own caps
  in the primary arm); jitter on pore sites (no exact stationary law; jitter is
  applied to the aggregator only); read thinning (assumed independent, update
  cost not counted).
- **Applied:** the frozen model and map, the 4- and 6-bit codebook,
  aggregator gain jitter, and priced programming and adaptation writes.

## Reading

- **Row answer:** yes under the relaxed constraints. With one-way couplings,
  k = 16, w ≥ 0.5 and an adapted or selected threshold, aggregator reads need
  5.2 to 6.9 times fewer reads than sparse thinned site reads at 4 and 6 bits
  and ±10 percent gain. At w = 1 the declared worst-block jitter-bias check
  fails, though the pooled bias is small. The design number is **5.2** (minimum over points,
  precisions and gain, w = 0.5, z-channels).
- **Constrained realization:** no. On the pore's archived map an aggregator
  reaches 5 sites, and the gain is under 3 everywhere. On the pore's
  coefficient caps it misses at low gain. Proposal 1's thesis depends on a
  fan-in feature the supported machine is not known to provide: a node that
  listens one-way to 16 model sites, with a bias range of about w·k.
- **Break-even requirement, not an advantage claim:** a one-way fan-in of
  k ≥ 16 per node, w ≥ 0.5 and a bias range of about 8 in the pore's field
  units (its field cap is 1.99). With host adaptation, the model must also stay
  resident at one operating point for at least 3 to 5 estimates to repay the
  writes, at the sealed ratio.
- **For `directed-relay-latent`,** which depends on this row: the row did not
  stop on back-action or on a gain under 3 in the declared arm. The k ≤ 5
  placement limit applies to its fan-in too.

## What this probe does not establish

- No hardware claim and no task advantage. The pore's digital exact
  reference computes loading in about 11 ms, and reads are compared inside one
  sampler.
- Not a sampled check. Chain mixing, the update cost of thinning, and pilot
  reads during adaptation are not modelled.
- It does not unhold `native-adsorption-eligibility` and authorizes no
  sampling.
- The second case named in the proposal note (susceptibility on the planar
  ferromagnet) was not run. The note put it after a pass. It is not a queue
  row.
- The source card on dithered one-bit quantization and stochastic computing
  is not written. It was owed before a protocol, and none follows. No novelty
  is claimed.

## Evidence obligations

From the dated index (`loop/evidence-obligations.json`, checked
2026-11-11), reconciled against main:

- **EO-003** (record the adsorption probe's unpowered read allocation as a
  lesson; source: direction review 2026-11-04; director and documentation
  author; due before a proposal reuses that read allocation). The index says
  `open`, but `docs/knowledge/lessons.md` already has the 2026-10-20 row
  ("Size a sampled allocation against the exact variance..."), so the index
  entry looks stale. This probe does not reuse that 64-read allocation.
- EO-001, EO-002 and EO-004 concern rounding rules, and EO-005 concerns the
  four-week evaluation. None applies to this row.

## Reproduce

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python docs/research/aggregator_read_gain_probe/probe.py --self-test
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python docs/research/aggregator_read_gain_probe/probe.py --output <new>.json
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python docs/research/aggregator_read_gain_probe/posthoc.py --output <new>.json
```

`posthoc.py` reads the committed `probe.json.gz` beside it. After review its
loader was switched from the uncompressed file to the gzipped one; nothing
else in it changed, and the decompressed bytes hash to the value above.
