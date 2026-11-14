# Z1 features that would change an answer

2026-11-13. **Analysis, not evidence.** This note collects results already on
main. It runs nothing new, and every number is quoted from the source linked
in its row, with that source's evidence class. Joules are
`calibrated_projection` values under the sealed profile
([Z1 hardware model](../z1-hardware-model.md): 7.09 fJ per update, 1.692 pJ
per boundary read, 153.6 pJ per full SRAM node write). That profile excludes
host energy, I/O latency, idle and board power. No row is a hardware claim.

Charter amendment 2 asks which hardware requirements would change the answer
([agenda](2026-10-20-lab-research-agenda.md)). Since October the lab's
results have mostly ended as conditional requirements: a result holds only
if Z1 has some feature, or only at some price Z1 does not publish. This page
puts them in one place. It has two uses. Someone who talks to the hardware
side can ask about the right constants. When a constant is published, the
archive can be re-priced without new runs.

## Constants whose publication would re-price the archive

These need no new runs. Each lists the archived results that move.

| Constant (unpublished) | What moves | Size of the move |
| --- | --- | --- |
| **Clamp-only cost** (a clamp change is priced as a full write) | Every task cost: the denoisers, the pore, the thinned-draw path | The L = 8 denoiser runs 3.2e-7 J per 16-draw estimate at the write constant and 9.7e-9 J with clamp loads at the read constant, a factor of 33 ([four-week evaluation](2026-11-11-four-week-evaluation.md), clause 3) |
| **Re-reading one clamp load** (can a clamped array be read repeatedly without a reload?) | The `t4` estimate path | The clamp floor per 64-draw estimate falls from 6.3e-7 J to 9.8e-9 J, 64x, with quality unchanged ([thinned-draw probe](2026-10-18-dtm-exact-posterior-lattice-thinned-probe.md), exploration) |
| **Coupling precision, coefficient range and scale granularity** | Every bit-count result and the margin rule | See rows 4 and 5 below; nobody can design against a bit count until these are known ([D-0009](2026-11-08-direction-review.md), "Rubric 4") |
| **Cost, precision and placement of one-way couplings** (the model says they are supported but does not price them) | The aggregator and relay results | See rows 6 and 7 |
| **Host latency and serialized core I/O** | Every elapsed-time statement | Sweep times are lower bounds; current I/O is core-wise and serialized, which the profile idealizes away |

## Features, with the number each would change

| # | Feature | What changes with it | Without it | Source and scope |
| --- | --- | --- | --- | --- |
| 1 | **Cheap clamp change** (clamp-only constant near the read constant) | Denoiser break-even falls from 102 to 3.1 pJ per MAC | I/O is 99 percent of the projected joule on the denoisers and about 93 percent on the pore | [Four-week evaluation](2026-11-11-four-week-evaluation.md) clause 3; [DTM small image](../experiment-reports/2026-10-14-dtm-small-image/). The 114-parameter logistic stays more accurate at any price |
| 2 | **Per-replica temperature control at zero node writes** | Tempering (exchange every 4 sweeps) costs 3.0x to 48.7x the cheapest ordinary baseline and keeps its 4x to 16x sweep-time advantage | 117x to 1,886x, since every accepted swap rewrites two replicas | [Exchange cost projection](../experiment-reports/2026-10-07-exchange-cost-projection/findings.md); zero-field Ising targets, n ≤ 16. Still not an energy win |
| 3 | **In-place coefficient update cheaper than a full node write** (ramps, reflashes) | Annealed penalty ramps and reflash-based programs become affordable | A domain-wall ramp's writes cost about 1,000 times its sampling (P = 8). The codon run's margin over the idealized GPU Potts bound shrinks from 160x to about 2.5x | [Domain-wall probe](2026-11-11-domainwall-anneal-probe.md), "Resource accounting" (analysis); [codon card](../knowledge/sources/extropic-codon-optimization.md) (analysis, assumes every reflash writes every node) |
| 4 | **Coupling range × input fan-in of at least 4.3 logit units per clamped input bit** | A 500-operation compiled program in the PAsymSwap gate family survives at 95 percent | Cap 4 with one copy tops out at 91.3 percent; each extra copy is a 153.6 pJ write per clamp change | [P-0011](2026-11-13-margin-budget-copies-probe.md), exact. Covers that family only, not M4I's |
| 5 | **Enough coefficient bits at a usable scale** | The bit counts below hold | Below them, the stated tolerance fails | 4 bits under `split` for the L = 8 denoiser ([DTM small image](../experiment-reports/2026-10-14-dtm-small-image/)); 6 bits with fields in units of J for the bias memory ([AM coupling bits](../experiment-reports/2026-10-08-am-coupling-bits/)); 4 field bits under class diffusion or 9 under nearest rounding for the pore ([P-0009](../experiment-reports/2026-11-02-adsorption-precision-requirement/)); 5 bits for the placed tie-limit latent ([P-0010](2026-11-05-rounding-rule-transfer-probe.md)) |
| 6 | **One-way fan-in of at least 16 model sites per node, w ≥ 0.5, bias range about 8 in the pore's field units** (the pore's cap is 1.99) | Loading reads on the pore fall 5.2x | On the archived map a node reaches 5 sites and the gain is 1.1 to 2.7 | [P-0012](2026-11-13-aggregator-read-gain-probe.md), exact. Reads are about 16 nJ against about 44 nJ of writes per pore trajectory, so this moves the smaller term |
| 7 | **An update schedule other than two colours** (sequential phases, or freezing a block while relays settle) | A one-way relay tree carries a global latent: 0.0052 at 8 bits, 8 extra sites | 0.065 to 0.079 even with ideal relays (staleness) | [P-0013](2026-11-13-directed-relay-latent-probe.md), exact, one 2 × 3 instance. Extra phases would cost writes. Not a proposed row |
| 8 | **Degree above 16** | The bias-design associative memory (hidden degree 24, visible degree up to 128) and the E0 kernel (degree 17) place natively | Both run with placement relaxed and flagged | [Constraint-layer plan](2026-10-13-constraint-layer-plan.md) |

**Already tolerated, not a requirement:** static ±10 percent per-site beta
jitter was negligible in the three studies that isolated it: the L = 8
denoiser, the bias memory (±30 percent within 0.003) and the sampled 8-bit
`t4` arm ([four-week evaluation](2026-11-11-four-week-evaluation.md),
clause 1). The pore was not tested for jitter. In P-0012, applying aggregator
gain jitter, the declared worst-block bias check at w = 1 failed (0.0116),
while the pooled bias stayed under 0.001. Per-sweep and coupling-value noise
were never tested.

## What no feature on this list changes

- **The quality misses.** On the one task with a costed digital baseline, the
  digital model is more accurate at any price (clause 3). Where the digital
  method is costly (`t4`, 1.74e6 MACs), the lattice model misses tolerance by
  0.0117 against 0.01 at its finite tie, a representation limit (P-0008).
  Features 1 to 3 lower cost. None turns a miss into a pass.
- **The headroom gap.** No archived task has digital cost headroom, native
  placement, a user and a tolerance together. That is the four-week
  evaluation's un-park condition. A feature list cannot supply it; only a
  named task can.

Features 1 and 2 are the two whose publication would most change a recorded
comparison. Feature 1 moves every task cost. Feature 2 decides whether the
lab's tempering defaults are an energy recommendation or only a sweep-time
one. Features 4 to 8 matter only once a program needs them.

## Not done

No run, no queue row, and no change to any archive or frozen source. The
numbers were checked against the linked sources on 2026-11-13. The joule
arithmetic that this note quotes from D-0010, the domain-wall probe and the
codon card is analysis in those sources, and is labelled that way here.
