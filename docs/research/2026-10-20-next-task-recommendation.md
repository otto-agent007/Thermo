# Thermo: next-direction decision memo
Date: October 20, 2026
Status: owner selected the native-adsorption eligibility screen; exploratory result accompanies this memo

**After the selected probe:** hold this operating point. Exact 8-bit loading
error reaches 0.011078 against the 0.01 tolerance, and sampled uncertainty
remains unresolved. The [probe report](2026-10-20-native-adsorption-eligibility-probe.md)
records the result and stops before a recorded study. The ranking below is
the rationale for selecting the probe, not a claim that it passed.

**Recommendation:** select native grand-canonical adsorption / local occupation thermodynamics for one capped representation-and-baseline screen. Rank narrow surface-code decoding second and heterogeneous two-choice channel assignment third. None currently has demonstrated economic advantage. Adsorption is the best architectural hypothesis; surface-code decoding is the most immediately standardized application benchmark.

## What the existing evidence permits

At the start of this review, main had no open research-queue row and no open PR. D-0005's owner decision chooses no further premise test in the closed denoising line and calls for a representation-led next task. After reviewing the recommendation, the owner asked to write it up and proceed. This PR records that selection and the bounded exploratory screen; it does not reopen denoising or change a frozen protocol.

| Evidence | Actual result | Implication |
|---|---|---|
| P-0008, exact reference | Tied-copy **tie-limit** MPM gaps: 8 bits 0.0003; 6 bits 0.0025; 4 bits 0.0752 | Precision can preserve the limiting representation; this is not a finite-tie hardware operating point. |
| P-0008, software simulation | Best priced K=1024 gap 0.0166; long oracle pair pooled mechanism estimate 0.0117, tolerance 0.01 | Frozen stop rule fires. The mechanism attribution is finite-w representation; the narrow margin and intervals do not overturn the owner's decision. |
| P-0008 development | Sampled gap worsens from 0.0197 at tie 1.75 to about 0.039 at tie 3 | Stronger ties are not a free representation repair. |
| PR #129 exploration | Thinned continuation at K=1024: approximately 1.05e-7 J per 64-draw estimate at write pricing, gap about 0.0122 | Lower cost did not establish task quality. Exploration and calibrated projection, not recorded positive evidence or measured hardware. |
| AM and constraint layer | Six-bit AM recall result used relaxed placement; hidden degree 24 and visible degree up to 128 | A good low-precision result is not evidence that dense memory fits degree 16. |
| Earlier planar/exchange studies | Difficult large cold targets remained problematic; exchanging replicas can destroy projected energy gains | Do not propose generic sampler improvement, free exchange or optimistic equilibrium assumptions. |

Sources: [P-0008 findings](https://github.com/otto-agent007/Thermo/blob/main/docs/experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md), [D-0005 and owner decision](https://github.com/otto-agent007/Thermo/pull/128), [thinned exploration](https://github.com/otto-agent007/Thermo/pull/129), [current queue](https://github.com/otto-agent007/Thermo/blob/main/docs/research-queue.md), [constraint layer](https://github.com/otto-agent007/Thermo/blob/main/docs/research/2026-10-13-constraint-layer-plan.md), [current project notes](https://github.com/otto-agent007/Thermo/blob/main/CLAUDE.md).

These results close the selected construction under its rules. They are not a theorem that all global information, denoising or stochastic hardware is impossible. Respecting the closure requires choosing a different task, not another encoding or estimator for the archived images.

## The decisive hardware filter

The implemented offset generators are (1,0), (2,1), (2,3), (4,1), with quarter-turn rotations. All offsets change checkerboard parity. Therefore the graph is bipartite; degree at most 16 alone does not prove placement.

Consequences derived from that graph:

- Nonnegative Max-Cut is solved by its two-color partition.
- Weighted independent set on a bipartite graph has a polynomial min-cut/vertex-cover solution.
- Generic sparse graphs, square-grid diagonals, parity checks and fixed-composition constraints are not automatically native.
- Mixed-sign problems need a frustration check: some become an easy attractive model after spin sign changes.

For the next task, prefer finite pair interactions on actual allowed offsets, no replicated global variable, no hard constraints enforced by extreme penalties, and a useful output whose accuracy survives coefficient rounding.

The sealed projection charges 7.09 fJ/update, 1.692 pJ/node read and 153.6 pJ/full node SRAM write: approximately 239 updates per read and 21,664 per write. The underlying appendix excludes host contributions and idealizes node-level access; actual core transfers can touch extra nodes. Never infer unpublished core dimensions or free on-chip aggregation. [Repository topology](https://github.com/otto-agent007/Thermo/blob/main/src/thermo_lab/meta_ebm_topology.py); [upstream architecture and Appendix B](https://arxiv.org/html/2608.01615v2).

## Three candidate families

| Rank and concrete task | Representation and latent risk | Read/write consequences | Conventional baseline and useful value |
|---|---|---|---|
| **1. Equilibrium adsorption in a fixed narrow pore**: estimate loading, working capacity and local occupation correlations | One binary occupation per accessible site; finite local pair energies and local chemical-potential/wall fields. A grand-canonical reservoir permits single-site changes, so no fixed-total-occupancy penalty. A narrow cubic pore can fit directly; arbitrary bulk 3D geometry cannot be assumed to fit. | Geometry and couplings stay resident across many observations. Each chemical-potential change still writes fields; snapshots require node reads and host reductions. Correlation limits the value of each draw. Sparse physical footprint can impose substantial overhead. | Exact transfer matrix for narrow pores; compiled GCMC with lookup tables/multispin or vectorized updates; valid cluster methods, then GPU batching at useful scale. Value: lower total energy/time to a specified adsorption-property accuracy for a validated material model. |
| **2. Surface-code decoding with perfect syndrome extraction and independent bit-flip errors**: choose a logical correction class | Each fixed logical sector has a nearest-neighbor pairwise Ising representation with boundary fields, degree at most four. Evaluate sectors separately rather than introducing one globally connected sector spin. Equal-magnitude signed coefficients are attractive for limited precision. Correlated errors/noisy measurement histories require a different representation. | Syndrome-dependent signs must be programmed each query. Sector scoring needs repeated configuration reads and host computation even though final output is one logical bit. Streaming latency and interface costs matter. | PyMatching v2; exact maximum-likelihood decoding in the favorable noise regime; conventional implementation of the same stochastic decoder. Value: lower logical failure at a fixed end-to-end cost/deadline, or lower cost at the same failure rate. |
| **3. Heterogeneous two-choice wireless channel reassignment**: minimize interference using legitimate per-radio channel options | One binary variable/radio; every 2x2 pairwise interference table maps exactly to fields and one coupling. Heterogeneous channel lists can produce frustration. No one-hot or global latent is necessary. Real interference graphs may contain prohibited edges; two-choice restriction and graph truncation can destroy usefulness. | Program each interference snapshot once, read and score multiple candidates. Changed traffic weights require new writes; best-of selection and throughput validation are host costs. | CP-SAT/MILP with bounds plus a competent specialized bipartite binary optimizer/local search. Compare unrestricted channel alternatives too. Value: interference reduction that preserves actual goodput/fairness at lower total allocation cost. |

### Why adsorption ranks first

Here the physical state has the same meaning as the modeled state. The candidate does not need a learned global latent, approximate one-hot constraint or infinite-strength equality chain. Its plausible advantage is sustained local stochastic computation on a resident physical model, paid for only when configurations are read.

There is an application beyond spin-model diagnostics: published lattice-gas GCMC work reproduces measured hydrocarbon adsorption and predicts adsorbate microstructure. That supports the usefulness of adsorption calculations, **not** the claim that its full aerogel model maps to Thermo or that this proposed tiny pore is calibrated to methane. [ORNL primary record, Gruszkiewicz et al.](https://www.ornl.gov/publication/supercritical-fluid-adsorption-weakly-attractive-solids-universal-scaling-laws), DOI 10.1021/acs.jpcc.8b04390.

A concrete finite pairwise model class is available in Schneider et al., Eq. 22: cubic single-occupancy sites, nearest-neighbor attraction and site fields. Their wall-field convention is printed with a positive energy contribution and positive wall field; it must not be silently changed to reproduce attractive-wall behavior. Their hysteresis results are also not equilibrium benchmarks. Use the literal equation for a model-class screen, not a claimed reproduction of experimental adsorption. [Primary paper](https://www.nature.com/articles/s41598-017-07406-2).

The largest objections are strong digital methods, limited embedding scalability, and the absence so far of a validated material workload with demonstrated digital headroom. Avoiding coexistence may ease mixing but also makes digital computation easier. A positive small screen establishes eligibility, not usefulness or commercial advantage.

### Why decoding is second

Two fixed sector models avoid the global-latent problem. The output is a useful decision, not recovery of an individual corrupted pattern. Hutter–Wootton–Loss provide a local, single-temperature MCMC sector decision rule, but its sampled mean-energy comparison is a heuristic for a partition-function comparison, not an exact identity. [Primary algorithm](https://arxiv.org/pdf/1302.2669).

The strongest counterargument is that exact maximum-likelihood decoding is already O(n²) for noiseless syndrome extraction with independent X/Z errors—the regime with the cleanest mapping. PyMatching provides a highly optimized practical comparator. Thus this is a good second screen, with significant risk that digital specialization and input programming remove all headroom. [Exact decoder](https://arxiv.org/abs/1405.4883); [Sparse Blossom / PyMatching v2](https://quantum-journal.org/papers/q-2025-01-20-1600/).

A future selected screen could fix two code distances and two error rates, compare paired logical-failure rates, and count actual per-node programming/readout. Stop before scaling if even ideal sampling lacks a useful quality-cost frontier. Do not broaden to higher-order noise models to manufacture difficulty.

### Why channel assignment is third

Weighted Soft List Coloring is a genuine channel-assignment formulation. The two-choice specialization is this memo's proposed restriction, not the cited paper's demonstrated Thermo workload. It avoids one-hot penalties, but its actual interference graph and available choices must survive application validation. General NP-hardness says nothing about hardness on this restricted lattice. [Primary application](https://pure.kaist.ac.kr/en/publications/a-conflict-aware-channel-assignment-in-multi-radio-multi-channel-/).

Reject ordinary two-identical-channel Max-Cut and native bipartite MWIS as advantage targets. Require direct placement and a full-network goodput check before any sampler comparison. Strong bipartite Boolean quadratic methods already exist; CPU Gibbs alone would be an inadequate baseline. [Specialist optimization research](https://arxiv.org/abs/1605.02038).

## Proposed first adsorption experiment

**Decision number:** whether one fixed allocation of counted updates, reads and writes estimates normalized loading and working capacity to one percentage point in a directly represented finite-pore model, at 8-bit coefficients and 10% static beta jitter. Finding a minimum qualifying allocation would require a separately approved study and a matched conventional baseline.

The owner selected this direction for the cheap eligibility probe permitted by the research loop. The detailed [fixed choices](native_adsorption_probe/fixed-choices.md) were written before sampling. These thresholds remain research decision thresholds, not validated material requirements. The probe has a 30 CPU-minute limit; the two-hour recorded study below is conditional and is not authorized by this note.

### Model and exact reference

Use one open 3x3x16 rectangular pore: 144 occupation variables, 327 nearest-neighbor bonds, maximum degree six. With grand-canonical energy

\[
G(n)=-\epsilon\sum_{\langle ij\rangle} n_i n_j+
\sum_i(\phi_i-\mu)n_i,\qquad n_i\in\{0,1\},
\]

the spin substitution n=(1+s)/2 gives

\[
J_{ij}=\epsilon/4,\qquad
h_i=(\mu-\phi_i)/2+\epsilon z_i/4
\]

under energy convention -sum J s_i s_j -sum h_i s_i. Boundary coordination z_i must be included. Choose the literal published wall-field prescription for this screen and state its repulsive sign; no attractive-wall or real-fluid calibration claim.

A direct map exists:

\[
(x,y,z)\longmapsto(x+4z,\;y+z).
\]

The three edge directions become (1,0), (0,1), (4,1), all permitted offsets. The map is injective for x=0,1,2. A static arithmetic check performed for this memo verified all 327 edges and no coordinate collisions. That initial check was algebraic. The separately documented [eligibility probe](2026-10-20-native-adsorption-eligibility-probe.md) adds exploratory software evidence.

The bounding box is 63x18, or 1,134 lattice slots for 144 active variables. This poor packing is material: report active-node lower-bound cost and charge any additional core/footprint accesses required by the supported interface. This is a synthetic-lattice placement proof, not a claim that it fits an unpublished physical core.

A transverse slice has nine variables; a 512-state transfer matrix gives an exact finite-system reference. Use efficient factorized transfer operations, not an artificially expensive dense implementation.

### Fixed screen, capped at two CPU-hours

- One geometry; proposed dimensionless temperature beta*epsilon=1 with epsilon=1. Use phi=+3*epsilon on sites adjoining the four lateral walls (x=0 or 2, or y=0 or 2), zero elsewhere, with open ends. This is a declared model-class test point, not a calibrated material temperature. No fitted model, new sampler or latent variables.
- Fix three chemical potentials from the full-precision reference at low/middle/high loading (approximately 0.2, 0.5, 0.8), then freeze them. Do not retune after rounding/jitter. Middle loading can be difficult; do not call it automatically away from coexistence.
- Compute full-precision, 6-bit and 8-bit exact equilibrium references with the existing codebook and split caps: coupling cap epsilon/4 and a single field cap equal to the largest absolute field across the three frozen chemical potentials. Freeze these caps across all arms.
- Sample the 8-bit/10%-jitter operating case, 6-bit/10% diagnostic, and an ideal control using the existing local two-color update law. Use three independent seed sets and both empty/full starts. Compare each start separately with the reference before pooling; opposite transient biases must not cancel.
- Maximum 4,096 sweeps per trajectory; fixed burn-in/read schedule declared before execution, no adaptive extensions. A concrete default is 1,024 burn-in sweeps followed by 64 snapshots separated by 48 sweeps.
- Score mean loading at each chemical potential, working capacity (high minus low loading), and a fixed aggregate nearest-neighbor occupation statistic. Use independent seed groups for uncertainty and autocorrelation-aware diagnostics within runs. Three jitter maps do not establish reliability over a device population.
- Nonuniform site beta can alter the stationary law; do not replace it by an assumed equilibrium at average beta. Compare the actual jittered update process with the intended exact target.
- Benchmark an aggregate-only factorized transfer calculation (avoid computing unneeded per-edge marginals) and compiled GCMC on the same outputs and tolerances. Include a valid cluster method if it is competitive; any later scale study also needs batched GPU comparison. Digital implementations already exploit checkerboards and multispin coding. [Primary GPU implementation](https://arxiv.org/abs/1007.3726).

Only the fixed, exploratory allocation is run in this PR. A recorded study would require a protocol approved at its concrete commit, and must first survive the probe's stop rule. This memo specifies a model-class screen; a source-calibrated real-material workload has not been selected.

### Go/no-go rules

**Numerical eligibility:** the 8-bit/10%-jitter operating case must have maximum absolute loading error at most 0.01, working-capacity error at most 0.01, and aggregate pair-occupation error at most 0.02. Confidence bounds must support those claims, and empty/full starts must agree within the same declared budget. These resolutions would support distinguishing candidate capacities separated by materially more than 0.02; they do not authorize a material ranking from the single-pore screen.

**Practical success target:** at least **3x lower end-to-end energy at matched task accuracy** than the cheapest qualifying digital baseline. Threefold is a proposed margin against projection uncertainty, not a hardware claim. Report time to accuracy separately. Training is absent; programming, initialization, readout, host reductions and repeated parameter queries are included.

For accounting, with U updates, R node reads, W non-clamp full-node writes and C clamp writes counted separately,

\[
E_{\mathrm{array}}=
7.09\,10^{-15}U+1.692\,10^{-12}R+
1.536\,10^{-10}W+e_C C.
\]

Report the existing clamp band e_C in [1.692e-12,1.536e-10] J and both actual counts and excluded costs. Do not double-count clamp writes inside W. Extra physical/core accesses must be visible.

Price digital MACs at the queue's 0.1, 1 and 10 pJ/MAC, but **do not convert table lookups, integer/bitwise operations, RNG or memory traffic into invented MACs**. Report their counts and energy assumptions separately; use measured digital system energy when available. An unknown digital-energy denominator cannot establish a numerical advantage.

For a defensibly measured or priced qualifying digital baseline, publish the residual host/interface allowance

\[
E_{\mathrm{host,max}}=E_{\mathrm{digital}}/3-E_{\mathrm{array}}.
\]

A nonpositive allowance even with favorable array accounting rejects the proposed threefold advantage. A positive allowance is a necessary condition, not proof that a real system meets it.

**Advance:** numerical eligibility plus credible positive cost headroom justifies one subsequent source-calibrated, larger application comparison. The narrow transfer-matrix case alone never justifies a broad TSU advantage claim. Without defensible digital pricing, a numerical pass permits only resolving the missing cost measurement, not a scale campaign.

**Stop/park:** failed 8-bit representation fidelity; unresolved empty/full-start disagreement or uncertainty at the fixed cap; no qualifying quality-cost point; dependence on unsupported transfers, ignored footprint, global penalties or tied copies; or no scientifically validated native workload beyond an easy narrow model. No tolerance widening, extra tie sweep, sampler tournament or automatic replacement row follows.

## November 11 and the owner decision

Keep the charter evaluation separate from this new hypothesis. Existing evidence supports: design numbers were produced; the thinned exploration is cost-competitive above approximately 1 pJ/MAC under the owner's reading, but remains about 0.002 short on quality; no tested task wins on both. A new proposal does not revise that outcome.

The owner selected **the capped native-adsorption screen**, with the numerical and cost stops above. Its [exploratory result](2026-10-20-native-adsorption-eligibility-probe.md) determines whether to promote or park this allocation. The queue change in this PR records that disposition for owner review; merging remains acceptance. No denoising rerun or frozen-protocol change is part of this work. Surface-code decoding and channel assignment remain alternatives for a future explicit selection, not automatically queued follow-ups.
