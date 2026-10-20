# Thermo research agenda: representation, architecture and useful computation

**Owner selection:** 2026-10-20. **Repository acceptance:** by merge of this
documentation change. This implements [charter amendment 2](../../PROJECT_CHARTER.md#amendment-2-2026-10-20-representation-architecture-and-useful-computation).
It is a synthesis and selection guide, not a protocol, gatekeeper direction
review, new queue row or authorization to run an experiment.

## Mission and working hypothesis

Discover when a constrained physical stochastic system can deliver a useful
answer more economically than the strongest practical digital method, and
which representations and architectures make that possible.

The working hypothesis is that resident, local models with tolerable
coefficient error and modest output requirements can convert inexpensive
physical updates into useful information. Each condition can fail. A local
model may be digitally easy; a difficult distribution may require too many
reads; a small field error may alter the answer; a calibration may cost more
than the task. The research objective is to predict those boundaries with
stated scope, including evidence against the hypothesis.

## What the existing work supports

| Finding and source | Evidence and scope | Research implication; what remains open |
| --- | --- | --- |
| [M4H/M4I cap diagnosis](2026-09-23-cap-leakage-analysis.md) and [M4I report](../experiment-reports/2026-09-25-kernel-capacity-screen/summary.md) | Exact-reference analyses and bounded software studies; the tested caps and features did not meet the conservation objective. | Test fixed representation choices before optimizing execution. This does not prove conservation is impossible in every representation. |
| [P-0007](../experiment-reports/2026-10-16-dtm-baseline-headroom/findings.md) | Recorded exact references, software simulation and calibrated projections. Candidate families had headroom against the tested digital ladder, but the local-latent denoiser did not exploit it. | Digital difficulty and hardware suitability must be established independently. The ladder is not proof of the minimum digital cost. |
| [P-0008](../experiment-reports/2026-10-17-dtm-exact-posterior-lattice/findings.md) and [D-0005](2026-10-18-direction-review.md) | The 8-bit tie-limit reference closely represents the tested posterior; the finite-tie, finite-budget construction misses its frozen quality gate. The long oracle-pair diagnostic supports a finite-tie limitation with a small margin. | Distinguish an ideal representation limit from the model and dynamics actually executed. This is scoped to the tested construction; it is not a theorem against global latent variables. The owner nevertheless closed this research line. |
| [October 18 thinned-draw exploration](2026-10-18-dtm-exact-posterior-lattice-thinned-probe.md) | Exploration and calibrated projection: approximately 1.05e-7 J per 64-draw estimate for thinned continuation at draw spacing K=1024 under write-constant pricing, while task error remains about 0.002 beyond tolerance. | Reuse can lower projected I/O cost without producing a qualifying answer. It does not reopen the closed denoising line. |
| [Adsorption probe and independent audit](2026-10-20-native-adsorption-eligibility-probe.md) | Exploration with exact calculations and software samples. Exact 8-bit midpoint error is 0.011078 against 0.01; coherent field rounding explains the shift. Sampled intervals fail qualification even in the ideal control. | Native local structure alone does not ensure sufficient precision or statistical power. The result holds this allocation; it neither rejects all adsorption nor demonstrates harmful jitter or persistent mixing failure. |
| Same adsorption probe: digital baseline and placement | Exploration: aggregate-only exact digital reference about 11 ms per point; 144 active sites in a 1,134-position bounding box. Full digital energy and actual physical core/access costs are unknown. | A small eligibility case is not an advantage case. Bounding-box occupancy is not a measured chip utilization rate. |

No row above establishes physical-hardware advantage. Historical results
retain their original protocols, evidence labels, thresholds and disposition.
The exploration rows guide decisions without becoming recorded-study evidence.

## Questions that define the program

| Question | Decision it should inform | Evidence that would challenge the proposed direction |
| --- | --- | --- |
| What resource supplies the benefit: local parallel updates, persistent state, randomness, or avoided data movement? | Which architecture and workload to pursue. | The benefit disappears after matching the digital algorithm, output and total costs. |
| Which mathematical structures survive placement, finite coefficients and device variation? | Eligible representation classes and precision/connectivity requirements. | A bounded prediction fails on predeclared cases excluded from its development, or fidelity requires an unsupported feature. |
| Where does a digitally difficult task remain physically practical as it scales? | Whether a small result merits a larger comparison. | The relevant size/geometry loses placement, accuracy or cost headroom before digital methods become expensive. More sites alone do not establish this crossover. |
| What scientific or engineering decision needs the output, and how accurate must it be? | Task, observable and tolerance selection. | Model uncertainty overwhelms the decision margin, or the useful application needs interactions absent from the eligible model. |
| What belongs in the physical system, and what belongs in digital preparation or readout? | The division of work and interface requirements. | Calibration, host processing, state transfer or sufficient-confidence readout consumes the proposed saving. |
| Which single architecture change would remove a demonstrated bottleneck? | A quantified hardware requirement worth external testing. | The priced change offers no useful improvement, or another constraint immediately removes the gain. |
| What can be established here, and what requires a device or external collaborator? | Whether to prove a scoped result, measure a digital baseline, or seek hardware validation. | A claimed device benefit depends on unmeasured parameters or on a simulator implementing features the device lacks. |

## How to choose the next question

First identify a concrete design decision. For a task proposal, identify a
credible application and its strongest conventional route to the required
output. Use the [three-family comparison](2026-10-20-next-task-recommendation.md)
as a starting point, not a permanent ranking. Adsorption, surface-code
decoding and channel assignment are hypotheses with different risks; no
replacement task is selected by this agenda.

Summarize the following in the existing proposal, without a new form or
additional approval stage:

1. **User and decision:** what output changes a scientific or engineering
   choice; why the accuracy and confidence requirement is appropriate.
2. **Mechanism and prediction:** which representation or architecture
   property should matter, on which bounded family, and what observation
   would contradict the explanation.
3. **Constrained realization:** topology and footprint, coefficient range
   and precision, variation model, update law, initialization and output
   path. Separate the nominal target from each altered representation or
   process; do not assume site-dependent gains preserve its equilibrium.
4. **Comparison and costs:** the strongest practical baseline at matched
   output quality; setup/calibration, all access costs and independent
   information obtained. Declare reuse counts and applicability of exact
   algorithms, approximations, lookup tables or learned surrogates.
5. **Cheapest decisive check:** use existing evidence, an analytical bound
   or a bounded exploratory probe before a recorded study. Set the
   representation-error and statistical-precision budgets before sampling;
   account for correlated observations and independent replication.
6. **Decision rule:** explicit advance, stop and unresolved outcomes; what
   reusable conclusion or hardware requirement would remain after a
   negative. State what the proposed result cannot establish.

For a task advantage claim, identify cost headroom before optimizing a
sampler. For a representation or architecture study, name the numerical
design requirement it can resolve even without an application win. Neither
path permits cherry-picked cases, a weaker baseline, or a change to a
completed study's acceptance criterion.

A counterfactual architecture comparison should change one named feature,
retain the constrained reference, state all relaxations, and include the
feature's costs where known. If those costs are unknown, report a break-even
allowance or requirement rather than an advantage. This is analytical
co-design, not assumed access to a future device.

## Immediate disposition and November 11

The strategic synthesis is the work delivered here. The next selection
should resolve the joint gap in **application validity and digital cost
headroom**, or identify a sharply bounded architecture requirement from
existing results. No new sampler comparison is the default next step.

The denoising line stays closed under D-0005 and the owner's subsequent
decision. The tested adsorption allocation stays held. Balanced rounding,
extra sampling, a changed pore and a different task are unselected ideas,
not follow-ups authorized by the strategic shift. The suggested 0.003
representation budget from discussion is not an adopted protocol or a
validated application tolerance.

At the **2026-11-11** evaluation:

- Assess amendment 1 against its original requirements: a task's precision
  and variation tolerance, useful-sample update/read/write counts, and
  projected cost at stated quality against a conventional baseline. State
  negatives and missing evidence without resetting the deadline.
- Separately summarize the supported representation relationships, their
  model/parameter scope, and the hypotheses still needing independent
  tests. Do not generalize a failed construction to an entire task family.
- Identify one consequential next owner decision, including the option to
  stop or park a direction when no credible advantage case or useful
  design requirement remains. Do not fill an empty queue merely to keep the
  loop running.

The [research loop](../research-loop.md) and owner-curated
[queue](../research-queue.md) retain control of selection, protocol approval,
execution and acceptance. No scheduler, Hermes profile, approval gate,
frozen artifact or external service is changed by this document.
