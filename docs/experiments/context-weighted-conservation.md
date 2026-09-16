# Matched K4 context-weighting comparison

Predeclared September 16, 2026, following the recorded PR #27 result. Test whether
exact logical context weights improve conservation and directed-hop fidelity
under the same bounded optimizer. The earlier uniform objective improved local
averages while worsening pathwise survival and suppressing hops.

## Frozen control and single intervention

Use the complete committed local-conservation-tradeoff/study.json from PR #27.
Its canonical artifact digest is
`sha256:7b9e7a88e149bf888f3920f772b6109120944257dad71b7dbf8a4f099a6bd206`
and result digest is
`sha256:04be52b0465cafbfd3281109e8606c570b8b837fce4494a5b12a117ea77517f8`.
Pin the complete artifact, authenticate its three original M1 sources, and
replay its uniform search before using it as a control. Sources share one
initial matrix; source seeds are not independent fits. Preserve all five
control cells, including the logical reference and frozen initial parameters.

Derive the existing exact target-context trace from the canonical 25-site,
500-operation logical program, initialized at site (0,0). Use exact unsmoothed
pre-gate contexts and equal-occurrence pooling by sorted target hash. For group
g, w[g,a] is its mean logical parent-context probability over its occurrences.
The 37 profiles have zero weight for 11. These are frozen logical contexts,
not contexts sampled or updated from a fitted model. Persist the derived trace
identity and full pooled profiles in the result, rebuilding them on reload.

Only the objective's parent-context weights change. For each group optimize
F_w = sum_a w[a] sum_b (P[a,b]-T[a,b])^2,
C_w = sum_a w[a] sum_b P[a,b] M[a,b], and J_w = F_w + lambda*C_w,
where M indicates a particle-count change and P is the exact visible K4 law.
Differentiate the actual four-sweep uniform-reset hidden-then-output law.
Retain beta 1, float64, the same nine parameters and caps [-2,2].

## Identical bounded search policy

Use penalties [0,1,10], the archived start and nine zeros, exactly 100 projected
updates per start, and step 1/(1+lambda). Independently restart every group and
penalty. Select the minimum objective among archived start, archived endpoint,
zero start and zero endpoint, breaking ties in that order. Reuse the optimizer
policy without changing the old uniform arithmetic or evidence. Group
multiplicity does not scale optimizer gradients or learning rates.

The new weighted arm performs 22,200 group updates. Its matched uniform control
has the same 22,200-update budget; replay work is additional verification work.
No adaptive budget, new starts, stopping on improvement, smoothing, warm starts
from trained controls, extra loss terms, or post-outcome penalty selection.
Stop after this grid, regardless of scientific outcome.

## Evaluation and descriptive joint screen

Retain every PR #27 exact measurement for the new cells: uniform all-parent
failure and row TV, squared fidelity, unconditional and conditional hop MAE,
directed-hop asymmetry MAE, full local tables and errors, and all 500 exact
killed-survival rows. Report mean 00 failure and mean hop probability explicitly.
The parent-11 error remains visible even though its training weight is zero.

For all eight cells also report exact-target-context weighted F and C, averaging
groups with multiplicity/500. This equals the average over the 500 logical
pre-gate contexts. These weighted local measurements are not the fitted model's
actual context distribution or its terminal occupancy loss.

For each penalty compare the weighted arm with (a) its same-penalty uniform
control and (b) the frozen initial model. Record signed changes in final exact
survival, unconditional hop MAE and asymmetry MAE. A descriptive joint screen
passes only if survival is strictly higher and neither error increases. Use
literal float64 comparisons, no tuned tolerance. This is not a significance
test, an absolute fidelity certificate, or a release gate. Preserve all cells
and all comparisons even if none pass. Conditional hop error cannot substitute
for either unconditional error in this screen.

## Integrity, tests and research limits

The request binds the pinned control input identity, canonical context-source,
initial-state and pooling policies, objective, budget and evaluation rules.
Bind computed profiles, endpoint tables, fits and comparisons in the result.
On reload, pin and replay the control, reconstruct context weights, replay every
new update and measurement, and reject altered evidence even with repaired
digests. Render only validated evidence, write completion last, use a fresh
output directory, and retain runtime provenance.

Check weighted analytic gradients against central finite differences, zero
support and uniform-weight agreement, update-policy reuse, occurrence-weighted
measurement equivalence, and joint-screen rejection of survival-only gains.
Test source/profile/policy/result tampering and failed-publication boundaries.
Preserve all prior gates and numerical contracts. Scientific outcomes are
non-gating. Metrics are exact_reference up to floating-point arithmetic; there
are no samples, confidence intervals, hardware measurements, convergence
certificates, or optimal-capacity claims. This one-factor bounded comparison
cannot establish that context weighting is a general remedy.
