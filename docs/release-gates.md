# Release gates

Per-study release-gate requirements. This text moved here verbatim from
AGENTS.md on 2026-09-25; only the headings are new. AGENTS.md keeps the
research rules, the base gate commands and a one-line index of these gates.
Each study's frozen protocol under [`experiments/`](experiments/) remains
authoritative, and a gate's requirements here must not be weakened.

## M0–M1: trajectory estimator, composed finite-Gibbs and composed refinement

The trajectory estimator gate validates an exact-categorical three-site
microcircuit and non-gating seeded estimator evidence. It does not perform
parameter refinement or a finite-Gibbs-horizon composed-program comparison.
The one-step refinement gate preserves those checks, applies one bounded
shared-parameter update, and records the exact objective before and after. It
remains a bounded exact-categorical experiment, not the 25-site or
finite-Gibbs-horizon comparison.

The full composed finite-Gibbs gate is a NumPy `software_simulation`: it
samples 32,768 complete 25-site, 500-occurrence trajectories per seed for the
`independent`, `target_context`, and `model_context` frozen artifact families
at equilibrium and `K = 1, 2, 4, 8, 16, 30`. Use one PCG64 uniform vector per
occurrence across all family/horizon comparison cells. Treat the exact target
checkpoints and frozen local tables as `exact_reference`; treat sampled
occupancy error, leakage, invariant, final-marginal, and paired-comparison
metrics as `software_simulation`. The 89 improved / 37 worsened / 0 unchanged
final paired error/leakage results are non-gating descriptive outcomes. Do not
extend this gate into live THRML sampling, official Thermalizers, hosted or
hardware claims, iterative optimization, or full 25-site trajectory-level
parameter refinement within that frozen-artifact gate.

The separate composed trajectory-refinement gate performs exactly one
equilibrium update across 37 shared nine-parameter groups. Preserve the three
independent 32,768-trajectory roles (occupancy, gradient, held-out evaluation),
same-parent non-propagated references, learning rate 0.01, and bounds [-2, 2].
Retain the historical plug-in squared terminal occupancy objective before and
after, including its finite-batch bias. The v2 evidence contract also records
the unbiased order-two U-statistic before/after and signed after-minus-before
difference, plus paired delete-one jackknife SE and approximate normal 95%
interval from the joined terminal second-moment counts. A wholly negative
interval means improved, a wholly positive interval regressed, and otherwise
inconclusive; all conclusions are descriptive and non-gating. The per-run
interval is within-evaluation uncertainty conditional on the frozen parameter
pair, distinct from across-seed aggregate intervals. Bind the estimator,
uncertainty, conclusion, and scientific-status policies in the checked request
and versioned result digests. Reject v1 records as v2 evidence, retaining their
original plug-in semantics as historical results. Reconstruct counts, moments,
derived values, policies, and digests at aggregation/reporting boundaries.
Require exact row consistency for proven identical columns and exact
positive semidefiniteness of the full centered Gram matrix; these necessary
conditions are not a complete binary-realizability test. Retain exhaustive
tiny-fixture calibration documenting possible near-zero and small-sample
undercoverage, without tuning the checked estimator or budgets. Keep
exact three-site gradient checks and deep persisted-record validation at
aggregation/reporting boundaries. No iterative or finite-Gibbs refinement is
included.

## M2: frozen-pair finite-sweep audit

The separate M2 frozen-pair audit consumes validated M1 source records, freezes
their actual initial/updated parameters, and evaluates all seven canonical
horizons without training. Fresh numerical compilation is not a bitwise
historical fixture: compare legacy and M1 statistics using exact replay from
the same frozen parameters, and retain historical values under their original
identities. Require exact equilibrium terminal-count replay against the
supplied source; report whether its identity matches the historical PR #20
pair. Keep local endpoint tables exact_reference and sampled program metrics
software_simulation. Use one PCG64 uniform vector per occurrence across all
14 member/horizon cells. Preserve M1 uncertainty limitations and mark intervals
as pointwise rather than simultaneous. Validate source/request/table identities,
joined moments, particle histograms, and leakage margins on reload and before
reporting. Algorithmic sweep/pbit counts are not measured device operations.

After generating the three M1 source records above, the M2 release gate is:

~~~bash
uv run thermo-lab audit-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/frozen-pair-finite-sweeps
~~~

Use a fresh destination and require full_three_seed_release in completion.json.
Do not interpret negative or inconclusive scientific results as integrity
failures. M2 introduces neither finite-sweep gradients nor iterative training.

## M3: exact finite-sweep gradient contract

The separate M3 exact finite-sweep gradient gate preserves the existing
three-site, two-operation fixture and nine shared parameters. Differentiate
the uniform reset and all hidden-then-output sweeps at K=1,2,4,8,16,30. Require
endpoint-score, visible chain-rule, independent Bernoulli-autodiff, and existing
kernel finite-difference agreement, including untied occurrence derivatives
and their shared sum. Preserve beta 1, caps [-2,2], step 1e-6, exact tolerance
1e-12, and finite-difference tolerance 1e-7. Reject equilibrium-score and
missing-occurrence negative controls at K=1. Keep all evidence exact_reference
and restore scoped JAX x64 configuration. Strictly reconstruct the persisted
request and numerical evidence before reporting; write completion last.
This gate performs no update or sampled full-program gradient estimation.

```bash
uv run thermo-lab check-finite-sweep-gradients \
  --output-dir results/finite-sweep-gradient-contract
```

## M4: bounded finite-sweep refinement

The separate M4 gate validates sampled finite-sweep gradients and performs
exactly five updates at K=4 from the actual supplied M1 initial parameters.
Preserve the predeclared protocol in
docs/experiments/bounded-finite-sweep-refinement.md: beta 1, float64, learning
rate 0.01, bounds [-2,2], independent 32,768-trajectory occupancy and gradient
roles at each update, and untouched 32,768-pair final evaluation. Select update
five in advance. Use true finite endpoint scores, same-main-parent independent
references that never propagate, and sum shared occurrences before reducing
gradient second moments. Training-role loss diagnostics are not held-out
checkpoint comparisons. Preserve M1 uncertainty limitations and label sampled
results software_simulation; endpoint and microcircuit references remain
exact_reference. Normalize signed zero in sampler identities. Reconstruct
source lineage, role seeds, tables, updates, counts and joined moments on
reload; replay gradient moments within the fixed numerical tolerance while
binding their exact stored values in digests. Keep cached summaries immutable
and never cache externally supplied results. Write completion last; scientific
improvement is non-gating and no convergence or device-cost claim follows.

After generating the three M1 sources, the M4 release gate is:

```bash
uv run thermo-lab refine-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/bounded-finite-sweep-refinement
```

Require full_three_seed_release, updates_per_run=5, horizon=4, and
selected_checkpoint=5 in completion.json. A partial diagnostic is insufficient
for this release gate. CI must preserve the full evidence and show its report.

## M4B: matched training budget

The separate M4B gate follows docs/experiments/matched-training-budget.md.
Extract the original M1 source records from the three committed M4 seed JSON
files as documented in README.md, then run:

```bash
uv run thermo-lab compare-training-laws \
  results/m4b-sources/seed-0000000000.json \
  results/m4b-sources/seed-0000000001.json \
  results/m4b-sources/seed-0000000002.json \
  --output-dir results/matched-training-budget
```

Require all three seeds, updates_per_arm=5, selected_checkpoint=5,
evaluation_horizon=4, and matched_hardware_cost=false in completion.json.
Preserve the checked archived identities; never substitute regenerated sources.
Both training occupancy and gradient laws belong to their respective arm.
Match endpoint draws and updates, not energy or hardware sweeps. Bind the new
21-role seed namespace, every update, and three final paired comparisons to
replayable evidence. Primary signed loss is finite minus equilibrium; negative
favors finite training. Retain particle leakage, interval limitations, and
non-gating outcomes. M4B does not change M1–M4 or establish convergence,
inference-sample savings, or a device advantage. CI runs the complete M4B study
and preserves its bounded artifacts. Run the existing gates without weakening
their thresholds, provenance, or historical evidence boundaries.

## M4C: conservation diagnostic

The conservation diagnostic follows docs/experiments/conservation-leakage-diagnostic.md.
Run `uv run python -m thermo_lab.conservation_audit --output-dir results/conservation-diagnostic`
with a fresh destination. Require full_three_seed_release, seeds [0,1,2], horizons
[4,"equilibrium"], 32768 samples, 500 operations, and zero parameter updates.
Keep exact killed-path survival separate from full-sampler terminal leakage and
returns to one particle. Authenticate the same archived M1 initial sources; do
not substitute regenerated or trained parameters. New streams do not reuse M4B
evaluation. Replay all persisted counts before reporting and write completion
last. Local failure probabilities describe frozen parameters, not a proved
optimal representational limit. Preserve the older evidence and gates.

## M4D: local conservation trade-off

The local conservation trade-off follows docs/experiments/local-conservation-tradeoff.md.
Run `uv run python -m thermo_lab.conservation_tradeoff_audit --output-dir results/local-conservation-tradeoff`
with a fresh destination. Require full_grid, all source_seeds [0,1,2], one shared
initial matrix, 37 groups, penalties [0,1,10], two starts, 100 updates per start,
22,200 group updates, K4, 500 operations, and zero samples. Authenticate all source
records and replay every projected update, endpoint selection, and exact metric
before reporting; write completion last. Preserve uniform parent/context weights
and fixed step 1/(1+lambda). Report every penalty, conservation, unconditional and
conditional hop fidelity, asymmetry, and exact killed survival. A fixed local
search establishes attained results, not convergence, optimal capacity, or a
physical-device benefit. Preserve all prior evidence and gates.

## M4E: matched context weighting

The matched context-weighting study follows docs/experiments/context-weighted-conservation.md.
Run `uv run python -m thermo_lab.context_conservation_audit --output-dir results/context-weighted-conservation`
with a fresh destination. Require full_grid, source_seeds [0,1,2], one shared
initial matrix, 37 groups, penalties [0,1,10], two starts and 100 updates per start,
22,200 updates in each arm, control_replayed, K4, 500 operations, zero samples,
eight evaluated cells, and six descriptive comparisons. Pin and replay the
complete committed PR #27 uniform artifact before comparison. Rebuild exact
unsmoothed logical contexts with equal-occurrence pooling; derived profiles
belong in the result. Preserve the uniform optimizer's arithmetic and identities.
Report all-parent errors, occurrence-weighted logical-context errors, killed
survival, and unconditional hop/asymmetry errors together. A joint screen passes
only for strict survival gain and neither error increasing; it is descriptive,
not a release gate or fidelity certificate. Replay all new updates and profiles
before reporting; write completion last and preserve every older gate.

Context-study schema 1.1 / request v2 preserves the exact complete uniform-control
pin while checking its independently rebuilt numerical outputs at absolute
tolerance 1e-12 and relative tolerance zero. Requests, sources, structure, scalar
types, discrete decisions and penalty labels remain exact. Only derived result
and joint-table digests may differ across runtimes; the original pinned values
remain the comparison reference. New weighted evidence still requires strict
complete replay. Preserve schema 1.0's original strict semantics and the older
uniform-study validator. See the dated CI correction in the study protocol.

## M4F: asymmetry preservation

The separate M4F study follows docs/experiments/asymmetry-preservation.md.
Run `uv run python -m thermo_lab.asymmetry_preservation_audit --output-dir results/asymmetry-preservation`
with a fresh destination. Require full_study, three source seeds sharing one
matrix, 37 groups, lambda=mu=1, two starts, 100 updates, 7,400 new and 7,400
control group updates, three consumed reference cells replayed, four evaluated
cells and two comparisons, K4, 500 operations, and zero samples. Pin the complete
context archive, authenticate sources, and replay only the consumed weighted-1,
frozen and logical cells at the declared compatibility tolerance. Keep archived
weights/metrics and strict new-result replay. The primary screen requires
survival >= weighted-1, asymmetry MAE <= frozen and hop MAE <= weighted-1; it is
non-gating. Completion follows successful report replay. No coefficient search
or additional updates are allowed. CI must exercise a baseline NumPy CPU path
and retain the full evidence; preserve all older gates and historical validators.

## M4G: task quality versus inference budget

The M4G decision-component preflight follows the unchanged
`docs/experiments/task-quality-inference-budget.md` protocol. Run
`uv run python -m thermo_lab.quality_budget_preflight --output-dir results/quality-budget-preflight`
with a fresh destination. Require `component_preflight_complete`,
`full_m4g_ready=false`, zero fits and zero evaluation cells executed. This gate
authenticates three archives, checks 213 distinct roles, the planned 21-fit/60-cell
ledger, binomial coverage/inversion, correlated count decisions and nonmonotone
budget brackets. It must replay persisted evidence before reporting and write
completion last. Preserve literal float64 thresholds without an epsilon. This
component gate does not replace the matched runner, its training-law validation,
or the full-study preflight and release.

The separate M4G training-law component gate is
`uv run python -m thermo_lab.quality_budget_training_preflight --output-dir results/quality-budget-training-preflight`.
Use a fresh destination; require `training_law_component_complete`,
`full_m4g_ready=false`, zero fits, updates and study evaluation cells. Preserve
the existing three-site/two-operation fixture, all six M3 horizon contracts,
the equilibrium reference, independent law-specific occupancy/gradient roles,
and shared-occurrence reduction. Replay realized counts and gradient moments
independently at the versioned numerical tolerances; validate all persisted
evidence before reporting and write completion last. Keep the 42 validation
roles distinct from every study role. This gate does not certify the full
five-update runner or permit omission of its remaining integrity checks.


The M4G training-runner component gate is
`uv run python -m thermo_lab.quality_budget_runner_preflight --output-dir results/quality-budget-training-runner`.
Use a fresh destination; require `training_runner_component_complete`, 21
three-site fixture fits and 105 fixture updates, zero study fits/evaluation cells,
and `full_m4g_ready=false`. Authenticate all 21 production requests without
executing them. Preserve the separate 210-role diagnostic namespace, all seven
laws, five projected updates, fifth-checkpoint selection, exact stored gradient
identity and versioned numerical replay. Independently replay every evolving
fixture checkpoint and reject rehashed changes before reporting; completion is
last. A production bank must validate all 21 ordered fits before evaluation.
The held-out evaluator and integrated full-study preflight/review remain required
before production fitting. Do not interpret fixture success as task-quality,
inference-savings, or hardware evidence.

The integrated M4G gate is
`uv run python -m thermo_lab.quality_budget_full_preflight --output-dir results/quality-budget-full-preflight`.
It exercises all 21 fixture fits, 60 held-out fixture cells and 18 paired
comparisons; independently replay terminal counts, joined moments and killed
survival. Require `integrated_preflight_complete`, zero production fits/cells,
and separate code-bound statistical and implementation approvals before running
`thermo_lab.quality_budget_release`. Preserve the frozen protocol and all 213
production roles. Freeze all 21 fifth checkpoints before any held-out evaluation;
execute each distinct cell once per generation/replay phase with one PCG64
uniform vector per operation, including invalid-particle trajectories.
Persist all counts, histograms, paired moments, exact local errors and survival.
Use simultaneous acceptance bounds for decisions; U-statistic and paired
jackknife estimates remain descriptive. Report the complete finite-fit grid's
training cost and separate CPU work categories without device claims.

Production execution writes `execution.json` with final release pending. Require
independent statistical and implementation review of the resulting evidence,
all canonical repository gate commands successful, and complete persisted replay
before `finalize_release` writes `completion.json` last. Runtime provenance stays
outside scientific request identity but must validate and match its execution
record digest. Reject missing/stale review, provenance and gate records, changed
command arguments, rehashed numerical changes, and numeric type substitutions.
Negative scientific results do not fail an otherwise valid release.

## Exact return fixture

The separate exact return-fixture gate is
`uv run python -m thermo_lab.return_fixture_study --output-dir results/return-fixture-study`
with a fresh destination. Preserve the existing two-operation archive and
M4G requests. Require three sites, edges (0,1),(1,2),(0,1), K4, 64 complete
visible paths, nine shared parameters and zero samples. Check all three exact
gradients, six arms at 201 evaluator calls each, first-failure killed survival,
the path-objective difference and separately reported local fidelity. Validate
the complete persisted record before reporting; write completion last. The
study is exact-reference evidence on one small fixture, not full M4G quality,
inference savings or physical-hardware evidence.

## Return-fixture fidelity pilot

The distinct fidelity penalty pilot follows
`docs/experiments/return-fixture-fidelity-pilot.md`. Run
`uv run python -m thermo_lab.return_fixture_fidelity_pilot --output-dir results/return-fixture-fidelity-pilot`
with a fresh destination. Require four weights (0, 1, 10, 100), 201 exact
evaluator calls per arm, independently checked forward/reverse penalty
gradients, zero samples, complete persisted replay and completion written last.
Preserve the previous return-fixture study and its source hashes. Report
per-direction hop errors and survival together; pilot thresholds are not
full M4G quality or inference-savings evidence.

## Return-fixture complete-row pilot

The separate complete-row fidelity pilot follows
`docs/experiments/return-fixture-full-row-pilot.md`. Run
`uv run python -m thermo_lab.return_fixture_full_row_pilot --output-dir results/return-fixture-full-row-pilot`
with a fresh destination. Freeze all 16 visible input/output errors, weights
0/1/10/100, 201 calls per arm, the <=0.005 maximum entry error and >=0.95
survival gate before execution. Check every shared gradient, retain all
proposals and complete numerical replay before completion. Preserve older
fixture archives and keep this exact three-site result separate from M4G
quality, inference sampling and physical-hardware evidence.

## M4G survival-gradient audit

The saved M4G survival-gradient audit follows
`docs/experiments/survival-gradient-audit.md`. Run
`uv run python -m thermo_lab.survival_gradient_audit --output-dir results/survival-gradient-audit`
with a fresh destination. Require 21 fits, 126 checkpoints, 105 recorded updates,
zero new fits/samples, all seven small-fixture derivative checks and strict
persisted replay before reporting/completion. Authenticate the six pinned M4G
release files; never regenerate or tune training from these diagnostic results.
Keep own-training-law survival slopes separate from held-out task-quality claims.

## Exact fixture objective-by-step comparison

The exact fixture objective-by-step comparison follows
`docs/experiments/fixture-objective-step-comparison.md`. Run
`uv run python -m thermo_lab.fixture_objective_study --output-dir results/fixture-objective-study`
with a fresh destination. Require six arms, 201 evaluator calls per arm plus
19 preflight evaluations, zero samples and full persisted replay before
completion. Preserve K4, the two-operation shared fixture, caps [-2,2], and all
rejected proposals. Two path-aware objectives coincide on this fixture; never
report their agreement as independent evidence or claim full M4G quality.

## M4H: raised-cap path-KL screen

The raised-cap path-KL screen (M4H) follows
`docs/experiments/raised-cap-path-kl-screen.md`. Run
`uv run python -m thermo_lab.raised_cap_screen --output-dir results/raised-cap-path-kl-screen`
with a fresh destination. Require three arms (caps 2, 4 and 6), 37 groups,
20 predeclared starts per group, K4 fitting, zero samples, a passing preflight
and complete persisted replay before completion. Authenticate the pinned
local-trade-off and M4C archives by SHA-256. Keep the study's capped K law
bitwise equal to `finite_sweep_joint_law` inside [-2, 2] and leave the shared
evaluators unchanged, since archived fixture studies bind their hashes. Cap 2
is an ungated control; the exact screen is not sampled M4G quality,
inference-savings or hardware evidence, and the cap is a Thermo convention.
The full fit-and-replay takes about 25 minutes, so CI relies on the existing
unit-test job running `tests/unit/test_raised_cap_screen.py`, which pins the request and replays
every archived metric from the stored fits without refitting. Rerun the full
command locally whenever the runner or its bound sources change.

## M4I: one-feature kernel-capacity screen

The one-feature kernel-capacity screen (M4I) follows
`docs/experiments/kernel-capacity-screen.md` and closes the conservation line;
M5 is next regardless of its outcome. Run
`uv run python -m thermo_lab.kernel_capacity_screen --output-dir results/kernel-capacity-screen`
with a fresh destination (`--workers` sets fitting processes; the record must
not depend on it). Require four new arms (second hidden spin and
output-output coupling, each at caps 2 and 4), the M4H base comparators
replayed from the pinned archive, 21 predeclared starts per group, K4
fitting, zero samples, a passing preflight and complete persisted replay
before completion. Import M4H helpers rather than editing
`raised_cap_screen.py`. The families' spin-update counts are algorithmic, not
device operations; no hardware or Z1-connectivity claim follows. CI runs
`tests/unit/test_kernel_capacity_screen.py` through the unit-test job,
including the archived-evidence replay without refitting.

## M5a: exact meta-EBM cap baseline

Follow `docs/experiments/meta-ebm-cap-baseline.md`. Run the full CPU study with
`uv run python -m thermo_lab.meta_ebm_cap_baseline --output-dir results/meta-ebm-cap-baseline --workers 4 --fit-workers 8`
in a fresh directory. If interrupted, resume in that same directory with the
same command plus `--resume`; the runner authenticates its checkpoint against
the frozen request and implementation hashes. Require ten targets (both
readings, five seeds), nine
caps, both compile methods, all 180 chains, zero samples, passing integrity,
and full persisted replay before completion. `--workers` limits the dense
mixing and chain phases; `--fit-workers` only increases independent fitting
concurrency. The run log and status record explain caught failures and the last
known phase after an uncatchable stop. Checkpoints are recovery data, not
evidence. Autosave recovery coverage is
`uv run pytest tests/unit/test_meta_ebm_cap_baseline.py -k 'run_checkpoints_completed_work_and_resumes_without_repeating_it or resume_rejects_checkpoint_for_changed_request_before_work' -q`.
If final replay is interrupted, it restarts from the durable generated archive;
it never refits or loses generated results, but repeats replay computation.
Preserve the frozen scientific choices; a negative descriptive outcome is
valid evidence.

The recorded archive and report are under
`docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline/`. The ordinary unit
test `test_recorded_archive_pins_request_result_and_provenance` pins the
request and archive metadata, and
`test_recorded_archive_replays_every_fit_objective_without_refitting`
re-evaluates all 1,080 selected fit objectives from stored parameters. The
slow CI job runs
`test_recorded_archive_replays_stored_parameters_without_refitting`, which
replays the mixing scans and chain metrics for one target and its 18 chains
without new fits; the full ten-target replay was completed locally. A
recovered archive must retain its original generation provenance separately
from new replay provenance. Never reconstruct missing provenance by guessing.

The runner reads its own source hash from disk at start and again when it
writes the record, and spawned workers re-import it per phase. Don't switch
branches or edit the module in the checkout while a run is in progress.

## M5b: inner thermalization and precision sensitivity

The [approved protocol](experiments/meta-ebm-finite-thermalization.md) has a
runner and [bounded runtime calibration](research/2026-09-28-m5b-runtime.md).
The [full completion record](experiment-reports/2026-09-28-meta-ebm-thermalization/completion.json)
and [findings](experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md)
passed the gate: 80 references, 720 new cells, zero samples and all 800
persisted units replayed. The first attempt stopped without a terminal marker
after 688 generated units; resume retained them and completed the study.
Require the explicit inner update order, current-output initialization,
K>=1, lambda diagnostics, independent local enumeration, compatible-joint
control, adaptive large-K agreement and separate precision comparison.
Require all 80 source chains and 720 new cells, zero samples, passing
numerical integrity and full persisted replay before claiming completion.

Run the complete study from a clean, fixed checkout:

```bash
uv run python -m thermo_lab.meta_ebm_thermalization \
  --output-dir results/meta-ebm-thermalization --workers 3
uv run python -m thermo_lab.meta_ebm_thermalization \
  --output-dir results/meta-ebm-thermalization --workers 3 --resume
```

Use `--benchmark` with a distinct output directory for the single B/0,
variational, cap-1 source chain plus K=4 and 8-bit cells. Repeat the same
flags with `--resume` after interruption. Its marker is
`m5b_benchmark_complete`, which is never a production completion marker.
The full grid requires `meta_ebm_thermalization_complete`, 80 reference
chains, 720 new cells, zero samples, passing integrity and all units replayed.

The measured planning estimate is roughly 2–3 hours for generation plus
replay with three workers, or 5–6 hours with one, on the calibrated 8-CPU,
8-GiB session. These are estimates from one representative base chain, not
a full-grid timing guarantee. The runner caps concurrency at four and clamps
requests exceeding CPU/memory admission to the admitted count (minimum one),
logging the reason. Admission subtracts reclaimable inactive page cache from
cgroup usage, or uses MemAvailable when no cgroup limit exists. Every worker
has one BLAS thread. Numerical-integrity failures retain completed
work and stop; changing the numerical method requires a protocol amendment.

The real recovery exercise sent SIGTERM after the first reference autosave,
observed exit 130, resumed that directory and verified the saved reference
was unchanged, with all generation and persisted replay completed afterward.
Checkpoint each completed cell and replay unit using the shared persistence
helpers; reject changes to source, archive or request hashes on resume.
Finite-K transitions depend on the incoming output, so do not reuse M5a's
memoryless `sweep` implementation or edit its archived source to add support.
Selected replay tests belong in the existing CI jobs; the full grid stays a
local gate. The exploratory optimizer probe is not a completion gate and
must never replace the original M5a parameters.

## M5c: degree repair and placement on a synthetic offset lattice

The [frozen protocol](experiments/meta-ebm-synthetic-topology.md) covers the
primary arm only (reading B, variational, cap 1). Require the authenticated
M5b archive and its five pinned implementations, all 15 original chains
replayed against M5b, 11 J-only refits at `maxiter` 20,000, 30 new outer
cells (J-only prune and refit at K = 4, 32 and the limit), 60 placements
verified by site-level enumeration at the origin and in the packed patch,
five patches, zero topology violations, zero samples and full persisted
replay.

Run the study from a clean, fixed checkout into a fresh directory:

```bash
uv run python -m thermo_lab.meta_ebm_topology \
  --output-dir results/meta-ebm-topology --workers 3 --placement-workers 6
```

`completion.json` must show `meta_ebm_topology_complete`, 15 original
replays, 11 refits, 30 new outer cells, 60 placements, 5 patches, zero samples
and zero topology violations. `--benchmark` runs seed 0 only and publishes
`m5c_benchmark_complete`, which never satisfies this gate.

The seed-0 benchmark took 266 s (218 s generation, 48 s replay) on the 8-CPU
host. The [recorded full study](experiment-reports/2026-09-30-meta-ebm-topology/findings.md)
took 17.5 minutes (909 s generation, 141 s replay) and passed this gate. It is
below the 30-minute threshold and has no autosave layer; an interrupted run
starts again in a new directory. Replay recomputes every outer metric and checks
every stored layout exactly. It never refits and never re-solves a placement,
because time-limited solves can differ between machines. Never edit the five
M5b-pinned sources; M5c behavior lives in `meta_ebm_topology.py`. CI replays
a bounded subset of the committed archive; the full run stays a local gate.

## Exploratory fixed-budget sampling

**Portable entry point for the five sampling studies below:** see
[sampling portability](research/sampling-portability.md). Use
`python -m thermo_lab.sampling_portability <original_module_name>` to run on
hosts without readable cgroup metadata. New requests authenticate the adapter.
The fixed-budget numerical replay has a separate gate:
`portable-completion.json` must show
`status=fixed_budget_sampling_portable_replay_complete`, `cells_replayed=108`,
`float_atol=2e-12`, `float_rtol=0`, and `max_abs_difference <= float_atol`.
It preserves the original bitwise completion. The four later replay gates and
their declared tolerances are unchanged.

Read the [frozen protocol](experiments/fixed-budget-sampling.md). This is a
bounded exploratory comparison, separate from the M4/M5 milestone gates.
Use CPU and a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run --frozen python -m thermo_lab.fixed_budget_sampling \
  --output-dir results/fixed-budget-sampling-new
uv run --frozen pytest tests/unit/test_fixed_budget_sampling.py
```

The runner writes the frozen request before sampling, validates exact and
empirical fixtures, runs all 12 targets × three methods × three budgets, then
replays persisted evidence before writing `completion.json` last. Completion
must show `status=exploratory_fixed_budget_complete` and `cells_replayed=108`.
Scientific success is not an integrity gate. The 23-second recorded generation
is below the autosave threshold; interruption requires a new output directory.

For archive replay, extract the [report's](experiment-reports/2026-10-01-fixed-budget-sampling/findings.md)
`evidence.tar.gz` into a fresh directory and run the same module with `--replay`
and `--output-dir` pointing to the extracted `fixed-budget-sampling/` directory.
The unit test checks archive/member hashes and replays all 108 archived cells
without resampling in a subprocess with single-thread BLAS and JAX x64 disabled,
matching the recorded runtime. Strict bitwise replay is sensitive to numerical
reduction order; see the report's portability note. Replay authenticates the request, evaluator sources and
packed traces and reconstructs every exact reference, initialization, fixture
check, accuracy metric, work count and exchange rate. Historical trajectories
and timings are not regenerated. Preserve the hash-bound runner and helpers.

## Exploratory sampling time to accuracy and posterior transfer

Follow the [frozen protocol](experiments/sampling-time-to-accuracy.md). Use a
fresh output directory and keep other tests/studies idle during timing:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run --frozen python -m thermo_lab.sampling_time_to_accuracy \
  --output-dir results/sampling-time-new
uv run --frozen pytest tests/unit/test_sampling_time_to_accuracy.py
```

The full grid has 18 targets, five budgets and 330 cells: three sampling arms
everywhere, with two analytic global-flip estimators on six zero-field graphs.
Completion requires `status=exploratory_sampling_time_complete`,
`cells_replayed=330`, and `decisions_replayed=66`. Qualification failures are
scientific results, not integrity failures. Count all replica redraws and
exchange work, and include exchanges, host transfer and estimation in measured
warm batch time. Keep compilation and initialization separate. This short
exploratory run has no checkpoint layer; restart interrupted generation in a
fresh directory.

Replay with `--replay` and the extracted evidence directory checks source,
request and trace hashes; exact/empirical fixtures; initialization; all exact
references, estimates, errors, work counts and exchange rates; timing medians;
and sustained threshold decisions. It does not recreate wall-clock times or
rerun generation-only prefix/repeat checks. Float-derived values use the fixed
absolute tolerance 2e-12, while hashes, identities, counts and decision outcomes
remain exact. The unchanged October 1 sampler is imported, not edited.

## Exploratory fresh-seed symmetry plus tempering

Follow the [frozen protocol](experiments/symmetry-tempering.md), keeping other
tests and studies idle during measurement:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run --frozen python -m thermo_lab.symmetry_tempering \
  --output-dir results/symmetry-tempering-new
uv run --frozen pytest tests/unit/test_symmetry_tempering.py
```

Six fresh zero-field graphs, four arms, five budgets and 16 independent trials
produce 120 cells and 24 decisions. Completion requires
`status=exploratory_symmetry_tempering_complete`, `cells_replayed=120` and
`decisions_replayed=24`. Both tempering arms pay all five replicas' redraws and
exchange attempts and retain only cold samples. Qualification failures remain
scientific results. Warm timing includes the analytic estimator; compilation
and initialization are separate. No hardware claim follows.

Replay with `--replay` on the extracted directory authenticates sources,
request and traces, then recomputes references, estimates, work, exchange rates,
timing medians and all sustained-threshold decisions. It does not reproduce
historical timings or rerun generation-only prefix/repeat checks. Use the
inherited fixed numeric tolerance 2e-12 and exact hashes/discrete values.
Generation is short and has no checkpoint layer; restart an interrupted run
in a fresh directory. Preserve all earlier hash-bound sources unchanged.

## Changing evidence and causal restart policy

Use the [frozen protocol](experiments/changing-evidence.md), a fresh directory,
CPU, single-thread BLAS and JAX x64 disabled:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run --frozen python -m thermo_lab.changing_evidence \
  --output-dir results/changing-evidence-new
uv run --frozen pytest tests/unit/test_changing_evidence.py
```

Complete production evidence requires `status=changing_evidence_complete`,
`cells_replayed=416` and `query_estimates_replayed=83200`. Development and
held-out streams are disjoint; fit the predictor and persist its coefficients
before held-out generation. Test the frozen policy at the same sampling budget
as its fixed baselines. Negative predictive or policy results are scientific
outcomes, not integrity failures. Count initialization and monitoring overhead
in the warm query pipeline, and report compilation/model setup separately.

Replay with `--replay` authenticates all artifacts and sources, reconstructs
exact references and all per-query estimates/errors/decisions/work counts,
checks state continuity or seeded reset initialization, refits predictors on
the saved development data, and checks held-out predictions. Timing medians
are reconstructed from raw observations; historical timing is not reproduced.
All prior evaluators remain unchanged. The one-condition eight-stream runtime
calibration took 10.03 seconds for 26 cells plus 0.59 seconds for replay,
well below the 30-minute checkpoint requirement after scaling to 16 conditions.
The short production study has no restart layer; interrupted generation uses
a fresh directory. Neither generation nor verification has a wall-clock cutoff.

## Conditional estimates on identical trajectories

Use the [frozen protocol](experiments/conditional-estimation.md), a fresh output
directory, CPU and single-thread BLAS. Keep other studies/tests idle while
measuring timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run --frozen python -m thermo_lab.conditional_estimation \
  --output-dir results/conditional-estimation-new
uv run --frozen pytest tests/unit/test_conditional_estimation.py
```

Completion requires `status=conditional_estimation_complete`,
`trajectory_cells_replayed=96`, `estimator_cells_replayed=192` and
`query_estimates_replayed=76800`. Sixteen independent seeds pair both estimators
on identical trajectories; every timed repeat must be bitwise identical to its
warm-up. Compare errors and estimator-inclusive CPU costs, with no outcome
threshold gating completion. Separate synthetic decision utility from average
probability error and physical-hardware evidence.

Replay with `--replay` authenticates sources and traces, checks grid identities,
initialization/continuity, and reconstructs references, both estimators,
metrics, work counts and timing medians. It does not regenerate historical
trajectories or timings. Exact enumeration and zero-coupling tests independently
check the conditional-expectation formulas; archive replay is part of the unit
gate. All earlier and newly archived evaluators remain immutable.

A six-cell, sixteen-stream calibration took 7.85 seconds plus 0.34 seconds
replay. Production took 75.90 seconds plus 6.02 seconds replay, below the
30-minute checkpoint threshold. Restart interrupted generation in a fresh
directory. Neither generation nor verification has a wall-clock cutoff.

## THRML finite-sweep contract (E0 / A1)

Use the [frozen protocol](experiments/thrml-finite-sweep-contract.md), CPU
only, a fresh output directory:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python -m thermo_lab.thrml_finite_sweep_contract \
  --output-dir results/thrml-finite-sweep-contract-new
uv run pytest tests/unit/test_thrml_finite_sweep_contract.py
```

Completion requires `status=thrml_finite_sweep_contract_complete`, `cells=64`,
`chains_per_cell=400000`, `controls_gate_passed=true` and `replayed=true`. The
runner stops before any THRML call if a gated negative control does not
separate on the exact side. A failed cell is a scientific result, not an
integrity failure. The run took 76 s including compilation, so there is no
autosave or resume layer; restart in a fresh directory. CI replays the
committed archive (`--replay`) through the unit test; it does not resample.
