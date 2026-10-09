# CLAUDE.md

@AGENTS.md

AGENTS.md above holds the research rules, the base gate commands and an index
of study gates. The full per-study gate requirements are in
`docs/release-gates.md`, and per-study descriptions are in `docs/studies.md`.
This file adds the current state of the project and the habits that keep the
work pointed at the right problem.

## Where the project stands (updated 2026-10-09)

- **Charter amendment 1** (`PROJECT_CHARTER.md`, "Amendments", merged in
  PR #101 on 2026-10-09) sets which questions come first. The standing
  question is what a fixed-topology, low-precision, I/O-expensive p-bit array
  can do usefully, at what task quality and projected cost, against a
  conventional machine doing the same task. The order of work is a
  hardware-constraint layer (queue row `constraint-layer`), then one task (a
  small chained-EBM denoiser, `dtm-small-image`; associative-memory recall is
  second), with the conventional baseline inside the same row. Sampler
  comparisons on random instances are now probes, and the meta-EBM line
  stays at contract checks and placement. A recorded study must name, in its
  protocol's first paragraph, a number someone will design against;
  anything else is a probe under `docs/research/`. Research rules 8 to 10:
  run under the declared hardware constraints by default and say which a
  study relaxed; state the resource accounting with every comparison; where
  a task exists, report task-level error per unit of projected cost.
- The conservation line (M4B–M4I) is **closed** as of PR #49. M4H and M4I
  confirmed the diagnosis in `docs/research/2026-09-23-cap-leakage-analysis.md`:
  the ±2 field/coupling cap chosen on Sep 1 put a ceiling of about
  `L00 · L01 ≈ e^(-4c)` on leakage. Raising the cap and adding one kernel
  feature brought S(500) to 89.6%, still short of 95%. Don't propose more
  conservation pilots unless the owner reopens the line.
- **M5 (topology-aware meta-EBM)** is under way:
  `docs/experiments/topology-aware-meta-ebm.md`. Its first stage, the M5a
  fully connected cap baseline, is recorded in
  `docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline/`. Later stages
  start with the approved M5b protocol in
  `docs/experiments/meta-ebm-finite-thermalization.md`: inner hidden/output
  sweeps and a separate rounding comparison, retaining M5a parameters.
  The runner is implemented with per-cell/replay autosave and bounded runtime
  calibration in `docs/research/2026-09-28-m5b-runtime.md`. The full study and
  persisted replay are recorded in
  `docs/experiment-reports/2026-09-28-meta-ebm-thermalization/`.
  M5c (`docs/experiments/meta-ebm-synthetic-topology.md`) is recorded in
  `docs/experiment-reports/2026-09-30-meta-ebm-topology/`: a J-only degree
  refit plus exact placement on a synthetic offset lattice, at about 1.5×
  the parity-bound input copies. Complete execution costs remain open.
  **Reading B** of the meta-EBM energy prefactors is the standing choice for
  all M5-derived work (owner decision 2026-10-03, recorded in
  `docs/knowledge/lessons.md`); don't run reading A arms unless the question
  is about the literal appendix formula.
- Native algorithm/inference exploration has started: the October 1
  Max-Cut, restart-greedy and denoising pilots are preserved under
  `docs/experiment-reports/2026-10-01-exploratory-pilots/`. The fresh-seed
  fixed-budget sampling comparison is under
  `docs/experiment-reports/2026-10-01-fixed-budget-sampling/`: tempering improved
  probability accuracy per total redraw. The October 2 follow-up under
  `docs/experiment-reports/2026-10-02-sampling-time-to-accuracy/` measures warm
  execution costs and changes the recommendation: use symmetry when available,
  independent Gibbs for the tested denoising posteriors, and tempering for
  difficult biased graphs. A post-hoc symmetry-plus-tempering estimator passes
  all six zero-field cases. Its fresh-seed timing is now recorded under
  `docs/experiment-reports/2026-10-02-symmetry-tempering/`: the combination
  qualifies 6/6, symmetry-aware ordinary baselines 5/6, and unaugmented
  tempering 2/6. Against the best qualifying ordinary baseline it is
  4.05–12.33x faster on four targets, close on one, and alone qualifies on one.
  All 120 cells and 24 decisions replayed. The changing-evidence follow-up in
  `docs/experiment-reports/2026-10-02-changing-evidence/` completed 416 cells
  and 83,200 query estimates with full replay. Retention saves initialization
  work but can add history dependence. State-aware failure prediction improves
  held-out ranking, but its equal-work restart policy does not establish an
  improvement over retaining state. Exact enumeration wins the measured CPU
  timings at 12 spins. A post-hoc iid reference identifies substantial finite-
  sample estimation noise. Future policy work should predict intervention
  benefit and use fresh held-out streams; do not tune against the current test
  seeds. These study runners and the original sampler are hash-bound.
  The fresh-seed conditional-estimation follow-up in
  `docs/experiment-reports/2026-10-02-conditional-estimation/` replays 96 shared
  trajectory cells / 192 estimator cells. At T=16, conditional averaging lowers
  marginal error by 26.1% (Gibbs) and 29.9% (tempering), with median paired CPU
  overhead 26.3%/7.4%. Weak-regime gains are 56–64%; strong-checkerboard gains
  only 4–5%. Gibbs alarm-decision regret worsens despite better probability
  error. Use conditional estimates as an inference baseline, retain separate
  utility checks, and test any new mechanism on fresh seeds. This evaluator
  and protocol are also hash-bound.
  Other charter directions (`PROJECT_CHARTER.md`),
  including Potts, associative memory and Torx state-space inference, remain
  open. Prefer these tracks to extending M4.
- **E0 (THRML finite-sweep contract)** is complete in two stages. Stage A
  (`docs/experiment-reports/2026-10-03-thrml-finite-sweep-contract/`) shows
  THRML 0.1.4's block-Gibbs sweep matches the exact p0 T^K law on a five-spin
  chain in 64/64 cells. Stage B
  (`docs/experiment-reports/2026-10-03-thrml-m5a-kernel-inner-sweep/`) runs one
  compiled M5a kernel (reading B, variational, cap 1, seed 0, site 1) and
  matches the archived M5b inner-K law in 6/6 cells, so M5b's finite-K
  matrices describe what THRML executes for that site. THRML's `hinton_init`
  draws sigmoid(β b), a heuristic; model it as such. The other 59 kernels, a
  GPU float32 arm (A2) and execution costs remain open. Both are CPU software
  evidence against exact references; CI replays the archives without
  resampling.
- **The Potts track** (charter track A) has started. Stage A
  (`docs/experiment-reports/2026-10-06-thrml-potts-contract/`) shows THRML's
  categorical Gibbs sampler matches the exact p0 T^K law of a three-label,
  three-colour Potts patch in 84/84 cells, including both categorical factor
  classes, clamping and a q = 2 encoding of E0's chain. THRML's categorical
  conditional is softmax(beta * local field), with no factor of 2, unlike
  its spin sampler. Stage B
  (`docs/experiment-reports/2026-10-06-potts-symmetry-tempering/`) finds the
  Ising symmetry-plus-tempering result carries over to cold (beta 16)
  antiferromagnetic Potts targets, 6/6 against 5/6 and 4-16x smaller budget
  on three, but not to beta 8, where symmetry-aware ordinary Gibbs wins 4/6
  because tempering keeps only its cold replica. Stage C
  (`docs/experiment-reports/2026-10-06-potts-trapping-policy/`) is negative:
  a pilot-based switching policy detects trapping (AUC 0.81) but has regret
  8 against 3 for always-tempering, so default to tempering plus symmetry at
  budgets of 1024 and above. It is the second adaptive policy (after the
  changing-evidence restart policy) that fails to beat a simple baseline;
  don't design a third without the owner.
- **Associative memory** (charter track A) has its first study
  (`docs/experiment-reports/2026-10-07-am-binary-emulation/`). On pairwise
  binary units, ordinary binary hidden units with a negative bias stand in for
  a dense memory's categorical hidden unit: within 0.01 of its equilibrium recall
  at coupling range 14 to 26. One-hot inhibition is range-hungry, and
  domain-wall chains don't mix. Stage A2
  (`docs/experiment-reports/2026-10-07-am-categorical-reference/`) reran the
  sampled categorical reference label-first: it helps only at 4 to 16 sweeps.
  The reference barely mixes at the cold β (16) its equilibrium needs, and the
  bias design still beats it within budgets (16 of 24 pairs, never short).
  Check a reference arm's own dynamics before trusting a comparison with it,
  and test a proposed cause on study-scale data before writing it up. A
  cue-informed start is the open variant; it waits for the owner. Source cards
  for the area are in `docs/knowledge/sources/`.
  The coupling-bits study
  (`docs/experiment-reports/2026-10-08-am-coupling-bits/`) programs the bias
  design through the M5b codebook: 6 bits keep exact equilibrium recall within
  0.01 when the step equals the coupling J, against 10 with separate coupling
  and field caps and 12 with one full-scale cap. Encode its fields relative to
  J. Static ±10% per-site beta jitter is negligible in every cell.
- **Exchange cost projection**
  (`docs/experiment-reports/2026-10-07-exchange-cost-projection/`) prices
  replica exchange in the sealed Z1 Appendix-B model, where an accepted swap is
  a full SRAM write of both replicas (about 21,665 Gibbs updates per p-bit).
  Tempering plus symmetry never wins on projected energy at any tested
  exchange interval (475x to 7,459x the cheapest ordinary baseline when
  exchanging every sweep; 117x to 1,886x at every 4 sweeps, which keeps the
  same qualifying budgets on 5/6 targets; intervals of 16 and above lose the
  sweep-time advantage). It keeps a 4x to 16x elapsed-sweep advantage on four
  targets and is the only qualifying arm on one. The recorded "default to
  tempering" recommendations are sweep-budget recommendations; say which
  resource a target is bound by before repeating them. New tempering arms
  should default to an exchange every 4 sweeps. A zero-write per-replica
  temperature control is the hardware feature that would reconcile the two
  views. The cost model excludes host latency, so sweep time is a lower bound.
- **Planar Ising scaling**
  (`docs/experiment-reports/2026-10-07-planar-ising-scaling/`) is the first
  sampling study outside exact enumeration: open grids of 64, 256 and 1024
  spins with an exact Kac-Ward reference (`planar_ising_scaling.kac_ward`,
  valid for any planar zero-field graph; reuse it before writing another
  enumerator). Past 64 spins only tempering qualifies: five cold chains and a
  long chain never reach the threshold at 256 or 1024 spins within 4096
  sweeps, every tempering arm does at 256 to 1024 sweeps. Mixed-sign grids at
  beta 4 are out of reach for every arm at 256 spins and above (pre-registered;
  a temperature and budget statement, not a sampler one). The archived
  five-replica ladder's exchange acceptance collapses with size (0.38 to 0.018
  on the cold pair); do not reuse it above 64 spins without an N-scaled ladder.
  With the thin ladder, k = 4 cost two of three 1024-spin seeds a budget step,
  so the k = 4 default is for n <= 16 until re-tested with a scaled ladder.
  The October 8 probe (`docs/research/2026-10-08-ladder-probe.md`,
  exploration only) found that a ladder needs 25 to 33 replicas at 1024 spins
  to recover acceptance, that where it works (ferro) the thin ladder already
  qualified, and that where qualification is missing (mixed, beta 4) a working
  ladder cuts the error by 40 percent and then plateaus; at 100k sweeps the
  nine-replica arm reaches 0.077 and nothing qualifies, while a 100k-sweep
  anneal matches the exact energy per spin within 0.6 percent. Treat the
  scaled-ladder study as closed by that probe unless the owner wants it
  recorded, and treat frustrated grids at beta 4 as an optimization target
  (energy per spin, annealing arms) rather than a sampling target.
- **Planar annealing**
  (`docs/experiment-reports/2026-10-08-planar-annealing/`) is that
  optimization study: mixed-sign open grids of 64, 256 and 576 spins against
  exact transfer-matrix ground states (`planar_annealing.ground_state_energy`,
  max-plus over 2^L window states, L <= 24; `thermal_energy` gives the exact
  q(beta) at any beta where Kac-Ward is ill-conditioned). Four parallel
  annealed restarts are the best arm at every size and budget and qualify at
  1e-3 per spin on all 64-spin targets, but the advantage shrinks from 4x to
  23x at 64 spins to 1.3x to 2.2x at 256 and 576, where no arm gets within
  1e-3 of the ground state within 65,536 sweeps (pre-registered negative).
  Annealing to beta 8 or 16 makes no difference: the gap is trapping during
  the ramp. The nine-replica ladder at cold beta 4 reports its own thermal
  offset, not a search result. At equal p-bit updates the restarts win only
  at 64 spins or at the largest budget, so say which accounting (elapsed
  sweeps or updates) a restart claim uses. Its open follow-ups (schedule
  shape, a planar matching solver for L = 32, an N-scaled cold ladder) stay
  notes, not queue rows, under charter amendment 1.
- **Planar 16-offset ferro** (`docs/experiment-reports/2026-10-08-planar-16-offset-ferro/`,
  loop P-0001): #92 holds on a maximal planar subgraph of the Z1 16-offset
  rule (ferro, beta 4, about 42 percent long edges, paired grids). Only
  tempering qualifies at 256 and 1024 spins within 4096 sweeps; the
  nine-replica ladder keeps 256 and 1024 sweeps on every target. The thin
  five-replica ladder loses more than tenfold cold-end acceptance on the long
  edges and qualifies at 1024 spins with no margin; the probe's predicted slip
  did not reproduce. Says nothing about the non-planar degree-16 lattice.
- **The improvement harness is parked** (2026-10-07,
  `docs/improvement-harness.md`). It has never been run, and its research track
  only tunes a fixture from the closed conservation line. Don't extend it or
  treat it as the self-improvement plan.
- **The research loop** (`docs/research-loop.md`, started 2026-10-08) is the
  self-improvement plan. THERMES takes the top open row of the owner-curated
  `docs/research-queue.md`, probes it, drafts a protocol PR and waits for the
  owner's `approve` on Discord before running; the owner's merge accepts
  results. Add or reorder queue rows only when the owner asks, and don't start
  work on a row the loop has claimed (its branch is `research/<row>`). A
  director (`thermes-director`, Fable 5.1) reviews the direction after each
  accepted study and weekly, as a docs-only PR under `docs/research/`; its
  proposed queue changes are advice the owner adopts or not.
- `docs/roadmap.md` has the one-row-per-milestone status table. Keep it current.

## Check assumptions before building process

A review on 2026-09-24 found that roughly two weeks of preflights, pilots and
audits had gone into a quantity that one fixed parameter choice capped
structurally. Nobody noticed, because each new layer took the layer below it
as given. To avoid a repeat:

- **Before you design a study, list its fixed choices** (caps, model family,
  horizon, kernel features, schedule) and ask whether any of them bounds the
  target metric. A small exploratory probe (like `docs/research/cap_probe.py`)
  that runs in minutes is worth more than another preflight layer. Label its
  output as exploration and never present it as recorded evidence.
- **When two consecutive milestones fail the same way, stop and ask the owner**
  whether the premise is wrong. Don't design a third variant on your own.
- **Scale the process to the question.** The default is one frozen protocol,
  one runner, one replay test and one report. Add component, training-law or
  runner preflights only when the owner asks for them or the study really
  costs hours. M4G's five-stage chain is not the template.
- Say so plainly when a result is negative or a gate is structurally
  unreachable. Don't soften it into "inconclusive".

## Engineering habits

- **Reuse the shared provenance layer**: `hashing.py` (`canonical_json`,
  `canonical_sha256`), `records.py`, `provenance.py`, `persistence.py`
  (`atomic_write_text`), `evidence.py`. Don't write another per-study
  digest, record or rendering stack. If a new study needs a helper that an old
  study owns, import it. Extract it into a shared module only when that
  doesn't change any archived source hash. The existing per-study
  `run_study`/`render_report` copies stay as they are, because each study
  pins its own source bytes and a refactor would break archive validation.
- **Never edit hash-bound evaluators**, e.g. `finite_sweep_gradients.py`,
  `conservation_diagnostic.py` or `raised_cap_screen.py`. Archived studies pin
  their SHA-256, so an edit breaks archive validation. Put new behavior in
  study-local code and add a bitwise-equality test against the old evaluator.
  Shared modules can be pinned too: the changing-evidence and
  conditional-estimation archives pin `provenance.py` and `records.py`, so
  upstream release pins live in `release_pins.py`.
- **CI stays fast.** Don't add a GitHub workflow per study: the `unit-rest`
  job already runs `pytest tests/unit -m "not slow"`. A short exact study
  (about a minute) can join the `fixture-study` matrix in
  `.github/workflows/scientific.yml`. For a long study, CI runs a unit test
  that pins the request and replays archived metrics without refitting, and
  the full run stays a local gate.
- **Keep evidence small.** Gzip new archives (`*.json.gz`, as the recent
  studies do) and keep them bounded. Don't commit another multi-megabyte
  uncompressed JSON; if a study genuinely needs one, ask the owner first.
- **README is a one-page status**, not a caveat log. Add a study's
  description to `docs/studies.md`, its gate requirements to
  `docs/release-gates.md` with one index row in AGENTS.md, and its result to a
  report under `docs/experiment-reports/` linked from the roadmap.
- CPU remains the default for gates and the THERMES scheduler. The owner
  approved local GTX 1050 Ti experiments and tests on October 3, 2026 under
  the compatibility, provenance and validation conditions in
  [docs/environments.md](docs/environments.md#gpu-use). Keep exact enumerators
  and archived replays on CPU; GPU software remains simulation evidence.

## Commands

```bash
uv sync --frozen
uv run ruff format . && uv run ruff check .
uv run pytest tests/unit -m "not slow"       # the tests CI's unit-rest job runs
uv run pytest tests/unit/test_<module>.py    # iterate on one module first
```

CI splits the long test runs into balanced jobs and runs each on all runner
cores with `uv run --with pytest-xdist==3.8.0 --with execnet==2.1.2 pytest ...
-n auto` (see the comment on the `test` matrix in `scientific.yml`), which
works locally too. Never add pytest-xdist (or anything else) to
`pyproject.toml` for this: archived studies pin `uv.lock` by SHA-256.

The full `uv run pytest` and the gate list in AGENTS.md are slow. Run the
targeted tests while iterating, then the full gates for anything you touched
before you open a PR. Python is pinned to 3.11 and packages live under
`src/thermo_lab` (never `import thermo`).

## Git

Work on a branch, never directly on `main`, and open PRs against `main`.
Commit or push only when asked.
