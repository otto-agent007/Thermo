# CLAUDE.md

@AGENTS.md

AGENTS.md above holds the research rules, the base gate commands and an index
of study gates. The full per-study gate requirements are in
`docs/release-gates.md`, and per-study descriptions are in `docs/studies.md`.
This file adds the current state of the project and the habits that keep the
work pointed at the right problem.

## Where the project stands (updated 2026-09-25)

- The conservation line (M4B–M4I) is **closed** as of PR #49. M4H and M4I
  confirmed the diagnosis in `docs/research/2026-09-23-cap-leakage-analysis.md`:
  the ±2 field/coupling cap chosen on Sep 1 put a ceiling of about
  `L00 · L01 ≈ e^(-4c)` on leakage. Raising the cap and adding one kernel
  feature brought S(500) to 89.6%, still short of 95%. Don't propose more
  conservation pilots unless the owner reopens the line.
- **M5 (topology-aware meta-EBM)** is next: `docs/experiments/topology-aware-meta-ebm.md`.
- Charter tracks that haven't started yet (`PROJECT_CHARTER.md`): native THRML
  algorithms (Potts, associative memory, Max-Cut), and Torx Bayesian and
  state-space inference. When you're choosing what comes after M5, look here
  first rather than extending M4.
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
- Run on CPU only. The owner declined GPU work on the local GTX 1050 Ti.

## Commands

```bash
uv sync --frozen
uv run ruff format . && uv run ruff check .
uv run pytest tests/unit -m "not slow"       # what CI's unit-rest job runs
uv run pytest tests/unit/test_<module>.py    # iterate on one module first
```

The full `uv run pytest` and the gate list in AGENTS.md are slow. Run the
targeted tests while iterating, then the full gates for anything you touched
before you open a PR. Python is pinned to 3.11 and packages live under
`src/thermo_lab` (never `import thermo`).

## Git

Work on a branch, never directly on `main`, and open PRs against `main`.
Commit or push only when asked.
