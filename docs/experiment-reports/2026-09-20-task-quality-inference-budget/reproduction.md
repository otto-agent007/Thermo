# M4G reproduction and evidence boundaries

The protocol remains byte-identical to commit
`ae2ecac3be45501d52ac60113ee6ccc71e6adc0c` (Git blob
`8633d5643e7ea5f3dcc29df1e085dc9eb5af3f1b`). The implementation and tests were
frozen before production under digest
`sha256:0eaf7344c1c27a7e5c3964610f5eee2fd38d629ef0c0b22655d817bb78e952f4`.
This digest covers every Python source/test, experiment configuration, frozen
dependency files and protocol; it excludes observed results, timings and prose.

## Prerequisites and study

From this repository with the frozen environment, use fresh output directories:

```bash
uv sync --frozen
uv run python -m thermo_lab.quality_budget_full_preflight --output-dir results/m4g-preflight
uv run python -m thermo_lab.quality_budget_release \
  --preflight-dir results/m4g-preflight \
  --review-record docs/experiment-reports/2026-09-20-task-quality-inference-budget/integrated-preflight/review.json \
  --output-dir results/m4g-reproduction
```

The checked preproduction review is valid only for its exact implementation and
preflight digest. Different code or numerical preflight evidence requires fresh
independent statistical and implementation review before fitting. No production
roles are consumed by the preflight. Diagnostic training and evaluation streams
are separate from all 213 production roles.

The generation phase freezes 21 fits before sampling the 60 distinct held-out
cells. Each cell uses 32,768 complete 500-operation trajectories, including
invalid-particle paths. The 18 paired comparisons reuse those realized cells.
Persisted replay is another execution phase, not an additional statistical
replication. Internally generated immutable training summaries may be cached;
externally supplied results are never trusted as cached expectations.

## Final release

`execution.json` is an execution marker with final release pending. Obtain two
independent reviews of the production evidence, then record every required
repository gate's actual command, zero return code and elapsed seconds. The
canonical command plan checks configs, seeds, test partitions and packaging;
only output/source locations and CPU affinity are operational variations.

```python
from thermo_lab.quality_budget_release import finalize_release, validate_release

finalize_release(
    "results/m4g-reproduction",
    "path/to/production-evidence-review.json",
    "path/to/repository-gates.json",
)
validate_release("results/m4g-reproduction")
```

The finalizer numerically replays all persisted scientific evidence, validates
runtime provenance, reports and CSV, then copies/reloads the post-study review
and gate records. It writes `completion.json` last. A negative scientific
finding is compatible with successful integrity checks. A completed release
cannot be silently overwritten.

## Runtime accounting and plots

`provenance.json` records CPU wall-time spans separately for prerequisite checks,
generation, and persisted replay/reporting. Inclusive spans overlap. Exclusive
spans subtract instrumented children and can be summed within a phase, but omit
uninstrumented top-level dispatch. Exact table/feature construction, stochastic
sampling, source authentication, numerical replay, report/CSV generation and I/O
are separate categories. Provenance collection and final metadata I/O are
excluded. CPU affinity and competing validation work affect these observed
software timings; they are not physical device latency, energy or power.

The scientific ledger counts the complete six-fit finite grid per seed. It
includes main gradient endpoints and independent non-propagated references.
Equilibrium work is retained with a null finite-sweep count, never zero cost.
Every inference cell retains the same independent sample count.

`plot_quality.py` reads the archived `study.json` and writes the SVG with ordinary
Matplotlib. Lines connect tested K values only. The figure does not assert
monotonicity or interpolate a passing budget. Oracle diagnostics remain in the
CSV/report. All three seeds are displayed individually.
