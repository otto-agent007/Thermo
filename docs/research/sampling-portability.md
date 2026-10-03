# Portable execution of the frozen sampling studies

PR #68's first GitHub run exposed two host assumptions that the cloud run did
not: five generators read `/sys/fs/cgroup/cpu.max` and `memory.max` without
handling unavailable files, and the first study's replay compares derived
float64 metrics bitwise. Its archived replay fails locally too when OpenBLAS
selects the Haswell CPU kernel, despite single-thread settings. The portable
replay measured a maximum absolute difference of 1.11e-16 in that probe.

The original five evaluators, protocols, archives and historical completion
records remain unchanged. `thermo_lab.sampling_portability` supplies an
explicit adapter for new runs and a separate numerical audit for the first
archive. It changes no sampler, model, seed, metric, work count or timing scope.

## New runs

Use the adapter's positional study name instead of the original module entry
point, with a fresh output directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=false
uv run python -m thermo_lab.sampling_portability changing_evidence \
  --output-dir results/changing-evidence-portable
```

The available names are `fixed_budget_sampling`, `sampling_time_to_accuracy`,
`symmetry_tempering`, `changing_evidence` and `conditional_estimation`.

The adapter executes the original `run_study` code with a private copy of its
globals, replacing only its `Path` lookup. All numerical helpers are the
original objects; no module globals or process-wide filesystem functions are
modified. Only an `OSError` reading those two exact cgroup paths produces
`"unavailable"`. Readable values are preserved. Missing evidence, unwritable
outputs and other filesystem errors still fail. Unavailable means unknown,
not unlimited or a guessed resource allocation.

New requests identify the adapter and hash/snapshot its source alongside the
frozen evaluator. Caller-supplied requests are copied before augmentation.
Observed metadata stays in runtime provenance and does not enter the requested
input digest. The old direct generator commands retain their original host
requirements.

## Replay

After extracting an archive, use the same entry point with `--replay`:

```bash
uv run python -m thermo_lab.sampling_portability fixed_budget_sampling \
  --output-dir results/fixed-budget-sampling --replay
```

For the four later studies this invokes their unchanged replay functions and
existing declared tolerances. For fixed-budget sampling it runs a separate
auditor using the frozen scientific functions. It retains every source,
request and trace hash check; seeded initialization and shape checks; fixture
thresholds; and recomputation of all metrics, exact references and work counts.
Structures, types, integers, booleans and strings must match exactly. Finite
floating values use **absolute tolerance 2e-12 and relative tolerance zero**,
matching the later time-to-accuracy study. Nonfinite values on either side fail.

This audit writes `portable-completion.json` with status
`fixed_budget_sampling_portable_replay_complete`, the verifier's source hash,
input artifact hashes, tolerance, maximum observed difference and checked cell
count. It explicitly says `numerical_not_bitwise`. It never overwrites
`completion.json`; the original bitwise replay remains available on a matching
numerical environment. Portable numerical agreement is not a claim of bitwise
reproducibility, fresh sampling or reproduced historical timings.

Regression coverage simulates unavailable metadata for every study, exercises
the Haswell BLAS kernel only on CPUs with AVX2/FMA support, and rejects changed
metrics, integer work counts, traces, source files and nonfinite values. The
archive test checks that historical completion bytes remain unchanged.
