# QEC work inventory and reproduction sources

Start with the [consolidated research record](../2026-11-14-qec-research-record.md).
This directory preserves the recent owner-directed QEC sequence performed outside
the live Thermo checkout. It is a retrospective import, not a claim that all
original standalone tools have been integrated into `thermo_lab`.

## What is in the repository

- `reports/*.txt`: byte-preserving original report/audit snapshots. They retain
  historical tense, formatting and relative references; those refer to their
  original packages, not necessarily this repository. The current interpretation
  and corrected model labels are in the consolidated record.
- `source-records/**/*.txt`: byte-preserving middle-study reports, protocols,
  audits, execution-repair records and bounded JSON summaries. Each retains its
  original filename plus `.txt`; historical status is not current execution state.
- `original-archives.json`: exact full-archive filenames, sizes and SHA-256 values
  for retrieving the original reproduction sources from the owner’s Library.
- `plans/*.txt`: three original uncommitted planning documents, retained as
  history. A planning-time “held/not run” label does not override later approvals
  or results. The observable-specific sampler-reliability plan was not executed.
- `imported-source-hashes.json`: SHA-256 and source-member mapping for directly
  imported snapshots. Do not silently edit these snapshots.
- `archives/precision-diagnostic.tar.gz`: the unmodified complete 89,036-byte
  latest tiny-model diagnostic, including source, protocol, fixtures, outputs,
  receipts and its independent audit. The adjacent JSON authenticates the original
  compressed bytes.
- `tests/unit/test_qec_precision_archive.py`: a portable independent replay of
  exact fixture distributions, every saved interval and synthetic proxy counts.
  It reads the archive in place without extraction or executing archived code.

The source-only prototype inside the archive is not a new installed decoder.
Its original manifest contains absolute paths and pins the original Python
executable. Recreating its exact worker-supervisor run elsewhere needs a new,
separately identified manifest; never rewrite the archived manifest to hide that
portability limitation. The numerical evidence replay below does not depend on
those paths or original interpreter bytes.

```bash
uv run pytest tests/unit/test_qec_precision_archive.py
```

This enumerates no more than 1,024 fault vectors per model. It checks all 22
models, 976 mass intervals and 450 available posterior intervals. It is not an
efficacy run, d5 rerun, a supervisor fault-injection test or a benchmark of a
production solver. The original completed-run restart/reuse audit remains in the
archive and is distinct from this portable numerical replay.

## Complete original archives

Full standalone artifacts were delivered to the owner through ChatGPT Library.
Use their exact names and authenticated content hashes in `original-archives.json`
to locate them. Intermediate checkpoint/preparation archives are not substitutes
for completed results; incomplete attempts remain preserved in their final
recovery histories. Do not sum copied packages as independent experimental runs.

The repo carries compact evidence records rather than every duplicated trace
and dependency tree. In particular, the d3 refinement package contains roughly
101 MiB of expanded material (18 MiB compressed), much of it per-cell traces and
receipts. Its full package remains the replay source, not this compact report
snapshot. The 300,000 d5 shot arrays and later sliced-reference outputs likewise
belong to their original packages. A summary alone cannot reproduce them.

## Coverage limits and future work

This inventory covers the recoverable recent QEC research sequence, original
plans and the reliability repairs needed to execute it. Earlier Thermo research
already lives in the existing studies/roadmap/knowledge records; this import does
not duplicate or reinterpret those archived studies. No claim is made to have
reconstructed every prior conversation, tentative idea or unavailable external
agent state.

There is no new scientific execution permission, queue reordering, automatic
resume, hardware result or remote publication in this import. All result claims
remain scoped to their particular input models and selections. A production
implementation, actual d5 precision repair, calibrated selection gate or larger
comparison would be separate work.

The broader repository was spot-checked, not exhaustively re-audited: earlier M5,
E0, Potts, planar and related work already has studies/roadmap coverage; D-0005
and thinned-draw probes have dated research notes. THERMES deployed profile state
is external to this repository. `docs/research-loop.md` still mentions Fable 5.1
despite an earlier request for 5.2; this is flagged as possible documentation or
configuration drift, not verified deployed state. No profile setting was changed.

## Tensor-path follow-up

`archives/tensor-precision-feasibility.tar.gz` preserves the complete, unmodified
65,229-byte ten-fixture feasibility artifact. Its adjacent JSON authenticates
the archive; `reports/tensor-precision.txt` and `tensor-precision-audit.txt`
preserve its report and independent review. This adds an actual tensor-algorithm
interval port, not a production backend or a d5 result. Its absolute-path manifest
must not be rewritten to imply relocated execution continuity. Original replay
commands and limitations are in the report; no new study was run for this import.
