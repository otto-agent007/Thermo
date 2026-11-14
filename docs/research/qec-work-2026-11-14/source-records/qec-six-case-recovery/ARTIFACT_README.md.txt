# QEC short-batch execution repair

This is an execution-layer repair package, not new d5 research results.

- `executor.py`: short synchronous subprocesses, CPU/memory/wall bounds, parent-death protection, durable reservation/receipt/ownership protocol.
- `adapter.py`: candidate-prefix continuation, actual Tesseract and beliefmatching calls, archived baseline metadata preservation, unchanged sliced contraction, conditional numerical interval assembly.
- `test_executor.py`, `test_adapter.py`: 21 synthetic tests.
- `synthetic-e2e/`: preserved complete tiny synthetic run with exact rational enumeration oracle.
- `validate_fixture.py`: regenerates that run into a new absent `synthetic-e2e` directory.
- `INDEPENDENT_EXECUTOR_REVIEW.md`: independent checks and limits.
- `PREPARED_RECOVERY.json`: historical recovery proposal; it remains non-executing and still records the unresolved original-process gate.

No old source or output was overwritten, and no remaining d5 case was launched.

The synthetic baseline/observed bits are metadata test values. They are not four newly benchmarked decoders. The Tesseract/beliefmatching adapter calls did execute on the synthetic DEM. Class probabilities were independently enumerated with exact rational binary64 input values and enclosed by the final intervals.

## Portability

Manifests and receipts retain producer-local absolute paths for provenance. To replay elsewhere, install the pinned requirements and stage a new explicitly identified synthetic run in the new workspace, preserving content hashes and recording the path translation. Do not silently edit old manifests or treat producer paths as current consumer paths. The production recovery selection and budget must be bound in a new approved manifest before launch.
