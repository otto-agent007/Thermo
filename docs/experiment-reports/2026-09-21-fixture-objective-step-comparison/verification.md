# Verification record

All required gates passed on clean implementation commit `386338d` before this documentation-only archive was added.

- 34 recorded commands, all exit code zero: six environment/lint/build/package checks, three exhaustive disjoint test partitions, and 25 experiment gates.
- 2,094 tests passed: 1,645 unit/upstream, 193 integration A, and 256 integration B. No tests were omitted from the collected suite.
- The 22 focused new tests also passed separately before the full regression run.
- Wheel and source distribution include both new modules; experiment configurations were checked by the packaging gate.
- Independent scientific/code review approved the implementation; see review.md for additional enumeration and gradient checks.
- Fresh gate generation and persisted replay reproduced the original scientific result digest exactly: `sha256:8bb139d5793d850c70eb7521c0bbe7bf6875f29ba4d8774fa3cb2217e8e5d3d4`.
- Archive compression and decompressed byte hashes were checked against archive-metadata.json.

Exact commands, exit codes and CPU execution durations appear in verification-commands.json. Durations are verification provenance, not matched training costs, hardware latency or energy measurements. Test partitions ran concurrently on separate CPU affinity sets with a stable Git checkout.

The archive contains the full compressed study, original completion/provenance records, human findings, review and verification. Scientific completion does not establish full M4G task quality. No publication or merge was performed.
