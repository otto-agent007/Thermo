# Research queue

The owner curates this file. The [research loop](research-loop.md) takes the
top `open` row that has no proposal yet, probes it, drafts a frozen protocol
and asks the owner on Discord before running anything. It never adds, edits
or reorders rows; a study's report PR only marks its own row `done`.

One row is one question, one target family and one metric. The `Row` id
becomes the branch `research/<row>` and the protocol
`docs/experiments/<row>.md`, so use lowercase letters, digits and hyphens.
Statuses: `open` (the loop may take it), `held` (skip for now), `done`,
`dropped`. Keep `|` out of the cells. A row the owner declined is skipped
until its text changes.

| Row | Status | Question | Target family | Metric | Notes |
| --- | --- | --- | --- | --- | --- |
| `planar-16-offset-ferro` | open | Do the planar Ising scaling allocations (PR #92) hold on the hardware-shaped graph? | Zero-field ferromagnetic Ising on the planar subgraph of the 16-offset lattice; the thin five-replica ladder; the exact Kac-Ward reference imported from `planar_ising_scaling` | Which arms qualify, and at what sweep budget, against the exact reference | Known-good regime that confirms #92 on the chip's graph. Ferro targets only; the frustrated-grid track is reserved for the owner's annealing study. |
| `am-coupling-bits` | done | How does associative-memory recall fall with coupling precision and per-site beta noise? | Stage A's bias-binary design with 4-, 6- and 8-bit couplings from the M5b codebook and ±10% per-site beta jitter | Recall against coupling bits and beta jitter, using stage A's exact references | Produces a spec number: the coupling bits needed to keep recall within tolerance. |
| `e0-remaining-kernels` | open | Does THRML match the archived exact inner-K law for every compiled M5a kernel, as E0 stage B showed for one? | The other 59 compiled M5a kernels (reading B) at K = 1, 2 and 4, with stage B's contract, tolerances and controls | Cells passing on output rate and joint, per kernel; control rejections | Pure replication. Stage B took about 11 CPU-minutes for one kernel, so all 59 is roughly 11 CPU-hours before parallelism; the probe sizes it, and a run over 8 CPU-hours needs the owner's say-so. |
