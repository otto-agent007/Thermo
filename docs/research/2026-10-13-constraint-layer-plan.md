# Constraint layer: engineering plan (P-0004)

Engineering plan for queue row `constraint-layer`, dated 2026-10-13. The owner
answered P-0004's earlier question by making this row
engineering: "A code PR adds thermo_lab.constraints plus bitwise unit tests,
with E0 and AM checked with placement relaxed and flagged. Mark the row done
when it merges." Under `docs/research-loop.md` step 1 an engineering row gets
this plan instead of a protocol. It has no production run, evidence archive,
`completion.json`, study gate, studies entry or release-gate entry, because
nothing new is measured.

The basis is the probe of 2026-10-10
([note](2026-10-10-constraint-layer-probe.md),
[script](constraint_layer_probe.py),
[JSON](2026-10-10-constraint-layer-probe.json)). It is exploration, not
evidence. Its throwaway prototype layer already reproduced every check below,
so the build moves tested probe code into a module and pins it with unit tests.

This PR holds this plan only. It has no `src/` or `tests/` code. The build
writes them after the owner approves.

## Fixed choices and what each bounds

- **Placement of over-degree kernels.** This bounds the E0 check. The E0
  stage B kernel (reading B, variational, cap 1, seed 0, site 1) has output
  degree 17, so the lattice rejects it unchanged. The E0 check runs with
  `placement` relaxed and flagged. The placed M5c refit is checked against its
  own exact law, not against E0's.
- **Placement of the AM `bias` design.** This bounds the AM check. The design
  is complete bipartite: hidden degree 24, free visible degree 8 to 128,
  against 16. The AM check runs with `placement` relaxed and flagged (D-0001).
- **Cap rule.** "Step = J" is defined only for models whose couplings share
  one magnitude, like the AM design. M5b uses an absolute cap of 1.0. The
  request declares the cap rule per model: `absolute`, `full`, `split` or
  `coupling`. It does not bound agreement.
- **Codebook domain.** `round_parameters` accepts bits in {4, 8, 12} and
  raises when a value exceeds the cap. The layer's codebook clips and takes any
  width. The two agree bitwise on `round_parameters`' domain. The layer calls
  `round_parameters` itself on that domain, so the M5b check is reuse, not
  reimplementation.
- **BLAS-dependent float64 fields.** These bound "bitwise" for stationary and
  trajectory metrics. Rounding summaries, layouts, histograms, recall terms
  and correct-bit counts are compared bitwise. Stationary-solve metrics are
  compared at 1e-15 absolute (lesson of 2026-10-06; the probe saw 8.3e-17).
- **Definition of a counted update.** It does not bound agreement. It changes
  the priced numbers. The accounting record declares `update_rule`:
  `whole_array` (every participating free p-bit per sweep) or
  `active_kernel` (only the active kernel's free p-bits; M5b's sequential
  schedule).
- **MILP re-solve cost.** It does not bound agreement; it bounds CI time.
  Re-solving all 60 M5c placements costs about 41 CPU-minutes at archived
  timings, so CI re-solves the two fastest (1.3 s each) and the rest are slow.
- **Float32 THRML.** THRML programs run in float32 on CPU, as every archived
  THRML study did. The request records `dtype`. Exact laws stay float64.

## Modules and public API

One package, `src/thermo_lab/constraints/`. It imports the hash-bound modules
read-only and edits none of them. Torx and THRML calls stay inside
`constraints/thrml_lowering.py`, the adapter rule of AGENTS.md.

`constraints/request.py`
- `CONSTRAINTS = ("placement", "quantization", "beta_jitter",
  "chromatic_schedule", "no_replica_exchange")`.
- `ConstraintRequest`, a frozen dataclass: `model_id: str`, `bits: int | None`,
  `cap_rule: str`, `cap: float | None`, `jitter: float`, `jitter_words:
  tuple[int, ...] | None`, `update_rule: str`, `dtype: str`, `relaxed:
  tuple[str, ...]` (sorted, each from `CONSTRAINTS`, with a `reasons`
  mapping). `to_canonical()` and `digest()` use `thermo_lab.hashing`. The
  lattice rules come from `Z1HardwareProfile().connection_rules` and the
  profile hash is part of the request. Changing any relaxed flag changes the
  digest.

`constraints/model.py`
- `PairwiseModel`, a frozen dataclass: `sites: tuple[tuple[int, int], ...] |
  None`, `edges: tuple[tuple[int, int], ...]`, `weights`, `bias` (float64
  arrays), `free`, `clamped`, `clamped_input`, `first_colour`. Validated on
  construction: no self-edges, no duplicate edges, free and clamped disjoint.

`constraints/lattice.py`
- `colour(site) -> int`, `(u + v) mod 2`.
- `check(model) -> LatticeReport` with `edges`, `offset_violations`,
  `colour_violations`, `max_degree`, `degree_violations`, `distinct_sites`,
  `ok`.
- `place_m5c(vector, structure) -> PairwiseModel`. It calls
  `meta_ebm_topology.place_kernel` and lowers the layout with clamped copy
  nodes, as in the probe's `lower_placed_kernel`.
- `lower_m5c_layout(layout, vector, structure) -> PairwiseModel` for archived
  layouts, without re-solving.
- `planar_target(target) -> PairwiseModel`. It calls
  `planar_16_offset_ferro.build_target` and `target_positions`.
- `unplaced(model_like, colours) -> PairwiseModel`. Sites are `None`; it
  carries only a colour per node. Using it requires `placement` in
  `relaxed`.

`constraints/quantize.py`
- `codebook(x, cap, bits) -> ndarray`, the M5b rule with clipping.
- `cap_for(rule, couplings, fields, bits) -> (coupling_cap, field_cap)` for
  `absolute`, `full`, `split` and `coupling`; it rejects `coupling` when the
  nonzero coupling magnitudes are not all equal.
- `quantize(model, request) -> (PairwiseModel, RoundingSummary)`. On
  `round_parameters`' domain it delegates to `round_parameters`.
- `rounding_summary(vectors, cap, bits)` with M5b's fields and reduction
  order.

`constraints/jitter.py`
- `static_gains(words, sizes, jitter) -> list[ndarray]`: one
  `default_rng(words)`, then `uniform(1 - j, 1 + j, n)` per group in order.
  This is `am_coupling_bits.gains`' draw order, so the AM words reproduce its
  gains bitwise. Gains are fixed per request, never redrawn per sweep.

`constraints/gibbs.py`
- `exact_chromatic_kernels(model, x_batch) -> (kernels, states)`, float64
  enumeration of one two-colour sweep, bounded by `MAX_EXACT_FREE = 12` free
  nodes (AGENTS.md: bounded enumerators).
- No replica exchange and no per-replica temperatures exist in the API. A
  request with `no_replica_exchange` relaxed is rejected: no study needs it
  yet, so there is nothing to relax it for.

`constraints/thrml_lowering.py`
- `lower(model, request) -> (ebm, program, free_blocks)`: THRML `IsingEBM`
  with one free block per colour and one clamped block, as in the probe.
- With `jitter > 0` it uses `am_coupling_bits.GainSpinConditional` (imported,
  not copied), so jittered programs share AM's arithmetic.
- `lower_am(zq, gains, p, n, cue)` returns AM's own `gain_structure(...)
  .gain_program(...)` built from layer-quantized terms and layer gains. The AM
  sampled check uses it, as the probe did; bitwise counts need AM's node and
  factor order, which the generic lowering does not promise.

`constraints/accounting.py`
- `count(request, *, logical_pbits, physical_pbits_used, free_pbits, sweeps,
  node_reads, node_full_sram_writes, clamp_state_changes, active_updates=None)
  -> Z1OperationCounts`, through `Z1OperationCounts.constant_participation`.
- `price(request, counts) -> dict` with sampling, read and write energy, I/O
  share, `evidence_class` (`calibrated_projection`), the request digest and the
  relaxed list. Updates, reads, full SRAM writes and clamp changes stay
  separate fields. These are algorithmic counts, not device operations.

## Hardware constraints applied and relaxed (rule 8)

Applied by default: placement on the 16-offset lattice with a two-colour
schedule, coupling and bias quantization under a declared cap rule, static
per-site beta jitter (default 0, declared), chromatic block Gibbs only, no
replica exchange, and separate update, read, write and clamp counts.

Relaxed in this build's own checks, each flagged in the hashed request:

| Check | Relaxed | Why |
| --- | --- | --- |
| M5b rounding | `placement`, `beta_jitter` | M5b kernels are sequential per-site kernels; the check is the codebook alone |
| M5c placement | `quantization`, `beta_jitter` | The archive placed float64 refits |
| Planar graphs | `quantization`, `beta_jitter` | The archive used float32 ferro couplings |
| AM 6-bit recall | `placement` | Hidden degree 24 and visible degree up to 128 against 16 (owner decision, D-0001) |
| E0 stage B law | `placement`, `quantization`, `beta_jitter` | Kernel output degree 17 against 16 (owner decision); E0 used float64 parameters cast to float32 |

## Archives and samplers reproduced bit for bit

Every source is authenticated by its committed SHA-256 before use, through the
studies' own `load_source` or `load_kernel` helpers.

1. **M5b rounding** (via the M5b archive that `meta_ebm_topology.load_source`
   pins). All 120 reading-B precision cells: rounded arrays equal
   `round_parameters` and the rounding summary equals the archived `rounding`
   field, bitwise. Three cells' chain metrics recomputed from layer-rounded
   vectors: `bias` and `sweep_tv_*` bitwise, stationary fields at 1e-15.
   Probe: 120/120 and 3/3.
2. **M5c placement** (`2026-09-30-meta-ebm-topology/study.json.gz`). All 60
   archived layouts pass the layer's lattice check (max degree 16, no offset
   or colour violations) and reproduce `verify_layout`'s archived fields
   bitwise. Re-solved placements equal the archived layout bitwise. Probe:
   60/60 and 9/9 re-solved.
3. **Planar 16-offset ferro** (`2026-10-08-planar-16-offset-ferro/evidence.tar.gz`).
   All 18 targets rebuild bitwise and pass the lattice check (max degree 9).
   Probe: 18/18.
4. **am-coupling-bits at 6 bits, step = J**
   (`2026-10-08-am-coupling-bits/study.json.gz`). **Placement relaxed.** All
   96 exact units give `coupling/6` recall bitwise through the layer's
   quantizer. Archived THRML runs at P = 8, 12-bit cue, jitter 0 and 0.1,
   with layer gains, give the archived correct-bit counts bitwise. Probe:
   96/96 and 2/2.
5. **E0 stage B** (`2026-10-03-thrml-m5a-kernel-inner-sweep/study.json.gz`).
   **Placement relaxed.** The layer's unplaced THRML lowering, with E0's keys,
   reproduces the archived histograms bitwise. Probe: 3/3 cells tried
   (y0-/K1, y0+/K1, y0-/K2); the build checks all 6. With placement applied
   the layer cannot reproduce E0, because the placed kernel is the M5c refit
   (J[4] masked). Two placed checks still run and are labelled as such: the
   placed exact chromatic law equals the refit's own `inner_kernel` law at
   1e-14 (probe 1.9e-15), and THRML on the placed model passes the refit law
   within E0's archived tolerances for that cell (probe: output 0.0045 against
   0.0089, joint 0.0188 against 0.0197). The test records the placed-versus-E0
   gap (probe up to 0.31, 0.40, 0.38 at K = 1, 2, 4) as a documented
   difference, not a failure.
6. **Pricing.** One priced example per constraint (placement, quantization,
   beta jitter, chromatic schedule with no exchange), each with its relaxed
   list in the hashed request. The probe's counts are pinned: M5c outer sweep
   at K = 4 is 288 updates, 12 reads, 344 writes, 272 clamp changes; AM recall
   at P = 8 is 5,120 updates, 12 reads, 32 writes; planar 1024-spin chain is
   4,194,304 updates, 786,432 reads, 1,024 writes. A test checks that flipping
   one relaxed flag changes the digest.

Evidence classes: rounding, placement and exact laws are `exact_reference`;
THRML histograms and AM correct-bit counts are `software_simulation` (CPU,
float32); pricing is `calibrated_projection` of algorithmic counts. No
hardware claim.

## Tests

New file `tests/unit/test_constraints.py`, CPU only, no network, single-thread
BLAS pinned in the module as the archived studies do.

Fast subset (runs in `pytest tests/unit -m "not slow"` and CI; about 60 s
total from the probe's timings):

- Request: canonical digest stable; relaxed list sorted and validated;
  flipping one flag changes the digest; unknown constraint rejected;
  `no_replica_exchange` cannot be relaxed.
- Lattice: offsets equal `meta_ebm_topology.OFFSETS`, `planar_16_offset_ferro.OFFSETS`
  and `Z1HardwareProfile.interior_offsets()`; the checker rejects a bad
  offset, a same-colour edge and degree 17 (the E0 kernel itself).
- Quantizer: equals `round_parameters` bitwise on {4, 8, 12}; clips outside
  the cap; `coupling` rule rejects mixed magnitudes.
- M5b: all 120 rounding summaries bitwise (under 1 s).
- M5c: all 60 archived layouts lattice-checked and `verify_layout` fields
  bitwise; re-solve the two fastest placements (`4|7`, `1|4`, about 1.3 s
  each) and compare layouts.
- Planar: all 18 targets rebuilt bitwise and lattice-checked (about 1 s).
- AM: 8 of 96 exact units bitwise (one per cell and seed class, about 1.5 s);
  the two P = 8 sampled runs bitwise (about 2 s).
- E0: one cell (y0-/K1), first input batch only (16 of 1,024 inputs; E0 keys
  are folded per batch, so the batch's histogram rows are bitwise
  comparable), unplaced lowering. Placed exact law against `inner_kernel`
  on 64 inputs.
- Pricing: the four priced examples equal their pinned counts.

Slow subset (`@pytest.mark.slow`; about 25 CPU-minutes on this box, mostly
JAX threads):

- M5b chain metrics for the 3 cells (about 24 s).
- AM: all 96 exact units (about 19 s).
- E0: all 6 cells unplaced, full 1,024 inputs at 65,536 chains (about 33 s per
  K = 1 cell, 57 s per K = 2, about 6 minutes total); placed exact law on all
  1,024 inputs; THRML placed run, one cell (about 37 s).
- M5c: re-solve the 9 placements under 5 s each (about 25 s). Full re-solve
  of all 60 (about 41 CPU-minutes) stays out of the test suite and is
  documented as a manual command in the As-built section.

## Build estimate

About 6 hours of agent work, plus about 30 CPU-minutes to run the slow tests
once. Most of the code exists in the probe script; the work is module
boundaries, validation, the fast-subset selection, docstrings, `ruff` and the
full unit gate. Deliverables: `src/thermo_lab/constraints/` (about 600
lines), `tests/unit/test_constraints.py` (about 400 lines), an "As built"
section appended to this plan, and the queue row marked done. No new
dependencies. No new archive. No production run.

## What the build does not do

- It does not edit any hash-bound source or committed archive.
- It does not place over-degree kernels by chains or by masking; that changes
  the law, and the owner chose to flag the relaxation instead.
- It does not add replica exchange or free temperature schedules.
- It does not add CLAUDE.md, AGENTS.md or roadmap entries; this card does not
  ask for them. Suggested text for the owner can go in the build PR body.
