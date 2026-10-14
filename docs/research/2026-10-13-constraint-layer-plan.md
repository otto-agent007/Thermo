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

## As built

Built 2026-10-13 by THERMES (loop card t_2522f9e4, PR #107). Everything
above this heading is the approved plan, unchanged. This section records what
the build delivered and where it differs.

### Modules

Package `src/thermo_lab/constraints/`, about 800 lines:

- `request.py`: `CONSTRAINTS`, `ConstraintRequest`, `to_canonical()`,
  `digest()`.
- `model.py`: `PairwiseModel`.
- `lattice.py`: `colour`, `check -> LatticeReport`, `require_placement`,
  `solve_m5c_layout`, `lower_m5c_layout`, `place_m5c`, `planar_target`,
  `build_planar_target`, `unplaced`, `unplaced_m5a_kernel`.
- `quantize.py`: `codebook`, `quantize_array`, `rounding_summary`, `cap_for`,
  `quantize`, `quantize_terms`.
- `jitter.py`: `static_gains`, `request_gains`.
- `gibbs.py`: `MAX_EXACT_FREE = 12`, `free_states`,
  `exact_chromatic_kernels`, `propagate`.
- `thrml_lowering.py`: `lower`, `lower_am`, `final_states`. This is the only
  module that imports THRML.
- `accounting.py`: `count`, `price`.

It imports `meta_ebm_cap_baseline`, `meta_ebm_thermalization_core`,
`meta_ebm_topology`, `planar_16_offset_ferro`, `am_coupling_bits` and
`hardware.z1` read-only. It edits no file outside the new package. No new
dependency.

### Usage

```python
from thermo_lab.constraints import (
    ConstraintRequest,
    check,
    count,
    exact_chromatic_kernels,
    planar_target,
    price,
    quantize,
)

request = ConstraintRequest(
    model_id="L16 planar ferro, 6-bit",
    bits=6,
    cap_rule="full",  # quantization applies
    relaxed=("beta_jitter",),
    reasons={"beta_jitter": "noise-free baseline arm"},
)
model = planar_target(target)  # a planar-16-offset-ferro target dict
assert check(model).ok  # 16-offset lattice, two colours, degree <= 16
model_q, rounding = quantize(model, request)
counts = count(
    request,
    logical_pbits=256,
    physical_pbits_used=256,
    free_pbits=256,
    sweeps=4096,
    node_reads=256,
    node_full_sram_writes=256,
)
priced = price(request, counts)  # calibrated_projection; carries request.digest()
```

THRML sampling is `thrml_lowering.lower(model, request)`. It returns the
`IsingEBM`, the program and the free colour blocks. `final_states` runs
chains to exactly K sweeps. Exact laws for at most 12 free nodes come from
`exact_chromatic_kernels`.

### Tests

`tests/unit/test_constraints.py` has 43 tests: 32 fast and 11 `slow`. Every
archive is checked against its committed SHA-256 before use, or loaded
through the study's own hash-checking loader. The fast subset took 13 s on
this box. The slow subset took 742 s wall (3,619 s user; JAX uses all
cores).

Archive and sampler checks:

| Plan item | Test | Bitwise check | Subset |
| --- | --- | --- | --- |
| 1 M5b rounding | `test_m5b_rounding_all_120_reading_b_precision_cells_bitwise` | 120/120 cells. Rounded arrays equal `round_parameters`, and the summary equals the archived `rounding` field | fast |
| 1 M5b metrics | `test_m5b_chain_metrics_three_cells_cross_host_within_1e15` | 3 cells. Every metric within 1e-15, in a pinned subprocess; `sweep_tv_*` bitwise with `THERMO_ARCHIVE_HOST=1` | slow |
| 2 M5c layouts | `test_m5c_all_60_layouts_on_lattice_and_verify_layout_cross_host_replay_rule` | 60/60 pass `check` (max degree 16), and `verify_layout`'s archived fields match under M5c's replay rule; bitwise with `THERMO_ARCHIVE_HOST=1` | fast |
| 2 M5c re-solve | `test_m5c_fast_resolved_placement_equals_archived_layout[4\|7, 1\|4]` | MILP layout equals the archived layout | fast |
| 2 M5c re-solve | `test_m5c_resolved_placements_under_5s_equal_archived_layouts` | The other 7 placements under 5 s | slow |
| 3 Planar | `test_planar_all_18_targets_rebuild_bitwise_and_lie_on_lattice` | 18/18 rebuilt bitwise and on the lattice (max degree 9) | fast |
| 4 AM exact | `test_am_6bit_exact_recall_relaxed_placement_cross_host_within_1e12` | 8 units: `coupling/6` recall within 1e-12 on any host; bitwise with `THERMO_ARCHIVE_HOST=1` | fast |
| 4 AM exact | `test_am_6bit_exact_recall_all_96_units_relaxed_placement_cross_host_within_1e12` | 96/96 units, same rule | slow |
| 4 AM sampled | `test_am_6bit_sampled_p8_relaxed_placement_bitwise[0.0, 0.1]` | P = 8, 12-bit cue, seed 7400: correct-bit counts bitwise at jitter 0 and 0.1 | fast |
| 5 E0 | `test_e0_relaxed_placement_cross_host_first_batch_y0minus_k1_within_frozen_tolerance` | y0-/K1, first 16 inputs, within E0's frozen output and joint tolerances; bitwise with `THERMO_ARCHIVE_HOST=1` | fast |
| 5 E0 | `test_e0_relaxed_placement_cross_host_all_cells_archived_verdicts[6 cells]` | All 6 cells, 1,024 inputs, 65,536 chains: E0's own `evaluate` with frozen tolerances gives the archived passes, closest K and control rejections; bitwise with `THERMO_ARCHIVE_HOST=1` | slow |
| 5 E0 placed | `test_e0_placed_refit_exact_law_equals_its_inner_kernel_64_inputs` | Placed refit's exact law against its own `inner_kernel` law, at most 1e-14 | fast |
| 5 E0 placed | `test_e0_placed_refit_exact_law_all_inputs_and_documented_gap_to_e0` | The same on 1,024 inputs. Records the gap to E0's law (documented, not a failure) | slow |
| 5 E0 placed | `test_e0_placed_refit_thrml_within_archived_tolerance_one_cell` | THRML on the placed refit, y0-/K1, within E0's archived output and joint tolerances | slow |
| 6 Pricing | `test_pricing_four_examples_equal_pinned_counts` | 288/12/344/272, 5,120/12/32/12 (twice) and 4,194,304/786,432/1,024/0 | fast |

The other fast tests cover four things:

- Request: the digest is canonical; flipping or rewording one relaxed flag
  changes it; an unknown constraint is rejected; `no_replica_exchange`
  cannot be relaxed; quantization and jitter fields are validated.
- Lattice: the offsets equal those of M5c, planar-16-offset-ferro and the Z1
  profile. `check` rejects a bad offset, a same-colour edge, degree 17 and
  the E0 kernel itself. An unplaced model requires `placement` relaxed.
- Quantizer and gains: `round_parameters` is matched bitwise at 4, 8 and 12
  bits. Values outside the cap clip. The `coupling` rule rejects mixed
  magnitudes. `static_gains` equals `am_coupling_bits.gains`.
- Exact kernels: rows sum to one, and the enumerator is bounded at 12 free
  nodes. A pinned-BLAS subprocess check covers the environment.
- Jittered lowering: `lower` with static gains (through
  `GainSpinConditional`) on a placed four-free-node model agrees with the
  exact gain-weighted one-sweep law to within 6 sigma per state (65,536
  chains, fixed key). This is a statistical self-check, not an archive
  reproduction.

Results of the slow run on this box: 10 of 11 passed on the first run. The
96-unit AM exact test failed in the last bits because the shell lacked the
BLAS pin (deviation 1 below). After the subprocess fix it passed, and both
AM exact tests passed with the pin deliberately unset. The placed-versus-E0
output-law gap was 0.307, 0.395 and 0.382 at K = 1, 2 and 4. These equal the
probe's numbers. They are a restated exploration value, not a new
measurement.

### Relaxed checks, flagged

The names come from the hashed `ConstraintRequest.relaxed` in each test.

| Check | Relaxed |
| --- | --- |
| M5b rounding | `placement`, `beta_jitter` |
| M5c placement, E0 placed refit | `quantization`, `beta_jitter` |
| Planar | `quantization`, `beta_jitter` |
| AM exact and jitter-0 sampled | `placement`, `beta_jitter` |
| AM jitter-0.1 sampled | `placement` |
| E0 unplaced | `placement`, `quantization`, `beta_jitter` |

Every AM and E0 test that relies on relaxed placement says
`relaxed_placement` in its name.

Cross-host relaxation (review round 1 and the fix after merge, deviation
7). Six tests are bitwise only on the archive host and say `cross_host` in
their name: the two AM exact tests, the two E0 unplaced tests, the M5c
layout test and the M5b metrics test. Off that host they check a
declared tolerance or the archived verdicts, as the studies' own replays do.
This is a test-side relaxation of the comparison, not of a hardware
constraint, so it is not a `ConstraintRequest` field and the request digests
are unchanged. `THERMO_ARCHIVE_HOST=1` restores the bitwise assertion.

| Check | Off the archive host | Source of the rule |
| --- | --- | --- |
| AM exact recall | every value within 1e-12 of the archive | `am_coupling_bits.replay` (`_match`, rtol = atol = 1e-12) |
| E0 histograms, fast | first 16 rows of y0-/K1 within the frozen output and joint tolerances | E0 `evaluate` pass rule |
| E0 histograms, slow | E0 `evaluate` with frozen tolerances gives the archived `output_pass`, `joint_pass`, `closest_k` and control rejections | E0 `replay` (frozen tolerances; histograms authenticated by digest, not regenerated) |
| M5c layouts | integer fields exact; `conditional_error` within rel 1e-9, abs 1e-10 | `meta_ebm_topology.replay` (placements through `m5a._close`) |
| M5b metrics | every metric within 1e-15, `sweep_tv_*` included | the plan's own bound for BLAS-dependent M5b fields (M5b's replay allows 1e-10) |

### Deviations from the plan

1. **The AM exact check runs in a pinned subprocess.** Under threaded
   OpenBLAS, 4 of the 8 fast AM units differ in the last bits. One example:
   0.9949744706144386 against the archived 0.9949744706144596. With
   `OPENBLAS_NUM_THREADS=1` all 96 are bitwise. Setting the pin inside the
   test module does not work, because pytest has already loaded numpy. Both
   AM exact tests therefore run in a subprocess with the pin set, as the M5b
   metric test does. The AM archive was produced single-threaded. This is
   the same BLAS effect the plan already bounds for M5b's stationary fields.
2. The AM fast subset is 8 units: each of the 6 cells at seed 7400, plus
   P8/c12 and P128/c8 at seed 7415. The plan said "one per cell and seed
   class" without naming them.
3. `ConstraintRequest` adds `reasons`, as the plan's text implies, and
   `bits=None` / `cap_rule="none"` when quantization is relaxed. The cap is
   set only by the `absolute` rule.
4. `PairwiseModel` gains an optional `colours` field so that `unplaced`
   models carry a colour per node. Placed models derive their colours from
   their sites.
5. `lattice.unplaced_m5a_kernel` is a named helper for E0's node order
   (inputs, hidden, output; hidden block first).
6. The full re-solve of all 60 M5c placements stays out of the suite, as
   planned. To run it by hand (about 41 CPU-minutes), call
   `lattice.solve_m5c_layout(refit, structure)` for each `seed|site` key in
   `docs/experiment-reports/2026-09-30-meta-ebm-topology/study.json.gz` and
   compare it with `placements[key]["layout"]`. Use the test helpers
   `_m5c_vectors` and `_resolve` with `OPENBLAS_NUM_THREADS=1`.
7. **AM exact and E0 histograms are bitwise only on the archive host.**
   PR #107's CI at 3915dd0 (run 38479335702, `slow-contracts`, ubuntu-latest,
   4 xdist workers) failed both, though the BLAS pin was set: AM exact
   P128/c12 units were not bitwise, and E0 histograms differed in 5 of 6
   cells (all but y0+/K1). Locally (i7-7700K, the host that wrote both
   archives) all pass bitwise. The cause is CPU-dependent float64 BLAS and
   XLA float32 code generation, not threading. Neither study's replay ever
   promised cross-host bitwise equality: AM replays exact values at 1e-12,
   and E0 authenticates histograms by digest and recomputes only the
   evaluation. The four affected tests were renamed `cross_host` and check
   the rules in the table above; with `THERMO_ARCHIVE_HOST=1` they also
   assert bitwise equality. After the merge, main's CI at 6c9e5f4 (run
   38503799344, `unit-rest`) failed the M5c layout test on placement 0|2,
   though the PR's run had passed it: `conditional_error` is a float64
   rounding residue of about 1e-15, and its last bits follow the runner's
   BLAS kernels. With `OPENBLAS_CORETYPE` set to Prescott, Nehalem or
   Sandybridge it differs on 8 of 60 placements locally; Haswell, Zen and
   this box's own kernels match. That test is now `cross_host` too and uses
   M5c's replay rule. The slow M5b metrics test had the same latent fault:
   under the Prescott and Sandybridge kernels `sweep_tv_m5a` is not bitwise
   (up to 1.1e-16; worst field 4.4e-16), so off the archive host every
   metric is checked within 1e-15. Under the Prescott kernels the rest of
   the suite passes unchanged (M5b rounding, M5c re-solve, planar, AM
   sampled, pricing and the existing `cross_host` tests).

No hash-bound source or committed archive changed. Nothing was written under
`results/`. No new number was measured.

### What a later row must declare

A row that runs through this layer must state all of the following in its
protocol and hashed request:

- one `ConstraintRequest` per model, with its digest;
- the relaxed constraints, each with a reason;
- `bits`, `cap_rule` and `cap` when quantization applies;
- `jitter` and `jitter_words` when jitter is positive;
- `update_rule` (`whole_array` or `active_kernel`);
- `dtype`.

If placement applies, the model must pass `require_placement`. A model over
degree 16 is not placed by chains or masking; it runs with `placement`
relaxed, or the row is changed. Results keep their evidence classes: exact
laws are `exact_reference`, THRML runs are `software_simulation`, and
`price` output is a `calibrated_projection` of algorithmic counts. None is
hardware evidence. Run on CPU with `OPENBLAS_NUM_THREADS=1` set before numpy
loads, or compare BLAS-dependent float64 fields numerically at about 1e-15.
