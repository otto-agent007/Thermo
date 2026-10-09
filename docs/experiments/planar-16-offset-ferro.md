# Planar 16-offset ferro: exploratory protocol (draft, proposal P-0001)

Drafted on 2026-10-08 for queue row `planar-16-offset-ferro`. Not frozen
until the owner approves it; no runner exists yet. Probe:
[`docs/research/2026-10-08-planar-16-offset-ferro-probe.md`](../research/2026-10-08-planar-16-offset-ferro-probe.md)
(exploration, one seed).

## Question

Do the planar Ising scaling allocations of
[PR #92](../experiment-reports/2026-10-07-planar-ising-scaling/findings.md)
hold on a planar subgraph of the Z1 16-offset lattice? In #92, on zero-field
ferro grids at beta = 4, five cold chains and one long chain qualify only at
64 spins, while every tempering arm qualifies at 64, 256 and 1024 spins (at
64, 256 and 1024 sweeps for the nine-replica ladder). The metric is which arms
qualify, and at what sweep budget, against the exact Kac-Ward reference.

## Fixed choices and what they bound

- **Graph: `greedy-long` planar subgraph of the 16-offset rule.** The rule
  ((1,0), (2,1), (2,3), (4,1) and their rotations, from
  `Z1HardwareProfile.interior_offsets`) is degree 16 and not planar, so a
  planar subgraph must be chosen. On an L x L patch, all candidate edges are
  sorted longest first, ties broken by a permutation seeded with the target's
  coupling seed; each edge is kept if its straight segment crosses no kept
  segment. At L = 32 this keeps 1982 edges (grid: 1984), 42 percent of them
  non-grid ((1,2) 252, (1,4) 580, (2,3) 1), with no isolated sites.
  *Bounds:* any planar bipartite graph has at most 2N - 4 edges, so mean
  degree stays below 4 whatever the choice. This study tests the chip's long
  edges in a planar arrangement, not its degree. The non-planar lattice has
  no exact reference here; that is a limit of the row's target family.
  `greedy-random` (seeded order, all lengths mixed) behaved the same in the
  probe and is not an arm; it loses 9 percent of edges and isolates sites.
- **Paired grid control.** The open L x L grid of #92 with the same fresh
  coupling seeds, so that graph effects are not confounded with coupling
  draws. *Bounds:* nothing; it is the comparison.
- **Couplings.** Weights {1..5}/5 drawn as in #92
  (`default_rng(seed * 1000 + L).integers(1, 6, E) / 5`), all negative (the
  archived sign convention; ferromagnetic by gauge on a bipartite graph). Ferro
  only, per the row; no mixed variant (that track is reserved for the owner's
  annealing study).
- **Temperature.** beta = 4, as archived. Ordered regime.
- **Ladders.** The thin five-replica ladder [0.25, 0.5, 1, 2, 4] and the
  nine-replica geometric ladder from #92, unchanged and imported. *Bounds:*
  the five-replica arms at L = 32. In the probe their cold-pair acceptance fell
  from 0.015 (grid) to 0.003 to 0.008 on the 16-offset subgraphs, and they
  qualified only at 4096 sweeps, at 0.036 to 0.045 against 0.05. This is the
  choice that sets the metric at L = 32. An N-scaled ladder is not an arm (the
  October 8 ladder probe closed that question for ferro grids).
- **Budgets T = 64, 256, 1024, 4096, 16384.** 16384 is added so that the
  expected five-replica slip at L = 32 reads as a budget rather than a
  censored "not reached"; 16 is dropped because no arm qualified there in #92
  or the probe. *Bounds:* qualification is censored at 16384. Long and
  independent are expected to stay censored at L >= 16.
- **Sizes L = 8, 16, 32** (64, 256, 1024 sites), open patches, interior rule
  only (no edge-of-chip rule). The L = 32 reference takes about 25 s.
- **Arms, at equal elapsed sweeps (as #92).** `long` (one chain, 5T sweeps),
  `independent` (five cold chains, T sweeps), `tempering5-k1`,
  `tempering5-k4`, `tempering9-k1`, `tempering9-k4`. Exchange intervals 1 and 4.
- **Threshold and estimator.** Mean trial edge-correlation MAE <= 0.05,
  sustained at all larger tested budgets; window (T/4, T] (quarter burn-in);
  16 trials. Unchanged from #92. Edge correlations are flip-invariant, so the
  symmetry estimator is the identity and is not an arm.
- **Cost model.** Each cell priced with #92's `price` function, imported
  unchanged (sealed Z1 profile, `published` and `beta_knob` conventions,
  R x N physical p-bits). Secondary; it does not enter the decision.

## Pre-registered expectations (from the one-seed probe)

1. At L = 16 and 32 on `greedy-long`, `long` and `independent` reach 0.05 on
   no target within 4096 sweeps, as on the grid. They may qualify at 16384 at
   L = 16 (#92 extrapolated about a million sweeps; the probe's L = 16 long
   chain was at 0.058 at 4096).
2. The nine-replica arms qualify at the same budget as the paired grid at
   every size (probe: 64, 256, 1024).
3. The five-replica arms slip one fourfold step against the paired grid at
   L = 32 (probe: 1024 grid, 4096 greedy-long for k = 1), on at least two of
   three seeds.
4. At L = 8, five cold chains may fail on `greedy-long` (probe: 0.073 at
   4096 against 0.039 on the grid); not gating.

## Exact anchor and evidence classes

The Kac-Ward determinant for the planar zero-field Ising model (ln Z and every
edge correlation), as in `planar_ising_scaling.kac_ward`, generalised to an
explicit straight-line embedding by taking site positions as an argument. The
matrix construction is identical; the study-local function must equal the
imported one bitwise on grids. Before any sampling: brute force on at least 12
patches of up to 16 sites containing non-grid edges (ferro at beta 4 and mixed
sign at beta 1, the latter only to exercise the turning angles of crossing-free
long edges) to 1e-9 in ln Z and per edge; the finite-difference check of the
two weakest correlations on every target, below 1e-4 (probe: at most 2.3e-9).
The transfer-matrix cross-check of #92 needs a grid and is kept for the grid
control at L <= 16 only.

Kernel checks: one colour-A-then-colour-B sweep is exactly stationary on a
16-offset patch small enough to enumerate (residual <= 1e-12); the
neighbour-list sampler reproduces `planar_ising_scaling.compile_sampler`'s
window sums and exchange flags bitwise on an 8 x 8 grid (probe: identical);
`tempering5-k1` on a 4 x 4 `greedy-long` patch reaches brute force within 0.01
mean edge error at 65,536 sweeps (probe on a greedy patch: 7e-6).

Sweeps are `software_simulation` (CPU, JAX float32). References are
`exact_reference` (float64). Z1 energies and sweep times are
`calibrated_projection` with the profile's exclusions (no host latency).
Nothing runs on hardware, and the subgraph is a synthetic patch of the
published local rule, not the physical chip map
(`exact_physical_graph_available = false`).

## Primary decision rule (fixed in advance)

For each arm, size L in {16, 32} and seed, compare the qualifying budget on
`greedy-long` with the paired grid. Per (arm, L), over three seeds:

- **holds**: equal budget on at least two of three seeds and never more than
  one fourfold step apart;
- **slips**: `greedy-long` later by at least one step on at least two seeds;
- **improves**: earlier by at least one step on at least two seeds;
- **mixed**: anything else. A censored budget (not reached by 16384) counts
  as later than 16384.

Row verdict, stated in the report as one of three sentences:

- **#92 holds on the 16-offset subgraph** if (a) at L = 16 and 32 `long` and
  `independent` qualify on no `greedy-long` target within 4096 sweeps while
  every tempering arm qualifies on 3/3 within 16384, and (b) both nine-replica
  arms are `holds` at L = 16 and 32.
- **#92 holds with a thin-ladder slip** if (a) and (b) hold and any
  five-replica arm is `slips`.
- **#92 does not hold** otherwise; the report names the failing clause.

L = 8 decisions and the comparison with #92's archived budgets (grid seeds
400 to 402) are reported descriptively and do not gate.

## What one recorded sample means

The vector of edge products s_i s_j over all edges of the retained replicas
after one complete two-colour sweep (cold replica for tempering, all five for
independent, the single chain for long). Trials, not sweeps, are the
replication unit.

## Seeds

Coupling and graph seeds 4510, 4511, 4512 (the probe used 4500; #92 used
400 to 402). JAX root 20261013, folded with target index and trial, split
into initialization and sampling keys; one initialization per trial shared by
all arms (as #92). Root 20261012 belongs to the probe and is not reused.

## Budget and CPU estimate

18 targets (two graphs x three sizes x three seeds), six arms, 16 trials,
horizon 16384 (long chain 81,920 sweeps). From the probe's timings: about
900 s per L = 32 target, 260 s per L = 16, 80 s per L = 8, so about 2.1 h wall
for one process and **about 4 to 4.5 CPU-hours** (XLA uses about two cores
per process). With two workers about 1.1 h wall. Replay adds about 4 minutes
(references only, no sweeps). Under the 8 CPU-hour flag.

## Autosave and resume

Over 30 minutes, so it follows the autosave contract in
`docs/experiment-runner.md`. The work unit is one (target, arm) cell group:
its window counts, accepted-exchange counts and timings are written
atomically to `units/<target>__<arm>.npz` with a digest in a unit index.
`--resume` re-authenticates the request and sources, verifies every completed
unit by digest and runs only the missing ones. Interruption test: kill the
runner after at least one unit, resume in the same directory, and require the
completed archive to equal an uninterrupted run of the same small request
bitwise. Resume command:
`uv run python -m thermo_lab.planar_16_offset_ferro --output-dir results/planar-16-offset-ferro --resume`.

## Persistence and archive size

Per (target, arm): the count of +1 edge products in each budget's window as
uint16 (budgets x trials x edges; at 16384 an int16 sum would overflow, the
longest window holds 61,440 products), and the accepted exchanges per pair
slot in each budget's window as uint16 (budgets x trials x pairs) instead of
per-sweep flags. Plus the request, graph edge lists, references, cells,
decisions and summaries in JSON. Replay authenticates sources, request and
counts, rebuilds every graph from its seed and checks its edge list, recomputes
every reference and check, then every estimate, error, price and decision,
tolerance 2e-12 relative. It does not regenerate sweeps.

Estimated archive: about 3.5 MB gzipped (the #92 ferro sums were 0.44 MB per
L = 32 target; six L = 32 targets dominate). If the archive exceeds 4 MB,
the runner still completes, but the executor does not commit it and asks the
owner.

## Gate

`uv run python -m thermo_lab.planar_16_offset_ferro --output-dir results/planar-16-offset-ferro`;
`completion.json`, written last after a full replay, must show
`status=planar_16_offset_ferro_complete`, `targets=18`, `cells_replayed=540`,
`decisions_replayed=108`, `references_recomputed=18`,
`reference_checks_passed=true`, `kernel_checks_passed=true`,
`graph_rebuild_passed=true`. CPU only, single-thread BLAS
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`), `JAX_ENABLE_X64=false`. CI
replays the archive only.

## Stop rules

- Any exact or kernel check fails: stop before sampling and report; do not
  loosen a tolerance.
- A unit fails to verify on resume: rerun only that unit; if it fails twice,
  stop and report.
- Wall time above 2x the estimate (about 4.5 h): checkpoint and report.
- Archive above 4 MB: do not commit; ask the owner.
- No arm, budget, seed or graph is added or removed after seeing errors.

## What a negative result means

"#92 does not hold" means that on a planar arrangement of the chip's long
edges at the same size, the allocation that worked on the grid changes: either
a cold or long chain starts to qualify (the long edges shorten mixing), or a
nine-replica arm needs a larger budget than on the grid. Either is a
statement about this graph family at beta = 4, 16 trials and budgets up to
16384. It says nothing about the non-planar degree-16 lattice, the physical
chip or frustrated targets. A five-replica slip alone is expected and is not
a negative.
