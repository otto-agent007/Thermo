# THRML execution of an M5a kernel's inner sweep (E0 stage B)

**Status: frozen 2026-10-03 and run the same day.** The exact side, the
tolerance rule and the negative-control gate were fixed before any THRML
sampling at the recorded chain count. Change any value only through a new
protocol version. The study runs in about 15 minutes on one CPU and has no
autosave or resume layer; a fresh output directory is the restart. Owner
approval for the run and for reading B as the standing choice: 2026-10-03.

## Question and scope

E0 ([protocol](thrml-finite-sweep-contract.md)) certified that THRML 0.1.4's
state after K block-Gibbs sweeps is p0 T^K for the five-spin chain's exact
sweep kernel. M5a compiles a target conditional into a site kernel of hidden
spins w and one output spin y; M5b's finite-K law
([protocol](meta-ebm-thermalization.md)) states what K inner sweeps of that
kernel do to the output at every fixed blanket input x. Those finite-K
matrices had never been run through THRML. This study asks: when THRML
executes one compiled M5a kernel's inner sweep, clamped to its blanket inputs
as M5b does, is the state after exactly K sweeps the archived exact inner-K
law?

Evidence classes. Exact references are `exact_reference`: float64 NumPy, the
archived M5b law through the hash-bound `meta_ebm_thermalization_core.powered_rates`
and a study-local enumeration of the 2^(n_h+1)-state (w, y) joint. THRML
cells are `software_simulation`: THRML 0.1.4 on CPU in float32. Nothing here
is hardware evidence, and inner sweeps are not device operations.

## Fixed choices

Every value below is in the hashed request (`study_request()` in
`src/thermo_lab/thrml_m5a_kernel_inner_sweep.py`).

- Kernel: the M5c primary arm (reading B, variational, cap 1), seed 0, site 1.
  Its parameter vector is read from the M5b archive
  `docs/experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz`
  through `meta_ebm_topology.load_source` (archive SHA-256 and pinned sources
  authenticated) and `meta_ebm_topology.references`; its structure comes from
  `meta_ebm_cap_baseline.structures(make_target(0, "B"))`. No hash-bound file
  is edited. Site 1 has the largest blanket (k = 10, 1,024 inputs) and most
  hidden spins (n_h = 7, 256 joint states) among the seed-0 sites, tied with
  site 10, and the slowest archived inner contraction (lambda_max 0.78,
  k* = 131), so finite K matters most there. The vector's SHA-256 is recorded
  in the archive and checked at replay.
- THRML model: `IsingEBM` over x (10 `SpinNode`, clamped block), w (7, free
  block), y (1, free block); beta = 1; edges x-y with weights J, x-w with
  weights A, w-y with weights beta_a; biases b on w and h on y; numeric dtype
  float32, declared in the request. This encodes E0's verified conventions
  (energy sign, beta, {-1,+1} encoding, block order, clamping, one warm-up
  step = one sweep) for the M5a energy
  `E = -(J.x+h) y - sum_a (A_a.x+b_a) w_a - sum_a beta_a w_a y`.
- Sweep: hidden block then output block, matching M5b's `inner_kernel`
  (hidden-first, output-second). Hidden spins start at -1, as M5b declares;
  the first hidden block forgets them exactly, so the law after K sweeps
  depends only on x and the incoming output y0.
- Cells: K in {1, 2, 4} x y0 in {-1, +1} = 6 cells, each covering all 1,024
  blanket inputs. `SamplingSchedule(n_warmup=K, n_samples=1,
  steps_per_sample=1)`, so the one recorded state is the state after exactly
  K sweeps.
- N = 65,536 independent chains per input per cell (6.3 x 10^6 chains per
  cell, 4.0 x 10^8 chain-sweeps in total). One recorded sample is one chain's
  (w, y) state after K sweeps at one fixed input. Each chain has its own JAX
  key folded in from one root seed (20261004), the cell index and the input
  batch; the key is split into an init key and a sampling key, and the init
  key is discarded because the protocol's initial state is deterministic.
- Statistics per cell: (i) the maximum over inputs of |P_hat(y_K = +1 | x,
  y0) - p(x)| against the archived law `powered_rates` (q(x) +
  lambda(x)^K (1[y0 = +1] - q(x))); (ii) the maximum over inputs of the total
  variation between the 256-bin (w, y) histogram and the study-local joint
  law p0 T_x^K. The study-local joint must reproduce the archived output law
  to 1e-10 at every input before anything runs (observed 2e-15).
- Tolerance per cell: the 0.999 quantile of the same worst-over-inputs
  statistic that an exact multinomial(N, law) sample itself shows, from 4,000
  (output) and 1,000 (joint) exact-side draws with a fixed NumPy seed. With 6
  cells and two statistics, about 0.012 false failures are expected under the
  null. Observed tolerances: 0.0079 to 0.0091 on the output rate, 0.019 to
  0.022 on the joint TV.

What could cap the metric. The kernel's slowest input contracts at
lambda = 0.78 per sweep, so at K = 4 the laws for K, K-1 and K+1 still
differ by 0.055 to 0.082 against a tolerance near 0.009; the K cells keep
their power to distinguish K from its neighbours, unlike E0's fast chain.
Float32 on the THRML side places a floor near 1e-7 in the conditionals, far
below the tolerances. N fixes the tolerance and the runtime (about 100 s per
cell at N = 65,536); a mismatch smaller than 0.008 in the output rate is not
detectable here. One site, one seed, one arm: the result is about this
kernel's execution, not about every compiled site.

## Negative controls

A control is a wrong exact reference. Before sampling, every control must
separate from the right law by more than the cell's output tolerance; the
runner stops with `controls-gate.json` otherwise. After sampling, the same
wrong references must be rejected by the samples.

| Convention | Control | Wrong reference |
| --- | --- | --- |
| What counts as a sweep | `off_by_one` | p0 T^(K+1); K-1 also reported as the nearest-law diagnostic |
| Block order | `output_first` | output block then hidden block |
| Clamp value | `negated_inputs` | every blanket input negated |
| Finite K vs limit | `marginal_limit` | the exact marginal q(x) (K -> infinity) |

24 checks (4 controls x 6 cells). Observed exact separations before the
first THRML call: 0.055 (off-by-one at K = 4) to 0.93.

## Outputs and completion

`results/<dir>/`: `study.json.gz` (request, bounded exact references,
controls, histograms, evaluation), `summary.md`, `provenance.json`,
`run.log`, and `completion.json` written last by the runner. The per-input
laws are dropped from the archive; replay recomputes them.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run python -m thermo_lab.thrml_m5a_kernel_inner_sweep \
  --output-dir results/thrml-m5a-kernel-inner-sweep
uv run python -m thermo_lab.thrml_m5a_kernel_inner_sweep \
  --replay docs/experiment-reports/2026-10-03-thrml-m5a-kernel-inner-sweep/study.json.gz \
  --output-dir results/thrml-m5a-kernel-inner-sweep-replay
```

Replay authenticates the M5b archive, recomputes the exact laws, tolerances
and control separations from the archived request, rejects any drift, and
re-evaluates the archived histograms against the result digest. `--light`
reuses the archived tolerances instead of redrawing them (about 20 s instead
of 2.5 min); the unit test uses it, the gate uses the full replay.

`completion.json` must show `status=thrml_m5a_kernel_inner_sweep_complete`,
`cells=6`, `inputs_per_cell=1024`, `chains_per_input=65536`,
`controls_gate_passed=true`, `autosave="none; ..."`, and the counts of output
cells passed, joint cells passed, cells nearest to the exact K, and controls
rejected. A failed cell is a scientific result about THRML's execution of
this kernel or Thermo's reading of it, not an integrity failure.
