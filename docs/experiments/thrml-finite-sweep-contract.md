# THRML finite-sweep contract on the five-spin chain (E0 / A1)

**Status: frozen 2026-10-03 and run the same day.** The exact side, the
tolerance rule and the negative-control gate were fixed before any THRML
sampling. The chain count was raised once, on the exact side only, before the
first THRML call; the request records why. Change any value only through a new
protocol version. The study runs in about 80 s on one CPU, so it has no
autosave or resume layer; a fresh output directory is the restart.

## Question and scope

After exactly K ordered block-Gibbs sweeps from a declared initial
distribution p0, does THRML 0.1.4's state distribution equal p0 T^K, where T
is the exact 32x32 one-sweep kernel of the five-spin chain that the existing
THRML gate samples (`configs/experiments/thrml-ising-chain.toml`)?

The existing gate compares long-run marginals and total variation after 200
warm-up sweeps. Every finite-K statement that Thermo builds on THRML (M5 inner
thermalization, the K=30 PAsymSwap budget, future finite-budget studies)
assumes that one THRML sweep is the kernel Thermo writes down. That assumption
had not been checked. This study checks it. It is backlog item E0 and
dependency check A1 in the October research directions. Stage B of the
backlog draft (one M5a site through THRML) is out of scope here.

Evidence classes. Exact references are `exact_reference`: float64 NumPy, 2^5
enumeration, kernel powers. THRML cells are `software_simulation`: THRML 0.1.4
on CPU in float32. Nothing here is hardware evidence, and sweep counts are not
device operations.

## Fixed choices

Every value below is in the hashed request (`study_request()` in
`src/thermo_lab/thrml_finite_sweep_contract.py`).

- Model: the biases, couplings and beta = 0.8 of the pinned config, energy
  `-beta*(sum b_i s_i + sum J_ij s_i s_j)`, spins in {-1,+1}, numeric dtype
  float32 on the THRML side (declared in the request), float64 on the exact
  side.
- Free blocks {0,2,4} then {1,3} (`forward`) and {1,3} then {0,2,4}
  (`reversed`). Both are independent sets, so one block update is a product of
  single-site conditionals P(s_i=+1 | rest) = sigmoid(2 beta h_i), with h_i the
  local field. Each order is compared with its own kernel.
- Initial distributions: all spins -1 (point mass); uniform over 32 states;
  THRML's `hinton_init`. From the 0.1.4 source (`thrml/models/ising.py`),
  `hinton_init` draws each site independently with
  P(S_i = 1) = sigmoid(beta * bias_i), so Thermo models it as that product law.
  K = 0 checks the initializer alone and separates sigmoid(beta b_i) from
  sigmoid(2 beta b_i) (exact separation between the two laws 0.0454).
- Sweep counts K in {0, 1, 2, 3, 4, 8, 16, 30}, run as
  `SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)`, so the one
  recorded state is the state after exactly K sweeps.
- Clamped arms: spin 0 pinned to +1 and, separately, to -1 through THRML's
  `clamped_blocks`, with free blocks {2,4} then {1,3}; hinton init restricted
  to the 16 consistent states. The reference is the same kernel construction
  restricted to those states, and the stationary law is the conditional
  Boltzmann law.
- N = 400,000 independent chains per cell. One recorded sample is one chain's
  full five-spin state after K sweeps. Each chain has its own JAX key, split
  into an init key and a sampling key, folded in from one root seed
  (20261003) and the cell index. Chains are independent, so N is also the
  effective sample count.
- Statistic: total variation between the 32-bin empirical histogram and
  p0 T^K.
- Tolerance per cell: the 0.999 quantile of the TV that a multinomial(N,
  p0 T^K) sample itself shows, from 4000 exact-side draws with a fixed NumPy
  seed. With 64 cells, about 0.06 false failures are expected under the null.
- Off-by-one diagnostic per cell: TV of the histogram to p0 T^(K-1) and
  p0 T^(K+1) beside p0 T^K. The nearest law is reported, and the comparison
  is called decisive only when both neighbours sit more than the cell
  tolerance from p0 T^K on the exact side.

Cells: 2 orders x 3 inits x 8 budgets free, plus 2 clamped arms x 8 budgets
= 64 cells.

What could cap the metric. The chain is five spins at beta 0.8 and mixes in
two to three sweeps, so beyond K = 3 every init agrees with the stationary law
to within tolerance and the test loses its power to distinguish K from K+-1.
That is a property of the model, not of THRML; the K <= 2 cells and the
all-minus init carry the finite-K information. Float32 on the THRML side
places a floor near 1e-7 in the conditionals, far below the 0.004 to 0.005
tolerances. N fixes the tolerance: raising it sharpens the K = 2 and clamped
comparisons but never the K >= 4 ones.

## Convention checks and their negative controls

Each convention is an explicit arm. A control is a wrong exact reference that
must be rejected by the samples at the cell tolerance.

| Convention | Arm | Named control |
| --- | --- | --- |
| Energy sign | every cell | `flipped_j`: all couplings negated |
| Temperature | every cell | `inverse_beta`: beta replaced by 1/beta |
| Spin encoding | K = 0 hinton cells; single-site conditional unit test | sigmoid(2 beta b) product vs sigmoid(beta b) product |
| Initialization | K = 0 cells, all three inits | the other two inits' laws |
| Block order | both orders, each vs its own kernel | `reversed_order`: the other order's kernel |
| Clamping | clamped+ and clamped- arms | mass outside the consistent states; unconditioned law |
| What counts as a sweep | every K >= 1 cell | `off_by_one`: p0 T^(K+1), and the K-1/K+1 diagnostic |

Controls are evaluated at K in {1, 2} for every free init and order. The gate
before sampling: every control from the all-minus init must separate from the
right reference by more than the cell tolerance on the exact side. At N =
40,000 the reversed-order and off-by-one controls at K = 2 did not separate
(0.006 to 0.012 against a 0.0155 tolerance), so N was raised to 400,000 before
the first THRML call. Controls from the uniform and hinton inits at K = 2 are
reported but not gated: those inits start close to stationary, so the wrong
references converge as well. That is again a model property.

## Outputs and completion

`results/<dir>/`: `study.json.gz` (request, exact references, controls,
histograms, evaluation), `summary.md`, `provenance.json`, `run.log`, and
`completion.json` written last by the runner. Replay:

```bash
uv run python -m thermo_lab.thrml_finite_sweep_contract \
  --replay docs/experiment-reports/2026-10-03-thrml-finite-sweep-contract/study.json.gz \
  --output-dir results/thrml-finite-sweep-contract-replay
```

recomputes the exact side from the archived request, rejects any drift in
the references or tolerances, re-evaluates the archived histograms and
checks the result digest.

`completion.json` must show `status=thrml_finite_sweep_contract_complete`,
`cells=64`, `controls_gate_passed=true`, `chains_per_cell=400000`, and the
counts of cells passed, controls rejected and off-by-one-decisive cells. A
failed cell is a scientific result about THRML's kernel or Thermo's reading of
it, not an integrity failure.
