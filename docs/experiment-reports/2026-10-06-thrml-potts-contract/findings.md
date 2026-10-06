# THRML's categorical sweep is Thermo's Potts kernel: 84 of 84 cells match p0 T^K

THRML 0.1.4's `CategoricalGibbsConditional`, running a three-label Potts model
on a six-site patch with a three-colour block schedule, reproduces the exact
729-state law p0 T^K at every tested budget K in {0, 1, 2, 3, 4, 8, 16}, for
both block orders, both initial laws, both categorical factor classes and a
clamped arm. A two-label categorical encoding of E0's five-spin chain
reproduces E0's own exact spin kernel. **All 84 cells pass** their
predeclared tolerance (0.999 multinomial quantile at N = 400,000 chains per
cell). The negative-control gate passed on the exact side before sampling, and
the samples reject **112 of 112** wrong references. All 44 cells where the
off-by-one comparison is decisive sit closest to p0 T^K. The result is
positive: no convention mismatch.

Exact references are `exact_reference` (float64 NumPy, 3^6 and 2^5
enumeration, kernel powers). THRML cells are `software_simulation` (THRML
0.1.4, CPU, float32). Nothing here is hardware evidence; sweep counts are not
device operations. The
[frozen protocol](../../experiments/thrml-potts-finite-sweep-contract.md)
lists every fixed choice. The run took 167 s on CPU from commit `8c518f4`
(clean tree): 9.0 s exact side, 53 s of compilation and 86 s of execution
summed over cells (at most 1.1 s and 3.7 s per cell), and 15 s in-run replay.
Replaying the archive reproduces the request digest, result digest, archive
SHA-256 and every count in `completion.json`; only `provenance_digest`
differs, because a replay records its own provenance.

## Per arm

| Arm | Cells | TV to p0 T^K (K >= 1) | Tolerance | Largest TV / tolerance | Largest per-site marginal error |
| --- | --- | --- | --- | --- | --- |
| Potts, q = 3, 729 states | 56 / 56 | 0.0087 to 0.0126 | 0.0108 to 0.0131 | 0.97 | 0.0027 |
| Clamped (site 1 = 2) | 14 / 14 | 0.0062 to 0.0079 | 0.0080 to 0.0089 | 0.89 | 0.0026 |
| Bridge, q = 2 E0 chain | 14 / 14 | 0.0022 to 0.0043 | 0.0049 to 0.0052 | 0.86 | 0.0018 |

The K = 0 cells check only Thermo's own initial draws and pass trivially
(point masses) or at the uniform law (TV 0.016 to 0.017 against 0.018).

The two factor classes are indistinguishable. The mean TV / tolerance over
K >= 1 cells is 0.880 (generic) against 0.872 (square) on the Potts arm, 0.836
against 0.807 clamped, and 0.660 against 0.695 on the bridge.
`SquareCategoricalEBMFactor`, which merges interaction groups and is the
class performance code will use, executes the same kernel.

## What each convention check found

| Control | Wrong reference | Samples' TV to it | Exact separation | Rejected |
| --- | --- | --- | --- | --- |
| `negated_energy` | beta -> -beta | 0.935 to 0.987 | 0.936 to 0.987 | 16 / 16 |
| `doubled_beta` | spin-style factor 2 in the softmax | 0.309 to 0.399 | 0.310 to 0.398 | 16 / 16 |
| `transposed_w` | W_e[c_b, c_a] | 0.575 to 0.701 | 0.576 to 0.700 | 16 / 16 |
| `label_shift` | labels read as c + 1 mod 3 | 0.778 to 0.814 | 0.778 to 0.813 | 16 / 16 |
| `reversed_order` | the other order's kernel | 0.066 to 0.350 | 0.066 to 0.348 | 16 / 16 |
| `off_by_one` | p0 T^(K+1) | 0.027 to 0.278 | 0.024 to 0.278 | 16 / 16 |
| `clamp_as_zero` | free sites see site 1 as label 0 | 0.382 to 0.482 | 0.382 to 0.482 | 4 / 4 |
| bridge `doubled_beta` | E0 kernel at 2 beta | 0.170 to 0.173 | 0.170 to 0.173 | 4 / 4 |
| bridge `label_swap` | label 0 read as spin +1 | 0.155 to 0.221 | 0.155 to 0.220 | 4 / 4 |
| bridge `off_by_one` | p0 T^(K+1) | 0.007 to 0.129 | 0.006 to 0.129 | 4 / 4 |

Tolerances are at most 0.013 on the Potts arm, 0.009 clamped and 0.005 on the
bridge. The tightest gated control is the bridge `off_by_one` at K = 2: exact
separation 0.0063 against a 0.0049 tolerance (1.29x), the same K = 2 neighbour
gap that E0 found on this chain.

- **Energy sign and softmax scale.** THRML's categorical conditional samples
  softmax(theta) with theta the sum of the factor weights at the node, so with
  weights beta h and beta W it draws P(c_i = k | rest) proportional to
  exp(beta * local field). There is no factor of 2, unlike the spin sampler's
  sigmoid(2 gamma). Both the sign flip and the spin-style factor are rejected
  in every cell.
- **Table orientation and labels.** A pair factor over (head, tail) node
  groups reads W[e, c_head, c_tail] for both endpoints; THRML moves the axes
  itself when the tail node is updated. Label k is the k-th entry of each
  table.
- **Three colours and block order.** Each order matches its own kernel. The
  reversed order is rejected in all 16 cells, including the uniform init at
  K = 2 (0.066 against 0.012).
- **What counts as a sweep.**
  `SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)` returns the
  state after exactly K updates of all three blocks, as E0 found for two
  blocks. Decisive cells are Potts all-zero K = 1 to 4, Potts uniform K = 1 to
  3, clamped K = 1 to 3 and bridge K = 1 and 2, each in both factor classes.
  From K = 8 on, the chain is within 0.0005 of stationary on the exact side,
  so K and K +- 1 are indistinguishable there; that is a property of the
  model.
- **Clamping.** A categorical node clamped to label 2 stays at 2 in every
  chain, and the free sites follow the conditionals given label 2.
- **Spin and categorical paths agree.** Writing E0's chain as h_i[c] =
  b_i s(c) and W_e[c_a, c_b] = J_e s(c_a) s(c_b), with label 1 as spin +1,
  reproduces E0's exact spin kernel at every K. Thermo can move an Ising model
  to categorical nodes, or mix the two in later work, without changing its
  kernel.

## Disclosed before the frozen run

- The exploratory probe (`docs/research/potts_contract_probe.py`) sampled
  THRML on this graph with different parameters (seed 0) before the protocol
  was frozen.
- A pipeline smoke at N = 20,000 with the controls gate bypassed (two
  off-by-one controls lack power at that N, as expected) ran end to end and
  passed 82 of 84 cells. The two exceedances were marginal (TV / tolerance
  1.03 and 1.002), both in square-class cells. A comparison of all K >= 1
  cells found mean TV / tolerance 0.848 (generic) and 0.845 (square) against
  0.836 for fresh multinomial draws from the same laws, so no change was
  made. That smoke is not evidence.
- The first frozen run used the committed protocol but an uncommitted runner
  (`git_dirty`). After the runner was committed, a re-run from the clean tree
  reproduced the identical result digest. The archive here is the clean run.

## What this does and does not settle

THRML's categorical Gibbs path, both factor classes, three-colour schedules
and categorical clamping now have a checked per-sweep contract against
Thermo's exact kernel. Potts stage B, a native inference or optimization study
on fresh Potts targets, can cite it as its convention check. Not tested: more
than three labels or three colours, uint8 overflow, mixed spin-categorical
factors, models beyond enumeration, THRML initializers, and float32 at scale.
Nothing here supports a speed, hardware or execution-cost claim.

## Files

- `study.json.gz` (0.86 MiB, SHA-256
  `cbc76fec1068627ad8625b76c3329e682119658f6a9d35afb5f2b282b9db2d4b`): request, exact references (42 laws, off-by-one
  neighbours, stationary laws, 56 control laws), 84 histograms, evaluation and
  the result digest.
- `summary.md`: the per-cell and per-control tables rendered from the
  archive.
- `completion.json`: 84 cells (56 Potts, 14 clamped, 14 bridge), 84 passes,
  112 of 112 controls rejected, 44 decisive cells all closest to K, replayed,
  no autosave layer.
- `provenance.json`, `run.log`: runtime (JAX 0.10.2 CPU, x64 off, Python
  3.11.15, commit 8c518f4) and per-cell compile/execute seconds.
