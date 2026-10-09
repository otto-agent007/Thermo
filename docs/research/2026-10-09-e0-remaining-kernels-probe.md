# THRML on the other 59 compiled M5a kernels: control power and cost (probe, exploration)

Exploration on 2026-10-09 for queue row `e0-remaining-kernels` (proposal
P-0003). Script: [`e0_remaining_kernels_probe.py`](e0_remaining_kernels_probe.py);
raw numbers: [`2026-10-09-e0-remaining-kernels-probe.json`](2026-10-09-e0-remaining-kernels-probe.json).
**Not recorded evidence.** No THRML histogram was evaluated against a law,
no archive, no replay, no gate. Probe-only seeds (NumPy 9900, JAX 9901). It
exists to find the fixed choice that bounds the metric and to size the run.

## What ran

- Exact side for all 60 compiled kernels of the M5c primary arm (reading B,
  variational, cap 1, seeds 0 to 4, sites 0 to 11), through stage B's
  study-local helpers and the authenticated M5b archive. For every kernel and
  every stage B cell (K in {1, 2, 4} x y0 in {-1, +1}): the study-local joint
  law against the archived `powered_rates` law, stage B's 0.999 output
  tolerance at N = 65,536 from 4,000 multinomial draws, and the exact
  separation of each of stage B's four wrong references.
- Stage B's joint-tolerance draw (1,000 draws) timed on one cell of the
  largest kernel by inputs x states (seed 1, site 9: 2,048 inputs, 256
  states).
- Stage B's `run_cell` timed on 32 inputs per cell, K in {1, 2, 4}, for five
  kernel shapes (n_h 3 to 8, 64 to 2,048 inputs) at N = 65,536. Every
  histogram summed to N.

Total 511 s wall on this box (i7-7700K), most of it exact-side enumeration.

## Findings

1. **The kernels differ far more than stage B's one site did.** Blankets
   range from 5 to 11 inputs (32 to 2,048 inputs per kernel), hidden counts
   from 1 to 8 (4 to 512 joint states), and lambda_max from 0.09 to 0.92.
   The 59 kernels hold 31,232 inputs, 30.5x stage B's 1,024. The study-local
   joint reproduces the archived law to 3.9e-15 on every kernel.
2. **Stage B's control gate is the fixed choice that bounds the metric.**
   Stage B stops the whole study if any control fails to separate by more
   than the cell's output tolerance. On the 59 kernels, 72 of 1,416 control
   checks do not separate (34 more separate by less than 2x the tolerance).
   They fall on 16 kernels, all fast mixers (lambda_max 0.09 to 0.46); every
   kernel above 0.46 separates everywhere. The failures are
   off-by-one (27 cells at K = 4, 4 at K = 2), the K -> infinity marginal
   (19 at K = 4, 3 at K = 2), output-first order (17 at K = 4) and negated
   inputs (2, both on seed 0, site 6). At seed 3, site 4 (lambda_max 0.09)
   the K = 4 law is within 2e-5 of K = 5 and of the limit: no N can separate
   them. Run unchanged, the study would stop before its first THRML call.
   This is a property of the kernels, not of THRML.
3. **K = 1 keeps full power everywhere.** Every control separates at K = 1
   on every kernel, by at least 5.8x the tolerance. So the per-sweep
   convention checks (sweep count, block order, clamp sign) stay decisive on
   all 59 kernels at K = 1; at K = 4 they are decisive only on slow kernels.
4. **Output and joint pass tests are unaffected.** Output tolerances are
   0.0059 to 0.0097. Pass or fail on the right law does not depend on control
   power; only the "is the sample nearer K than K + 1" reading does.
5. **Multiplicity.** 354 cells x 2 statistics = 708 tests at stage B's 0.999
   quantile give about 0.7 false failures expected under the null, against
   0.012 in stage B. A family-level rule is needed before the run.
6. **Cost.** THRML execution at N = 65,536 costs 0.019 to 0.029 s per input
   at K = 1, 0.034 to 0.054 at K = 2, 0.061 to 0.100 at K = 4 with JAX's
   default intra-op threads (about 3 cores busy). Pinned to one core it is
   2.1x to 2.2x slower (0.062, 0.108, 0.201 s per input on the 256-state
   kernel). Summed over the 59 kernels and both y0: about 10,600 s
   multithreaded, about 22,500 CPU-seconds single-core (6.3 CPU-hours).
   Joint tolerances scale with inputs x states: 31 s for one cell of the
   largest kernel, about 2,000 s for all 354 cells. Exact laws and output
   tolerances: 400 s. A full replay redraws the joint tolerances again.
   **Total about 7.5 to 8.5 CPU-hours**, at or over the 8 CPU-hour flag.
   The owner's 11 CPU-hour figure scaled stage B's wall time; stage B's
   inputs were among the largest, so the per-kernel mean is smaller, but its
   wall time was multithreaded.
7. **Archive.** Stage B's dense histograms (1.57 million bins) gzipped to
   1.19 MB. The 59 kernels hold 34 million bins, about 26 MB gzipped. A
   committed archive under 4 MB has to drop the dense joint histograms; the
   per-input output counts (187,392 ints) and per-input joint TVs fit in
   about 1 MB gzipped (measured on synthetic data of the same shape).

## What this changes for the protocol

- Make control eligibility per cell, fixed in advance: a control is
  **powered** in a cell when its exact separation exceeds 2x the cell's
  output tolerance; only powered controls must be rejected. Unpowered ones
  are reported with their separation and do not stop the study. The only
  pre-sampling stop is a K = 1 control without power, which the probe says
  will not happen.
- Report "nearest K" only in cells where off-by-one is powered.
- Fix a family-level decision rule for 708 tests before the run.
- Persist output counts and joint TVs in the archive; keep the dense
  histograms as a separately hashed, uncommitted sidecar.
- Autosave per kernel; the run is hours.
