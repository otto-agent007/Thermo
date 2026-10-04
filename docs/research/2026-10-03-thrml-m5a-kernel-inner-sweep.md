# After E0 stage B: M5's finite-K matrices describe what THRML does

*October 3, 2026. CPU software evidence and exact references only.*

The [stage B study](../experiment-reports/2026-10-03-thrml-m5a-kernel-inner-sweep/findings.md)
ran one compiled M5a site kernel (M5c primary arm: reading B, variational,
cap 1, seed 0, site 1) through THRML 0.1.4 block Gibbs for exactly K in
{1, 2, 4} inner sweeps, clamped to each of its 1,024 blanket inputs as M5b
does. The state after K sweeps matches the archived M5b inner-K law in 6 of 6
cells on both the output rate and the 256-state (hidden, output) joint, at
tolerances of 0.008 to 0.009 and 0.019 to 0.022 drawn from the exact
multinomial at N = 65,536 chains per input. All 24 wrong references (K+1,
output-first order, negated clamps, the K -> infinity marginal) are rejected,
and the K+1 law sits 0.055 or more away even at K = 4, so the sweep count is
decisive at this site (lambda_max 0.78).

What changes. The M5b and M5c finite-K statements (stationary bias and target
TV at K in {1, 2, 4}) rest on a per-sweep contract checked on a kernel Thermo
compiled, with clamping and a two-block hidden/output schedule, not only on
E0's toy chain. B4 (THRML executing the M5c placed patch) can cite this as its
convention check and needs only the float32 tolerance question from A2.

What stays open. One site, one seed, one arm; 59 compiled kernels were not
run. Execution costs are not measured here and no number in this study is a
device operation. The tolerance rule was the most expensive step: 118 s of
exact-side multinomial draws against 29 to 103 s per THRML cell. Replay has a
`--light` path that reuses the archived tolerances; the gate still runs the
full redraw.
