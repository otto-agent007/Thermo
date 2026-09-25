# The ±2 parameter cap sets a conservation ceiling

*Review note, September 23, 2026. This is exact-reference analysis of the kernel family, not a new study. No protocol, archive, or recorded result was changed. Script: [`cap_probe.py`](cap_probe.py).*

## Summary

The five-spin PAsymSwap kernel (one hidden spin, no output–output coupling, nine parameters) cannot keep empty edges and occupied edges both leak-free at once. Across every cap tested, the best attainable pair satisfies

**L₀₀ · L₀₁ ≈ e^(−4c)**

Here L₀₀ is the empty-edge leakage P(output ≠ 00 | input 00), L₀₁ is the occupied-edge leakage P(output ∈ {00, 11} | input 01 or 10), and c is the field/coupling cap. The relation holds both at equilibrium and at K = 4. At the current cap c = 2, the floor on the product is 3.4 × 10⁻⁴. Reaching M4G's 95% survival over 500 operations needs about 10⁻⁴ leakage per operation (see the Sep 21 research note), which puts the product near 10⁻⁷. The kernel family can reach that at c = 4, and easily at c = 6.

The cap was lowered from 4 to 2 on Sep 1 (see the "Checked-cap revision" section of `docs/superpowers/specs/2026-08-31-independent-pasym-swap-compiler-design.md`) so the fits would pass the K = 30 residual gate. The conservation failures recorded from M4C through M4G, and in the return-fixture pilots, are consistent with that change being the main cause.

## Result

Each cell below is the lowest empty-edge leakage found with occupied-edge leakage held at or below ε. The search used L-BFGS-B inside [−c, c]⁹ from 60 starts per cell: half at random corners of the box, half uniform.

| Cap c | K | ε = 10⁻² | ε = 10⁻³ | Product | e^(−4c) |
|---|---|---|---|---|---|
| 2 | eq | 3.32e-2 | 3.37e-1 | 3.4e-4 | 3.4e-4 |
| 2 | 4 | 3.42e-2 | 4.05e-1 | 3.4–4.1e-4 | |
| 3 | eq | 6.08e-4 | 6.16e-3 | 6.1e-6 | 6.1e-6 |
| 3 | 4 | 5.93e-4 | 6.23e-3 | 5.9–6.2e-6 | |
| 4 | eq | 1.11e-5 | 1.12e-4 | 1.1e-7 | 1.1e-7 |
| 4 | 4 | 1.12e-5 | 1.12e-4 | 1.1e-7 | |
| 6 | eq | 4.6e-9 | 3.9e-8 | ≈4e-11 | 3.8e-11 |
| 6 | 4 | 6.4e-9 | 2.7e-8 | ≈3e-11 | |

Across caps 2–6 the product matches e^(−4c) to within the optimizer's resolution. K = 4 is essentially the same as equilibrium, so low-leakage kernels at higher caps do not need long mixing.

## Why the product is fixed

Given the hidden spin, the two outputs are sampled independently, and each output's logit is a sum of terms bounded by c. Producing the anticorrelated outputs 01/10 from input 01 needs a strong hidden–output coupling. Producing 00 from input 00 regardless of the hidden spin needs input and field terms that overpower that same coupling. The margin for each of these two requirements is bounded by about 2c in logit units. Improving one requirement's margin spends the other's, and together they cap the achievable product at about e^(−4c). I haven't written this up as a formal proof. The numbers above are the evidence.

## What this means for 500-operation survival

On the 5×5 torus schedule, each site belongs to 4 of the 50 edge operations in every macrostep. So with a uniform visitation assumption, a run has about 460 empty-edge operations and 40 occupied-edge operations. With one kernel for every operation, total hazard is about 460·L₀₀ + 40·L₀₁. Minimizing that along the c = 2 frontier gives a hazard of about 4.8, which is **≈0.8% survival even with hop fidelity ignored**.

The recorded runs do better than that bound (M4E reached 8.28%) because the 37 grouped kernels can specialize: a group the particle rarely visits can push L₀₀ down. This bound therefore does not cover the grouped model. The trade-off still applies to every group the particle does visit, and specialization has not come close to 95%.

Spot check with hop fidelity (hop error ≤ 0.01) added:

- At c = 2, no fit I found reached more than about 0.1% survival under the uniform-visitation bound.
- At c = 6, K = 4, I found fits at about 97% survival for the two extreme-hop targets. The median target reached only 22% with the same search. The penalized search was unstable, so read these as proof that such fits exist, not as a frontier.

## Why cap 4 failed on Sep 1

Cap 4 failed because the compiler fit to equilibrium and landed on parameters that mixed slowly (K = 30 residual 0.41). The table shows fast-mixing, low-leakage kernels exist at cap 4 and cap 6 at K = 4. Lowering the cap passed the gate by making the whole family weaker, when the real problem was that the objective ignored mixing.

## Recommended next step

Recompile at c = 4 and c = 6 against the finite-K endpoint law directly: the same differentiated finite-K conditional that M3 validated, with the target-visitation-weighted path KL from the Sep 21 note as the objective. Keep the M4G quality contract unchanged. Before any sampled study, report the per-group L₀₀/L₀₁ pairs and the exact 500-operation survival. The step is cheap because it is all exact computation, and it answers the capacity question that the Sep 21 note left open.

## Limits

- The frontier comes from local optimization with multiple starts, not global certification. The measured products are upper bounds on the true minimum. The e^(−4c) match suggests they are tight, but that is not proven.
- The survival figures use uniform visitation and a single kernel. The grouped, context-weighted setup is not bounded here.
- Hop fidelity is only spot-checked, and only at c = 2 and c = 6.
- Larger caps may have hardware implications. The cap was a Thermo convention, not a published Z1 limit.
- The cross-check against `thermo_lab.thermodynamic_kernel` (equilibrium and K = 1, 4, 30) agrees to 2e-14.

## Reproduce

From a Thermo checkout:

```bash
uv run python docs/research/cap_probe.py
```

The run takes about 5 minutes on 2 CPU cores and prints the cross-check followed by the table above.
