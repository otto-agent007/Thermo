# Margin budget with clamped input copies: probe (P-0011)

2026-11-13. **Exploration, not a recorded study and not evidence.** Probe
for queue row `margin-budget-copies`, proposal 3 of the
[mechanism proposals](2026-11-12-mechanism-proposals.md). Every number is an
`exact_reference` conditional of the five-spin PAsymSwap kernel family of
[`cap_probe.py`](cap_probe.py), computed on a CPU. No sampling, no training,
no compiler, no hardware. Write energies are the sealed profile's
`calibrated_projection`.

**Finding: the law holds, and the probe answers the row.** With r clamped
copies per input bit and every coupling capped at c, the best leakage product
found lies between 0.63 and 2.44 times e^(−4cr) in all 24 cells (caps 1 to 4,
r in {1, 2, 4}, equilibrium and K = 4). The 10-times falsifier does not
fire. At equilibrium the floor is now a theorem for this family:
L₀₀·L₀₁ ≥ (1 − L₀₀)(1 − L₀₁)·e^(−4cr). The hidden-reweighting gap is closed
analytically at equilibrium and only checked numerically at K = 4. The
design number: 95 percent uniform-visitation survival over 500 operations
needs **c·r ≥ 4.29** by the bound; the kernels found reach it at c·r of
about 4.46 when r ≥ 2. A single input-hidden coupling acts like one more copy
of the bit it sees: that pair's product approaches e^(−8c), not e^(−4c).

The strict reading of the prediction ("within optimizer resolution") fails
at r ≥ 2 by a stable factor of 2 (0.3 decades) while the two search seeds
agree to 0.03 decades. That factor is a prefactor of the found frontier,
not search noise. It sits well inside the falsifier.

## Fixed choices and whether each can bound the metric

| Choice | Value | Can it bound the metric? |
| --- | --- | --- |
| Family | cap_probe's nine parameters: hidden field, two output fields, four input-output couplings, two hidden-output couplings; no output-output coupling | **Yes, by construction.** The theorem below holds because the hidden spin does not see the inputs. M4I's output-output coupling is a different family and is not covered. |
| Copies | r in {1, 2, 4}; copy m of input bit k couples to output i by J[k, m, i] in [−c, c] | This is what the row asks. Every conditional depends on the copies only through Σₘ J[k, m, i], whose range is exactly [−rc, rc], so the arm is searched exactly as the nine-parameter family with input couplings bounded by rc and all else by c. |
| Cap | c in {1, 2, 3, 4} on every field and coupling | **Yes.** It sets the exponent with r. Fields and hidden-output couplings stay at c; they do not enter the floor, but they bound reachable splits of L₀₀ against L₀₁ (the factor 2 at r ≥ 2). |
| Hidden arm | r = 1 plus one input-hidden coupling k on input bit 1, in [−c, c] | **Yes.** It raises the pair (00, 01) floor to e^(−4(c + c)); the uncoupled pair (00, 10) keeps e^(−4c). |
| Horizon | equilibrium and K = 4 uniform-reset hidden-then-output sweeps (cap_probe's convention) | Not at these caps: K = 4 matches equilibrium to within the search's spread. No theorem at K = 4. |
| Leakage definition | L₀₀ = P(out ≠ 00 given 00), summed over the three wrong words; L₀₁ = P(out in {00, 11} given 01); occupied leakage = max(L₀₁, L₁₀) | By definition. cap_probe's 1 − C₀₀ floors at about 1e-16, which would have capped the r = 4 cells; the sum agrees with it to 7e-15 where both are valid. |
| Search | L-BFGS-B in the box, cap_probe's penalty objective, occupied leakage ≤ ε with ε = e^(−2cr ± 1); two independent seeds, 6 starts each, alternating random corners and uniform points | **Could.** Local search gives upper bounds on the minimum. On the low side the theorem replaces the search; on the high side the found products are the evidence. |
| Survival | 460 empty-edge and 40 occupied-edge operations (20 per occupied input), one kernel, hop fidelity ignored, S = (1 − L₀₀)⁴⁶⁰(1 − L₀₁)²⁰(1 − L₁₀)²⁰ | **Yes.** The visitation counts are the cap note's 5×5 torus assumption; the grouped, context-weighted model is not bounded. |

## What ran

[`probe.py`](margin_budget_copies_probe/probe.py), committed as `269c2e2`
with every fixed choice in its docstring before it ran. That first run
crashed in post-processing (an underflow in the survival-bound helper)
after 1,146 CPU-seconds and saved nothing. The fix and a cut from 16 to 6
starts per seed, to stay under the 30 CPU-minute ceiling, are commit
`6a670f3`. Output: [`probe.json`](margin_budget_copies_probe/probe.json)
(107,103 bytes, SHA-256
`8967bd1ae554b7d9b316a38524a283910b9120f019ef8abb264dd069fa62abbe`).
The second run took 441 CPU-seconds, 116 s wall on 4 workers. With the
crashed run and a 39 CPU-second timing check, the probe used about 27
CPU-minutes.

1. **Cross-checks.** The probe's conditional matches `cap_probe.conditional`
   to 5e-15, which matches `thermo_lab.thermodynamic_kernel` to 2.5e-14
   (equilibrium and K = 1, 4, 30). The hidden arm matches brute-force
   enumeration of the five-spin Boltzmann law to 1e-15 and an explicitly
   built 8-state chain at K = 4 to 3e-16.
2. **Control.** `cap_probe.min_empty_leak`, imported unedited (60 starts,
   seed = cap), reproduces the archived r = 1 table at caps 2 to 4,
   equilibrium and K = 4, ε = 1e-2 and 1e-3: all twelve cells within
   0.994 to 1.053 times the archived values.
3. **Product search**, 24 copies cells and 8 hidden-arm cells.
4. **Survival search**, the same cells, maximizing S directly.
5. **Theorem check**, 500 random parameter draws per copies cell and horizon,
   plus every search result.

## The equilibrium floor, closed analytically

Write s for the input spins and J₁ᵢ for the summed couplings from input bit 1
to output i. The joint weight of (hidden h, outputs o) given the input is
exp(h_h·h + Σᵢ oᵢ(hᵢ + Σₖ sₖJₖᵢ + h·j_hᵢ)). Flipping bit 1 from 00 to 01
multiplies it by ρ(o) = exp(2 Σᵢ oᵢ J₁ᵢ), which does not depend on h. So
p(o | 01) is p(o | 00) tilted by ρ(o): the hidden spin's reweighting cancels
exactly. With q = p(· | 00),

L₀₁ = (q₀₀ρ₀₀ + q₁₁ρ₁₁) / Σ q ρ ≥ q₀₀ρ₀₀ / (q₀₀ρ₀₀ + q₀₁ρ₀₁ + q₁₀ρ₁₀),

and ρ₀₁/ρ₀₀ = e^(4J₁₁), ρ₁₀/ρ₀₀ = e^(4J₁₀), so with J\* = max(J₁₀, J₁₁, 0) ≤ rc,

**L₀₀·L₀₁ ≥ (1 − L₀₀)(1 − L₀₁)·e^(−4J\*) ≥ (1 − L₀₀)(1 − L₀₁)·e^(−4cr).**

The pair (00, 10) gives the same with bit 0. It needs no fixed hidden state,
holds for any hidden field and hidden-output couplings, and is reached
(ratio 1.00 to 1.03) by the best r = 1 kernels at caps 2 to 4. No
equilibrium random draw (0 of 6,000; smallest ratio 1.10) or equilibrium
search result (smallest ratio 1.003) violates it.

With an input-hidden coupling k on bit 1, ρ depends on h. Running the same
argument on the joint (h, o) gives a floor of e^(−4(J\* + |k|)) for the pair
(00, 01): the coupling adds to the margin budget like one more copy.

At K = 4 the chain's law is not a tilt of anything, and the argument does
not apply. The floor's form is only checked numerically there: every
copies-arm K = 4 search result sits above it (smallest ratio 1.003), as do
6,000 random draws (smallest ratio 2.08).

## Result

Best product found over e^(−4cr). Seed spread is |log₁₀| of the ratio of
the two seeds' best products.

| c | r | c·r | Equilibrium | K = 4 | Largest seed spread (decades) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0.633 | 0.639 | 0.00 |
| 2 | 1 | 2 | 0.950 | 1.10 | 0.07 |
| 3 | 1 | 3 | 0.995 | 0.998 | 0.03 |
| 4 | 1 | 4 | 1.02 | 1.00 | 0.30 |
| 1 | 2 | 2 | 1.84 | 1.76 | 0.03 |
| 2 | 2 | 4 | 1.60 | 2.00 | 0.01 |
| 3 | 2 | 6 | 2.00 | 2.00 | 0.00 |
| 4 | 2 | 8 | 2.44 | 2.00 | single seed at eq (1 of 4 searches feasible) |
| 1 | 4 | 4 | 2.00 | 2.00 | 0.01 |
| 2 | 4 | 8 | 2.00 | 2.00 | 0.00 |
| 3 | 4 | 12 | 2.00 | 2.00 | 0.00 |
| 4 | 4 | 16 | 2.00 | 2.00 | single seed at eq (2 of 4 feasible) |

Values below 1 at c·r = 1 and 2 come from the (1 − L₀₀)(1 − L₀₁) factor,
which is far from 1 when L₀₁ is near e^(−1); against the full floor the
r = 1 cells sit at 1.00 to 1.16 and the r ≥ 2 cells at 1.6 to 2.4. At
c = 4, r ≥ 2 and equilibrium the search often
failed to meet ε = e^(−2cr ± 1) (e^(−15) to e^(−33)), so those cells rest on
one or two feasible searches. They still agree with K = 4, where all four
searches were feasible.

**Hidden arm** (r = 1, one input-hidden coupling on bit 1):

| c | Coupled pair (00, 01) over e^(−4c), eq / K = 4 | Over e^(−8c), eq | Both pairs over e^(−4c), eq / K = 4 |
| --- | --- | --- | --- |
| 1 | 0.096 / 0.097 | 5.2 | 0.64 / 0.65 |
| 2 | 2.9e-3 / 3.3e-3 | 8.6 | 0.95 / 0.95 |
| 3 | 5.5e-5 / 7.7e-5 | 8.9 | 0.99 / 0.99 |
| 4 | 1.0e-6 / 1.9e-6 | 9.0 | 2.28 / 1.00 |

The coupled pair beats e^(−4c) by up to six decades and stays within 10
times of its own floor e^(−8c). With occupied leakage taken as the max over
both inputs, the uncoupled bit holds the product at e^(−4c), as the theorem
says. A hidden unit that sees an input buys margin additively, as one more
copy would; this probe tested no majority amplification.

## Design number: survival over 500 operations

The bound gives an analytic ceiling on uniform-visitation survival for each
c·r (equilibrium, no input-hidden coupling):

| c·r | 2 | 3 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- |
| Survival ceiling | 0.0075 | 0.511 | 0.913 | 0.998 | 1.000 |

The ceiling reaches 95 percent at **c·r = 4.29**. The proposal's "about 4"
was 0.3 short. The kernels the search found reach their ceiling at r = 1
(c = 3: 0.511; c = 4: 0.913) but sit at the factor-2 frontier at r ≥ 2
(c·r = 4: 0.879 at c = 1, 0.901 at c = 2). A factor 2 shifts c·r by
ln 2 / 4 = 0.17, so the found kernels need about **c·r = 4.46**. On the
grid the smallest c·r reaching 95 percent is 6 (c = 3, r = 2: 0.998 at
equilibrium, 0.998 at K = 4).

Integer choices that clear 4.46: r = 5 at cap 1, r = 3 at cap 2, r = 2 at
caps 3 and 4. Cap 4 with r = 1 cannot reach 95 percent in this family
(ceiling 0.913). That is consistent with M4I's 89.61 percent at cap 4, which
was a different family with an output-output coupling; this probe does not
explain M4I.

**Cost of copies.** Each clamped copy is a full write under the sealed
profile, 153.6 pJ (`calibrated_projection`). An operation that re-clamps its
two input bits writes 2r sites: 307.2 pJ at r = 1, 921.6 pJ at r = 3 (cap 2),
614.4 pJ at r = 2 (cap 3). Copies trade writes for cap; a larger cap trades
coefficient range or precision instead. Which is cheaper on Z1 depends on a
cap or precision cost that is not published, so no comparison is made here.

## Mechanism, model and limits

- **Intended model:** a gate that maps 00 to 00 and 01, 10 to a one-particle
  word, with bounded coefficients. **Representation:** r clamped spins per
  input bit, exactly equivalent to one input with coupling range rc.
  **Update process:** equilibrium, and K = 4 uniform-reset hidden-then-output
  Gibbs sweeps. **Estimator:** none; every number is an exact conditional.
- **Decision this informs:** how many input copies, or how large a cap, a
  compiled gate of this family needs for a 500-operation program to survive
  at 95 percent. The answer is c·r ≥ 4.29 as a bound, about 4.46 as found.
- **Digital baseline and headroom:** not a task claim. A digital gate does
  not leak, so there is no cost headroom to state; the row asks for a design
  rule, not an advantage.
- **Scope:** this kernel family only, single kernel, uniform visitation, hop
  fidelity ignored. The theorem needs the hidden spin to be blind to the
  inputs; with an input-hidden coupling the floor moves by that coupling.
  The factor-2 frontier at r ≥ 2 is a search result, not a certificate;
  the true minimum lies between the floor and it.
- **Novelty unchecked.** The tilt argument is elementary. No literature
  search was done and no novelty is claimed. The row's source card on
  noisy-gate reliability bounds (von Neumann 1956; Evans and Schulman) is
  owed before any protocol; this probe answers the row, so none follows,
  and the card is still not written.
- **No reopening.** This explains a closed result. It does not reopen the
  conservation line or the parked Thermalizers work, and it authorizes no
  compiler, pilot or sampled study.

## Reproduce

From a Thermo checkout, about 2 minutes on 4 cores:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run python docs/research/margin_budget_copies_probe/probe.py
```
