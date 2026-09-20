# Conservation after M4G: research findings and next experiments

Researched September 20, 2026 by three independent agents covering constrained
kernels, trajectory objectives, and official Torx/THRML capabilities. Primary
papers and official documentation were checked against the pinned local code.
This note proposes future work; it changes neither the frozen M4G protocol nor
its completed evidence, and reports no new fitting or simulations.

## Recommendation

Start with a **no-fit exact survival-gradient audit of the saved M4G checkpoints**.
It can separate objective conflict, weak parameter movement and shared-gradient
cancellation without another large stochastic study. Follow with a tiny joint
sampler feasibility experiment if we want to test architectural conservation.
A new ideal logical baseline would duplicate one we already have.

M4G showed that K30 can pass marginal-loss and local-fidelity requirements while
failing conservation. Trained terminal outputs are approximately 45% empty,
41% one-particle and 14% multiple-particle. All 60 cells fail the joint contract.
These results describe the fixed five-update procedure; they do not establish a
capacity limit or prove that a different objective would succeed.
See [the completed study](../experiment-reports/2026-09-20-task-quality-inference-budget/summary.md).

## 1. Differentiate the quantity we need to preserve

Finite-horizon safety has a direct dynamic-programming formulation: optimize the
probability of staying inside the allowed state set throughout the computation.
This supports a whole-path conservation objective instead of relying on terminal
marginal error. The control paper does not prove feasibility for our bounded,
shared nonlinear parameters. [Abate et al. (2008)](https://www.cs.ox.ac.uk/people/alessandro.abate/publications/APLS08.pdf)

**Our deduction for the existing implementation:** the killed chain already
tracks just 25 valid particle locations. Let Q_t be its substochastic transition
matrix, a_t=a_(t-1) Q_t, and S=a_500 1. A forward/backward calculation can
compute the gradient of log S through all operations, summing repeated uses of
each shared group. This avoids enumerating the full 2^25 state space. It still
measures conservation at completed-operation endpoints, as M4G does.

The first-exit probability decomposes exactly as

\[
1-S=\sum_{t=1}^{500} a_{t-1}(\mathbf 1-Q_t\mathbf 1).
\]

These weights are the actual surviving occupancies of the model. They differ
from M4F's fixed logical-context weights. M4F already tested local conservation
and asymmetry penalties; repeating that suggestion would add little.

**First proposed diagnostic, no fitting:**

1. At the original initialization and saved checkpoints, compute exact grad log S.
2. Check selected directional derivatives against finite differences on the
   bounded fixture before consuming the full checkpoints.
3. Compare its dot product with every actual projected M4G parameter displacement
   against the observed exact survival change.
4. Break contributions down by occurrence/group and first-exit particle creation
   versus destruction. Report displacement norms and shared-term cancellation.

Negative directions identify objective conflict at those checkpoints. Positive
but tiny movements support a limited conclusion about the attained procedure.
Neither establishes global impossibility or convergence.

A possible later objective is derived from unnormalized valid terminal masses
m_i=P(survives every operation and finishes at i). If q sums to one and r=m/S,

\[
J=\sum_i q_i\log(q_i/m_i)=-\log S+\mathrm{KL}(q\Vert r).
\]

This is our proposed joint valid-output objective, not a result supplied by a
cited paper. It prices survival and conditional target fidelity together without
choosing another local penalty coefficient. A pilot would need a new predeclared
horizon, update/step policy and endpoint selection; keep the original full-output
quality gates and count exact computation separately. Improving J alone would
not establish task success. Explicit constraints are also worth considering,
but guarantees for policy optimization do not automatically transfer to this
parameterized kernel family. [Achiam et al. (2017)](https://proceedings.mlr.press/v70/achiam17a.html)

A naive terminal survival reward can be nearly absent: from the existing M4G
probabilities, 32,768 trajectories imply approximately 0.00014, 0.12, 13.4 and
1,107 survivors at K1, K2, K4 and K30. These are arithmetic expectations, not new
simulations. A correct score estimator may therefore have poor finite-batch
signal; exact marginalization avoids that issue for this particular survival
quantity. General unbiased-gradient and baseline theory does not guarantee a
useful observed gradient in such a batch. [Schulman et al. (2015)](https://arxiv.org/abs/1506.05254)

## 2. Conservation can be structural, but it changes the sampler

Spin-exchange dynamics provide a number-preserving update primitive.
[Kawasaki (1966)](https://doi.org/10.1103/PhysRev.145.224)
Fixed-cardinality swap chains likewise stay inside allowed support, although
mixing results for strongly Rayleigh measures do not automatically apply to our
hidden-variable model or fixed schedule.
[Anari, Oveis Gharan and Rezaei (2016)](https://arxiv.org/abs/1602.05242)

**Repository-specific deduction:** our target has
p_ij=0.1 sigmoid(l_j-l_i), p_ji=0.1 sigmoid(l_i-l_j). Their sum is 0.1 and their
ratio is exp(l_j-l_i). Thus asymmetric hops are compatible with detailed balance
under one-particle weights proportional to exp(l_i).

An exact interpretation is: leave 00 and 11 fixed; for a singly occupied pair,
hold with probability 0.9, otherwise resample particle location from the two
Gibbs weights. This is algebraically the **existing M4F logical reference**, which
already has survival one and zero local errors. It is not a new baseline or a
native implementation of the nine-parameter Ising compilation. An ordered
composition preserves the shared equilibrium distribution but need not itself
be reversible.

A prospective constrained law could multiply the local Boltzmann weight by
1[y_0+y_1=x_0+x_1] and normalize on that support. It needs a joint output-pair
update and valid initialization: single-bit Gibbs under an exact cardinality
constraint cannot change either output without leaving the allowed sector.
Finite penalties suppress forbidden probability rather than making it zero.
Ground-state penalty constructions do not certify finite-temperature pathwise
conservation. [Lucas (2014)](https://arxiv.org/abs/1302.5843)

**Proposed tiny software feasibility experiment:** select one frozen group in
advance. Compare the current independent-output sampler and a joint constrained
sampler at K1/K4 for all four clamped parents. Use the same count-valid starting
outputs in both arms and retain the historical uniform-reset law separately.
Enumerate the transition matrices, then compare actual pinned THRML draws with
the exact laws. Check forbidden-support zeros, normalization and directed-hop
errors. Count deterministic gating, retained state, pair writes and random draws.
Passing would establish a changed software law's implementation, not successful
thermal compilation or hardware savings. A whole-program constrained composition
would be a later experiment; current conditional-hop diagnostics already answer
part of its local-fidelity question.

## 3. What the public APIs support

The pinned THRML 0.1.4 implementation was inspected alongside official docs.
No dependency upgrade is needed for these software capabilities.

| Capability | Consequence |
| --- | --- |
| Ordered free blocks and fixed clamped blocks | Schedules and input clamping are explicit; changes define a new execution law. |
| Built-in Ising output block | It draws conditionally independent Bernoulli outputs; grouping two nodes does not enforce a joint cardinality constraint. |
| SuperBlock | Samplers read the same preceding global state; this does not introduce a coupled two-bit heat bath. |
| Custom conditional samplers and categorical nodes | A new sampler with count-preserving support can be implemented in software. |
| Added output-output coupling | Software graphs support it, but valid coloring or a joint update is needed because the outputs become dependent. |

Sources: [THRML block sampling](https://docs.thrml.ai/en/latest/api-block-sampling.html),
[THRML sampling tutorial](https://docs.thrml.ai/en/latest/01_all_of_thrml.html).
A custom sampler's public interface does not imply a matching device primitive.

The official announcement still describes Thermalizers as upcoming and early
API access as GPU simulation; Z1 system access is described prospectively.
Its announced sparse Ising/chromatic-Gibbs model does not establish native hard
constrained pair updates. [Extropic announcement](https://extropic.ai/writing/from-one-to-one-billion)
The Thermalizers paper's demonstrations are software. The separate Torx paper
includes X0/XTR-0 experiments, which do not validate this PAsymSwap chain on Z1.
[Thermalizing Stochastic Programs](https://arxiv.org/abs/2608.01615),
[Torx framework paper](https://arxiv.org/abs/2608.01612).

## 4. A useful architectural bound, with a precise limit

The following are deductions from the current energy and update rule in
[thermodynamic_kernel.py](../../src/thermo_lab/thermodynamic_kernel.py), not
additional experimental results or a cited-paper theorem.

At beta one with all parameters bounded by two, each output field has magnitude
at most eight. Let delta=sigmoid(-16). Since outputs are conditionally
independent given the final hidden spin, every completed sweep has local
count-change probability at least 2 delta (1-delta), approximately 2.25e-7.
Mixing over hidden states retains the bound. It gives S500 <= 0.99988747,
which is far too weak to rule out the required 0.95 survival.

A stronger row-specific relation follows because changing the one hidden spin
changes an output's log odds by at most eight. Conditioning on the other output
only changes mixture weights, so the visible row obeys

\[
p_{00}p_{11}\ge e^{-8}p_{01}p_{10}.
\]

For a singly occupied input, write f=p00+p11 and let h be unconditional hop
probability. Then

\[
f^2\ge4e^{-8}h(1-f-h).
\]

At h=0.05 this requires row leakage at least approximately 0.00795. This exposes
a real fidelity/conservation tension in the one-hidden-spin, uncoupled-output
family. It is **not a global impossibility proof**: the acceptance gate averages
hop error uniformly over groups, while survival follows changing visited
contexts. A useful global bound would need to connect both. This derivation
should receive dedicated mathematical review before becoming a research gate.

## Decision boundary for next work

The exact-gradient audit is the least disruptive first step. A joint-sampler
pilot answers a different architectural question and must have its own protocol.
Neither calls for changing M4G's completed evidence. The original M5 milestone
remains separate; these findings inform the choice of a follow-up rather than
silently adding training, new thresholds or an architecture claim.
