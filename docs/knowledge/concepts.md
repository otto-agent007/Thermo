# Concepts

Short definitions for terms that recur across the sources and Thermo's
studies. Each entry says where Thermo uses the idea. For evidence classes and
sample terminology, the [evidence policy](../evidence-policy.md) is
authoritative.

## Dynamics

**Gibbs sampling (single-site, block, chromatic).** Resample one spin, or a
set of conditionally independent spins, from its conditional distribution. A
*systematic sweep* visits sites in fixed order. Chromatic (block) Gibbs updates
each graph color in parallel, which is how THRML and the TSU execute. Thermo
counts complete sweeps and color phases separately from recorded states.

**Finite thermalization.** Running K sweeps instead of mixing to equilibrium.
The output distribution then depends on K and the start state. M2–M4 and M5b
study it for discrete kernels. Whitelam's finite *observation time* is the
continuous analogue.

**Overdamped Langevin dynamics.** dx = −∇V(x) dt + √(2kT) dW. The continuous
substrate of Whitelam's papers and of Normal Computing's hardware. With a
quadratic V it is an Ornstein–Uhlenbeck process.

**Ornstein–Uhlenbeck (OU) process.** Linear Langevin dynamics,
dx = −A(x − μ) dt + √(2kT) dW. Its transition density is Gaussian with a
closed-form mean and covariance, so it has exact references for any time.
Basis of thermodynamic linear algebra and `thermox`.

**Euler–Maruyama discretization.** x_{t+Δt} = x_t − ∇V(x_t)Δt + √(2kTΔt) ξ.
Simulators use it. It adds discretization bias that is separate from sampling
error, and a study should report the two separately.

## Training

**Trajectory likelihood (Onsager–Machlup action).** The log-probability of a
whole discretized path. Under Euler–Maruyama it is a sum of Gaussian terms in
x_{t+Δt} − x_t + ∇V(x_t)Δt. Whitelam trains by maximizing it for a target
path. When V is linear in the parameters, it is quadratic in them.

**Finite-horizon versus equilibrium training.** Optimizing the model's
behavior at the horizon it will actually run for, versus at its stationary
distribution. M3 checked the finite-horizon gradient for discrete sweeps. M4B
compared the two training laws at matched budgets and found no demonstrated
advantage.

**Reverse-trajectory (time-reversal) training.** Training a generator to
reproduce the time reverse of a noising process. It is the basis of diffusion
models and of generative thermodynamic computing.

## Error and mixing

**Stationary bias.** The distance between a compiled chain's stationary
distribution and the target. M5a measures it against coupling caps.

**Dobrushin coefficient ρ_D.** A contraction coefficient for a Markov kernel.
When ρ_D < 1, a per-sweep error η_sweep gives a guaranteed stationary bias of
at most η_sweep/(1 − ρ_D). In M5a this is the only quantity called a bound.

**SLEM.** Second-largest eigenvalue modulus of a transition matrix. It
estimates the mixing rate but doesn't give the Dobrushin bound. The
Thermalizers paper's plotted floor uses it (see the M5 research note).

## Thermodynamics

**Heat and work along a path.** For a potential V changed by parameters over
time, the work done on the system is W = ∫ ∂_t V dt and the heat released to
the bath is Q = W − ΔV (Sekimoto's stochastic energetics). Whitelam's
generative paper claims that reverse-trajectory training minimizes Q.

**Entropy production.** The log ratio of forward to time-reversed path
probabilities, which is zero on average only for reversible dynamics. For a
finite Markov chain with transition matrix P and stationary π, the rate is
½ Σᵢⱼ (πᵢPᵢⱼ − πⱼPⱼᵢ) ln(πᵢPᵢⱼ / πⱼPⱼᵢ). A systematic sweep of reversible
single-site updates is generally not reversible, so its rate is positive.

Heat and entropy production are properties of a model. They are not
device-energy measurements, and Thermo would not label them
`calibrated_projection` or `physical_hardware`.

## Hardware vocabulary

**p-bit.** A probabilistic bit whose state fluctuates with a tunable bias.
Z1's sampling cells are p-bits.

**TSU (thermodynamic sampling unit).** Extropic's term for hardware that runs
block Gibbs over p-bits. Z1 is Extropic's TSU. Thermo has no access to one.

**SPU (stochastic processing unit).** Normal Computing's term for hardware
built from coupled continuous oscillators, targeting OU and Langevin dynamics.
Thermo has no access to one either.

**Calibrated projection.** A cost computed from a cited, versioned hardware
model, such as the Z1 Appendix-B operation model. It is never a measurement.
