# Native adsorption eligibility probe: fixed choices

2026-10-20. Exploration, capped at 30 CPU-minutes. The owner selected the
adsorption recommendation in the research-decision conversation. This is the
cheap probe permitted by `docs/research-loop.md`, not a recorded study or a
new denoising premise. These choices are written before sampling.

## Question and expectation

Can the proposed native representation survive finite precision and static
per-node beta variation at one fixed read/update allocation? The likely
obstacles are uncertainty with only 64 retained states per trajectory and a
strong exact digital baseline. A pass would establish numerical eligibility
only. No minimum budget, material accuracy, or economic advantage is inferred.

## Model

- One open 3×3×16 box: 144 binary occupations, 327 nearest-neighbor edges.
- Grand energy `G = -sum_edges n_i*n_j + sum_i (phi_i-mu)*n_i`, beta=1.
- Literal positive wall field phi=3 when x is 0 or 2, or y is 0 or 2;
  phi=0 otherwise. Open ends. This is a repulsive-wall model-class screen,
  not a real-fluid calibration or reproduction of a published geometry.
- Spin map: `J=1/4`, `h_i=(mu-phi_i)/2+degree_i/4`, with actual boundary degree.
- Physical coordinates `(x+4*z,y+z)`, allowed offsets (1,0), (0,1), (4,1).
  No copies, auxiliary spins, global constraints or replica exchange.
- Determine three chemical potentials using the full-precision exact transfer
  reference at mean occupancies 0.2, 0.5, 0.8, Brent bracket [-10,10],
  absolute tolerance 1e-12. Freeze these points for every subsequent arm.
- Exact references: full precision, 6 bits, 8 bits. Reuse
  `constraints.quantize_array`. Coupling cap 0.25; field cap is the maximum
  absolute field across all three nominal models. This is a **family-wide
  split cap**, explicitly applied to arrays, not the layer's per-model
  `quantize(..., cap_rule="split")`. Record both caps and rounded arrays.
- Float64. At most 12 transverse bits in the transfer reference; at most
  16 sites in its independent brute-force test.

## Sampling allocation

- Arms: unrounded/unit-gain ideal control; 8-bit/±10% gain operating case;
  6-bit/±10% gain diagnostic. No 6-bit result can substitute for 8-bit.
- Three independent seed groups, each with empty and full starts at every
  chemical potential. Each gain field is reused across bits, starts and
  chemical potentials within its seed group; independent groups are the
  replication unit. Gain seed words `[20261020,seed,0]`, seed=0,1,2.
- Gains use `constraints.static_gains`, one uniform[0.9,1.1] draw per site,
  fixed throughout a trajectory. Ideal control uses all ones.
- Independent sampler stream words `[20261020,seed,arm,mu_index,start,1]`.
  Arms ideal/8-bit/6-bit are 0/1/2; chemical-potential indices 0/1/2;
  starts empty/full are 0/1.
- Two-color local updates, color 0 then 1; conditional probability
  `sigmoid(2*g_i*(h_i+sum_j J_ij*s_j))`. Reuse the constraint-layer law.
  Nonuniform gains are not assumed to preserve a Gibbs equilibrium.
- Exactly 4096 sweeps per trajectory. Burn in for 1024; retain 64 complete
  states, one every 48 additional sweeps. A retained state is not an
  independent sample. No adaptive thinning, restart, extra seed or extension.
- Retain compact means, eight consecutive batch means, lag-one diagnostics,
  and final-state hashes; no raw trajectory dump.

## Observable and uncertainty checks

Report loading and mean occupied-pair fraction at each chemical potential,
and working capacity (high minus low loading). Compare both rounded exact
references and actual sampled outputs with the nominal exact reference.
Do not compare jittered outputs only with rounded targets to hide bias.

Report pointwise two-sided 95% Student-t intervals across the three seed
groups both for each start separately and with starts averaged within each
group; working capacity uses paired groups. Check each start against the
target before pooling, so opposing transient biases cannot cancel.
These are exploratory, with only two degrees of freedom,
not simultaneous certification of a device population. Show empty/full
differences and autocorrelation/batch diagnostics separately. For each
observable, the paired three-seed 95% interval of full-minus-empty estimates
must also lie inside its tolerance: ±0.01 for loading and working capacity,
±0.02 for pair occupation. This start-agreement rule is fixed before running.

Numerical eligibility requires loading and capacity absolute error at most
0.01, pair error at most 0.02, supporting uncertainty, and agreement between
starts at this budget. A point estimate alone is insufficient. Failure or
unresolved uncertainty parks this allocation; no automatic larger study or
tolerance change. If the ideal control is unresolved, do not attribute the
failure specifically to hardware variation.

## Baseline and costs

Use an aggregate-only factorized 512-state exact transfer calculation,
checked against the per-site/per-edge reference and enumeration on small
open boxes. Propagate occupation and pair-count moments with the partition
vector instead of computing every edge marginal. Time setup and all requested observables,
five repeats per point; report timing as software execution only. A narrow
pore is deliberately a strong digital-baseline case. This probe does not
implement or benchmark a new production sampler, optimized GCMC or GPU code.

Per trajectory: 589824 algorithmic updates, 9216 node reads, 288 full node
writes (144 parameter programming plus 144 charged initialization writes),
zero clamps. No free on-chip reductions. Apply the sealed profile via its
existing pricing functions; label every joule `calibrated_projection`.
No clamp pricing uncertainty occurs because clamp count is zero.

Show a separate 1134-slot bounding-box sensitivity: program/initialize all
slots and let all slots update while reading only the 144 model sites.
It is a scenario, not a measured physical footprint or a cost bound. Actual
core access amplification, host work, idle/board power and I/O latency remain
unknown. Never infer a physical core shape from the synthetic lattice.

Digital total energy is unknown. MAC-only arithmetic at 0.1, 1, 10 pJ/MAC
cannot price log-add-exp, lookup tables, RNG, integer operations or memory.
Do not claim an energy ratio without a qualifying complete denominator.
The eventual practical target remains 3× lower total energy at matched
accuracy; residual host allowance would be `E_digital/3-E_array`.

## Persistence and stopping

One run, fresh JSON path, strict finite JSON, source and requested-input
hashes, package versions, git provenance and CPU/wall timing. A process CPU
limit enforces the 30-minute ceiling. Validate algebra, placement, the local
conditional law, transfer reference and evidence arithmetic before sampling.
Replay calculations from the persisted summaries; stochastic reproduction
is unnecessary for this decision probe. No production `completion.json`.

If this probe answers the promotion question, publish the result in a
docs-only PR and stop. Any subsequent recorded study requires a separately
reviewable protocol and approval of its concrete commit under the research
loop. No provisional proposal ID is invented here.
