# Frozen-program conservation diagnostic

Predeclared September 16, 2026, following merged PR #25. This study asks where
one-particle conservation first fails, and how later returns to one particle
change the terminal leakage statistic. It does not tune or train parameters.

## Fixed inputs and budget

Use all three original M1 initial parameter matrices embedded in the committed
M4 seed records. Authenticate the complete sources through the existing archive
importer. Preserve 25 sites, 500 ordered operations, 37 groups, beta 1, float64,
and parameter bounds [-2,2]. Evaluate K=4 and local equilibrium. Equilibrium
is an oracle control without an assigned finite sweep cost. Do not use trained
M4B parameters or reuse its evaluation draws.

For each seed use 32,768 trajectories initialized with a particle at site zero.
The new stream is `SeedSequence([0x434F4E53, source_seed]).generate_state(1,
dtype=uint64)[0]`, passed to PCG64. Share one uniform vector per occurrence
between K4 and equilibrium by replaying that stream. Horizons are correlated;
source seeds are the replication units. All comparisons are descriptive, with
no statistical acceptance threshold or confidence intervals.

## Exact reference and sampled diagnostic

Marginalize the hidden endpoint bit to obtain each 4-by-4 visible transition
law. Compute local conservation failure probabilities separately for parents
00, 01, 10, and 11. Starting from site zero, propagate an unnormalized vector
of probabilities for the 25 single-particle states, killing a path on its
first exit from that sector. At an edge (l,r), outside mass survives only on
00 -> 00; mass at l or r survives only on outputs 10 or 01. Retain per-operation
survival probability and first-exit mass by parent. This bounded substochastic
recurrence is exact_reference (up to float64 arithmetic), not a full 2^25-state
terminal distribution. No path may re-enter this exact survival calculation.

Separately execute the existing joint-endpoint sampling law without killing,
repairing, or rejecting any draw. Retain per-operation particle histograms,
occupancy counts, local parent/output transition counts, first-exit counts by
parent, ever-exited counts, and returns from an invalid particle count to one.
The histogram's count-one bin includes returned paths; distinguish it from
never-exited survival. Check particle balance from the local transition counts.
Report terminal unbiased occupancy loss beside leakage, and the share that
ends valid after at least one earlier failure. Keep raw trajectories transient.

Local probabilities and the killed recurrence are exact_reference; empirical
histograms, first exits, terminal loss, and recoveries are software_simulation.
The study may identify contributing local transitions and accumulation, but
cannot establish optimal representational capacity, causal effects of caps,
or the benefit of a new loss/architecture. Endpoint draws are not executed
Gibbs updates or physical device operations. No hardware timing/energy claim.

## Integrity and outputs

Bind canonical requested input identities, source archive digests, parameters,
horizons, stream, and protocol in each request. Bind derived table-value digests
in the result alongside bounded per-operation evidence. Reload by authenticating sources, rebuilding tables, and replaying
the complete diagnostic; a repaired checksum alone cannot authorize changed
counts. Generate reports only from validated artifacts; write completion last.
Use a fresh destination. Require all three seeds and both horizons for a full
study. Preserve source identities and output artifacts in the repository.

Tests must compare the killed recurrence against exhaustive tiny full-state
propagation, distinguish exit-and-return from uninterrupted survival, validate
local creation/destruction accounting, reproduce existing sampler terminal
counts, and reject tampered evidence. Scientific outcomes are non-gating.

Request-identity clarification during review: computed tables belong in the
result identity, consistent with AGENTS.md's requested-input hash boundary.
The initial wording placed table values in the request; the implementation
already used the result boundary. This correction changes no parameter,
random stream, estimator, horizon, sample count, result, or scientific decision.
