# Fixed-budget cold-target sampling: exploratory protocol

Frozen before sampling on 2026-10-01. This bounded CPU experiment asks whether
one longer chain, five independent chains, or five-replica parallel tempering
estimates the same cold distribution more accurately per spin update.

## Fixed choices and structural limits

- Six weighted degree-three graphs: sizes 12 and 16, seeds 100, 101, 102.
  Each is a Hamiltonian ring plus a disjoint random perfect matching; integer
  weights are uniform 1..5. These seeds were not in the earlier pilots.
- Two targets per graph: zero fields and independent fields chosen uniformly
  from {-0.15, +0.15} with NumPy SeedSequence([20261003, n, seed]). Couplings
  are float32(-weight/5), fields float32, and cold inverse temperature is 4.
  Exact enumeration uses float64 arithmetic on these rounded coefficients.
- Zero-field marginals are structurally 0.5. Primary accuracy is therefore
  total variation of the joint law of spins 0,1,2,3 (16 bins); edge-correlation
  MAE and spin-marginal MAE are secondary. The weak-field targets prevent a
  symmetry-only solution. This projection cannot certify the full joint law.
- All arms use systematic single-site Gibbs, site order 0..n-1, implemented
  in JAX. This isolates the allocation strategy; it is not a THRML benchmark.
- Tempering ladder: inverse temperatures [0.25, 0.5, 1, 2, 4], unchanged for
  every graph. After each sweep attempt adjacent exchanges on alternating
  even/odd pairs, two attempts per sweep. Accept with log probability capped
  at zero: (beta_i-beta_j)*(q(s_j)-q(s_i)), where q = h.s + sum J*s_i*s_j.
- Sixteen independently seeded trials per target, root JAX key 20261003,
  fold in target index then trial index, split initialization/sampling keys.
  Initialize five uniform states. The long chain starts from the fifth state;
  the independent and tempered arms share all five. Fold method index into
  sampling keys. A trial, not an individual state, is the replication unit.

## Budget and observations

At each T in {64,256,1024}, run one cold chain for 5T sweeps, five independent
cold chains for T sweeps, or five tempered replicas for T sweeps. Every arm
spends exactly 5*T*n spin redraws per trial, including burn-in. Record initial
states and every completed sweep; tempering records after the exchange.
Discard the initial observation and first quarter of sweeps at each budget.
Thus long/independent arms retain 15T/4 cold states, tempering retains 3T/4.
Shorter budgets are prefixes of the full run. Pool independent cold chains
only within their trial. Count 2T swap attempts separately; no equal-runtime,
equal-energy, or hardware-operation claim follows from matching redraws.

## Evaluation and decision

Report all target/method/budget cells and all per-trial metrics. Compare mean
trial error per target, then average equally over the six graphs separately
for each field variant. Retain worst cases and exchange acceptance rates.
There are three graph seeds shared across sizes; field variants are paired,
not twelve independent problem-family replications. No state-based confidence
intervals or universal superiority claim. A useful exploratory signal is at
least a twofold reduction in primary mean error versus the better baseline
at the largest budget in both field variants; failure is a valid result.

Before the full run, validate a three-spin nonzero-field fixture by exact
enumeration: the composed Gibbs sweep preserves the target, replica exchange
satisfies detailed balance, and empirical cold probabilities of every arm
are within TV 0.04 using 16 trials and T=4096. A reversed swap sign must fail
the exact balance check. These are checks, not parameter selection.

## Evidence and execution

One runner, one replay test, one report. Persist the request before sampling,
including source/lock/protocol hashes and all realized graph parameters.
Save packed traces separately in NPZ. Label sampled results
`software_simulation` and enumerated references `exact_reference`. Record CPU,
versions and JAX x64 configuration. Separate compilation, warm-up, and median
of three synchronized full-batch executions; initialization, transfer and
scoring are excluded from batch timing. Repeated timings reuse identical
keys and are not extra trials. No GPU or new dependency.

Use a fresh output directory. Replay verifies hashes, initialization, fixture
checks, all exact references, budget counts and saved accuracy metrics;
it does not regenerate stochastic trajectories or historical timings.
Write completion last, after replay. Archive compressed evidence and the
unchanged runner. Expected runtime is minutes, below the autosave threshold.
