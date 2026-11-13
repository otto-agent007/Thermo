# Signal Propagation and Noisy Circuits

```toml
id = "noisy-gate-reliability"
title = "Signal Propagation and Noisy Circuits"
kind = "paper"
status = "asserted"
last_checked = 2026-11-12

[source]
url = "https://doi.org/10.1109/18.796377"
version = "IEEE Trans. Inf. Theory 45(7), 2367-2373, November 1999"
released = 1999-11-01
authors = "William S. Evans and Leonard J. Schulman (primary); also J. von Neumann; K. Y. Camsari, R. Faria, B. M. Sutton and S. Datta; J. Martens, A. Chattopadhyay, T. Pitassi and R. Zemel; G. Montufar, N. Ay and K. Ghazi-Zahedi"
note = "Card for a literature, keyed to Evans and Schulman 1999. Read 2026-11-12 at abstract level: Evans and Schulman (abstract; thresholds as restated in arXiv:2507.15108); von Neumann, 'Probabilistic Logics and the Synthesis of Reliable Organisms from Unreliable Components', Automata Studies, 43-98, 1956 (citation and secondary summaries, text not read); Camsari et al., 'Stochastic p-bits for Invertible Logic', arXiv:1610.00377 (abstract and figure captions); Martens et al., 'On the Representational Efficiency of Restricted Boltzmann Machines', NeurIPS 2013 (abstract and introduction); Montufar et al., 'Geometry and Expressive Power of Conditional Restricted Boltzmann Machines', JMLR 16, 2405-2436, 2015 (abstract). A short search, not a survey."

[[claims]]
id = "C1"
status = "asserted"
text = "von Neumann (1956): with gates that each fail independently with a fixed probability below a threshold, redundancy and majority restoration compute any function reliably at a constant-factor increase in depth; three-input majority restoration stops working at a gate error of 1/6."

[[claims]]
id = "C2"
status = "asserted"
text = "Evans and Schulman (1999): information about an input decays geometrically with the number of noisy gates it crosses, which gives a lower bound on the depth increase reliable computation needs and an upper bound on tolerable gate noise; for k-input gates reliable computation is impossible at gate error above (1 - 1/sqrt(k))/2."

[[claims]]
id = "C3"
status = "asserted"
text = "Camsari et al. (2017): a Boltzmann machine of p-bits with symmetric couplings can implement a truth table, with an overall interaction strength setting how strictly the table is obeyed; such gates can be joined by one-way (partially directed) connections into larger circuits such as a 32-bit adder; the error of an AND gate with clamped inputs is studied against an added noise field."

[[claims]]
id = "C4"
status = "asserted"
text = "Martens et al. (2013): a restricted Boltzmann machine with enough hidden units realizes any distribution up to an error that vanishes exponentially in the magnitude of its parameters, and some simple distributions cannot be represented efficiently when the weights are bounded."

[[claims]]
id = "C5"
status = "asserted"
text = "Montufar et al. (2015): conditional restricted Boltzmann machines have upper and lower bounds on the hidden units needed to approximate every conditional distribution within a tolerance, including deterministic conditionals."
```

## Claim

Two literatures bear on a stochastic gate with bounded coefficients. The
noisy-circuit line takes a gate's error probability as given and asks what
redundancy buys: majority restoration works below a noise threshold, and
information decays with depth however the circuit is built. The
Boltzmann-machine line asks what a network with given units and weights can
represent: approximation error falls exponentially with parameter magnitude,
and bounded weights cost hidden units.

## Method

- von Neumann multiplexes each wire into a bundle and restores it by
  majority after each gate. The gate error is an input to the analysis.
- Evans and Schulman bound the mutual information a noisy channel passes,
  then sum that bound over paths in the circuit.
- Camsari et al. design couplings so the truth-table rows are the low-energy
  states, and scale all couplings by one strength to sharpen the gate.
- The representational results are constructions and counting arguments on
  the model class, not on a sampler.

## Relevance to Thermo

- P-0011 ([probe](../../research/2026-11-13-margin-budget-copies-probe.md))
  proved a leakage floor e^(−4cr) for the five-spin PAsymSwap family at
  equilibrium. In this literature's terms the floor fixes a gate's error from
  its coefficient cap and fan-in, which is the quantity von Neumann and
  Evans and Schulman take as given. The two are complements, not the same
  result.
- **Novelty reading (2026-11-12).** That gate error falls exponentially in
  coupling magnitude is known (C3's interaction strength, C4's parameter
  magnitude). The search found no statement of the cap-times-fan-in floor for
  a conditional kernel with a hidden unit blind to the inputs. That is "not
  found in a short search", not "new": the argument is one paragraph of
  algebra and may exist elsewhere. P-0011 claims no novelty and this card
  does not add one.
- P-0011's clamped copies are fan-in redundancy before the gate. von
  Neumann's bundles are redundancy after it, restored by majority. The
  majority-amplification arm P-0011 did not test is the von Neumann
  construction; D-0010 names it as dropped.
- C3 is the nearest prior art for one-way couplings between Boltzmann
  blocks, which `aggregator-read-gain` and `directed-relay-latent` relax
  (rule 8). It is a simulation design, not a statement about Z1.
- C2 applies to any relay tree: a signal crossing d noisy binary relays loses
  information geometrically in d. That is the lossy fan-in stop
  `directed-relay-latent` pre-registers.

## Limits

- Abstract-level reading throughout; von Neumann's text was not read. The
  1/6 and (1 − 1/√k)/2 thresholds are taken from secondary restatements.
- The noisy-circuit model has independent, input-independent gate failures.
  A Boltzmann gate's error depends on its input, as P-0011's L₀₀ and L₀₁ do,
  so the thresholds do not transfer as numbers.
- None of these sources concerns finite sweep budgets, coefficient
  quantization or a priced write per input copy.
- Any Thermo row that builds on a gate-reliability claim should read the
  primary sources in full first.
