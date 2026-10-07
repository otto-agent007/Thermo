# Large Associative Memory Problem in Neurobiology and Machine Learning

```toml
id = "krotov-hopfield-large-memory"
title = "Large Associative Memory Problem in Neurobiology and Machine Learning"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/2008.06996v3"
version = "v3"
released = 2021-04-27
authors = "Dmitry Krotov and John Hopfield"

[[claims]]
id = "C1"
status = "asserted"
text = "Dense associative memories and modern Hopfield networks are effective descriptions of a network with feature neurons and hidden (memory) neurons that interact only through two-body synapses between the layers; there are no synapses among feature neurons or among memory neurons. Integrating out the hidden neurons recovers the dense models."

[[claims]]
id = "C2"
status = "asserted"
text = "Model B: with hidden Lagrangian L_h = log sum_mu exp(h_mu), the hidden outputs are softmax(h), a contrastive normalization across the whole hidden layer, and the reduced energy is the log-sum-exp energy of 'Hopfield Networks is All You Need' (Ramsauer et al., 2020)."

[[claims]]
id = "C3"
status = "asserted"
text = "Dense associative memories permit storage and reliable retrieval of a number of memories exponential in the dimension of feature space; in all the paper's models capacity is also bounded by the number of hidden neurons."
```

## Claim

The many-body energies of dense associative memory do not need many-body
synapses. A network of feature neurons and hidden memory neurons, coupled only
in pairs across the two layers, has the same fixed points once the hidden
neurons are integrated out.

## Method

The paper uses continuous-time rate dynamics with an energy function built from
two Lagrangians, one per layer (equation 2). Model A has additive Lagrangians and
recovers the polynomial and exponential dense memories. Model B uses a softmax
in the hidden layer and recovers the attention-like modern Hopfield network.
Model C, a spherical-memory variant, is new. Published at ICLR 2021.

## Relevance to Thermo

- This is the closest published framing of Thermo's associative-memory study.
  Its two-body coupling is between layers only. The softmax of model B is a
  layer-wide normalization, not a set of pairwise synapses.
- Thermo's open question is whether stochastic binary hidden units with
  pairwise mutual inhibition can play the softmax's role on sampling
  hardware, and at what cost in sweeps. The paper does not address that
  question, so it is not answered here.

## Cautions

- The model is deterministic and continuous-time, with symmetric synapses
  assumed for the energy to exist. It is not a Boltzmann sampler, and its
  hidden units are real-valued.
- "Biological plausibility" is defined in the paper only as the absence of
  many-body synapses.
