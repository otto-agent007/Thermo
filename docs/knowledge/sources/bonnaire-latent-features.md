# On the role of non-linear latent features in bipartite generative neural networks

```toml
id = "bonnaire-latent-features"
title = "On the role of non-linear latent features in bipartite generative neural networks"
kind = "paper"
status = "asserted"
last_checked = 2026-10-06

[source]
url = "https://arxiv.org/abs/2506.10552v1"
version = "v1"
released = 2025-06-12
authors = "Tony Bonnaire, Giovanni Catania, Aurélien Decelle and Beatriz Seoane"

[[claims]]
id = "C1"
status = "asserted"
text = "Restricted Boltzmann machines with binary hidden nodes and extensive connectivity suffer from reduced critical capacity, limiting their effectiveness as associative memories."

[[claims]]
id = "C2"
status = "asserted"
text = "Introducing local biases and richer hidden-unit priors (multi-state, ReLU-like) restores ordered retrieval phases and markedly improves recall, even at finite temperature."
```

## Claim

How well a restricted Boltzmann machine works as an associative memory depends
on its hidden-unit prior. Plain binary hidden units lower the critical capacity.
Local biases or richer hidden units restore retrieval.

## Method

The paper uses statistical mechanics of disordered systems (phase diagrams of
bipartite energy-based models) and draws connections to the Hopfield model.

## Relevance to Thermo

- It explains the associative-memory probe's finding that a Hebbian bipartite
  memory with binary hidden units recalls worse than Hopfield.
- It suggests a cheap arm with no inhibition: binary hidden units with a
  negative local bias. That arm stays bipartite, so it keeps two-block Gibbs
  sweeps.

## Cautions

- These are thermodynamic phase diagrams in a large-system limit. The abstract
  gives no finite-size recall numbers or mixing times.
