# Training thermodynamic computers by gradient descent

```toml
id = "whitelam-gradient-descent"
title = "Training thermodynamic computers by gradient descent"
kind = "paper"
status = "asserted"
last_checked = 2026-09-29

[source]
url = "https://arxiv.org/abs/2509.15324v1"
version = "v1"
released = 2025-09-18
authors = "Stephen Whitelam"
published = "https://doi.org/10.1073/pnas.2528413123"

[code]
url = "https://github.com/swhitelam/thermo_gd"
license = "none"
note = "C++. No license file, so Thermo may read it but not copy or vendor it."

[[claims]]
id = "C1"
status = "asserted"
text = "The parameters of a thermodynamic computer can be trained by gradient descent to perform a computation at a chosen observation time."

[[claims]]
id = "C2"
status = "asserted"
text = "On an image classification task, the estimated thermodynamic advantage over a digital implementation exceeds seven orders of magnitude. This is an estimate from a model of the computer, not a measurement."
```

## Claim

The parameters of a thermodynamic computer can be trained by gradient descent
to perform a computation at a chosen observation time. On an image
classification task, the paper estimates a thermodynamic advantage of more than
seven orders of magnitude over a digital implementation.

## Method

A network of real-valued spins evolves by overdamped Langevin dynamics in the
potential V(x) = Σᵢ (J₂xᵢ² + J₄xᵢ⁴) + Σᵢ bᵢxᵢ + Σ₍ᵢⱼ₎ Jᵢⱼxᵢxⱼ (the
potential written in the code README). Inputs are clamped spins. A digital
neural network is trained first, and its activations define an idealized
trajectory. Training maximizes the probability that the Langevin computer
generates that trajectory. The computation is read from the output spins at
the observation time.

## Relevance to Thermo

- It is the continuous-time counterpart of Thermo's finite-horizon work: M3
  checked exact finite-sweep gradients for a discrete Gibbs circuit, and this
  paper trains for a finite observation time rather than equilibrium.
- V is linear in the parameters, so the discretized trajectory log-likelihood
  is quadratic in them. A small instance therefore has an exact gradient and a
  closed-form optimum, which makes an exact contract possible. See backlog
  item E1.

## Cautions

- The seven-orders-of-magnitude figure is an estimate from a model of the
  computer, not a hardware measurement. Treat it like a `calibrated_projection`
  claim whose cost model Thermo has not reviewed.
- The released code runs inference from a trained `parameters.dat`. Its README
  says results vary run to run because the RNG is seeded from clock time.
  Without a license, Thermo may read it but must not copy or vendor it.
- The substrate is continuous and analog-like. It does not map directly onto
  Z1's discrete p-bits, so no Z1 conclusion follows from it.
