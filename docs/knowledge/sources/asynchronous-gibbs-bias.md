# Ensuring Rapid Mixing and Low Bias for Asynchronous Gibbs Sampling

```toml
id = "asynchronous-gibbs-bias"
title = "Ensuring Rapid Mixing and Low Bias for Asynchronous Gibbs Sampling"
kind = "paper"
status = "asserted"
last_checked = 2026-11-12

[source]
url = "https://arxiv.org/abs/1602.07415v3"
version = "v3"
released = 2016-02-24
authors = "Christopher De Sa, Kunle Olukotun and Christopher Re"
note = "ICML 2016, PMLR 48:1567-1576. Read 2026-11-12: abstract, introduction and contribution list of v3. The theorems and their constants were not read."

[[claims]]
id = "C1"
status = "asserted"
text = "Asynchronous (Hogwild) Gibbs sampling, in which updates read stale values of other variables, can be biased: unlike sequential Gibbs it does not always produce samples arbitrarily close to the target distribution."

[[claims]]
id = "C2"
status = "asserted"
text = "The mixing time of asynchronous Gibbs can be up to exponentially greater than that of the corresponding sequential chain."

[[claims]]
id = "C3"
status = "asserted"
text = "For models with bounded total influence, if sequential Gibbs reaches a small sparse variation distance from the target in O(n) steps, the asynchronous sampler reaches the same distance in O(1) more steps; sparse variation distance measures error on marginals of few variables, where total variation overstates the effect of bias."

[[claims]]
id = "C4"
status = "asserted"
text = "For models satisfying Dobrushin's condition (total influence below one), the mixing-time bounds of sequential and asynchronous Gibbs differ only by a factor of 1 + O(1/n)."
```

## Claim

Running Gibbs updates on stale neighbour values changes the chain: it can be
biased and can mix far more slowly. Both effects are controlled when each
variable's conditional depends weakly enough on the others, measured by the
total influence that appears in Dobrushin's condition.

## Method

- Staleness is modelled statistically: each update reads values that may lag
  by a bounded delay.
- Bias is measured on marginals of small sets of variables (sparse variation
  distance), not on the full joint.
- The bounds are in terms of total influence α, the largest summed effect of
  the other variables on one variable's conditional.

## Relevance to Thermo

- `directed-relay-latent`
  ([note](../../research/2026-11-12-mechanism-proposals.md), proposal 2)
  hypothesizes that a latent broadcast through one-way relay trees behaves
  like blocked Gibbs with a stale, noisy latent, with bias of the form
  staleness times influence. This card is the nearest published form of that
  hypothesis: C3 and C4 give the conditions under which staleness is
  harmless.
- **Those conditions point the wrong way for a global latent.** A template
  latent coupled to every pixel is a high-influence variable by design, and
  P-0008's posterior is the regime where one variable's value moves many
  conditionals. Dobrushin's condition (C4) should not be expected to hold, so
  C1 and C2 are the applicable warnings, not C3 and C4.
- A relay tree differs from the paper's model in two ways the row would have
  to handle itself: the latent's value is not only stale but passed through
  noisy binary relays, and the couplings are one-way, so the chain has no
  target distribution in the paper's sense until one is defined.
- The Aadit card records the related inter-device staleness ratio
  ([card](aadit-million-pbits.md), C2). Thermo's M5a and M5b already use
  Dobrushin-type quantities for compiled kernels; the same α would be the
  first thing to compute on the reduced instance.

## Limits

- Only the abstract, introduction and contribution list were read. The
  bias and mixing bounds are stated here qualitatively; their constants and
  exact hypotheses must be read before any row relies on them.
- The paper treats random-scan Gibbs on a symmetric model with bounded
  delay. Chromatic block updates, one-way couplings and relay noise are
  outside it.
- Nothing here is evidence for or against the relay construction. It says
  which published condition to check first.
