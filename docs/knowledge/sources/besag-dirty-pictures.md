# On the Statistical Analysis of Dirty Pictures

```toml
id = "besag-dirty-pictures"
title = "On the Statistical Analysis of Dirty Pictures"
kind = "paper"
status = "asserted"
last_checked = 2026-10-16

[source]
url = "https://doi.org/10.1111/j.2517-6161.1986.tb01412.x"
version = "JRSS B 48(3), 259-302, July 1986 (with discussion)"
released = 1986-07-01
authors = "Julian Besag"

[[claims]]
id = "C1"
status = "asserted"
text = "Iterated conditional modes (ICM), a deterministic site-by-site update to the locally most probable label given the data and the current neighbours, gives good reconstructions and depends mainly on the local characteristics of the Markov random field prior, not on its large-scale properties."

[[claims]]
id = "C2"
status = "asserted"
text = "The MAP estimate is a global choice under zero-one loss on the whole image; the marginal posterior mode at each pixel minimises the expected number of misclassified pixels, which is the right estimate when the loss counts pixel errors."
```

## Claim

Image reconstruction under a Markov random field prior can be done well by
local rules, because what matters for pixelwise error is the prior's local
behaviour. The paper argues for estimates matched to the loss: the marginal
posterior mode (MPM) for pixel error, rather than the global MAP.

## Method

ICM updates each pixel to its conditional mode given the data and its
neighbours, sweeping until no pixel changes. It is compared with simulated
annealing toward MAP on synthetic scenes that were not drawn from the
assumed prior.

## Relevance to Thermo

- Thermo's Bayes floor is the MPM error, which is the right floor for the
  pixel-error metric the DTM studies report. This card is the reference for
  that choice.
- The paper's thesis, that local characteristics dominate, is the reason a
  local digital denoiser is near-optimal on a single-MRF family. The
  `dtm-baseline-headroom` probe confirms it on Thermo's mixtures: raising
  flip noise widens the gap of a 7 x 7 window, but a full-image logistic
  closes it
  ([probe](../../research/2026-10-16-dtm-baseline-headroom-probe.md)).
- Headroom for a sampler over a local model needs a global latent, which a
  single MRF prior does not have.

## Cautions

- ICM is deterministic and finds a local optimum; it is not a sampler and
  is not an exact MPM.
- The paper's synthetic scenes and noise models differ from Thermo's
  mixtures; its numerical results do not transfer.
