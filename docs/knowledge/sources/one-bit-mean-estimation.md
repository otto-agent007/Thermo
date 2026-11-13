# Mean Estimation from One-Bit Measurements

```toml
id = "one-bit-mean-estimation"
title = "Mean Estimation from One-Bit Measurements"
kind = "paper"
status = "asserted"
last_checked = 2026-11-12

[source]
url = "https://arxiv.org/abs/1901.03403v4"
version = "v4"
authors = "Alon Kipnis and John C. Duchi (primary); also I. Lau and J. Scarlett; A. Alaghi and J. P. Hayes; T. Baker and J. P. Hayes"
note = "Card for a technique family, keyed to Kipnis and Duchi (v4 dated 2022-05-09). Read 2026-11-12 at abstract and introduction level: Kipnis and Duchi, arXiv:1901.03403v4 and the earlier 'Mean Estimation from Adaptive One-bit Measurements', arXiv:1708.00952; Lau and Scarlett, 'Order-Optimal Sequential 1-Bit Mean Estimation in General Tail Regimes', arXiv:2604.07796v2 (abstract); Alaghi and Hayes, 'Survey of Stochastic Computing', ACM TECS 12(2s), article 92, 2013 (abstract and introduction); Baker and Hayes on large-scale stochastic adders (abstract). A short search, not a survey."

[[claims]]
id = "C1"
status = "asserted"
text = "Kipnis and Duchi: when each Gaussian sample may be reported by one bit, the asymptotic squared error of any estimator of the mean is at least pi/2 times that of the unrestricted sample mean, so the one-bit constraint costs a factor pi/2 in effective sample size."

[[claims]]
id = "C2"
status = "asserted"
text = "Kipnis and Duchi: an adaptive scheme, in which each bit's threshold depends on earlier bits, attains the pi/2 factor; no adaptive scheme does better."

[[claims]]
id = "C3"
status = "asserted"
text = "Lau and Scarlett (2026): an adaptive estimator built only from randomized threshold queries is order-optimal for one-bit mean estimation, and when each one-bit query may depend on m local samples the communication cost falls by roughly a factor of m."

[[claims]]
id = "C4"
status = "asserted"
text = "Alaghi and Hayes (2013): stochastic computing represents a number as the fraction of ones in a bitstream processed by very simple circuits; each extra bit of precision doubles the bitstream length, so run time grows exponentially with precision."

[[claims]]
id = "C5"
status = "asserted"
text = "Baker and Hayes: summing many bitstreams is a key stochastic-computing operation; multiplexer adders are small but inaccurate, and accumulative parallel counters are accurate but large."
```

## Claim

Estimating a mean from one bit per sample loses only a constant factor,
π/2, against using the samples themselves, provided the threshold sits near
the mean, which an adaptive scheme arranges. Letting each bit summarize
several local samples divides the number of bits needed by about that
number. Separately, stochastic computing shows what a one-bit stream costs
when it is the number format: precision is bought with exponentially long
streams.

## Method

- A one-bit message is the indicator of a threshold on the sample. Its Fisher
  information about the mean is largest when the threshold is at the mean.
- Adaptive schemes move the threshold toward the running estimate.
- Dither (a random threshold) trades some of that information for robustness
  to not knowing where the mean is.

## Relevance to Thermo

- The proposal behind `aggregator-read-gain`
  ([note](../../research/2026-11-12-mechanism-proposals.md), proposal 1)
  computed an independent-site ceiling of (2/π)·k for reading one aggregator
  p-bit over k sites. **That ceiling is C1**: a block sum of k sites carries
  k site-reads of information, and one thresholded bit keeps 2/π of it. The
  host-adapted threshold that the row's probe found necessary is C2.
  Aggregating m samples before the bit is C3. None of the three is new.
- What a Thermo row can add is the constrained case these sources do not
  treat: sites correlated because a block must be contiguous on the 16-offset
  lattice, a logistic dither fixed by the p-bit, a bias limited by the field
  cap, and a slot that sees fewer than 16 sites of the model. P-0012 (PR #157)
  measured those; its number, not the mechanism, is the contribution.
- C4 is the caution for any tree of aggregators or relays: a value carried as
  a rate of binary events needs about ε^(−2) events for precision ε, as the
  CN101 card already records ([card](normal-cn101.md)).
- C5 is the caution for `directed-relay-latent`'s fan-in: adding many binary
  streams accurately is the expensive part of stochastic computing, and a
  thresholding unit is not an adder.

## Limits

- Abstract and introduction level only. C1 and C2 are for independent
  Gaussian samples and asymptotic risk. A block sum of 16 binary sites is
  discrete and far from asymptotic; the lab's exact Fisher calculation is the
  number to use, with this card as its interpretation.
- The communication-constrained setting prices bits. Thermo prices node reads
  and writes under a sealed projection, and a threshold change is a write.
- These sources assume the summarizing device is free and exact. On Z1 the
  aggregator is a p-bit reached by a one-way coupling whose cost is
  unpublished.
