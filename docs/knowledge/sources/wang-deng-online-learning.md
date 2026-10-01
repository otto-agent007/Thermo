# Online local learning for generative thermodynamic computing

```toml
id = "wang-deng-online-learning"
title = "Online local learning for generative thermodynamic computing"
kind = "paper"
status = "asserted"
last_checked = 2026-09-29

[source]
url = "https://arxiv.org/abs/2609.15439v1"
version = "v1"
released = 2026-09-14
authors = "Huilin Wang and Weibing Deng"

[code]
note = "None linked from arXiv as of the last check."

[[claims]]
id = "C1"
status = "asserted"
text = "Generative thermodynamic computers can be trained online, updating parameters at each integration step with the reverse-path Onsager-Machlup objective."

[[claims]]
id = "C2"
status = "asserted"
text = "On MNIST, online and batch training reach comparable validation performance."

[[claims]]
id = "C3"
status = "asserted"
text = "Online-trained models dissipate less heat during sampling and need lower precision to store their couplings. Both are properties of simulated models."
```

## Claim

Generative thermodynamic computers ([Whitelam](whitelam-generative.md)) can be
trained online, updating parameters at each integration step with the
reverse-path Onsager–Machlup objective rather than accumulating gradients over
whole trajectories. On MNIST, online and batch training reach comparable
validation performance. Online-trained models dissipate less heat during
sampling and need lower precision to store their couplings.

## Relevance to Thermo

- Local, per-step updates are closer to what hardware could do without a host
  round trip, which the charter treats as a first-class cost.
- The precision finding echoes M5b's bit-precision comparison on discrete
  kernels. A shared question is how coupling precision trades against
  stationary bias.
- It is a natural secondary arm for backlog item E2 (online versus batch
  training on the same toy target).

## Cautions

- It is a very recent preprint. Nothing has been reproduced independently.
- The heat and precision claims are properties of simulated models.
