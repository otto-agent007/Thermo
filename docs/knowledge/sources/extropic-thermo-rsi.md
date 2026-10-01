# First sparks of thermodynamic recursive intelligence

```toml
id = "extropic-thermo-rsi"
title = "First sparks of thermodynamic recursive intelligence"
kind = "blog_post"
status = "asserted"
last_checked = 2026-10-01

[source]
url = "https://extropic.ai/writing/baby-thermo-rsi"
version = "2026-10-01; SHA-256 of the extracted page text begins e012e16a3692e44f"
released = 2026-10-01
note = "Blog post only; no paper or preprint."
authors = "Neagoe, Verdon and Morton (Extropic, with Prime Intellect)"

[[claims]]
id = "C1"
status = "asserted"
text = "Post-training Qwen3.6-35B-A3B with GRPO for 100 steps on about 50 Hinton-era reproduction tasks, with reward 0.7 exec + 0.3 rubric, raised held-out reward from 0.127 to 0.361."

[[claims]]
id = "C2"
status = "asserted"
text = "Held-out chart, zero-shot for off-the-shelf models: GPT-5.5 0.747, Claude Opus 4.8 0.464, Qwen3.6 post-trained 0.361, Qwen3.5 post-trained 0.356, Qwen3.6 base 0.127, Qwen3-30B-Instruct 0.109, GLM-5.2 0.073."

[[claims]]
id = "C3"
status = "asserted"
text = "The post-trained model is highly competitive with much larger frontier models. This is the authors' reading; on their own chart GPT-5.5 and Claude Opus 4.8 score higher."
```

## Claim

Extropic post-trained Qwen3.6-35B-A3B (35B total, 3B active) with GRPO for
100 steps on about 50 coding tasks adapted from reproductions of Hinton-era
connectionist experiments (Boltzmann machines, wake-sleep, a bars RBM). The
reward is r = 0.7 · exec + 0.3 · rubric:

- **exec** is deterministic. 0.15 rewards code that runs. The remaining 0.85
  rewards reproducing the reference task's core metrics, such as FID or loss.
- **rubric** is scored by Nemotron 3 Super 120B as an LLM judge against
  criteria derived in advance from the original work: architecture match,
  procedure followed, reproducibility.

Held-out reward rose from 0.127 to 0.361. The post calls the result "highly
competitive with much larger frontier models". Its held-out chart (Fig. 2,
values from the figure's alt text; whiskers are standard error across
rollouts) reads:

| Model | Held-out reward |
| --- | --- |
| GPT-5.5 | 0.747 |
| Claude Opus 4.8 | 0.464 |
| Qwen3.6, post-trained | 0.361 |
| Qwen3.5, post-trained | 0.356 |
| Qwen3.6, base | 0.127 |
| Qwen3-30B-Instruct | 0.109 |
| GLM-5.2 | 0.073 |

The off-the-shelf models ran zero-shot. The stated next steps are agents that
devise new TSU learning rules, tests in simulation, then feedback from real
chips "as our first large-scale chips come online in 2027".

## Relevance to Thermo

- It is the closest outside analogue to Thermo's own agent research loop
  (`thermo-harness`), on the same public stack:
  [THRML](extropic-thrml.md), [Torx](extropic-torx.md) and
  [Thermalizers](extropic-thermalizers.md).
- The two loops check results differently. Extropic scores candidates with
  metric reproduction plus an LLM judge. Thermo scores against frozen exact
  evaluators and never infers an owner decision from a metric.
- The task suite behind it is public: see
  [hinton-problems](cybertronai-hinton-problems.md).

## Cautions

- Every number is `asserted`. The post-trained weights, the ~50 adapted tasks,
  the held-out split, the rubrics and the scoring code are not released.
- On its own chart, GPT-5.5 and Claude Opus 4.8 outscore the post-trained
  model. "Competitive" is the authors' reading.
- The research-loop and GRPO-step animations are labelled illustrative.
- The motivating "100x and 10,000x more efficient" range links to Extropic's
  Z1 post and to [arXiv:2510.23972](extropic-dtm-hardware.md). The 10,000×
  end is that paper's hardware projection, not a measurement.
- The tasks reproduce known experiments. Nothing here is a new algorithm or a
  hardware result.
