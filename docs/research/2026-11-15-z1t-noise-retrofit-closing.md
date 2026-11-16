# Z1T-0 noise retrofit line: closed

2026-11-15. A closing note, not new evidence. The owner closed this line in
session on 2026-11-15 after three fine-tuning runs ended the same way.
Everything cited is `software_simulation` exploration on the released Z1T-0
checkpoint with output sampling noise only; nothing here is a physical-Z1
result or a recorded study. Sources: the
[first results note](2026-11-15-z1t-noise-finetune-results.md), the
[follow-up proposal](2026-11-15-z1t-noise-quality-followup.md) and the
[follow-up results note](2026-11-15-z1t-noise-quality-followup-results.md).

## What was asked

Extropic's Z1T computes each tanh-linear unit as the mean of N p-bit
samples. Can the released model be fine-tuned to tolerate N = 32, and at
what cost to its quality?

## What was found

Next-token loss in nats on OpenWebText, 64 windows of 256 tokens.

| Model | Noise-free | N = 32 | Recovery | Noise-free cost |
| --- | ---: | ---: | ---: | ---: |
| Released | 4.002 | 10.746 | | |
| Clean-trained control | 3.996 | 10.521 | 0.033 | none |
| Noise-trained | 5.422 | 6.003 | 0.703 | 1.420 |
| Noise-trained, embedding scale frozen | 6.424 | 7.082 | 0.543 | 2.422 |
| Noise-trained by distillation | 6.418 | 6.759 | 0.591 | 2.416 |

1. **The released model is unusable at 32 samples per unit**, and matched
   clean fine-tuning does not change that. Its loss is 9.184 at N = 128,
   6.901 at 512 and 4.916 at 2,048, so it is still 0.91 nats above its
   noise-free loss at 2,048 samples. This finding is evaluation only and
   does not depend on any training choice made here.
2. **Noise-aware fine-tuning recovers most of the gap at N = 32 and costs
   noise-free quality on every route tried.** The best point is the first
   run: 70 percent recovery for 1.42 nats. Freezing the embedding scale or
   distilling to the released model's hidden state kept a little less
   robustness and cost a further nat.
3. **Each noise-trained model reaches its own degraded noise-free loss by
   about 512 samples.** After fine-tuning, sampling noise is no longer the
   binding limit; the lost quality is.

By the reading rules set before the runs: partial recovery in all three,
and robustness at 32 samples and the original quality were not obtained
together.

## What the logs suggest, untested

All three runs ended with a similar amplification of the input to the first
normalisation, reached through different parameters. The product of the
embedding scale and the first DyT gain, relative to the released model, is
7.1 in the first run (scale 9.69, gain 4.42 to 3.24), 7.7 with the scale
frozen (gain 4.42 to 34.1) and 9.0 under distillation (scale 7.12, gain 4.42
to 5.55). That would explain why freezing the scale did not remove the
robustness. It is a reading of logged values. No experiment here tests it,
and it identifies no mechanism for the quality loss.

## What this does and does not cover

- A released model was retrofitted for 2,000 updates of four 256-token
  sequences, one training seed per run. Nothing was trained from scratch.
- Only the 2.2 million body parameters, the DyT gains and one embedding
  scale were trained. The embedding and the output layer, 1.2 billion
  parameters, stayed frozen because training them does not fit a 4 GB card.
  A frozen output layer cannot adapt to shifted hidden states, so part of
  the quality cost may come from that mismatch and not from the noise.
- Only output sampling noise is modelled. Input encoding, 4-bit weights,
  placement and analog error are relaxed.
- So the closed claim is narrow: **within this retrofit budget, with the
  embedding and output layer frozen, robustness at 32 samples cost at least
  1.4 nats.** It is not a claim that Z1T cannot be both robust and good.

## What would reopen it

Any one of these, by the owner's decision. Three variants have ended the
same way, so a fourth is not to be designed without it.

- Training a Z1T from scratch with sampling noise in the loop. This is the
  fairest test. A narrow model fits the local card, but may not behave like
  the 12,288-wide one.
- The same retrofit with the embedding and output layer trainable, which
  needs roughly 20 GB of GPU memory.
- A noise-trained checkpoint, or a stated sample count per unit, published
  by Extropic.

## What stays open

- Evaluation-only work on the released model is not part of this line: a
  measured map of loss against sample count per site, and the model under
  noise and 4-bit weights together.
- The held row `z1t-sampled-head` is unaffected. It concerns the output
  layer on a noise-free body.
- The archived weights of the two follow-up runs (steps 500 to 2,000) and
  the first run's final weights are kept outside git and can be read by
  `z1t_site_probe.py`.
