# Z1T: sparse transformer-like models for probabilistic hardware

```toml
id = "extropic-z1t"
title = "Z1T: sparse transformer-like models for probabilistic hardware"
kind = "blog_post"
status = "asserted"
last_checked = 2026-11-15

[source]
url = "https://extropic.ai/writing/z1t/"
version = "read 2026-10-01; SHA-256 of the extracted page text begins fa554291f7b35f6f"
released = 2026-09-04
authors = "Verdon, Neagoe, Lockwood and Morton (Extropic)"
note = "Blog post. Open weights at huggingface.co/extropic-ai; training recipe at github.com/extropic-ai/sparse-transformers. Re-read on 2026-11-15 through an automated page reader that returns a summary of the page, not its raw text: C1 to C5 matched, C6 was confirmed and C7 to C9 were added. The page hash in `version` is from the 2026-10-01 read and was not recomputed."

[[claims]]
id = "C1"
status = "asserted"
text = "Z1 has 269,568 p-bits in 8 cores and 2,135,904 hardwired couplings, each p-bit of degree 16. It runs chromatic Gibbs sampling at a 50 MHz internal update clock and draws under 1 W."

[[claims]]
id = "C2"
status = "asserted"
text = "Z1T (gated convolutional attention, 4-bit weights, 4 incoming edges per output) needs about an order of magnitude more training FLOPs than GPT-2 for the same loss. The log-log fit reaches GPT-2-small's loss at about 9.5e19 FLOPs. The scaling results do not include activation quantization."

[[claims]]
id = "C3"
status = "asserted"
text = "Projected energy is 294.52 nJ per token (8.74 nJ Z1 sampling, 285.78 nJ FPGA) against an H100 at 40.9 uJ (10% MFU, about 139x), 8.17 uJ (50%, about 28x) and 4.09 uJ (100%, about 14x). Both sides exclude the final logit readout, which on the FPGA would bring Z1T to about 136.4 uJ per token. Z1-FPGA data movement and chip count are not modelled."

[[claims]]
id = "C4"
status = "asserted"
text = "The sparse tanh layers that run on Z1 alone are about 468x to 4,680x more energy efficient than the H100, from 100% down to 10% MFU."

[[claims]]
id = "C5"
status = "asserted"
text = "Serial latency is 58.8 us per token (about 17,000 tokens/s), against 102 us for a batch-1 torch.compile H100 run at 0.006% MFU. The authors say batching would make the H100 substantially more efficient and recommend Z1 for decode, not prefill."

[[claims]]
id = "C6"
status = "asserted"
text = "The latency estimate assumes 25 sequential sampling layers of 32 samples each at 50 MHz, 16.0 us per token, with the samples averaged to estimate each scalar tanh-linear output. The post calls this conservative because it does not count samples drawn in parallel. It is an assumption of the projected latency: the post does not report model quality at 32 samples, and the estimate excludes the final vocabulary-logit computation."

[[claims]]
id = "C7"
status = "asserted"
text = "The 58.8 us breaks down as 38.0 us of FPGA orchestration (38 operations at 1 us), 16.0 us of Z1 sampling, 3.2 us of reads, 1.3 us of writes and about 0.3 us of averaging, pooling, positional encoding and residual adds. Per token there are 8 FPGA-to-Z1 write steps and 5 Z1-to-FPGA read steps at 25.6 Gbit/s."

[[claims]]
id = "C8"
status = "asserted"
text = "The energy and latency model is for L = 4, D = 512, context 1,024, 4 heads, kernel size 4 and fan-in 4. Each output node reads 4 input values of 4 p-bits each (dy4p: four +/-1 p-bits weighted 1/4 to 1/32, 16 levels per sample), which uses all 16 couplings of a Z1 p-bit; a value is recovered as the mean over N samples. Constants: 1.3e-14 J per Z1 sample, 0.2 pJ per FPGA matmul operation, 3.0 pJ per FPGA scalar operation, 1.5 W FPGA static power, 0.177 pJ per H100 FP32 FLOP."

[[claims]]
id = "C9"
status = "asserted"
text = "Not stated in the post: whether the reported losses were evaluated with sampling noise, whether training was noise-aware, step counts and optimizer settings, and whether the scaling figure used quantized weights. The post does not name a Z1T-0 model; it says the open weights are from one of the larger Z1T training runs."
```

## Claim

Extropic maps a sparse transformer-like model onto Z1 and an FPGA
co-processor. Z1 computes sparse tanh-linear layers as averages of p-bit
samples, using a 4-p-bit dyadic encoding (dy4p) for continuous values. The FPGA
does the dense and arithmetic work. The headline is "over 100x energy
efficiency gains vs GPUs", with a scaling law in which connectivity is an
extra variable.

## Relevance to Thermo

- It is the source of the "100x" in the
  [thermo-RSI post](extropic-thermo-rsi.md), alongside the
  [diffusion-hardware paper](extropic-dtm-hardware.md) for "10,000x".
- Its energy figures are a hardware model. Under the
  [evidence policy](../../evidence-policy.md), anything Thermo derives from
  them is `calibrated_projection` and must carry the assumptions in C3.
- Z1's fixed degree-16 coupling graph is a real hardware graph to compare with
  M5c's synthetic offset lattice. M5c's complete execution costs remain open.
- The training recipe and weights are public, so the scaling claim (C2) is
  checkable in software, unlike the energy claims.
- C6 is why Thermo's Z1T noise runs use 32 samples per unit. Together with C2
  and C9 it frames their result: the released checkpoint was scored at the
  sample count the latency estimate assumes, a condition the post's reported
  losses do not cover
  ([results](../../research/2026-11-15-z1t-noise-finetune-results.md)). Those
  runs model output sampling noise only. Under C8 a unit's inputs are also
  sampled 16-level values, which the runs do not model.

## Cautions

- The ">100x" holds only at the H100's 10% MFU operating point and with the
  logit readout excluded. At 50% MFU the stated ratio is about 28x.
- The authors say the energy figures are projections from Z1's theoretical
  consumption, anchored to experiments with similar p-bits on X0. Z1 itself was
  not measured.
- The FPGA accounts for over 95% of the projected energy. The post's "up to
  1000x" is a hypothetical for a future chip without it.
- If more samples come from parallel p-bits, as C6 allows, the cost moves from
  time to p-bit count, and chip count is not modelled (C3).
- The scaling figure's caption says body parameters exclude the final
  vocabulary matmul, while the connectivity section says its counts include
  it (2026-11-15 read).
- The die figure gives 215,904 coupling parameters, while the text gives
  2,135,904 coupling edges. C1 uses the text's number; treat the figure as a
  likely typo.
