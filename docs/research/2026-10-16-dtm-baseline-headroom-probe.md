# dtm-baseline-headroom probe (P-0007)

Date: 2026-10-16. Label: exploration, not evidence. Script:
[`dtm_baseline_headroom_probe.py`](dtm_baseline_headroom_probe.py). Output:
[`2026-10-16-dtm-baseline-headroom-probe.json`](2026-10-16-dtm-baseline-headroom-probe.json).
CPU only (`JAX_PLATFORMS=cpu`, single-thread BLAS), Python 3.11, probe seed
20261016 (never reused by a protocol). The three recorded runs used 1,610
CPU-s; with a 300-image smoke run the probe used about 28 CPU-minutes. No EBM
was trained.

Evidence classes in this note:

- The mixture data, test sets and Bayes MPM floor are `exact_reference`
  (float64 transfer matrix, the P-0006 `GridIsing` code imported unchanged).
- Ladder errors are `software_simulation` (float32 JAX on CPU, one training
  seed, 1,000 test images per family).
- Joules are `calibrated_projection`: the DTM's from the sealed Z1 profile
  through `dtm_small_image.price_sample`, the baseline's from MACs per image
  at a stated constant.

Nothing here is a hardware measurement.

## Findings

1. **Breaking locality works; raising noise alone does not.**
   - On the P-0006 family the 7 x 7 logistic is 0.004 above Bayes (0.0659),
     which reproduces P-0006 (Bayes 0.067, logistic 0.071).
   - Raising q to 0.3, 0.35 and 0.4 on that family widens the 7 x 7 gap to
     0.013, 0.024 and 0.034. The full-image logistic (290 parameters, 14,400
     MACs) stays within 0.006 to 0.008 of Bayes in each. A wider window
     recovers it, so higher noise only measures receptive field.
   - Zero field (no templates) at q = 0.35 behaves the same way: the
     full-image logistic is 0.004 above Bayes.
2. **Families whose posterior depends on component identity leave a gap.**
   Gap of the best declared rung to Bayes (standard error 0.002 to 0.003):

   | Family (L = 8) | Noisy | Bayes | Best declared rung, gap | Smallest rung within 0.01 |
   | --- | --- | --- | --- | --- |
   | P-0006 anchor, q 0.2 | 0.200 | 0.066 | cnn2 w8, 0.001 | logistic 7 x 7 (3,136 MACs) |
   | Anchor, q 0.4 | 0.399 | 0.140 | cnn2 w8, 0.006 | logistic full (14,400) |
   | Four templates, J 0.35, h0 0.5, q 0.35 | 0.350 | 0.219 | cnn2 w32, 0.016 | none |
   | Eight smooth templates, same | 0.349 | 0.214 | cnn2 w32, 0.035 | none |
   | Sixteen smooth templates, same | 0.352 | 0.239 | cnn2 w8, 0.024 | none |
   | Quadrant parity, J 0.3, h0 0.6, q 0.35 | 0.349 | 0.143 | cnn2 w32, 0.014 | none |
   | Quadrant parity, J 0.55, h0 0.3, q 0.3 | 0.299 | 0.096 | cnn2 w8, 0.010 | cnn2 w8 (128,000) |
   | Corner key, J 0.3, h0 0.6, q 0.35 | 0.347 | 0.129 | cnn2 w32, 0.014 | none |

   "cnn2 w32" is 27,329 parameters and 1.74 million MACs per image. In five
   families no declared rung, including that one, came within 0.01 of Bayes.
3. **The ladder's training budget bounds the metric.** This is the main
   finding.
   - One extra rung outside the row's list (the 8-wide two-layer net with
     each hidden layer's channel means fed back, a global-pooling net of
     3,881 parameters and 243,200 MACs) reached 0.0075 on quadrant parity
     and 0.016 on the corner key.
   - The same rung diverged on the four-template and sixteen-template
     families (gaps 0.093 and 0.222), at a fixed learning rate of 0.01 and
     2,000 steps. That is an optimisation failure, not a capacity limit.
   - Wide nets were trained for 2,000 to 3,000 steps at one learning rate
     and one seed. A gap of 0.014 to 0.035 under that budget does not show
     that no cheap digital model reaches the floor.
   - A protocol must therefore fix the ladder's learning rate by validation,
     train it longer, take the best of several seeds, and include a
     global-pooling rung. Otherwise the row would measure under-training
     instead of receptive field.
4. **Break-even MACs.** The DTM at P-0006's operating point (T = 2, K = 64,
   L = 8) costs 2.01e-8 J per sample and 3.21e-7 J per 16-draw estimate at
   the profile's write constant. With clamp loads at the read constant it
   costs 9.7e-9 J per estimate.
   - The baseline's cost equals that at 3.2e5 MACs per image at 1 pJ per
     MAC (9.7e3 MACs at the read-constant end).
   - At 0.1 pJ per MAC the break-even is 3.2e6 MACs (9.7e4 at the read
     end). At 10 pJ per MAC it is 3.2e4 MACs (970 at the read end).
   - Every rung that reaches the floor on a local family costs 3,136 to
     128,000 MACs. Against the 3,136-MAC logistic the DTM is cheaper only at
     10 pJ per MAC with clamp loads at the read constant. Against the
     128,000-MAC net it is cheaper at 10 pJ per MAC at both ends of the
     band, and at 0.1 and 1 pJ per MAC only at the read-constant end.
   - The verdict therefore turns on the clamp constant at every rung the
     probe saw reach the floor. That is the 33x band D-0003 warned about.
5. **Cost.**
   - The exact posterior at L = 8 takes 0.2 CPU-s per 500 images per
     component. At 16 components, 1,000 images take about 7 CPU-s.
   - A ladder fit takes 5 to 80 CPU-s (2,000 to 3,000 steps, batch 128). The
     two-layer 32-wide net is the most expensive.
   - At L = 16 the exact posterior costs about 0.38 CPU-s per image per
     component. The L = 16 DTM costs about 5 times the L = 8 DTM in THRML
     and 8 times in training (P-0006 probe, finding 7).
6. **Sizing for a study.** Ladder stage with a learning-rate grid, three
   seeds and nine families: about 3 to 4 CPU-hours. DTM stage on at most three
   candidate families at the recorded operating point: about 1 CPU-hour per
   family (scaled from P-0006's 4.8 CPU-hours for 168 cells). Total about 7
   CPU-hours at L = 8, under the 8 CPU-hour flag. L = 16 is not sized into
   the study.

## Fixed choices and whether each could bound the metric

| Choice | Probe setting | Could it bound the metric? |
| --- | --- | --- |
| Family kind | template mixtures, parity, corner key, zero field | **Yes.** Noise and window size alone give no gap; component identity does |
| Coupling J, field h0, noise q | J 0.3 to 0.6, h0 0 to 0.6, q 0.2 to 0.4 | Yes, through the floor and the gap. The study fixes the probe's families |
| Ladder list | the row's seven rungs | **Yes.** Without a pooling rung the ladder misses the cheapest global model |
| Ladder training | Adam, one rate, 2,000 to 3,000 steps, one seed | **Yes, binding.** Divergence and under-training inflate gaps |
| X (cost threshold) | not yet fixed; break-even at 1 pJ per MAC is 3.2e5 MACs | Yes: it decides candidacy. The protocol fixes it |
| Operating point | P-0006's T = 2, 4 bits, K = 64 | Sets the DTM's joule; not re-tuned by the row |
| Image size | L = 8 | Cost only; L = 16 is outside the probe |
| Test set | 1,000 images | Resolution: standard error 0.002 to 0.003 on a gap |
| Within-0.01 rule | point estimate, one seed | Resolution; the protocol pairs and replicates |

The fixed choice that bounds the metric is the ladder's training budget,
with the missing global-pooling rung next to it. The family kind decides
whether any gap exists at all, and component identity is the only lever the
probe found that does.
