# Error diffusion and quantization bias correction

```toml
id = "error-diffusion-quantization-bias"
title = "Error diffusion and quantization bias correction"
kind = "paper"
status = "asserted"
last_checked = 2026-11-05

[source]
url = "https://arxiv.org/abs/1906.04721v3"
version = "v3"
released = 2019-06-11
authors = "M. Nagel, M. van Baalen, T. Blankevoort and M. Welling (primary); also R. W. Floyd and L. Steinberg; M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos and T. Blankevoort; E. Frantar, S. Ashkboos, T. Hoefler and D. Alistarh"
note = "Card for a technique family, keyed to Nagel et al. 2019 (ICCV). Also read 2026-11-05: Floyd and Steinberg, Proc. SID 17(2), 75-77, 1976 (citation checked, text not read); AdaRound, arXiv:2004.10568v2 (ICML 2020, abstract); GPTQ, arXiv:2210.17323v2 (ICLR 2023, abstract)."

[[claims]]
id = "C1"
status = "asserted"
text = "Floyd and Steinberg (1976): quantizing grey levels to a few output levels site by site and passing each site's rounding error on to unprocessed neighbours preserves local average intensity, which nearest-level thresholding loses."

[[claims]]
id = "C2"
status = "asserted"
text = "Nagel et al. (2019, data-free quantization): weight rounding shifts the expected output of a layer, a bias in the error, and correcting that shift without data (bias correction), with per-channel range equalization, gives near-float 8-bit accuracy on common vision networks without fine-tuning."

[[claims]]
id = "C3"
status = "asserted"
text = "Nagel et al. (2020, AdaRound): rounding each weight to nearest is not optimal for the task loss; choosing up or down per weight from a second-order loss approximation (a binary optimization relaxed and solved layer-wise) quantizes ResNet18 and ResNet50 weights to 4 bits within 1 percent accuracy without fine-tuning."

[[claims]]
id = "C4"
status = "asserted"
text = "Frantar et al. (2023, GPTQ): quantizing weights one at a time and updating the not-yet-quantized weights to compensate the error, using approximate second-order information, brings 175-billion-parameter GPT models to 3 or 4 bits per weight with negligible accuracy loss."
```

## Claim

Rounding each coefficient to its nearest level minimizes each error alone,
not their combined effect. When many errors act on the same output, choosing
the rounding direction so that errors cancel (error diffusion, bias
correction) or so that they cost least under a model of the loss (AdaRound,
GPTQ) beats nearest rounding at the same bit count.

## Method

- Error diffusion carries each site's residual to sites not yet rounded, in a
  fixed order. The running sum of errors stays within half a step.
- Bias correction measures the mean output shift caused by rounding and
  subtracts it from a bias term.
- AdaRound and GPTQ choose roundings against a quadratic model of the task
  loss, which needs second-order information (data or a Hessian estimate).

## Relevance to Thermo

- P-0009's `diffuse_class_family` is error diffusion restricted to classes
  of equal nominal coefficient
  ([findings](../../experiment-reports/2026-11-02-adsorption-precision-requirement/findings.md)).
  It is the cheapest member of this family: it needs no data and no loss
  model, only the nominal coefficients. It keeps each class's sum within
  half a step, which cancels the coherent shift that nearest rounding
  produces when a whole class sits at the same fraction of a step.
- The family predicts where class diffusion should not help: when the
  coefficients that matter are distinct per site (no large classes), or when
  the loss depends on individual coefficients rather than class sums. That
  is the case the `rounding-rule-transfer` probe tests on P-0008's model.
- AdaRound and GPTQ are the step up if class diffusion fails: they target
  the loss, not the class sum. Thermo has exact losses for its small models,
  so a loss-aware rounding is computable there without data. Using one is a
  new rule and needs its own row.

## Limits

- The technique is known. What a Thermo study can add is a measured bit
  requirement on a specific model under a stated rule, not the rule.
- These sources concern deterministic networks and images. None concerns
  Boltzmann-machine couplings or equilibrium observables; transfer to them is
  what the lab's rows test.
- Only abstracts were read for C2 to C4, and only the citation for C1.
  Claims are stated at abstract level.
