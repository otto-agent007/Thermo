# Z1T-0 noise training: the clean-quality cost was not avoided

2026-11-15. Evidence: **software_simulation; exploration**. These are
owner-requested runs on the local GTX 1050 Ti, outside the loop. They are not
a recorded study and not a physical-Z1 result. The proposal and its reading
rule were written before the runs in
`2026-11-15-z1t-noise-quality-followup.md` (PR #170; not merged when this
note was written). They follow the closed row `z1t-noise-recovery` and its
[results note](2026-11-15-z1t-noise-finetune-results.md). No queue row,
training or hardware claim follows from this record.

## Outcome

Neither route kept robustness at 32 samples while reducing the clean-quality
cost. Both made the cost larger. By the pre-registered reading: within 2,000
updates of fine-tuning, robustness at 32 samples and the original quality
were not obtained together by either route. That does not show they are
incompatible for a model trained with noise from the start.

- `noscale` (embedding scale frozen at 1): recovery 0.543 on OpenWebText,
  which is **partial** (between 0.35 and 0.60). Robustness was not lost, so
  the rule does not name the embedding scale as the mechanism. Clean loss
  rose by 2.42 nats, 1.00 more than in the first run.
- `distill` (match the released model's noise-free final hidden state):
  recovery 0.591, which is **partial**, just under the 0.60 bar. The cost
  was 2.42 nats against the 0.36 bar for "avoided". Its noise-free loss is
  1.00 nats above the first run's noise-trained model.
- Neither run is better at the operating point. N = 32 loss is above the
  first run's 6.003 by 1.08 (`noscale`) and 0.76 (`distill`) nats.

## Numbers

Next-token loss in nats, mean ± SEM over 64 windows. N = 32 losses average
three noise draws per window. The first run's values are from its
[snapshot](2026-11-15-z1t-noise-results/summary.json).

| OpenWebText | Clean | N = 32 | Recovery | Clean cost vs start |
| --- | ---: | ---: | ---: | ---: |
| Start (released model) | 4.002 ± 0.065 | 10.746 ± 0.082 | | |
| First run, noisy arm | 5.422 ± 0.071 | 6.003 ± 0.067 | 0.703 | 1.420 ± 0.045 |
| `noscale` | 6.424 ± 0.073 | 7.082 ± 0.063 | 0.543 | 2.422 ± 0.063 |
| `distill` | 6.418 ± 0.064 | 6.759 ± 0.064 | 0.591 | 2.416 ± 0.058 |

| WikiText-103 | Clean | N = 32 | Recovery | Clean cost vs start |
| --- | ---: | ---: | ---: | ---: |
| Start | 5.101 ± 0.048 | 11.560 ± 0.055 | | |
| First run, noisy arm | 6.689 ± 0.047 | 7.167 ± 0.039 | 0.680 | 1.588 ± 0.032 |
| `noscale` | 7.495 ± 0.037 | 7.947 ± 0.032 | 0.559 | 2.394 ± 0.047 |
| `distill` | 7.274 ± 0.036 | 7.565 ± 0.037 | 0.619 | 2.173 ± 0.049 |

Paired differences against the first run's noisy arm, same windows (OWT):

| Run | Clean, run − first noisy | N = 32, run − first noisy | N = 32, run − first control |
| --- | ---: | ---: | ---: |
| `noscale` | +1.002 ± 0.030 | +1.079 ± 0.024 | −3.439 ± 0.040 |
| `distill` | +0.996 ± 0.029 | +0.756 ± 0.023 | −3.762 ± 0.040 |

Sample-count curve on OWT at the last step, one noise draw per window:

| N | 128 | 512 | 2,048 |
| --- | ---: | ---: | ---: |
| First run, noisy arm | 5.590 | 5.464 | 5.433 |
| `noscale` | 6.604 | 6.467 | 6.432 |
| `distill` | 6.522 | 6.444 | 6.425 |

As in the first run, each noise-trained model reaches its own degraded clean
loss by about 512 samples. More samples do not bring back the released
model's 4.002.

Training trajectory (OWT clean / N = 32 at each archived evaluation):

| Step | 500 | 1,000 | 1,500 | 2,000 |
| --- | --- | --- | --- | --- |
| `noscale` | 5.280 / 8.821 | 6.367 / 7.302 | 6.450 / 7.132 | 6.424 / 7.082 |
| `distill` | 5.383 / 7.724 | 6.248 / 6.901 | 6.418 / 6.803 | 6.418 / 6.759 |

Most of the clean-loss rise happens by step 1,000, the end of the N
annealing. The distillation loss (normalised squared error of the final
hidden state) fell from 1.47 at step 1 to 0.84 at step 2,000. The noisy
model never got close to the released model's hidden state.

## Pairing check

The rule allows paired comparisons only if each run's step-0 per-window
noise-free losses match the first run's `start.json` to 1e-3 nats. Both
runs match exactly (maximum absolute difference 0.0) on all 64 windows, for
OWT and WikiText-103, noise-free and at N = 32. The paired comparisons above
are therefore valid.

## What did not change, and what bounds the answer

Settings are the first run's noisy arm with one flag changed:
`--no-emb-scale` or `--loss distill`. That is 2,000 updates of four
256-token sequences, seed 0, body LR 3e-4, gain LR 3e-3, N annealed from
1,024 to 32 over the first half, and the same evaluation. There is one
training seed per run and no new control arm; the first run's control is
used as saved. Only output sampling noise is modelled. Input encoding,
4-bit weights, placement and analog error are relaxed. `distill` matches
the final hidden state only. The window SEMs do not account for all corpus
dependence.

The two runs make the first run's 0.70 recovery look like the better point
in this family, not a lower bound. Each change kept most of the robustness
and cost more clean quality. Nothing here identifies the mechanism. The
saturation probe (`z1t_site_probe.py`) can now read the archived weights at
steps 500 to 2,000 of both runs, because the hardened runner kept them.

## Run record

- Runner: `docs/research/z1t_noise_finetune.py` at `05a0059`, the hardened
  version. GTX 1050 Ti, driver 580.196.07, Python 3.12.3, jax/jaxlib
  0.12.0.1, float32 body, float16 head, x64 off. Full provenance is in each
  run's `run-manifest.json`.
- GPU smoke checks for both variants passed first. Each printed the CPU
  reference step-0 clean loss, 2.10935 on OWT.
- `noscale` started at 01:10 PST. The weekly Hermes self-update
  (`hermes-weekly-update.timer`, `hermes update --yes`) restarted the Hermes
  gateway at 04:04:57. That killed the gateway-tracked job at step 1,220
  (exit 143). It also killed the queued `distill` run, which had written only
  an empty log. The training did not fail.
- The chain was relaunched at 09:50 as its own systemd user unit
  (`systemd-run --user`), outside the gateway. `noscale` resumed from its
  step-1,000 checkpoint under the hardened identity check, recorded as a
  second invocation. Its training loss at step 1,010 (7.5036) matched the
  killed attempt's. `noscale` finished at 12:25 PST, about 8.6 s per step
  steady. `distill` finished at 16:27 PST, about 6.6 s per step.
- No `SAVE FAILED`, traceback or non-finite loss. Every archived step
  (500, 1,000, 1,500, 2,000) of both runs is `verified: true` in its
  `summary.json`.

## Evidence retained

[`2026-11-15-z1t-noise-quality-results/`](2026-11-15-z1t-noise-quality-results/artifact-manifest.json)
holds byte copies of each run's `summary.json`, `start.json`,
`provenance.json`, `run-manifest.json`, invocation records, `noisy/final.json`,
`noisy/log.json` and the per-step archive manifests. The artifact manifest
records every copied file's SHA-256, checked equal to its source at capture.
It also records the SHA-256 and size of the weights and checkpoints that were
not copied: 8.8 MB per archived step and a 26 MB recovery checkpoint per run,
all kept outside git.

`compare.py` in that folder replays the comparison above from the committed
copies and the first run's snapshot. It does not evaluate a model.
`comparison.json` is its output, and a rerun fails if the output differs.
This checks arithmetic only: it is not a model rerun or a training replay.
