# Z1T-0 noise training: is the clean-quality cost avoidable?

2026-11-15. **A proposal with its reading rule, written before the runs.**
Owner-run exploration on the local GTX 1050 Ti, outside the loop, with the
merged `z1t_noise_finetune.py` (no code change). Evidence class of anything it
produces: `software_simulation`. No queue row is added; this follows the
closed row `z1t-noise-recovery` and its
[results note](2026-11-15-z1t-noise-finetune-results.md).

## Question

The first run recovered 70.3 percent of the gap at 32 samples per unit, and
paid for it: the noise-trained model's noise-free loss rose from 4.002 to
5.422 nats on OpenWebText while its embedding scale grew from 1 to 9.69.
Is that cost a property of making this model robust, or of the route this
training took?

## Two runs, one change each

Each is the first run's noisy arm with one flag changed and everything else
at the first run's settings (2,000 updates of four 256-token sequences, seed
0, body LR 3e-4, gain LR 3e-3, N annealed 1,024 to 32 over the first half,
64 evaluation windows, three noise draws at N = 32). Each uses a fresh
output folder, as the hardened runner requires.

| Run | Change | What it tests |
| --- | --- | --- |
| `noscale` | `--no-emb-scale` | The embedding scale is frozen at 1. If the scale was the mechanism, recovery should mostly disappear. |
| `distill` | `--loss distill` | The noisy model is trained to match the released model's noise-free final hidden state (normalised squared error) instead of the next-token loss. It asks for robustness without leaving the original model. |

Evaluation is unchanged in both: next-token loss in nats, noise-free and at
N = 32, on OpenWebText and WikiText-103, with the sample-count curve at the
last step. Every evaluation is archived with its weights.

## Comparison and reading rule (set before the runs)

Reference values are the first run's, on OpenWebText: start 4.002 noise-free
and 10.746 at N = 32; noise-trained 5.422 and 6.003; control 3.996 and 10.521.

Paired comparisons across runs are made per window against the first run's
saved per-window values. They are valid only if the windows are the same:
check first that each new run's step-0 noise-free per-window losses agree
with the first run's `start.json` to 1e-3 nats. If they do not, report
unpaired means only and say so.

For each run, with "recovery" the fraction of the start gap closed at N = 32
and "cost" the rise in noise-free loss over the start:

- **Robustness kept:** recovery of 0.60 or more. **Lost:** below 0.35 (half
  of the first run's 0.70). In between: partial.
- **Cost reduced:** noise-free loss below the first run's noise-trained
  model by more than 2 SEM (paired). **Cost avoided:** cost of 0.36 nats or
  less (a quarter of 1.42).
- **Better at the operating point:** N = 32 loss below 6.003 by more than
  2 SEM (paired).

Readings:

- `noscale` loses robustness: the embedding scale was the mechanism, and the
  first run's cost goes with it.
- `noscale` keeps robustness with reduced cost: the scale was not needed and
  the cost was partly a side effect of that route.
- `distill` keeps robustness with cost avoided: robustness and the original
  quality are compatible on this model at this budget.
- Neither reduces the cost while keeping robustness: say plainly that within
  2,000 updates of fine-tuning, robustness at 32 samples and the original
  quality were not obtained together by either route. That would not show
  they are incompatible for a model trained with noise from the start.

## Fixed choices that bound the answer

- A released model is retrofitted for 2,000 updates; nothing is trained
  from scratch. One training seed per run.
- `distill` matches the final hidden state only, not the logits or the
  intermediate layers, and its training loss is not in nats.
- Only output sampling noise is modelled. Input encoding, 4-bit weights,
  placement and analog error are relaxed, as in the first run.
- The first run's folder cannot be extended by the hardened runner, so its
  control is used as saved; no new control arm is trained.

## Run steps

Checked on CPU at smoke scale on commit `05a0059`: both commands run end to
end, archive each evaluation and write a one-arm summary. Neither variant has
run on the GPU; the `distill` path in particular is untested there.

```bash
S=/path/to/Thermo/docs/research      # a checkout of main, 05a0059 or later
cd /mnt/2TBHDD/z1t-finetune
IN="--repo sparse-transformers --weights model.eqx --config config.json \
    --tokenizer tokenizer.json --owt owt100k.parquet --wikitext wt103_test.parquet"

# two-minute checks first (separate folders, safe to delete afterwards)
python $S/z1t_noise_finetune.py $IN --arms noisy --no-emb-scale --out smoke_noscale --smoke
python $S/z1t_noise_finetune.py $IN --arms noisy --loss distill --out smoke_distill --smoke

# the two runs, back to back; the second starts even if the first stops
nohup sh -c "
  python $S/z1t_noise_finetune.py $IN --arms noisy --no-emb-scale \
      --out run_n32_noscale > run_n32_noscale.log 2>&1
  python $S/z1t_noise_finetune.py $IN --arms noisy --loss distill \
      --out run_n32_distill > run_n32_distill.log 2>&1
" > /dev/null 2>&1 &
```

Estimate, not a measurement: about five and a half hours each at the first
run's 8.8 seconds per update, so about eleven hours for both. If a run stops,
re-running the same command resumes it; changing any flag or file is refused
and needs a fresh folder. Post each `summary.json`.
