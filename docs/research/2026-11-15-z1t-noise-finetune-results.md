# Z1T-0 noise fine-tuning: completed two-arm exploration

2026-11-15 UTC. Evidence: **software_simulation; exploration**, not a recorded
study or physical-Z1 result. Retrospective record of the owner-run experiment
in [the proposal](2026-11-14-z1t-noise-finetune-proposal.md). No new experiment,
queue authorization, training, or hardware claim follows from this record.

## Outcome

Noise-aware training substantially improves N=32 loss beyond matched clean
fine-tuning, but recovery is **partial**, with a substantial clean-quality
tradeoff. Both arms reached step 2,000. Values are next-token loss in nats;
lower is better.

| Evaluation | Original clean | Original N32 | Noisy-trained clean | Noisy-trained N32 | Control clean | Control N32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenWebText | 4.002117 | 10.746358 | 5.421706 | 6.003219 | 3.995715 | 10.521048 |
| WikiText-103 | 5.100924 | 11.560240 | 6.689298 | 7.166972 | 5.108116 | 11.346528 |

Paired N32 noisy-minus-control differences are **-4.517829 ± 0.049395 SEM**
on OpenWebText and **-4.179556 ± 0.056828 SEM** on WikiText-103. The proposal's
more-than-two-SEM attribution criterion is met. Recovery against the original
clean-to-noisy gap is 70.3287% and 68.0144%, respectively; clean training alone
recovers 3.3408% and 3.3086%. The predeclared 90% recovery criterion is not met.
These are paired-window summaries from one training seed per arm, not
confidence intervals over independent training replications or hardware.

The OWT residual gap to original clean quality is 2.001102 nats: 1.419589
is the clean degradation of the trained model, and 0.581513 is its remaining
clean-to-N32 noise penalty. About 70.94% of that remaining gap is clean
quality loss. This is an arithmetic decomposition, not a causal mechanism.
The embedding scale changes from 1 to 9.694904 under noisy training versus
1.079842 in control; no ablation establishes that scaling caused the gain.

## Sample-count tradeoff

OWT evaluation at higher N uses one noise draw per window, unlike three at N32.

| N | Original | Noisy-trained | Clean-trained control |
| --- | ---: | ---: | ---: |
| 32 | 10.746358 | 6.003219 | 10.521048 |
| 128 | 9.184392 | 5.590226 | 8.878216 |
| 512 | 6.901006 | 5.464200 | 6.586246 |
| 2048 | 4.915627 | 5.432502 | 4.766710 |

At N2048 the noisy-trained model is worse than control by 0.665792 nats.
Thus robustness at N32 is not general quality recovery across sample budgets.
More samples alone approach this trained model's degraded clean limit.

## Fixed scope and numerical contract

The original checkpoint, frozen embedding and output head, and roughly 2.2M
trainable body parameters plus embedding log-scale are described in the
proposal. Each arm uses 2,000 updates of four 256-token sequences, same data
order and optimizer settings, seed 0, body LR 0.0003 and gain LR 0.003. N
anneals from 1024 to 32 over the first half; **annealing steps above target
use Gaussian forward noise**, while target-N training uses binomial forward
samples with the approximate Gaussian-reparameterized backward rule.
Evaluation uses binomial output means, 64 windows per dataset and three
noise draws per window at N32. Additional N curves use one draw each.

Saved provenance records GTX 1050 Ti (4096 MiB), Python 3.12.3, JAX/jaxlib
0.12.0.1, Equinox 0.13.8, Optax 0.2.9 and NumPy 2.5.4. Body/logits are float32,
head storage float16, x64 disabled. Upstream revision is
`8c0f7e66cfea794f70e81fcf289e179d9f58ed0d`. No synchronized all-in timing or
energy comparison is claimed here. This is not a THRML placement experiment.

## Evidence retained and independent verification

The adjacent [result snapshot](2026-11-15-z1t-noise-results/summary.json)
contains the final summary. The directory also retains original start,
per-arm final values and logs, saved provenance, and an
[artifact manifest](2026-11-15-z1t-noise-results/artifact-manifest.json).
These are byte copies captured from `run_n32` after completion; original
artifacts were not edited. The manifest records source location and SHA-256s.

Independent arithmetic replay used only persisted per-window values, not the
runner's helper functions: all row means/SEMs, paired contrasts/SEMs and
recovery fractions agree to absolute tolerance 1e-12; signatures agree;
all metric arrays contain 64 windows; both final steps are 2000. This checks
arithmetic consistency, **not** a new model evaluation, numerical sampler
validation, training replay, or proof that historical cache inputs matched.
No matching runner process was observed at inspection.

### Checkpoint retention gap

The runner overwrites `ckpt.npz` and `body_params.npz` at each evaluation.
The original run directories contain final noisy and control weights and
checkpoints. Independent control checkpoint copies at steps 500, 1000,
1500 and 2000 are retained outside git in
`/home/user1/thermo-z1t-checkpoint-archive`; step 500 has two identical copies.
Their manifest SHA-256s and ZIP-member CRC checks passed during this review.
Their inventory is included in the artifact manifest; large checkpoint
binaries are not copied into this documentation directory.

No intermediate noisy-arm checkpoint was found in that archive or its run
directory. Its log retains intermediate metrics, **not** the corresponding
weights or optimizer states. Consequently, historical noisy checkpoints
cannot presently be re-evaluated or interpolated. Any broader recovery claim
requires locating an independent copy; regeneration is not archival recovery.

### Provenance and resume limitation

Source audit of `z1t_noise_finetune.py` finds that `eval_sig` and `train_sig`
contain settings but omit input/script hashes, upstream source identity and
runtime identity (`chunk` is also omitted). Cache reuse, checkpoint resume
and cross-invocation final-arm collection compare these settings only.
`provenance.json` records input/script hashes but is replaced on each
invocation before cache lookup. Thus changed inputs or code could silently
reuse stale results or continue a hybrid run with compatible array shapes;
paired arrays are matched by position rather than token-window identity.
A settings mismatch can restart into the same output directory.

**This is a demonstrated source-code vulnerability, not demonstrated
contamination of this run.** The snapshot's provenance is what remained at
capture; it cannot by itself prove every earlier invocation's identity.
No runner was restarted or patched during documentation. Future repair should
fail closed against an immutable input/code manifest, preserve per-invocation
provenance, bind artifacts and paired windows to identities, and test unchanged
resume plus changed-input rejection. Legacy artifacts must not be retroactively
presented as having checks they never enforced.

## Interpretation and next decision

The matched control supports a scoped noise-training benefit under this
software model. It does not show a general training recipe, sample-efficient
hardware execution, or recovery of original model quality. Output-only noise
relaxes dy4p input encoding, 4-bit weights, coupling range, placement and analog
error. Original pretraining overlap with OWT remains unknown; training seed
replication is absent. Window SEMs do not account for all corpus dependence.

The useful next design question is preserving clean capability while retaining
N32 robustness, not automatically extending noise-only training. Existing log
points can inform a clean/noisy tradeoff review, with checkpoint availability
explicit. A mixed clean/noisy objective, teacher preservation or sample-allocation
study is a separate proposal, not authorized by this result. No queue status
was changed by this retrospective documentation.
