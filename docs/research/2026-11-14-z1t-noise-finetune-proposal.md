# Z1T-0 under p-bit sampling noise: exploration and an owner-run fine-tune

2026-11-14. **A proposal and exploration, not evidence.** Written in an owner
session with Claude and added to the queue at the owner's request as the held
row `z1t-noise-recovery`. Every number here is `software_simulation` on a CPU
unless it is marked as a source's own claim; none is a Z1 measurement or a
projection of one, and this note states no energy or latency figure. Source
card: [extropic-z1t](../knowledge/sources/extropic-z1t.md).

## Where this sits

This is not a task row and it names no task with digital headroom. It checks
one premise of a published Extropic design on Extropic's own released model:
the Z1T post's latency model assumes N = 32 p-bit samples per tanh-linear
output, and the released training code has no sampling noise in it. The
question is whether the released model works at that sample count, and if
not, whether training with the noise brings it back.

Every hardware constraint is relaxed except output sampling noise: float
weights, no placement on the 16-offset lattice, no dy4p input encoding, no
analog error, no reads or writes counted. A recovery under these conditions
is necessary for the published design and not sufficient.

## What the exploration found

Z1T-0 is the checkpoint at `huggingface.co/Extropic-AI/Z1T-0` (`model.eqx`,
SHA-256 `02577f67…246dcc`; config: vocabulary 50,257, context 256, 4 layers,
width 12,288, fan-in 4, tanh in every sparse layer). Numbers are next-token
loss in nats on 256-token windows, from a NumPy port that matches Extropic's
JAX model to 2e-6 on a small model. They are in
[`2026-11-14-z1t-exploration.json`](2026-11-14-z1t-exploration.json). The
port and evaluation scripts are in the owner's Claude project, not in this
repository; step 0 of the script below reproduces the central number on its
own.

- **Against GPT-2 small** (80 windows each, same windows): 5.08 against 3.78
  on WikiText-103 test, and 3.92 against 3.33 on an OpenWebText sample. The
  OpenWebText documents may be in Z1T-0's training split, which would favour
  it. On OpenWebText the two models agree over the first 16 positions (4.48
  and 4.51) and the gap is all at longer context.
- **Sampling noise** (OpenWebText, 15 windows, clean loss 3.78): replacing
  each tanh-linear output y by the mean of N samples with P(+1) = (1 + y)/2
  costs +1.76 ± 0.03 at N = 1024, +4.09 ± 0.11 at N = 256 and +6.75 ± 0.14 at
  N = 32. At N = 2^20 the loss returns to the clean value (3.646 against
  3.648 on 4 windows), so the sampler is unbiased.
- **Where it hurts** (N = 1024, one site at a time, 4 windows): +0.03 at
  `qkv`, +0.04 at `proj1`, +1.0 at `out_proj` and +0.9 at `proj2`. The two
  damaging sites write into the residual stream, whose updates have RMS 0.02
  to 0.11, so their tanh units sit near zero, where the sampling variance
  (1 − y²)/N is largest.
- **4-bit weights** cost +0.015 with one scale per output row and +1.49 with
  one scale per layer (15 windows).
- **The checkpoint itself:** the stored positional encoding is 1e-31 times
  the sinusoid (the released code weight-decayed it until upstream fix #4),
  and 27 percent of the tanh-linear units have all weights and bias below
  1e-6. Every `proj1` is 37 percent dead, which is e^-1: `proj2` reads four
  random hidden units per output, so that share is never read.

## The run

[`z1t_noise_finetune.py`](z1t_noise_finetune.py) trains two arms from the
same checkpoint with the same data order and settings: `noisy` (sampling
noise on every tanh-linear output) and `control` (no noise). It then
evaluates both on the same windows with the same random keys.

### Fixed choices, and what each could bound

| Choice | Value | What it could bound |
| --- | --- | --- |
| What trains | The 2.2M sparse-body parameters and one embedding scale; embedding and output head frozen | The model can only fix the noise by saturating its writes or by scaling the embedding. A failure may mean the frozen 1.2e9 embedding and head parameters had to move |
| Budget | 2,000 steps of 4 × 256 tokens (about 2e6 tokens) per arm | Too little to rule recovery out. A negative reads "not recovered in this budget" |
| Noise model | Output side only; exact binomial forward pass at N = 32 with a Gaussian-reparameterised gradient | Optimistic: input encoding noise and analog error are left out |
| Annealing | N from 1,024 to 32 over the first half | Halves the steps spent at the target N |
| Learning rates | 3e-4 body, 3e-3 for log-space scales, untuned | Either arm may be under-trained; both arms share them |
| Evaluation | 64 OpenWebText windows from the last 3,000 documents of a 100k sample, 3 noise draws each; 64 WikiText-103 test windows | Training uses the first 8,000 documents of the same sample; Z1T-0's own training split is unknown |
| Precision | float32 body, float16 output head | Clean loss differs from the float32 port by at most 0.001 nats per token |

A small CPU comparison (noisy arm, 250 steps of 32 tokens, 8 windows) put
these defaults ahead of the first design (one learning rate, no annealing,
no embedding scale): 32-sample loss 8.66 against 9.32 from a start of 11.07,
paired difference −0.65 ± 0.16. The clean loss on those windows is 4.33, so
this says the defaults are better, not that recovery works.

### How the result will be read

Fixed before the run. `summary.json` reports, per data set, the paired
difference in 32-sample loss between the arms and the recovered fraction
(the share of the start's noise-induced loss increase that an arm removed).

- **Attribution:** the noisy arm's 32-sample OpenWebText loss must be below
  the control's by more than 2 SEM, or the change is ordinary fine-tuning.
- **Recovered:** recovered fraction of at least 0.9 on OpenWebText.
- **Partial:** between 0.5 and 0.9. Read the sample-count curve (128, 512,
  2,048) for how many samples the fine-tuned model needs.
- **Not recovered in this budget:** below 0.5. This does not show the design
  cannot work; see the budget row above.

### Running it (owner, local GTX 1050 Ti)

The run takes an estimated 4 to 5 GPU-hours, far past the loop's 30
CPU-minute probe ceiling, and needs packages that are not in `uv.lock`
(optax, tokenizers, pyarrow, a CUDA build of jax). It therefore runs in its
own virtual environment outside `uv`, and the row is `held` so that THERMES
does not take it. Nothing in `pyproject.toml` or `uv.lock` changes.

The [GPU conditions](../environments.md#gpu-use) apply as follows.

- The result is `software_simulation` and exploration; it is not a recorded
  study and has no gate.
- The script writes `provenance.json` (device, driver, jax and jaxlib builds,
  `JAX_PLATFORMS`, matmul precision, input SHA-256s) and sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- CPU against GPU: the script has only run on CPU (jax 0.12.0.1). Before the
  full run, `--smoke` on the GPU should print clean losses of 2.109
  (`owt_clean`) and 4.858 (`wt103_clean`) at step 0, the CPU values, to
  about 0.001. Whether `jax[cuda12]` supports this Pascal card is unchecked.

```bash
python3 -m venv venv && source venv/bin/activate
pip install "jax[cuda12]" equinox jaxtyping optax tokenizers pyarrow numpy
python -c "import jax; print(jax.devices())"   # must list a CudaDevice
git clone https://github.com/extropic-ai/sparse-transformers
wget https://huggingface.co/Extropic-AI/Z1T-0/resolve/main/model.eqx
wget https://huggingface.co/Extropic-AI/Z1T-0/resolve/main/config.json
wget https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
wget -O owt100k.parquet https://huggingface.co/datasets/Elriggs/openwebtext-100k/resolve/main/data/train-00000-of-00001-10a4de027237eace.parquet
wget -O wt103_test.parquet https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet

ARGS="--repo sparse-transformers --weights model.eqx --config config.json \
  --tokenizer tokenizer.json --owt owt100k.parquet --wikitext wt103_test.parquet"
python z1t_noise_finetune.py $ARGS --smoke --out smoke      # a few minutes
python z1t_noise_finetune.py $ARGS --out run_n32            # resumes if rerun
```

The script was checked on CPU: its hand-written backward pass matches
`jax.grad` to 3e-7 on a small Z1T built with Extropic's code, its sampler
matches the binomial distribution, and an interrupted run resumes to
bit-identical weights. `sparse-transformers` was read at commit `8c0f7e6`.

## What follows

The owner posts `summary.json` and `provenance.json`. A result note under
`docs/research/` then records the outcome with its evidence class and closes
or drops the row. No protocol, runner or gate follows from this note, and a
positive result would not open a Z1-priced study by itself.

## Checkpoints: recovery and research history are separate (added 2026-11-14)

The first version of the script, which the first full run used, wrote
`ckpt.npz` and `body_params.npz` at every evaluation under the same names. That
is right for a crash-recovery snapshot and wrong as the only copy: it kept
the run restartable and discarded the models at the intermediate evaluations.
Nothing in this note justified that; "restartable" was treated as if it meant
"all the work is preserved". From that run the per-evaluation losses survive
in `<arm>/log.json` and the final weights of each arm in `body_params.npz`.
The noisy arm's weights at steps 500, 1,000 and 1,500 do not. The control
arm's intermediate checkpoints survive only because they were copied by hand
outside the run directory while it ran (recorded in the results note,
`2026-11-15-z1t-noise-finetune-results.md`, which arrives with PR #169).

The script now keeps both, and so does `z1t_sampled_head.py`:

- **Recovery:** `<arm>/ckpt.npz`, one latest snapshot, replaced atomically.
- **Research record:** `<arm>/archive/step_NNNNNN/` at every evaluation, never
  overwritten and made read-only: the weights, `record.json` (that step's
  evaluation, per-window losses, settings, and the provenance of the
  invocation that wrote it, since a resumed run rewrites `provenance.json`)
  and `manifest.json` with SHA-256 digests. Weights for a step that is already
  archived are left alone if identical and written beside it as `_v2` if they
  differ, so a rerun with other settings cannot replace or drop anything.
- **Verification:** every save is written to a temporary file, forced to
  disk, read back and compared with memory before it takes its final name.
  A save that fails stops the run with `SAVE FAILED`; it can be resumed from
  the last verified checkpoint. `summary.json` lists each archived step and
  whether it still matches its manifest.

Checked on CPU at smoke scale: a write that alters or truncates the data is
caught and the previous file is left in place; a damaged snapshot is
reported, not replaced; an interrupted and resumed run archives the same
weights, digest for digest, as an uninterrupted one. The checkpoint format is
unchanged, so a run started with the first version resumes under this one and
archives from then on. `z1t_site_probe.py --steps 500,1000` reads archived
steps, to follow saturation through training.
