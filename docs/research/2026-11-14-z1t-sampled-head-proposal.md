# Z1T-0 follow-ups: a sampled output head, and where the samples are needed

2026-11-14. **A proposal and exploration, not evidence.** Written in an owner
session with Claude. The owner asked for the follow-up runs to be set up; this
PR proposes the held row `z1t-sampled-head` for the owner to accept or strike.
Every number here is `software_simulation` on a CPU with small samples unless
it is marked as a source's own claim. None is a Z1 measurement or a
projection of one, and this note states no energy or latency figure:
arithmetic is counted in multiply-accumulates (MACs) per token, which is a
count. Source card: [extropic-z1t](../knowledge/sources/extropic-z1t.md).
Numbers: [exploration JSON](2026-11-14-z1t-sampled-head-exploration.json).

## Where this sits

Neither piece is a task row and neither names a task with digital headroom.
Both follow the held row `z1t-noise-recovery`
([note](2026-11-14-z1t-noise-finetune-proposal.md)) on Extropic's released
Z1T-0.

A p-bit sample can be used three ways: averaged (what Z1T does, and what the
noise-recovery run tests), saturated (the p-bit becomes a comparator), or
taken as the answer. Choosing the next token is the one place in a language
model where the answer is a sample. Z1T-0 does that step with a dense
50,257 x 12,288 readout off the chip, which is 617.6M of its MACs per token
and is left out of the Z1T post's energy comparison (source's own statement).

1. **Site probe** (`z1t_site_probe.py`, evaluation only). Reads the bodies the
   noise-recovery run writes. It answers how the noise-trained body lowered
   its write noise (saturation or signal scale) and how few samples the
   internal sites need. It is part of reading that row's result, not a new row.
2. **Sampled head** (`z1t_sampled_head.py`, the proposed row). Draws the first
   L bits of the next token as L yes/no samples and measures the nats lost.

Hierarchical output heads are an old digital technique and are just as cheap
on a digital chip. A good result would show that the output step can be
handed to single p-bit draws at a known cost in nats. It would not show an
advantage for the hardware. Every hardware constraint is relaxed: noise-free
body, float weights, no placement (a 12,288-wide logit per draw does not fit
a degree-16 node), no reads or writes counted.

## The sampled head

The released output vectors are split in half L times (balanced 2-means in
256 PCA dimensions), giving a binary tree whose depth-L groups hold V / 2^L
tokens. `tree_L` draws L bits in turn, each one Bernoulli sample with logit
`e_node . h + b_node`, then picks the token inside the group with the released
rows. That last step equals the dense model's own conditional exactly, so
every nat lost is lost in the sampled bits. Per token it costs
(L + V / 2^L) x d MACs against V x d.

A node's loss depends only on its own vector, so one set of node vectors
gives every depth at once and the loss splits exactly by level. The script
reports each level's excess over the dense model. `softmax_S` chooses the same
depth-S groups with one 2^S-way softmax (same parameter count) and prices the
binary factorisation. `*_random` puts tokens at random leaves and prices the
grouping against no grouping. `softmax_km` is one softmax over 256 free
k-means clusters of the same vectors, with the released rows inside the
cluster; `softmax_L8 - softmax_km` prices the balanced tree's grouping
against an unconstrained one. Each head is a normalised distribution, so losses are exact nats.
The script first checks that a tree with exact branch probabilities
reproduces the dense loss (agreement to 4e-6 nats in the runs below).

## Exploration

CPU, 200 training windows (51k tokens), 150 steps, 16 test windows of 256
tokens from OpenWebText, tree depth 10. Paired difference to the dense head
in nats (mean, SEM over windows); dense is 3.764.

| Head | MACs removed | Stage 0 | Trained | Trained, random tree |
| --- | --- | --- | --- | --- |
| `tree_L2` | 4x | +0.163 | +0.058 (0.010) | +0.181 |
| `tree_L4` | 16x | +0.390 | +0.201 (0.027) | +0.474 |
| `tree_L8` | 246x | +1.009 | +0.616 (0.061) | +1.360 |
| `tree_L10` | 851x | +1.373 | +0.896 (0.084) | +1.844 |
| `softmax_L8` | 111x | +1.005 | +0.551 (0.060) | +1.274 |
| `softmax_km` | 116x | +0.831 | +0.466 (0.044) | n/a |

- Each sampled level costs 0.03 to 0.15 nats here, least at the top of the
  tree (0.030 and 0.028 for the first two bits).
- Eight binary draws cost 0.064 (0.008) more than one 256-way softmax over
  the same groups.
- The grouping is worth 0.7 to 0.9 nats against a random tree, in line with
  the published hierarchical-softmax results.
- Forcing balanced halves costs 0.085 (0.026) against free k-means clusters
  under the same softmax, on the same windows and fit. The k-means clusters
  hold 31 to 472 tokens; the head's expected cluster size is 178.
- This run is small and data-limited: 51k tokens against 12.6M node
  parameters. Its dev loss had flattened by step 150, which says the data
  ran out, not that the head converged. The full run has 20 times the data.

One parametrisation was tried and dropped (same JSON): a shared rank-256
projection for the node logits. Truncating the released head to rank 256
costs 1.70 nats before training and 1.14 after a short fit. The head is not
low rank, so that choice would have capped the tree. Node vectors are full
width instead.

Unbalanced k-means clusters were first tried as the main two-level head. The
tree is the main construction because it gives every depth from one fit and
a bounded group size, and the clusters stay in the run as `softmax_km`.

Site probe, smoke scale only (released model, 2 windows of 32 tokens): the two
sites that write into the residual stream have mean (1 - y^2) of 0.995, so
they are not saturated at all, and at N = 32 their sampling noise is 2.3 and
1.7 times the RMS of the residual signal they are added to. Units per block:
qkv 24,580, out 12,288, p1 49,152, p2 12,288, so the internal sites are 75%
of the samples.

Why not a recall add-on instead: on 80 OpenWebText windows the released
model's gap to GPT-2 small is 0.596 nats (SEM 0.053). Tokens that repeat a
bigram from earlier in the window are 11% of tokens and 22% of that gap (95%
bootstrap 14 to 31). The rest grows with position from zero over the first 16
tokens. A lookup mechanism of the kind that closes the gap for other
attention-free models would address about a fifth of it at this context.

## Fixed choices

| Choice | Value | Bounds the metric? |
| --- | --- | --- |
| Body | released Z1T-0, frozen, noise-free, states stored as float16 | Yes for transfer: says nothing about a noisy or fine-tuned body |
| Tree | balanced 2-means bisection of output vectors in 256 PCA dims | Yes: worth 0.7 to 0.9 nats against a random tree, and 0.085 behind free clusters in the exploration; `softmax_km` measures this in the run |
| Node logit | linear in h, full width | Possibly: a branch's exact log-odds is not linear in h |
| Within-group step | the dense model's own conditional | No, by construction; its V / 2^L rows are counted |
| Depth | 12 bits (4,095 node vectors per tree, 50M parameters) | Memory bound on a 4 GB GPU; depths 13 to 16 are not measured |
| Training | true next token, Adam 3e-4, 2 epochs of 4,000 windows (1.0M tokens) | Yes for deep levels: few tokens per node; read the per-level excess |
| Selection | best dev step per head; test and WikiText-103 reported | No |

## Reading rule (set before the run)

Headline: the paired test difference `tree_L - dense` at L = 8 (246x fewer
MACs) and L = 12 (2,070x), with its SEM.

- At most 0.05 nats: small. 0.05 to 0.15: moderate (0.14 is the penalty of a
  naive tree in the published text8 comparison). Above 0.15: large.
- The grouping matters if `tree_L - tree_L_random` is below zero by more
  than 2 SEM.
- `tree_L8 - softmax_L8` is the cost of binary draws against one softmax.
- `softmax_L8 - softmax_km` is the cost of the balanced grouping. If it is
  more than half of `tree_L8 - dense`, say that the grouping bounded the
  result.
- If the dev loss is still falling by more than 0.01 nats per evaluation at
  the last step, report "not converged in this budget" and do not quote the
  number as a floor.
- A large loss is a negative for this construction (frozen body, this tree,
  linear branches). It is not evidence about trees trained with the body.

## Run steps

Both scripts import `z1t_noise_finetune.py` from their own directory, so run
them from a checkout of this repository, with the same files and packages
as that run (nothing from `uv.lock`). Run them after the fine-tune finishes:
the 4 GB card cannot hold two jobs.

```bash
S=/path/to/Thermo/docs/research
cd /mnt/2TBHDD/z1t-finetune   # or wherever the inputs are
IN="--repo sparse-transformers --weights model.eqx --config config.json \
    --tokenizer tokenizer.json --owt owt100k.parquet"

# 1. site probe: evaluation only, reads run_n32/{noisy,control}/body_params.npz
python $S/z1t_site_probe.py $IN --run run_n32 --out run_n32/site_probe

# 2. sampled head: needs about 28 GB under --out (25 GB of training states)
python $S/z1t_sampled_head.py $IN --wikitext wt103_test.parquet --out run_head
```

Add `--smoke` to either for a CPU check first (about two minutes each here).
Re-running the head command resumes its cache. If the head's training step
runs out of GPU memory, use `--levels 11`. Estimates, not measurements:
the probe is roughly 3,200 evaluations and the head caches 4,400 windows and
trains 2,000 steps, so plan for one to three GPU-hours each. My estimates for
the fine-tune were low by a factor of two to three.

Post `run_n32/site_probe/site_probe.json` and `run_head/summary.json` with
their `provenance.json`.
