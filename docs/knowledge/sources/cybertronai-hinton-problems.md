# hinton-problems

```toml
id = "cybertronai-hinton-problems"
title = "hinton-problems"
kind = "task_catalog"
status = "code_released"
last_checked = 2026-10-01

[source]
url = "https://github.com/cybertronai/hinton-problems"
version = "ff9aa60"
released = 2026-08-30
runtime = "NumPy; matplotlib for figures only"
authors = "Sutro Group (cybertronai)"
license = "The Unlicense"

[[claims]]
id = "C1"
status = "code_released"
text = "Pure-NumPy implementations of synthetic learning problems from Hinton's experimental papers, 1981 to 2022, each with paper-comparison metrics, runnable on a laptop CPU."

[[claims]]
id = "C2"
status = "asserted"
text = "RESULTS.md grades 27 stubs as reproducing the paper, 27 as partial and 1 as not replicating."
```

## What it is

A catalog of synthetic learning problems from Geoffrey Hinton's experimental
papers, 1981 to 2022: the shifter, bars, encoders, Boltzmann machines,
wake-sleep, DBN/DBM, MultiMNIST, Forward-Forward and others. Each problem is
one folder with a pure-NumPy model, training, evaluation and a README that
compares against the paper. `RESULTS.md` grades each one `yes` (27),
`partial` (27, gap documented) or `no` (1). Every stub runs on a laptop CPU in
under about five minutes per seed. The README says AI agents built the catalog
(~661M tokens across 63 sessions).

## Use in Thermo

None yet. Extropic's [thermo-RSI post](extropic-thermo-rsi.md) links it as
the task data behind its post-training results, so it is the only public part
of that benchmark.

## Cautions

- It holds the reference implementations, not Extropic's ~50 adapted tasks,
  held-out split, rubrics or scoring. Scores on it are not comparable to the
  post's chart without rebuilding that scoring.
- A `partial` stub's reference metric is the stub's result, not the paper's.
  Check its README before using it as a target.
- READMEs can disagree with their own code. In a local exploration run on
  2026-10-01 (commit `ff9aa60`, NumPy 2.4.3, CPU; not recorded evidence):
  - `bars-rbm --seed 2` recovered 7/8 bars as stated, but gave purity 0.877
    and reconstruction MSE 0.0088 against the README's 0.90 and 0.016.
  - `encoder-4-2-4`'s run command uses `--seed 2`, which ended at 75%
    accuracy. Its results table describes seed 0, which matched exactly
    (restarts at epochs 80 and 160, 100%).

  Before scoring anything against these tasks, fix which metric, seed and
  commit is the target.
- Its documents disagree on the count (53 v1 stubs, 54 with the DBN add-on,
  "55 of 55"). Pin a commit and count from the tree.
- Matplotlib is used for figures only. Keep it out of Thermo's core install.
