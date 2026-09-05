# Model-Context PAsymSwap Continuation Handoff

## Resume point

- Repository: `otto-agent007/Thermo`
- Branch: `feat/model-context-pasym-swap`
- Base: merged `main` commit `d559da3` (PR #9)
- Working tree: clean at the time this handoff was written.
- Do not merge or push without the user's explicit approval.

## Completed checkpoints

- `62c5a5c` — approved one-pass mean-field design.
- `0429e35` — approved implementation plan.
- `f2a5b25` — deterministic 500-occurrence model-context trace and 37-profile pooling.
- `c6a87b9` — checked model-context TOML, schema, factory, dispatch, and identity hash.
- `1a080fe` — four-start model-context compiler warm-started from the target-context artifact.
- `93f1eea` — occurrence-weighted own-model-profile KL acceptance utility.

## Scientific scope

This is a one-pass first-moment factorization diagnostic. It uses the frozen target-context artifact conditional to update endpoint means, pools contexts by target hash, and does not propagate the full 25-site joint distribution. It is therefore not an exact composed-program result, fixed-point procedure, REINFORCE study, or hardware claim.

## Next implementation work

1. Expand `model_context_pasym_swap_results.py` into the strict persisted three-way result contract: uniform baseline, target-context artifact, and model-context artifact. Its reload validation must recompute deterministic quantities without rerunning SciPy or THRML.
2. Add `ThrmlModelContextPAsymSwapBackend`; rebuild upstream target-context artifacts in memory, derive exactly one model trace, compile exactly one model artifact/profile, and sample model artifacts only.
3. Add runner, aggregate, record-schema, and report branches. Preserve the independent and target-context branches unchanged.
4. Add integration tests, docs, CI command, package membership checks, full local gates, review, then PR preparation.

## Required commands before completion

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv run thermo-lab smoke
uv run thermo-lab run configs/experiments/thrml-model-context-pasym-swap.toml --seeds 0,1,2 --output-dir results/model-context-pasym-swap
uv build
git diff --check
```

## Continuation prompt

Use this prompt in a new ChatGPT Work coding chat:

> Continue Thermo on branch `feat/model-context-pasym-swap`. Read `AGENTS.md`, `docs/superpowers/specs/2026-09-05-model-context-pasym-swap-design.md`, `docs/superpowers/plans/2026-09-05-model-context-pasym-swap.md`, and this handoff. Verify the repo state first. Finish the remaining model-context integration tasks, run all required verification, prepare a PR, and do not push or merge without asking me.
