# CLAUDE.md

@AGENTS.md

AGENTS.md is the shared source of research rules, project state, engineering
habits, Git policy, and handoff requirements for Claude, Codex, and ChatGPT
Work. Keep shared guidance there so every agent follows the same rules.

Read the relevant frozen protocol under `docs/experiments/` and its gate in
`docs/release-gates.md` before running, changing, or reviewing a study. Study
descriptions are in `docs/studies.md`; current evidence is indexed by the
roadmap. Use targeted tests while iterating, then the gates for changed areas.

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/unit/test_<module>.py
```

The non-slow unit suite is `uv run pytest tests/unit -m "not slow"`;
CI partitions it into several jobs. Full study runs are separate from fast
component tests. Follow AGENTS.md for the complete release requirements.
