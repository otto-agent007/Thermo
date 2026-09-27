# Verification

- Full CPU command: `uv run python -m thermo_lab.meta_ebm_cap_baseline
  --output-dir results/meta-ebm-cap-baseline --workers 4 --fit-workers 8`.
  Generation took 3,604.9 s (60.1 min); the complete persisted replay took
  1,961.2 s (32.7 min) and passed before `completion.json` was written.
- The archive records request digest
  `sha256:85ca494706e6f7e2a2634d431e249bace08ea60a098b93a067164cf0a09e4c6d`,
  result digest
  `sha256:474fd591f31dda2ff43acdd69f3e07b84e19adb0ba51af488ab6fd6293b4e036`,
  and archive SHA-256
  `e9b59a929c613fde7260580d672a0c5e307b2afc6980b236ea4e9fcf8b79346a`.
  It contains 10 targets, 180 chains, zero samples, and reports integrity and
  persisted replay as passed.
- Runtime: four matrix/chain workers and eight independent fit workers. The
  cgroup allowed eight CPU cores and 8 GiB memory; peak memory was
  7,178,928,128 bytes (6.69 GiB), and all OOM event counters remained zero.
- `uv run pytest tests/unit/test_meta_ebm_cap_baseline.py -m 'not slow' -q`:
  45 passed, 1 slow replay test deselected in 76.66 s. The one-target archived
  replay test passed separately in 228.12 s (`1 passed, 45 deselected`), so its
  measured runtime fits the 25-minute slow CI job timeout. Ruff and format
  checks passed. The full repository suite was stopped after more than 18
  minutes at about 2% progress, with no failures reported before it was
  stopped.

The study is an exact CPU reference for the fully connected capped kernels.
It makes no hardware, Z1, latency, power or energy claim. Replay interruption
restarts validation from the durable generated archive, so it repeats
validation time but never loses or regenerates the study results.
