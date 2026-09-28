# M5b verification

- Fixed run checkout: `282ba450e9d54cde1f938806233a8810b2edfc12`.
  Command: `uv run python -m thermo_lab.meta_ebm_thermalization
  --output-dir results/meta-ebm-thermalization --workers 3`, followed by
  the same command with `--resume` after interruption. No executing source
  file changed.
- Attempt 1 started 2026-09-28 18:05:19 UTC and saved 688 of 800 generation
  units by 19:02:37 UTC. It left no terminal marker. Attempt 2 started
  19:36:17 UTC, reused those 688 units, completed generation and saved 475
  replay receipts before its execution session also ended without a terminal
  marker. The exact stop causes are unknown; neither recorded a numerical
  error or OOM. Attempt 3 started 22:49:12 UTC, reused all generated units
  and 475 replay receipts, and finished the remaining replay in 2,482.99
  seconds (41.38 minutes) with three workers.
- `completion.json`: `meta_ebm_thermalization_complete`, 80 reference chains,
  720 new cells, zero samples, `integrity=true`, `replayed=true`. All 800
  generation units and 800 replay receipts are in the final local checkpoint;
  `exit.json` recorded code 0. The final archive SHA-256 matches the marker:
  `4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674`.
- The completed attempt's cgroup peak was 6,559,191,040 bytes (6.11 GiB)
  against an 8-GiB limit; all recorded OOM counters were zero. These are run
  diagnostics, not device performance evidence. `provenance.json` contains
  per-unit timings and all three attempts.
- `uv run pytest tests/unit/test_meta_ebm_thermalization.py -m 'not slow' -q`:
  18 passed, one slow test deselected. The separate slow archived-base-chain
  metrics replay passed. Repository-wide Ruff lint and format checks,
  relative Markdown links and Git whitespace checks passed. The numerical
  and archive replay requirements are specified in the
  [release gate](../../release-gates.md#m5b-inner-thermalization-and-precision-sensitivity).

The [findings](findings.md) explain representative paired comparisons. The
[full generated summary](summary.md) and [compressed archive](study.json.gz)
retain every seed and the protocol's separate stationary, transient,
contraction and rounding measurements.
