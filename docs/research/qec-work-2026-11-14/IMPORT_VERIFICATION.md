# Documentation import verification

Verified on November 14, 2026, based on Thermo main `0ad4d56`, in isolated
branch `docs/qec-research-record`. This records integration checks, not additional
scientific experiments.

- Frozen offline dependency sync and lock check passed.
- Full Ruff check and format check passed (647 Python files).
- Bounded relevant test batch passed: 7 tests in 1.28 seconds, comprising
  `tests/unit/test_qec_precision_archive.py` and
  `tests/unit/test_quality_budget_release_contract.py`.
- All imported-source SHA-256 entries and primary retrospective links passed.
- `git diff --check` passed.
- Full suite collected 2,788 tests but exceeded a 300-second wrapper limit
  (exit 124) during `test_composed_pasym_swap_runner.py`, after 19 visible
  passes. No failure was printed, but the remaining suite is **not verified**.
  The relevant bounded batch above completed separately and can be rerun.

The import changes documentation and adds one read-only archive replay test.
It does not change production algorithms, dependencies, archived evaluator
bytes, queue decisions or runtime configuration. Unrelated user work was
preserved; this record does not imply a commit, push or merge.

Independent review found no blocking issue. It verified all 98 snapshot hashes:
88 against their transfer members and 10 reports/plans against original sources.
Five complete archive hashes were independently recomputed locally. The other
nine full-archive hashes were checked against the supplied inventory, not full
archive bytes. The reviewer checked model labels, negative results, costs,
selection limitations and links, and separately reran the new numerical test
(1 passed, 0.29 seconds). Earlier expensive studies were not replayed by this
import review.
