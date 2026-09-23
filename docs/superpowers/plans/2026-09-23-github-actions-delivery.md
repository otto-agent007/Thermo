# GitHub Actions delivery implementation plan

> Execute natively in the existing isolated worktree. Preserve the reviewed PR42 tree as the baseline. One independent whole-branch review follows implementation.

**Goal:** Make every Thermo pull request produce one trustworthy CI decision, and every successful main build produce a verifiable dashboard delivery bundle.

**Architecture:** A single `Thermo CI` workflow calls the existing scientific and dashboard workflows. All existing scientific commands, assertions, budgets and test shards remain gates. A separate delivery workflow consumes the exact successful main-run artifact without rebuilding or executing files from it. Private Sites publication remains an authenticated operator handoff because this environment exposes no durable GitHub-to-Sites deployment credential.

**Scope:** Python 3.11, Node 24, frozen dependency installs; public repository artifacts contain only existing public report/UI data. Keep the private site's audience unchanged. No merge, automatic dependency merge, scientific regeneration or new scientific claim.

## Tasks

- [x] 1. Implement and test delivery trust boundaries. Build a deterministic bundle of the compiled static dashboard with a file manifest, source SHA and run ID. Reject incomplete evidence, unsafe archive members, changed files, mismatched identities and artifacts from unsuccessful/non-main/non-push runs. Test using standard-library unittest and small file fixtures.
- [x] 2. Consolidate the six workflows with one fail-closed `CI required` job. Pin action commits, disable persisted checkout credentials, add timeouts and PR cancellation, use `merge_group`, and preserve every scientific gate. Run the dashboard browser tests against its compiled export and upload that exact build. Lint workflow syntax with checksum-pinned actionlint 1.7.12.
- [x] 3. Deliver successful main-run bundles through a separate read-only workflow. Validate the upstream run using the GitHub API, download only its named artifact, verify its contents without extraction/execution, and retain it with a clear publication handoff. Support an explicitly selected successful run for rollback preparation. Add Dependabot for action pins and dashboard npm updates, plus a main ruleset import and runbook.
- [ ] 4. Verify scripts, workflow syntax, preservation of scientific commands, dashboard tests/build/static-browser CI; independent review; push a stacked PR on PR42. Live branch protection and production authentication are activation requirements, not claims about settings changed by committing YAML.

## Review focus

- A failed, cancelled or skipped child must never produce a green aggregate.
- Pull-request/fork artifacts must never enter the trusted delivery path.
- Bundle SHA, run ID, run attempt and repository must match the selected run.
- Malformed tar entries, duplicate names, symlinks and oversized payloads must be rejected without extraction.
- Automatic delivery cannot select an older main commit; explicit rollback preparation can select an older successful main run, but does not deploy it.

## Validation and decisions

Initial inspection: PR42 remains open; main is unprotected and omits the full-row work. This PR is stacked on PR42 to avoid mixing implementation diffs. Scientific workflow bodies will be compared mechanically against the baseline, with only action pins, runner/tool setup and reporting metadata allowed to differ.

Implementation verification: 14 delivery tests pass, 17 dashboard tests pass, production build/typecheck pass, actionlint 1.7.12 passes, and Ruff passes for pipeline scripts/tests. Mechanical comparison confirms every scientific run command, assertion, matrix, environment and timeout is preserved. Static Chromium and full scientific validation will run on the PR in GitHub.

Independent review: one important finding (failed-job reruns reuse earlier dashboard artifacts) and two smaller hardening findings (recheck binding and decompressed metadata limits). All addressed in one fix pass with regression cases; 14/14 delivery tests pass. No findings deferred.

GitHub validation on initial branch head: compiled dashboard browser tests, packaging and upload pass. Pipeline lint exposed ShellCheck SC2016 on intentionally literal Markdown backticks; summary copy now avoids those constructs without weakening lint. Final head remains subject to the complete CI gate.
