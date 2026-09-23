"""Delivery must reject the wrong build or damaged payload before any publication."""

import copy
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dashboard_bundle as bundle
from select_delivery import select_run

SHA = "a" * 40
REPO = "otto-agent007/Thermo"


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / "dist"
        (self.dist / "data" / "cells").mkdir(parents=True)
        (self.dist / "assets").mkdir()
        (self.dist / "index.html").write_text("<html>Thermo</html>")
        (self.dist / "assets" / "app.js").write_text('console.log("Thermo")')
        self.snapshot = {
            "mode": "snapshot",
            "availability": "available",
            "study": {"cells": [{"id": f"cell/{i}"} for i in range(60)]},
            "recentStudies": [
                {"id": x, "availability": "available"}
                for x in [
                    "full-row",
                    "hop-fidelity",
                    "return-fixture",
                    "objective-steps",
                    "survival-audit",
                ]
            ],
        }
        for i in range(60):
            (self.dist / "data" / "cells" / f"cell~{i}.json").write_text("{}")
        self.write_snapshot()

    def write_snapshot(self):
        (self.dist / "data" / "project.json").write_text(json.dumps(self.snapshot))

    def build(self, name="output"):
        dest = self.root / name
        bundle.build(self.dist, dest, SHA, REPO, "123", "1")
        return dest

    def verify(self, dest, sha=SHA, run_id="123", attempt="1"):
        return bundle.verify(dest, sha, REPO, run_id, attempt)

    def test_roundtrip_and_reproducible_archive(self):
        first, second = self.build(), self.build("second")
        manifest = self.verify(first)
        self.assertEqual(manifest["source_commit"], SHA)
        self.assertEqual(
            (first / "dashboard.tar.gz").read_bytes(), (second / "dashboard.tar.gz").read_bytes()
        )

    def test_identity_mismatch_fails(self):
        dest = self.build()
        for kwargs in [{"sha": "b" * 40}, {"run_id": "124"}, {"attempt": "2"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.verify(dest, **kwargs)

    def test_missing_evidence_or_cell_detail_blocks_build(self):
        for change in ["study", "recent"]:
            original = copy.deepcopy(self.snapshot)
            if change == "study":
                self.snapshot["availability"] = "unavailable"
            else:
                self.snapshot["recentStudies"][0]["availability"] = "unavailable"
            self.write_snapshot()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(change)
            self.snapshot = original
        self.write_snapshot()
        (self.dist / "data/cells/cell~0.json").unlink()
        with self.assertRaises(ValueError):
            self.build()

    def test_symlink_not_packaged(self):
        (self.dist / "leak").symlink_to(self.root)
        with self.assertRaises(ValueError):
            self.build()

    def test_archive_rejects_changed_extra_duplicate_and_link_entries(self):
        dest = self.build()
        original = (dest / "dashboard.tar.gz").read_bytes()
        for case in ["changed", "extra", "duplicate", "link", "traversal"]:
            with self.subTest(case=case):
                with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as src:
                    entries = [(m, src.extractfile(m).read()) for m in src.getmembers()]
                if case == "changed":
                    entries[0] = (entries[0][0], b"changed")
                else:
                    name = {
                        "extra": "extra.txt",
                        "duplicate": entries[0][0].name,
                        "link": "dist/link",
                        "traversal": "../outside",
                    }[case]
                    member = tarfile.TarInfo(name)
                    if case == "link":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                    entries.append((member, b""))
                with tarfile.open(dest / "dashboard.tar.gz", "w:gz") as out:
                    for member, data in entries:
                        member.size = len(data)
                        out.addfile(member, io.BytesIO(data))
                # Update transport checksum: internal manifest verification must still fail.
                bundle.write_checksums(dest)
                with self.assertRaises(ValueError):
                    self.verify(dest)

    def test_sidecar_or_archive_corruption_fails(self):
        for name in ["release.json", "dashboard.tar.gz", "SHA256SUMS"]:
            dest = self.build(name.replace(".", "-"))
            with (dest / name).open("ab") as stream:
                stream.write(b"corruption")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(dest)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.run = {
            "id": 123,
            "run_attempt": 1,
            "head_sha": SHA,
            "head_branch": "main",
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/ci.yml",
            "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO},
        }

    def test_successful_main_run(self):
        self.assertEqual(select_run(self.run, REPO, "123", SHA)["sha"], SHA)

    def test_reject_wrong_event_repo_branch_status_workflow_and_identity(self):
        cases = [
            ("event", "pull_request"),
            ("conclusion", "failure"),
            ("conclusion", "cancelled"),
            ("status", "in_progress"),
            ("head_branch", "feature"),
            ("path", ".github/workflows/other.yml"),
            ("id", 456),
            ("head_sha", "not-a-sha"),
            ("head_repository", {"full_name": "attacker/Thermo"}),
        ]
        for key, value in cases:
            run = {**self.run, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                select_run(run, REPO, "123", SHA)

    def test_stale_main_requires_explicit_rollback_preparation(self):
        with self.assertRaises(ValueError):
            select_run(self.run, REPO, "123", "b" * 40)
        result = select_run(self.run, REPO, "123", "b" * 40, rollback=True)
        self.assertEqual(result["sha"], SHA)

    def test_all_children_required(self):
        from ci_gate import require_success

        require_success({"quality": {"result": "success"}, "tests": {"result": "success"}})
        for result in ["failure", "cancelled", "skipped", "unknown"]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                require_success({"quality": {"result": "success"}, "tests": {"result": result}})
        with self.assertRaises(ValueError):
            require_success({})


if __name__ == "__main__":
    unittest.main()
