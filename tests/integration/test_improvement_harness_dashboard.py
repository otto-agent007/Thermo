"""Dashboard catalog execution and provenance-bound screenshot retention."""

import subprocess

import pytest

from thermo_lab.improvement_harness import dashboard
from thermo_lab.improvement_harness.checks import run_checks
from thermo_lab.improvement_harness.plan import Plan


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    baseline, candidate, records = (
        tmp_path / name for name in ("baseline", "candidate", "records")
    )
    for worktree in (baseline, candidate):
        (worktree / "dashboard/test-results").mkdir(parents=True)
        for name in dashboard.SCREENSHOTS:
            (worktree / "dashboard/test-results" / name).write_bytes(b"stale")
    plan = Plan(
        schema_version=1,
        track="dashboard",
        objective="Readable layout",
        baseline_commit="a" * 40,
        allowed_paths=("dashboard/src/",),
        primary_metric="checks",
        direction="pass",
        threshold=1,
        max_candidates=2,
        wall_seconds=1800,
    )
    return plan, baseline, candidate, records


@pytest.mark.parametrize("mode", ["passed", "failed", "missing", "unavailable"])
def test_only_fresh_passing_screenshots_are_retained(evaluation, monkeypatch, mode):
    plan, baseline, candidate, records = evaluation

    def runner(argv, **kwargs):
        if argv[-1] == "test:browser":
            if mode != "missing":
                for name in dashboard.SCREENSHOTS:
                    (kwargs["cwd"] / "test-results" / name).write_bytes(b"fresh")
            if mode == "failed":
                return subprocess.CompletedProcess(argv, 1, stdout=b"assertion", stderr=b"")
            if mode == "unavailable":
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout=b"",
                    stderr=b"Executable doesn't exist. Please run npx playwright install",
                )
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(
        dashboard,
        "run_checks",
        lambda track, cwd, seconds, *, record_dir: run_checks(
            track, cwd, seconds, record_dir=record_dir, runner=runner
        ),
    )
    result = dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=records)
    assert result["visual_evidence"] == ("available" if mode == "passed" else "unavailable")
    retained = list((records / "candidate/artifacts").glob("*.png"))
    assert len(retained) == (3 if mode == "passed" else 0)
    assert result["research_outcome"] == "not_applicable"
    assert "aesthetic_score" not in result
    if mode == "unavailable":
        assert result["execution"] == "unavailable"


def test_record_destination_must_be_outside_worktrees(evaluation):
    plan, baseline, candidate, _ = evaluation
    with pytest.raises(ValueError, match="outside"):
        dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=candidate / "evidence")


def test_symlink_observation_destination_is_rejected(evaluation, monkeypatch, tmp_path):
    plan, baseline, candidate, records = evaluation
    records.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (records / "baseline").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        dashboard.evaluate_dashboard(plan, baseline, candidate, record_dir=records)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_playwright_refuses_unrelated_server_and_binds_config_worktree(tmp_path, role):
    """Exercise Playwright's real webServer startup before browser availability."""
    import json
    import shutil
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from pathlib import Path

    source_dashboard = Path(__file__).resolve().parents[2] / "dashboard"
    node_modules = source_dashboard / "node_modules"
    if not (node_modules / "@playwright/test/cli.js").is_file():
        pytest.skip("dashboard Playwright dependency unavailable")
    worktree_dashboard = tmp_path / role / "dashboard"
    (worktree_dashboard / "tests").mkdir(parents=True)
    (worktree_dashboard / "node_modules").symlink_to(node_modules, target_is_directory=True)
    config_path = worktree_dashboard / "tests/playwright.config.ts"
    config_path.write_text((source_dashboard / "tests/playwright.config.ts").read_text())
    server_path = worktree_dashboard / "tests/browser-server.ts"
    server_source = (source_dashboard / "tests/browser-server.ts").read_text()
    server_path.write_text(server_source)
    for directory in ("server", "shared", "tests/fixtures"):
        shutil.copytree(source_dashboard / directory, worktree_dashboard / directory)
    shutil.copy2(
        source_dashboard / "tests/proposal-fixture.ts",
        worktree_dashboard / "tests/proposal-fixture.ts",
    )
    (worktree_dashboard / "package.json").write_text('{"type":"module"}')
    (worktree_dashboard / "tests/browser.spec.ts").write_text(
        'import { test } from "@playwright/test";\n'
        'test("wrong server", async ({ page }) => { await page.goto("/"); });\n'
    )
    # Loading from another cwd must still select this configuration's worktree.
    inspected = subprocess.run(
        [
            "node",
            "--import",
            str(node_modules / "tsx/dist/loader.mjs"),
            "--input-type=module",
            "-e",
            f"import config from {json.dumps(str(config_path))}; "
            f"import {{ archive }} from {json.dumps(str(source_dashboard / 'server/catalog.ts'))}; "
            "console.log(JSON.stringify({config: config.default ?? config, archive}));",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    inspected_data = json.loads(inspected.stdout)
    config = inspected_data["config"]
    shutil.copytree(
        source_dashboard.parent / inspected_data["archive"],
        worktree_dashboard.parent / inspected_data["archive"],
    )
    assert config["use"]["baseURL"] == "http://127.0.0.1:5174"
    assert config["webServer"]["url"] == "http://127.0.0.1:5174"
    assert config["webServer"]["command"] == "node --import tsx tests/browser-server.ts"
    assert 'host: "127.0.0.1", port: 5174, strictPort: true' in server_source

    class WrongServer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"UNRELATED WORKTREE")

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), WrongServer) as server:
        # Preserve the checked config behavior while reserving a test-only port.
        config_path.write_text(config_path.read_text().replace("5174", str(server.server_port)))
        server_path.write_text(server_source.replace("5174", str(server.server_port)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            observed = subprocess.run(
                [
                    "node",
                    str(node_modules / "@playwright/test/cli.js"),
                    "test",
                    "-c",
                    str(worktree_dashboard / "tests/playwright.config.ts"),
                ],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert observed.returncode != 0
    assert "already used" in observed.stdout + observed.stderr
    assert not list(worktree_dashboard.rglob("*.png"))
    assert Path(config["webServer"]["cwd"]).resolve() == worktree_dashboard.resolve()

    # A fresh owned server must run from this worktree even when Playwright was
    # invoked elsewhere. APIRequestContext exercises that binding without a
    # browser download or claiming visual UI verification.
    (worktree_dashboard / "index.html").write_text(f"<h1>{worktree_dashboard}</h1>")
    (worktree_dashboard / "tests/browser.spec.ts").write_text(
        'import { test, expect } from "@playwright/test";\n'
        'test("owned worktree", async ({ request }) => {\n'
        '  const response = await request.get("/");\n'
        f"  expect(await response.text()).toContain({json.dumps(str(worktree_dashboard))});\n"
        '  const proposals = await request.get("/data/proposals.json");\n'
        "  expect((await proposals.json()).items[0].id).toBe("
        '"12345678-1234-4234-8234-123456789abc");\n'
        "});\n"
    )
    owned = subprocess.run(
        ["node", str(node_modules / "@playwright/test/cli.js"), "test", "-c", str(config_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert owned.returncode == 0, owned.stdout + owned.stderr
    assert "1 passed" in owned.stdout
