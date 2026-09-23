"""Bind a delivery to one successful main-push Thermo CI run."""

import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen


def select_run(run, repository, run_id, main_sha, rollback=False):
    if (
        not re.fullmatch(r"[1-9][0-9]*", run_id)
        or str(run.get("id")) != run_id
        or run.get("repository", {}).get("full_name") != repository
        or run.get("head_repository", {}).get("full_name") != repository
        or run.get("path") != ".github/workflows/ci.yml"
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", ""))
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] < 1
    ):
        raise ValueError("Not a successful main-push run of the expected CI workflow")
    if not rollback and run["head_sha"] != main_sha:
        raise ValueError("Run is stale: automatic delivery requires current main")
    return {
        "sha": run["head_sha"],
        "run_id": run_id,
        "ci_attempt": str(run["run_attempt"]),
    }


def select_build(selection, jobs):
    producers = [job for job in jobs if job.get("name") == "dashboard / dashboard"]
    if not producers or any(type(job.get("run_attempt")) is not int for job in producers):
        raise ValueError("No identifiable dashboard producer job")
    latest = max(job["run_attempt"] for job in producers)
    matches = [job for job in producers if job["run_attempt"] == latest]
    if len(matches) != 1:
        raise ValueError("Ambiguous dashboard producer")
    job = matches[0]
    uploads = [
        step for step in job.get("steps", []) if step.get("name") == "Preserve exact tested build"
    ]
    if (
        not 1 <= latest <= int(selection["ci_attempt"])
        or str(job.get("run_id")) != selection["run_id"]
        or job.get("head_sha") != selection["sha"]
        or job.get("status") != "completed"
        or job.get("conclusion") != "success"
        or len(uploads) != 1
        or uploads[0].get("conclusion") != "success"
    ):
        raise ValueError("Latest dashboard producer did not successfully retain this source build")
    return {
        **selection,
        "attempt": str(latest),
        "artifact": f"dashboard-{selection['run_id']}-{latest}",
    }


def require_same_selection(selection, expected):
    if json.loads(expected) != selection:
        raise ValueError("Source CI/build selection changed while preparing delivery; retry")


def run_jobs(repository, run_id):
    jobs = []
    for page in range(1, 101):
        batch = api(
            f"repos/{repository}/actions/runs/{run_id}/jobs?filter=all&per_page=100&page={page}"
        )
        jobs.extend(batch["jobs"])
        if len(batch["jobs"]) < 100:
            return jobs
    raise ValueError("Too many historical jobs; cannot establish unique producer")


def api(path):
    request = Request(
        "https://api.github.com/" + path,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["SOURCE_RUN_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("Run ID must be a positive integer")
    run = api(f"repos/{repository}/actions/runs/{run_id}")
    main_sha = api(f"repos/{repository}/git/ref/heads/main")["object"]["sha"]
    rollback = (
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        and os.environ.get("PREPARE_ROLLBACK") == "true"
    )
    selection = select_build(
        select_run(run, repository, run_id, main_sha, rollback), run_jobs(repository, run_id)
    )
    if os.environ.get("EXPECTED_SELECTION"):
        require_same_selection(selection, os.environ["EXPECTED_SELECTION"])
    Path(os.environ["RUNNER_TEMP"], "delivery.json").write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n"
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in selection.items():
            stream.write(f"{key}={value}\n")
    print(
        f"Selected {selection['sha']} from run {run_id}, "
        f"CI attempt {selection['ci_attempt']}, build attempt {selection['attempt']}"
    )


if __name__ == "__main__":
    main()
