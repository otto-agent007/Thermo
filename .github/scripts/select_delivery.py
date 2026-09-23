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
        "attempt": str(run["run_attempt"]),
        "artifact": f"dashboard-{run_id}-{run['run_attempt']}",
    }


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
    selection = select_run(run, repository, run_id, main_sha, rollback)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in selection.items():
            stream.write(f"{key}={value}\n")
    print(f"Selected {selection['sha']} from run {run_id}, attempt {selection['attempt']}")


if __name__ == "__main__":
    main()
