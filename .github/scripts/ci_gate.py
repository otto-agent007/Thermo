"""A skipped or cancelled dependency is not a successful CI gate."""

import json
import os


def require_success(needs):
    if not needs or any(job.get("result") != "success" for job in needs.values()):
        raise ValueError("Every CI dependency must succeed: " + json.dumps(needs, sort_keys=True))


if __name__ == "__main__":
    require_success(json.loads(os.environ["CI_NEEDS"]))
    print("All required CI groups passed.")
