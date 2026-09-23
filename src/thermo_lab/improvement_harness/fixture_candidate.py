"""Protected subprocess entry point for extracting the editable candidate hook."""

import json

from thermo_lab.research_candidates.three_site import propose_parameters
from thermo_lab.trajectory_reinforce import build_checked_fixture


def main() -> None:
    fixture = build_checked_fixture()
    values = propose_parameters(fixture)
    print(json.dumps({"parameters": values}, allow_nan=False))


if __name__ == "__main__":
    main()
