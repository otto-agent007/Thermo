"""Protected driver staged beside the editable hook inside the hook sandbox.

The sandbox holds only the standard library and the staged hook package, so
the hook receives plain inputs and cannot import the scorer.
"""

import json
import sys

from thermo_lab.research_candidates.three_site import propose_parameters


def main() -> None:
    payload = json.load(sys.stdin)
    values = propose_parameters(tuple(payload["parameters"]), payload["cap"])
    print(json.dumps({"parameters": list(values)}, allow_nan=False))


if __name__ == "__main__":
    main()
