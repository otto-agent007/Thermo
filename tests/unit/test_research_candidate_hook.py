"""Contract for the editable research hook, run by the harness check catalog."""

import ast
import math
import sys
from pathlib import Path

from thermo_lab.research_candidates.three_site import propose_parameters
from thermo_lab.trajectory_reinforce import build_checked_fixture

HOOK = Path(__file__).parents[2] / "src/thermo_lab/research_candidates/three_site.py"


def _inputs():
    fixture = build_checked_fixture()
    return fixture.model_parameters.values, fixture.parameter_cap


def test_hook_returns_nine_finite_values_inside_the_cap():
    parameters, cap = _inputs()
    values = tuple(propose_parameters(parameters, cap))
    assert len(values) == 9
    for value in values:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(value)
        assert abs(value) <= cap


def test_hook_is_deterministic():
    parameters, cap = _inputs()
    assert tuple(propose_parameters(parameters, cap)) == tuple(propose_parameters(parameters, cap))


def test_hook_imports_only_the_standard_library():
    """The sandbox provides nothing else; this check names the offending import."""
    tree = ast.parse(HOOK.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not available in the hook sandbox"
            modules.append(node.module or "")
    outside = [name for name in modules if name.split(".")[0] not in sys.stdlib_module_names]
    assert outside == [], f"hook imports outside the standard library: {outside}"
