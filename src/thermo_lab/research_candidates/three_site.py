"""Editable parameter proposal for the bounded three-site fixture only."""

from thermo_lab.trajectory_reinforce import TrajectoryFixture


def propose_parameters(fixture: TrajectoryFixture) -> tuple[float, ...]:
    """Start with the observed baseline; no scientific improvement is assumed."""
    return fixture.model_parameters.values
