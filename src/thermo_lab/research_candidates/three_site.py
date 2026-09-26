"""Editable parameter proposal for the bounded three-site fixture only.

The harness runs this hook in a sandbox holding only the Python standard
library. It receives the checked initial parameters and the symmetric cap,
never the scorer, and must return nine finite values inside the cap.
"""


def propose_parameters(parameters: tuple[float, ...], cap: float) -> tuple[float, ...]:
    """Start with the observed baseline; no scientific improvement is assumed."""
    return parameters
