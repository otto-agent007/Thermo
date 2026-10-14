"""Declared, hashed constraint requests for the hardware-constraint layer.

A request names one model and states which Z1-shaped constraints apply to it.
Every constraint applies by default; a relaxed constraint is listed in
``relaxed`` with a reason, and both are part of the canonical digest, so a
result produced with a constraint relaxed can never share a digest with one
produced under it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from thermo_lab.hardware.z1 import Z1HardwareProfile
from thermo_lab.hashing import canonical_sha256

SCHEMA = "thermo_lab.constraints.request.v1"
CONSTRAINTS = (
    "placement",
    "quantization",
    "beta_jitter",
    "chromatic_schedule",
    "no_replica_exchange",
)
UNRELAXABLE = ("no_replica_exchange",)
CAP_RULES = ("absolute", "full", "split", "coupling")
UPDATE_RULES = ("whole_array", "active_kernel")
DTYPES = ("float32", "float64")
CODEBOOK = "L = 2^(b-1) - 1; step = cap / L; q = step * clip(rint(x / step), -L, L)"
SCHEDULE = "chromatic two-colour block Gibbs, colour (u + v) mod 2, no replica exchange"
JITTER_RULE = (
    "static per-node gain g ~ uniform(1 - j, 1 + j), one default_rng(words) per request, "
    "groups drawn in order; the conditional field is multiplied by g; never redrawn per sweep"
)


@dataclass(frozen=True)
class ConstraintRequest:
    """One model's constraint declaration.

    ``bits``/``cap_rule``/``cap`` describe quantization; with ``quantization``
    relaxed they must be ``None``/``"none"``/``None``. ``cap`` is required only by
    the ``absolute`` rule. ``jitter`` is the half-width of the static gain draw and
    ``jitter_words`` its ``default_rng`` seed words; with ``beta_jitter`` relaxed
    the jitter must be 0. ``reasons`` maps every relaxed constraint to its reason.
    """

    model_id: str
    bits: int | None = None
    cap_rule: str = "none"
    cap: float | None = None
    jitter: float = 0.0
    jitter_words: tuple[int, ...] | None = None
    update_rule: str = "whole_array"
    dtype: str = "float32"
    relaxed: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        relaxed = tuple(self.relaxed)
        if len(set(relaxed)) != len(relaxed):
            raise ValueError("relaxed constraints must be unique")
        for name in relaxed:
            if name not in CONSTRAINTS:
                raise ValueError(f"unknown constraint {name!r}")
            if name in UNRELAXABLE:
                raise ValueError(f"{name} cannot be relaxed: the layer has no replica exchange")
        object.__setattr__(self, "relaxed", tuple(sorted(relaxed)))
        reasons = dict(self.reasons)
        if set(reasons) != set(relaxed):
            raise ValueError("reasons must name exactly the relaxed constraints")
        if not all(isinstance(v, str) and v for v in reasons.values()):
            raise ValueError("every relaxed constraint needs a non-empty reason")
        object.__setattr__(self, "reasons", dict(sorted(reasons.items())))
        if self.jitter_words is not None:
            object.__setattr__(self, "jitter_words", tuple(int(w) for w in self.jitter_words))
        self._validate_quantization()
        self._validate_jitter()
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")

    def _validate_quantization(self) -> None:
        if "quantization" in self.relaxed:
            if self.bits is not None or self.cap_rule != "none" or self.cap is not None:
                raise ValueError("quantization is relaxed: bits=None, cap_rule='none', cap=None")
            return
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits < 2:
            raise ValueError("quantization applies: bits must be an integer >= 2")
        if self.cap_rule not in CAP_RULES:
            raise ValueError(f"cap_rule must be one of {CAP_RULES}")
        if self.cap_rule == "absolute":
            if self.cap is None or not math.isfinite(self.cap) or self.cap <= 0:
                raise ValueError("the absolute cap rule needs a positive finite cap")
        elif self.cap is not None:
            raise ValueError("cap is set only by the absolute cap rule")

    def _validate_jitter(self) -> None:
        if not math.isfinite(self.jitter) or not 0.0 <= self.jitter < 1.0:
            raise ValueError("jitter must lie in [0, 1)")
        if "beta_jitter" in self.relaxed:
            if self.jitter != 0.0 or self.jitter_words is not None:
                raise ValueError("beta_jitter is relaxed: jitter=0 and no jitter_words")
        elif self.jitter > 0.0 and not self.jitter_words:
            raise ValueError("jitter > 0 needs jitter_words")

    def applies(self, constraint: str) -> bool:
        if constraint not in CONSTRAINTS:
            raise ValueError(f"unknown constraint {constraint!r}")
        return constraint not in self.relaxed

    def to_canonical(self) -> dict[str, Any]:
        profile = Z1HardwareProfile()
        return {
            "schema": SCHEMA,
            "model_id": self.model_id,
            "lattice": {
                "connection_rules": [list(r) for r in profile.connection_rules],
                "interior_degree": profile.interior_degree,
                "colours": profile.colors,
                "profile_id": profile.profile_id,
                "profile_hash": profile.profile_hash,
            },
            "quantization": {
                "codebook": CODEBOOK,
                "bits": self.bits,
                "cap_rule": self.cap_rule,
                "cap": self.cap,
            },
            "beta_jitter": {
                "rule": JITTER_RULE,
                "jitter": self.jitter,
                "words": None if self.jitter_words is None else list(self.jitter_words),
            },
            "schedule": SCHEDULE,
            "replica_exchange": False,
            "update_rule": self.update_rule,
            "dtype": self.dtype,
            "relaxed": list(self.relaxed),
            "reasons": dict(self.reasons),
        }

    def digest(self) -> str:
        return canonical_sha256(self.to_canonical())
