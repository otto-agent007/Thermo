"""Immutable bounded summaries for deterministic weighted graph walks."""

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from thermo_lab.records import FrozenModel


class GraphWalkVariantResult(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    order: Literal["canonical", "reverse"]
    final_occupancy: tuple[float, ...]
    checkpoint_occupancies: tuple[tuple[float, ...], ...]
    final_half_l1: float = Field(ge=0)
    max_trajectory_half_l1: float = Field(ge=0)
    final_max_abs_error: float = Field(ge=0)
    max_one_particle_leakage: float = Field(ge=0)
    max_normalization_error: float = Field(ge=0)
    minimum_state_probability: float


class GraphWalkOrderSensitivity(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    final_half_l1: float = Field(ge=0)
    max_trajectory_half_l1: float = Field(ge=0)


class GraphWalkAcceptance(FrozenModel):
    passed: StrictBool
    checks: tuple[str, ...]


class WeightedGraphWalkSummary(FrozenModel):
    source_reference: str
    node_labels: tuple[str, ...]
    declared_resolutions: tuple[StrictInt, ...]
    checkpoint_times: tuple[float, ...]
    exact_final_occupancy: tuple[float, ...]
    variants: tuple[GraphWalkVariantResult, ...]
    order_sensitivity: tuple[GraphWalkOrderSensitivity, ...]
    acceptance: GraphWalkAcceptance

    @model_validator(mode="after")
    def validate_summary(self) -> "WeightedGraphWalkSummary":
        structural_fields = (
            ("node_labels", self.node_labels),
            ("declared_resolutions", self.declared_resolutions),
            ("checkpoint_times", self.checkpoint_times),
            ("exact_final_occupancy", self.exact_final_occupancy),
            ("variants", self.variants),
            ("order_sensitivity", self.order_sensitivity),
        )
        for field_name, values in structural_fields:
            if not values:
                raise ValueError(f"{field_name} must not be empty")
        if not self.acceptance.checks:
            raise ValueError("acceptance checks must not be empty")

        if (
            any(resolution < 1 for resolution in self.declared_resolutions)
            or tuple(sorted(set(self.declared_resolutions))) != self.declared_resolutions
        ):
            raise ValueError("declared_resolutions must be strictly increasing and unique")

        node_count = len(self.node_labels)
        if len(self.exact_final_occupancy) != node_count:
            raise ValueError("exact_final_occupancy width must equal node count")

        pairs = [(variant.resolution, variant.order) for variant in self.variants]
        if len(set(pairs)) != len(pairs):
            raise ValueError("variants must contain one row per resolution and order pair")

        declared_resolutions = set(self.declared_resolutions)
        variant_resolutions = {variant.resolution for variant in self.variants}
        if variant_resolutions != declared_resolutions:
            raise ValueError("variant resolutions must match declared resolutions")
        for resolution in self.declared_resolutions:
            orders = {
                variant.order for variant in self.variants if variant.resolution == resolution
            }
            if orders != {"canonical", "reverse"}:
                raise ValueError("variants must contain exactly two orders for each resolution")

        for variant in self.variants:
            if len(variant.final_occupancy) != node_count:
                raise ValueError("variant final occupancy width must equal node count")
            if len(variant.checkpoint_occupancies) != len(self.checkpoint_times):
                raise ValueError("variant checkpoint count must equal checkpoint_times")
            if any(len(checkpoint) != node_count for checkpoint in variant.checkpoint_occupancies):
                raise ValueError("variant checkpoint occupancy width must equal node count")

        sensitivity_resolutions = [item.resolution for item in self.order_sensitivity]
        if len(set(sensitivity_resolutions)) != len(sensitivity_resolutions):
            raise ValueError("order_sensitivity must contain one row per resolution")
        if set(sensitivity_resolutions) != declared_resolutions:
            raise ValueError("order_sensitivity resolutions must match declared resolutions")
        return self
