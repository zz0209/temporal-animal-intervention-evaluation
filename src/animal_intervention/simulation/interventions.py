from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class InterventionAction:
    """A time-bounded multiplicative intervention on selected nodes."""

    name: str
    action_type: str
    target_nodes: tuple[str, ...]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    contact_multiplier: float = 1.0
    susceptibility_multiplier: float = 1.0
    infectivity_multiplier: float = 1.0
    recovery_rate_multiplier: float = 1.0
    rewiring_fraction: float = 0.0
    rewiring_mode: str = "none"

    def __post_init__(self) -> None:
        if pd.Timestamp(self.end_time) <= pd.Timestamp(self.start_time):
            raise ValueError("intervention end_time must follow start_time")
        for field_name in (
            "contact_multiplier",
            "susceptibility_multiplier",
            "infectivity_multiplier",
            "recovery_rate_multiplier",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not 0 <= self.rewiring_fraction <= 1:
            raise ValueError("rewiring_fraction must be between zero and one")
        if self.rewiring_mode not in {"none", "uniform_partner_substitution"}:
            raise ValueError(f"unsupported rewiring_mode: {self.rewiring_mode}")
        if self.rewiring_fraction > 0 and self.rewiring_mode == "none":
            raise ValueError("positive rewiring_fraction requires a rewiring mode")

    def is_target(self, node_id: str) -> bool:
        return str(node_id) in self.target_nodes

    def active_at(self, time: pd.Timestamp) -> bool:
        time = pd.Timestamp(time)
        return pd.Timestamp(self.start_time) <= time < pd.Timestamp(self.end_time)


def neutral_action(start_time: pd.Timestamp, end_time: pd.Timestamp) -> InterventionAction:
    return InterventionAction(
        name="no_intervention",
        action_type="none",
        target_nodes=(),
        start_time=pd.Timestamp(start_time),
        end_time=pd.Timestamp(end_time),
    )
