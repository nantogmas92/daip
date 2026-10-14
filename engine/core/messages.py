"""Typed inter-module records: operational events, factor messages, candidates.

Every record here is an owner-stamped, versioned, frozen message.  Modules
communicate *only* through these records so that any single module (most
importantly the Algorithm 3 backend) can be replaced without touching the rest
of the stack.

Ownership
---------
``AirportOperationalEvent``    owned by the simulation layer (physical truth).
``FactorUpdateMessage``        owned by Algorithm 1.
``InterventionCandidate``      owned by Algorithm 2.
``ValidatedInterventionBundle``owned by the counterfactual validator.

Downstream modules may read a record but never mutate it; the records are
frozen dataclasses so the rule is mechanical rather than a convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# Physical / operational events
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "FLIGHT_ON_BLOCK",          # aircraft arrived at stand, turnaround released
    "TASK_START",
    "TASK_COMPLETE",
    "GSE_DISPATCHED",
    "GSE_ARRIVED",
    "GSE_TRAVEL_DELAY",
    "GSE_BREAKDOWN",
    "GSE_RESTORED",
    "CREW_UNAVAILABLE",
    "CREW_RESTORED",
    "SERVICE_OVERRUN",
    "APRON_CONGESTION",
    "STAND_RELEASE",
    "STAND_RELEASE_DELAY",
    "ARRIVAL_DELAY",
    "AIRCRAFT_READY",
    "CONTROL_TICK",
)


@dataclass(frozen=True)
class AirportOperationalEvent:
    """One observed change in the physical apron state."""

    event_id: int
    simulation_time: float
    event_type: str
    flight_id: str = ""
    task_id: str = ""
    resource_ids: Tuple[str, ...] = ()
    location_id: str = ""
    old_state: Dict[str, float] = field(default_factory=dict)
    new_state: Dict[str, float] = field(default_factory=dict)
    observed_delta: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "simulator"
    seed: int = 0

    def as_row(self, experiment_id: str, epoch: int) -> Dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "seed": self.seed,
            "epoch": epoch,
            "event_id": self.event_id,
            "simulation_time": self.simulation_time,
            "event_type": self.event_type,
            "flight_id": self.flight_id,
            "task_id": self.task_id,
            "resource_ids": "|".join(self.resource_ids),
            "location_id": self.location_id,
            "delta_keys": "|".join(self.observed_delta.keys()),
            "delta_values": "|".join("%.6g" % v
                                     for v in self.observed_delta.values()),
            "confidence": self.confidence,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Factor-graph propagation messages
# ---------------------------------------------------------------------------

MESSAGE_TYPES = (
    "ETA_CHANGED",
    "TASK_COMPLETION_CHANGED",
    "AIRCRAFT_READY_TIME_CHANGED",
    "CRITICALITY_CHANGED",
    "RESOURCE_BINDING_CHANGED",
    "RESERVATION_CHANGED",
    "SLACK_CHANGED",
    "FEASIBILITY_CHANGED",
    "ACTION_SENSITIVITY_CHANGED",
)

# Messages that belong to the sensitivity stream rather than the state stream.
SENSITIVITY_MESSAGES = ("ACTION_SENSITIVITY_CHANGED", "CRITICALITY_CHANGED")


@dataclass(frozen=True)
class FactorUpdateMessage:
    """One state- or sensitivity-stream message emitted by Algorithm 1."""

    epoch: int
    message_type: str
    source_node: str
    destination_factor: str
    old_value: float
    new_value: float
    delta: float
    active_set_before: FrozenSet[str]
    active_set_after: FrozenSet[str]
    causal_support: Tuple[str, ...]
    timestamp: float = 0.0

    @property
    def is_sensitivity(self) -> bool:
        return self.message_type in SENSITIVITY_MESSAGES


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

INTERVENTION_TYPES = (
    "REROUTE_GSE",              # continuous: reduce a vehicle ETA
    "REASSIGN_GSE",             # move a task to another compatible vehicle
    "ACTIVATE_RESERVE_GSE",     # bring a reserve unit into service
    "SWAP_GSE_SEQUENCE",        # exchange service order of two flights
    "REASSIGN_CREW",            # move a task to another qualified crew
    "SWAP_CREW_SEQUENCE",       # exchange crew service order
    "CHANGE_SERVICE_POSITION",  # move a task to another stand-side position
    "ADVANCE_POSITION_ADMISSION",
    "RESEQUENCE_TASKS",         # reorder two order-free services of one flight
    "REASSIGN_STAND",           # high-cost, optional
)

CLASS_OF_TYPE = {
    "REROUTE_GSE": "gse",
    "REASSIGN_GSE": "gse",
    "ACTIVATE_RESERVE_GSE": "gse",
    "SWAP_GSE_SEQUENCE": "gse",
    "REASSIGN_CREW": "crew",
    "SWAP_CREW_SEQUENCE": "crew",
    "CHANGE_SERVICE_POSITION": "stand",
    "ADVANCE_POSITION_ADMISSION": "stand",
    "RESEQUENCE_TASKS": "scheduling",
    "REASSIGN_STAND": "stand",
}


@dataclass(frozen=True)
class InterventionCandidate:
    """A feasible intervention carrying positive predicted causal leverage."""

    candidate_id: str
    candidate_type: str
    affected_flights: Tuple[str, ...]
    affected_tasks: Tuple[str, ...]
    affected_resources: Tuple[str, ...]
    target_violation: Tuple[str, ...]
    predicted_leverage: Dict[str, float]        # violation id -> Lambda_z(u)
    causal_path: Tuple[str, ...]
    side_effects: Dict[str, float]
    resource_requirements: Dict[str, float]
    feasibility: bool
    uncertainty: Dict[str, float]
    validity_epoch: int = 0
    theta: float = 0.0
    cost: float = 1.0
    bottleneck_class: str = "mixed"
    sufficient: bool = False
    leverage_fraction: float = 0.0
    origin: str = "structural"                  # "gradient" | "structural"
    harm: float = 0.0    # priority-weighted damage inflicted on other flights

    @property
    def entities(self) -> Tuple[str, ...]:
        return self.affected_resources

    def leverage_on(self, z: str) -> float:
        return self.predicted_leverage.get(z, 0.0)

    @property
    def signature(self) -> Tuple[str, ...]:
        return (self.candidate_type,) + self.affected_resources + ("%.4g" % self.theta,)


@dataclass(frozen=True)
class ValidatedInterventionBundle:
    """An intervention bundle whose *joint* consequences have been propagated."""

    bundle_id: str
    candidate_ids: Tuple[str, ...]
    predicted_target_effects: Dict[str, float]
    validated_target_effects: Dict[str, float]
    resource_feasibility: bool
    safety_feasibility: bool
    affected_flights: Tuple[str, ...]
    affected_resources: Tuple[str, ...]
    new_violations: Tuple[str, ...] = ()
    validation_epoch: int = 0
    model_error: float = 0.0
    rounds: int = 1

    @property
    def executable(self) -> bool:
        return (self.resource_feasibility and self.safety_feasibility
                and len(self.candidate_ids) > 0)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One closed-loop decision epoch, logged for offline analysis."""

    experiment_id: str = ""
    seed: int = 0
    epoch: int = 0
    simulation_time: float = 0.0
    representation: str = "R6"
    propagation_mode: str = "A2"
    candidate_mode: str = "C2"
    validation_mode: str = "V1"
    backend: str = "cpsat"
    regime: str = "base"
    n_flights_active: int = 0
    n_stands: int = 0
    n_gse: int = 0
    disturbance_class: str = "none"
    # Algorithm 1
    factors_total: int = 0
    factors_touched: int = 0
    messages: int = 0
    criticality_switches: int = 0
    propagation_depth: int = 0
    sccs_resolved: int = 0
    t_prop_ms: float = 0.0
    # Algorithm 2
    universe_size: int = 0
    candidate_size: int = 0
    recall_cand: float = float("nan")
    precision_cand: float = float("nan")
    t_cand_ms: float = 0.0
    budget_exhausted: bool = False
    n_gse_cand: int = 0
    n_crew_cand: int = 0
    n_sched_cand: int = 0
    n_stand_cand: int = 0
    # Algorithm 3
    interaction_edges: int = 0
    bundle_size: int = 0
    t_select_ms: float = 0.0
    t_validate_ms: float = 0.0
    t_interaction_ms: float = 0.0
    t_total_ms: float = 0.0
    solver_status: str = "none"
    repair_fraction: float = 0.0
    new_violations: int = 0
    bundle_model_error: float = 0.0
    # violation context
    n_violations: int = 0
    residual_pre: float = 0.0
    residual_post: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["t_dec_ms"] = self.t_total_ms or (self.t_prop_ms + self.t_cand_ms
                           + self.t_select_ms + self.t_validate_ms + self.t_interaction_ms)
        row["propagation_fraction"] = (
            self.factors_touched / self.factors_total if self.factors_total else 0.0)
        row["causal_sparsity"] = 1.0 - row["propagation_fraction"]
        row["crr"] = (1.0 - self.candidate_size / self.universe_size
                      if self.universe_size else 0.0)
        return row
