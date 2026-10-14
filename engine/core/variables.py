"""Typed state variables of the dynamic causal factor graph.

A variable carries the quadruple

    X_x = (v_x, tau_x, A_x, S_x)

where ``v_x`` is the current predicted or observed value, ``tau_x`` a version
counter, ``A_x`` the current *binding* (active) cause set and ``S_x`` the
sparse set of nonzero action sensitivities

    S_x = { (u, d x / d u) : d x / d u != 0 }.

The core is deliberately domain-neutral: nothing in this module knows what an
aircraft, a stand or a baggage tractor is.  The airport meaning of each
variable kind is attached by ``airport/adapter.py``; the kind strings below are
the only contract between the two layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

NEG_INF = -1e18
POS_INF = 1e18

# ---------------------------------------------------------------------------
# Variable kinds.  The airport reading is given for orientation only; the core
# treats them as opaque labels except where noted.
# ---------------------------------------------------------------------------
VARIABLE_KINDS = (
    "eta",      # predicted arrival of a mobile resource at a service position
    "S",        # service-task start time
    "C",        # service-task completion time
    "Rv",       # GSE (vehicle) readiness for a task
    "Rh",       # crew readiness for a task
    "Rp",       # service-position / stand-capacity readiness
    "ready",    # aircraft-ready (departure-enabling) time C_f^ready
    "L",        # turnaround slack L_f = d_f - C_f^ready
    "mu",       # apron-edge congestion multiplier
    "W",        # crew workload state
)

# Kinds that expose an exogenous control handle u.  A variable of one of these
# kinds seeds its own sensitivity entry with d x / d x = 1, which is what makes
# the sparse stream S_x reachable from an intervention.
HANDLE_KINDS = ("eta", "Rv", "Rh", "Rp")


@dataclass
class Variable:
    """One typed node of the factor graph."""

    vid: str
    kind: str
    value: float = 0.0
    version: int = 0
    active_set: FrozenSet[str] = frozenset()
    sens: Dict[str, float] = field(default_factory=dict)
    producer: Optional[str] = None          # id of the factor that outputs it
    consumers: List[str] = field(default_factory=list)
    entity: str = ""                        # owning flight / resource id
    decision_facing: bool = False           # belongs to Xi_t
    layer: str = "task"                     # task | resource | spatial | flight
    stamp: float = 0.0                      # simulation time of last update

    # -- convenience -------------------------------------------------------
    @property
    def is_handle(self) -> bool:
        return self.kind in HANDLE_KINDS

    # Kinds whose value is a predicted clock time; for these the prediction
    # horizon is how far ahead of the last update the prediction lies.
    TIME_KINDS = ("eta", "S", "C", "Rv", "Rh", "Rp", "ready")

    @property
    def horizon(self) -> float:
        """Prediction horizon v_x - stamp for time-valued variables, else 0."""
        if self.kind in self.TIME_KINDS and abs(self.value) < 1e17:
            return max(0.0, self.value - self.stamp)
        return 0.0

    @property
    def causal_support(self):
        """Identifiers of the inputs currently binding this variable."""
        return tuple(sorted(self.active_set))

    def snapshot(self):
        return (self.value, self.version, self.active_set, dict(self.sens),
                self.stamp)

    def restore(self, snap) -> None:
        (self.value, self.version, self.active_set, self.sens,
         self.stamp) = snap

    def __repr__(self) -> str:              # pragma: no cover - debugging aid
        return "<%s %s=%.3f A=%s>" % (self.kind, self.vid, self.value,
                                      sorted(self.active_set))
