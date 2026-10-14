"""Aircraft stands and stand-side service positions.

A stand is not merely a destination.  It is a finite spatial resource

    S_f = (geometry, service positions, access state, occupancy, type
           compatibility)

containing service positions with their own capacities and access rules.
Different service vehicles require different access zones around the aircraft,
so two services needing the same position contend even when neither is a
safety conflict, and two services at different positions may still contend
through the operator's configurable compatibility matrix (``safety_constraints``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .turnaround_tasks import POSITIONS

# How many vehicles a position can host at once.  Holds are wide enough for a
# tractor train and a loader on large aircraft; door positions are not.
POSITION_CAPACITY: Dict[str, int] = {
    "NOSE": 1,
    "FWD_LEFT": 1,
    "AFT_LEFT": 1,
    "FWD_RIGHT": 1,
    "AFT_RIGHT": 1,
    "FWD_HOLD": 1,
    "AFT_HOLD": 1,
    "WING": 1,
}

# Aircraft types a stand class can accept.
STAND_COMPATIBILITY: Dict[str, Tuple[str, ...]] = {
    "contact_large": ("REGIONAL", "NARROW", "WIDE"),
    "contact_small": ("REGIONAL", "NARROW"),
    "remote_large": ("REGIONAL", "NARROW", "WIDE"),
    "remote_small": ("REGIONAL", "NARROW"),
}


@dataclass
class ServicePosition:
    """One access zone around the aircraft on one stand."""

    pid: str                       # "T3:FWD_LEFT"
    stand: str
    position: str
    capacity: int = 1
    occupants: List[str] = field(default_factory=list)   # task ids
    reservations: List[Tuple[str, float, float]] = field(default_factory=list)

    def free(self) -> bool:
        return len(self.occupants) < self.capacity


@dataclass
class Stand:
    """An aircraft stand with its service positions and access state."""

    stand_id: str
    node: str                      # apron-graph node
    stand_class: str
    contact: bool
    positions: Dict[str, ServicePosition] = field(default_factory=dict)
    occupant: Optional[str] = None                # flight currently on-block
    access_state: str = "open"                    # open | restricted | closed
    released_at: float = float("nan")

    def accepts(self, actype: str) -> bool:
        return actype in STAND_COMPATIBILITY.get(self.stand_class, ())

    def position_of(self, code_position: str) -> Optional[ServicePosition]:
        return self.positions.get(code_position)

    def as_row(self) -> Dict[str, object]:
        return {
            "stand_id": self.stand_id,
            "node": self.node,
            "stand_class": self.stand_class,
            "contact": self.contact,
            "occupant": self.occupant or "",
            "access_state": self.access_state,
            "n_positions": len(self.positions),
        }


def build_stands(apron, wide_share: float = 0.35,
                 position_capacity: Optional[Dict[str, int]] = None
                 ) -> Dict[str, Stand]:
    """Instantiate stands on every apron node of kind ``stand``."""
    caps = dict(POSITION_CAPACITY)
    if position_capacity:
        caps.update(position_capacity)
    stands: Dict[str, Stand] = {}
    nodes = apron.nodes_of_kind("stand")
    for i, nid in enumerate(nodes):
        contact = nid.startswith("T")
        big = (i % max(1, int(round(1.0 / max(1e-6, wide_share)))) == 0)
        cls = ("contact_large" if big else "contact_small") if contact else \
              ("remote_large" if big else "remote_small")
        st = Stand(stand_id=nid, node=nid, stand_class=cls, contact=contact)
        for p in POSITIONS:
            st.positions[p] = ServicePosition("%s:%s" % (nid, p), nid, p, caps[p])
        stands[nid] = st
    return stands


def position_id(stand: str, position: str) -> str:
    return "%s:%s" % (stand, position)


def alternative_positions(code: str) -> Tuple[str, ...]:
    """Stand-side positions from which a service may alternatively be worked.

    Only a few services admit an alternative access zone in practice; the rest
    return an empty tuple, so ``CHANGE_SERVICE_POSITION`` is instantiated only
    where it is physically meaningful.
    """
    return {
        "CATER_F": ("AFT_LEFT",),
        "CATER_A": ("FWD_LEFT",),
        "BAGUNLD": ("FWD_HOLD",),
        "BAGLD": ("FWD_HOLD",),
        "CARGOUNLD": ("AFT_HOLD",),
        "CARGOLD": ("AFT_HOLD",),
        "LAV": ("FWD_RIGHT",),
        "WATER": ("AFT_RIGHT",),
    }.get(code, ())
