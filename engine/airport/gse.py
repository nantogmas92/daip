"""Heterogeneous ground support equipment.

Each mobile unit carries

    v = (type, capacity, capabilities, location, availability, route, status)

and, crucially for this study, a *service sequence*: the ordered list of tasks
it is currently committed to across flights.  That sequence is what turns a
fleet into a cross-flight coupling mechanism: a delay to the vehicle's current
task shifts the release time of its next task, which may belong to a different
aircraft that has experienced no disturbance of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Fleet classes and the service codes each can perform.  A class may be
# capable of several services (a baggage tractor also serves cargo on narrow
# bodies); capability is checked per unit, not per class, so heterogeneity
# inside a class is representable.
GSE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "bag_tractor":  ("BAGUNLD", "BAGLD"),
    "cargo_loader": ("CARGOUNLD", "CARGOLD"),
    "catering":     ("CATER_F", "CATER_A"),
    "clean_van":    ("CLEAN",),
    "water":        ("WATER",),
    "lavatory":     ("LAV",),
    "fuel_truck":   ("FUEL",),
    "gpu":          ("GPU",),
    "maint_van":    ("INSPECT",),
    "pushback":     ("PBPREP",),
}

# Nominal fleet composition per 10 simultaneous turnarounds, before the
# utilisation multiplier of the scenario configuration is applied.
NOMINAL_FLEET = {
    "bag_tractor": 6.0,
    "cargo_loader": 2.0,
    "catering": 3.0,
    "clean_van": 3.0,
    "water": 2.0,
    "lavatory": 2.0,
    "fuel_truck": 3.0,
    "gpu": 4.0,
    "maint_van": 2.0,
    "pushback": 3.0,
}


@dataclass
class GSEResource:
    """One mobile ground support unit."""

    resource_id: str
    resource_type: str
    capability_set: Tuple[str, ...]
    current_location: str                 # apron node id
    availability_time: float = 0.0
    current_assignment: Optional[str] = None      # task id
    route: Tuple[str, ...] = ()                   # current edge sequence
    status: str = "idle"                  # idle | moving | serving | down | reserve
    speed: float = 1.0
    capacity: float = 1.0
    reserve: bool = False
    home: str = ""
    sequence: List[str] = field(default_factory=list)   # committed task order
    # telemetry
    distance: float = 0.0
    deadhead: float = 0.0
    busy_time: float = 0.0
    n_reassign: int = 0

    def can_serve(self, code: str) -> bool:
        return code in self.capability_set

    @property
    def available(self) -> bool:
        return self.status not in ("down",)

    def as_row(self) -> Dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "capability_set": "|".join(self.capability_set),
            "current_location": self.current_location,
            "availability_time": self.availability_time,
            "current_assignment": self.current_assignment or "",
            "route": "|".join(self.route),
            "status": self.status,
            "reserve": self.reserve,
            "distance": self.distance,
            "deadhead": self.deadhead,
            "busy_time": self.busy_time,
        }


def build_fleet(rng, n_flights: int, utilisation: str, apron,
                reserve_fraction: float = 0.10,
                speed_spread: float = 0.12) -> Dict[str, GSEResource]:
    """Size and place the fleet for a given offered load.

    ``utilisation`` scales the fleet *inversely*: a high-utilisation regime
    supplies fewer units for the same flight bank, which raises the number of
    flights each unit must serve in sequence and therefore strengthens the
    cross-flight causal coupling that this study is about.
    """
    scale = {"low": 1.35, "medium": 1.0, "high": 0.76, "extreme": 0.62}
    mult = scale.get(utilisation, 1.0) * max(1.0, n_flights) / 10.0
    depots = apron.nodes_of_kind("depot") or ["D0"]
    staging = apron.nodes_of_kind("staging") or depots

    fleet: Dict[str, GSEResource] = {}
    for cls, nominal in sorted(NOMINAL_FLEET.items()):
        n = max(1, int(round(nominal * mult)))
        n_res = int(round(n * reserve_fraction))
        for i in range(n):
            rid = "%s_%d" % (cls, i)
            home = depots[i % len(depots)] if i % 3 else staging[i % len(staging)]
            fleet[rid] = GSEResource(
                resource_id=rid,
                resource_type=cls,
                capability_set=GSE_CLASSES[cls],
                current_location=home,
                home=home,
                speed=max(0.7, rng.gauss(1.0, speed_spread)),
                reserve=(i >= n - n_res and n_res > 0),
                status="reserve" if (i >= n - n_res and n_res > 0) else "idle",
            )
    return fleet


def utilisation_of(fleet: Dict[str, GSEResource], horizon: float) -> Dict[str, float]:
    """Per-class utilisation U_GSE over the observed horizon."""
    by_class: Dict[str, List[GSEResource]] = {}
    for v in fleet.values():
        by_class.setdefault(v.resource_type, []).append(v)
    out: Dict[str, float] = {}
    for cls, units in by_class.items():
        denom = max(1e-9, horizon * len(units))
        out[cls] = sum(u.busy_time for u in units) / denom
    return out
