"""Human service crews.

Crews are represented separately from equipment because they are qualified,
not merely capable, and because they are shared across flights on their own
service sequence.  For this study the human model emphasises

    qualification + availability + service duration + workload,

which is what couples a crew into the causal graph.  A workload-dependent
service-duration multiplier is available and is exercised only in the optional
sensitivity analysis; the primary claims of the paper do not depend on it, and
the default configuration sets its gain to zero so that no result rests on an
unvalidated fatigue model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Qualification classes and the services each may perform.
SKILL_SETS: Dict[str, Tuple[str, ...]] = {
    "baggage":   ("BAGUNLD", "BAGLD", "CARGOUNLD", "CARGOLD"),
    "catering":  ("CATER_F", "CATER_A"),
    "cleaning":  ("CLEAN",),
    "servicing": ("WATER", "LAV"),
    "fuelling":  ("FUEL",),
    "gate":      ("DEBOARD", "BOARD"),
    "maint":     ("INSPECT",),
    "pushback":  ("PBPREP",),
}

NOMINAL_CREWS = {
    "baggage": 5.0,
    "catering": 3.0,
    "cleaning": 3.0,
    "servicing": 2.0,
    "fuelling": 3.0,
    "gate": 4.0,
    "maint": 2.0,
    "pushback": 3.0,
}

SKILL_SPREAD = {"low": 0.05, "medium": 0.18, "high": 0.38}


@dataclass
class CrewResource:
    """One qualified service crew."""

    crew_id: str
    qualification_set: Tuple[str, ...]
    current_location: str
    availability_time: float = 0.0
    current_assignment: Optional[str] = None
    workload_state: float = 0.0
    skill: str = ""
    rate: float = 1.0                 # service-duration multiplier (<1 faster)
    workload_gain: float = 0.0        # kappa; zero unless the study enables it
    recovery: float = 0.0009
    status: str = "idle"              # idle | serving | unavailable
    sequence: List[str] = field(default_factory=list)
    worked: float = 0.0
    waiting: float = 0.0
    workload_integral: float = 0.0

    def qualified(self, code: str) -> bool:
        return code in self.qualification_set

    @property
    def available(self) -> bool:
        return self.status != "unavailable"

    def as_row(self) -> Dict[str, object]:
        return {
            "crew_id": self.crew_id,
            "qualification_set": "|".join(self.qualification_set),
            "current_location": self.current_location,
            "availability_time": self.availability_time,
            "current_assignment": self.current_assignment or "",
            "workload_state": self.workload_state,
            "status": self.status,
            "worked": self.worked,
            "waiting": self.waiting,
        }


def build_crews(rng, n_flights: int, utilisation: str, apron,
                heterogeneity: str = "medium",
                workload_gain: float = 0.0) -> Dict[str, CrewResource]:
    """Size and place the crew pool for a given offered load."""
    scale = {"low": 1.35, "medium": 1.0, "high": 0.78, "extreme": 0.64}
    mult = scale.get(utilisation, 1.0) * max(1.0, n_flights) / 10.0
    spread = SKILL_SPREAD.get(heterogeneity, 0.18)
    depots = apron.nodes_of_kind("depot") or ["D0"]
    staging = apron.nodes_of_kind("staging") or depots
    home_pool = depots + staging

    crews: Dict[str, CrewResource] = {}
    for skill, nominal in sorted(NOMINAL_CREWS.items()):
        n = max(1, int(round(nominal * mult)))
        for i in range(n):
            cid = "%s_c%d" % (skill, i)
            crews[cid] = CrewResource(
                crew_id=cid,
                qualification_set=SKILL_SETS[skill],
                current_location=home_pool[i % len(home_pool)],
                skill=skill,
                rate=max(0.6, rng.gauss(1.0, spread)),
                workload_gain=workload_gain * (1.0 + rng.uniform(-spread, spread)),
            )
    return crews


def utilisation_of(crews: Dict[str, CrewResource], horizon: float) -> Dict[str, float]:
    by_skill: Dict[str, List[CrewResource]] = {}
    for c in crews.values():
        by_skill.setdefault(c.skill, []).append(c)
    out: Dict[str, float] = {}
    for skill, members in by_skill.items():
        denom = max(1e-9, horizon * len(members))
        out[skill] = sum(m.worked for m in members) / denom
    return out


def workload_fairness(crews: Dict[str, CrewResource]) -> float:
    """Variance of realised crew workload (lower is more balanced)."""
    vals = [c.worked for c in crews.values()]
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)
