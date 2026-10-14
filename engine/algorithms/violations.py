"""Violation detection.

A violation is any decision-facing constraint whose residual r_z is positive.
The detector is representation-dependent by construction: an R0 model can only
see per-task lateness because it has no aircraft-ready variable at all, and no
representation below R3 can see a service-position violation.  That is the
mechanism by which representation fidelity limits what a controller is even
able to react to, and it is deliberate rather than an implementation shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Violation:
    zid: str                 # target variable, e.g. "L:F17"
    kind: str                # deadline | position | resource
    residual: float          # r_z = max(0, -L) >= 0
    direction: int           # d_z in {-1, +1}
    entity: str
    weight: float = 1.0

    @property
    def is_active(self) -> bool:
        return self.residual > 1e-9


class ViolationDetector:
    def __init__(self, model, guard: float = 0.0):
        self.model = model
        self.guard = guard

    def detect(self, now: float) -> List[Violation]:
        g = self.model.g
        scn = self.model.sim.scn
        out: List[Violation] = []
        for vid, v in g.var.items():
            if v.kind != "L":
                continue
            r = max(0.0, -v.value)
            if r <= self.guard:
                continue
            ent = v.entity
            fl = scn.flights.get(ent)
            w = fl.weight if fl is not None else 1.0
            out.append(Violation(vid, "deadline", r, +1, ent, w))
        out.sort(key=lambda z: (-z.weight * z.residual, z.zid))
        return out

    def residual_vector(self, now: float) -> Dict[str, float]:
        return dict((z.zid, z.residual) for z in self.detect(now))

    def weighted_residual(self, now: float) -> float:
        return sum(z.weight * z.residual for z in self.detect(now))
